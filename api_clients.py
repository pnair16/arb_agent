"""
api_clients.py
==============
Asynchronous API clients for Kalshi and Polymarket.

Handles:
- Rate limiting (respects 429 Retry-After headers)
- Exponential backoff retries on 429/500 errors
- Connection timeouts and malformed JSON guards
- Explicit fast-fail with clear messages on 401/403 (auth errors)
- Returns raw market data as plain Python dicts for downstream processing

Auth notes
----------
Kalshi REST API v2 requires a Bearer token for ALL endpoints.
Set the KALSHI_API_KEY environment variable (or pass via --kalshi-key CLI flag).
Without it every request returns 401 and zero markets are fetched.

Polymarket Gamma API is fully public — no authentication required.
"""

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# Sports-related keywords checked against multiple Kalshi fields.
# The filter is intentionally broad — better to over-fetch and let the
# fuzzy matcher discard irrelevant pairs than to silently drop real markets.
KALSHI_SPORTS_KEYWORDS = frozenset(
    {
        "sports", "nfl", "nba", "mlb", "nhl", "ufc", "mma",
        "soccer", "tennis", "golf", "ncaa", "cbb", "cfb",
        "football", "basketball", "baseball", "hockey",
    }
)

# Known Polymarket sports tag IDs (checked in order; first non-empty wins).
# Run `python diagnose.py` to refresh these if Polymarket changes taxonomy.
POLYMARKET_SPORTS_TAG_IDS = [6, 7, 16, 21]  # common observed IDs for Sports

# HTTP request settings
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.0  # doubles on each retry: 1 → 2 → 4 → 8


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

async def _request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any] | list[Any] | None:
    """
    Perform an HTTP request with exponential-backoff retry logic.

    Retries on:
      - HTTP 429  (Too Many Requests) — also honours Retry-After header
      - HTTP 5xx  (Server errors)
      - aiohttp connection / timeout exceptions

    Fast-fails (no retry) on:
      - HTTP 401/403  — auth problem; retrying won't help, log clearly
      - HTTP 404      — resource missing; retrying won't help

    Returns the parsed JSON payload or None if all retries are exhausted
    or a non-retryable error is encountered.
    """
    backoff = BASE_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.request(method, url, **kwargs) as response:
                # --- Rate-limited ---
                if response.status == 429:
                    retry_after = float(response.headers.get("Retry-After", backoff))
                    logger.warning(
                        "Rate-limited by %s (attempt %d/%d). Sleeping %.1fs.",
                        url, attempt, MAX_RETRIES, retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue

                # --- Auth failure — fast-fail with actionable message ---
                if response.status in (401, 403):
                    logger.error(
                        "AUTH ERROR %d from %s. "
                        "For Kalshi: set the KALSHI_API_KEY environment variable. "
                        "Skipping this endpoint.",
                        response.status, url,
                    )
                    return None  # no point retrying

                # --- Not found — fast-fail ---
                if response.status == 404:
                    logger.warning("404 Not Found: %s — skipping.", url)
                    return None

                # --- Server errors — retry with backoff ---
                if response.status >= 500:
                    logger.warning(
                        "Server error %d from %s (attempt %d/%d). Sleeping %.1fs.",
                        response.status, url, attempt, MAX_RETRIES, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                # --- Any other 4xx — log and fast-fail ---
                if response.status >= 400:
                    logger.error(
                        "Client error %d from %s — skipping (not retried).",
                        response.status, url,
                    )
                    return None

                # --- Success ---
                try:
                    return await response.json(content_type=None)
                except Exception as exc:
                    logger.error("Failed to parse JSON from %s: %s", url, exc)
                    return None

        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            logger.warning(
                "Network error on %s (attempt %d/%d): %s. Sleeping %.1fs.",
                url, attempt, MAX_RETRIES, exc, backoff,
            )
            await asyncio.sleep(backoff)
            backoff *= 2

    logger.error("All %d retries exhausted for %s", MAX_RETRIES, url)
    return None


# ---------------------------------------------------------------------------
# Kalshi client
# ---------------------------------------------------------------------------

def _kalshi_is_sports(market: dict[str, Any]) -> bool:
    """
    Return True if this Kalshi market is sports-related.

    Checks multiple fields because Kalshi uses different categorisation
    schemes across market types:
      - category    e.g. "Sports"
      - series_ticker e.g. "NFLWINNER", "KBASK", "UFCFIGHT"
      - event_ticker  e.g. "NFL-2024-SUPERBOWL"
      - title / subtitle  plain-text description
    """
    haystack = " ".join(
        str(market.get(field, ""))
        for field in ("category", "series_ticker", "event_ticker", "title", "subtitle")
    ).lower()
    return any(kw in haystack for kw in KALSHI_SPORTS_KEYWORDS)


class KalshiClient:
    """
    Thin async client for the Kalshi REST API v2.

    Authentication:
        All Kalshi endpoints require a Bearer token.
        Pass the token as the Authorization header in the shared aiohttp
        session (set KALSHI_API_KEY env var — main.py handles injection).
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def fetch_sports_markets(self) -> list[dict[str, Any]]:
        """
        Fetch all *open* Kalshi markets that are sports-related.

        Kalshi paginates with cursor-based scheme (cursor / limit params).
        Iterates until the API returns no next cursor.

        Each returned market dict contains at minimum:
          ticker, title, subtitle, series_ticker, event_ticker,
          category, status, yes_bid, yes_ask, last_price, close_time
        """
        all_markets: list[dict[str, Any]] = []
        url = f"{KALSHI_BASE_URL}/markets"
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor

            payload = await _request_with_retry(
                self._session, "GET", url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            )

            if payload is None:
                # Could be auth error (401) or network failure — already logged
                break

            markets: list[dict[str, Any]] = payload.get("markets", [])
            sports = [m for m in markets if _kalshi_is_sports(m)]
            all_markets.extend(sports)

            logger.debug(
                "Kalshi page: %d total markets, %d sports.", len(markets), len(sports)
            )

            # A cursor of "" (empty string) or None both mean "last page"
            cursor = payload.get("cursor") or None
            if not cursor or not markets:
                break

        logger.info("Kalshi: fetched %d sports markets total.", len(all_markets))
        return all_markets


# ---------------------------------------------------------------------------
# Polymarket client
# ---------------------------------------------------------------------------

class PolymarketClient:
    """
    Thin async client for the Polymarket Gamma API (public, no auth needed).

    The Gamma API is the REST interface for market metadata including CLOB
    mid-prices expressed as probabilities in [0, 1].
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def _discover_sports_tag_id(self) -> int | None:
        """
        Query /tags to find the correct numeric ID for the Sports tag.
        Returns the first matching tag ID, or None if the endpoint fails.

        This is necessary because the tag ID can change across Polymarket
        deployments and the hardcoded constant may be stale.
        """
        payload = await _request_with_retry(
            self._session, "GET", f"{POLYMARKET_GAMMA_URL}/tags",
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        )
        if not payload:
            return None

        tags: list[dict[str, Any]] = payload if isinstance(payload, list) else payload.get("tags", [])
        sports_keywords = {"sport", "nfl", "nba", "mlb", "nhl", "ufc", "soccer", "tennis"}

        for tag in tags:
            label = str(tag.get("label", "") + " " + tag.get("slug", "")).lower()
            if any(kw in label for kw in sports_keywords):
                tag_id = tag.get("id")
                if tag_id is not None:
                    logger.info("Polymarket: discovered sports tag_id=%s (%r).", tag_id, label.strip())
                    return int(tag_id)
        return None

    async def _fetch_page(
        self, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Fetch one page and normalise the response to a flat list."""
        payload = await _request_with_retry(
            self._session, "GET", f"{POLYMARKET_GAMMA_URL}/markets",
            params=params,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        )
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        return payload.get("markets", [])

    async def fetch_sports_markets(self) -> list[dict[str, Any]]:
        """
        Fetch active Polymarket markets tagged as Sports.

        Strategy:
          1. Query /tags to discover the live sports tag ID.
          2. Try each known fallback tag ID if discovery fails.
          3. If tag-based filtering returns 0 results, fall back to fetching
             all active markets and keyword-filtering on question text.

        The Gamma API uses offset-based pagination (offset + limit).
        """
        # Step 1: discover live tag ID
        sports_tag_id: int | None = await self._discover_sports_tag_id()

        # Fall back to known candidates if discovery failed
        tag_ids_to_try: list[int | None] = (
            [sports_tag_id] if sports_tag_id else []
        ) + POLYMARKET_SPORTS_TAG_IDS

        # Deduplicate while preserving order
        seen: set[int] = set()
        unique_tag_ids: list[int] = []
        for tid in tag_ids_to_try:
            if tid is not None and tid not in seen:
                seen.add(tid)
                unique_tag_ids.append(tid)

        all_markets: list[dict[str, Any]] = []
        page_size = 100

        for tag_id in unique_tag_ids:
            offset = 0
            tag_markets: list[dict[str, Any]] = []

            while True:
                page = await self._fetch_page({
                    "active": "true",
                    "closed": "false",
                    "tag_id": tag_id,
                    "limit": page_size,
                    "offset": offset,
                })
                tag_markets.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size

            if tag_markets:
                logger.info(
                    "Polymarket: tag_id=%d returned %d markets.", tag_id, len(tag_markets)
                )
                all_markets = tag_markets
                break  # found a working tag ID — stop trying others
            else:
                logger.debug("Polymarket: tag_id=%d returned 0 markets, trying next.", tag_id)

        # Step 2: fallback — keyword filter on all active markets
        if not all_markets:
            logger.warning(
                "Polymarket: no markets from tag IDs %s. "
                "Falling back to full-market keyword filter.",
                unique_tag_ids,
            )
            sports_kw = frozenset(
                {"nfl", "nba", "mlb", "nhl", "ufc", "soccer", "tennis",
                 "football", "basketball", "baseball", "hockey", "golf", "mma"}
            )
            offset = 0
            while True:
                page = await self._fetch_page({
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                })
                if not page:
                    break
                for mkt in page:
                    q = mkt.get("question", "").lower()
                    if any(kw in q for kw in sports_kw):
                        all_markets.append(mkt)
                if len(page) < page_size:
                    break
                offset += page_size

        logger.info("Polymarket: fetched %d sports markets total.", len(all_markets))
        return all_markets
