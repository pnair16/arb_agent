"""
api_clients.py
==============
Asynchronous API clients for Kalshi and Polymarket.

Handles:
- Rate limiting (respects 429 Retry-After headers)
- Exponential backoff retries on 429/500 errors
- Connection timeouts and malformed JSON guards
- Returns raw market data as plain Python dicts for downstream processing
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

# Sports-related category tags / keywords used to filter Kalshi markets
KALSHI_SPORTS_TAGS = ["Sports", "NFL", "NBA", "MLB", "NHL", "UFC", "Soccer", "Tennis"]

# Polymarket sports tag id (used to pre-filter requests)
POLYMARKET_SPORTS_TAG_ID = 6  # tag_id=6 → Sports on Gamma API

# HTTP request settings
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.0  # doubles on each retry


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

    Returns the parsed JSON payload or None if all retries are exhausted.
    """
    backoff = BASE_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status == 429:
                    retry_after = float(response.headers.get("Retry-After", backoff))
                    logger.warning(
                        "Rate-limited by %s (attempt %d/%d). Sleeping %.1fs.",
                        url, attempt, MAX_RETRIES, retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue

                if response.status >= 500:
                    logger.warning(
                        "Server error %d from %s (attempt %d/%d). Sleeping %.1fs.",
                        response.status, url, attempt, MAX_RETRIES, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                response.raise_for_status()

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

class KalshiClient:
    """
    Thin async client for the Kalshi REST API v2.

    Only reads public market data — no authentication required for
    listing active markets and their current prices.
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def fetch_sports_markets(self) -> list[dict[str, Any]]:
        """
        Fetch all *open* Kalshi markets that belong to sports categories.

        Kalshi paginates with a cursor-based scheme (cursor / limit params).
        We iterate until no next cursor is returned.

        Each returned market dict contains at minimum:
          ticker, title, status, yes_bid, yes_ask, no_bid, no_ask,
          category, subtitle, event_ticker, close_time, ...
        """
        all_markets: list[dict[str, Any]] = []
        url = f"{KALSHI_BASE_URL}/markets"
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {
                "status": "open",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            payload = await _request_with_retry(
                self._session, "GET", url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            )

            if payload is None:
                logger.error("Kalshi: received None payload; aborting pagination.")
                break

            markets: list[dict[str, Any]] = payload.get("markets", [])

            # Filter to sports-relevant markets by checking category tag
            sports_markets = [
                m for m in markets
                if any(
                    tag.lower() in (m.get("category", "") + " " + m.get("series_ticker", "")).lower()
                    for tag in KALSHI_SPORTS_TAGS
                )
            ]
            all_markets.extend(sports_markets)

            cursor = payload.get("cursor")
            if not cursor or not markets:
                break

        logger.info("Kalshi: fetched %d sports markets.", len(all_markets))
        return all_markets

    async def fetch_market_orderbook(self, ticker: str) -> dict[str, Any] | None:
        """
        Fetch the current order book for a single market by ticker.
        Returns the best yes_bid as a proxy for live pricing.
        """
        url = f"{KALSHI_BASE_URL}/markets/{ticker}"
        payload = await _request_with_retry(
            self._session, "GET", url,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        )
        if payload is None:
            return None
        return payload.get("market")


# ---------------------------------------------------------------------------
# Polymarket client
# ---------------------------------------------------------------------------

class PolymarketClient:
    """
    Thin async client for the Polymarket Gamma API.

    The Gamma API is the public REST interface for market metadata,
    including CLOB (Central Limit Order Book) mid-prices expressed
    as probabilities in [0, 1].
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def fetch_sports_markets(self) -> list[dict[str, Any]]:
        """
        Fetch active Polymarket markets tagged as Sports.

        Gamma API supports offset-based pagination via `offset` and `limit`.
        We iterate until fewer results than the page size are returned.

        Each market dict contains at minimum:
          id, question, description, outcomes, outcomePrices,
          active, closed, tags, ...
        """
        all_markets: list[dict[str, Any]] = []
        url = f"{POLYMARKET_GAMMA_URL}/markets"
        offset = 0
        page_size = 100

        while True:
            params: dict[str, Any] = {
                "active": "true",
                "closed": "false",
                "tag_id": POLYMARKET_SPORTS_TAG_ID,
                "limit": page_size,
                "offset": offset,
            }

            payload = await _request_with_retry(
                self._session, "GET", url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            )

            if payload is None:
                logger.error("Polymarket: received None payload; aborting pagination.")
                break

            # Gamma returns a top-level list or a dict with a "markets" key
            if isinstance(payload, list):
                page = payload
            else:
                page = payload.get("markets", [])

            all_markets.extend(page)

            if len(page) < page_size:
                break  # last page

            offset += page_size

        logger.info("Polymarket: fetched %d sports markets.", len(all_markets))
        return all_markets
