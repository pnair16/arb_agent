"""
normalizer.py
=============
Entity resolution and odds normalization layer.

Responsibilities:
1. Normalize raw market titles from both platforms into a canonical string
   suitable for fuzzy comparison (strip stop words, punctuation, etc.).
2. Match Kalshi markets to Polymarket markets using fuzzy string similarity
   (thefuzz / python-Levenshtein).
3. Convert each platform's native pricing into an implied probability in [0, 1].
4. Return a typed `NormalizedMarket` dataclass for the arb engine.

Matching strategy
-----------------
Both platforms express markets as binary YES/NO questions, but the phrasing
differs wildly:
  Kalshi  → "Lakers to win vs. Celtics"
  Poly    → "Will the Lakers beat the Celtics tonight?"

We reduce each title to a bag of meaningful tokens (team names, player names,
sport keywords) and compare with fuzz.token_sort_ratio, which is robust to
word-order differences.  A threshold of 75 is used by default — tune this
value based on observed false-positive/negative rates during research.
"""

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Any

from thefuzz import fuzz  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Similarity threshold (0–100).  Pairs below this score are not matched.
# ---------------------------------------------------------------------------
MATCH_THRESHOLD = 75

# ---------------------------------------------------------------------------
# Stop words that add no semantic value for entity-resolution purposes.
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "will", "the", "a", "an", "be", "to", "vs", "versus",
        "at", "in", "on", "of", "and", "or", "who", "which",
        "win", "beat", "defeat", "over", "game", "match",
        "tonight", "today", "this", "week", "season",
        "nfl", "nba", "mlb", "nhl", "ufc", "mls",  # league names
    }
)


# ---------------------------------------------------------------------------
# Dataclass output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedMarket:
    """
    A matched, normalized market pair ready for arbitrage analysis.

    Probabilities are expressed as floats in [0.0, 1.0].
    A value of None means the data was unavailable or malformed.
    """
    event_name: str           # canonical human-readable label
    sport: str                # inferred sport tag (e.g. "NBA")

    # Kalshi raw and derived fields
    kalshi_ticker: str
    kalshi_prob_yes: float | None  # implied prob of YES outcome
    kalshi_prob_no: float | None   # implied prob of NO outcome (1 - yes if binary)

    # Polymarket raw and derived fields
    poly_id: str
    poly_prob_yes: float | None
    poly_prob_no: float | None

    # Match quality metadata
    match_score: int = field(default=0, compare=False)


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Convert a market title to a lowercase, punctuation-free, stop-word-free
    string suitable for fuzzy comparison.

    Example:
        "Will the Lakers beat the Celtics tonight?" → "lakers celtics"
    """
    # Lower-case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenise and drop stop words
    tokens = [t for t in text.split() if t and t not in _STOP_WORDS]
    return " ".join(tokens)


def _infer_sport(text: str) -> str:
    """Heuristically infer a sport label from raw market text."""
    t = text.lower()
    sport_keywords: list[tuple[str, str]] = [
        ("nfl", "NFL"), ("nba", "NBA"), ("mlb", "MLB"), ("nhl", "NHL"),
        ("ufc", "UFC"), ("mma", "UFC"), ("soccer", "Soccer"),
        ("tennis", "Tennis"), ("golf", "Golf"),
        # fallback team-sport indicators
        ("football", "NFL"), ("basketball", "NBA"),
        ("baseball", "MLB"), ("hockey", "NHL"),
    ]
    for keyword, label in sport_keywords:
        if keyword in t:
            return label
    return "Sports"


# ---------------------------------------------------------------------------
# Kalshi odds → implied probability
# ---------------------------------------------------------------------------

def kalshi_price_to_prob(yes_price_cents: Any) -> float | None:
    """
    Convert Kalshi's YES contract price (in cents, 0–100) to an implied
    probability in [0.0, 1.0].

    Kalshi prices ARE the market-implied probability expressed as cents
    on the dollar:  $0.65 contract → 65% probability → return 0.65.

    We use the mid-point of the best bid/ask when available, otherwise
    fall back to the last traded price.
    """
    try:
        price = float(yes_price_cents)
    except (TypeError, ValueError):
        return None

    if not (0.0 <= price <= 100.0):
        logger.debug("Kalshi: out-of-range price %s; discarding.", yes_price_cents)
        return None

    return round(price / 100.0, 6)


def _kalshi_mid_price(market: dict[str, Any]) -> float | None:
    """
    Extract the YES mid-price from a Kalshi market dict.

    Prefers (yes_bid + yes_ask) / 2 for a fair mid.
    Falls back to last_price or yes_bid alone.
    """
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    last_price = market.get("last_price")

    if yes_bid is not None and yes_ask is not None:
        try:
            mid = (float(yes_bid) + float(yes_ask)) / 2.0
            return kalshi_price_to_prob(mid)
        except (TypeError, ValueError):
            pass

    if last_price is not None:
        return kalshi_price_to_prob(last_price)

    if yes_bid is not None:
        return kalshi_price_to_prob(yes_bid)

    return None


# ---------------------------------------------------------------------------
# Polymarket odds → implied probability
# ---------------------------------------------------------------------------

def polymarket_price_to_prob(price: Any) -> float | None:
    """
    Convert a Polymarket CLOB share price (already in [0, 1] USDC terms)
    to an implied probability.

    Polymarket's Gamma API returns `outcomePrices` as a JSON-encoded list
    of strings like '["0.65", "0.35"]' where index 0 = YES, index 1 = NO.
    This function accepts a single already-parsed float/string value.
    """
    try:
        prob = float(price)
    except (TypeError, ValueError):
        return None

    if not (0.0 <= prob <= 1.0):
        logger.debug("Polymarket: out-of-range price %s; discarding.", price)
        return None

    return round(prob, 6)


def _polymarket_probs(market: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    Extract (yes_prob, no_prob) from a Polymarket Gamma market dict.

    outcomePrices is a list aligned with outcomes:
      outcomes      = ["Yes", "No"]
      outcomePrices = ["0.65", "0.35"]
    """
    outcomes: list[str] = market.get("outcomes", [])
    prices_raw = market.get("outcomePrices", [])

    # outcomePrices may arrive as a JSON-encoded string in some API versions
    if isinstance(prices_raw, str):
        import json
        try:
            prices_raw = json.loads(prices_raw)
        except Exception:
            return None, None

    if len(outcomes) != 2 or len(prices_raw) != 2:
        return None, None

    # Identify YES / NO index by outcome label
    yes_idx: int | None = None
    no_idx: int | None = None
    for idx, label in enumerate(outcomes):
        low = label.lower()
        if low in ("yes", "true", "1"):
            yes_idx = idx
        elif low in ("no", "false", "0"):
            no_idx = idx

    # Default: first outcome is YES, second is NO
    if yes_idx is None:
        yes_idx = 0
    if no_idx is None:
        no_idx = 1

    return (
        polymarket_price_to_prob(prices_raw[yes_idx]),
        polymarket_price_to_prob(prices_raw[no_idx]),
    )


# ---------------------------------------------------------------------------
# Main matching function
# ---------------------------------------------------------------------------

def match_and_normalize(
    kalshi_markets: list[dict[str, Any]],
    poly_markets: list[dict[str, Any]],
    threshold: int = MATCH_THRESHOLD,
) -> list[NormalizedMarket]:
    """
    Match Kalshi markets to Polymarket markets using fuzzy string similarity,
    then normalize probabilities for matched pairs.

    Algorithm:
      1. Pre-compute normalized tokens for every Polymarket market (O(P)).
      2. For each Kalshi market, score it against every Polymarket market
         using token_sort_ratio (handles reordered words) and pick the
         best match above `threshold`.
      3. Each Polymarket market can only be claimed by one Kalshi market
         (greedy best-match; good enough for MVP research purposes).

    Returns a list of NormalizedMarket dataclasses.
    """
    matched: list[NormalizedMarket] = []

    # Pre-compute normalized poly titles for efficiency
    poly_normalized: list[str] = [
        _normalize_text(m.get("question", "")) for m in poly_markets
    ]

    # Track which poly markets have been matched (avoid double-matching)
    claimed_poly_indices: set[int] = set()

    for kalshi_market in kalshi_markets:
        k_title_raw = kalshi_market.get("title", "") or kalshi_market.get("subtitle", "")
        k_title_norm = _normalize_text(k_title_raw)

        if not k_title_norm:
            continue

        best_score = 0
        best_poly_idx: int | None = None

        for poly_idx, p_title_norm in enumerate(poly_normalized):
            if poly_idx in claimed_poly_indices:
                continue
            if not p_title_norm:
                continue

            score = fuzz.token_sort_ratio(k_title_norm, p_title_norm)
            if score > best_score:
                best_score = score
                best_poly_idx = poly_idx

        if best_poly_idx is None or best_score < threshold:
            continue  # no good match found

        poly_market = poly_markets[best_poly_idx]
        claimed_poly_indices.add(best_poly_idx)

        # --- Kalshi probabilities ---
        k_prob_yes = _kalshi_mid_price(kalshi_market)
        k_prob_no = (1.0 - k_prob_yes) if k_prob_yes is not None else None

        # --- Polymarket probabilities ---
        p_prob_yes, p_prob_no = _polymarket_probs(poly_market)

        # Canonical event name: prefer the more verbose Polymarket question
        poly_question = poly_market.get("question", "")
        event_name = poly_question if len(poly_question) > len(k_title_raw) else k_title_raw

        sport = _infer_sport(k_title_raw + " " + kalshi_market.get("category", ""))

        normalized = NormalizedMarket(
            event_name=event_name,
            sport=sport,
            kalshi_ticker=kalshi_market.get("ticker", ""),
            kalshi_prob_yes=k_prob_yes,
            kalshi_prob_no=k_prob_no,
            poly_id=str(poly_market.get("id", "")),
            poly_prob_yes=p_prob_yes,
            poly_prob_no=p_prob_no,
            match_score=best_score,
        )
        matched.append(normalized)

        logger.debug(
            "Matched (score=%d): '%s'  ↔  '%s'",
            best_score, k_title_raw, poly_question,
        )

    logger.info(
        "Normalizer: %d Kalshi × %d Polymarket → %d matched pairs.",
        len(kalshi_markets), len(poly_markets), len(matched),
    )
    return matched
