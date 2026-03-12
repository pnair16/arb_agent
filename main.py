"""
main.py
=======
Async orchestration loop for the Arbitrage Scanner MVP.

Pipeline (runs every SCAN_INTERVAL_SECONDS):
  1. Ingest  → fetch live markets from Kalshi and Polymarket concurrently
  2. Match   → fuzzy-match events across platforms and normalise probabilities
  3. Analyse → calculate arb margins and persist qualifying rows to SQLite

This module is purely observational — it reads market data and logs findings.
No orders are placed, no accounts are accessed, no positions are taken.

Usage:
    python main.py [--interval 10] [--db arb_opportunities.db] [--verbose]
    python main.py --mock           # run one cycle with synthetic data (no API needed)
    python main.py --show-recent 20 # print last 20 DB rows and exit

Environment variables:
    KALSHI_API_KEY    — Required for Kalshi REST API (all endpoints need auth)
    SCAN_INTERVAL     — Override the default scan interval in seconds

Kalshi auth note
----------------
Kalshi's trading API requires authentication for every endpoint, including
market listing.  Without KALSHI_API_KEY the Kalshi client will receive 401
responses and log a clear error message.  Sign up at kalshi.com, create an
API key under Account → API Access, and export it:

    export KALSHI_API_KEY="your-api-key-here"
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp

from api_clients import KalshiClient, PolymarketClient
from arb_engine import evaluate_and_log, fetch_recent_opportunities, init_db
from normalizer import match_and_normalize

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("arb_scanner.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL", "10"))
DEFAULT_DB_PATH = Path("arb_opportunities.db")


# ---------------------------------------------------------------------------
# Mock data for --mock mode
# ---------------------------------------------------------------------------

def _mock_kalshi_markets() -> list[dict[str, Any]]:
    """
    Realistic synthetic Kalshi market dicts that mirror the actual API schema.
    Used by --mock to exercise the full pipeline without network access.

    Prices are in cents (0–100).  Some pairs are deliberately near-arb to
    produce interesting DB rows.
    """
    return [
        {
            "ticker": "NBA-2024-LAKERS-WIN",
            "event_ticker": "NBA-2024-LAKERS-CELTICS",
            "series_ticker": "KBASK",
            "title": "NBA",
            "subtitle": "Lakers to win vs Celtics",
            "category": "Sports",
            "status": "open",
            "yes_bid": 44,
            "yes_ask": 46,
            "last_price": 45,
        },
        {
            "ticker": "NFL-2024-CHIEFS-SB",
            "event_ticker": "NFL-2024-SUPERBOWL",
            "series_ticker": "NFLWINNER",
            "title": "NFL",
            "subtitle": "Will the Chiefs win the Super Bowl?",
            "category": "Sports",
            "status": "open",
            "yes_bid": 31,
            "yes_ask": 33,
            "last_price": 32,
        },
        {
            "ticker": "UFC-JONES-FIGHT",
            "event_ticker": "UFC-308",
            "series_ticker": "UFCFIGHT",
            "title": "UFC",
            "subtitle": "Jon Jones to win UFC 308",
            "category": "Sports",
            "status": "open",
            "yes_bid": 68,
            "yes_ask": 72,
            "last_price": 70,
        },
        {
            "ticker": "NBA-2024-CELTICS-WIN",
            "event_ticker": "NBA-2024-CELTICS-HEAT",
            "series_ticker": "KBASK",
            "title": "NBA",
            "subtitle": "Celtics to win vs Heat",
            "category": "Sports",
            "status": "open",
            "yes_bid": 72,
            "yes_ask": 76,
            "last_price": 74,
        },
    ]


def _mock_polymarket_markets() -> list[dict[str, Any]]:
    """
    Realistic synthetic Polymarket Gamma market dicts.
    outcomePrices is a JSON-encoded string (as in the real Gamma API).
    Some prices are deliberately offset from the Kalshi mocks to create margin.
    """
    return [
        {
            "id": "poly-lakers-celtics-001",
            "question": "Will the Lakers beat the Celtics?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.58", "0.42"]',   # YES=0.58 (Kalshi YES=0.45 → potential arb)
            "active": True,
            "closed": False,
            "tags": [{"id": 6, "label": "Sports"}],
        },
        {
            "id": "poly-chiefs-superbowl-001",
            "question": "Will the Kansas City Chiefs win the Super Bowl?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.30", "0.70"]',   # near-miss, small margin
            "active": True,
            "closed": False,
            "tags": [{"id": 6, "label": "Sports"}],
        },
        {
            "id": "poly-jones-ufc308-001",
            "question": "Will Jon Jones win at UFC 308?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.65", "0.35"]',   # YES=0.65, Kalshi YES=0.70
            "active": True,
            "closed": False,
            "tags": [{"id": 6, "label": "Sports"}],
        },
        {
            "id": "poly-celtics-heat-001",
            "question": "Boston Celtics to beat the Miami Heat",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.71", "0.29"]',
            "active": True,
            "closed": False,
            "tags": [{"id": 6, "label": "Sports"}],
        },
    ]


# ---------------------------------------------------------------------------
# Single scan cycle
# ---------------------------------------------------------------------------

async def run_scan_cycle(
    kalshi: KalshiClient,
    poly: PolymarketClient,
    db_path: Path,
    mock: bool = False,
) -> int:
    """
    Execute one full ingest → match → analyse cycle.

    If mock=True, use synthetic data instead of live API calls so the full
    pipeline can be verified without network access or API credentials.

    Returns the number of rows logged to the database this cycle.
    """
    logger.info("=== Scan cycle started%s ===", " [MOCK]" if mock else "")

    if mock:
        kalshi_markets = _mock_kalshi_markets()
        poly_markets = _mock_polymarket_markets()
    else:
        kalshi_task = asyncio.create_task(kalshi.fetch_sports_markets())
        poly_task = asyncio.create_task(poly.fetch_sports_markets())
        kalshi_markets, poly_markets = await asyncio.gather(kalshi_task, poly_task)

    if not kalshi_markets and not poly_markets:
        logger.warning(
            "Both APIs returned empty results this cycle. "
            "Check KALSHI_API_KEY is set and the network is reachable."
        )
        return 0

    logger.info(
        "Ingested: %d Kalshi markets, %d Polymarket markets.",
        len(kalshi_markets), len(poly_markets),
    )

    matched = match_and_normalize(kalshi_markets, poly_markets)
    if not matched:
        logger.info(
            "No matched pairs this cycle. "
            "If this persists, try lowering MATCH_THRESHOLD in normalizer.py (currently 75)."
        )
        return 0

    logged = evaluate_and_log(matched, db_path=db_path)

    true_arb = [o for o in logged if o.arb_margin_pct > 0]
    if true_arb:
        logger.warning("!!! %d GENUINE ARB OPPORTUNITIES FOUND !!!", len(true_arb))
        for opp in true_arb:
            logger.warning(
                "  → %s  |  K-%s@%.4f  ↔  P-%s@%.4f  |  margin=+%.2f%%",
                opp.matched_event_name[:70],
                opp.kalshi_side, opp.kalshi_price,
                opp.poly_side, opp.poly_price,
                opp.arb_margin_pct,
            )

    logger.info("=== Scan cycle complete — %d rows logged ===", len(logged))
    return len(logged)


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    init_db(db_path)

    logger.info(
        "Arbitrage Scanner MVP starting.  Interval=%ds  DB=%s  Mock=%s",
        args.interval, db_path, args.mock,
    )

    if not args.mock and not os.getenv("KALSHI_API_KEY"):
        logger.warning(
            "KALSHI_API_KEY is not set. Kalshi requests will return 401 and "
            "no Kalshi markets will be fetched. "
            "Set it with: export KALSHI_API_KEY='your-key-here'"
        )

    headers: dict[str, str] = {}
    kalshi_api_key = os.getenv("KALSHI_API_KEY")
    if kalshi_api_key:
        headers["Authorization"] = f"Bearer {kalshi_api_key}"
        logger.info("Kalshi API key loaded from environment.")

    connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(
        headers=headers, connector=connector, timeout=timeout,
    ) as session:
        kalshi = KalshiClient(session)
        poly = PolymarketClient(session)

        scan_count = 0
        total_logged = 0

        while True:
            scan_count += 1
            logger.info("--- Starting scan #%d ---", scan_count)
            try:
                n = await run_scan_cycle(kalshi, poly, db_path, mock=args.mock)
                total_logged += n
            except asyncio.CancelledError:
                logger.info("Scan loop cancelled. Shutting down cleanly.")
                break
            except Exception as exc:
                logger.exception("Unhandled error in scan #%d: %s", scan_count, exc)

            if args.mock:
                logger.info("Mock mode: single cycle complete. Exiting.")
                break

            logger.info(
                "Cumulative rows logged: %d.  Next scan in %ds.",
                total_logged, args.interval,
            )
            await asyncio.sleep(args.interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="arb_scanner",
        description=(
            "Observational arbitrage scanner for Kalshi and Polymarket. "
            "No trades are placed — data is logged to a local SQLite database."
        ),
    )
    parser.add_argument(
        "--interval", type=int, default=SCAN_INTERVAL_SECONDS, metavar="SECONDS",
        help=f"Seconds between scan cycles (default: {SCAN_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH), metavar="PATH",
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help=(
            "Run one cycle with synthetic data to verify the pipeline "
            "without API credentials or network access"
        ),
    )
    parser.add_argument(
        "--show-recent", type=int, default=0, metavar="N",
        help="Print the N most recent logged opportunities and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _configure_logging(args.verbose)

    if args.show_recent > 0:
        rows = fetch_recent_opportunities(limit=args.show_recent, db_path=Path(args.db))
        if not rows:
            print("No opportunities found in database.")
            print("Tip: run `python main.py --mock` first to populate with synthetic data,")
            print("     or set KALSHI_API_KEY and run without --mock for live data.")
        else:
            print(f"{'timestamp':<26} {'sport':<8} {'margin%':>8}  event")
            print("-" * 100)
            for row in rows:
                print(
                    f"{row['timestamp']:<26} {row['sport']:<8} "
                    f"{row['arb_margin_pct']:>7.2f}%  "
                    f"K-{row['kalshi_side']}@{row['kalshi_price']:.3f} "
                    f"vs P-{row['poly_side']}@{row['poly_price']:.3f}  "
                    f"{row['matched_event_name'][:55]}"
                )
        sys.exit(0)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user (KeyboardInterrupt).")
        sys.exit(0)
