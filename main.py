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

Environment variables (optional):
    KALSHI_API_KEY    — Bearer token if Kalshi ever requires auth for live data
    SCAN_INTERVAL     — Override the default scan interval in seconds
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import aiohttp

from api_clients import KalshiClient, PolymarketClient
from arb_engine import evaluate_and_log, init_db, fetch_recent_opportunities
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
    # Quieten noisy third-party loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL", "10"))
DEFAULT_DB_PATH = Path("arb_opportunities.db")


# ---------------------------------------------------------------------------
# Single scan cycle
# ---------------------------------------------------------------------------

async def run_scan_cycle(
    kalshi: KalshiClient,
    poly: PolymarketClient,
    db_path: Path,
) -> int:
    """
    Execute one full ingest → match → analyse cycle.

    Returns the number of rows logged to the database this cycle.
    """
    logger.info("=== Scan cycle started ===")

    # Step 1 — Ingest both platforms concurrently
    kalshi_task = asyncio.create_task(kalshi.fetch_sports_markets())
    poly_task = asyncio.create_task(poly.fetch_sports_markets())

    kalshi_markets, poly_markets = await asyncio.gather(
        kalshi_task, poly_task, return_exceptions=False
    )

    if not kalshi_markets and not poly_markets:
        logger.warning("Both APIs returned empty results. Skipping this cycle.")
        return 0

    logger.info(
        "Ingested: %d Kalshi markets, %d Polymarket markets.",
        len(kalshi_markets), len(poly_markets),
    )

    # Step 2 — Match and normalise
    matched = match_and_normalize(kalshi_markets, poly_markets)

    if not matched:
        logger.info("No matched pairs found this cycle.")
        return 0

    # Step 3 — Evaluate arbitrage and log to DB
    logged = evaluate_and_log(matched, db_path=db_path)

    # Surface any genuine arbitrage opportunities clearly
    true_arb = [o for o in logged if o.arb_margin_pct > 0]
    if true_arb:
        logger.warning(
            "!!! %d GENUINE ARB OPPORTUNITIES FOUND THIS CYCLE !!!",
            len(true_arb),
        )
        for opp in true_arb:
            logger.warning(
                "  → %s  |  K-%s @ %.4f  ↔  P-%s @ %.4f  |  margin=+%.2f%%",
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

    # Initialise database schema (idempotent)
    init_db(db_path)

    logger.info(
        "Arbitrage Scanner MVP starting up.  Interval=%ds  DB=%s",
        args.interval, db_path,
    )

    # Optional: inject Kalshi API key as Authorization header if provided
    headers: dict[str, str] = {}
    kalshi_api_key = os.getenv("KALSHI_API_KEY")
    if kalshi_api_key:
        headers["Authorization"] = f"Bearer {kalshi_api_key}"
        logger.info("Kalshi API key loaded from environment.")

    # Shared aiohttp session — reused across all scan cycles for efficiency
    connector = aiohttp.TCPConnector(
        limit=20,           # max total concurrent connections
        ttl_dns_cache=300,  # cache DNS resolutions for 5 minutes
    )
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(
        headers=headers,
        connector=connector,
        timeout=timeout,
    ) as session:
        kalshi = KalshiClient(session)
        poly = PolymarketClient(session)

        scan_count = 0
        total_logged = 0

        while True:
            scan_count += 1
            logger.info("--- Starting scan #%d ---", scan_count)

            try:
                n_logged = await run_scan_cycle(kalshi, poly, db_path)
                total_logged += n_logged
            except asyncio.CancelledError:
                logger.info("Scan loop cancelled. Shutting down cleanly.")
                break
            except Exception as exc:
                # Log and continue — never let a single scan crash the loop
                logger.exception(
                    "Unhandled exception in scan cycle #%d: %s", scan_count, exc
                )

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
        "--interval",
        type=int,
        default=SCAN_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=f"Seconds between scan cycles (default: {SCAN_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        metavar="PATH",
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--show-recent",
        type=int,
        default=0,
        metavar="N",
        help="Print the N most recent logged opportunities and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _configure_logging(args.verbose)

    # --show-recent mode: query DB and exit without starting the scan loop
    if args.show_recent > 0:
        rows = fetch_recent_opportunities(limit=args.show_recent, db_path=Path(args.db))
        if not rows:
            print("No opportunities found in database.")
        else:
            print(f"{'timestamp':<26} {'sport':<8} {'margin%':>8}  event")
            print("-" * 90)
            for row in rows:
                print(
                    f"{row['timestamp']:<26} {row['sport']:<8} "
                    f"{row['arb_margin_pct']:>7.2f}%  "
                    f"K-{row['kalshi_side']}@{row['kalshi_price']:.3f} "
                    f"vs P-{row['poly_side']}@{row['poly_price']:.3f}  "
                    f"{row['matched_event_name'][:50]}"
                )
        sys.exit(0)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user (KeyboardInterrupt).")
        sys.exit(0)
