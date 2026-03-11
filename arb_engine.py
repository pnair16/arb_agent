"""
arb_engine.py
=============
Arbitrage calculator and SQLite persistence layer.

Core formula
------------
A cross-platform arbitrage opportunity exists when you can simultaneously
back OPPOSITE sides of the same event on two different platforms and
guarantee a profit regardless of outcome:

    Back YES on platform A at implied probability P_A_yes
    Back NO  on platform B at implied probability P_B_no

    Total cost per unit = (1 / P_A_yes) + (1 / P_B_no)

    If total_cost < 1.0  → arbitrage exists
    Margin % = (1 - total_cost) * 100

We evaluate both directions (Kalshi-YES / Poly-NO and Kalshi-NO / Poly-YES)
and log all opportunities found above a configurable minimum margin.

Database schema
---------------
Table: arb_opportunities
  id                   INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp            TEXT    (ISO-8601, UTC)
  sport                TEXT
  matched_event_name   TEXT
  kalshi_ticker        TEXT
  poly_id              TEXT
  kalshi_side          TEXT    ('YES' or 'NO')
  kalshi_price         REAL    (implied probability)
  poly_side            TEXT    ('YES' or 'NO')
  poly_price           REAL    (implied probability)
  arb_margin_pct       REAL    (percentage, positive = opportunity)
  match_score          INTEGER (fuzzy match confidence 0-100)
"""

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from normalizer import NormalizedMarket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path("arb_opportunities.db")

# Only log rows where the arb margin exceeds this threshold (in percent).
# Set to a negative value to capture ALL scanned pairs, including non-arb.
MIN_ARB_MARGIN_PCT = -5.0  # log anything with margin > -5% for research purposes


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArbOpportunity:
    """
    Represents a single evaluated cross-platform combination.

    arb_margin_pct > 0  → genuine arbitrage opportunity
    arb_margin_pct ≤ 0  → no arb, but near-miss data still valuable
    """
    timestamp: str
    sport: str
    matched_event_name: str
    kalshi_ticker: str
    poly_id: str
    kalshi_side: str       # 'YES' or 'NO'
    kalshi_price: float    # implied probability
    poly_side: str         # 'YES' or 'NO' (the OPPOSING side)
    poly_price: float      # implied probability
    arb_margin_pct: float  # (1 - 1/P_K - 1/P_P) * 100
    match_score: int       # fuzzy match confidence


# ---------------------------------------------------------------------------
# Database management
# ---------------------------------------------------------------------------

def init_db(db_path: Path = DB_PATH) -> None:
    """
    Create the SQLite database and the arb_opportunities table if they
    do not already exist.  Safe to call on every startup (idempotent).
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arb_opportunities (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                sport               TEXT    NOT NULL,
                matched_event_name  TEXT    NOT NULL,
                kalshi_ticker       TEXT    NOT NULL,
                poly_id             TEXT    NOT NULL,
                kalshi_side         TEXT    NOT NULL,
                kalshi_price        REAL    NOT NULL,
                poly_side           TEXT    NOT NULL,
                poly_price          REAL    NOT NULL,
                arb_margin_pct      REAL    NOT NULL,
                match_score         INTEGER NOT NULL
            )
        """)
        # Index for fast retrieval by timestamp and sport
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_arb_timestamp
            ON arb_opportunities (timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_arb_sport
            ON arb_opportunities (sport)
        """)
        conn.commit()
    logger.info("Database initialised at %s", db_path)


@contextmanager
def _get_connection(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields an open SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _insert_opportunity(conn: sqlite3.Connection, opp: ArbOpportunity) -> None:
    """Insert a single ArbOpportunity row into the database."""
    conn.execute(
        """
        INSERT INTO arb_opportunities
            (timestamp, sport, matched_event_name, kalshi_ticker, poly_id,
             kalshi_side, kalshi_price, poly_side, poly_price,
             arb_margin_pct, match_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            opp.timestamp,
            opp.sport,
            opp.matched_event_name,
            opp.kalshi_ticker,
            opp.poly_id,
            opp.kalshi_side,
            opp.kalshi_price,
            opp.poly_side,
            opp.poly_price,
            opp.arb_margin_pct,
            opp.match_score,
        ),
    )


# ---------------------------------------------------------------------------
# Arbitrage calculation
# ---------------------------------------------------------------------------

def _calc_arb_margin(prob_a: float, prob_b: float) -> float:
    """
    Calculate the arbitrage margin for backing prob_a on platform A and
    prob_b on platform B (opposing sides).

    margin = (1 - 1/prob_a - 1/prob_b) * 100

    A positive margin means guaranteed profit; negative means a loss.
    """
    if prob_a <= 0.0 or prob_b <= 0.0:
        return float("-inf")
    return round((1.0 - (1.0 / prob_a) - (1.0 / prob_b)) * 100.0, 4)


def evaluate_and_log(
    markets: list[NormalizedMarket],
    db_path: Path = DB_PATH,
    min_margin_pct: float = MIN_ARB_MARGIN_PCT,
) -> list[ArbOpportunity]:
    """
    Evaluate all matched market pairs for arbitrage, log qualifying rows to
    SQLite, and return the list of logged opportunities.

    For each matched pair we evaluate TWO directions:
      Direction 1: Back Kalshi-YES vs. Polymarket-NO
      Direction 2: Back Kalshi-NO  vs. Polymarket-YES

    Parameters
    ----------
    markets       : List of NormalizedMarket from the normalizer module.
    db_path       : Path to the SQLite database file.
    min_margin_pct: Only log rows whose margin exceeds this threshold.

    Returns
    -------
    List of ArbOpportunity objects that were persisted to the database.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    logged: list[ArbOpportunity] = []

    with _get_connection(db_path) as conn:
        for mkt in markets:
            # Skip markets with missing probability data
            if any(
                v is None
                for v in (
                    mkt.kalshi_prob_yes, mkt.kalshi_prob_no,
                    mkt.poly_prob_yes, mkt.poly_prob_no,
                )
            ):
                logger.debug(
                    "Skipping '%s': missing probability data.", mkt.event_name
                )
                continue

            # Annotate types after None-guard (mypy hint)
            k_yes: float = mkt.kalshi_prob_yes  # type: ignore[assignment]
            k_no: float = mkt.kalshi_prob_no    # type: ignore[assignment]
            p_yes: float = mkt.poly_prob_yes    # type: ignore[assignment]
            p_no: float = mkt.poly_prob_no      # type: ignore[assignment]

            # --- Direction 1: Kalshi YES ↔ Polymarket NO ---
            margin_d1 = _calc_arb_margin(k_yes, p_no)

            opp_d1 = ArbOpportunity(
                timestamp=now_utc,
                sport=mkt.sport,
                matched_event_name=mkt.event_name,
                kalshi_ticker=mkt.kalshi_ticker,
                poly_id=mkt.poly_id,
                kalshi_side="YES",
                kalshi_price=k_yes,
                poly_side="NO",
                poly_price=p_no,
                arb_margin_pct=margin_d1,
                match_score=mkt.match_score,
            )

            if margin_d1 > min_margin_pct:
                _insert_opportunity(conn, opp_d1)
                logged.append(opp_d1)
                _log_result(opp_d1)

            # --- Direction 2: Kalshi NO ↔ Polymarket YES ---
            margin_d2 = _calc_arb_margin(k_no, p_yes)

            opp_d2 = ArbOpportunity(
                timestamp=now_utc,
                sport=mkt.sport,
                matched_event_name=mkt.event_name,
                kalshi_ticker=mkt.kalshi_ticker,
                poly_id=mkt.poly_id,
                kalshi_side="NO",
                kalshi_price=k_no,
                poly_side="YES",
                poly_price=p_yes,
                arb_margin_pct=margin_d2,
                match_score=mkt.match_score,
            )

            if margin_d2 > min_margin_pct:
                _insert_opportunity(conn, opp_d2)
                logged.append(opp_d2)
                _log_result(opp_d2)

    if logged:
        logger.info("Logged %d rows to database (min_margin=%.2f%%).", len(logged), min_margin_pct)
    else:
        logger.info("No rows above min_margin threshold (%.2f%%) this scan.", min_margin_pct)

    return logged


def _log_result(opp: ArbOpportunity) -> None:
    """Emit a human-readable log line for an opportunity."""
    flag = "*** ARB ***" if opp.arb_margin_pct > 0 else "near-miss"
    logger.info(
        "[%s] %s | %s | K-%s @ %.4f  ↔  P-%s @ %.4f | margin=%.2f%%",
        flag,
        opp.sport,
        opp.matched_event_name[:60],
        opp.kalshi_side,
        opp.kalshi_price,
        opp.poly_side,
        opp.poly_price,
        opp.arb_margin_pct,
    )


# ---------------------------------------------------------------------------
# Convenience query helpers (for external analysis / debugging)
# ---------------------------------------------------------------------------

def fetch_recent_opportunities(
    limit: int = 50,
    only_positive: bool = False,
    db_path: Path = DB_PATH,
) -> list[sqlite3.Row]:
    """
    Return the most recent logged rows from the database.

    Parameters
    ----------
    limit         : Maximum number of rows to return.
    only_positive : If True, filter to rows with arb_margin_pct > 0.
    db_path       : Path to the SQLite database file.
    """
    with _get_connection(db_path) as conn:
        where = "WHERE arb_margin_pct > 0" if only_positive else ""
        rows = conn.execute(
            f"""
            SELECT *
            FROM   arb_opportunities
            {where}
            ORDER  BY timestamp DESC
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
    return rows
