"""
diagnose.py
===========
Standalone diagnostic tool — fires one request to each API and pretty-prints
the raw JSON so you can see exactly what field names and structures are live.

Run:  python diagnose.py
"""

import asyncio
import json
import aiohttp

KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"


async def probe_kalshi(session: aiohttp.ClientSession) -> None:
    print("\n" + "=" * 60)
    print("KALSHI — single page (limit=3, status=open)")
    print("=" * 60)
    url = f"{KALSHI_BASE_URL}/markets"
    params = {"status": "open", "limit": 3}
    async with session.get(url, params=params) as r:
        print(f"HTTP status: {r.status}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        raw = await r.text()
        try:
            data = json.loads(raw)
        except Exception:
            print("RAW (non-JSON):", raw[:500])
            return

    markets = data.get("markets", data if isinstance(data, list) else [])
    print(f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else 'LIST'}")
    print(f"Markets in page: {len(markets)}")

    if markets:
        sample = markets[0]
        print("\n--- First market keys ---")
        print(json.dumps(list(sample.keys()), indent=2))
        print("\n--- First market (selected fields) ---")
        for field in [
            "ticker", "title", "subtitle", "series_ticker",
            "category", "status", "yes_bid", "yes_ask",
            "last_price", "result", "close_time", "event_ticker",
        ]:
            print(f"  {field!r}: {sample.get(field)!r}")

    # Also probe events endpoint
    print("\n--- Kalshi /events (limit=3) ---")
    async with session.get(
        f"{KALSHI_BASE_URL}/events",
        params={"status": "open", "limit": 3},
    ) as r2:
        print(f"HTTP status: {r2.status}")
        try:
            d2 = await r2.json(content_type=None)
            print(f"Top-level keys: {list(d2.keys()) if isinstance(d2, dict) else 'LIST'}")
            events = d2.get("events", [])
            if events:
                ev = events[0]
                print(f"Event keys: {list(ev.keys())}")
                print(f"  category: {ev.get('category')!r}")
                print(f"  series_ticker: {ev.get('series_ticker')!r}")
                print(f"  title: {ev.get('title')!r}")
        except Exception as exc:
            print(f"Could not parse: {exc}")


async def probe_polymarket(session: aiohttp.ClientSession) -> None:
    print("\n" + "=" * 60)
    print("POLYMARKET GAMMA — markets endpoint (limit=3)")
    print("=" * 60)
    url = f"{POLYMARKET_GAMMA_URL}/markets"
    params = {"active": "true", "closed": "false", "limit": 3}
    async with session.get(url, params=params) as r:
        print(f"HTTP status: {r.status}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        raw = await r.text()
        try:
            data = json.loads(raw)
        except Exception:
            print("RAW (non-JSON):", raw[:500])
            return

    markets = data if isinstance(data, list) else data.get("markets", [])
    print(f"Top-level type: {type(data).__name__}")
    print(f"Markets in page: {len(markets)}")

    if markets:
        sample = markets[0]
        print("\n--- First market keys ---")
        print(json.dumps(list(sample.keys()), indent=2))
        print("\n--- First market (selected fields) ---")
        for field in [
            "id", "question", "outcomes", "outcomePrices",
            "active", "closed", "tags", "groupItemTitle",
            "clobTokenIds", "bestBid", "bestAsk", "lastTradePrice",
        ]:
            val = sample.get(field)
            # Truncate long values
            display = repr(val)[:120] if val is not None else "MISSING"
            print(f"  {field!r}: {display}")

    # Check tags endpoint
    print("\n--- Polymarket /tags ---")
    async with session.get(f"{POLYMARKET_GAMMA_URL}/tags") as r2:
        print(f"HTTP status: {r2.status}")
        try:
            tags_data = await r2.json(content_type=None)
            tags = tags_data if isinstance(tags_data, list) else tags_data.get("tags", [])
            sports_tags = [
                t for t in tags
                if "sport" in str(t).lower() or "nfl" in str(t).lower()
                or "nba" in str(t).lower() or "soccer" in str(t).lower()
            ]
            print(f"Total tags: {len(tags)}")
            print("Sports-related tags:")
            for t in sports_tags[:10]:
                print(f"  {t}")
        except Exception as exc:
            print(f"Could not parse tags: {exc}")

    # Now try with tag_id=6
    print("\n--- Polymarket /markets?tag_id=6 (limit=3) ---")
    async with session.get(url, params={"active": "true", "limit": 3, "tag_id": 6}) as r3:
        print(f"HTTP status: {r3.status}")
        try:
            d3 = await r3.json(content_type=None)
            page = d3 if isinstance(d3, list) else d3.get("markets", [])
            print(f"Markets with tag_id=6: {len(page)}")
            if page:
                print(f"  Sample question: {page[0].get('question')!r}")
        except Exception as exc:
            print(f"Could not parse: {exc}")


async def main() -> None:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=20),
        headers={"User-Agent": "arb-scanner-diagnostic/1.0"},
    ) as session:
        await probe_kalshi(session)
        await probe_polymarket(session)


if __name__ == "__main__":
    asyncio.run(main())
