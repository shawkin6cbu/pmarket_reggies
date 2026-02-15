#!/usr/bin/env python3
"""
ingest.py - Polymarket Data Ingestion

This script does two things:
1. BACKFILL: Pull historical price data for active markets
2. POLL: Continuously capture live prices (run as daemon later)

Run modes:
    python ingest.py backfill     # One-time historical pull
    python ingest.py poll         # Continuous live capture (Ctrl+C to stop)
    python ingest.py status       # Show what's in the database

=============================================================================
HOW POLYMARKET'S API STRUCTURE WORKS
=============================================================================

There are TWO separate APIs you need to understand:

1. GAMMA API (https://gamma-api.polymarket.com)
   - This is the "metadata" API
   - Returns market info: question text, volume, whether it's active, etc.
   - Each market has a `clobTokenIds` field — this is what you need for prices
   - A market can have multiple CLOB tokens (Yes and No are separate tokens)

2. CLOB API (https://clob.polymarket.com)  
   - This is the "trading" API
   - Returns actual prices and historical data
   - You query it with a `clob_token_id` (NOT the market ID)
   - The `/prices-history` endpoint gives you historical timeseries

The relationship:
    
    Gamma API                          CLOB API
    ---------                          --------
    Market "Will X happen?"  ------>   Token ID abc123 (YES token)
         |                             Token ID def456 (NO token)
         |
         +-- question: "Will X happen?"
         +-- volume: $50,000
         +-- clobTokenIds: ["abc123", "def456"]
         +-- active: true
         +-- closed: false

For our purposes, we only care about the YES token price (p_yes).
The NO token is just (1 - p_yes), so it's redundant.

=============================================================================
WHAT THE HISTORICAL ENDPOINT RETURNS
=============================================================================

GET https://clob.polymarket.com/prices-history
    ?market=<clob_token_id>
    &interval=max          # Get all available history
    &fidelity=5            # 5-minute resolution

Response:
{
    "history": [
        {"t": 1697875200, "p": 0.45},   # t = unix timestamp, p = price
        {"t": 1697875500, "p": 0.46},   # 5 minutes later
        {"t": 1697875800, "p": 0.44},   # 5 minutes later
        ...
    ]
}

The price `p` is always between 0 and 1 (0 = 0%, 1 = 100%).

=============================================================================
"""

import sqlite3
import requests
import time
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Rate limiting - be nice to their servers
GAMMA_DELAY = 0.1      # 100ms between Gamma requests
CLOB_DELAY = 0.15      # 150ms between CLOB requests (they rate limit harder)

# Filtering criteria for markets
MIN_VOLUME = 10_000    # Minimum $10k volume (filters dead markets)
MAX_MARKETS = 200      # Don't pull more than this many markets

# Polling configuration (for live mode)
POLL_INTERVAL = 300    # 5 minutes = 300 seconds

# Database path (relative to project root)
DB_PATH = Path(__file__).parent.parent / "data" / "polymarket.db"


# =============================================================================
# DATABASE SETUP
# =============================================================================

def get_db_connection() -> sqlite3.Connection:
    """
    Create or connect to the SQLite database.
    
    SQLite is perfect for this project because:
    - Zero setup (it's just a file)
    - ACID compliant (won't corrupt on crash)
    - Fast enough for millions of rows
    - You can query it with regular SQL
    
    The database file lives at: data/polymarket.db
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    return conn


def init_database(conn: sqlite3.Connection) -> None:
    """
    Create tables if they don't exist.
    
    Two tables:
    
    1. `markets` - Metadata about each market
       - Updated periodically (markets can become inactive, close, etc.)
       - One row per market
    
    2. `snapshots` - Price observations
       - Append-only (NEVER delete or update)
       - One row per (market, timestamp) pair
       - This is your raw data; everything else is derived from this
    
    Why append-only for snapshots?
    - You can always regenerate features and labels from raw data
    - If you find a bug in your feature code, just re-run it
    - Historical data never changes, so why would you edit it?
    """
    cursor = conn.cursor()
    
    # Markets table - metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            clob_token_id TEXT PRIMARY KEY,
            condition_id TEXT,
            question TEXT,
            slug TEXT,
            volume REAL,
            active INTEGER,
            closed INTEGER,
            end_date TEXT,
            outcome TEXT,           -- 'Yes' or 'No' (we only store Yes tokens)
            market_slug TEXT,       -- URL-friendly name
            updated_at INTEGER      -- When we last refreshed this row
        )
    """)
    
    # Snapshots table - price timeseries (append-only!)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clob_token_id TEXT NOT NULL,
            ts INTEGER NOT NULL,
            price REAL NOT NULL,
            source TEXT NOT NULL,
            fetched_at INTEGER NOT NULL,
            UNIQUE(clob_token_id, ts)
        )
    """)
    
    # Indexes for fast queries
    # 
    # Why these indexes matter:
    # - idx_snapshots_market_ts: When computing features, you query 
    #   "give me all prices for market X between time A and B"
    #   Without this index: full table scan (slow)
    #   With this index: direct lookup (fast)
    #
    # - idx_snapshots_ts: For queries like "what happened across all markets
    #   at time T" (useful for correlation analysis later)
    #
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_market_ts 
        ON snapshots(clob_token_id, ts)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_ts 
        ON snapshots(ts)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_markets_active 
        ON markets(active, closed)
    """)
    
    conn.commit()
    print(f"Database initialized at: {DB_PATH}")


# =============================================================================
# API FUNCTIONS
# =============================================================================

def fetch_markets() -> list[dict]:
    """
    Fetch all active markets from the Gamma API.
    
    The Gamma API uses cursor-based pagination:
    - First request: GET /markets?limit=100&active=true&closed=false
    - Response includes a `next_cursor` field
    - Next request: GET /markets?limit=100&cursor=<next_cursor>
    - Repeat until no more cursor
    
    We filter for:
    - active=true (market is accepting trades)
    - closed=false (market hasn't resolved yet)
    
    Returns a list of market dicts with fields like:
    - clobTokenIds: list of token IDs for this market
    - question: "Will X happen by Y date?"
    - volume: total trading volume in dollars
    - slug: URL-friendly identifier
    """
    print("Fetching markets from Gamma API...")
    
    all_markets = []
    cursor = None
    page = 1
    
    while True:
        # Build request URL
        params = {
            "limit": 100,
            "active": "true",
            "closed": "false",
        }
        if cursor:
            params["cursor"] = cursor
            
        url = f"{GAMMA_API}/markets"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching markets: {e}")
            break
            
        data = response.json()
        
        # Handle both list response and paginated response
        if isinstance(data, list):
            markets = data
            cursor = None
        else:
            markets = data.get("data", data.get("markets", []))
            cursor = data.get("next_cursor")
        
        if not markets:
            break
            
        all_markets.extend(markets)
        print(f"  Page {page}: fetched {len(markets)} markets (total: {len(all_markets)})")
        
        page += 1
        time.sleep(GAMMA_DELAY)
        
        # Stop if no more pages or we hit our limit
        if not cursor or len(all_markets) >= MAX_MARKETS:
            break
    
    # Filter by volume
    high_volume = [m for m in all_markets if float(m.get("volume") or 0) >= MIN_VOLUME]
    print(f"Filtered to {len(high_volume)} markets with volume >= ${MIN_VOLUME:,}")
    
    return high_volume[:MAX_MARKETS]


def fetch_price_history(clob_token_id: str, fidelity: int = 5) -> list[dict]:
    """
    Fetch historical prices for a single CLOB token.
    
    Parameters:
    - clob_token_id: The token to fetch (from market's clobTokenIds)
    - fidelity: Resolution in minutes (5 = one data point per 5 minutes)
    
    Returns list of {"t": unix_timestamp, "p": price} dicts.
    
    The API parameters:
    - market: The CLOB token ID (confusingly named, it's not the market ID)
    - interval: "max" means "give me all available history"
    - fidelity: Resolution in minutes
    
    Note: The API might not have data going back forever. Typically you get
    a few months of history for active markets.
    """
    url = f"{CLOB_API}/prices-history"
    params = {
        "market": clob_token_id,
        "interval": "max",
        "fidelity": fidelity,
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("history", [])
    except requests.RequestException as e:
        print(f"  Error fetching history for {clob_token_id}: {e}")
        return []


def fetch_current_price(clob_token_id: str) -> Optional[float]:
    """
    Fetch the current midpoint price for a token.
    
    Used for live polling. The midpoint is (best_bid + best_ask) / 2,
    which is the "fair" price between buyers and sellers.
    
    Returns None if the request fails.
    """
    url = f"{CLOB_API}/midpoint"
    params = {"token_id": clob_token_id}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data.get("mid", data.get("midpoint", 0)))
    except (requests.RequestException, ValueError, TypeError) as e:
        print(f"  Error fetching price for {clob_token_id}: {e}")
        return None


# =============================================================================
# DATA STORAGE FUNCTIONS
# =============================================================================

def save_market(conn: sqlite3.Connection, market: dict) -> Optional[str]:
    """
    Save or update a market's metadata.
    
    Returns the YES token's clob_token_id, or None if no valid token found.
    
    Why we only care about the YES token:
    - Every binary market has a YES and NO token
    - Their prices always sum to ~$1 (minus small spread)
    - So if p_yes = 0.60, then p_no = 0.40
    - Storing both is redundant; we only need one
    - Convention: we store the YES token
    """
    # Extract the YES token ID
    # NOTE: The API returns these as JSON strings, not lists!
    clob_token_ids_raw = market.get("clobTokenIds", "[]")
    outcomes_raw = market.get("outcomes", '["Yes", "No"]')
    
    # Parse JSON strings if needed
    if isinstance(clob_token_ids_raw, str):
        try:
            clob_token_ids = json.loads(clob_token_ids_raw)
        except json.JSONDecodeError:
            clob_token_ids = []
    else:
        clob_token_ids = clob_token_ids_raw or []
    
    if isinstance(outcomes_raw, str):
        try:
            outcomes = json.loads(outcomes_raw)
        except json.JSONDecodeError:
            outcomes = ["Yes", "No"]
    else:
        outcomes = outcomes_raw or ["Yes", "No"]
    
    # Find the YES token
    yes_token_id = None
    if len(clob_token_ids) >= 1:
        # Usually the first token is YES, but let's be careful
        if len(outcomes) >= 1 and outcomes[0].lower() == "yes":
            yes_token_id = clob_token_ids[0]
        elif len(clob_token_ids) >= 2 and len(outcomes) >= 2:
            # Find which index has "Yes"
            for i, outcome in enumerate(outcomes):
                if outcome.lower() == "yes" and i < len(clob_token_ids):
                    yes_token_id = clob_token_ids[i]
                    break
        
        # Fallback: just use the first token
        if not yes_token_id:
            yes_token_id = clob_token_ids[0]
    
    if not yes_token_id:
        return None
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO markets (
            clob_token_id, condition_id, question, slug, volume,
            active, closed, end_date, outcome, market_slug, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(clob_token_id) DO UPDATE SET
            volume = excluded.volume,
            active = excluded.active,
            closed = excluded.closed,
            updated_at = excluded.updated_at
    """, (
        yes_token_id,
        market.get("conditionId"),
        market.get("question"),
        market.get("slug"),
        float(market.get("volume") or 0),
        1 if market.get("active") else 0,
        1 if market.get("closed") else 0,
        market.get("endDate"),
        "Yes",
        market.get("slug"),
        int(time.time()),
    ))
    conn.commit()
    
    return yes_token_id


def save_snapshots(
    conn: sqlite3.Connection,
    clob_token_id: str,
    history: list[dict],
    source: str = "historical"
) -> int:
    """
    Save price history to the snapshots table.
    
    Uses INSERT OR IGNORE to skip duplicates — this is crucial because:
    - You might run backfill multiple times
    - Historical data doesn't change
    - We don't want to fail or create duplicates
    
    The UNIQUE(clob_token_id, ts) constraint in the schema ensures
    we never have two rows for the same market at the same timestamp.
    
    Returns the number of NEW rows inserted (not counting skipped duplicates).
    """
    if not history:
        return 0
    
    cursor = conn.cursor()
    now = int(time.time())
    
    # Prepare batch insert
    rows = [
        (clob_token_id, point["t"], point["p"], source, now)
        for point in history
        if "t" in point and "p" in point
    ]
    
    # Get count before insert
    cursor.execute(
        "SELECT COUNT(*) FROM snapshots WHERE clob_token_id = ?",
        (clob_token_id,)
    )
    count_before = cursor.fetchone()[0]
    
    # Batch insert with ignore duplicates
    cursor.executemany("""
        INSERT OR IGNORE INTO snapshots (clob_token_id, ts, price, source, fetched_at)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    
    # Get count after insert
    cursor.execute(
        "SELECT COUNT(*) FROM snapshots WHERE clob_token_id = ?",
        (clob_token_id,)
    )
    count_after = cursor.fetchone()[0]
    
    return count_after - count_before


# =============================================================================
# MAIN COMMANDS
# =============================================================================

def cmd_backfill():
    """
    One-time historical data pull.
    
    This is what you run first to populate your database with months
    of historical data. After this, you have enough to start training.
    
    Steps:
    1. Fetch list of active, high-volume markets from Gamma
    2. For each market, fetch full price history from CLOB
    3. Store everything in SQLite
    
    This might take 10-30 minutes depending on how many markets
    and how much history each has. The script prints progress so
    you can see it's working.
    """
    print("=" * 60)
    print("BACKFILL MODE - Pulling historical data")
    print("=" * 60)
    
    conn = get_db_connection()
    init_database(conn)
    
    # Step 1: Get markets
    markets = fetch_markets()
    if not markets:
        print("No markets found. Check your network connection.")
        return
    
    print(f"\nWill fetch history for {len(markets)} markets...")
    print("-" * 60)
    
    # Step 2: Fetch history for each market
    total_snapshots = 0
    successful_markets = 0
    
    for i, market in enumerate(markets, 1):
        question = market.get("question", "Unknown")[:50]
        volume = float(market.get("volume") or 0)
        
        print(f"\n[{i}/{len(markets)}] {question}...")
        print(f"  Volume: ${volume:,.0f}")
        
        # Save market metadata
        token_id = save_market(conn, market)
        if not token_id:
            print("  ⚠ No valid token ID found, skipping")
            continue
        
        print(f"  Token: {token_id[:20]}...")
        
        # Fetch price history
        history = fetch_price_history(token_id)
        if not history:
            print("  ⚠ No history available")
            time.sleep(CLOB_DELAY)
            continue
        
        # Save snapshots
        new_rows = save_snapshots(conn, token_id, history, source="historical")
        total_snapshots += new_rows
        successful_markets += 1
        
        # Calculate date range
        timestamps = [p["t"] for p in history]
        start_date = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        end_date = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)
        
        print(f"  ✓ Saved {new_rows} snapshots ({len(history)} total in response)")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        
        # Rate limit
        time.sleep(CLOB_DELAY)
    
    # Final summary
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Markets processed: {successful_markets}/{len(markets)}")
    print(f"Total snapshots saved: {total_snapshots:,}")
    print(f"Database location: {DB_PATH}")
    
    # Show database stats
    cmd_status()


def cmd_poll():
    """
    Continuous live polling mode.
    
    Run this AFTER backfill to keep your data fresh. It will:
    1. Load all active markets from the database
    2. Every 5 minutes, fetch current price for each
    3. Save to snapshots table with source='live'
    4. Repeat forever (Ctrl+C to stop)
    
    Why poll instead of websocket?
    - Simpler to implement and debug
    - 5-minute resolution doesn't need real-time
    - If the script crashes, you just restart it (no state to recover)
    - Websockets require handling disconnects, reconnects, etc.
    
    For a first project, polling is the right choice.
    """
    print("=" * 60)
    print("POLL MODE - Live data capture")
    print(f"Polling every {POLL_INTERVAL} seconds (Ctrl+C to stop)")
    print("=" * 60)
    
    conn = get_db_connection()
    init_database(conn)
    
    # Get list of active markets from database
    cursor = conn.cursor()
    cursor.execute("""
        SELECT clob_token_id, question FROM markets 
        WHERE active = 1 AND closed = 0
    """)
    markets = cursor.fetchall()
    
    if not markets:
        print("No active markets in database. Run 'backfill' first.")
        return
    
    print(f"Tracking {len(markets)} active markets\n")
    
    poll_count = 0
    try:
        while True:
            poll_count += 1
            now = int(time.time())
            now_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            
            print(f"[Poll #{poll_count}] {now_str}")
            
            saved = 0
            errors = 0
            
            for token_id, question in markets:
                price = fetch_current_price(token_id)
                
                if price is not None:
                    # Save single snapshot
                    save_snapshots(
                        conn, 
                        token_id, 
                        [{"t": now, "p": price}],
                        source="live"
                    )
                    saved += 1
                else:
                    errors += 1
                
                time.sleep(CLOB_DELAY)
            
            print(f"  Saved: {saved}, Errors: {errors}")
            
            # Wait until next poll
            elapsed = time.time() - now
            sleep_time = max(0, POLL_INTERVAL - elapsed)
            print(f"  Sleeping {sleep_time:.0f}s until next poll...\n")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\nPolling stopped by user.")
        cmd_status()


def cmd_status():
    """
    Show database statistics.
    
    Quick health check to see:
    - How many markets you're tracking
    - How many snapshots you have
    - Date range of your data
    - Data source breakdown (historical vs live)
    """
    print("\n" + "-" * 60)
    print("DATABASE STATUS")
    print("-" * 60)
    
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run 'python ingest.py backfill' first.")
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Market counts
    cursor.execute("SELECT COUNT(*) FROM markets")
    total_markets = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM markets WHERE active = 1 AND closed = 0")
    active_markets = cursor.fetchone()[0]
    
    # Snapshot counts
    cursor.execute("SELECT COUNT(*) FROM snapshots")
    total_snapshots = cursor.fetchone()[0]
    
    cursor.execute("SELECT source, COUNT(*) FROM snapshots GROUP BY source")
    by_source = dict(cursor.fetchall())
    
    # Date range
    cursor.execute("SELECT MIN(ts), MAX(ts) FROM snapshots")
    min_ts, max_ts = cursor.fetchone()
    
    print(f"Markets tracked:     {total_markets:,} ({active_markets} active)")
    print(f"Total snapshots:     {total_snapshots:,}")
    print(f"  Historical:        {by_source.get('historical', 0):,}")
    print(f"  Live:              {by_source.get('live', 0):,}")
    
    if min_ts and max_ts:
        min_date = datetime.fromtimestamp(min_ts, tz=timezone.utc)
        max_date = datetime.fromtimestamp(max_ts, tz=timezone.utc)
        print(f"Date range:          {min_date.date()} to {max_date.date()}")
    
    # Sample some markets
    print("\nTop 5 markets by snapshot count:")
    cursor.execute("""
        SELECT m.question, COUNT(*) as cnt
        FROM snapshots s
        JOIN markets m ON s.clob_token_id = m.clob_token_id
        GROUP BY s.clob_token_id
        ORDER BY cnt DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        q = row[0][:45] if row[0] else "Unknown"
        print(f"  {row[1]:>8,} - {q}...")
    
    print("-" * 60)
    conn.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Command-line interface.
    
    Usage:
        python ingest.py backfill   # Pull historical data (run this first)
        python ingest.py poll       # Start live polling
        python ingest.py status     # Check database stats
    """
    parser = argparse.ArgumentParser(
        description="Polymarket data ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest.py backfill   # Pull historical data (run this first!)
    python ingest.py poll       # Start live polling (run in background)
    python ingest.py status     # Check what's in the database
        """
    )
    parser.add_argument(
        "command",
        choices=["backfill", "poll", "status"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    if args.command == "backfill":
        cmd_backfill()
    elif args.command == "poll":
        cmd_poll()
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()