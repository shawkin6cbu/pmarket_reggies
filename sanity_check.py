#!/usr/bin/env python3
"""
sanity_check.py - Verify your data looks reasonable after backfill

Run this AFTER ingest.py backfill to catch common problems:
- Missing data / gaps
- Duplicate timestamps
- Price values outside [0, 1]
- Markets with too little history

Usage:
    python sanity_check.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

DB_PATH = Path(__file__).parent / "data" / "polymarket.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run 'python ingest.py backfill' first.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 60)
    print("DATA SANITY CHECK")
    print("=" * 60)
    
    issues = []
    
    # -------------------------------------------------------------------------
    # Check 1: Price values in valid range [0, 1]
    # -------------------------------------------------------------------------
    print("\n[1] Checking price ranges...")
    
    cursor.execute("""
        SELECT COUNT(*) FROM snapshots WHERE price < 0 OR price > 1
    """)
    bad_prices = cursor.fetchone()[0]
    
    if bad_prices > 0:
        issues.append(f"❌ {bad_prices} snapshots have price outside [0, 1]")
        print(f"  ❌ Found {bad_prices} invalid prices")
    else:
        print("  ✓ All prices in valid range [0, 1]")
    
    # -------------------------------------------------------------------------
    # Check 2: Duplicate timestamps (should be 0 due to UNIQUE constraint)
    # -------------------------------------------------------------------------
    print("\n[2] Checking for duplicates...")
    
    cursor.execute("""
        SELECT clob_token_id, ts, COUNT(*) as cnt
        FROM snapshots
        GROUP BY clob_token_id, ts
        HAVING cnt > 1
        LIMIT 5
    """)
    duplicates = cursor.fetchall()
    
    if duplicates:
        issues.append(f"❌ Found duplicate (market, timestamp) pairs")
        for row in duplicates:
            print(f"  ❌ Token {row['clob_token_id'][:20]}... ts={row['ts']} appears {row['cnt']} times")
    else:
        print("  ✓ No duplicate timestamps")
    
    # -------------------------------------------------------------------------
    # Check 3: Markets with very little data
    # -------------------------------------------------------------------------
    print("\n[3] Checking data coverage per market...")
    
    cursor.execute("""
        SELECT 
            m.clob_token_id,
            m.question,
            COUNT(s.id) as snapshot_count,
            MIN(s.ts) as first_ts,
            MAX(s.ts) as last_ts
        FROM markets m
        LEFT JOIN snapshots s ON m.clob_token_id = s.clob_token_id
        GROUP BY m.clob_token_id
        ORDER BY snapshot_count ASC
    """)
    
    markets_data = cursor.fetchall()
    
    sparse_markets = [m for m in markets_data if m['snapshot_count'] < 100]
    if sparse_markets:
        print(f"  ⚠ {len(sparse_markets)} markets have < 100 snapshots:")
        for m in sparse_markets[:5]:
            q = m['question'][:40] if m['question'] else "Unknown"
            print(f"    - {m['snapshot_count']:>5} snapshots: {q}...")
        if len(sparse_markets) > 5:
            print(f"    ... and {len(sparse_markets) - 5} more")
    else:
        print("  ✓ All markets have 100+ snapshots")
    
    # -------------------------------------------------------------------------
    # Check 4: Gap analysis (missing data points)
    # -------------------------------------------------------------------------
    print("\n[4] Checking for gaps in timeseries...")
    
    # Sample a few markets for gap analysis
    cursor.execute("""
        SELECT clob_token_id, question FROM markets 
        WHERE clob_token_id IN (
            SELECT clob_token_id FROM snapshots 
            GROUP BY clob_token_id 
            HAVING COUNT(*) > 500
            LIMIT 5
        )
    """)
    sample_markets = cursor.fetchall()
    
    expected_interval = 5 * 60  # 5 minutes in seconds
    gap_threshold = 60 * 60    # Flag gaps > 1 hour
    
    for market in sample_markets:
        cursor.execute("""
            SELECT ts FROM snapshots 
            WHERE clob_token_id = ? 
            ORDER BY ts
        """, (market['clob_token_id'],))
        
        timestamps = [row[0] for row in cursor.fetchall()]
        
        large_gaps = []
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            if gap > gap_threshold:
                large_gaps.append(gap)
        
        q = market['question'][:35] if market['question'] else "Unknown"
        if large_gaps:
            max_gap_hours = max(large_gaps) / 3600
            print(f"  ⚠ {q}...")
            print(f"      {len(large_gaps)} gaps > 1 hour (max: {max_gap_hours:.1f}h)")
        else:
            print(f"  ✓ {q}... - no major gaps")
    
    # -------------------------------------------------------------------------
    # Check 5: Recent data freshness
    # -------------------------------------------------------------------------
    print("\n[5] Checking data freshness...")
    
    cursor.execute("SELECT MAX(ts) FROM snapshots")
    latest_ts = cursor.fetchone()[0]
    
    if latest_ts:
        latest_date = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        age_days = (now - latest_date).days
        
        print(f"  Most recent data: {latest_date.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Data age: {age_days} days")
        
        if age_days > 7:
            issues.append(f"⚠ Data is {age_days} days old - consider running backfill again")
    
    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    cursor.execute("SELECT COUNT(*) FROM markets")
    print(f"Total markets:     {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM snapshots")
    print(f"Total snapshots:   {cursor.fetchone()[0]:,}")
    
    cursor.execute("SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM snapshots GROUP BY clob_token_id)")
    avg = cursor.fetchone()[0]
    if avg:
        print(f"Avg per market:    {avg:,.0f}")
    
    cursor.execute("SELECT MIN(ts), MAX(ts) FROM snapshots")
    min_ts, max_ts = cursor.fetchone()
    if min_ts and max_ts:
        min_date = datetime.fromtimestamp(min_ts, tz=timezone.utc)
        max_date = datetime.fromtimestamp(max_ts, tz=timezone.utc)
        days = (max_date - min_date).days
        print(f"Date range:        {min_date.date()} to {max_date.date()} ({days} days)")
    
    # -------------------------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ ALL CHECKS PASSED")
        print("\nYour data looks good! You can proceed to Chunk 2 (EDA).")
    print("=" * 60)
    
    conn.close()


if __name__ == "__main__":
    main()