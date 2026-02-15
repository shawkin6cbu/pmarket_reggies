"""
features.py — Chunk 3: Feature Engineering v1

Turns raw price snapshots into model-ready rows.
Each row represents ONE prediction opportunity: a (market, timestamp) pair.

=== WHAT THIS SCRIPT DOES ===

For every snapshot in the database, it computes features by looking BACKWARD
in time (using only information available at that moment), and a label by
looking FORWARD in time (what actually happened next).

    PAST ◄────────────── NOW ──────────────► FUTURE
    [features computed    │    label computed
     from this window]    │    from this window]

This separation is critical. Features can only use past data.
Labels use future data. Mixing them up is called "leakage" and it
will make your model look amazing in training and worthless in production.

=== FEATURES WE COMPUTE ===

1. PRICE LEVEL
   - p: current price (0 to 1)
   - p_complement: 1 - p (distance from certainty)
   - p_from_center: |p - 0.5| (distance from maximum uncertainty)

   Why: Markets near 0 or 1 behave differently than markets near 0.5.
   A market at 0.95 can't go up much more. A market at 0.50 has max
   uncertainty and tends to be more volatile.

2. PRICE VELOCITY (Δp over multiple windows)
   - dp_3: price change over last 3 samples (30 min)
   - dp_6: price change over last 6 samples (1 hour)
   - dp_12: price change over last 12 samples (2 hours)
   - dp_36: price change over last 36 samples (6 hours)

   Why: Momentum. If price has been going up, does it keep going?
   Multiple windows capture momentum at different timescales.
   Short windows = recent momentum. Long windows = trend.

3. PRICE ACCELERATION (Δ²p = change in the change)
   - ddp_3: acceleration over 30-min windows
   - ddp_6: acceleration over 1-hour windows

   Why: Is momentum increasing or fading? If dp_6 was +0.01 an hour
   ago and is +0.02 now, momentum is accelerating. If it was +0.02
   and is now +0.01, it's decelerating (potential reversal).

   Computed as: dp_N(now) - dp_N(N_samples_ago)
   Example for ddp_3: dp_3(now) - dp_3(3 samples ago)
                     = [p(t) - p(t-3)] - [p(t-3) - p(t-6)]

4. ROLLING VOLATILITY (standard deviation of returns over windows)
   - vol_6: volatility over last 1 hour
   - vol_12: volatility over last 2 hours
   - vol_36: volatility over last 6 hours

   Why: Volatile markets are more likely to move ≥ threshold.
   This is probably the single most predictive feature for "will
   it move?" questions. A market that's been dead flat for 6 hours
   is unlikely to suddenly jump.

   Computed as: std(price changes) over the window, NOT std(prices).
   We want volatility of returns, not volatility of levels.

5. RELATIVE POSITION FEATURES
   - p_vs_6h_high: (p - 6h_low) / (6h_high - 6h_low)
   - p_vs_6h_mean: p - rolling_mean_36

   Why: Is the current price near its recent high or low?
   Near the high = maybe exhausted, could reverse.
   Near the low = maybe bouncing, could rise.
   This captures mean-reversion vs trend-continuation patterns.

6. TIME FEATURES
   - hour_sin, hour_cos: time of day encoded as circular features
   - is_weekend: Saturday/Sunday indicator

   Why: Markets might behave differently during US trading hours vs
   overnight, or weekdays vs weekends (lower activity).
   
   We encode hour as sin/cos rather than raw number because hour 23
   and hour 0 are adjacent but numerically far apart. Sin/cos
   preserves this circular relationship.

7. TIME-TO-EXPIRY (if available)
   - days_to_expiry: calendar days until market end date
   - log_days_to_expiry: log(days_to_expiry + 1)

   Why: Markets approaching expiry behave very differently. They
   converge toward 0 or 1 as the outcome becomes more certain.
   A market with 2 days left is much more likely to make a big
   move than one with 6 months left (all else equal).

=== OUTPUT FORMAT ===

A CSV file with columns:
    clob_token_id, ts, [all features], label

One row per (market, timestamp). Rows with NaN features (not enough
history) or NaN labels (not enough future data) are dropped.

=== USAGE ===

    cd /mnt/ml-data/projects/polymarket-predictor
    python src/features.py

    # Or with custom parameters:
    python src/features.py --threshold 0.001 --horizon 36 --output data/features.csv

=== IMPORTANT NOTES ===

- Features look BACKWARD only (no leakage)
- Labels look FORWARD only
- Rows at the start of each market's history get dropped (not enough lookback)
- Rows at the end get dropped (not enough future for the label)
- This is expected and correct — you're trading a few thousand rows
  for guaranteed no-leakage
"""

import sqlite3
import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
DB_PATH = Path(__file__).parent.parent / "data" / "polymarket.db"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "features.csv"

# Prediction parameters (from EDA results in Chunk 2)
DEFAULT_THRESHOLD = 0.001   # 0.1 cent move
DEFAULT_HORIZON = 36        # 36 samples × 10 min = 6 hours

# Feature windows (in number of samples, where 1 sample = 10 minutes)
#
# Why these specific windows?
# - 3 samples = 30 min  (short-term noise / immediate momentum)
# - 6 samples = 1 hour  (medium-term momentum)
# - 12 samples = 2 hours (longer trend)
# - 36 samples = 6 hours (same as prediction horizon — "where were we 6h ago?")
#
# Rule of thumb: you want windows shorter than, equal to, and longer than
# your prediction horizon. This lets the model see patterns at multiple scales.
VELOCITY_WINDOWS = [3, 6, 12, 36]
ACCELERATION_WINDOWS = [3, 6]
VOLATILITY_WINDOWS = [6, 12, 36]

# Minimum samples required per market to include it
# We need at least max(lookback, horizon) samples on each end,
# plus enough in the middle for meaningful data.
# 36 (lookback) + 36 (horizon) + 50 (minimum usable rows) = 122
MIN_SAMPLES_PER_MARKET = 122


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load snapshots and market metadata from SQLite.
    
    Returns two DataFrames:
    - snapshots: all price observations (clob_token_id, ts, price)
    - markets: metadata including end_date for time-to-expiry
    
    We convert timestamps to proper datetime objects here so we don't
    have to deal with raw unix timestamps everywhere else.
    """
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run 'python src/ingest.py backfill' first.")
        sys.exit(1)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    print("Loading data from database...")
    
    # Load snapshots — only the columns we need for features
    snapshots = pd.read_sql("""
        SELECT clob_token_id, ts, price 
        FROM snapshots 
        ORDER BY clob_token_id, ts
    """, conn)
    
    # Load markets — we need end_date for time-to-expiry feature
    markets = pd.read_sql("""
        SELECT clob_token_id, question, end_date, volume
        FROM markets
    """, conn)
    
    conn.close()
    
    # Convert timestamps
    snapshots['datetime'] = pd.to_datetime(snapshots['ts'], unit='s', utc=True)
    
    print(f"  Snapshots loaded: {len(snapshots):,}")
    print(f"  Markets loaded:   {len(markets):,}")
    print(f"  Date range: {snapshots['datetime'].min()} to {snapshots['datetime'].max()}")
    
    return snapshots, markets


# =============================================================================
# FEATURE COMPUTATION — THE CORE OF CHUNK 3
# =============================================================================

def compute_features_for_market(df, market_end_date=None):
    """
    Compute all features for a single market's price series.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: ts, price, datetime
        Must be sorted by ts ascending
    market_end_date : str or None
        ISO date string for the market's expiry (e.g., "2026-03-01")
        Used to compute time-to-expiry feature
    
    Returns
    -------
    DataFrame with all original columns plus computed features.
    Rows that don't have enough history for lookback will have NaN
    in their feature columns — that's expected.
    
    IMPORTANT: This function ONLY looks backward in time for features.
    Labels are computed separately to keep the boundary crystal clear.
    """
    # Work on a copy so we don't modify the original
    df = df.copy()
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: Price returns (the building block for everything else)
    # ─────────────────────────────────────────────────────────────
    #
    # "returns" = price change from one sample to the next
    # This is the raw ingredient for velocity, acceleration, and volatility.
    #
    # Why returns instead of raw prices? Because a price of 0.50 doesn't
    # mean anything without context, but a return of +0.005 means "price
    # went up half a cent." Returns are stationary-ish; prices are not.
    
    df['return'] = df['price'].diff()
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Price level features
    # ─────────────────────────────────────────────────────────────
    
    df['p'] = df['price']
    df['p_complement'] = 1.0 - df['price']
    df['p_from_center'] = (df['price'] - 0.5).abs()
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: Velocity — price change over various windows
    # ─────────────────────────────────────────────────────────────
    #
    # dp_N = p(t) - p(t - N)
    #
    # This is NOT the same as summing N returns. If there are gaps in
    # the data, diff(N) handles it correctly because we're working with
    # sorted, indexed data. We're using .diff(N) which computes the
    # difference between the current row and the row N positions back.
    #
    # Positive dp_N means price went up over the last N samples.
    # The magnitude tells you how much.
    
    for w in VELOCITY_WINDOWS:
        df[f'dp_{w}'] = df['price'].diff(w)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: Acceleration — change in velocity
    # ─────────────────────────────────────────────────────────────
    #
    # ddp_N = dp_N(t) - dp_N(t - N)
    #
    # In plain English: "Is the momentum getting stronger or weaker?"
    #
    # Example with N=3 (30 min):
    #   At time t:   dp_3 = +0.005 (price went up 0.5 cents in last 30 min)
    #   At time t-3: dp_3 = +0.002 (price went up 0.2 cents in that 30 min)
    #   ddp_3 = 0.005 - 0.002 = +0.003 (momentum is accelerating upward)
    #
    # Positive ddp = accelerating (momentum building)
    # Negative ddp = decelerating (momentum fading, potential reversal)
    # Zero ddp = constant velocity (steady trend)
    
    for w in ACCELERATION_WINDOWS:
        velocity = df['price'].diff(w)
        df[f'ddp_{w}'] = velocity.diff(w)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Rolling volatility
    # ─────────────────────────────────────────────────────────────
    #
    # vol_N = std(returns) over the last N samples
    #
    # We use .rolling(N).std() on the RETURNS column, not the price
    # column. This is standard practice in finance:
    #
    # - std(prices) tells you "how spread out are the price levels"
    #   which depends on the price level itself (useless)
    # - std(returns) tells you "how much does the price typically
    #   change per period" (what we actually want)
    #
    # Higher volatility → more likely to cross the threshold
    # Lower volatility → market is quiet, less likely to move
    #
    # min_periods=w//2 means we'll compute volatility even with some
    # missing returns in the window, but require at least half the
    # window to be present. This prevents very noisy estimates from
    # tiny samples while not throwing away too many rows at the start.
    
    for w in VOLATILITY_WINDOWS:
        df[f'vol_{w}'] = df['return'].rolling(
            window=w, 
            min_periods=max(w // 2, 2)  # At least 2 points, or half the window
        ).std()
    
    # ─────────────────────────────────────────────────────────────
    # STEP 6: Relative position features
    # ─────────────────────────────────────────────────────────────
    #
    # Where is the current price relative to its recent range?
    #
    # p_vs_6h_high is like a "stochastic oscillator" in technical analysis:
    #   0.0 = price is at its 6-hour low
    #   1.0 = price is at its 6-hour high
    #   0.5 = price is in the middle of its range
    #
    # p_vs_6h_mean tells you if price is above or below its recent average.
    #   Positive = above average (maybe overbought)
    #   Negative = below average (maybe oversold)
    #
    # These features capture mean-reversion signals. Markets that have
    # drifted far from their mean might snap back.
    
    rolling_high = df['price'].rolling(window=36, min_periods=18).max()
    rolling_low = df['price'].rolling(window=36, min_periods=18).min()
    rolling_mean = df['price'].rolling(window=36, min_periods=18).mean()
    
    # Avoid division by zero when high == low (flat market)
    price_range = rolling_high - rolling_low
    price_range = price_range.replace(0, np.nan)
    
    df['p_vs_6h_high'] = (df['price'] - rolling_low) / price_range
    df['p_vs_6h_mean'] = df['price'] - rolling_mean
    
    # ─────────────────────────────────────────────────────────────
    # STEP 7: Time features
    # ─────────────────────────────────────────────────────────────
    #
    # Time of day, encoded cyclically with sin/cos.
    #
    # Why sin/cos instead of raw hour number?
    # 
    # If you use hour as a raw number (0, 1, 2, ... 23), the model
    # thinks hour 23 and hour 0 are far apart (distance = 23).
    # But they're actually adjacent! 11:50 PM is only 10 minutes
    # from 12:00 AM.
    #
    # Sin/cos encoding maps the 24-hour cycle onto a circle:
    #   hour_sin = sin(2π × hour/24)
    #   hour_cos = cos(2π × hour/24)
    #
    # Now hour 23 and hour 0 are close together in this 2D space,
    # which correctly represents their temporal proximity.
    #
    # The model gets TWO features (sin and cos) which together
    # uniquely identify any time of day. You need both — sin alone
    # can't distinguish 3 AM from 9 PM (both have the same sin value).
    
    hour_frac = df['datetime'].dt.hour + df['datetime'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_frac / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour_frac / 24.0)
    
    # Day of week: is it a weekend?
    # Polymarket is 24/7 but activity patterns differ on weekends
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 8: Time-to-expiry
    # ─────────────────────────────────────────────────────────────
    #
    # How many days until this market resolves?
    #
    # Markets approaching expiry converge toward 0 or 1. The rate of
    # convergence accelerates as expiry approaches (like a bond
    # approaching maturity — "pull to par").
    #
    # We use log(days + 1) because the effect of time-to-expiry is
    # nonlinear: the difference between 2 days and 1 day matters much
    # more than the difference between 102 days and 101 days.
    # Log compression captures this.
    #
    # The +1 prevents log(0) when expiry is today.
    
    if market_end_date and market_end_date != 'None' and pd.notna(market_end_date):
        try:
            end_dt = pd.to_datetime(market_end_date, utc=True)
            df['days_to_expiry'] = (end_dt - df['datetime']).dt.total_seconds() / 86400.0
            df['days_to_expiry'] = df['days_to_expiry'].clip(lower=0)  # No negative days
            df['log_days_to_expiry'] = np.log1p(df['days_to_expiry'])  # log(x+1)
        except Exception:
            df['days_to_expiry'] = np.nan
            df['log_days_to_expiry'] = np.nan
    else:
        df['days_to_expiry'] = np.nan
        df['log_days_to_expiry'] = np.nan
    
    return df


def compute_labels(df, horizon, threshold):
    """
    Compute the binary label: did price rise by >= threshold within horizon?
    
    This is intentionally a separate function from feature computation
    to make the boundary between "past information" and "future information"
    absolutely explicit.
    
    Parameters
    ----------
    df : DataFrame
        Must have 'price' column, sorted by time
    horizon : int
        Number of samples to look ahead (36 = 6 hours at 10-min intervals)
    threshold : float
        Minimum price increase to count as y=1 (0.001 = 0.1 cents)
    
    Returns
    -------
    DataFrame with additional columns:
        future_price: price at t + horizon
        price_change: future_price - current_price
        label: 1 if price_change >= threshold, else 0
    
    Rows at the end of the series (within `horizon` of the end) will have
    NaN labels because we can't see their future. These get dropped later.
    """
    df = df.copy()
    
    # shift(-horizon) takes the value `horizon` rows AHEAD
    # This is the price at time t + horizon
    df['future_price'] = df['price'].shift(-horizon)
    df['price_change'] = df['future_price'] - df['price']
    df['label'] = (df['price_change'] >= threshold).astype(float)
    
    # Mark rows where we can't compute the label as NaN
    df.loc[df['future_price'].isna(), 'label'] = np.nan
    
    return df


# =============================================================================
# FEATURE MATRIX ASSEMBLY
# =============================================================================

# The canonical list of feature columns.
# This is defined once so that training code and feature code always agree
# on what columns exist and in what order.
FEATURE_COLUMNS = [
    # Price level (3)
    'p', 'p_complement', 'p_from_center',
    # Velocity (4)
    'dp_3', 'dp_6', 'dp_12', 'dp_36',
    # Acceleration (2)
    'ddp_3', 'ddp_6',
    # Volatility (3)
    'vol_6', 'vol_12', 'vol_36',
    # Relative position (2)
    'p_vs_6h_high', 'p_vs_6h_mean',
    # Time (3)
    'hour_sin', 'hour_cos', 'is_weekend',
    # Time-to-expiry (2)
    'days_to_expiry', 'log_days_to_expiry',
]


def build_feature_matrix(snapshots, markets, horizon, threshold):
    """
    Main orchestrator: processes all markets and assembles the final table.
    
    For each market:
    1. Extract that market's price series
    2. Check it has enough data
    3. Compute features (looking backward)
    4. Compute labels (looking forward)
    5. Drop rows with NaN features or labels
    6. Append to the master table
    
    Parameters
    ----------
    snapshots : DataFrame
        All price observations
    markets : DataFrame
        Market metadata (needed for end_date)
    horizon : int
        Prediction horizon in samples
    threshold : float
        Binary label threshold
    
    Returns
    -------
    DataFrame with columns: clob_token_id, ts, [features], label
    """
    print(f"\nBuilding feature matrix...")
    print(f"  Horizon:   {horizon} samples ({horizon * 10 / 60:.1f} hours)")
    print(f"  Threshold: {threshold}")
    print(f"  Features:  {len(FEATURE_COLUMNS)}")
    
    # Create a lookup for market end dates
    end_dates = dict(zip(markets['clob_token_id'], markets['end_date']))
    
    all_rows = []
    market_ids = snapshots['clob_token_id'].unique()
    skipped_small = 0
    skipped_empty = 0
    
    for i, token_id in enumerate(market_ids):
        # Extract this market's data
        mkt = snapshots[snapshots['clob_token_id'] == token_id].copy()
        mkt = mkt.sort_values('ts').reset_index(drop=True)
        
        # Skip markets with too little data
        if len(mkt) < MIN_SAMPLES_PER_MARKET:
            skipped_small += 1
            continue
        
        # Compute features (backward-looking only)
        end_date = end_dates.get(token_id)
        mkt = compute_features_for_market(mkt, market_end_date=end_date)
        
        # Compute labels (forward-looking only)
        mkt = compute_labels(mkt, horizon=horizon, threshold=threshold)
        
        # Select only the columns we need for the output
        output_cols = ['clob_token_id', 'ts'] + FEATURE_COLUMNS + ['label']
        
        # Some columns might not exist if end_date was missing
        available_cols = [c for c in output_cols if c in mkt.columns]
        mkt_out = mkt[available_cols].copy()
        
        # Drop rows with NaN in features or label
        # This removes:
        #   - First ~36 rows (not enough lookback for all features)
        #   - Last 36 rows (not enough future for the label)
        #   - Any rows where data gaps caused NaN features
        before_drop = len(mkt_out)
        mkt_out = mkt_out.dropna(subset=['label'])
        
        # For features, we allow NaN in expiry columns (not all markets have dates)
        # but require all other features
        required_features = [c for c in FEATURE_COLUMNS 
                           if c not in ('days_to_expiry', 'log_days_to_expiry')]
        mkt_out = mkt_out.dropna(subset=required_features)
        
        after_drop = len(mkt_out)
        
        if after_drop == 0:
            skipped_empty += 1
            continue
        
        all_rows.append(mkt_out)
        
        # Progress update every 20 markets
        if (i + 1) % 20 == 0 or (i + 1) == len(market_ids):
            print(f"  Processed {i+1}/{len(market_ids)} markets "
                  f"({sum(len(r) for r in all_rows):,} rows so far)")
    
    # Combine all markets into one DataFrame
    if not all_rows:
        print("ERROR: No valid rows produced. Check your data.")
        sys.exit(1)
    
    result = pd.concat(all_rows, ignore_index=True)
    
    print(f"\n  Markets skipped (too few samples): {skipped_small}")
    print(f"  Markets skipped (all rows NaN):    {skipped_empty}")
    print(f"  Markets included:                  {len(all_rows)}")
    print(f"  Total rows:                        {len(result):,}")
    
    return result


# =============================================================================
# DIAGNOSTICS — SANITY CHECKS ON THE FEATURE MATRIX
# =============================================================================

def print_diagnostics(df):
    """
    Print summary statistics so you can catch problems before training.
    
    Things to look for:
    - Label balance: should be ~16% based on EDA
    - Feature ranges: nothing crazy (all features should be finite, bounded)
    - NaN counts: should be zero in required features, maybe some in expiry
    - Market coverage: how many markets contributed rows
    """
    print("\n" + "=" * 60)
    print("FEATURE MATRIX DIAGNOSTICS")
    print("=" * 60)
    
    # ── Label distribution ──
    label_counts = df['label'].value_counts()
    total = len(df)
    pos = label_counts.get(1.0, 0)
    neg = label_counts.get(0.0, 0)
    base_rate = pos / total if total > 0 else 0
    
    print(f"\nLabel distribution:")
    print(f"  y=0 (no move):    {neg:>8,} ({neg/total:.1%})")
    print(f"  y=1 (price rose): {pos:>8,} ({pos/total:.1%})")
    print(f"  Base rate:        {base_rate:.2%}")
    
    if base_rate < 0.05:
        print("  ⚠ WARNING: Base rate below 5%! Severe class imbalance.")
        print("    Consider lowering threshold or increasing horizon.")
    elif base_rate < 0.10:
        print("  ⚠ CAUTION: Base rate below 10%. Manageable but watch metrics.")
    elif base_rate > 0.50:
        print("  ⚠ CAUTION: Base rate above 50%. Your threshold might be too easy.")
    else:
        print("  ✓ Base rate looks healthy for binary classification.")
    
    # ── Feature statistics ──
    print(f"\nFeature statistics:")
    print(f"  {'Feature':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'NaN':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            s = df[col]
            nan_count = s.isna().sum()
            if nan_count < len(s):  # At least some non-NaN values
                print(f"  {col:<22} {s.mean():>10.5f} {s.std():>10.5f} "
                      f"{s.min():>10.5f} {s.max():>10.5f} {nan_count:>8,}")
            else:
                print(f"  {col:<22} {'all NaN':>42} {nan_count:>8,}")
        else:
            print(f"  {col:<22} {'MISSING':>42}")
    
    # ── Market coverage ──
    market_counts = df['clob_token_id'].value_counts()
    print(f"\nMarket coverage:")
    print(f"  Markets with rows:  {len(market_counts)}")
    print(f"  Rows per market:")
    print(f"    Min:    {market_counts.min():,}")
    print(f"    Median: {int(market_counts.median()):,}")
    print(f"    Max:    {market_counts.max():,}")
    print(f"    Total:  {market_counts.sum():,}")
    
    # ── Time range ──
    min_ts = df['ts'].min()
    max_ts = df['ts'].max()
    min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc)
    max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)
    days = (max_dt - min_dt).days
    print(f"\nTime range:")
    print(f"  {min_dt.strftime('%Y-%m-%d %H:%M')} to {max_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  ({days} days)")
    
    # ── Feature correlations with label ──
    print(f"\nFeature correlation with label (Pearson):")
    print(f"  Top positive correlations = feature tends to predict y=1")
    print(f"  Top negative correlations = feature tends to predict y=0")
    
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    corrs = df[available_features + ['label']].corr()['label'].drop('label')
    corrs = corrs.dropna().sort_values(ascending=False)
    
    print(f"\n  {'Feature':<22} {'Correlation':>12}")
    print(f"  {'-'*22} {'-'*12}")
    # Show top 5 positive and top 5 negative
    for col in list(corrs.head(5).index) + list(corrs.tail(5).index):
        print(f"  {col:<22} {corrs[col]:>12.4f}")
    
    print(f"\n  Note: Low correlations are NORMAL for market prediction.")
    print(f"  Anything above |0.05| is worth paying attention to.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature engineering for Polymarket predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/features.py                              # Default params
    python src/features.py --threshold 0.002 --horizon 36  # Custom threshold
    python src/features.py --output data/features_v2.csv   # Custom output path
        """
    )
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Price change threshold for y=1 (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON,
                        help=f'Prediction horizon in samples (default: {DEFAULT_HORIZON})')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help=f'Output CSV path (default: {DEFAULT_OUTPUT})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHUNK 3: FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Threshold: {args.threshold} ({args.threshold*100:.1f} cents)")
    print(f"  Horizon:   {args.horizon} samples ({args.horizon * 10 / 60:.1f} hours)")
    print(f"  Output:    {args.output}")
    
    start_time = time.time()
    
    # Load raw data
    snapshots, markets = load_data()
    
    # Build feature matrix
    df = build_feature_matrix(snapshots, markets, args.horizon, args.threshold)
    
    # Print diagnostics
    print_diagnostics(df)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  Output:    {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows:      {len(df):,}")
    print(f"  Columns:   {len(df.columns)}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"\nNext step: Chunk 4 — Labels + baseline model")
    print(f"  Run: python src/train_eval.py")


if __name__ == "__main__":
    main()
