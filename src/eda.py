#!/usr/bin/env python3
"""
eda.py - Exploratory Data Analysis for Polymarket data

Run this AFTER ingest.py backfill to understand your data before building features.

Usage:
    cd /mnt/ml-data/projects/polymarket-predictor
    python src/eda.py

What this script answers:
1. What do price movements look like?
2. How often does our target event (≥2 cent move in 2hr) occur?
3. Are there gaps, duplicates, weird values?
4. What's the typical volatility?
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# CONFIG
# =============================================================================

DB_PATH = Path(__file__).parent.parent / "data" / "polymarket.db"
HORIZON_SAMPLES = 36    # 6 hours at 10-min intervals
THRESHOLD = 0.001       # 0.1 cent move

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load markets and snapshots from database."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run 'python src/ingest.py backfill' first.")
        return None, None
    
    conn = sqlite3.connect(DB_PATH)
    
    markets = pd.read_sql("SELECT * FROM markets", conn)
    print(f"Markets: {len(markets)}")
    
    snapshots = pd.read_sql("SELECT * FROM snapshots", conn)
    print(f"Snapshots: {len(snapshots):,}")
    
    # Convert timestamp to datetime
    snapshots['dt'] = pd.to_datetime(snapshots['ts'], unit='s', utc=True)
    snapshots = snapshots.sort_values(['clob_token_id', 'ts']).reset_index(drop=True)
    
    print(f"Date range: {snapshots['dt'].min()} to {snapshots['dt'].max()}")
    
    conn.close()
    return markets, snapshots


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def plot_sample_markets(markets, snapshots):
    """Plot price timeseries for top markets."""
    print("\n" + "=" * 60)
    print("1. SAMPLE PRICE TIMESERIES")
    print("=" * 60)
    
    # Get top 6 markets by snapshot count
    top_markets = (
        snapshots.groupby('clob_token_id')
        .size()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, token_id in enumerate(top_markets):
        data = snapshots[snapshots['clob_token_id'] == token_id]
        
        name = markets[markets['clob_token_id'] == token_id]['question'].values
        name = name[0][:45] + '...' if len(name) > 0 else 'Unknown'
        
        axes[i].plot(data['dt'], data['price'], linewidth=0.8)
        axes[i].set_title(name, fontsize=9)
        axes[i].set_ylabel('P(Yes)')
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.suptitle('Price Timeseries for Top 6 Markets', y=1.02, fontsize=14)
    plt.savefig(DB_PATH.parent.parent / 'results' / 'eda_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved: results/eda_timeseries.png")
    return top_markets


def compute_base_rate(markets, snapshots):
    """Compute how often y=1 occurs (price rises ≥2 cents in 2 hours)."""
    print("\n" + "=" * 60)
    print("2. BASE RATE ANALYSIS")
    print("=" * 60)
    print(f"Target: Price increases by ≥{THRESHOLD} in {HORIZON_SAMPLES * 10} minutes")
    print()
    
    all_labels = []
    
    for token_id in snapshots['clob_token_id'].unique():
        df = snapshots[snapshots['clob_token_id'] == token_id].sort_values('ts').copy()
        
        # Future price (24 samples = 2 hours ahead)
        df['future_price'] = df['price'].shift(-HORIZON_SAMPLES)
        df['price_change'] = df['future_price'] - df['price']
        df['label'] = (df['price_change'] >= THRESHOLD).astype(float)
        df.loc[df['future_price'].isna(), 'label'] = np.nan
        
        all_labels.append(df[['clob_token_id', 'ts', 'price', 'price_change', 'label']])
    
    labeled_df = pd.concat(all_labels, ignore_index=True)
    valid = labeled_df[labeled_df['label'].notna()]
    
    base_rate = valid['label'].mean()
    
    print(f"Total samples with valid labels: {len(valid):,}")
    print(f"Samples where y=1: {int(valid['label'].sum()):,}")
    print(f"Samples where y=0: {int((1 - valid['label']).sum()):,}")
    print()
    print(f">>> BASE RATE: {base_rate:.2%} <<<")
    print()
    
    if base_rate < 0.10:
        print("⚠ Base rate is LOW (<10%). Consider:")
        print("  - Lowering threshold (e.g., 0.015 or 0.01)")
        print("  - Increasing horizon (e.g., 4 hours)")
    elif base_rate > 0.40:
        print("⚠ Base rate is HIGH (>40%). Consider:")
        print("  - Raising threshold (e.g., 0.03 or 0.04)")
        print("  - Decreasing horizon (e.g., 1 hour)")
    else:
        print("✓ Base rate is in good range (10-40%)")
    
    # Base rate by market
    by_market = valid.groupby('clob_token_id')['label'].agg(['mean', 'count'])
    by_market.columns = ['base_rate', 'n_samples']
    
    print(f"\nBase rate variation across markets:")
    print(f"  Min:    {by_market['base_rate'].min():.2%}")
    print(f"  Max:    {by_market['base_rate'].max():.2%}")
    print(f"  Std:    {by_market['base_rate'].std():.2%}")
    
    return labeled_df, base_rate


def analyze_price_changes(labeled_df):
    """Analyze distribution of 2-hour price changes."""
    print("\n" + "=" * 60)
    print("3. PRICE CHANGE DISTRIBUTION")
    print("=" * 60)
    
    changes = labeled_df[labeled_df['price_change'].notna()]['price_change']
    
    print(f"2-hour price change statistics:")
    print(f"  Mean:   {changes.mean():+.4f}")
    print(f"  Median: {changes.median():+.4f}")
    print(f"  Std:    {changes.std():.4f}")
    print(f"  Min:    {changes.min():+.4f}")
    print(f"  Max:    {changes.max():+.4f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(changes, p)
        print(f"  {p}th: {val:+.4f}")
    
    # Where does threshold fall?
    pct_above = (changes >= THRESHOLD).mean() * 100
    print(f"\n{THRESHOLD} threshold = {100 - pct_above:.1f}th percentile")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(changes, bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
    axes[0].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('2-Hour Price Change')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of 2-Hour Price Changes')
    axes[0].legend()
    
    axes[1].hist(changes, bins=100, range=(-0.1, 0.1), edgecolor='black', alpha=0.7)
    axes[1].axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
    axes[1].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('2-Hour Price Change')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Zoomed: -10% to +10%')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(DB_PATH.parent.parent / 'results' / 'eda_price_changes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nSaved: results/eda_price_changes.png")
    
    return changes


def analyze_gaps(snapshots):
    """Check for gaps in the timeseries."""
    print("\n" + "=" * 60)
    print("4. DATA QUALITY: GAPS")
    print("=" * 60)
    
    all_gaps = []
    for token_id in snapshots['clob_token_id'].unique():
        df = snapshots[snapshots['clob_token_id'] == token_id].sort_values('ts')
        gaps = df['ts'].diff().dropna()
        all_gaps.append(gaps)
    
    all_gaps = pd.concat(all_gaps)
    gaps_min = all_gaps / 60
    
    print(f"Gap statistics (minutes):")
    print(f"  Expected: 5.0")
    print(f"  Mean:     {gaps_min.mean():.1f}")
    print(f"  Median:   {gaps_min.median():.1f}")
    print(f"  Max:      {gaps_min.max():.1f} ({gaps_min.max()/60:.1f} hours)")
    
    pct_exact = (all_gaps == 300).mean() * 100
    pct_close = ((all_gaps >= 290) & (all_gaps <= 310)).mean() * 100
    
    print(f"\nGaps exactly 5 min: {pct_exact:.1f}%")
    print(f"Gaps within ±10s of 5 min: {pct_close:.1f}%")
    
    # Count large gaps
    large_gaps = (gaps_min > 60).sum()
    print(f"Gaps > 1 hour: {large_gaps:,}")
    
    if pct_close > 90:
        print("\n✓ Data is well-sampled (>90% gaps are ~5 min)")
    else:
        print(f"\n⚠ Only {pct_close:.1f}% of gaps are ~5 min. Check for missing data.")


def analyze_volatility(markets, snapshots):
    """Compute volatility by market."""
    print("\n" + "=" * 60)
    print("5. VOLATILITY ANALYSIS")
    print("=" * 60)
    
    volatilities = []
    
    for token_id in snapshots['clob_token_id'].unique():
        df = snapshots[snapshots['clob_token_id'] == token_id].sort_values('ts').copy()
        df['price_diff'] = df['price'].diff()
        
        # 1-hour rolling std (12 samples)
        vol = df['price_diff'].rolling(12).std().mean()
        
        volatilities.append({
            'clob_token_id': token_id,
            'volatility': vol
        })
    
    vol_df = pd.DataFrame(volatilities)
    vol_df = vol_df.merge(markets[['clob_token_id', 'question', 'volume']], on='clob_token_id')
    
    print(f"Volatility statistics (1-hr rolling std):")
    print(f"  Mean:   {vol_df['volatility'].mean():.5f}")
    print(f"  Median: {vol_df['volatility'].median():.5f}")
    print(f"  Std:    {vol_df['volatility'].std():.5f}")
    
    print(f"\nMost volatile markets:")
    for _, row in vol_df.nlargest(3, 'volatility').iterrows():
        print(f"  {row['volatility']:.5f} - {row['question'][:50]}...")
    
    print(f"\nLeast volatile markets:")
    for _, row in vol_df.nsmallest(3, 'volatility').iterrows():
        print(f"  {row['volatility']:.5f} - {row['question'][:50]}...")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(vol_df['volatility'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Average 1-Hour Volatility')
    plt.ylabel('Number of Markets')
    plt.title('Distribution of Market Volatilities')
    plt.savefig(DB_PATH.parent.parent / 'results' / 'eda_volatility.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nSaved: results/eda_volatility.png")


def analyze_autocorrelation(snapshots, top_markets):
    """Check for momentum/mean-reversion in price changes."""
    print("\n" + "=" * 60)
    print("6. AUTOCORRELATION (MOMENTUM CHECK)")
    print("=" * 60)
    
    sample_token = top_markets[0]
    df = snapshots[snapshots['clob_token_id'] == sample_token].sort_values('ts').copy()
    df['price_change'] = df['price'].diff()
    
    print(f"Analyzing: {sample_token[:30]}...")
    print()
    
    # Compute autocorrelations
    lags = [1, 3, 6, 12, 24]  # 5min, 15min, 30min, 1hr, 2hr
    print("Autocorrelation of price changes:")
    for lag in lags:
        ac = df['price_change'].autocorr(lag=lag)
        label = f"{lag * 5} min"
        print(f"  Lag {lag:2d} ({label:>6}): {ac:+.4f}")
    
    print()
    print("Interpretation:")
    print("  Positive = momentum (trends continue)")
    print("  Negative = mean reversion (trends reverse)")
    print("  Near zero = random walk (no predictable pattern)")
    
    ac1 = df['price_change'].autocorr(lag=1)
    if abs(ac1) > 0.05:
        print(f"\n✓ Lag-1 autocorr = {ac1:+.4f} — there may be signal to exploit")
    else:
        print(f"\n⚠ Lag-1 autocorr = {ac1:+.4f} — price changes look random")


def print_summary(markets, snapshots, base_rate):
    """Print final summary table."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    days = (snapshots['dt'].max() - snapshots['dt'].min()).days
    
    print(f"  Markets:              {len(markets)}")
    print(f"  Snapshots:            {len(snapshots):,}")
    print(f"  Days of data:         {days}")
    print(f"  Avg per market:       {len(snapshots) // len(markets):,}")
    print(f"  Base rate (y=1):      {base_rate:.2%}")
    print(f"  Threshold:            {THRESHOLD}")
    print(f"  Horizon:              {HORIZON_SAMPLES * 10} minutes")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    if 0.10 <= base_rate <= 0.40:
        print("✓ Base rate looks good. Proceed to Chunk 3 (feature engineering).")
    else:
        print("⚠ Consider adjusting threshold/horizon before Chunk 3.")
        print("  Edit THRESHOLD and HORIZON_SAMPLES in this script and re-run.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create results directory
    results_dir = DB_PATH.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    markets, snapshots = load_data()
    if markets is None:
        return
    
    # Run analyses
    top_markets = plot_sample_markets(markets, snapshots)
    labeled_df, base_rate = compute_base_rate(markets, snapshots)
    analyze_price_changes(labeled_df)
    analyze_gaps(snapshots)
    analyze_volatility(markets, snapshots)
    analyze_autocorrelation(snapshots, top_markets)
    
    # Summary
    print_summary(markets, snapshots, base_rate)


if __name__ == "__main__":
    main()
