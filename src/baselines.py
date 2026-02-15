#!/usr/bin/env python3
"""
baselines.py — Chunk 4: Labels + Baseline Models

Establishes the floor that any real model must beat.
If your fancy ML model can't outperform these trivially simple
predictors on out-of-sample data, something is wrong with your
pipeline — not your model complexity.

=== WHAT THIS SCRIPT DOES ===

1. Loads the feature matrix from Chunk 3
2. Splits it by TIME (not randomly!) into train/val/test
3. Evaluates several braindead baseline predictors
4. Prints a comparison table with log loss, Brier score, calibration

=== THE BASELINES ===

1. "Always 0.5" — predict 50% probability for everything
   This is the absolute worst reasonable predictor. It knows nothing
   about the data. If your model can't beat this, you have no signal.

2. "Predict base rate" — predict the training set's base rate for everything
   This is smarter than 0.5: it knows that y=1 happens ~27% of the time,
   so it predicts 0.27 for every row. This is the minimum bar for log loss.
   Any model that can't beat this hasn't learned anything beyond class frequency.

3. "No change" — predict 0 for everything (price won't move)
   Since y=0 is the majority class (~73%), this gets high accuracy.
   But accuracy is misleading for imbalanced problems. We care about
   log loss and calibration instead.

4. "Momentum" — predict based on recent price direction
   If price went up recently (dp_6 > 0), predict higher probability of
   continued upward movement. This is the simplest "intelligent" baseline.
   It uses actual features but in the most naive way possible.

=== WHY TIME SPLITS, NOT RANDOM SPLITS ===

Random train/test splits are WRONG for time series data.

If you randomly shuffle rows, your training set will contain data from
February 3rd and your test set will contain data from February 2nd.
The model is literally trained on future data and tested on past data.

Walk-forward split:
    |======= TRAIN (70%) =======|=== VAL (15%) ===|=== TEST (15%) ===|
    Jan 5 ──────────────────── Jan 27 ──────── Jan 31 ──────── Feb 5

The model only ever sees the past during training and is evaluated
on the future. This mimics real-world deployment where you train on
historical data and predict forward.

=== METRICS EXPLAINED ===

Log Loss (lower is better):
    -mean(y * log(p) + (1-y) * log(1-p))
    Heavily penalizes confident wrong predictions. If you predict 0.99
    and the answer is 0, you get hammered. This is the primary metric.

Brier Score (lower is better):
    mean((p - y)²)
    Mean squared error of your probabilities. Less punishing of
    confident errors than log loss. Easier to interpret: perfect = 0,
    random guessing at base rate = base_rate × (1 - base_rate).

Calibration:
    When you predict "30% chance of moving up," does it actually move
    up 30% of the time? We bin predictions into buckets (0.0-0.1,
    0.1-0.2, etc.) and check if the actual frequency matches.
    Perfect calibration = predicted probability equals actual frequency.

=== USAGE ===

    cd /mnt/ml-data/projects/polymarket-predictor
    python src/baselines.py

    # Custom feature file:
    python src/baselines.py --features data/features_v2.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Train/val/test split ratios (by time, NOT by random shuffle)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit: whatever's left)


# =============================================================================
# METRICS
# =============================================================================

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Binary cross-entropy / log loss.
    
    Parameters
    ----------
    y_true : array-like
        Actual labels (0 or 1)
    y_pred : array-like
        Predicted probabilities (0 to 1)
    eps : float
        Small constant to avoid log(0). Clips predictions to [eps, 1-eps].
    
    Returns
    -------
    float : mean log loss (lower is better)
    
    Why we implement this ourselves instead of using sklearn:
    - Fewer dependencies
    - You understand exactly what's being computed
    - sklearn.metrics.log_loss does the same thing
    
    The formula:
        L = -1/N × Σ [y_i × log(p_i) + (1 - y_i) × log(1 - p_i)]
    
    Intuition: If y=1 and you predicted p=0.9, loss is -log(0.9) = 0.105 (small).
    If y=1 and you predicted p=0.1, loss is -log(0.1) = 2.303 (huge!).
    The penalty is asymmetric — being confidently wrong is much worse
    than being uncertain.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def brier_score(y_true, y_pred):
    """
    Brier score — mean squared error of probability predictions.
    
    Returns
    -------
    float : mean Brier score (lower is better, 0 = perfect)
    
    Reference values:
    - Always predict 0.5:     0.25
    - Predict base rate 0.27: 0.27 × 0.73 ≈ 0.197
    - Perfect predictions:    0.0
    
    Brier score decomposes into three components:
    1. Reliability (calibration error)
    2. Resolution (how much predictions vary)
    3. Uncertainty (inherent unpredictability)
    
    We care most about #1 and #2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def calibration_table(y_true, y_pred, n_bins=10):
    """
    Compute a calibration table: binned predictions vs actual frequencies.
    
    For each bin (e.g., predictions between 0.2 and 0.3), compute:
    - The average prediction in that bin
    - The actual fraction of y=1 in that bin
    - How many samples fell in that bin
    
    Perfect calibration: avg_predicted ≈ actual_frequency for every bin.
    
    Parameters
    ----------
    y_true : array-like
        Actual labels (0 or 1)
    y_pred : array-like
        Predicted probabilities
    n_bins : int
        Number of bins (default: 10 for deciles)
    
    Returns
    -------
    DataFrame with columns: bin_lower, bin_upper, avg_predicted,
                           actual_frequency, count
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        
        # Which predictions fall in this bin?
        if i < n_bins - 1:
            mask = (y_pred >= lo) & (y_pred < hi)
        else:
            mask = (y_pred >= lo) & (y_pred <= hi)  # Include 1.0 in last bin
        
        count = mask.sum()
        
        if count > 0:
            avg_pred = y_pred[mask].mean()
            actual_freq = y_true[mask].mean()
        else:
            avg_pred = np.nan
            actual_freq = np.nan
        
        rows.append({
            'bin': f'{lo:.1f}-{hi:.1f}',
            'avg_predicted': avg_pred,
            'actual_frequency': actual_freq,
            'count': count
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# DATA SPLITTING
# =============================================================================

def time_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Split data by time into train/val/test sets.
    
    CRITICAL: We sort by timestamp and split sequentially.
    NO SHUFFLING. The training set is always the oldest data,
    validation is the middle, test is the most recent.
    
    Why not random split?
    - Time series data has autocorrelation (nearby points are similar)
    - Random split leaks future information into training
    - Real deployment always predicts the future from the past
    - Walk-forward validation mimics reality
    
    Parameters
    ----------
    df : DataFrame
        Must have 'ts' column (unix timestamp)
    train_ratio : float
        Fraction of data for training (default: 0.70)
    val_ratio : float
        Fraction of data for validation (default: 0.15)
    
    Returns
    -------
    train_df, val_df, test_df : DataFrames
    """
    # Sort by time (should already be sorted, but be safe)
    df = df.sort_values('ts').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Print split info
    def ts_to_date(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
    
    print(f"\nTime-based split:")
    print(f"  Train: {len(train_df):>8,} rows  "
          f"({ts_to_date(train_df['ts'].min())} to {ts_to_date(train_df['ts'].max())})")
    print(f"  Val:   {len(val_df):>8,} rows  "
          f"({ts_to_date(val_df['ts'].min())} to {ts_to_date(val_df['ts'].max())})")
    print(f"  Test:  {len(test_df):>8,} rows  "
          f"({ts_to_date(test_df['ts'].min())} to {ts_to_date(test_df['ts'].max())})")
    
    # Sanity check: no time overlap
    assert train_df['ts'].max() <= val_df['ts'].min(), "Train/val overlap!"
    assert val_df['ts'].max() <= test_df['ts'].min(), "Val/test overlap!"
    
    # Check base rates across splits (they should be roughly similar)
    train_rate = train_df['label'].mean()
    val_rate = val_df['label'].mean()
    test_rate = test_df['label'].mean()
    
    print(f"\n  Base rates:  train={train_rate:.1%}  val={val_rate:.1%}  test={test_rate:.1%}")
    
    if abs(train_rate - test_rate) > 0.10:
        print("  ⚠ WARNING: Base rate shifted significantly between train and test.")
        print("    This means market behavior changed over time (concept drift).")
        print("    Your model may struggle. This is normal and important to know.")
    
    return train_df, val_df, test_df


# =============================================================================
# BASELINE PREDICTORS
# =============================================================================

class AlwaysPredict:
    """
    Predicts the same constant probability for every sample.
    
    This is the simplest possible "model." If you predict the base rate,
    you get the best possible log loss for a constant predictor.
    
    Any real model that can't beat this hasn't learned anything.
    """
    def __init__(self, value):
        self.value = value
        self.name = f"Always {value:.3f}"
    
    def predict(self, X):
        return np.full(len(X), self.value)


class MomentumBaseline:
    """
    Predicts based on recent price direction.
    
    Logic:
    - If dp_6 > 0 (price went up in last hour): predict higher than base rate
    - If dp_6 < 0 (price went down): predict lower than base rate
    - If dp_6 = 0 (no change): predict base rate
    
    The prediction is: base_rate + scale * dp_6
    
    This is clipped to [0.01, 0.99] to avoid degenerate probabilities.
    
    The 'scale' parameter controls how much the momentum shifts the
    prediction. It's calibrated on the training set to minimize log loss.
    
    This is the simplest "intelligent" baseline — it actually uses
    a feature. If your ML model can't beat this, the additional features
    and complexity aren't adding value.
    """
    def __init__(self):
        self.base_rate = None
        self.scale = None
        self.name = "Momentum (dp_6)"
    
    def fit(self, train_df):
        """
        Calibrate the momentum baseline on training data.
        
        We try different scale values and pick the one that minimizes
        log loss on the training set. This is a bit like hyperparameter
        tuning, but the "model" is so simple that overfitting isn't a concern.
        """
        self.base_rate = train_df['label'].mean()
        
        y_true = train_df['label'].values
        dp6 = train_df['dp_6'].values
        
        best_loss = float('inf')
        best_scale = 0
        
        # Try different scales
        for scale in [0, 0.5, 1, 2, 5, 10, 20, 50]:
            preds = np.clip(self.base_rate + scale * dp6, 0.01, 0.99)
            loss = log_loss(y_true, preds)
            if loss < best_loss:
                best_loss = loss
                best_scale = scale
        
        self.scale = best_scale
        print(f"  Momentum baseline: base_rate={self.base_rate:.3f}, "
              f"best_scale={self.scale}")
    
    def predict(self, df):
        dp6 = df['dp_6'].values
        preds = np.clip(self.base_rate + self.scale * dp6, 0.01, 0.99)
        return preds


class VolatilityBaseline:
    """
    Predicts based on recent volatility.
    
    Logic: Higher volatility → higher probability of threshold crossing.
    
    This captures the intuition that active markets are more likely to
    move than quiet ones. Since volatility had the second-highest
    correlation with the label in our diagnostics, this might actually
    be hard to beat.
    
    Prediction: base_rate + scale * (vol_12 - mean_vol_12) / std_vol_12
    
    We z-score the volatility so the scale parameter is interpretable
    regardless of the raw volatility magnitude.
    """
    def __init__(self):
        self.base_rate = None
        self.scale = None
        self.vol_mean = None
        self.vol_std = None
        self.name = "Volatility (vol_12)"
    
    def fit(self, train_df):
        self.base_rate = train_df['label'].mean()
        self.vol_mean = train_df['vol_12'].mean()
        self.vol_std = train_df['vol_12'].std()
        
        if self.vol_std < 1e-10:
            self.scale = 0
            return
        
        y_true = train_df['label'].values
        vol_z = (train_df['vol_12'].values - self.vol_mean) / self.vol_std
        
        best_loss = float('inf')
        best_scale = 0
        
        for scale in [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
            preds = np.clip(self.base_rate + scale * vol_z, 0.01, 0.99)
            loss = log_loss(y_true, preds)
            if loss < best_loss:
                best_loss = loss
                best_scale = scale
        
        self.scale = best_scale
        print(f"  Volatility baseline: base_rate={self.base_rate:.3f}, "
              f"best_scale={self.scale}, vol_mean={self.vol_mean:.5f}")
    
    def predict(self, df):
        vol_z = (df['vol_12'].values - self.vol_mean) / self.vol_std
        preds = np.clip(self.base_rate + self.scale * vol_z, 0.01, 0.99)
        return preds


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_baseline(name, y_true, y_pred):
    """
    Compute all metrics for one baseline predictor.
    
    Returns a dict with the results.
    """
    ll = log_loss(y_true, y_pred)
    bs = brier_score(y_true, y_pred)
    cal = calibration_table(y_true, y_pred)
    
    return {
        'name': name,
        'log_loss': ll,
        'brier': bs,
        'calibration': cal,
        'predictions': y_pred
    }


def print_results(results, y_true, split_name="Test"):
    """
    Print a formatted comparison table of all baselines.
    """
    base_rate = y_true.mean()
    
    print(f"\n{'=' * 70}")
    print(f"BASELINE RESULTS ON {split_name.upper()} SET")
    print(f"{'=' * 70}")
    print(f"Base rate: {base_rate:.1%}  |  N = {len(y_true):,}")
    print(f"\n  {'Model':<28} {'Log Loss':>10} {'Brier':>10} {'vs Base Rate':>14}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*14}")
    
    # "Predict base rate" log loss is the reference
    ref_ll = log_loss(y_true, np.full(len(y_true), base_rate))
    
    for r in results:
        # Compare to base rate predictor
        diff = r['log_loss'] - ref_ll
        direction = "worse" if diff > 0 else "better"
        diff_str = f"{abs(diff):.4f} {direction}"
        
        print(f"  {r['name']:<28} {r['log_loss']:>10.4f} {r['brier']:>10.4f} {diff_str:>14}")
    
    # Print calibration for the best model
    best = min(results, key=lambda x: x['log_loss'])
    print(f"\nCalibration table for best baseline: {best['name']}")
    print(f"  {'Bin':<10} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    for _, row in best['calibration'].iterrows():
        if row['count'] > 0:
            pred_str = f"{row['avg_predicted']:.3f}" if pd.notna(row['avg_predicted']) else "n/a"
            act_str = f"{row['actual_frequency']:.3f}" if pd.notna(row['actual_frequency']) else "n/a"
            print(f"  {row['bin']:<10} {pred_str:>10} {act_str:>10} {row['count']:>8,}")
    
    print(f"\n  Reading the calibration table:")
    print(f"  - 'Predicted' = average model output for samples in this bin")
    print(f"  - 'Actual' = fraction of samples that were actually y=1")
    print(f"  - If Predicted ≈ Actual, the model is well-calibrated in that bin")
    print(f"  - Most baselines only predict one value, so only one bin has data")


def print_what_to_beat(results):
    """
    Summarize the targets for Chunk 5.
    """
    best = min(results, key=lambda x: x['log_loss'])
    
    print(f"\n{'=' * 70}")
    print(f"TARGETS FOR CHUNK 5 (Your model must beat these)")
    print(f"{'=' * 70}")
    print(f"  Best baseline: {best['name']}")
    print(f"  Log loss to beat:  {best['log_loss']:.4f}")
    print(f"  Brier to beat:     {best['brier']:.4f}")
    print(f"\n  If your logistic regression can't beat {best['log_loss']:.4f} on the")
    print(f"  test set, STOP and check for bugs before adding complexity.")
    print(f"  Common causes: feature leakage, wrong time split, bad features.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chunk 4: Baseline model evaluation"
    )
    parser.add_argument('--features', type=str, default=str(FEATURES_PATH),
                        help=f'Path to features CSV (default: {FEATURES_PATH})')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CHUNK 4: BASELINE MODELS")
    print("=" * 70)
    
    # ── Load data ──
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"ERROR: Features file not found at {features_path}")
        print("Run 'python src/features.py' first (Chunk 3).")
        sys.exit(1)
    
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # ── Time split ──
    train_df, val_df, test_df = time_split(df)
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # ── Build baselines ──
    print(f"\nBuilding baselines...")
    
    train_base_rate = y_train.mean()
    
    baselines = []
    
    # Baseline 1: Always predict 0.5
    b1 = AlwaysPredict(0.5)
    baselines.append(b1)
    
    # Baseline 2: Predict training base rate
    b2 = AlwaysPredict(train_base_rate)
    b2.name = f"Base rate ({train_base_rate:.3f})"
    baselines.append(b2)
    
    # Baseline 3: Momentum
    b3 = MomentumBaseline()
    b3.fit(train_df)
    baselines.append(b3)
    
    # Baseline 4: Volatility
    b4 = VolatilityBaseline()
    b4.fit(train_df)
    baselines.append(b4)
    
    # ── Evaluate on validation set ──
    val_results = []
    for b in baselines:
        preds = b.predict(val_df)
        result = evaluate_baseline(b.name, y_val, preds)
        val_results.append(result)
    
    print_results(val_results, y_val, split_name="Validation")
    
    # ── Evaluate on test set ──
    test_results = []
    for b in baselines:
        preds = b.predict(test_df)
        result = evaluate_baseline(b.name, y_test, preds)
        test_results.append(result)
    
    print_results(test_results, y_test, split_name="Test")
    
    # ── Summary ──
    print_what_to_beat(test_results)
    
    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save a summary CSV for later comparison
    summary_rows = []
    for r in test_results:
        summary_rows.append({
            'model': r['name'],
            'split': 'test',
            'log_loss': r['log_loss'],
            'brier': r['brier'],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / 'baseline_results.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Results saved to {summary_path}")
    
    print(f"\nNext step: Chunk 5 — Logistic regression")
    print(f"  Run: python src/train_eval.py")


if __name__ == "__main__":
    main()
