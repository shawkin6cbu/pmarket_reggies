#!/usr/bin/env python3
"""
train_eval.py — Chunk 5: First Real Model (Logistic Regression)

Trains a logistic regression with regularization using walk-forward
evaluation. This is the first model that actually learns from all
features simultaneously.

=== WHY LOGISTIC REGRESSION FIRST ===

Logistic regression is the right first model because:

1. It's a linear model — easy to understand what it's doing
2. It produces calibrated probabilities (unlike many other models)
3. It's fast to train — you can iterate quickly
4. It sets a meaningful baseline for nonlinear models (Chunk 6)
5. If it can't beat the baselines, something is wrong with your
   features or pipeline, and adding complexity won't help

=== WHAT THIS SCRIPT DOES ===

1. Loads features from Chunk 3
2. Splits by time into train/val/test (same split as Chunk 4)
3. Standardizes features (mean=0, std=1) using ONLY training stats
4. Trains logistic regression with L2 regularization
5. Tunes regularization strength on validation set
6. Evaluates on test set and compares to baselines
7. Prints calibration table and feature importances

=== STANDARDIZATION ===

Logistic regression needs standardized features because:
- Features on different scales get different effective regularization
- dp_3 ranges from -0.5 to +0.5, while days_to_expiry goes 0 to 200
- Without standardization, the model would ignore dp_3 and overweight
  days_to_expiry simply because it has larger numbers

We compute mean and std from the TRAINING set only, then apply
those same statistics to val and test. This prevents leakage:
if we computed stats on the whole dataset, the training features
would be influenced by future data distributions.

=== REGULARIZATION ===

L2 regularization (Ridge) adds a penalty term: λ × Σ(weights²)

This prevents the model from putting all its faith in one feature
and makes predictions more robust. Too little regularization →
overfitting (great on train, bad on test). Too much → underfitting
(model barely uses the features).

We try several values of C (= 1/λ) and pick the one with the
best validation log loss. C is sklearn's regularization parameter:
larger C = less regularization, smaller C = more regularization.

=== USAGE ===

    cd /mnt/ml-data/projects/polymarket-predictor
    python src/train_eval.py

    # Custom parameters:
    python src/train_eval.py --features data/features.csv
"""

import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# We use sklearn for logistic regression. It's the standard choice
# and handles the optimization correctly. No need to implement
# gradient descent from scratch for a first project.
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required.")
    print("Install it: pip install scikit-learn --break-system-packages")
    print("  or: conda install scikit-learn")
    sys.exit(1)

warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Same split ratios as baselines.py — MUST match so we compare apples to apples
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Feature columns — same list as features.py
# This is the single source of truth for what features the model sees.
FEATURE_COLUMNS = [
    'p', 'p_complement', 'p_from_center',
    'dp_3', 'dp_6', 'dp_12', 'dp_36',
    'ddp_3', 'ddp_6',
    'vol_6', 'vol_12', 'vol_36',
    'p_vs_6h_high', 'p_vs_6h_mean',
    'hour_sin', 'hour_cos', 'is_weekend',
    'days_to_expiry', 'log_days_to_expiry',
]

# Regularization values to try
# C = 1/λ, so smaller C = stronger regularization
# We search across several orders of magnitude
C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# =============================================================================
# METRICS (same as baselines.py — duplicated for self-containment)
# =============================================================================

def log_loss(y_true, y_pred, eps=1e-15):
    """Binary cross-entropy loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def brier_score(y_true, y_pred):
    """Mean squared error of probability predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def calibration_table(y_true, y_pred, n_bins=10):
    """Binned calibration: predicted probability vs actual frequency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_pred >= lo) & (y_pred < hi)
        else:
            mask = (y_pred >= lo) & (y_pred <= hi)
        
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
# DATA LOADING AND SPLITTING
# =============================================================================

def load_and_split(features_path):
    """
    Load features and perform time-based split.
    
    Returns train, val, test DataFrames plus the feature matrix arrays.
    """
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Sort by time and split
    df = df.sort_values('ts').reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    def ts_to_date(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
    
    print(f"\nTime-based split:")
    print(f"  Train: {len(train_df):>8,} rows  "
          f"({ts_to_date(train_df['ts'].min())} to {ts_to_date(train_df['ts'].max())})")
    print(f"  Val:   {len(val_df):>8,} rows  "
          f"({ts_to_date(val_df['ts'].min())} to {ts_to_date(val_df['ts'].max())})")
    print(f"  Test:  {len(test_df):>8,} rows  "
          f"({ts_to_date(test_df['ts'].min())} to {ts_to_date(test_df['ts'].max())})")
    
    train_rate = train_df['label'].mean()
    val_rate = val_df['label'].mean()
    test_rate = test_df['label'].mean()
    print(f"  Base rates:  train={train_rate:.1%}  val={val_rate:.1%}  test={test_rate:.1%}")
    
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df):
    """
    Extract feature matrices and standardize using TRAINING statistics only.
    
    Returns
    -------
    X_train, X_val, X_test : numpy arrays (standardized features)
    y_train, y_val, y_test : numpy arrays (labels)
    scaler : fitted StandardScaler (for later use / inspection)
    feature_names : list of feature column names
    
    === WHY WE STANDARDIZE ===
    
    Logistic regression finds weights w such that:
        P(y=1) = sigmoid(w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ)
    
    With L2 regularization, it penalizes Σ(wᵢ²). If x₁ ranges from
    -0.5 to +0.5 and x₂ ranges from 0 to 200, then w₂ needs to be
    ~400x smaller than w₁ to have a similar effect. The regularization
    penalty treats them equally, so it effectively regularizes x₁
    much more than x₂. Standardizing puts all features on the same
    scale so regularization is fair.
    
    === LEAKAGE PREVENTION ===
    
    We fit the scaler on TRAINING data only:
        scaler.fit(X_train)  # learns mean, std from training
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)    # uses training mean/std
        X_test = scaler.transform(X_test)  # uses training mean/std
    
    If we fit on the entire dataset, the training features would be
    centered using statistics that include future data. Subtle, but
    it's a form of leakage.
    """
    # Check which features are available
    available = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in train_df.columns]
    
    if missing:
        print(f"\n  ⚠ Missing features (will skip): {missing}")
    
    # Handle NaN in expiry features — fill with median from training set
    # This is a simple imputation strategy. More sophisticated approaches
    # exist but aren't worth the complexity for a first model.
    for col in ['days_to_expiry', 'log_days_to_expiry']:
        if col in available:
            median_val = train_df[col].median()
            for split_df in [train_df, val_df, test_df]:
                split_df[col] = split_df[col].fillna(median_val)
    
    # Extract feature matrices
    X_train = train_df[available].values.astype(np.float64)
    X_val = val_df[available].values.astype(np.float64)
    X_test = test_df[available].values.astype(np.float64)
    
    y_train = train_df['label'].values.astype(np.float64)
    y_val = val_df['label'].values.astype(np.float64)
    y_test = test_df['label'].values.astype(np.float64)
    
    # Check for any remaining NaN/inf
    for name, X in [('train', X_train), ('val', X_val), ('test', X_test)]:
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠ {name}: {nan_count} NaN, {inf_count} inf values found")
            # Replace with 0 as a last resort
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train only
    X_val = scaler.transform(X_val)           # transform with train stats
    X_test = scaler.transform(X_test)         # transform with train stats
    
    print(f"\n  Features used: {len(available)}")
    print(f"  Feature matrix shapes: train={X_train.shape}, "
          f"val={X_val.shape}, test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, available


# =============================================================================
# MODEL TRAINING
# =============================================================================

def tune_regularization(X_train, y_train, X_val, y_val):
    """
    Try different regularization strengths and pick the best one.
    
    We train one model per C value and evaluate on the validation set.
    The C with the lowest validation log loss wins.
    
    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    
    Returns
    -------
    best_model : fitted LogisticRegression with best C
    results : list of dicts with C, train_loss, val_loss for each attempt
    
    === ABOUT THE C PARAMETER ===
    
    In sklearn, C = 1/λ where λ is the regularization strength.
    
    C = 0.001 → very strong regularization → weights forced near zero
                → model barely uses features → likely underfitting
    
    C = 10.0  → very weak regularization → weights can be large
                → model fully trusts features → risk of overfitting
    
    The sweet spot is usually somewhere in between. For most problems
    with ~20 features and ~100K training samples, C between 0.01 and 1.0
    tends to work well.
    
    === SOLVER CHOICE ===
    
    We use 'lbfgs' (Limited-memory BFGS), which is:
    - The default in modern sklearn
    - Good for small-to-medium datasets
    - Handles L2 regularization natively
    - Converges reliably for well-conditioned problems
    
    Alternative: 'saga' is better for very large datasets (>1M rows)
    but slower per iteration. With 160K training rows, lbfgs is fine.
    """
    print(f"\nTuning regularization (C)...")
    print(f"  {'C':>8} {'Train Loss':>12} {'Val Loss':>12} {'Val Brier':>12} {'Note':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    results = []
    best_val_loss = float('inf')
    best_model = None
    best_C = None
    
    for C in C_VALUES:
        # Train
        model = LogisticRegression(
            C=C,
            penalty='l2',           # Ridge regularization
            solver='lbfgs',         # Reliable general-purpose solver
            max_iter=1000,          # Enough iterations to converge
            random_state=42,        # Reproducibility
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        
        train_loss = log_loss(y_train, train_probs)
        val_loss = log_loss(y_val, val_probs)
        val_brier = brier_score(y_val, val_probs)
        
        # Track best
        note = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_C = C
            note = "← best"
        
        print(f"  {C:>8.3f} {train_loss:>12.4f} {val_loss:>12.4f} "
              f"{val_brier:>12.4f} {note:>10}")
        
        results.append({
            'C': C,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_brier': val_brier,
        })
    
    print(f"\n  Best C: {best_C}")
    
    # Check for overfitting signal
    best_result = [r for r in results if r['C'] == best_C][0]
    overfit_gap = best_result['val_loss'] - best_result['train_loss']
    
    if overfit_gap > 0.05:
        print(f"  ⚠ Train-val gap: {overfit_gap:.4f} — possible overfitting")
        print(f"    Try stronger regularization (smaller C) or fewer features")
    elif overfit_gap < 0.001:
        print(f"  Train-val gap: {overfit_gap:.4f} — model may be underfitting")
        print(f"    Could try weaker regularization (larger C)")
    else:
        print(f"  Train-val gap: {overfit_gap:.4f} — looks healthy")
    
    return best_model, results


# =============================================================================
# EVALUATION AND REPORTING
# =============================================================================

def evaluate_model(model, X, y, split_name):
    """
    Full evaluation of the trained model on a given split.
    
    Returns dict with all metrics.
    """
    probs = model.predict_proba(X)[:, 1]
    
    ll = log_loss(y, probs)
    bs = brier_score(y, probs)
    cal = calibration_table(y, probs)
    
    return {
        'split': split_name,
        'log_loss': ll,
        'brier': bs,
        'calibration': cal,
        'predictions': probs,
    }


def print_comparison(model_result, baseline_results_path):
    """
    Print side-by-side comparison of model vs baselines.
    """
    print(f"\n{'=' * 70}")
    print(f"MODEL vs BASELINES (Test Set)")
    print(f"{'=' * 70}")
    
    # Load baseline results
    if baseline_results_path.exists():
        baselines = pd.read_csv(baseline_results_path)
        baselines = baselines[baselines['split'] == 'test']
        
        print(f"\n  {'Model':<30} {'Log Loss':>10} {'Brier':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        
        for _, row in baselines.iterrows():
            print(f"  {row['model']:<30} {row['log_loss']:>10.4f} {row['brier']:>10.4f}")
        
        print(f"  {'─'*30} {'─'*10} {'─'*10}")
        print(f"  {'Logistic Regression':<30} "
              f"{model_result['log_loss']:>10.4f} {model_result['brier']:>10.4f}")
        
        # Compare to best baseline
        best_baseline_ll = baselines['log_loss'].min()
        improvement = best_baseline_ll - model_result['log_loss']
        
        if improvement > 0:
            print(f"\n  ✓ BEATS best baseline by {improvement:.4f} log loss")
            pct = (improvement / best_baseline_ll) * 100
            print(f"    ({pct:.2f}% relative improvement)")
        else:
            print(f"\n  ✗ WORSE than best baseline by {abs(improvement):.4f} log loss")
            print(f"    Debug checklist:")
            print(f"    - Check for feature leakage (features using future data)")
            print(f"    - Verify time split is correct (train before val before test)")
            print(f"    - Try removing features that might be noisy")
            print(f"    - Check if standardization is working correctly")
    else:
        print(f"\n  Logistic Regression: log_loss={model_result['log_loss']:.4f}, "
              f"brier={model_result['brier']:.4f}")
        print(f"\n  (Run baselines.py first to get comparison numbers)")


def print_calibration(cal_df):
    """Print calibration table with interpretation."""
    print(f"\nCalibration Table:")
    print(f"  {'Bin':<10} {'Predicted':>10} {'Actual':>10} {'Count':>8} {'Error':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    
    total_cal_error = 0
    total_count = 0
    
    for _, row in cal_df.iterrows():
        if row['count'] > 0 and pd.notna(row['avg_predicted']):
            error = abs(row['avg_predicted'] - row['actual_frequency'])
            pred_str = f"{row['avg_predicted']:.3f}"
            act_str = f"{row['actual_frequency']:.3f}"
            err_str = f"{error:.3f}"
            total_cal_error += error * row['count']
            total_count += row['count']
            print(f"  {row['bin']:<10} {pred_str:>10} {act_str:>10} "
                  f"{row['count']:>8,} {err_str:>8}")
    
    if total_count > 0:
        ece = total_cal_error / total_count
        print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
        if ece < 0.02:
            print(f"  ✓ Excellent calibration")
        elif ece < 0.05:
            print(f"  ✓ Good calibration")
        elif ece < 0.10:
            print(f"  ~ Acceptable calibration, could be improved")
        else:
            print(f"  ⚠ Poor calibration — consider Platt scaling or isotonic regression")


def print_feature_importance(model, feature_names, scaler):
    """
    Print feature importances from the logistic regression coefficients.
    
    === INTERPRETING COEFFICIENTS ===
    
    Because we standardized the features, the coefficients are
    directly comparable in magnitude. A larger absolute coefficient
    means the feature has more influence on the prediction.
    
    Positive coefficient → higher feature value → higher predicted
    probability of y=1 (price rise)
    
    Negative coefficient → higher feature value → lower predicted
    probability of y=1
    
    Example: if vol_12 has coefficient +0.3, then when vol_12 is
    1 standard deviation above its mean, the log-odds of y=1
    increase by 0.3, which shifts the predicted probability upward.
    
    CAVEAT: Coefficients in correlated features can be misleading.
    If vol_6 and vol_12 are highly correlated, the model might put
    weight on one and not the other, even though both are informative.
    Don't interpret individual coefficients too literally.
    """
    print(f"\nFeature Importance (Standardized Coefficients):")
    print(f"  {'Feature':<22} {'Coefficient':>12} {'|Coefficient|':>14}")
    print(f"  {'-'*22} {'-'*12} {'-'*14}")
    
    coefficients = model.coef_[0]
    
    # Sort by absolute value
    indices = np.argsort(np.abs(coefficients))[::-1]
    
    for idx in indices:
        name = feature_names[idx]
        coef = coefficients[idx]
        abs_coef = abs(coef)
        
        # Visual bar
        bar_len = int(abs_coef / max(np.abs(coefficients)) * 20)
        bar = '█' * bar_len
        direction = '+' if coef > 0 else '-'
        
        print(f"  {name:<22} {coef:>+12.4f} {abs_coef:>14.4f}  {direction} {bar}")
    
    print(f"\n  Intercept (bias): {model.intercept_[0]:+.4f}")
    
    # Interpretation help
    top_pos = [(feature_names[i], coefficients[i]) 
               for i in indices if coefficients[i] > 0][:3]
    top_neg = [(feature_names[i], coefficients[i]) 
               for i in indices if coefficients[i] < 0][:3]
    
    if top_pos:
        print(f"\n  Top predictors of UPWARD movement (y=1):")
        for name, coef in top_pos:
            print(f"    {name}: +{coef:.4f} (higher {name} → more likely to rise)")
    
    if top_neg:
        print(f"\n  Top predictors AGAINST movement (y=0):")
        for name, coef in top_neg:
            print(f"    {name}: {coef:.4f} (higher {name} → less likely to rise)")


def print_prediction_distribution(y_pred, y_true):
    """
    Show how the model's predictions are distributed.
    
    A good model should spread its predictions across a range.
    A model that predicts ~0.27 for everything hasn't learned much
    beyond the base rate.
    """
    print(f"\nPrediction Distribution (Test Set):")
    print(f"  Min:    {y_pred.min():.4f}")
    print(f"  25th:   {np.percentile(y_pred, 25):.4f}")
    print(f"  Median: {np.median(y_pred):.4f}")
    print(f"  75th:   {np.percentile(y_pred, 75):.4f}")
    print(f"  Max:    {y_pred.max():.4f}")
    print(f"  Std:    {y_pred.std():.4f}")
    
    spread = y_pred.max() - y_pred.min()
    if spread < 0.05:
        print(f"  ⚠ Very narrow spread — model predictions are nearly constant")
        print(f"    This means the model hasn't found much signal beyond the base rate")
    elif spread < 0.15:
        print(f"  ~ Moderate spread — model is making some differentiated predictions")
    else:
        print(f"  ✓ Good spread — model is confidently differentiating cases")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chunk 5: Train logistic regression with walk-forward evaluation"
    )
    parser.add_argument('--features', type=str, default=str(FEATURES_PATH),
                        help=f'Path to features CSV')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CHUNK 5: LOGISTIC REGRESSION")
    print("=" * 70)
    
    start_time = time.time()
    
    # ── Load and split ──
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"ERROR: Features file not found at {features_path}")
        print("Run 'python src/features.py' first (Chunk 3).")
        sys.exit(1)
    
    train_df, val_df, test_df = load_and_split(features_path)
    
    # ── Prepare features ──
    print(f"\nPreparing features...")
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, 
     scaler, feature_names) = prepare_features(train_df, val_df, test_df)
    
    # ── Tune regularization ──
    best_model, tune_results = tune_regularization(
        X_train, y_train, X_val, y_val
    )
    
    # ── Evaluate on all splits ──
    print(f"\n{'=' * 70}")
    print(f"EVALUATION")
    print(f"{'=' * 70}")
    
    train_result = evaluate_model(best_model, X_train, y_train, "train")
    val_result = evaluate_model(best_model, X_val, y_val, "val")
    test_result = evaluate_model(best_model, X_test, y_test, "test")
    
    print(f"\n  {'Split':<10} {'Log Loss':>10} {'Brier':>10} {'Base Rate':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Train':<10} {train_result['log_loss']:>10.4f} "
          f"{train_result['brier']:>10.4f} {y_train.mean():>10.1%}")
    print(f"  {'Val':<10} {val_result['log_loss']:>10.4f} "
          f"{val_result['brier']:>10.4f} {y_val.mean():>10.1%}")
    print(f"  {'Test':<10} {test_result['log_loss']:>10.4f} "
          f"{test_result['brier']:>10.4f} {y_test.mean():>10.1%}")
    
    # ── Compare to baselines ──
    baseline_results_path = RESULTS_DIR / 'baseline_results.csv'
    print_comparison(test_result, baseline_results_path)
    
    # ── Calibration ──
    print_calibration(test_result['calibration'])
    
    # ── Feature importance ──
    print_feature_importance(best_model, feature_names, scaler)
    
    # ── Prediction distribution ──
    print_prediction_distribution(test_result['predictions'], y_test)
    
    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model results alongside baselines
    model_summary = pd.DataFrame([{
        'model': 'Logistic Regression',
        'split': 'test',
        'log_loss': test_result['log_loss'],
        'brier': test_result['brier'],
        'C': best_model.C,
    }])
    
    summary_path = RESULTS_DIR / 'model_results.csv'
    model_summary.to_csv(summary_path, index=False)
    
    # Save tuning history
    tune_df = pd.DataFrame(tune_results)
    tune_path = RESULTS_DIR / 'tuning_history.csv'
    tune_df.to_csv(tune_path, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  Model results:   {summary_path}")
    print(f"  Tuning history:  {tune_path}")
    print(f"\nNext step: Chunk 6 — Gradient boosted trees (LightGBM/XGBoost)")


if __name__ == "__main__":
    main()