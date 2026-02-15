#!/usr/bin/env python3
"""
gbm_eval.py — Chunk 6: Gradient Boosted Trees

"Patterns humans wouldn't find"

This is where we let the model discover nonlinear relationships and
feature interactions that logistic regression can't capture.

=== WHY GRADIENT BOOSTED TREES ===

Logistic regression learns: w₁×vol_12 + w₂×p_from_center + ...
It can only combine features additively with fixed weights.

Gradient boosted trees learn rules like:
  "IF vol_12 > 0.005 AND p_from_center < 0.3 AND dp_6 > 0
   THEN probability is 0.45"

This captures:
- Nonlinear thresholds (vol matters a lot above some level, not below)
- Feature interactions (volatility matters more at certain price levels)
- Conditional relationships (momentum matters only when vol is high)

=== WHY LIGHTGBM ===

LightGBM is the standard choice for tabular data because:
- Fast training (histogram-based splitting)
- Handles mixed feature types well
- Built-in handling of missing values
- Good default hyperparameters
- Widely used in Kaggle competitions and production systems

XGBoost is equally valid. The differences are minor for this dataset size.

=== WHAT THIS SCRIPT DOES ===

1. Loads features (same split as Chunks 4-5)
2. Trains LightGBM with hyperparameter tuning
3. Compares against logistic regression and baselines
4. Analyzes feature importance (permutation + split-based)
5. Checks stability: are important features consistent across time?
6. Optionally runs SHAP for deeper interpretation

=== USAGE ===

    cd /mnt/ml-data/projects/polymarket-predictor
    python src/gbm_eval.py

    # Skip SHAP (faster):
    python src/gbm_eval.py --no-shap
"""

import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: LightGBM is required.")
    print("Install it: pip install lightgbm --break-system-packages")
    print("  or: conda install -c conda-forge lightgbm")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required.")
    print("Install it: pip install scikit-learn --break-system-packages")
    sys.exit(1)

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_PATH = Path(__file__).parent.parent / "data" / "features.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

FEATURE_COLUMNS = [
    'p', 'p_complement', 'p_from_center',
    'dp_3', 'dp_6', 'dp_12', 'dp_36',
    'ddp_3', 'ddp_6',
    'vol_6', 'vol_12', 'vol_36',
    'p_vs_6h_high', 'p_vs_6h_mean',
    'hour_sin', 'hour_cos', 'is_weekend',
    'days_to_expiry', 'log_days_to_expiry',
]

# Hyperparameter grid for LightGBM
# These are the parameters that matter most for performance.
#
# num_leaves: Controls tree complexity. More leaves = more complex trees.
#   Too many → overfitting. Too few → underfitting.
#   Rule of thumb: num_leaves < 2^max_depth
#
# learning_rate: How much each tree contributes. Lower = more trees needed
#   but usually better generalization. We pair it with n_estimators.
#
# n_estimators: Number of boosting rounds. With early stopping, this is
#   just an upper bound — training stops when validation loss plateaus.
#
# min_child_samples: Minimum data points in a leaf. Higher = more conservative.
#   Prevents the model from creating rules based on tiny subsets.
#
# reg_alpha (L1) and reg_lambda (L2): Regularization on leaf weights.
#   Similar purpose to C in logistic regression.
#
# subsample and colsample_bytree: Randomly use only a fraction of rows/features
#   per tree. Reduces overfitting by adding randomness (like dropout in neural nets).

PARAM_GRID = [
    # Config 1: Conservative (fewer leaves, stronger regularization)
    {
        'num_leaves': 15,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    # Config 2: Moderate
    {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 50,
        'reg_alpha': 0.01,
        'reg_lambda': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    # Config 3: More complex (more leaves, less regularization)
    {
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 30,
        'reg_alpha': 0.0,
        'reg_lambda': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
    },
    # Config 4: Slower learning, more trees
    {
        'num_leaves': 31,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'min_child_samples': 50,
        'reg_alpha': 0.01,
        'reg_lambda': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    # Config 5: Conservative with slow learning
    {
        'num_leaves': 15,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
    },
]


# =============================================================================
# METRICS
# =============================================================================

def log_loss(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def brier_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_pred - y_true) ** 2)


def calibration_table(y_true, y_pred, n_bins=10):
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
            rows.append({
                'bin': f'{lo:.1f}-{hi:.1f}',
                'avg_predicted': y_pred[mask].mean(),
                'actual_frequency': y_true[mask].mean(),
                'count': count
            })
        else:
            rows.append({
                'bin': f'{lo:.1f}-{hi:.1f}',
                'avg_predicted': np.nan,
                'actual_frequency': np.nan,
                'count': 0
            })
    return pd.DataFrame(rows)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_split(features_path):
    """Load features and split by time. Same logic as train_eval.py."""
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
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
    
    print(f"  Base rates:  train={train_df['label'].mean():.1%}  "
          f"val={val_df['label'].mean():.1%}  test={test_df['label'].mean():.1%}")
    
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df):
    """
    Extract feature matrices for LightGBM.
    
    KEY DIFFERENCE FROM LOGISTIC REGRESSION:
    LightGBM does NOT need standardization. Tree-based models split on
    thresholds, so the scale of features doesn't matter. A split at
    "vol_12 > 0.003" works the same whether vol_12 is raw or z-scored.
    
    We still handle NaN in expiry features, but skip the StandardScaler.
    This is actually an advantage of tree models — fewer preprocessing
    decisions to get wrong.
    """
    available = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    
    # Handle NaN in expiry columns
    for col in ['days_to_expiry', 'log_days_to_expiry']:
        if col in available:
            median_val = train_df[col].median()
            for split_df in [train_df, val_df, test_df]:
                split_df[col] = split_df[col].fillna(median_val)
    
    X_train = train_df[available].values.astype(np.float64)
    X_val = val_df[available].values.astype(np.float64)
    X_test = test_df[available].values.astype(np.float64)
    
    y_train = train_df['label'].values.astype(np.float64)
    y_val = val_df['label'].values.astype(np.float64)
    y_test = test_df['label'].values.astype(np.float64)
    
    # Replace any remaining NaN/inf
    for X in [X_train, X_val, X_test]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n  Features used: {len(available)}")
    print(f"  No standardization needed (tree-based model)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, available


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """
    Train LightGBM with hyperparameter search and early stopping.
    
    === EARLY STOPPING ===
    
    Instead of training for a fixed number of rounds, we monitor
    validation loss and stop when it hasn't improved for 50 rounds.
    This automatically finds the right number of trees for each
    configuration, preventing overfitting.
    
    Example: if we set n_estimators=2000 but validation loss plateaus
    at round 300, training stops at round 350 (300 + 50 patience).
    The model uses the weights from round 300.
    
    === VERBOSE CONTROL ===
    
    LightGBM can be very chatty. We suppress per-round output and
    only show summaries.
    """
    print(f"\nTraining LightGBM with {len(PARAM_GRID)} configurations...")
    print(f"  {'Config':>8} {'Leaves':>8} {'LR':>8} {'Val Loss':>10} "
          f"{'Val Brier':>10} {'Rounds':>8} {'Note':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
    
    best_val_loss = float('inf')
    best_model = None
    best_config_idx = None
    all_results = []
    
    for i, params in enumerate(PARAM_GRID):
        # LightGBM parameters
        lgb_params = {
            'objective': 'binary',        # Binary classification
            'metric': 'binary_logloss',   # Optimize log loss
            'boosting_type': 'gbdt',      # Standard gradient boosting
            'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'],
            'min_child_samples': params['min_child_samples'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'verbose': -1,                # Suppress output
            'feature_pre_filter': False,  # Allow different min_child_samples per config
            'seed': 42,
        }
        
        # Train with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress per-round logging
        ]
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=[val_data],
            callbacks=callbacks,
        )
        
        # Evaluate
        val_probs = model.predict(X_val)
        val_loss = log_loss(y_val, val_probs)
        val_brier = brier_score(y_val, val_probs)
        
        note = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_config_idx = i
            note = "← best"
        
        print(f"  {i+1:>8} {params['num_leaves']:>8} {params['learning_rate']:>8.3f} "
              f"{val_loss:>10.4f} {val_brier:>10.4f} {model.best_iteration:>8} {note:>8}")
        
        all_results.append({
            'config': i + 1,
            'num_leaves': params['num_leaves'],
            'learning_rate': params['learning_rate'],
            'n_estimators_used': model.best_iteration,
            'val_loss': val_loss,
            'val_brier': val_brier,
        })
    
    best_params = PARAM_GRID[best_config_idx]
    print(f"\n  Best config: #{best_config_idx + 1} "
          f"(leaves={best_params['num_leaves']}, "
          f"lr={best_params['learning_rate']}, "
          f"rounds={best_model.best_iteration})")
    
    return best_model, all_results


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_split_importance(model, feature_names):
    """
    Split-based importance: how often each feature was used to split.
    
    This tells you which features the model relied on for making decisions.
    A feature with high split importance was frequently chosen as the best
    way to partition the data.
    
    CAVEAT: Correlated features share importance. If vol_6 and vol_12
    are highly correlated, the model might use vol_6 in some trees and
    vol_12 in others, making both look moderately important even though
    they carry the same signal.
    """
    importance = model.feature_importance(importance_type='split')
    
    # Normalize to sum to 1
    total = importance.sum()
    if total > 0:
        importance_pct = importance / total
    else:
        importance_pct = importance
    
    # Sort by importance
    indices = np.argsort(importance_pct)[::-1]
    
    print(f"\nSplit-based Feature Importance:")
    print(f"  (How often each feature was used to make a split decision)")
    print(f"\n  {'Feature':<22} {'Splits':>8} {'Share':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8}")
    
    for idx in indices:
        name = feature_names[idx]
        splits = importance[idx]
        pct = importance_pct[idx]
        bar = '█' * int(pct / max(importance_pct) * 20)
        print(f"  {name:<22} {splits:>8} {pct:>7.1%}  {bar}")
    
    return dict(zip(feature_names, importance_pct))


def analyze_gain_importance(model, feature_names):
    """
    Gain-based importance: total reduction in loss from each feature's splits.
    
    While split importance counts how often a feature is used, gain importance
    measures how much each feature improves the model when it IS used.
    
    A feature could have low split importance (rarely used) but high gain
    importance (very helpful when it is used).
    """
    importance = model.feature_importance(importance_type='gain')
    
    total = importance.sum()
    if total > 0:
        importance_pct = importance / total
    else:
        importance_pct = importance
    
    indices = np.argsort(importance_pct)[::-1]
    
    print(f"\nGain-based Feature Importance:")
    print(f"  (Total loss reduction from each feature's splits)")
    print(f"\n  {'Feature':<22} {'Gain':>10} {'Share':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8}")
    
    for idx in indices:
        name = feature_names[idx]
        gain = importance[idx]
        pct = importance_pct[idx]
        bar = '█' * int(pct / max(importance_pct) * 20)
        print(f"  {name:<22} {gain:>10.1f} {pct:>7.1%}  {bar}")
    
    return dict(zip(feature_names, importance_pct))


def permutation_importance(model, X, y, feature_names, n_repeats=5):
    """
    Permutation importance: how much does loss increase when we scramble
    each feature?
    
    === HOW IT WORKS ===
    
    For each feature:
    1. Compute baseline loss with all features intact
    2. Randomly shuffle that one feature (breaking its relationship with y)
    3. Compute loss again
    4. The increase in loss = how much the model needed that feature
    
    Repeat n_repeats times and average (to reduce randomness).
    
    === WHY THIS IS BETTER THAN SPLIT/GAIN IMPORTANCE ===
    
    Split and gain importance are computed during training and can be
    misleading with correlated features. Permutation importance is
    computed on held-out data and directly measures the impact on
    actual predictions.
    
    If a feature has high split importance but low permutation importance,
    the model uses it but doesn't really need it (probably redundant
    with another feature).
    """
    print(f"\nPermutation Importance (on validation set, {n_repeats} repeats):")
    print(f"  (How much does loss increase when each feature is scrambled?)")
    
    baseline_loss = log_loss(y, model.predict(X))
    importances = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        losses = []
        
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, feat_idx])
            preds = model.predict(X_shuffled)
            losses.append(log_loss(y, preds))
        
        mean_loss = np.mean(losses)
        importance = mean_loss - baseline_loss
        importances[feat_name] = importance
    
    # Sort and print
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    max_imp = max(v for _, v in sorted_feats) if sorted_feats else 1
    
    print(f"\n  Baseline loss: {baseline_loss:.4f}")
    print(f"\n  {'Feature':<22} {'Δ Loss':>10} {'Interpretation':>20}")
    print(f"  {'-'*22} {'-'*10} {'-'*20}")
    
    for name, imp in sorted_feats:
        if imp > 0.001:
            interp = "Important"
        elif imp > 0.0001:
            interp = "Minor signal"
        elif imp > 0:
            interp = "Negligible"
        else:
            interp = "Not useful"
        
        bar_len = max(0, int(imp / max(max_imp, 1e-6) * 15))
        bar = '█' * bar_len
        print(f"  {name:<22} {imp:>+10.4f} {interp:>20}  {bar}")
    
    return importances


# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def check_stability(X_train, y_train, feature_names):
    """
    Check if feature importance is consistent across different time periods.
    
    === WHY THIS MATTERS ===
    
    If vol_12 is the most important feature in week 1 but irrelevant in
    week 3, your model is fragile — it learned patterns that don't persist.
    Stable importance means the signal is durable and likely to continue
    working in production.
    
    We split training data into 3 equal time chunks and train a separate
    model on each, then compare their feature importances.
    
    === WHAT TO LOOK FOR ===
    
    - Features that are consistently top-5 across all periods → reliable
    - Features that swing wildly → unreliable, model may be fitting noise
    - If rankings change dramatically → the market regime shifted
    """
    print(f"\n{'=' * 70}")
    print(f"FEATURE STABILITY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Training 3 models on different time windows...")
    
    n = len(X_train)
    chunk_size = n // 3
    
    period_importances = []
    
    for period in range(3):
        start = period * chunk_size
        end = start + chunk_size if period < 2 else n
        
        X_chunk = X_train[start:end]
        y_chunk = y_train[start:end]
        
        # Train a quick model on this chunk
        train_data = lgb.Dataset(X_chunk, label=y_chunk, feature_name=feature_names)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'min_child_samples': 50,
            'feature_pre_filter': False,
            'verbose': -1,
            'seed': 42,
        }
        
        model = lgb.train(params, train_data, num_boost_round=200)
        
        # Get gain importance
        imp = model.feature_importance(importance_type='gain')
        total = imp.sum()
        if total > 0:
            imp = imp / total
        
        period_importances.append(dict(zip(feature_names, imp)))
    
    # Compare across periods
    print(f"\n  Feature importance share by time period:")
    print(f"  {'Feature':<22} {'Period 1':>10} {'Period 2':>10} {'Period 3':>10} {'Stable?':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    # Sort by average importance
    avg_imp = {}
    for feat in feature_names:
        vals = [period_importances[p].get(feat, 0) for p in range(3)]
        avg_imp[feat] = np.mean(vals)
    
    sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
    
    stable_features = []
    unstable_features = []
    
    for feat, avg in sorted_feats:
        vals = [period_importances[p].get(feat, 0) for p in range(3)]
        cv = np.std(vals) / (np.mean(vals) + 1e-10)  # Coefficient of variation
        
        if cv < 0.5:
            stable = "✓ Stable"
            stable_features.append(feat)
        elif cv < 1.0:
            stable = "~ Mixed"
        else:
            stable = "✗ Unstable"
            unstable_features.append(feat)
        
        print(f"  {feat:<22} {vals[0]:>9.1%} {vals[1]:>9.1%} {vals[2]:>9.1%} {stable:>10}")
    
    print(f"\n  Stable features (CV < 0.5):   {len(stable_features)}")
    print(f"  Unstable features (CV > 1.0): {len(unstable_features)}")
    
    if unstable_features:
        print(f"  ⚠ Unstable: {', '.join(unstable_features)}")
        print(f"    These features may not generalize well to future data.")
    
    return period_importances


# =============================================================================
# SHAP ANALYSIS (OPTIONAL)
# =============================================================================

def run_shap_analysis(model, X_val, feature_names):
    """
    SHAP (SHapley Additive exPlanations) analysis.
    
    === WHAT SHAP DOES ===
    
    For each prediction, SHAP tells you exactly how much each feature
    contributed to pushing the prediction above or below the average.
    
    Example: For a specific sample where model predicts 0.45:
      - Average prediction is 0.27
      - vol_12 pushed it up by +0.08
      - p_from_center pushed it up by +0.06
      - dp_36 pushed it down by -0.03
      - etc.
    
    This is much more informative than global importance because it
    shows the DIRECTION and MAGNITUDE of each feature's effect.
    
    === INTERPRETATION ===
    
    SHAP summary plot (printed as text here):
    - Features sorted by mean |SHAP value| (most impactful first)
    - Shows whether high feature values push predictions up or down
    """
    try:
        import shap
    except ImportError:
        print("\n  SHAP not installed. Skipping.")
        print("  Install with: pip install shap --break-system-packages")
        return None
    
    print(f"\nSHAP Analysis (this may take a minute)...")
    
    # Use a sample for speed (SHAP on 30K+ rows is slow)
    sample_size = min(5000, len(X_val))
    indices = np.random.RandomState(42).choice(len(X_val), sample_size, replace=False)
    X_sample = X_val[indices]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values might be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Class 1 (price rises)
    else:
        shap_vals = shap_values
    
    # Global importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    indices_sorted = np.argsort(mean_abs_shap)[::-1]
    
    print(f"\n  SHAP Feature Importance (mean |SHAP value|):")
    print(f"  {'Feature':<22} {'Mean |SHAP|':>12} {'Direction':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*10}")
    
    for idx in indices_sorted:
        name = feature_names[idx]
        mean_shap = mean_abs_shap[idx]
        
        # Check direction: correlation between feature value and SHAP value
        corr = np.corrcoef(X_sample[:, idx], shap_vals[:, idx])[0, 1]
        if corr > 0.1:
            direction = "↑ Higher=up"
        elif corr < -0.1:
            direction = "↓ Higher=dn"
        else:
            direction = "~ Complex"
        
        bar = '█' * int(mean_shap / max(mean_abs_shap) * 15)
        print(f"  {name:<22} {mean_shap:>12.4f} {direction:>10}  {bar}")
    
    return shap_vals


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chunk 6: Gradient boosted trees with LightGBM"
    )
    parser.add_argument('--features', type=str, default=str(FEATURES_PATH))
    parser.add_argument('--no-shap', action='store_true',
                        help='Skip SHAP analysis (faster)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CHUNK 6: GRADIENT BOOSTED TREES (LightGBM)")
    print("=" * 70)
    
    start_time = time.time()
    
    # ── Load and split ──
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"ERROR: Features file not found at {features_path}")
        sys.exit(1)
    
    train_df, val_df, test_df = load_and_split(features_path)
    
    # ── Prepare features ──
    print(f"\nPreparing features...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names) = prepare_features(train_df, val_df, test_df)
    
    # ── Train ──
    best_model, tune_results = train_lightgbm(
        X_train, y_train, X_val, y_val, feature_names
    )
    
    # ── Evaluate ──
    print(f"\n{'=' * 70}")
    print(f"EVALUATION")
    print(f"{'=' * 70}")
    
    train_probs = best_model.predict(X_train)
    val_probs = best_model.predict(X_val)
    test_probs = best_model.predict(X_test)
    
    train_ll = log_loss(y_train, train_probs)
    val_ll = log_loss(y_val, val_probs)
    test_ll = log_loss(y_test, test_probs)
    
    train_bs = brier_score(y_train, train_probs)
    val_bs = brier_score(y_val, val_probs)
    test_bs = brier_score(y_test, test_probs)
    
    print(f"\n  {'Split':<10} {'Log Loss':>10} {'Brier':>10} {'Base Rate':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Train':<10} {train_ll:>10.4f} {train_bs:>10.4f} {y_train.mean():>10.1%}")
    print(f"  {'Val':<10} {val_ll:>10.4f} {val_bs:>10.4f} {y_val.mean():>10.1%}")
    print(f"  {'Test':<10} {test_ll:>10.4f} {test_bs:>10.4f} {y_test.mean():>10.1%}")
    
    # ── Compare all models ──
    print(f"\n{'=' * 70}")
    print(f"ALL MODELS COMPARISON (Test Set)")
    print(f"{'=' * 70}")
    
    baseline_path = RESULTS_DIR / 'baseline_results.csv'
    logreg_path = RESULTS_DIR / 'model_results.csv'
    
    print(f"\n  {'Model':<30} {'Log Loss':>10} {'Brier':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    
    # Load and print baselines
    if baseline_path.exists():
        baselines = pd.read_csv(baseline_path)
        baselines = baselines[baselines['split'] == 'test']
        for _, row in baselines.iterrows():
            print(f"  {row['model']:<30} {row['log_loss']:>10.4f} {row['brier']:>10.4f}")
    
    # Load and print logistic regression
    if logreg_path.exists():
        logreg = pd.read_csv(logreg_path)
        logreg = logreg[logreg['split'] == 'test']
        for _, row in logreg.iterrows():
            print(f"  {row['model']:<30} {row['log_loss']:>10.4f} {row['brier']:>10.4f}")
    
    print(f"  {'─'*30} {'─'*10} {'─'*10}")
    print(f"  {'LightGBM':<30} {test_ll:>10.4f} {test_bs:>10.4f}")
    
    # Show improvement over logistic regression
    if logreg_path.exists():
        logreg_ll = logreg[logreg['split'] == 'test']['log_loss'].values[0]
        imp_vs_lr = logreg_ll - test_ll
        if imp_vs_lr > 0:
            print(f"\n  ✓ Beats logistic regression by {imp_vs_lr:.4f} log loss "
                  f"({(imp_vs_lr/logreg_ll)*100:.2f}% relative)")
        elif imp_vs_lr < -0.001:
            print(f"\n  ✗ Worse than logistic regression by {abs(imp_vs_lr):.4f}")
            print(f"    This can happen with overfitting or concept drift.")
        else:
            print(f"\n  ~ Roughly tied with logistic regression (Δ={imp_vs_lr:.4f})")
            print(f"    Linear model may be sufficient for this problem.")
    
    # ── Calibration ──
    cal = calibration_table(y_test, test_probs)
    print(f"\nCalibration Table (Test Set):")
    print(f"  {'Bin':<10} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    
    total_cal_error = 0
    total_count = 0
    for _, row in cal.iterrows():
        if row['count'] > 0 and pd.notna(row['avg_predicted']):
            error = abs(row['avg_predicted'] - row['actual_frequency'])
            total_cal_error += error * row['count']
            total_count += row['count']
            print(f"  {row['bin']:<10} {row['avg_predicted']:>10.3f} "
                  f"{row['actual_frequency']:>10.3f} {row['count']:>8,}")
    
    if total_count > 0:
        ece = total_cal_error / total_count
        print(f"\n  ECE: {ece:.4f}")
    
    # ── Feature importance ──
    print(f"\n{'=' * 70}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'=' * 70}")
    
    split_imp = analyze_split_importance(best_model, feature_names)
    gain_imp = analyze_gain_importance(best_model, feature_names)
    perm_imp = permutation_importance(best_model, X_val, y_val, feature_names)
    
    # ── Stability ──
    stability = check_stability(X_train, y_train, feature_names)
    
    # ── SHAP ──
    if not args.no_shap:
        shap_vals = run_shap_analysis(best_model, X_val, feature_names)
    
    # ── Prediction distribution ──
    print(f"\nPrediction Distribution (Test Set):")
    print(f"  Min:    {test_probs.min():.4f}")
    print(f"  25th:   {np.percentile(test_probs, 25):.4f}")
    print(f"  Median: {np.median(test_probs):.4f}")
    print(f"  75th:   {np.percentile(test_probs, 75):.4f}")
    print(f"  Max:    {test_probs.max():.4f}")
    print(f"  Std:    {test_probs.std():.4f}")
    
    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model results
    model_summary = pd.DataFrame([{
        'model': 'LightGBM',
        'split': 'test',
        'log_loss': test_ll,
        'brier': test_bs,
    }])
    
    # Append to existing results
    summary_path = RESULTS_DIR / 'model_results.csv'
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Remove old LightGBM rows if re-running
        existing = existing[existing['model'] != 'LightGBM']
        combined = pd.concat([existing, model_summary], ignore_index=True)
    else:
        combined = model_summary
    combined.to_csv(summary_path, index=False)
    
    # Save tuning history
    tune_df = pd.DataFrame(tune_results)
    tune_df.to_csv(RESULTS_DIR / 'gbm_tuning_history.csv', index=False)
    
    # Save feature importance comparison
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'split_importance': [split_imp.get(f, 0) for f in feature_names],
        'gain_importance': [gain_imp.get(f, 0) for f in feature_names],
        'permutation_importance': [perm_imp.get(f, 0) for f in feature_names],
    })
    imp_df = imp_df.sort_values('permutation_importance', ascending=False)
    imp_df.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  Model results:        {summary_path}")
    print(f"  Feature importance:    {RESULTS_DIR / 'feature_importance.csv'}")
    print(f"  GBM tuning history:   {RESULTS_DIR / 'gbm_tuning_history.csv'}")
    print(f"\nNext step: Chunk 7 — Decision rule + paper trading")


if __name__ == "__main__":
    main()
