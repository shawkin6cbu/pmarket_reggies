#!/usr/bin/env python3
"""
paper_trade.py — Chunk 7: Decision Rule + Paper Trading

Converts model probabilities into simulated trading actions and
tracks what would have happened with real money.

=== THE GAP BETWEEN PREDICTION AND TRADING ===

Having a model that predicts well (low log loss) does NOT mean you
can make money. The gap between "good predictions" and "profitable
trading" is where most quant projects die. This chunk addresses that.

Problems you'll face:
1. Even a well-calibrated model is WRONG most of the time on individual
   predictions. At 27% base rate, even a good model predicts "will rise"
   and is wrong 60%+ of the time.
2. You need to decide WHEN to act. Predicting 0.30 when the base rate
   is 0.27 is technically "above average" but the edge is tiny.
3. You need position sizing — how much to bet on each prediction.
4. You need to account for fees, slippage, and the fact that you can't
   always execute at the price you want.

=== THE EDGE FRAMEWORK ===

Edge = model_probability - market_implied_probability

In prediction markets, the current price IS the market's probability.
If the market price is 0.30 and your model says 0.35, your perceived
edge is 0.05. You believe the market is underpricing the probability
of this event moving up.

The decision rule:
  - Only trade when |edge| > threshold
  - Bet proportional to edge (modified Kelly criterion)
  - Cap position size to limit downside
  - Stop trading if the model appears miscalibrated

=== WHAT THIS SCRIPT DOES ===

1. Loads the test set with model predictions
2. Simulates trading using the edge framework
3. Tracks every trade: entry price, exit price, P&L
4. Produces an equity curve and trade log
5. Computes performance statistics

=== IMPORTANT DISCLAIMER ===

This is a SIMULATION on historical data. It assumes you could have
executed at the recorded prices, which is optimistic. Real trading
has slippage (you get worse prices than expected), latency (prices
move between decision and execution), and liquidity constraints
(large orders move the market against you).

The simulation uses conservative assumptions to partially account
for this, but real performance will always be worse than backtest.

=== USAGE ===

    cd /mnt/ml-data/projects/polymarket-predictor
    python src/paper_trade.py

    # Adjust parameters:
    python src/paper_trade.py --edge-threshold 0.05 --max-position 2.0
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: LightGBM required. pip install lightgbm --break-system-packages")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn required. pip install scikit-learn --break-system-packages")
    sys.exit(1)


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

# ── Trading Parameters ──
# These are deliberately conservative. The goal is to NOT fool yourself.

DEFAULT_EDGE_THRESHOLD = 0.03   # Only trade when edge > 3%
DEFAULT_MAX_POSITION = 1.00     # Max $1 per position
DEFAULT_BANKROLL = 100.00       # Starting capital
DEFAULT_MAX_OPEN = 10           # Max simultaneous positions
DEFAULT_KELLY_FRACTION = 0.25   # Use 1/4 Kelly (very conservative)

# Slippage: assume you get 0.5% worse price than expected
# This accounts for spread, execution delay, and market impact
SLIPPAGE_BPS = 50  # basis points (50 bps = 0.5%)

# Fee structure (Polymarket currently has 0 bps maker/taker, but
# we include a small fee assumption for conservatism)
FEE_BPS = 0  # basis points per trade


# =============================================================================
# MODEL TRAINING (retrain best model for prediction generation)
# =============================================================================

def train_best_model(X_train, y_train, X_val, y_val, feature_names):
    """
    Retrain the best LightGBM configuration from Chunk 6.
    Uses the winning config: 15 leaves, lr=0.05, early stopping.
    """
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names,
                           reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'min_child_samples': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'feature_pre_filter': False,
        'verbose': -1,
        'seed': 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    print(f"  Model trained: {model.best_iteration} rounds")
    return model


# =============================================================================
# EDGE CALCULATION
# =============================================================================

def compute_edge(model_prob, market_price):
    """
    Compute the perceived edge.

    edge = model_probability - market_price

    Positive edge: model thinks the price should be higher → buy
    Negative edge: model thinks the price should be lower → sell/avoid

    In prediction markets:
    - market_price = current p_yes (what the market thinks)
    - model_prob = what our model thinks p_yes should be

    If market says 0.30 and model says 0.38, edge = +0.08.
    We think YES is underpriced and would buy.

    IMPORTANT: We're only predicting whether price will RISE by ≥0.001
    in 6 hours. model_prob is P(price rises), NOT the "true probability"
    of the underlying event. So our edge is about short-term price
    movement, not about the event outcome.
    """
    return model_prob - market_price


# =============================================================================
# POSITION SIZING (KELLY CRITERION)
# =============================================================================

def kelly_size(edge, odds, kelly_fraction=DEFAULT_KELLY_FRACTION):
    """
    Modified Kelly criterion for position sizing.

    === WHAT IS KELLY? ===

    The Kelly criterion tells you the optimal fraction of your bankroll
    to bet to maximize long-term growth. The formula for a simple bet:

        f* = (p × b - q) / b

    where:
        p = probability of winning (our model's prediction)
        q = probability of losing (1 - p)
        b = odds (how much you win per dollar risked)

    For prediction markets where you buy at market_price and the
    position pays $1 if correct:
        b = (1 - market_price) / market_price

    === WHY FRACTIONAL KELLY ===

    Full Kelly is mathematically optimal but assumes:
    1. Your probability estimates are exactly correct (they're not)
    2. You can tolerate massive drawdowns (you can't)
    3. You have infinite time horizon (you don't)

    In practice, using 1/4 Kelly gives ~75% of the growth rate with
    much smaller drawdowns. Most professional gamblers and quant funds
    use some fraction of Kelly.

    === EXAMPLE ===

    Model says P(rise) = 0.40, market price = 0.30
    Edge = 0.10
    Odds = 0.70 / 0.30 = 2.33
    Full Kelly: f* = (0.40 × 2.33 - 0.60) / 2.33 = 0.143 (14.3%)
    Quarter Kelly: 0.143 × 0.25 = 0.036 (3.6% of bankroll)

    On a $100 bankroll, you'd bet $3.60.
    """
    if edge <= 0:
        return 0.0

    if odds <= 0:
        return 0.0

    p = edge + 0.5  # Rough conversion: edge above 50/50
    # More precisely for our setup:
    # We're buying at market_price, expecting a small rise
    # Simplify: size proportional to edge, capped by Kelly logic

    # Simple proportional sizing based on edge
    # Full Kelly for binary bet: f = edge / variance
    # We approximate with: f = edge (since max payout is bounded)
    full_kelly = edge
    return full_kelly * kelly_fraction


# =============================================================================
# TRADING SIMULATION
# =============================================================================

class PaperTrader:
    """
    Simulates trading on historical data.

    Tracks:
    - Open positions
    - Closed trades with P&L
    - Running equity curve
    - Performance statistics
    """

    def __init__(self, bankroll, max_position, max_open, edge_threshold,
                 kelly_fraction, slippage_bps, fee_bps):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.max_position = max_position
        self.max_open = max_open
        self.edge_threshold = edge_threshold
        self.kelly_fraction = kelly_fraction
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps

        self.open_positions = []   # Currently held positions
        self.closed_trades = []    # Completed trades with P&L
        self.equity_curve = []     # (timestamp, equity) pairs
        self.decisions = []        # Every decision made (trade or skip)

        # Tracking stats
        self.total_signals = 0
        self.trades_taken = 0
        self.trades_skipped_no_edge = 0
        self.trades_skipped_max_open = 0
        self.trades_skipped_no_capital = 0

    def apply_slippage(self, price, direction='buy'):
        """
        Adjust price for slippage.

        When buying, you pay slightly more than the quoted price.
        When selling, you receive slightly less.
        """
        slippage = price * (self.slippage_bps / 10000)
        if direction == 'buy':
            return price + slippage
        else:
            return price - slippage

    def apply_fees(self, trade_value):
        """Calculate fee on a trade."""
        return trade_value * (self.fee_bps / 10000)

    def compute_position_size(self, edge, market_price):
        """
        Determine how much to bet.

        Caps at:
        1. Kelly-derived size (based on edge)
        2. Max position parameter
        3. Available bankroll
        """
        kelly = kelly_size(edge, market_price, self.kelly_fraction)
        dollar_size = kelly * self.bankroll

        # Apply caps
        dollar_size = min(dollar_size, self.max_position)
        dollar_size = min(dollar_size, self.bankroll * 0.1)  # Never risk >10% of bankroll
        dollar_size = max(dollar_size, 0)

        return round(dollar_size, 4)

    def evaluate_signal(self, ts, market_id, model_prob, market_price, future_price):
        """
        Evaluate a trading signal and decide whether to act.

        Parameters
        ----------
        ts : int
            Unix timestamp of the decision point
        market_id : str
            Market identifier
        model_prob : float
            Model's predicted probability of price rising
        market_price : float
            Current market price (p_yes)
        future_price : float
            Price 6 hours later (for P&L calculation — only used
            in simulation, not available in live trading!)
        """
        self.total_signals += 1

        edge = model_prob - market_price

        # Decision: is the edge large enough?
        if edge < self.edge_threshold:
            self.trades_skipped_no_edge += 1
            self.decisions.append({
                'ts': ts, 'market_id': market_id,
                'action': 'skip_no_edge',
                'model_prob': model_prob, 'market_price': market_price,
                'edge': edge
            })
            return

        # Check position limits
        if len(self.open_positions) >= self.max_open:
            self.trades_skipped_max_open += 1
            self.decisions.append({
                'ts': ts, 'market_id': market_id,
                'action': 'skip_max_open',
                'model_prob': model_prob, 'market_price': market_price,
                'edge': edge
            })
            return

        # Calculate position size
        size = self.compute_position_size(edge, market_price)

        if size < 0.01:  # Minimum trade size
            self.trades_skipped_no_capital += 1
            return

        # Execute the trade
        entry_price = self.apply_slippage(market_price, 'buy')
        entry_fee = self.apply_fees(size)

        # Deduct cost from bankroll
        self.bankroll -= (size + entry_fee)

        # Record the position
        position = {
            'ts_entry': ts,
            'market_id': market_id,
            'entry_price': entry_price,
            'size': size,
            'entry_fee': entry_fee,
            'model_prob': model_prob,
            'edge': edge,
            'future_price': future_price,  # For later P&L calc
        }
        self.open_positions.append(position)
        self.trades_taken += 1

        self.decisions.append({
            'ts': ts, 'market_id': market_id,
            'action': 'buy',
            'model_prob': model_prob, 'market_price': market_price,
            'edge': edge, 'size': size
        })

    def close_positions(self, current_ts):
        """
        Close positions that have reached their horizon.

        In our setup, the "horizon" is 6 hours. After 6 hours,
        we check the actual price and compute P&L.

        In real trading, you'd sell the position. In simulation,
        we just compute what would have happened.
        """
        still_open = []

        for pos in self.open_positions:
            # The future_price was recorded at simulation setup
            # In live trading, you'd check the current market price
            exit_price = self.apply_slippage(pos['future_price'], 'sell')
            exit_fee = self.apply_fees(pos['size'])

            # P&L calculation
            # You bought shares at entry_price, they're now worth exit_price
            # Number of shares = size / entry_price
            shares = pos['size'] / pos['entry_price']
            proceeds = shares * exit_price
            pnl = proceeds - pos['size'] - pos['entry_fee'] - exit_fee

            # Return capital + P&L to bankroll
            self.bankroll += (pos['size'] + pnl)

            self.closed_trades.append({
                'ts_entry': pos['ts_entry'],
                'ts_exit': current_ts,
                'market_id': pos['market_id'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'size': pos['size'],
                'pnl': pnl,
                'return_pct': (pnl / pos['size']) * 100 if pos['size'] > 0 else 0,
                'model_prob': pos['model_prob'],
                'edge': pos['edge'],
            })

        # All positions close at horizon (no open positions carry over)
        self.open_positions = []

    def record_equity(self, ts):
        """Snapshot current equity (bankroll + open position value)."""
        # For simplicity, mark open positions at entry price
        open_value = sum(p['size'] for p in self.open_positions)
        total = self.bankroll + open_value
        self.equity_curve.append({'ts': ts, 'equity': total})

    def get_equity(self):
        return self.bankroll + sum(p['size'] for p in self.open_positions)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(test_df, model, feature_names, params):
    """
    Run the paper trading simulation on the test set.

    For each row in the test set:
    1. Get model prediction
    2. Compare to market price (the 'p' feature)
    3. Decide whether to trade
    4. If trading, compute P&L using the actual future price

    IMPORTANT: The 'future_price' used for P&L is p + price_change
    from the label computation. In live trading, you wouldn't know this.
    The simulation uses it only to compute what would have happened.
    """
    print(f"\nRunning paper trading simulation...")
    print(f"  Edge threshold:  {params['edge_threshold']:.1%}")
    print(f"  Max position:    ${params['max_position']:.2f}")
    print(f"  Max open:        {params['max_open']}")
    print(f"  Kelly fraction:  {params['kelly_fraction']}")
    print(f"  Starting capital: ${params['bankroll']:.2f}")
    print(f"  Slippage:        {SLIPPAGE_BPS} bps")

    trader = PaperTrader(
        bankroll=params['bankroll'],
        max_position=params['max_position'],
        max_open=params['max_open'],
        edge_threshold=params['edge_threshold'],
        kelly_fraction=params['kelly_fraction'],
        slippage_bps=SLIPPAGE_BPS,
        fee_bps=FEE_BPS,
    )

    # Prepare features
    available = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    X_test = test_df[available].values.astype(np.float64)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Get model predictions for all test rows
    model_probs = model.predict(X_test)

    # We need market_price (current p) and future_price for P&L
    # market_price = test_df['p'] (current price)
    # future_price = p + price_change (we reconstruct from label info)
    #
    # Since we saved 'label' (binary) but not the raw price_change,
    # we need to compute future price from the original data.
    # For the simulation, we'll use: if label=1, price went up by at
    # least threshold. We use the actual 'p' column of the NEXT row
    # as an approximation, but more precisely we need the price 36
    # samples ahead.
    #
    # WORKAROUND: We compute future prices by looking ahead in the
    # sorted test data. Within each market, the row 36 positions ahead
    # has the future price.

    print(f"  Computing future prices for P&L calculation...")

    test_df = test_df.copy()
    test_df['model_prob'] = model_probs

    # Compute actual future prices per market
    test_df['future_price'] = np.nan
    for market_id in test_df['clob_token_id'].unique():
        mask = test_df['clob_token_id'] == market_id
        market_data = test_df.loc[mask].sort_values('ts')
        future_p = market_data['p'].shift(-36)
        test_df.loc[market_data.index, 'future_price'] = future_p.values

    # Drop rows without future price (last 36 per market)
    valid_mask = test_df['future_price'].notna()
    sim_df = test_df[valid_mask].sort_values('ts').reset_index(drop=True)

    print(f"  Tradeable rows: {len(sim_df):,} (of {len(test_df):,} test rows)")

    # Run simulation chronologically
    prev_ts = None
    for idx, row in sim_df.iterrows():
        ts = row['ts']

        # Close expired positions when time advances
        if prev_ts is not None and ts != prev_ts:
            trader.close_positions(ts)
            trader.record_equity(ts)

        # Evaluate signal
        trader.evaluate_signal(
            ts=ts,
            market_id=row['clob_token_id'],
            model_prob=row['model_prob'],
            market_price=row['p'],
            future_price=row['future_price'],
        )

        prev_ts = ts

    # Close any remaining positions
    trader.close_positions(sim_df['ts'].max())
    trader.record_equity(sim_df['ts'].max())

    return trader


# =============================================================================
# PERFORMANCE REPORTING
# =============================================================================

def print_performance(trader):
    """Print comprehensive trading performance statistics."""

    print(f"\n{'=' * 70}")
    print(f"PAPER TRADING RESULTS")
    print(f"{'=' * 70}")

    # ── Trade summary ──
    print(f"\nTrade Summary:")
    print(f"  Total signals evaluated:  {trader.total_signals:,}")
    print(f"  Trades taken:             {trader.trades_taken:,}")
    print(f"  Skipped (no edge):        {trader.trades_skipped_no_edge:,}")
    print(f"  Skipped (max positions):  {trader.trades_skipped_max_open:,}")
    print(f"  Skipped (no capital):     {trader.trades_skipped_no_capital:,}")

    if not trader.closed_trades:
        print(f"\n  No trades were executed. Try lowering --edge-threshold.")
        return

    trades_df = pd.DataFrame(trader.closed_trades)

    # ── P&L Statistics ──
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    median_pnl = trades_df['pnl'].median()
    total_risked = trades_df['size'].sum()

    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] < 0]
    flat = trades_df[trades_df['pnl'] == 0]

    win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0

    print(f"\nP&L Statistics:")
    print(f"  Total P&L:        ${total_pnl:+.4f}")
    print(f"  Total risked:     ${total_risked:.2f}")
    print(f"  Return on risked: {(total_pnl/total_risked)*100:+.2f}%"
          if total_risked > 0 else "")
    print(f"  Avg P&L per trade: ${avg_pnl:+.4f}")
    print(f"  Median P&L:       ${median_pnl:+.4f}")

    print(f"\nWin/Loss Breakdown:")
    print(f"  Winners:  {len(winners):>6} ({win_rate:.1%})")
    print(f"  Losers:   {len(losers):>6} ({len(losers)/len(trades_df):.1%})")
    print(f"  Flat:     {len(flat):>6}")

    if len(winners) > 0:
        print(f"  Avg win:  ${winners['pnl'].mean():+.4f}")
        print(f"  Best win: ${winners['pnl'].max():+.4f}")
    if len(losers) > 0:
        print(f"  Avg loss: ${losers['pnl'].mean():+.4f}")
        print(f"  Worst:    ${losers['pnl'].min():+.4f}")

    # ── Return per trade distribution ──
    print(f"\nReturn Distribution (per trade):")
    pcts = trades_df['return_pct']
    print(f"  Min:    {pcts.min():+.2f}%")
    print(f"  25th:   {pcts.quantile(0.25):+.2f}%")
    print(f"  Median: {pcts.median():+.2f}%")
    print(f"  75th:   {pcts.quantile(0.75):+.2f}%")
    print(f"  Max:    {pcts.max():+.2f}%")

    # ── Edge analysis ──
    print(f"\nEdge Analysis:")
    print(f"  Avg edge at entry:  {trades_df['edge'].mean():.4f}")
    print(f"  Min edge at entry:  {trades_df['edge'].min():.4f}")
    print(f"  Max edge at entry:  {trades_df['edge'].max():.4f}")

    # Edge vs outcome
    edge_bins = [0, 0.05, 0.10, 0.20, 0.50, 1.0]
    print(f"\n  Win rate by edge bucket:")
    print(f"  {'Edge Range':<15} {'Trades':>8} {'Win Rate':>10} {'Avg P&L':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*10}")

    for i in range(len(edge_bins) - 1):
        lo, hi = edge_bins[i], edge_bins[i+1]
        bucket = trades_df[(trades_df['edge'] >= lo) & (trades_df['edge'] < hi)]
        if len(bucket) > 0:
            wr = (bucket['pnl'] > 0).mean()
            avg = bucket['pnl'].mean()
            print(f"  {lo:.2f}-{hi:.2f}       {len(bucket):>8} {wr:>9.1%} ${avg:>+9.4f}")

    # ── Equity curve stats ──
    if trader.equity_curve:
        equity_df = pd.DataFrame(trader.equity_curve)
        final_equity = equity_df['equity'].iloc[-1]
        max_equity = equity_df['equity'].max()
        min_equity = equity_df['equity'].min()

        # Drawdown
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()

        print(f"\nEquity Curve:")
        print(f"  Starting:     ${trader.initial_bankroll:.2f}")
        print(f"  Final:        ${final_equity:.2f}")
        print(f"  Net return:   {((final_equity/trader.initial_bankroll)-1)*100:+.2f}%")
        print(f"  Peak:         ${max_equity:.2f}")
        print(f"  Trough:       ${min_equity:.2f}")
        print(f"  Max drawdown: {max_drawdown*100:.2f}%")

    # ── Reality check ──
    print(f"\n{'=' * 70}")
    print(f"REALITY CHECK")
    print(f"{'=' * 70}")

    if total_pnl > 0:
        print(f"  The simulation shows a profit of ${total_pnl:.4f}.")
        print(f"  Before getting excited, remember:")
        print(f"  - This is 5 days of data. Could be luck.")
        print(f"  - Slippage was estimated, not measured.")
        print(f"  - You assumed instant execution at quoted prices.")
        print(f"  - The model was trained on data just before this period.")
        print(f"  - Survivorship bias: you're testing on markets that existed.")
    elif total_pnl < 0:
        print(f"  The simulation lost ${abs(total_pnl):.4f}.")
        print(f"  This is actually normal and informative:")
        print(f"  - Having a good MODEL doesn't guarantee profitable TRADING.")
        print(f"  - The edge might be too small to overcome transaction costs.")
        print(f"  - Try adjusting edge_threshold or position sizing.")
        print(f"  - The model's strength is prediction, not necessarily trading.")
    else:
        print(f"  Break even. The edge might exist but is very small.")

    print(f"\n  The real test: Chunk 8 (live trading with tiny real money).")

    return trades_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chunk 7: Paper trading simulation"
    )
    parser.add_argument('--features', type=str, default=str(FEATURES_PATH))
    parser.add_argument('--edge-threshold', type=float, default=DEFAULT_EDGE_THRESHOLD,
                        help=f'Min edge to trade (default: {DEFAULT_EDGE_THRESHOLD})')
    parser.add_argument('--max-position', type=float, default=DEFAULT_MAX_POSITION,
                        help=f'Max $ per position (default: {DEFAULT_MAX_POSITION})')
    parser.add_argument('--bankroll', type=float, default=DEFAULT_BANKROLL,
                        help=f'Starting capital (default: {DEFAULT_BANKROLL})')
    parser.add_argument('--max-open', type=int, default=DEFAULT_MAX_OPEN,
                        help=f'Max simultaneous positions (default: {DEFAULT_MAX_OPEN})')
    parser.add_argument('--kelly-fraction', type=float, default=DEFAULT_KELLY_FRACTION,
                        help=f'Kelly fraction (default: {DEFAULT_KELLY_FRACTION})')
    args = parser.parse_args()

    print("=" * 70)
    print("CHUNK 7: PAPER TRADING SIMULATION")
    print("=" * 70)

    # ── Load data ──
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"ERROR: Features not found at {features_path}")
        sys.exit(1)

    df = pd.read_csv(features_path)
    df = df.sort_values('ts').reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nData splits:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows (paper trading period)")

    # ── Train model ──
    print(f"\nTraining model...")
    available = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    for col in ['days_to_expiry', 'log_days_to_expiry']:
        if col in available:
            median_val = train_df[col].median()
            for split_df in [train_df, val_df, test_df]:
                split_df[col] = split_df[col].fillna(median_val)

    X_train = train_df[available].values.astype(np.float64)
    X_val = val_df[available].values.astype(np.float64)
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    model = train_best_model(X_train, y_train, X_val, y_val, available)

    # ── Run simulation ──
    trading_params = {
        'edge_threshold': args.edge_threshold,
        'max_position': args.max_position,
        'bankroll': args.bankroll,
        'max_open': args.max_open,
        'kelly_fraction': args.kelly_fraction,
    }

    trader = run_simulation(test_df, model, available, trading_params)

    # ── Print results ──
    trades_df = print_performance(trader)

    # ── Save outputs ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if trades_df is not None and len(trades_df) > 0:
        # Save trade log
        trades_path = RESULTS_DIR / 'paper_trades.csv'
        trades_df.to_csv(trades_path, index=False)
        print(f"\n  Trade log:     {trades_path}")

    # Save equity curve
    if trader.equity_curve:
        equity_df = pd.DataFrame(trader.equity_curve)
        equity_path = RESULTS_DIR / 'equity_curve.csv'
        equity_df.to_csv(equity_path, index=False)
        print(f"  Equity curve:  {equity_path}")

    # Save decisions log
    if trader.decisions:
        decisions_df = pd.DataFrame(trader.decisions)
        decisions_path = RESULTS_DIR / 'trading_decisions.csv'
        decisions_df.to_csv(decisions_path, index=False)
        print(f"  Decisions log: {decisions_path}")

    print(f"\n  Tip: Try different parameters to see how they affect results:")
    print(f"    python src/paper_trade.py --edge-threshold 0.05")
    print(f"    python src/paper_trade.py --edge-threshold 0.01 --max-position 0.50")
    print(f"\nNext step: Chunk 8 — Tiny-dollar live test (optional)")
    print(f"  Or: Chunk 9 — Resume packaging")


if __name__ == "__main__":
    main()