# Polymarket Price Movement Predictor

## The Game (Don't Change This Mid-Project)

**Domain:** Polymarket prediction markets  
**Prediction target:** Binary classification  
**Question:** Will `p_yes` increase by ≥ 0.001 (0.1 cents) in the next 6 hours?

```
y = 1  if  p(t + 6hr) - p(t) >= 0.001
y = 0  otherwise
```

**Why these parameters?**  
- Data is 10-minute intervals (not 5-min as originally expected)
- Markets are less volatile than typical — 0.02 threshold gave only 1.8% base rate
- 0.001 threshold at 6hr gives ~16% base rate — good for learning
- Note: 0.1 cents is too small for real profit; this is a learning project first

---

## Data Spec

**Sampling resolution:** 10 minutes (actual from API, not 5 as originally expected)  
**Horizon:** 6 hours (36 samples forward)  
**Minimum market criteria:**  
- `active = true`  
- `closed = false`  
- Volume > $10k (filters out dead markets)

**Data sources:**
| API | Endpoint | What you get |
|-----|----------|--------------|
| Gamma | `GET /markets` | Market metadata, clob_token_ids, volume, status |
| CLOB | `GET /prices-history` | Historical price timeseries |
| CLOB | `GET /midpoint` | Live midpoint price (for polling later) |

---

## Feature Windows

With 10-min sampling, these are your lookback windows for features:

| Window | Duration | Samples | Use for |
|--------|----------|---------|---------|
| instant | 0 | 1 | Current price p |
| short | 30 min | 3 | Fast momentum |
| medium | 2 hours | 12 | Recent trend |
| long | 6 hours | 36 | Baseline drift |

---

## Metrics (All Three, Every Time)

1. **Log Loss** (primary) — penalizes confident wrong predictions  
2. **Brier Score** — mean squared error of probabilities  
3. **Calibration** — binned reliability (when you say 70%, is it right 70% of the time?)

Calibration bins: `[0.0-0.1, 0.1-0.2, ..., 0.9-1.0]`

**Baseline targets to beat:**
- Naive (always predict 0.5): log_loss ≈ 0.693, brier ≈ 0.25
- "No change" (predict base rate ~0.16): log_loss ≈ 0.51, brier ≈ 0.13

---

## Database Schema (SQLite)

```sql
-- Raw snapshots (append-only, never delete)
CREATE TABLE snapshots (
    id INTEGER PRIMARY KEY,
    clob_token_id TEXT NOT NULL,
    ts INTEGER NOT NULL,           -- unix timestamp
    price REAL NOT NULL,           -- p_yes (0 to 1)
    source TEXT NOT NULL,          -- 'historical' or 'live'
    fetched_at INTEGER NOT NULL,   -- when we pulled this
    UNIQUE(clob_token_id, ts)
);

-- Market metadata (update periodically)
CREATE TABLE markets (
    clob_token_id TEXT PRIMARY KEY,
    condition_id TEXT,
    question TEXT,
    slug TEXT,
    volume REAL,
    active INTEGER,                -- 0 or 1
    closed INTEGER,                -- 0 or 1
    end_date TEXT,
    updated_at INTEGER
);

-- Indexes for fast feature computation
CREATE INDEX idx_snapshots_market_ts ON snapshots(clob_token_id, ts);
CREATE INDEX idx_markets_active ON markets(active, closed);
```

**Why this schema:**
- `snapshots` is append-only — you can always regenerate features/labels
- `source` column lets you distinguish backfill vs live data
- Compound unique index prevents duplicates if you re-run backfill
- Separate `markets` table so you don't repeat metadata in every row

---

## Project Structure

```
polymarket-predictor/
├── README.md              # This file (the spec)
├── data/
│   └── polymarket.db      # SQLite database
├── src/
│   ├── ingest.py          # Chunk 1: data capture
│   ├── features.py        # Chunk 3: feature engineering  
│   ├── train_eval.py      # Chunk 5: model training
│   └── utils.py           # Shared helpers
├── notebooks/
│   └── eda.ipynb          # Chunk 2: exploration
├── models/                # Saved model artifacts
├── results/               # Evaluation outputs
└── requirements.txt
```

---

## Evaluation Protocol

**Always walk-forward (time-split):**
```
|-------- train --------|--- val ---|--- test ---|
      past                              future
```

Never shuffle. Never let future leak into past.

**Minimum test set:** 500+ predictions (for calibration to be meaningful)

---

## Rules That Keep This Healthy

1. Don't add new data sources until Chunk 6  
2. If you can't beat naive baseline on log loss OOS, stop and debug — don't add complexity  
3. Keep snapshots append-only; regenerate features from raw data  
4. Log every model run with: git hash, hyperparams, train/val/test metrics  
5. No live trading until paper trading shows positive edge for 2+ weeks

---

## Next Step (Chunk 1)

Write `ingest.py` that:
1. Fetches active markets from Gamma API
2. Pulls historical prices for each from CLOB API  
3. Stores everything in SQLite per the schema above

Target: Run it once, get 1M+ price snapshots across 50+ markets.