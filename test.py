import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("/mnt/ml-data/projects/polymarket-predictor/data/polymarket.db")
snapshots = pd.read_sql("SELECT * FROM snapshots", conn)

# Grid search: threshold x horizon
thresholds = [0.001, 0.002, 0.003, 0.005]
horizons = [12, 24, 36]  # 2hr, 4hr, 6hr at 10-min intervals

print(f"{'Threshold':<12} {'Horizon':<12} {'Base Rate':<12}")
print("-" * 36)

for horizon in horizons:
    for thresh in thresholds:
        labels = []
        for token_id in snapshots['clob_token_id'].unique():
            df = snapshots[snapshots['clob_token_id'] == token_id].sort_values('ts')
            future = df['price'].shift(-horizon)
            change = future - df['price']
            labels.extend(change.dropna().tolist())
        
        labels = np.array(labels)
        rate = (labels >= thresh).mean()
        hrs = horizon * 10 / 60
        print(f"{thresh:<12} {hrs:.1f}hr ({horizon}){'':<4} {rate:.1%}")
    print()