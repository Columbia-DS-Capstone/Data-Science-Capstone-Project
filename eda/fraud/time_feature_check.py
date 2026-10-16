"""ULB: validation PR-AUC with no time feature, hour of day, or raw Time, on two splits.

One untuned gradient-boosting fit, three model seeds. Rough guide only.

    uv run python -I eda/fraud/time_feature_check.py eda/output/data/creditcard.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(sys.argv[1])
V = [f"V{i}" for i in range(1, 29)]
all_cols = ["Time"] + V + ["Amount", "Class"]
d = df[~df.duplicated(all_cols)].reset_index(drop=True)          # dedupe first (same row, same timestamp)
d["hour"] = (d.Time / 3600) % 24
y = d.Class.to_numpy()
idx = np.arange(len(d))
tr, hold = train_test_split(idx, test_size=0.4, stratify=y, random_state=42)
va, te = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=42)
order = np.argsort(d.Time.to_numpy(), kind="stable")
a, b = int(len(d) * 0.6), int(len(d) * 0.8)
splits = {"stratified": (tr, va), "by time": (order[:a], order[a:b])}
sets = {"no time feature": V + ["Amount"], "hour of day": V + ["Amount", "hour"], "raw Time": V + ["Amount", "Time"]}
print("validation PR-AUC (one untuned gradient-boosting fit; chance = fraud rate ~0.0017)")
for sname, (t, v) in splits.items():
    row = []
    for fname, cols in sets.items():
        scores = []
        for seed in (0, 1, 2):
            m = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
            m.fit(d.loc[t, cols], y[t])
            scores.append(average_precision_score(y[v], m.predict_proba(d.loc[v, cols])[:, 1]))
        row.append(f"{fname}: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
    print(f"  {sname:>10}  |  " + "  |  ".join(row), f"  [val fraud rows: {int(y[v].sum())}]")
