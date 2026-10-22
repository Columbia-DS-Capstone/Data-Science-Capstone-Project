"""IEEE-CIS: how many columns stay under a missing-value limit, and what a boosting fit scores
on a 100k subsample using only those columns.

    uv run python -I eda/fraud/candidates/ieee_reduced_features.py data/external
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

root = Path(sys.argv[1])
tx = pd.read_csv(root / "ieee-fraud-detection" / "train_transaction.csv")
y = tx["isFraud"].to_numpy()
n = len(tx)
miss = tx.isna().mean()
feat = [c for c in tx.columns if c not in ("TransactionID", "isFraud")]
print("columns by missing share (features only, TransactionDT included):")
for t in (0.0, 0.05, 0.2, 0.5):
    print(f"  <= {int(t*100)}% missing: {int((miss[feat] <= t).sum())} of {len(feat)}")

def run(cols, label):
    sub, _ = train_test_split(np.arange(n), train_size=100_000, stratify=y, random_state=42)
    sub = np.sort(sub); ys = y[sub]
    idx = np.arange(len(sub))
    tr, hold = train_test_split(idx, test_size=0.4, stratify=ys, random_state=42)
    va, te = train_test_split(hold, test_size=0.5, stratify=ys[hold], random_state=42)
    X = tx.iloc[sub][cols].reset_index(drop=True)
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan)
        Xt = X.iloc[tr][cats].astype(str).where(X.iloc[tr][cats].notna())
        enc.fit(Xt)
        X[cats] = enc.transform(X[cats].astype(str).where(X[cats].notna()))
    mask = [c in cats for c in cols]
    m = HistGradientBoostingClassifier(categorical_features=mask if any(mask) else None, random_state=42, max_iter=200)
    m.fit(X.iloc[tr], ys[tr])
    pr = m.predict_proba(X.iloc[va])[:, 1]
    print(f"  {label}: {len(cols)} cols ({len(cats)} text), val PR-AUC {average_precision_score(ys[va], pr):.4f}, ROC-AUC {roc_auc_score(ys[va], pr):.4f}")

for t in (0.05, 0.2):
    cols = [c for c in feat if miss[c] <= t and c != "TransactionDT"]
    run(cols, f"features with <= {int(t*100)}% missing (TransactionDT dropped)")
cols = [c for c in feat if miss[c] <= 0.2 and c != "TransactionDT"]
print("text columns kept at <=20%:", [c for c in cols if not pd.api.types.is_numeric_dtype(tx[c])])
card = {c: int(tx[c].nunique()) for c in cols if not pd.api.types.is_numeric_dtype(tx[c])}
print("distinct values of text columns:", card)
