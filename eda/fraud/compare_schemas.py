"""ULB against IEEE-CIS: which columns line up, and what a ULB model keeps with only those.

    uv run python -I eda/fraud/compare_schemas.py eda/output/data/creditcard.csv data/external

data/external must hold ieee-fraud-detection/train_transaction.csv (not in the repo: Kaggle rules, 7B).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

ulb = pd.read_csv(sys.argv[1])
tx = pd.read_csv(sys.argv[2] + "/ieee-fraud-detection/train_transaction.csv")
idt = pd.read_csv(sys.argv[2] + "/ieee-fraud-detection/train_identity.csv", nrows=5)

print("ULB columns:", len(ulb.columns), "->", ["Time", "V1..V28", "Amount", "Class"])
print("IEEE-CIS transaction columns:", len(tx.columns), "| identity columns:", len(idt.columns) - 1)
fam = {}
for c in tx.columns:
    k = "V" if c.startswith("V") and c[1:].isdigit() else ("C" if c[0] == "C" and c[1:].isdigit() else ("D" if c[0] == "D" and c[1:].isdigit() else ("M" if c[0] == "M" and c[1:].isdigit() else ("card" if c.startswith("card") else ("addr/dist" if c.startswith(("addr", "dist")) else c)))))
    fam.setdefault(k, 0)
    fam[k] += 1
print("IEEE-CIS column families:", fam)
shared_names = sorted(set(ulb.columns) & set(tx.columns) - {"Class"})
print("column names the two files share:", shared_names[:6], "... total", len(shared_names), "(all of them are the V1..V28 names)")

print("\nSame name, different thing: V1")
v = ulb["V1"]
print(f"  ULB V1: mean {v.mean():.2f}, std {v.std():.2f}, min {v.min():.1f}, max {v.max():.1f}, missing {100*v.isna().mean():.0f}%, distinct values {v.nunique():,}")
w = tx["V1"]
print(f"  IEEE V1: mean {w.mean():.2f}, std {w.std():.2f}, min {w.min():.1f}, max {w.max():.1f}, missing {100*w.isna().mean():.0f}%, distinct values {w.nunique()}")
for c in ("V2", "V3"):
    print(f"  IEEE {c}: missing {100*tx[c].isna().mean():.0f}%, distinct {tx[c].nunique()}, max {tx[c].max():.0f}")
print("  IEEE V columns that are 0/1/integer-valued only:", int(sum(((tx[c].dropna() == tx[c].dropna().round()).all()) for c in tx.columns if c[0] == "V" and c[1:].isdigit())), "of 339")

print("\nWhat the two files have in common:")
print(f"  Amount: ULB median {ulb.Amount.median():.2f}, mean {ulb.Amount.mean():.1f} | IEEE TransactionAmt median {tx.TransactionAmt.median():.2f}, mean {tx.TransactionAmt.mean():.1f}")
print(f"  Time: ULB spans {ulb.Time.max()/3600:.0f} hours | IEEE TransactionDT spans {(tx.TransactionDT.max()-tx.TransactionDT.min())/86400:.0f} days")

# ULB detector using only the columns the two datasets share (amount and time)
V = [f"V{i}" for i in range(1, 29)]
d = ulb[~ulb.duplicated(["Time"] + V + ["Amount", "Class"])].reset_index(drop=True)
y = d.Class.to_numpy()
tr, hold = train_test_split(np.arange(len(d)), test_size=0.4, stratify=y, random_state=42)
va, _ = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=42)
print("\nULB validation PR-AUC (one untuned fit, 3 seeds; chance = 0.0017):")
for name, cols in (("all 30 columns", V + ["Amount", "Time"]), ("only amount + time (the shared columns)", ["Amount", "Time"]), ("only amount", ["Amount"])):
    s = []
    for seed in (0, 1, 2):
        m = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(d.loc[tr, cols], y[tr])
        s.append(average_precision_score(y[va], m.predict_proba(d.loc[va, cols])[:, 1]))
    print(f"  {name}: {np.mean(s):.3f} (±{np.std(s):.3f})")
