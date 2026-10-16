"""ULB: rows that repeat the same V1 to V28 and Amount at a different Time, versus exact copies.

    uv run python -I eda/fraud/time_twins.py eda/output/data/creditcard.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(sys.argv[1])
V = [f"V{i}" for i in range(1, 29)]
feat_notime = V + ["Amount"]
all_cols = ["Time"] + V + ["Amount", "Class"]
n = len(df)
print("rows", n, "fraud", int(df.Class.sum()))
print("Time is integer-valued:", bool((df.Time == df.Time.round()).all()), "| span hours:", round(df.Time.max() / 3600, 1))

# 1. exact copies (every column, Time included)
exact = df.duplicated(all_cols)
print("\n1) exact copies incl. Time:", int(exact.sum()), "| fraud:", int((exact & (df.Class == 1)).sum()))

# 2. same V1..V28 + Amount, ignoring Time (and Class)
twin = df.duplicated(feat_notime)
print("2) extra copies ignoring Time:", int(twin.sum()), "| of which also same Time:", int((twin & exact).sum()), "| different Time:", int((twin & ~exact).sum()))

# group structure for the different-time twins
key = df.groupby(feat_notime, sort=False).ngroup()
df["g"] = key
sz = df.groupby("g")["Time"].transform("size")
tw = df[sz > 1].copy()
print("   groups with copies:", tw.g.nunique(), "rows in them:", len(tw))
# label conflicts among twins (ignoring Time)
conf = tw.groupby("g")["Class"].nunique()
print("   groups with BOTH labels (ignoring Time):", int((conf > 1).sum()))
# fraud among twins
fr = tw[tw.Class == 1]
print("   fraud rows in twin groups:", len(fr), "| distinct groups:", fr.g.nunique())

# time gaps between consecutive members of a twin group, excluding same-second copies
tw = tw.sort_values(["g", "Time"])
tw["gap"] = tw.groupby("g")["Time"].diff()
gaps = tw["gap"].dropna()
gaps = gaps[gaps > 0]
print("\n3) gaps (seconds) between consecutive different-time twins: n =", len(gaps))
for name, lo, hi in [("<= 10 s", 0, 10), ("10 s to 1 min", 10, 60), ("1 min to 1 h", 60, 3600), ("1 h to 12 h", 3600, 43200), (">= 12 h", 43200, 1e9)]:
    k = int(((gaps > lo) & (gaps <= hi)).sum()) if lo > 0 else int((gaps <= hi).sum())
    print(f"   {name:>14}: {k:5d} ({100 * k / len(gaps):.1f}%)")
print("   median gap (s):", float(gaps.median()))

# amounts among twins
top = tw[~tw.duplicated(["g", "Time"])]["Amount"].value_counts().head(6)
print("\n4) most common Amount among twin rows:", dict(top))
print("   share of twin rows with Amount == 0:", round(100 * (tw.Amount == 0).mean(), 1), "%; Amount <= 2:", round(100 * (tw.Amount <= 2).mean(), 1), "%")
print("   distinct values of V1 within twin rows vs rows:", tw.V1.nunique(), "/", len(tw))

# 5. recommended policy: drop exact copies, then stratified 60/20/20 (orchestrator recipe)
d = df[~exact].reset_index(drop=True)
y = d.Class.to_numpy()
idx = np.arange(len(d))
tr, hold = train_test_split(idx, test_size=0.4, stratify=y, random_state=42)
va, te = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=42)
kk = d.groupby(feat_notime, sort=False).ngroup().to_numpy()
tr_keys = set(kk[tr])
print("\n5) after dedupe-first stratified split, rows with a Time-different twin in TRAIN:")
for nm, part in (("val", va), ("test", te)):
    has = np.array([k in tr_keys for k in kk[part]])
    fraud = int((has & (y[part] == 1)).sum())
    print(f"   {nm}: {int(has.sum())} of {len(part)} rows ({100 * has.mean():.1f}%), fraud rows: {fraud} of {int(y[part].sum())}")

# 6. time split (first 60% / next 20% / last 20% by Time) after dedupe
order = np.argsort(d.Time.to_numpy(), kind="stable")
a, b = int(len(d) * 0.6), int(len(d) * 0.8)
ttr, tva, tte = order[:a], order[a:b], order[b:]
tk = set(kk[ttr])
print("6) time split: rows in val/test with a twin in train (repeats of earlier behaviour):")
for nm, part in (("val", tva), ("test", tte)):
    has = np.array([k in tk for k in kk[part]])
    print(f"   {nm}: {int(has.sum())} of {len(part)} ({100 * has.mean():.1f}%), fraud rows: {int((has & (y[part] == 1)).sum())} of {int(y[part].sum())}")
