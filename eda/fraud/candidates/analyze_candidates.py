"""Measured facts about the three candidate fraud datasets, for the 16 Oct slides.

    uv run python -I eda/fraud/candidates/analyze_candidates.py insurance|ecom|ieee data/external

Expects fraud_oracle.csv, archive/Fraud_Data.csv and ieee-fraud-detection/train_*.csv under the
folder given. The data is not in the repo: IEEE-CIS forbids redistribution (Kaggle rules, 7B).

Prints a plain-text report. Reads only; writes nothing next to the data.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

SEED = 42
OUT = {}


def p(*a):
    print(*a, flush=True)


def split_idx(y, seed=SEED):
    """Same recipe as orchestrator/split.py: hold out 40%, then halve it, stratified."""
    idx = np.arange(len(y))
    tr, hold = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=seed)
    return tr, va, te


def time_idx(order):
    """First 60% / next 20% / last 20% by the given ordering (array of row positions)."""
    n = len(order)
    a, b = int(n * 0.6), int(n * 0.8)
    return order[:a], order[a:b], order[b:]


def fill_table(pos, neg, levels=(0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)):
    gap = neg - pos
    return {lv: (round(lv * gap), round(lv * gap / pos, 1)) for lv in levels}


def parts_report(name, y, tr, va, te):
    rows = []
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        rows.append((nm, len(ix), int(y[ix].sum()), 100 * y[ix].mean()))
    p(f"  {name}: " + " | ".join(f"{n}: {r} rows, {f} fraud ({pct:.2f}%)" for n, r, f, pct in rows))
    return {n: {"rows": r, "fraud": f, "rate_pct": round(pct, 3)} for n, r, f, pct in rows}


def hgb_score(X, y, tr, va, cat_cols, label):
    """Gradient boosting on ordinal-encoded categoricals. NaN allowed. One fit, no tuning."""
    Xc = X.copy()
    cats = [c for c in cat_cols if c in Xc.columns]
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan)
        Xc[cats] = enc.fit(Xc.iloc[tr][cats].astype(str).where(Xc.iloc[tr][cats].notna())).transform(
            Xc[cats].astype(str).where(Xc[cats].notna())
        )
    mask = [c in cats for c in Xc.columns]
    m = HistGradientBoostingClassifier(
        categorical_features=mask if any(mask) else None, random_state=SEED, max_iter=200
    )
    t = time.time()
    m.fit(Xc.iloc[tr], y[tr])
    pr = m.predict_proba(Xc.iloc[va])[:, 1]
    res = {
        "pr_auc": round(float(average_precision_score(y[va], pr)), 4),
        "roc_auc": round(float(roc_auc_score(y[va], pr)), 4),
        "prevalence_val": round(float(y[va].mean()), 4),
        "seconds": round(time.time() - t, 1),
    }
    p(f"  {label}: val PR-AUC {res['pr_auc']} (chance = {res['prevalence_val']}), ROC-AUC {res['roc_auc']}, {res['seconds']}s")
    return res


def logit_score(X, y, tr, va, cat_cols, label):
    cats = [c for c in cat_cols if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    ct = ColumnTransformer(
        [
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
        ]
    )
    m = make_pipeline(ct, LogisticRegression(max_iter=2000, C=1.0))
    m.fit(X.iloc[tr], y[tr])
    pr = m.predict_proba(X.iloc[va])[:, 1]
    res = {
        "pr_auc": round(float(average_precision_score(y[va], pr)), 4),
        "roc_auc": round(float(roc_auc_score(y[va], pr)), 4),
    }
    p(f"  {label}: val PR-AUC {res['pr_auc']}, ROC-AUC {res['roc_auc']}")
    return res


def cross_part_leak(df, tr, va, te, cols):
    """Rows of val/test that have an exact copy (on cols) in train."""
    key = pd.util.hash_pandas_object(df[cols], index=False)
    ks = set(key.iloc[tr])
    v = int(key.iloc[va].isin(ks).sum())
    t = int(key.iloc[te].isin(ks).sum())
    return v, t


# ----------------------------------------------------------------------------------------------
def insurance(root):
    f = root / "fraud_oracle.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    p("== Vehicle insurance claims (fraud_oracle.csv)")
    y = df["FraudFound_P"].to_numpy()
    n, pos = len(df), int(y.sum())
    p(f"  rows {n}, columns {df.shape[1]}, fraud {pos} ({100*pos/n:.2f}%)")
    OUT["rows"], OUT["fraud"], OUT["cols"] = n, pos, df.shape[1]

    # id-like columns and time coverage
    pn = df["PolicyNumber"]
    p(f"  PolicyNumber: min {pn.min()}, max {pn.max()}, unique {pn.nunique()}, equals row number: {bool((pn.to_numpy()==np.arange(1,n+1)).all())}")
    p(f"  Year values: {df['Year'].value_counts().sort_index().to_dict()}")
    p(f"  RepNumber unique: {df['RepNumber'].nunique()}")

    # missing / placeholder values
    p(f"  true NaN cells: {int(df.isna().sum().sum())}")
    ph = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            for v in ("0", "none", "None", "unknown", "?"):
                k = int((df[c].astype(str) == v).sum())
                if k:
                    ph.setdefault(c, {})[v] = k
    p(f"  string placeholders by column: {ph}")
    p(f"  Age == 0: {int((df['Age']==0).sum())}; AgeOfPolicyHolder values: {df['AgeOfPolicyHolder'].value_counts().to_dict()}")
    OUT["age_zero"] = int((df["Age"] == 0).sum())

    # duplicates
    d_all = int(df.duplicated().sum())
    d_noid = int(df.drop(columns=["PolicyNumber"]).duplicated().sum())
    p(f"  exact duplicate rows (all columns): {d_all}; ignoring PolicyNumber: {d_noid}")

    # cardinalities
    p("  columns (type, distinct): " + ", ".join(f"{c}[{'cat' if not pd.api.types.is_numeric_dtype(df[c]) else 'num'},{df[c].nunique()}]" for c in df.columns))

    # splits
    tr, va, te = split_idx(y)
    OUT["stratified"] = parts_report("stratified 60/20/20", y, tr, va, te)
    feat = [c for c in df.columns if c not in ("FraudFound_P", "PolicyNumber")]
    lv, lt = cross_part_leak(df.drop(columns=["PolicyNumber"]), tr, va, te, feat)
    p(f"  val/test rows with an exact copy in train (ignoring PolicyNumber): {lv} / {lt}")
    neg_tr, pos_tr = int((y[tr] == 0).sum()), int(y[tr].sum())
    OUT["fill"] = fill_table(pos_tr, neg_tr)
    p(f"  train positives {pos_tr}, negatives {neg_tr}, class gap {neg_tr-pos_tr}; fill rows (times positives): {OUT['fill']}")

    # signal
    X = df[feat]
    cats = [c for c in feat if not pd.api.types.is_numeric_dtype(df[c])]
    OUT["hgb"] = hgb_score(X, y, tr, va, cats, "gradient boosting")
    OUT["logit"] = logit_score(X, y, tr, va, cats, "logistic regression")

    # how much of the signal sits in a few columns
    p("  fraud rate by column, largest gaps:")
    gaps = []
    for c in feat:
        if df[c].nunique() <= 30:
            g = df.groupby(c)["FraudFound_P"].agg(["mean", "size"])
            g = g[g["size"] >= 100]
            if len(g) > 1:
                gaps.append((g["mean"].max() - g["mean"].min(), c, g["mean"].idxmax(), g["mean"].max(), g["mean"].idxmin(), g["mean"].min()))
    for gap, c, hi, hv, lo, lv_ in sorted(gaps, reverse=True)[:6]:
        p(f"    {c}: {hi} {100*hv:.1f}% vs {lo} {100*lv_:.1f}%")

    # uniformity hints
    for c in ("Month", "DayOfWeek", "WeekOfMonth"):
        vc = df[c].value_counts(normalize=True)
        p(f"  {c} share range: {100*vc.min():.1f}% to {100*vc.max():.1f}%")


# ----------------------------------------------------------------------------------------------
def ecom(root):
    df = pd.read_csv(root / "archive" / "Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
    p("== Fraud ecommerce (Fraud_Data.csv)")
    y = df["class"].to_numpy()
    n, pos = len(df), int(y.sum())
    p(f"  rows {n}, columns {df.shape[1]}, fraud {pos} ({100*pos/n:.2f}%)")
    OUT["rows"], OUT["fraud"] = n, pos
    p(f"  user_id unique: {df['user_id'].nunique()}; device_id unique: {df['device_id'].nunique()}; ip unique: {df['ip_address'].nunique()}")
    p(f"  true NaN cells: {int(df.isna().sum().sum())}")
    p(f"  signup {df['signup_time'].min()} to {df['signup_time'].max()}; purchase {df['purchase_time'].min()} to {df['purchase_time'].max()}")
    d = int(df.drop(columns=["user_id"]).duplicated().sum())
    p(f"  exact duplicate rows ignoring user_id: {d}")
    df["secs_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    quick = df["secs_to_purchase"] <= 60
    p(f"  purchase within 60 s of signup: {int(quick.sum())} rows, fraud rate {100*df.loc[quick,'class'].mean():.1f}%; others {100*df.loc[~quick,'class'].mean():.2f}%")
    p(f"  purchase_value: min {df['purchase_value'].min()}, max {df['purchase_value'].max()}, mean {df['purchase_value'].mean():.1f}")
    # devices shared by several users
    dev = df.groupby("device_id")["class"].agg(["size", "mean"])
    shared = dev[dev["size"] > 1]
    p(f"  devices used by more than one user: {len(shared)} devices covering {int(shared['size'].sum())} rows; fraud rate on those rows {100*(shared['size']*shared['mean']).sum()/shared['size'].sum():.1f}%")
    tr, va, te = split_idx(y)
    OUT["stratified"] = parts_report("stratified 60/20/20", y, tr, va, te)
    order = np.argsort(df["purchase_time"].to_numpy(), kind="stable")
    t_tr, t_va, t_te = time_idx(order)
    OUT["time"] = parts_report("by purchase time", y, t_tr, t_va, t_te)
    neg_tr, pos_tr = int((y[tr] == 0).sum()), int(y[tr].sum())
    OUT["fill"] = fill_table(pos_tr, neg_tr)
    p(f"  train positives {pos_tr}, class gap {neg_tr-pos_tr}; fill rows (times positives): {OUT['fill']}")

    feat_base = ["purchase_value", "source", "browser", "sex", "age"]
    cats = ["source", "browser", "sex"]
    X = df[feat_base].copy()
    OUT["hgb_basic"] = hgb_score(X, y, tr, va, cats, "gradient boosting, basic columns only")
    X2 = X.assign(secs_to_purchase=df["secs_to_purchase"], hour=df["purchase_time"].dt.hour)
    OUT["hgb_time"] = hgb_score(X2, y, tr, va, cats, "gradient boosting, + secs_to_purchase, hour")
    cnt = df.groupby("device_id")["user_id"].transform("count")
    X3 = X2.assign(device_users=cnt)
    OUT["hgb_dev"] = hgb_score(X3, y, tr, va, cats, "gradient boosting, + device sharing")


# ----------------------------------------------------------------------------------------------
def ieee(root):
    r = root / "ieee-fraud-detection"
    t0 = time.time()
    tx = pd.read_csv(r / "train_transaction.csv")
    idt = pd.read_csv(r / "train_identity.csv")
    p("== IEEE-CIS (train_transaction.csv + train_identity.csv)")
    p(f"  loaded in {time.time()-t0:.0f}s; transaction {tx.shape}, identity {idt.shape}")
    y = tx["isFraud"].to_numpy()
    n, pos = len(tx), int(y.sum())
    p(f"  rows {n}, fraud {pos} ({100*pos/n:.2f}%)")
    OUT["rows"], OUT["fraud"] = n, pos
    has_id = tx["TransactionID"].isin(idt["TransactionID"])
    p(f"  rows with an identity record: {int(has_id.sum())} ({100*has_id.mean():.1f}%); fraud rate with identity {100*tx.loc[has_id,'isFraud'].mean():.2f}%, without {100*tx.loc[~has_id,'isFraud'].mean():.2f}%")
    OUT["identity_share_pct"] = round(100 * has_id.mean(), 1)
    dt = tx["TransactionDT"]
    p(f"  TransactionDT span: {dt.min()} to {dt.max()} s = {(dt.max()-dt.min())/86400:.0f} days; sorted by time already: {bool(dt.is_monotonic_increasing)}")
    miss = tx.isna().mean()
    p(f"  columns with any missing: {int((miss>0).sum())} of {tx.shape[1]}; median column missing {100*miss.median():.1f}%; overall missing cells {100*tx.isna().mean().mean():.1f}%")
    p(f"  columns >50% missing: {int((miss>0.5).sum())}")
    obj = [c for c in tx.columns if not pd.api.types.is_numeric_dtype(tx[c])]
    p(f"  text columns in transaction table: {obj}")
    dup = int(tx.drop(columns=["TransactionID"]).duplicated().sum())
    p(f"  exact duplicate rows ignoring TransactionID: {dup}")

    tr, va, te = split_idx(y)
    OUT["stratified"] = parts_report("stratified 60/20/20 (full)", y, tr, va, te)
    order = np.argsort(dt.to_numpy(), kind="stable")
    t_tr, t_va, t_te = time_idx(order)
    OUT["time"] = parts_report("by time (full)", y, t_tr, t_va, t_te)
    neg_tr, pos_tr = int((y[tr] == 0).sum()), int(y[tr].sum())
    OUT["fill_full"] = fill_table(pos_tr, neg_tr, (0.01, 0.02, 0.05, 0.10, 0.25, 1.0))
    p(f"  full train positives {pos_tr}, negatives {neg_tr}, gap {neg_tr-pos_tr}; fill rows (times positives): {OUT['fill_full']}")

    # subsample keeping the fraud rate, then split the subsample
    sub_n = 100_000
    sub, _ = train_test_split(np.arange(n), train_size=sub_n, stratify=y, random_state=SEED)
    sub = np.sort(sub)
    ys = y[sub]
    s_tr, s_va, s_te = split_idx(ys)
    OUT["sub100k"] = parts_report("100k stratified subsample, stratified split", ys, s_tr, s_va, s_te)
    neg, posn = int((ys[s_tr] == 0).sum()), int(ys[s_tr].sum())
    OUT["fill_sub"] = fill_table(posn, neg, (0.10, 0.25, 0.50, 1.0))
    p(f"  subsample train positives {posn}, gap {neg-posn}; fill rows (times positives): {OUT['fill_sub']}")

    # signal check on the subsample: transaction table only, and with identity joined
    sx = tx.iloc[sub].drop(columns=["isFraud", "TransactionID"]).reset_index(drop=True)
    cats = [c for c in sx.columns if not pd.api.types.is_numeric_dtype(sx[c])]
    OUT["hgb_sub_tx"] = hgb_score(sx, ys, s_tr, s_va, cats, "gradient boosting, 100k subsample, transaction table")
    # time split of the subsample for a realism check
    s_order = np.argsort(tx["TransactionDT"].to_numpy()[sub], kind="stable")
    a, b, c_ = time_idx(s_order)
    OUT["time_sub"] = parts_report("100k subsample, by time", ys, a, b, c_)
    OUT["hgb_sub_time"] = hgb_score(sx, ys, a, b, cats, "gradient boosting, 100k subsample, time split")


if __name__ == "__main__":
    which, root = sys.argv[1], Path(sys.argv[2])
    {"insurance": insurance, "ecom": ecom, "ieee": ieee}[which](root)
    p("JSON " + json.dumps(OUT, default=str))
