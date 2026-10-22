"""Credit Card Fraud (ULB) analysis for issue #24: duplicates, a split by time, and what they mean.

Works offline from the raw CSV (Kaggle mlg-ulb/creditcardfraud); nothing is downloaded.

    uv run python eda/fraud/fraud_analysis.py --csv eda/output/data/creditcard.csv

Tables print to the terminal and are saved as CSV in --out (default eda/output/fraud/).
The stratified split uses the team recipe: two stratified steps, 60/20/20, seed 42.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
NO_TIME = [c for c in FEATURES if c != "Time"]
PARTS = ("train", "val", "test")

# Three ways to call two rows "the same", from strictest to loosest.
DEFINITIONS = {
    "exact row (Time, V1-V28, Amount, Class)": FEATURES + [TARGET],
    "same transaction, ignoring Class": FEATURES,
    "same V1-V28 and Amount, any Time": NO_TIME,
}


def group_ids(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """An id per row; rows with identical values in `cols` share an id (exact float match)."""
    return df.groupby(cols, sort=False).ngroup()


# --- duplicates -------------------------------------------------------------------------------


def duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate counts overall and among fraud rows, under each definition of 'the same'."""
    rows = []
    for scope, part in [("all rows", df), ("fraud rows", df[df[TARGET] == 1])]:
        for name, cols in DEFINITIONS.items():
            sizes = group_ids(part, cols).value_counts()
            repeated = sizes[sizes > 1]
            rows.append(
                {
                    "scope": scope,
                    "definition": name,
                    "rows": len(part),
                    "extra_copies": int(repeated.sum() - len(repeated)),
                    "groups_with_copies": len(repeated),
                    "rows_in_those_groups": int(repeated.sum()),
                }
            )
    return pd.DataFrame(rows)


def label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions identical in every feature but labelled both fraud and not fraud."""
    classes = df.groupby(group_ids(df, FEATURES))[TARGET].nunique()
    conflicting = classes[classes > 1]
    return pd.DataFrame({"groups_with_both_labels": [len(conflicting)]})


# --- splits -----------------------------------------------------------------------------------


def _label(df: pd.DataFrame, train, val, test) -> pd.Series:
    parts = pd.Series(pd.NA, index=df.index, dtype="object")
    parts.loc[train], parts.loc[val], parts.loc[test] = "train", "val", "test"
    return parts


def stratified_parts(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    train, hold = train_test_split(df.index, test_size=0.4, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(
        hold, test_size=0.5, stratify=df.loc[hold, TARGET], random_state=seed
    )
    return _label(df, train, val, test)


def time_parts(df: pd.DataFrame) -> pd.Series:
    """First 60% of rows by Time is train, the next 20% validation, the last 20% test."""
    order = df["Time"].sort_values(kind="stable").index
    n_train, n_val = round(len(df) * 0.6), round(len(df) * 0.2)
    return _label(df, order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :])


SPLITS = {"stratified": stratified_parts, "by time": time_parts}


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval for a proportion p measured on n cases."""
    return z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def part_summary(df: pd.DataFrame, parts: pd.Series) -> pd.DataFrame:
    rows = []
    for part in PARTS:
        d = df[parts == part]
        fraud = int(d[TARGET].sum())
        rows.append(
            {
                "part": part,
                "rows": len(d),
                "fraud": fraud,
                "fraud_rate_pct": round(100 * fraud / len(d), 3),
                "first_hour": round(d["Time"].min() / 3600, 1),
                "last_hour": round(d["Time"].max() / 3600, 1),
                "one_missed_case_pts_of_recall": round(100 / fraud, 2),
                "recall_ci_halfwidth_pts_at_0.8": round(100 * wilson_halfwidth(0.8, fraud), 1),
            }
        )
    return pd.DataFrame(rows)


def leakage(df: pd.DataFrame, parts: pd.Series) -> pd.DataFrame:
    """Exact-row copies inside each part, and validation/test rows that also sit in train."""
    ids = group_ids(df, FEATURES + [TARGET])
    train_ids = set(ids[parts == "train"])
    rows = []
    for part in PARTS:
        mask = parts == part
        sizes = ids[mask].value_counts()
        in_train = mask & ids.isin(train_ids) if part != "train" else mask & False
        rows.append(
            {
                "part": part,
                "extra_copies_inside_part": int(sizes[sizes > 1].sum() - (sizes > 1).sum()),
                "rows_also_in_train": int(in_train.sum()),
                "distinct_rows_also_in_train": int(ids[in_train].nunique()),
                "fraud_rows_also_in_train": int(df.loc[in_train, TARGET].sum()),
            }
        )
    return pd.DataFrame(rows)


def hour_of_day_summary(df: pd.DataFrame, block: int = 4) -> pd.DataFrame:
    """Fraud rate by time of day. Time counts seconds from the first transaction over two days."""
    hour_of_day = (df["Time"] / 3600) % 24
    blocks = (hour_of_day // block * block).astype(int)
    g = df.groupby(blocks)[TARGET].agg(rows="size", fraud="sum")
    g["fraud_rate_pct"] = (100 * g["fraud"] / g["rows"]).round(3)
    g.index = [f"{h:02d}-{h + block:02d}h" for h in g.index]
    return g.rename_axis("hour_of_day").reset_index()


# --- duplicate policies -----------------------------------------------------------------------


def apply_policy(df: pd.DataFrame, parts: pd.Series, policy: str, split_fn) -> tuple:
    """Return (frame, parts) after a duplicate policy. Rows removed by a policy leave the frame."""
    exact = group_ids(df, FEATURES + [TARGET])
    # A later copy within the same part. Comparing across parts would drop train rows just
    # because their twin sits in validation or test.
    copy_in_part = pd.DataFrame({"group": exact, "part": parts}).duplicated()
    if policy == "none":
        return df, parts
    if policy == "dedupe inside train only":
        drop = (parts == "train") & copy_in_part
        return df[~drop], parts[~drop]
    if policy == "drop val/test rows seen in train, dedupe inside parts":
        train_ids = set(exact[parts == "train"])
        drop = copy_in_part | ((parts != "train") & exact.isin(train_ids))
        return df[~drop], parts[~drop]
    if policy == "dedupe everything, then split":
        kept = df[~exact.duplicated()]
        return kept, split_fn(kept)
    raise ValueError(policy)


POLICIES = [
    "none",
    "dedupe inside train only",
    "drop val/test rows seen in train, dedupe inside parts",
    "dedupe everything, then split",
]


def policy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split_fn in SPLITS.items():
        base_parts = split_fn(df)
        for policy in POLICIES:
            d, parts = apply_policy(df, base_parts, policy, split_fn)
            row = {"split": split_name, "policy": policy}
            for part in PARTS:
                sub = d[parts == part]
                row[f"{part}_rows"], row[f"{part}_fraud"] = len(sub), int(sub[TARGET].sum())
            row["val_test_rows_seen_in_train"] = int(leakage(d, parts)["rows_also_in_train"].sum())
            row["fraud_removed"] = int(df[TARGET].sum() - d[TARGET].sum())
            rows.append(row)
    return pd.DataFrame(rows)


# --- recommended split ------------------------------------------------------------------------

# Part names in the row-ID files follow the EDA exports (validation, not val).
FILE_PART = {"train": "train", "val": "validation", "test": "test"}


def recommended_parts(df: pd.DataFrame, split_fn) -> tuple[pd.DataFrame, pd.Series]:
    """Drop exact duplicate rows (keeping the first), then split what is left."""
    return apply_policy(df, split_fn(df), "dedupe everything, then split", split_fn)


def fill_grid_table(df: pd.DataFrame, parts: pd.Series, levels: list[float]) -> pd.DataFrame:
    """Rows a fill level would add, from the training class gap, and how many times the real
    positives that is."""
    train = df[parts == "train"][TARGET]
    positives, negatives = int(train.sum()), int((1 - train).sum())
    gap = negatives - positives
    rows = [
        {
            "fill_level": level,
            "rows_added": round(level * gap),
            "times_real_positives": round(level * gap / positives, 1),
        }
        for level in levels
    ]
    return pd.DataFrame(rows).assign(train_positives=positives, class_gap=gap)


def write_splits(df: pd.DataFrame, out: Path) -> None:
    """Row-ID files for the recommended split, by stratified and by time, plus the drop list.

    source_row is the 0-based row position in the raw CSV (header not counted).
    """
    out.mkdir(parents=True, exist_ok=True)
    for tag, split_fn in [("", stratified_parts), ("time_", time_parts)]:
        kept, parts = recommended_parts(df, split_fn)
        for part in PARTS:
            ids = kept.index[(parts == part).loc[kept.index].to_numpy()]
            name = f"credit_card_fraud_{tag}{FILE_PART[part]}_row_ids.csv"
            pd.DataFrame({"source_row": ids}).to_csv(out / name, index=False)
    exact = group_ids(df, FEATURES + [TARGET])
    first = df.index.to_series().groupby(exact).transform("first")
    dropped = exact.duplicated()
    pd.DataFrame({"source_row": df.index[dropped], "copy_of_row": first[dropped]}).to_csv(
        out / "credit_card_fraud_dropped_duplicates.csv", index=False
    )


# --- run --------------------------------------------------------------------------------------


def show(title: str, table: pd.DataFrame, out: Path, name: str) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))
    table.to_csv(out / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("eda/output/data/creditcard.csv"))
    parser.add_argument("--out", type=Path, default=Path("eda/output/fraud"))
    parser.add_argument(
        "--write-splits", type=Path, help="folder for the recommended split's row-ID files"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"{args.csv}: {df.shape[0]:,} rows, {int(df[TARGET].sum())} fraud")
    show("Duplicates", duplicate_summary(df), args.out, "duplicates")
    show("Same features, both labels", label_conflicts(df), args.out, "label_conflicts")
    show("Fraud rate by time of day", hour_of_day_summary(df), args.out, "hour_of_day")

    for split_name, split_fn in SPLITS.items():
        parts = split_fn(df)
        show(f"Parts, {split_name} split", part_summary(df, parts), args.out, f"parts_{split_name}")
        show(
            f"Duplicates and leakage, {split_name}",
            leakage(df, parts),
            args.out,
            f"leak_{split_name}",
        )
        shared = int((parts.groupby(df["Time"]).nunique() > 1).sum())
        print(f"seconds shared by more than one part: {shared}")
    show("Duplicate policies", policy_table(df), args.out, "policies")

    kept, parts = recommended_parts(df, stratified_parts)
    levels = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    show(
        "Fill grid after dedupe, stratified",
        fill_grid_table(kept, parts, levels),
        args.out,
        "fill_grid",
    )
    if args.write_splits:
        write_splits(df, args.write_splits)
        print(f"\nwrote row-ID files to {args.write_splits}/")


if __name__ == "__main__":
    main()
