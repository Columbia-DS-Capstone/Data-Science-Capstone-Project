"""Augmented-training comparison: real training rows plus synthetic rows, at several fill levels.

For each generator, fill level and seed, the synthetic rows are added to the real training
rows only, encoded with the preprocessing fitted on the real training rows, and scored with
``run_models`` on the untouched real validation and test tables. Splits, models, threshold and
PR-AUC method match Noor's baseline (``tools/baselines.py``); only the model seed varies.

    uv run python -m agentic_gan.tools.augmented --generator smotenc
    uv run python -m agentic_gan.tools.augmented --generator ctgan --arm-label ctgan_v1
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agentic_gan.tools.baselines import run_models
from agentic_gan.tools.ctgan import IS_SYNTHETIC, fill_counts
from agentic_gan.tools.preprocessing import Preprocessor
from agentic_gan.tools.smotenc import smotenc_rows
from agentic_gan.utils.datasets import DATASETS, DatasetConfig

ROOT = Path(__file__).resolve().parents[3]
ID_COLUMN = "source_row"
SEEDS = (42, 43, 44, 45, 46)
FILL_LEVELS = (0.10, 0.25, 0.50)
# Provisional: top models by mean validation PR-AUC rank across the three datasets.
DEFAULT_MODELS = ("Stacking ensemble", "CatBoost", "LightGBM", "Gradient boosting")
REAL_ONLY = "real_only"
RESULT_COLUMNS = [
    "dataset", "arm", "fill_level", "n_synthetic", "seed", "model", "status", "selected",
    "val_pr_auc", "val_f1", "test_pr_auc", "test_f1", "test_precision", "test_recall",
    "test_roc_auc", "test_accuracy", "fit_seconds",
]
KEYS = ["dataset", "arm", "fill_level", "seed"]
DEFAULT_OUT = ROOT / "results" / "augmented_bank_marketing.csv"


def load_cleaned(config: DatasetConfig, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Read Noor's cleaned train, validation and test tables (row IDs kept for the check)."""
    folder = root / "data" / "interim" / config.name
    return {s: pd.read_csv(folder / f"{s}_cleaned.csv") for s in ("train", "validation", "test")}


def check_holdouts(tables: dict[str, pd.DataFrame], config: DatasetConfig) -> None:
    """Fail unless validation and test are real rows that are disjoint from training."""
    for split in ("validation", "test"):
        if IS_SYNTHETIC in tables[split].columns:
            raise ValueError(f"{split} contains an {IS_SYNTHETIC} column")
        if not tables[split][config.target].isin([0, 1]).all():
            raise ValueError(f"{split} target is not binary")
    ids = {s: set(t[ID_COLUMN]) for s, t in tables.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"{a} and {b} share source rows")


def _encode(pre: Preprocessor, df: pd.DataFrame, target: str):
    encoded = pre.transform_cleaned(df.reset_index(drop=True))
    return encoded.drop(columns=target), encoded[target]


def run_experiment(
    tables: dict[str, pd.DataFrame],
    config: DatasetConfig,
    generator: str,
    arm_label: str,
    pool: pd.DataFrame | None = None,
    seeds: Sequence[int] = SEEDS,
    fill_levels: Sequence[float] = FILL_LEVELS,
    model_names: Sequence[str] = DEFAULT_MODELS,
    include_real_only: bool = True,
) -> pd.DataFrame:
    """Return one row per fill level, seed and model.

    ``generator`` is "smotenc" (regenerated for every seed) or "ctgan" (a fixed ``pool`` of
    synthetic rows whose first n rows are used, so the fill levels are nested).
    """
    if generator not in ("smotenc", "ctgan"):
        raise ValueError("generator must be 'smotenc' or 'ctgan'")
    if generator == "ctgan" and pool is None:
        raise ValueError("ctgan needs a pool of synthetic rows")
    check_holdouts(tables, config)
    target = config.target
    real = {s: t.drop(columns=ID_COLUMN) for s, t in tables.items()}
    discrete = [c for c in config.discrete if c in real["train"].columns]

    pre = Preprocessor(config).fit(real["train"])  # real training rows only
    X_train, y_train = _encode(pre, real["train"], target)
    X_val, y_val = _encode(pre, real["validation"], target)
    X_test, y_test = _encode(pre, real["test"], target)
    counts = fill_counts(real["train"][target], fill_levels)

    frames = []
    for seed in seeds:
        levels = [(0.0, 0)] if include_real_only else []
        levels += [(f, counts[f]) for f in fill_levels]
        synthetic_pool = None
        if generator == "smotenc":
            synthetic_pool = smotenc_rows(
                real["train"], target, max(counts.values()), discrete_columns=discrete, seed=seed
            )
        elif pool is not None:
            synthetic_pool = pool
        for fill, n in levels:
            if n == 0:
                X, y, arm = X_train, y_train, REAL_ONLY
            else:
                extra = synthetic_pool.head(n)
                if len(extra) < n:
                    raise ValueError(f"pool has {len(extra)} rows, need {n}")
                X_syn, y_syn = _encode(pre, extra, target)
                X = pd.concat([X_train, X_syn], ignore_index=True)
                y = pd.concat([y_train.reset_index(drop=True), y_syn], ignore_index=True)
                arm = arm_label
            print(f"[{arm}] fill={fill:.2f} seed={seed} rows={len(X)}", flush=True)
            table = run_models(
                X, y, X_val, y_val, X_test, y_test, seed=seed, cv_folds=0,
                model_names=model_names,
            )
            table["dataset"], table["arm"] = config.name, arm
            table["fill_level"], table["n_synthetic"] = fill, n
            frames.append(table)
    return pd.concat(frames, ignore_index=True).reindex(columns=RESULT_COLUMNS)


def save_results(table: pd.DataFrame, path: Path) -> None:
    """Upsert by dataset, arm, fill level and seed, keeping other arms' rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        incoming = set(map(tuple, table[KEYS].drop_duplicates().to_numpy()))
        keep = [tuple(row) not in incoming for row in old[KEYS].to_numpy()]
        table = pd.concat([old.loc[keep], table], ignore_index=True)
    table.reindex(columns=RESULT_COLUMNS).to_csv(path, index=False)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, per arm and fill level.

    ``val_*`` are averaged over the models; ``test_*`` come from each seed's validation winner
    (the only model scored on test), as in the baseline.
    """
    keys = ["arm", "fill_level"]
    ok = table[table.status == "success"]
    val = ok.groupby([*keys, "seed"])[["val_pr_auc", "val_f1"]].mean().reset_index()
    val = val.groupby(keys)[["val_pr_auc", "val_f1"]].agg(["mean", "std"])
    test = ok[ok.selected].groupby(keys)[["test_pr_auc", "test_f1"]].agg(["mean", "std"])
    out = val.join(test)
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bank_marketing")
    parser.add_argument("--generator", choices=["smotenc", "ctgan"], required=True)
    parser.add_argument("--arm-label", help="Name of the arm in the results (default: generator)")
    parser.add_argument("--pool", type=Path, help="CTGAN file with the largest fill level")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--no-real-only", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    pool = None
    if args.generator == "ctgan":
        path = args.pool or ROOT / "data" / "interim" / config.name / "synthetic" / (
            "bank_ctgan_fill50.csv"
        )
        pool = pd.read_csv(path)
    table = run_experiment(
        load_cleaned(config), config, args.generator, args.arm_label or args.generator, pool,
        seeds=args.seeds, model_names=args.models, include_real_only=not args.no_real_only,
    )
    save_results(table, args.out)
    print(summarize(table).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
