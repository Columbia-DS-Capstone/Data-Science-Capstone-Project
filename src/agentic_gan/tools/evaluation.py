"""Deterministic evaluation tools: arm comparison, the reliable-gain rule, segments, number check.

The Evaluation agent's LLM only reads what these functions return; it never computes a number.
They read the training agent's `metrics.csv` (one row per arm, fill level, seed, model and split,
`selected` marking the model chosen on validation PR-AUC) and the prediction files it saves.

Reliable gain, as the rule is written down here: at a fill level, a metric has a reliable gain
when, on both validation and test, the mean paired lift over the original arm (same seeds) is
larger than the seed-to-seed spread of that lift (its sample standard deviation over the seeds),
and at least MIN_SEEDS seeds are paired. A lift below minus the spread on both is a reliable
loss; anything else, a lift inside the noise or one that differs between splits, is inconclusive.

With fewer than MIN_SEEDS seeds (one seed, 42, is the default) the uncertainty comes from a paired
bootstrap over the rows of each evaluation split instead: the 95% interval of the metric
difference, with positives and negatives resampled separately. A gain needs the interval's lower
bound above 0 on both validation and test, a loss its upper bound below 0 on both, and anything
else is inconclusive. The bootstrap covers the sampling noise of the evaluation rows only. Model
training and generator randomness are fixed by the seed, so it says nothing about them.
"""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from agentic_gan.tools.baselines import pr_auc_score

METRICS = ("pr_auc", "roc_auc", "f1", "accuracy", "precision", "recall")
SPLITS = ("validation", "test")
ARM_ORDER = ("original", "cleaned", "smotenc", "ctgan")
REFERENCE_ARM = "original"
MIN_SEEDS = 3
MAX_CATEGORIES = 10
N_BOOT = 1000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_METRICS = ("pr_auc", "roc_auc")
_BOOT_COLUMNS = ["boot_seed", "boot_diff", "boot_lo", "boot_hi", "n_boot"]
_CHUNK_CELLS = 4_000_000  # resample weights held at once: rows in a chunk times evaluation rows
_COLUMNS = ("dataset", "arm", "fill_level", "seed", "split", "status", "selected", *METRICS)
_ONE_ROW = ["dataset", "arm", "fill_level", "seed", "split"]
_GROUP = ["dataset", "arm", "fill_level", "split", "metric"]


def _selected_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """The successful models the training agent selected, on the validation and test splits."""
    missing = [name for name in _COLUMNS if name not in metrics]
    if missing:
        raise ValueError(f"metrics table is missing columns: {missing}")
    if "synthetic_weight" in metrics:
        weights = metrics["synthetic_weight"].fillna(1.0)
        if weights.groupby(metrics["dataset"]).nunique().gt(1).any():
            raise ValueError(
                "filter metrics to one synthetic_weight per dataset before comparisons"
            )
    return metrics[
        metrics["status"].astype(str).str.upper().eq("SUCCESS")
        & metrics["selected"].astype(str).str.lower().eq("true")
        & metrics["split"].isin(SPLITS)
    ]


# --- paired bootstrap -------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """The observed difference `other - base` in a metric, with its paired-bootstrap interval."""

    diff: float
    lo: float
    hi: float
    n_boot: int


class _Ranked:
    """One model's scores, sorted once, so many weighted resamples of the rows can be scored."""

    def __init__(self, y: np.ndarray, scores: np.ndarray):
        self.order = np.argsort(-scores, kind="stable")
        ranked = scores[self.order]
        self.ends = np.flatnonzero(np.append(ranked[1:] != ranked[:-1], True))  # last of each tie
        self.positive = y[self.order] == 1
        self.n_positive = int(self.positive.sum())
        self.n_negative = len(y) - self.n_positive

    def area(self, counts: np.ndarray, metric: str) -> np.ndarray:
        """`metric` of every resample; `counts[i, j]` is how often row j is drawn in resample i.

        This equals scoring the drawn rows one by one with `pr_auc_score` (trapezoidal, as in the
        metrics tables) or `roc_auc_score`. Both are areas under a curve with a point at each
        distinct score, and a resample only changes the weight each row carries along the curve.
        """
        weights = counts[:, self.order]
        true_pos = np.cumsum(weights * self.positive, axis=1)[:, self.ends]
        false_pos = np.cumsum(weights * ~self.positive, axis=1)[:, self.ends]
        start = np.zeros((len(counts), 1))
        recall = np.hstack([start, true_pos / self.n_positive])
        if metric == "roc_auc":
            across, height = np.hstack([start, false_pos / self.n_negative]), recall
        else:
            seen = true_pos + false_pos
            precision = np.where(seen > 0, true_pos / np.maximum(seen, 1), 1.0)
            across, height = recall, np.hstack([start + 1.0, precision])
        return np.sum(np.diff(across, axis=1) * (height[:, 1:] + height[:, :-1]) / 2, axis=1)


def paired_bootstrap(
    y_true,
    base_scores,
    other_scores,
    *,
    metric: str = "pr_auc",
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = BOOTSTRAP_ALPHA,
) -> BootstrapResult:
    """How far `other_scores` are above `base_scores` on the same rows, with an interval.

    Every resample draws the row indices of the positives, and of the negatives, with replacement
    (so the class counts never change) and scores both models on the same drawn rows. The interval
    is the `alpha / 2` and `1 - alpha / 2` percentiles of `metric(other) - metric(base)` over
    `n_boot` resamples. `diff` is the same difference on the rows as given. `metric` is `pr_auc`
    (trapezoidal, as in the metrics tables) or `roc_auc`. The inputs must be aligned: the same
    rows in the same order. The same `seed` gives the same result.
    """
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(f"metric must be one of {BOOTSTRAP_METRICS}, not {metric!r}")
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError("n_boot must be a positive whole number")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    y = np.asarray(y_true)
    base = np.asarray(base_scores, dtype=float)
    other = np.asarray(other_scores, dtype=float)
    if y.ndim != 1 or base.shape != y.shape or other.shape != y.shape:
        raise ValueError("y_true, base_scores and other_scores must be aligned: one value per row")
    if not (np.isfinite(base).all() and np.isfinite(other).all()):
        raise ValueError("scores must be finite")
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("y_true must hold both classes, 0 and 1")
    y = y.astype(int)
    score = pr_auc_score if metric == "pr_auc" else roc_auc_score
    observed = float(score(y, other) - score(y, base))

    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    base_ranked, other_ranked = _Ranked(y, base), _Ranked(y, other)
    rng = np.random.default_rng(seed)
    n_boot = int(n_boot)
    diffs = np.empty(n_boot)
    chunk = max(1, _CHUNK_CELLS // len(y))
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        counts = np.empty((size, len(y)), dtype=np.int32)
        for row in range(size):
            drawn = np.concatenate(
                [
                    positives[rng.integers(0, len(positives), len(positives))],
                    negatives[rng.integers(0, len(negatives), len(negatives))],
                ]
            )
            counts[row] = np.bincount(drawn, minlength=len(y))
        diffs[start : start + size] = other_ranked.area(counts, metric) - base_ranked.area(
            counts, metric
        )
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(observed, float(lo), float(hi), n_boot)


def load_selected_scores(
    models_dir: str | Path,
    metrics: pd.DataFrame,
    *,
    arm: str,
    fill_level: float,
    seed: int,
    split: str,
) -> pd.DataFrame:
    """The saved scores of the model selected for one arm, fill level, seed and split.

    `metrics` is the training agent's metrics table (`selected` marks the validation winner) and
    `models_dir` its models folder. Returns `source_row`, `y_true` and `probability`, sorted by
    `source_row`, so two arms' scores line up row for row.
    """
    # Imported here so the tools do not load the training agent unless scores are read.
    from agentic_gan.agents.training import ARM_LABELS, model_artifact_stem

    arms = {label: arm_enum for arm_enum, label in ARM_LABELS.items()}
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(arms)}")
    if "model" not in metrics:
        raise ValueError("metrics table is missing the model column")
    rows = _selected_rows(metrics)
    cell = rows[
        rows["arm"].eq(arm)
        & np.isclose(rows["fill_level"], fill_level)
        & rows["seed"].eq(seed)
        & rows["split"].eq(split)
    ]
    if len(cell) != 1:
        raise ValueError(
            f"expected one selected model for {arm} at fill {fill_level:g}, seed {seed}, {split}; "
            f"found {len(cell)}"
        )
    stem = model_artifact_stem(
        arms[arm], float(cell["fill_level"].iloc[0]), seed, cell["model"].iloc[0]
    )
    scores = pd.read_csv(Path(models_dir) / f"{stem}_{split}.csv")
    return scores[["source_row", "y_true", "probability"]].sort_values(
        "source_row", ignore_index=True
    )


def bootstrap_table(
    metrics: pd.DataFrame,
    models_dir: str | Path | Mapping[str, str | Path],
    *,
    reference: str = REFERENCE_ARM,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """The paired-bootstrap difference from the reference arm, per dataset, arm, level and split.

    Scores come from the selected models' saved predictions, at the lowest seed that the arm and
    the reference share. `models_dir` is the training agent's models folder, or a mapping from
    dataset name to one. Each row carries `boot_diff`, its interval `boot_lo` to `boot_hi` and
    the model seed it used, for each metric in `BOOTSTRAP_METRICS`.
    """
    rows = _selected_rows(metrics)
    table = []
    for dataset, group in rows.groupby("dataset", sort=False):
        if isinstance(models_dir, Mapping) and dataset not in models_dir:
            continue  # no saved scores for this dataset, so it gets no bootstrap rows
        folder = models_dir[dataset] if isinstance(models_dir, Mapping) else models_dir
        base = group[group["arm"].eq(reference)]
        if base.empty:
            continue
        base_level = float(base["fill_level"].iloc[0])
        others = group[group["arm"].ne(reference)]
        for (arm, level), cells in others.groupby(["arm", "fill_level"], sort=False):
            for split in SPLITS:
                shared = set(cells.loc[cells["split"].eq(split), "seed"]) & set(
                    base.loc[base["split"].eq(split), "seed"]
                )
                if not shared:
                    continue
                used = min(shared)
                kept = load_selected_scores(
                    folder, group, arm=reference, fill_level=base_level, seed=used, split=split
                )
                other = load_selected_scores(
                    folder, group, arm=arm, fill_level=level, seed=used, split=split
                )
                if not (
                    kept["source_row"].equals(other["source_row"])
                    and kept["y_true"].equals(other["y_true"])
                ):
                    raise ValueError(
                        f"{arm} at fill {level:g} and {reference} are scored on different rows "
                        f"({dataset}, seed {used}, {split})"
                    )
                for metric in BOOTSTRAP_METRICS:
                    result = paired_bootstrap(
                        kept["y_true"],
                        kept["probability"],
                        other["probability"],
                        metric=metric,
                        n_boot=n_boot,
                        seed=seed,
                    )
                    table.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "fill_level": level,
                            "split": split,
                            "metric": metric,
                            "boot_seed": int(used),
                            "boot_diff": result.diff,
                            "boot_lo": result.lo,
                            "boot_hi": result.hi,
                            "n_boot": result.n_boot,
                        }
                    )
    return pd.DataFrame(table, columns=[*_GROUP, *_BOOT_COLUMNS])


def compare_arms(
    metrics: pd.DataFrame,
    *,
    reference: str = REFERENCE_ARM,
    models_dir: str | Path | Mapping[str, str | Path] | None = None,
    n_boot: int = N_BOOT,
) -> pd.DataFrame:
    """Mean and seed-to-seed spread per dataset, arm, fill level, split and metric.

    Each arm, level and seed counts once, with the model the training agent selected. `diff_mean`
    and `diff_sd` are the paired difference from the reference arm at the same seeds (NaN for the
    reference itself); `n_paired` is how many seeds had a reference to pair with. With
    `models_dir` (the training agent's models folder, see `bootstrap_table`) the table also has the
    `boot_*` columns: the paired-bootstrap difference from the reference and its interval, which
    `reliable_gain` uses when fewer than MIN_SEEDS seeds are paired.
    """
    rows = _selected_rows(metrics)
    if rows.duplicated(_ONE_ROW).any():
        raise ValueError("more than one selected model per dataset, arm, fill level, seed, split")
    long = rows.melt(
        id_vars=_ONE_ROW, value_vars=list(METRICS), var_name="metric", value_name="value"
    ).dropna(subset=["value"])
    pair_on = ["dataset", "split", "metric", "seed"]
    base = long.loc[long["arm"] == reference].set_index(pair_on)["value"].rename("reference")
    if not base.index.is_unique:
        raise ValueError(f"the {reference} arm has more than one fill level or selected model")
    paired = long.join(base, on=pair_on)
    paired["diff"] = (paired["value"] - paired["reference"]).where(paired["arm"] != reference)
    table = paired.groupby(_GROUP, sort=False).agg(
        n_seeds=("value", "count"),
        mean=("value", "mean"),
        sd=("value", "std"),
        n_paired=("diff", "count"),
        diff_mean=("diff", "mean"),
        diff_sd=("diff", "std"),
    )
    table = table.reset_index()
    if models_dir is not None:
        boot = bootstrap_table(metrics, models_dir, reference=reference, n_boot=n_boot)
        table = table.merge(boot, on=_GROUP, how="left")
    rank = table["arm"].map({arm: i for i, arm in enumerate(ARM_ORDER)}).fillna(len(ARM_ORDER))
    metric_rank = table["metric"].map({metric: i for i, metric in enumerate(METRICS)})
    order = table.assign(rank=rank, metric_rank=metric_rank).sort_values(
        ["dataset", "rank", "fill_level", "split", "metric_rank"]
    )
    return order.drop(columns=["rank", "metric_rank"]).reset_index(drop=True)


@dataclass(frozen=True)
class GainVerdict:
    """The reliable-gain verdict for one arm and fill level, with the numbers behind it."""

    dataset: str
    arm: str
    fill_level: float
    metric: str
    verdict: str  # reliable gain, reliable loss, inconclusive or not enough paired seeds
    lift: dict[str, float]  # mean paired lift over the reference arm, by split
    spread: dict[str, float]  # seed-to-seed spread of that lift, or half its bootstrap interval
    reason: str
    basis: str = "seed spread"  # "paired bootstrap" when fewer than MIN_SEEDS seeds are paired
    interval: dict[str, tuple[float, float]] = field(default_factory=dict)  # bootstrap, by split

    @property
    def reliable(self) -> bool:
        return self.verdict == "reliable gain"


def _bootstrap_intervals(rows: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """The bootstrap difference and its interval for each split, or {} unless both have one."""
    if not {"boot_diff", "boot_lo", "boot_hi"} <= set(rows.columns):
        return {}
    found = {}
    for split in SPLITS:
        if split not in rows.index:
            return {}
        values = rows.loc[split, ["boot_diff", "boot_lo", "boot_hi"]]
        if values.isna().any():
            return {}
        found[split] = tuple(float(value) for value in values)
    return found


def _bootstrap_verdict(
    dataset: str, arm: str, fill_level: float, metric: str, boot: dict
) -> GainVerdict:
    lift = {split: boot[split][0] for split in SPLITS}
    interval = {split: (boot[split][1], boot[split][2]) for split in SPLITS}
    spread = {split: (hi - lo) / 2 for split, (lo, hi) in interval.items()}
    reason = "; ".join(
        f"{s} lift {lift[s]:.4f}, interval [{interval[s][0]:.4f}, {interval[s][1]:.4f}]"
        for s in SPLITS
    )
    if all(interval[split][0] > 0 for split in SPLITS):
        verdict = "reliable gain"
    elif all(interval[split][1] < 0 for split in SPLITS):
        verdict = "reliable loss"
    else:
        verdict = "inconclusive"
    return GainVerdict(
        dataset,
        arm,
        fill_level,
        metric,
        verdict,
        lift,
        spread,
        reason,
        "paired bootstrap",
        interval,
    )


def reliable_gain(
    comparison: pd.DataFrame,
    *,
    dataset: str,
    arm: str,
    fill_level: float = 0.0,
    metric: str = "pr_auc",
) -> GainVerdict:
    """Apply the reliable-gain rule (module docstring) to one row group of `compare_arms`.

    With at least MIN_SEEDS paired seeds the seed-spread rule decides. With fewer, the verdict
    comes from the bootstrap interval, if `compare_arms` was given the scores to compute it.
    """
    rows = comparison[
        comparison["dataset"].eq(dataset)
        & comparison["arm"].eq(arm)
        & np.isclose(comparison["fill_level"], fill_level)
        & comparison["metric"].eq(metric)
    ].set_index("split")
    boot = _bootstrap_intervals(rows)
    lift, spread = {}, {}
    for split in SPLITS:
        if split not in rows.index or rows.loc[split, "n_paired"] < MIN_SEEDS:
            if boot:
                return _bootstrap_verdict(dataset, arm, fill_level, metric, boot)
            reason = f"fewer than {MIN_SEEDS} seeds paired with the original arm on {split}"
            return GainVerdict(
                dataset, arm, fill_level, metric, "not enough paired seeds", lift, spread, reason
            )
        lift[split] = float(rows.loc[split, "diff_mean"])
        spread[split] = float(rows.loc[split, "diff_sd"])
    reason = "; ".join(f"{s} lift {lift[s]:.4f} vs spread {spread[s]:.4f}" for s in SPLITS)
    if all(lift[split] > spread[split] for split in SPLITS):
        verdict = "reliable gain"
    elif all(lift[split] < -spread[split] for split in SPLITS):
        verdict = "reliable loss"
    else:
        verdict = "inconclusive"
    interval = {split: (boot[split][1], boot[split][2]) for split in boot}
    return GainVerdict(
        dataset, arm, fill_level, metric, verdict, lift, spread, reason, interval=interval
    )


def segment_metrics(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    column: str,
    *,
    bins: int = 4,
    min_positives: int = 20,
) -> pd.DataFrame:
    """Precision, recall, F1 and PR-AUC of one model's predictions within segments of a column.

    `predictions` is a prediction file from the training agent (source_row, y_true, probability,
    prediction). `features` holds the same rows' `source_row` and the segment column; a numeric
    column with many values is cut into `bins` quantile bands. A column that is missing gives an
    empty table. Segments with fewer than `min_positives` positives are flagged `small`.
    """
    names = [
        "segment",
        "n",
        "positives",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "small",
        "segment_lower_bound",
        "segment_upper_bound",
    ]
    if column not in features:
        return pd.DataFrame(columns=names)
    values = (
        features.set_index("source_row")[column] if "source_row" in features else features[column]
    )
    if not values.index.is_unique:
        raise ValueError("features has repeated source_row values")
    bounds = {}
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > MAX_CATEGORIES:
        bands = pd.qcut(values, bins, duplicates="drop")
        # Store the displayed bounds as numbers so report verification can ground
        # interval labels (including qcut's adjusted lower endpoint).
        bounds = {str(band): (float(band.left), float(band.right)) for band in bands.cat.categories}
        values = bands.astype(str).where(bands.notna())
    data = predictions.assign(segment=predictions["source_row"].map(values)).dropna(
        subset=["segment"]
    )
    rows = []
    for segment, part in data.groupby("segment", sort=False):
        y, predicted = part["y_true"], part["prediction"]
        positives = int(y.sum())
        has_both = 0 < positives < len(part)
        rows.append(
            {
                "segment": str(segment),
                "segment_lower_bound": bounds.get(str(segment), (np.nan, np.nan))[0],
                "segment_upper_bound": bounds.get(str(segment), (np.nan, np.nan))[1],
                "n": len(part),
                "positives": positives,
                "precision": precision_score(y, predicted, zero_division=0),
                "recall": recall_score(y, predicted, zero_division=0) if positives else np.nan,
                "f1": f1_score(y, predicted, zero_division=0) if positives else np.nan,
                "pr_auc": pr_auc_score(y, part["probability"]) if has_both else np.nan,
                "small": positives < min_positives,
            }
        )
    table = pd.DataFrame(rows, columns=names)
    return table.sort_values("n", ascending=False, ignore_index=True)


@dataclass(frozen=True)
class NumberCheck:
    """Which numbers in a report were found in the results."""

    verified: bool
    checked: int
    unverified: tuple[str, ...]


# Digits not glued to letters or a dot, with thousands separators, decimals and a percent sign.
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:#+[ \t]*)?\d+[.)][ \t]+")


def _json_numbers(node) -> Iterable[float]:
    if isinstance(node, dict):
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            yield from _json_numbers(child)
    elif isinstance(node, int | float) and not isinstance(node, bool):
        yield node


def _result_numbers(result_files: Iterable[str | Path]) -> np.ndarray:
    values: list[float] = []
    for path in map(Path, result_files):
        if path.suffix == ".json":
            values.extend(_json_numbers(json.loads(path.read_text())))
        else:
            values.extend(pd.read_csv(path).select_dtypes("number").to_numpy().ravel())
    numbers = np.abs(np.asarray(values, dtype=float))
    return numbers[np.isfinite(numbers)]


def _found(token: str, known: np.ndarray) -> bool:
    percent = token.endswith("%")
    text = token.rstrip("%").replace(",", "")
    value = float(text)
    decimals = len(text.partition(".")[2])
    if "." not in text and not percent:
        return bool(np.any(np.abs(known - value) < 1e-9))  # a count or an id must match exactly
    options = [(value, decimals)] + ([(value / 100, decimals + 2)] if percent else [])
    return any(np.any(np.abs(known - v) <= 0.5 * 10.0**-d + 1e-9) for v, d in options)


def verify_report_numbers(
    report_text: str, result_files: Iterable[str | Path], *, allowed: Iterable[float] = ()
) -> NumberCheck:
    """Check that every number quoted in the report is in the results (CSV or JSON files).

    A decimal or percentage matches a result that rounds to it at the digits quoted (0.05 matches
    0.0524; 25% matches 0.25), ignoring sign so "dropped by 0.05" matches -0.05. A whole number
    such as a seed count must match exactly. `allowed` lists numbers the caller vouches for, such
    as the configured seeds. Numbers glued to letters (v1, pr_auc2) and list numbering are skipped.
    """
    known = np.concatenate([_result_numbers(result_files), np.abs(list(allowed))])
    tokens = _NUMBER.findall(_LIST_MARKER.sub("", report_text))
    unverified = tuple(dict.fromkeys(token for token in tokens if not _found(token, known)))
    return NumberCheck(not unverified, len(tokens), unverified)
