"""How the pipeline samples CTGAN rows: the generator recipe and the V2 sampling steps.

V1 asked CTGAN for minority rows directly. On Bank Marketing that skewed ``job`` (technicians were
42% of the fake subscribers against 15% of the real ones). V2 fixed it on the sampling side:

* sample without conditioning on the target and keep the minority-class rows;
* drop rows that break the dataset's consistency rules (the same ``RULES`` the quality gate uses);
* draw the final rows from a large pool so that the category shares of ``rake_columns`` match the
  real minority rows (raking).

A V3 option adds a selection step between sampling and raking: draw a larger pool, keep only the
synthetic rows that look most real (``realism``) or that sit near a real-data model's decision
boundary (``boundary``), then rake as before. ``none`` is V2.

``GeneratorRecipe`` holds the options of one attempt, ``recipe_for`` gives the first
attempt's values for a dataset, and ``sample_for_fill`` returns the rows for every fill level at
once. The fill levels are the first ``n`` rows of that table, so they are nested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from agentic_gan.tools.quality_fidelity import Rule

SAMPLING_MODES = ("conditional", "unconditional")
POOL_ROWS = 60_000
CONDITIONAL_POOL_FACTOR = 3  # conditional pool = this many times the largest fill
RAKE_BINS = 5  # a numeric rake column is cut into this many quantile bins of the real rows
# A numeric column with at most this many distinct real values is a code (SEX, EDUCATION, PAY_1),
# so it is raked by value; only a column with more values (day_of_month) is binned.
CODED_MAX_VALUES = 12
RAKE_ITERATIONS = 50
UNCONDITIONAL_MAX_ROUNDS = 100  # the target is matched by rejection, so many rounds are normal
SELECTION_MODES = ("none", "realism", "boundary")
REALISM_KEEP = 0.5  # share of the pool the realism classifier keeps (the most real-looking half)
SELECTION_FOLDS = 2  # the realism classifier scores each pool row from a fold that never saw it
METADATA_COLUMNS = ("source_row", "is_synthetic")


@dataclass(frozen=True)
class GeneratorRecipe:
    """The options of one CTGAN attempt: training settings, then the sampling options.

    ``selection_mode`` is a recipe choice, not a retry knob: ``"none"`` is V2, ``"realism"`` and
    ``"boundary"`` are V3. ``pool_multiplier`` sizes the pool they select from (times the largest
    fill) and ``boundary_low`` / ``boundary_high`` bound the predicted probability that
    ``boundary`` keeps.
    """

    epochs: int
    batch_size: int
    pac: int
    discriminator_steps: int
    sampling_mode: str
    rule_filter: bool
    rake_columns: list[str]
    selection_mode: str = "none"
    pool_multiplier: int = 5
    boundary_low: float = 0.10
    boundary_high: float = 0.60

    def __post_init__(self):
        if self.sampling_mode not in SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {SAMPLING_MODES}, not {self.sampling_mode!r}"
            )
        _check_selection(
            self.selection_mode, self.pool_multiplier, self.boundary_low, self.boundary_high
        )


def _check_selection(mode: str, multiplier: int, low: float, high: float) -> None:
    if mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}, not {mode!r}")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError("pool_multiplier must be a whole number of at least 1")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("boundary_low and boundary_high must satisfy 0 <= low < high <= 1")


def recipe_for(dataset: str) -> GeneratorRecipe:
    """The first attempt's options: the V2 recipe for Bank and Default, V1 for the rest."""
    if dataset == "bank_marketing":
        return GeneratorRecipe(
            epochs=300,
            batch_size=500,
            pac=5,
            discriminator_steps=3,
            sampling_mode="unconditional",
            rule_filter=True,
            rake_columns=["job", "month", "poutcome", "marital", "day_of_month"],
        )
    if dataset == "default":
        return GeneratorRecipe(
            epochs=300,
            batch_size=500,
            pac=5,
            discriminator_steps=3,
            sampling_mode="unconditional",
            rule_filter=True,
            rake_columns=["SEX", "EDUCATION", "MARRIAGE", "PAY_1"],
        )
    return GeneratorRecipe(
        epochs=300,
        batch_size=500,
        pac=10,
        discriminator_steps=1,
        sampling_mode="conditional",
        rule_filter=False,
        rake_columns=[],
    )


def rule_row_filter(rules: Sequence[Rule]) -> Callable[[pd.DataFrame], pd.Series]:
    """A ``sample_synthetic`` row filter that keeps rows breaking none of ``rules``."""

    def keep(frame: pd.DataFrame) -> pd.Series:
        broken = pd.Series(False, index=frame.index)
        for rule in rules:
            broken |= rule.violated(frame)
        return ~broken

    return keep


def rake_weights(
    pool: pd.DataFrame,
    real: pd.DataFrame,
    columns: Sequence[str],
    iterations: int = RAKE_ITERATIONS,
) -> np.ndarray:
    """Row weights, summing to 1, that make the pool's category shares match ``real``."""
    weights = np.ones(len(pool))
    for _ in range(iterations):
        for col in columns:
            target = real[col].value_counts(normalize=True)
            current = pd.Series(weights, index=pool.index).groupby(pool[col]).sum()
            current = current / current.sum()
            ratio = (target / current).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            weights = weights * pool[col].map(ratio).fillna(1.0).to_numpy()
    return weights / weights.sum()


def with_rake_bins(
    pool: pd.DataFrame, real: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Copies of both tables where each continuous rake column is replaced by a bin label.

    The bins are quantile bins of the real rows, so each holds about the same share of them.
    A text column, or a numeric one with at most ``CODED_MAX_VALUES`` distinct real values (an
    integer code such as SEX or PAY_1), is raked by its own values instead. Returns the pool,
    the real rows and the column names to rake on.
    """
    pool, real = pool.copy(), real.copy()
    names = []
    for col in columns:
        if col not in pool.columns or col not in real.columns:
            raise ValueError(f"rake column {col!r} is not in the generated and real rows")
        if not pd.api.types.is_numeric_dtype(real[col]) or real[col].nunique() <= CODED_MAX_VALUES:
            names.append(col)
            continue
        edges = np.unique(np.quantile(real[col].dropna(), np.linspace(0, 1, RAKE_BINS + 1)))
        edges = np.array([-np.inf, np.inf]) if len(edges) < 2 else edges
        edges[0], edges[-1] = -np.inf, np.inf
        name = f"__bin_{col}"
        for frame in (pool, real):
            frame[name] = pd.cut(frame[col], edges).astype(str)
        names.append(name)
    return pool, real, names


def raked_sample(
    pool: pd.DataFrame, real: pd.DataFrame, n: int, columns: Sequence[str], seed: int
) -> pd.DataFrame:
    """``n`` pool rows, drawn without replacement, whose shares match ``real`` on ``columns``."""
    pool = pool.reset_index(drop=True)
    binned_pool, binned_real, names = with_rake_bins(pool, real, columns)
    weights = rake_weights(binned_pool, binned_real, names)
    if int((weights > 0).sum()) < n:
        raise ValueError(f"only {int((weights > 0).sum())} pool rows can be drawn for {n} wanted")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(pool), size=n, replace=False, p=weights)
    return pool.iloc[rng.permutation(picked)].reset_index(drop=True)


def _features(frame: pd.DataFrame, target: str) -> list[str]:
    return [c for c in frame.columns if c != target and c not in METADATA_COLUMNS]


def _encoder(features: pd.DataFrame) -> ColumnTransformer:
    """One-hot for text columns, numbers as they are; fitted inside each classifier."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(features[c])]
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def synthetic_probability(
    real: pd.DataFrame, synthetic: pd.DataFrame, *, target: str, seed: int
) -> np.ndarray:
    """Each synthetic row's predicted probability of being synthetic, from a fold that never saw it.

    A forest is fitted on real minority rows against synthetic rows (half of each at a time, then
    the other half), so it cannot memorise the rows it scores. A low value means the row looks
    like the real minority rows.
    """
    columns = [c for c in _features(synthetic, target) if c in real.columns]
    table = pd.concat([real[columns], synthetic[columns]], ignore_index=True)
    label = np.r_[np.zeros(len(real), dtype=int), np.ones(len(synthetic), dtype=int)]
    score = np.full(len(table), np.nan)
    folds = StratifiedKFold(SELECTION_FOLDS, shuffle=True, random_state=seed)
    for fit, held in folds.split(table, label):
        model = make_pipeline(
            _encoder(table),
            RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=1,
                random_state=seed,
            ),
        )
        model.fit(table.iloc[fit], label[fit])
        score[held] = model.predict_proba(table.iloc[held])[:, 1]
    return score[len(real) :]


def boundary_probability(
    train: pd.DataFrame, synthetic: pd.DataFrame, *, target: str, positive, seed: int
) -> np.ndarray:
    """Each synthetic row's predicted probability of the positive class, from a LightGBM model
    fitted on the real training rows only (never validation or test)."""
    from lightgbm import LGBMClassifier  # lazy: it must not load beside torch in one process

    columns = [c for c in _features(train, target) if c in synthetic.columns]
    model = make_pipeline(
        _encoder(train[columns]),
        LGBMClassifier(n_estimators=150, verbosity=-1, n_jobs=2, random_state=seed),
    )
    model.fit(train[columns], train[target])
    classes = list(model.classes_)
    return model.predict_proba(synthetic[columns])[:, classes.index(positive)]


def select_rows(
    pool: pd.DataFrame,
    *,
    mode: str,
    train: pd.DataFrame,
    positive_rows: pd.DataFrame,
    target: str,
    positive,
    wanted: int,
    low: float,
    high: float,
    seed: int,
) -> pd.DataFrame:
    """The pool rows ``mode`` keeps, at least ``wanted`` of them; ``"none"`` keeps the pool."""
    if mode == "none":
        return pool
    if mode == "realism":
        score = synthetic_probability(positive_rows, pool, target=target, seed=seed)
        keep = max(wanted, int(np.ceil(REALISM_KEEP * len(pool))))
        order = np.argsort(score, kind="stable")[:keep]
        return pool.iloc[order].reset_index(drop=True)
    probability = boundary_probability(train, pool, target=target, positive=positive, seed=seed)
    inside = (probability >= low) & (probability <= high)
    if int(inside.sum()) < wanted:
        raise ValueError(
            f"only {int(inside.sum())} of {len(pool)} pool rows have a predicted probability "
            f"between {low} and {high}, fewer than the {wanted} wanted; raise pool_multiplier "
            "or widen the bounds"
        )
    return pool[inside].reset_index(drop=True)


def sample_for_fill(
    model,
    train: pd.DataFrame,
    positive_rows: pd.DataFrame,
    counts: Sequence[int] | Mapping[object, int],
    *,
    sampling_mode: str,
    rule_filter: bool,
    rake_columns: Sequence[str],
    rules: Sequence[Rule],
    pool_rows: int = POOL_ROWS,
    selection_mode: str = "none",
    pool_multiplier: int = 5,
    boundary_low: float = 0.10,
    boundary_high: float = 0.60,
    seed: int,
) -> pd.DataFrame:
    """Return ``max(counts)`` synthetic minority rows; each fill level takes the first n of them.

    ``train`` is the real training table (rows copying it are never returned) and
    ``positive_rows`` its minority-class rows, which the raking matches. With
    ``rule_filter``, rows that break any of ``rules`` are rejected while sampling.

    With ``selection_mode`` ``"realism"`` or ``"boundary"`` the pool is ``pool_multiplier`` times
    the largest fill, the selection keeps part of it, and the rows are then raked as in V2. Only
    ``train`` and ``positive_rows`` are used to fit the selection models.
    """
    # Imported here: the CTGAN tool loads torch, which the rest of this module does not need.
    from agentic_gan.tools import ctgan as ctgan_tool

    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, not {sampling_mode!r}")
    _check_selection(selection_mode, pool_multiplier, boundary_low, boundary_high)
    wanted = max(counts.values() if isinstance(counts, Mapping) else counts)
    if wanted < 1:
        raise ValueError("counts must include at least one fill level with a row")
    if positive_rows.empty:
        raise ValueError("positive_rows is empty, so there is nothing to match")
    target_value = positive_rows[model.target_col].iloc[0]
    rake = list(rake_columns)
    unconditional = sampling_mode == "unconditional"

    if selection_mode != "none":
        pool_n = pool_multiplier * wanted
    elif rake:
        pool_n = max(pool_rows, wanted) if unconditional else CONDITIONAL_POOL_FACTOR * wanted
    else:
        pool_n = wanted
    options = {
        "target_value": target_value,
        "seed": seed,
        "drop_duplicates_of": train,
        "row_filter": rule_row_filter(rules) if rule_filter and rules else None,
        "conditional": not unconditional,
    }
    if unconditional:
        options["max_rounds"] = UNCONDITIONAL_MAX_ROUNDS
    pool = ctgan_tool.sample_synthetic(model, pool_n, **options)
    if len(pool) < wanted:
        raise ValueError(f"sampled {len(pool)} usable rows, fewer than the {wanted} wanted")
    pool = select_rows(
        pool,
        mode=selection_mode,
        train=train,
        positive_rows=positive_rows,
        target=model.target_col,
        positive=target_value,
        wanted=wanted,
        low=boundary_low,
        high=boundary_high,
        seed=seed,
    )
    if not rake:
        return pool.head(wanted).reset_index(drop=True)
    return raked_sample(pool, positive_rows, wanted, rake, seed)
