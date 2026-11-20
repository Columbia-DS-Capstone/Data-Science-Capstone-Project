"""The generator recipe and the V2 sampling steps, with a stand-in for the fitted CTGAN.

The stand-in draws rows from a fixed table, so the real ``sample_synthetic`` runs (target match,
rule filter, copy check) without training a network.
"""

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.ctgan import TrainedCTGAN
from agentic_gan.tools.ctgan_sampling import (
    GeneratorRecipe,
    recipe_for,
    rule_row_filter,
    sample_for_fill,
    with_rake_bins,
)
from agentic_gan.tools.quality_fidelity import BANK_RULES, DEFAULT_RULES

JOBS = ["admin", "technician", "retired"]


class StubCTGAN:
    """Draws rows, with replacement, from a fixed table; ignores the condition like soft CTGAN."""

    def __init__(self, source: pd.DataFrame):
        self.source = source
        self.rng = np.random.default_rng(0)

    def set_random_state(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self, n, condition_column=None, condition_value=None):
        picked = self.rng.integers(0, len(self.source), n)
        return self.source.iloc[picked].reset_index(drop=True)


def bank_source(n=6000, seed=1, job_probs=(0.6, 0.3, 0.1), broken_share=0.2) -> pd.DataFrame:
    """Bank-shaped rows. ``broken_share`` of them break the Bank pdays/poutcome rule."""
    rng = np.random.default_rng(seed)
    contacted = rng.random(n) < 0.3
    frame = pd.DataFrame(
        {
            "job": rng.choice(JOBS, n, p=job_probs),
            "day_of_month": rng.integers(1, 29, n),
            "pdays": np.where(contacted, rng.integers(1, 300, n), -1),
            "poutcome": np.where(contacted, rng.choice(["failure", "success"], n), "unknown"),
            "previously_contacted": contacted.astype(int),
            "previous": np.where(contacted, rng.integers(1, 6, n), 0),
            "balance": rng.normal(1000, 300, n).round(3),
            "y": rng.integers(0, 2, n),
        }
    )
    broken = rng.random(n) < broken_share
    frame.loc[broken, "pdays"] = -1
    frame.loc[broken, "poutcome"] = "success"
    return frame


def fitted(source: pd.DataFrame) -> TrainedCTGAN:
    return TrainedCTGAN(
        model=StubCTGAN(source),
        target_col="y",
        discrete_columns=["job", "poutcome", "y"],
        columns=list(source.columns),
        dtypes={name: str(dtype) for name, dtype in source.dtypes.items()},
        params={},
    )


def reference(job_probs=(0.2, 0.3, 0.5), n=2000, seed=7) -> pd.DataFrame:
    """Real minority rows whose job and day shares differ from the generator's."""
    rng = np.random.default_rng(seed)
    frame = bank_source(n, seed=seed, job_probs=job_probs, broken_share=0)
    frame["day_of_month"] = rng.choice(np.arange(1, 29), n, p=_late_month_days())
    frame["y"] = 1
    return frame


def _late_month_days():
    weights = np.arange(1, 29, dtype=float)
    return weights / weights.sum()


def draw(model, train, real, counts, **overrides):
    options = {
        "sampling_mode": "unconditional",
        "rule_filter": True,
        "rake_columns": [],
        "rules": BANK_RULES,
        "pool_rows": 6000,
        "seed": 42,
    }
    options.update(overrides)
    return sample_for_fill(model, train, real, counts, **options)


def shares(frame, column):
    return frame[column].value_counts(normalize=True)


def test_recipe_for_bank_is_the_v2_recipe():
    recipe = recipe_for("bank_marketing")
    assert recipe == GeneratorRecipe(
        epochs=300,
        batch_size=500,
        pac=5,
        discriminator_steps=3,
        sampling_mode="unconditional",
        rule_filter=True,
        rake_columns=["job", "month", "poutcome", "marital", "day_of_month"],
    )


def test_recipe_for_default_is_the_v2_recipe():
    assert recipe_for("default") == GeneratorRecipe(
        epochs=300,
        batch_size=500,
        pac=5,
        discriminator_steps=3,
        sampling_mode="unconditional",
        rule_filter=True,
        rake_columns=["SEX", "EDUCATION", "MARRIAGE", "PAY_1"],
    )


@pytest.mark.parametrize("dataset", ["polish", "ieee_cis", "anything_else"])
def test_recipe_for_other_datasets_is_v1(dataset):
    assert recipe_for(dataset) == GeneratorRecipe(
        epochs=300,
        batch_size=500,
        pac=10,
        discriminator_steps=1,
        sampling_mode="conditional",
        rule_filter=False,
        rake_columns=[],
    )


def test_each_recipe_is_a_fresh_object():
    recipe = recipe_for("bank_marketing")
    recipe.rake_columns.append("education")
    assert "education" not in recipe_for("bank_marketing").rake_columns


def test_recipe_rejects_an_unknown_sampling_mode():
    with pytest.raises(ValueError, match="sampling_mode"):
        GeneratorRecipe(300, 500, 10, 1, "sideways", False, [])


@pytest.mark.parametrize("mode", ["unconditional", "conditional"])
@pytest.mark.parametrize("rake", [[], ["job"]])
def test_row_count_is_the_largest_fill(mode, rake):
    source = bank_source()
    train = source.head(300)
    rows = draw(
        fitted(source),
        train,
        reference(),
        [100, 250, 500],
        sampling_mode=mode,
        rake_columns=rake,
    )
    assert len(rows) == 500
    assert (rows["y"] == 1).all()
    assert (rows["is_synthetic"] == 1).all()
    assert list(rows.index) == list(range(500))


def test_counts_may_be_a_mapping():
    source = bank_source()
    rows = draw(fitted(source), source.head(300), reference(), {0.1: 40, 0.5: 200})
    assert len(rows) == 200


def test_no_row_equals_a_train_row():
    source = bank_source()
    train = source.head(1500)
    rows = draw(fitted(source), train, reference(), [800], rake_columns=["job"])
    real = set(map(tuple, train.itertuples(index=False)))
    made = set(map(tuple, rows[list(source.columns)].itertuples(index=False)))
    assert len(made) == len(rows)
    assert not made & real


def test_rule_filter_removes_rows_that_break_a_bank_rule():
    source = bank_source(broken_share=0.2)
    model, train = fitted(source), source.head(300)
    unfiltered = draw(model, train, reference(), [1000], rule_filter=False)
    filtered = draw(model, train, reference(), [1000], rule_filter=True)
    assert rule_row_filter(BANK_RULES)(unfiltered).sum() < len(unfiltered)
    assert rule_row_filter(BANK_RULES)(filtered).all()
    broken = pd.Series(False, index=filtered.index)
    for rule in BANK_RULES:
        broken |= rule.violated(filtered)
    assert not broken.any()


def test_rule_filter_is_off_when_the_recipe_says_so_or_no_rules_exist():
    source = bank_source(broken_share=0.2)
    model, train = fitted(source), source.head(300)
    off = draw(model, train, reference(), [1000], rule_filter=False)
    none = draw(model, train, reference(), [1000], rule_filter=True, rules=())
    for rows in (off, none):
        assert not rule_row_filter(BANK_RULES)(rows).all()


def test_raking_brings_category_shares_to_the_reference():
    # The generator over-produces job 0 and under-produces job 2 compared with the real rows.
    source = bank_source(n=40_000, broken_share=0, job_probs=(0.4, 0.3, 0.3))
    model, train = fitted(source), source.head(300)
    real = reference(job_probs=(0.3, 0.3, 0.4))
    plain = draw(model, train, real, [2000], pool_rows=12_000)
    raked = draw(model, train, real, [2000], pool_rows=12_000, rake_columns=["job"])
    want = shares(real, "job")
    assert (shares(plain, "job") - want).abs().max() > 0.08
    assert (shares(raked, "job") - want).abs().max() < 0.02


def test_raking_a_numeric_column_matches_its_quantile_bins():
    source = bank_source(n=40_000, broken_share=0)
    model, train, real = fitted(source), source.head(300), reference()
    raked = draw(model, train, real, [2000], pool_rows=12_000, rake_columns=["day_of_month"])
    edges = np.unique(np.quantile(real["day_of_month"], np.linspace(0, 1, 6)))
    edges[0], edges[-1] = -np.inf, np.inf
    want = pd.cut(real["day_of_month"], edges).value_counts(normalize=True)
    got = pd.cut(raked["day_of_month"], edges).value_counts(normalize=True)
    assert (got - want).abs().max() < 0.03
    assert "__bin_day_of_month" not in raked.columns


def test_the_same_seed_gives_the_same_rows_and_another_seed_does_not():
    source = bank_source()
    train = source.head(300)
    args = (fitted(source), train, reference(), [300])
    first = draw(*args, rake_columns=["job"], seed=5)
    again = draw(*args, rake_columns=["job"], seed=5)
    other = draw(*args, rake_columns=["job"], seed=6)
    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)


def test_smaller_fills_are_the_first_rows_of_the_largest():
    source = bank_source()
    rows = draw(
        fitted(source), source.head(300), reference(), [100, 250, 500], rake_columns=["job"]
    )
    for count in (100, 250):
        pd.testing.assert_frame_equal(rows.head(count), rows.iloc[:count])
    assert rows.head(100).index.tolist() == list(range(100))
    assert not rows.head(100).duplicated(subset=list(rows.columns)).any()


def test_too_few_usable_rows_raises():
    source = bank_source(n=40, broken_share=0)
    with pytest.raises(ValueError, match="fewer than"):
        draw(fitted(source), source.head(5), reference(), [500], pool_rows=500)


def test_an_unknown_rake_column_raises():
    source = bank_source()
    with pytest.raises(ValueError, match="rake column"):
        draw(fitted(source), source.head(300), reference(), [100], rake_columns=["education"])


def test_an_unknown_sampling_mode_raises():
    source = bank_source()
    with pytest.raises(ValueError, match="sampling_mode"):
        draw(fitted(source), source.head(300), reference(), [100], sampling_mode="sideways")


# --- Default-shaped toy table: integer codes, positive class 1 ------------------------------------


def default_source(n=6000, seed=3, education_probs=(0.5, 0.3, 0.2), broken_share=0.2):
    """Default-shaped rows. ``broken_share`` of them have a negative payment amount."""
    rng = np.random.default_rng(seed)
    frame = {
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "SEX": rng.choice([1, 2], n, p=[0.4, 0.6]),
        "EDUCATION": rng.choice([1, 2, 3], n, p=education_probs),
        "MARRIAGE": rng.choice([1, 2, 3], n),
    }
    for month in range(1, 7):
        frame[f"PAY_{month}"] = rng.choice([-1, 0, 1, 2], n)
    for month in range(1, 7):
        frame[f"PAY_AMT{month}"] = rng.integers(0, 20_000, n)
    frame["y"] = rng.integers(0, 2, n)
    table = pd.DataFrame(frame)
    table.loc[rng.random(n) < broken_share, "PAY_AMT3"] = -100
    return table


def default_reference(education_probs=(0.3, 0.3, 0.4), n=2000, seed=9):
    """Real defaulters whose education shares differ from the generator's."""
    table = default_source(n, seed=seed, education_probs=education_probs, broken_share=0)
    table["y"] = 1
    return table


def draw_default(source, real, counts, **overrides):
    options = {
        "sampling_mode": "unconditional",
        "rule_filter": True,
        "rake_columns": [],
        "rules": DEFAULT_RULES,
        "pool_rows": 12_000,
        "seed": 42,
    }
    options.update(overrides)
    return sample_for_fill(fitted(source), source.head(300), real, counts, **options)


def test_default_rule_filter_removes_rows_with_a_negative_payment_amount():
    source = default_source(n=10_000)
    real = default_reference()
    off = draw_default(source, real, [1000], rule_filter=False)
    on = draw_default(source, real, [1000], rule_filter=True)
    assert (off[[f"PAY_AMT{m}" for m in range(1, 7)]] < 0).any(axis=1).any()
    assert not (on[[f"PAY_AMT{m}" for m in range(1, 7)]] < 0).any(axis=1).any()
    assert rule_row_filter(DEFAULT_RULES)(on).all()


def test_default_raking_brings_education_shares_to_the_reference():
    # The generator over-produces education 1 and under-produces education 3. A 4,000-row draw
    # keeps the sampling noise (about 1 point) small enough to check the 2-point line.
    source = default_source(n=70_000, broken_share=0, education_probs=(0.4, 0.3, 0.3))
    real = default_reference(education_probs=(0.3, 0.3, 0.4))
    plain = draw_default(source, real, [4000], pool_rows=25_000)
    raked = draw_default(source, real, [4000], pool_rows=25_000, rake_columns=["EDUCATION"])
    want = shares(real, "EDUCATION")
    assert (shares(plain, "EDUCATION") - want).abs().max() > 0.05
    assert (shares(raked, "EDUCATION") - want).abs().max() < 0.02


def test_integer_codes_are_raked_by_value_and_only_continuous_columns_are_binned():
    source = default_source(n=3000)
    real = default_reference()
    real["day"] = np.arange(len(real)) % 31  # 31 distinct values: continuous
    pool = source.assign(day=np.arange(len(source)) % 31)
    _, _, names = with_rake_bins(pool, real, ["SEX", "EDUCATION", "MARRIAGE", "day"])
    assert names == ["SEX", "EDUCATION", "MARRIAGE", "__bin_day"]
