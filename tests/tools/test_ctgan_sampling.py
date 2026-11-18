"""The generator recipe and the V2 sampling steps, with a stand-in for the fitted CTGAN.

The stand-in draws rows from a fixed table, so the real ``sample_synthetic`` runs (target match,
rule filter, copy check) without training a network.
"""

import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.ctgan import TrainedCTGAN
from agentic_gan.tools.ctgan_sampling import (
    GeneratorRecipe,
    recipe_for,
    rule_row_filter,
    sample_for_fill,
)
from agentic_gan.tools.quality_fidelity import BANK_RULES

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


@pytest.mark.parametrize("dataset", ["default", "polish", "ieee_cis", "anything_else"])
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


# --- V3 selection modes ---------------------------------------------------------------------------

MODULE = "agentic_gan.tools.ctgan_sampling"


def selection_source():
    """Bank-shaped rows; half have an obviously synthetic `balance` (far above the real ones)."""
    source = bank_source(n=12_000, broken_share=0)
    shifted = np.random.default_rng(5).random(len(source)) < 0.5
    source.loc[shifted, "balance"] += 5000
    return source


def fake_boundary(monkeypatch, probability):
    """Replace the LightGBM scorer (it must not load beside torch) with ``probability(frame)``."""
    seen = []

    def scorer(train, synthetic, *, target, positive, seed):
        seen.append((train, synthetic))
        return np.asarray(probability(synthetic), dtype=float)

    monkeypatch.setattr(f"{MODULE}.boundary_probability", scorer)
    return seen


def in_band(frame):
    """A stand-in model: the probability rises with `balance`, from 0 at 500 to 1 at 2000."""
    return np.clip((frame["balance"].to_numpy() - 500) / 1500, 0, 1)


def test_recipe_selection_defaults_and_validation():
    recipe = recipe_for("bank_marketing")
    assert recipe.selection_mode == "none"
    assert recipe.pool_multiplier == 5
    assert (recipe.boundary_low, recipe.boundary_high) == (0.10, 0.60)
    with pytest.raises(ValueError, match="selection_mode"):
        GeneratorRecipe(300, 500, 10, 1, "conditional", False, [], selection_mode="best")
    with pytest.raises(ValueError, match="pool_multiplier"):
        GeneratorRecipe(300, 500, 10, 1, "conditional", False, [], pool_multiplier=0)
    with pytest.raises(ValueError, match="boundary_low"):
        GeneratorRecipe(300, 500, 10, 1, "conditional", False, [], boundary_low=0.7)


def test_none_returns_the_same_rows_as_v2():
    source = bank_source()
    args = (fitted(source), source.head(300), reference(), [200])
    plain = draw(*args, rake_columns=["job"], seed=11)
    explicit = draw(*args, rake_columns=["job"], seed=11, selection_mode="none")
    pd.testing.assert_frame_equal(plain, explicit)


@pytest.mark.parametrize("mode", ["realism", "boundary"])
def test_each_selection_mode_returns_the_largest_fill_without_train_copies(mode, monkeypatch):
    source = selection_source()
    train = source.head(1500)
    if mode == "boundary":
        fake_boundary(monkeypatch, lambda frame: np.full(len(frame), 0.3))
    rows = draw(
        fitted(source),
        train,
        reference(),
        [50, 100, 200],
        rake_columns=["job"],
        selection_mode=mode,
        seed=3,
    )
    assert len(rows) == 200
    assert (rows["y"] == 1).all()
    real = set(map(tuple, train.itertuples(index=False)))
    made = set(map(tuple, rows[list(source.columns)].itertuples(index=False)))
    assert not made & real


@pytest.mark.parametrize("mode", ["realism", "boundary"])
def test_kept_rows_come_from_the_pool_of_pool_multiplier_times_the_largest_fill(mode, monkeypatch):
    source = selection_source()
    pools = []
    import agentic_gan.tools.ctgan as ctgan_tool

    original = ctgan_tool.sample_synthetic

    def spy(model, n, **kwargs):
        pool = original(model, n, **kwargs)
        pools.append((n, pool))
        return pool

    monkeypatch.setattr(ctgan_tool, "sample_synthetic", spy)
    if mode == "boundary":
        fake_boundary(monkeypatch, lambda frame: np.full(len(frame), 0.3))
    rows = draw(
        fitted(source),
        source.head(300),
        reference(),
        [100, 200],
        selection_mode=mode,
        pool_multiplier=4,
        seed=3,
    )
    ((asked, pool),) = pools
    assert asked == 4 * 200
    columns = list(source.columns)
    assert set(map(tuple, rows[columns].itertuples(index=False))) <= set(
        map(tuple, pool[columns].itertuples(index=False))
    )


def test_selection_models_are_fitted_on_the_given_real_rows_only(monkeypatch):
    source = selection_source()
    train = source.head(1500)
    real = reference()
    boundary = fake_boundary(monkeypatch, lambda frame: np.full(len(frame), 0.3))
    realism = []
    original = __import__(MODULE, fromlist=["x"]).synthetic_probability

    def spy(real_rows, synthetic, *, target, seed):
        realism.append(real_rows)
        return original(real_rows, synthetic, target=target, seed=seed)

    monkeypatch.setattr(f"{MODULE}.synthetic_probability", spy)
    draw(fitted(source), train, real, [100], selection_mode="boundary", seed=3)
    draw(fitted(source), train, real, [100], selection_mode="realism", seed=3)
    assert boundary[0][0] is train  # the boundary model sees the real training table
    assert realism[0] is real  # the realism model sees the real minority rows


def test_realism_keeps_the_more_real_looking_rows():
    source = selection_source()
    plain = draw(fitted(source), source.head(300), reference(), [400], pool_rows=2000, seed=3)
    kept = draw(
        fitted(source),
        source.head(300),
        reference(),
        [400],
        selection_mode="realism",
        pool_multiplier=5,
        seed=3,
    )
    assert (plain["balance"] > 3000).mean() > 0.3  # about half the pool is obviously synthetic
    assert (kept["balance"] > 3000).mean() < 0.1


def test_boundary_keeps_only_rows_inside_the_bounds(monkeypatch):
    source = selection_source()
    fake_boundary(monkeypatch, in_band)
    rows = draw(
        fitted(source),
        source.head(300),
        reference(),
        [100],
        selection_mode="boundary",
        boundary_low=0.10,
        boundary_high=0.60,
        seed=3,
    )
    probability = in_band(rows)
    assert ((probability >= 0.10) & (probability <= 0.60)).all()


def test_boundary_with_too_few_rows_in_the_band_raises(monkeypatch):
    source = selection_source()
    fake_boundary(monkeypatch, lambda frame: np.full(len(frame), 0.95))
    with pytest.raises(ValueError, match="predicted probability"):
        draw(fitted(source), source.head(300), reference(), [100], selection_mode="boundary")


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="needs lightgbm")
def test_the_real_boundary_scorer_ranks_rows_by_the_response():
    # Run in its own process: LightGBM and torch (loaded by this file) crash together on macOS.
    script = textwrap.dedent(
        """
        import numpy as np, pandas as pd
        from agentic_gan.tools.ctgan_sampling import boundary_probability
        rng = np.random.default_rng(0)
        x = rng.normal(size=3000)
        train = pd.DataFrame({"x": x, "c": rng.choice(["a", "b"], 3000),
                              "y": (x + rng.normal(scale=0.5, size=3000) > 0.5).astype(int)})
        synth = pd.DataFrame({"x": np.r_[np.full(50, -2.0), np.full(50, 2.0)],
                              "c": ["a"] * 100, "is_synthetic": 1, "y": 1})
        p = boundary_probability(train, synth, target="y", positive=1, seed=0)
        assert p.shape == (100,) and ((p >= 0) & (p <= 1)).all()
        assert p[50:].mean() > p[:50].mean() + 0.5, (p[:50].mean(), p[50:].mean())
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-800:]
