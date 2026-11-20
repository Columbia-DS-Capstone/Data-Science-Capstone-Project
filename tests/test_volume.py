"""Validation-only fill selection and aligned saved-score loading."""

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools import volume
from agentic_gan.tools.evaluation import BootstrapResult


@pytest.fixture
def scores():
    rng = np.random.default_rng(17)
    y = np.tile([0, 1], 120)
    base = rng.random(len(y))
    return y, {0.0: base, 0.01: y + rng.normal(0, 0.1, len(y)), 0.1: base}


def test_best_reliable_level_is_chosen(scores):
    y, values = scores
    decision = volume.pick_fill(y, values, arm="ctgan", n_boot=80)
    assert decision.fill_level == 0.01
    assert decision.lo > 0
    assert [row["fill_level"] for row in decision.evidence] == [0.01, 0.1]
    assert "95% CI" in decision.reason


def test_no_reliable_lift_returns_cleaned(scores):
    y, values = scores
    decision = volume.pick_fill(y, {0.0: values[0.0], 0.1: values[0.0]}, arm="smotenc", n_boot=80)
    assert (decision.fill_level, decision.lift, decision.lo, decision.hi) == (0, 0, 0, 0)
    assert "0% (cleaned)" in decision.reason


def test_equal_lifts_choose_smaller_fill(scores):
    y, values = scores
    values[0.025] = values[0.01]
    values = dict(reversed(list(values.items())))
    assert volume.pick_fill(y, values, arm="ctgan", n_boot=80).fill_level == 0.01


def test_negative_lift_is_rejected(scores):
    y, values = scores
    decision = volume.pick_fill(y, {0.0: values[0.01], 0.1: 1 - y}, arm="ctgan", n_boot=80)
    assert decision.fill_level == 0
    assert decision.evidence[0]["lift"] < 0


def test_largest_lift_must_also_have_positive_lower_bound(monkeypatch):
    def bootstrap(y, base, other, **kwargs):
        return {
            0: BootstrapResult(0, 0, 0, 10),
            1: BootstrapResult(0.2, -0.01, 0.3, 10),
            2: BootstrapResult(0.1, 0.01, 0.2, 10),
            3: BootstrapResult(0.05, 0, 0.1, 10),
        }[other[0]]

    monkeypatch.setattr(volume, "paired_bootstrap", bootstrap)
    decision = volume.pick_fill([0, 1], {0.0: [0], 0.01: [1], 0.05: [2], 0.1: [3]}, arm="ctgan")
    assert decision.fill_level == 0.05


def test_missing_cleaned_reference_raises(scores):
    y, values = scores
    with pytest.raises(ValueError, match="0.0"):
        volume.pick_fill(y, {0.1: values[0.1]}, arm="ctgan")


def test_same_seed_reproduces_decision(scores):
    y, values = scores
    first = volume.pick_fill(y, values, arm="ctgan", n_boot=80, seed=19)
    assert first == volume.pick_fill(y, values, arm="ctgan", n_boot=80, seed=19)


@pytest.mark.parametrize("level", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_levels_raise(scores, level):
    y, values = scores
    with pytest.raises(ValueError, match="fill levels"):
        volume.pick_fill(y, {0.0: values[0.0], level: values[0.0]}, arm="ctgan")


def test_saved_scores_are_validation_only_and_row_aligned(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    pd.DataFrame(
        {
            "dataset": ["bank_marketing"] * 3,
            "seed": [42] * 3,
            "arm": ["cleaned", "ctgan", "smotenc"],
            "fill_level": [0, 0.01, 0.025],
        }
    ).to_csv(tmp_path / "models" / "metrics.csv", index=False)

    def load(folder, metrics, *, arm, fill_level, seed, split):
        assert split == "validation"
        return pd.DataFrame(
            {
                "source_row": [5, 8, 9, 10],
                "y_true": [0, 1, 0, 1],
                "probability": [0.1, 0.8, 0.2, 0.9],
            }
        )

    monkeypatch.setattr(volume, "load_selected_scores", load)
    decisions = volume.decisions_from_run(tmp_path, "bank_marketing", n_boot=10)
    assert [item.arm for item in decisions] == ["ctgan", "smotenc"]
    assert all(item.fill_level == 0 for item in decisions)

    def misaligned(*args, **kwargs):
        table = load(*args, **kwargs)
        if kwargs["arm"] != "cleaned":
            table.loc[0, "source_row"] = 99
        return table

    monkeypatch.setattr(volume, "load_selected_scores", misaligned)
    with pytest.raises(ValueError, match="align"):
        volume.decisions_from_run(tmp_path, "bank_marketing", n_boot=10)
