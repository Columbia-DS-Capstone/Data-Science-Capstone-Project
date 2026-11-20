"""Tests for tools/timegan.py.

Step 1: the data layer (step columns, layout, scaling, decoding, validation, save and load).
The toy table has the shape of the cleaned Default training split without ``source_row``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.quality_fidelity import (
    DEFAULT_PAY_CODES,
    DEFAULT_RULES,
    rule_violation_rate,
)
from agentic_gan.tools.selection import DEFAULT_WIDE_TEMPORAL

ROOT = Path(__file__).resolve().parents[2]
REAL_TRAIN = ROOT / "data" / "interim" / "default" / "train_cleaned.csv"

STATIC = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY = [f"PAY_{month}" for month in range(1, 7)]
BILL = [f"BILL_AMT{month}" for month in range(1, 7)]
PAID = [f"PAY_AMT{month}" for month in range(1, 7)]
TARGET = "y"
COLUMNS = [*STATIC, *PAY, *BILL, *PAID, TARGET]
STATIC_CODES = {"SEX": {1, 2}, "EDUCATION": set(range(7)), "MARRIAGE": set(range(4))}
ROWS = 300


# ---------------------------------------------------------------------------
# Module accessor (lazy import keeps torch out of the test-collection phase)
# ---------------------------------------------------------------------------

def _timegan():
    from agentic_gan.tools import timegan
    return timegan


def _default_steps() -> list[list[str]]:
    return _timegan().step_columns_from_wide(DEFAULT_WIDE_TEMPORAL, 6, newest_first=True)


def _violations(frame: pd.DataFrame) -> dict[str, float]:
    return {rule.name: rule_violation_rate(frame, rule) for rule in DEFAULT_RULES}


def _make_toy(rows: int = ROWS, seed: int = 7) -> pd.DataFrame:
    """A seeded Default-shaped table: real column order, all int64, 22% positives."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {
        "LIMIT_BAL": rng.integers(1, 81, rows) * 10_000,
        "SEX": rng.choice([1, 2], rows),
        "EDUCATION": rng.integers(0, 7, rows),
        "MARRIAGE": rng.integers(0, 4, rows),
        "AGE": rng.integers(21, 76, rows),
    }
    for name in PAY:
        data[name] = rng.integers(-2, 9, rows)
    for name in BILL:
        data[name] = rng.integers(-5_000, 200_000, rows)
    for name in PAID:  # about a fifth of the months have no payment, as in the real table
        data[name] = np.where(rng.random(rows) < 0.2, 0, rng.integers(1, 50_000, rows))
    positives = round(0.22 * rows)
    data[TARGET] = rng.permutation(np.r_[np.ones(positives), np.zeros(rows - positives)])
    return pd.DataFrame(data)[COLUMNS].astype("int64")


@pytest.fixture(scope="module")
def toy_df() -> pd.DataFrame:
    frame = _make_toy()
    assert list(frame.columns) == COLUMNS
    assert (frame.dtypes == "int64").all()
    assert frame[TARGET].sum() == 66
    assert frame[PAID].min().min() == 0
    assert _violations(frame) == {rule.name: 0.0 for rule in DEFAULT_RULES}
    return frame


@pytest.fixture(scope="module")
def layout(toy_df):
    return _timegan().fit_layout(toy_df, TARGET, STATIC, _default_steps())


# ---------------------------------------------------------------------------
# step_columns_from_wide
# ---------------------------------------------------------------------------

class TestStepColumns:
    def test_default_steps_run_april_to_september(self):
        steps = _default_steps()
        assert len(steps) == 6
        assert steps[0] == ["PAY_6", "BILL_AMT6", "PAY_AMT6"]
        assert steps[-1] == ["PAY_1", "BILL_AMT1", "PAY_AMT1"]
        assert [step[0] for step in steps] == [f"PAY_{month}" for month in range(6, 0, -1)]

    def test_oldest_first_input_keeps_its_order(self):
        steps = _timegan().step_columns_from_wide(
            ["a1", "a2", "b1", "b2"], 2, newest_first=False
        )
        assert steps == [["a1", "b1"], ["a2", "b2"]]

    def test_every_wide_column_is_used_once(self):
        flat = [name for step in _default_steps() for name in step]
        assert sorted(flat) == sorted(DEFAULT_WIDE_TEMPORAL)

    @pytest.mark.parametrize(
        ("wide", "n_steps", "message"),
        [
            (["a1", "a2", "b1"], 2, "equal width"),
            ([], 2, "equal width"),
            (["a1", "a2"], 1, "at least 2"),
            (["a1", "a1", "b1", "b2"], 2, "more than once"),
        ],
    )
    def test_bad_input_raises(self, wide, n_steps, message):
        with pytest.raises(ValueError, match=message):
            _timegan().step_columns_from_wide(wide, n_steps, newest_first=True)


# ---------------------------------------------------------------------------
# fit_layout
# ---------------------------------------------------------------------------

class TestFitLayout:
    def test_sizes(self, layout):
        assert (layout.n_steps, layout.n_step_features, layout.n_features) == (6, 3, 9)
        assert layout.features == (
            "step_feature_0", "step_feature_1", "step_feature_2", *STATIC, TARGET
        )

    def test_records_column_order_and_dtypes(self, layout, toy_df):
        assert list(layout.columns) == COLUMNS
        assert layout.dtypes == {name: "int64" for name in COLUMNS}
        assert layout.step_columns == tuple(tuple(step) for step in _default_steps())

    def test_step_feature_range_is_taken_over_all_steps(self, layout, toy_df):
        for position, group in enumerate((PAY, BILL, PAID)):
            assert layout.minimum[position] == toy_df[group].min().min()
            assert layout.maximum[position] == toy_df[group].max().max()

    def test_static_and_target_ranges(self, layout, toy_df):
        for offset, name in enumerate([*STATIC, TARGET]):
            assert layout.minimum[3 + offset] == toy_df[name].min()
            assert layout.maximum[3 + offset] == toy_df[name].max()

    def test_coded_features(self, layout):
        assert layout.coded == ("step_feature_0", "SEX", "EDUCATION", "MARRIAGE", TARGET)

    def test_status_codes_are_the_union_over_steps(self, layout, toy_df):
        union = sorted({int(v) for v in np.unique(toy_df[PAY].to_numpy())})
        assert [int(v) for v in layout.valid_values[0]] == union
        assert set(union) <= DEFAULT_PAY_CODES

    def test_every_feature_is_integer_valued(self, layout):
        assert all(layout.integer_valued)

    def test_target_is_coded_even_with_many_values(self, toy_df):
        frame = toy_df.copy()
        frame[TARGET] = np.arange(len(frame))
        fitted = _timegan().fit_layout(frame, TARGET, STATIC, _default_steps())
        assert len(fitted.valid_values[-1]) == len(frame)

    def test_does_not_change_the_frame(self, toy_df):
        before = toy_df.copy()
        _timegan().fit_layout(toy_df, TARGET, STATIC, _default_steps())
        pd.testing.assert_frame_equal(toy_df, before)


# ---------------------------------------------------------------------------
# to_sequences and from_sequences
# ---------------------------------------------------------------------------

class TestToSequences:
    def test_shape_dtype_and_range(self, layout, toy_df):
        array = _timegan().to_sequences(toy_df, layout)
        assert array.shape == (ROWS, 6, 9)
        assert array.dtype == np.float32
        assert array.min() == 0.0
        assert array.max() == 1.0

    def test_every_feature_spans_zero_to_one(self, layout, toy_df):
        array = _timegan().to_sequences(toy_df, layout)
        flat = array.reshape(-1, 9)
        assert (flat.min(axis=0) == 0.0).all()
        assert (flat.max(axis=0) == 1.0).all()

    def test_static_and_target_repeat_at_every_step(self, layout, toy_df):
        array = _timegan().to_sequences(toy_df, layout)
        for step in range(1, 6):
            np.testing.assert_array_equal(array[:, step, 3:], array[:, 0, 3:])

    def test_first_step_is_the_oldest_month(self, layout, toy_df):
        array = _timegan().to_sequences(toy_df, layout)
        low, high = layout.minimum[1], layout.maximum[1]
        april = (toy_df["BILL_AMT6"] - low) / (high - low)
        september = (toy_df["BILL_AMT1"] - low) / (high - low)
        np.testing.assert_allclose(array[:, 0, 1], april, atol=1e-6)
        np.testing.assert_allclose(array[:, 5, 1], september, atol=1e-6)

    def test_missing_column_raises(self, layout, toy_df):
        with pytest.raises(ValueError, match="AGE"):
            _timegan().to_sequences(toy_df.drop(columns="AGE"), layout)


class TestRoundTrip:
    def test_toy_table_round_trips_exactly(self, layout, toy_df):
        tg = _timegan()
        back = tg.from_sequences(tg.to_sequences(toy_df, layout), layout)
        pd.testing.assert_frame_equal(back, toy_df)

    @pytest.mark.skipif(not REAL_TRAIN.exists(), reason="Default training split not present")
    def test_real_training_split_round_trips_exactly(self):
        tg = _timegan()
        real = pd.read_csv(REAL_TRAIN).drop(columns="source_row")
        fitted = tg.fit_layout(real, TARGET, STATIC, _default_steps())
        back = tg.from_sequences(tg.to_sequences(real, fitted), fitted)
        pd.testing.assert_frame_equal(back, real)

    def test_constant_feature_scales_to_zero_and_decodes_to_itself(self, toy_df):
        tg = _timegan()
        frame = toy_df.copy()
        frame["AGE"] = 40
        fitted = tg.fit_layout(frame, TARGET, STATIC, _default_steps())
        array = tg.to_sequences(frame, fitted)
        age = fitted.features.index("AGE")
        assert (array[:, :, age] == 0.0).all()
        assert (tg.from_sequences(array, fitted)["AGE"] == 40).all()

    def test_float_column_keeps_its_dtype(self, toy_df):
        tg = _timegan()
        frame = toy_df.copy()
        frame["LIMIT_BAL"] = frame["LIMIT_BAL"] / 7.0
        fitted = tg.fit_layout(frame, TARGET, STATIC, _default_steps())
        assert not fitted.integer_valued[fitted.features.index("LIMIT_BAL")]
        back = tg.from_sequences(tg.to_sequences(frame, fitted), fitted)
        assert back["LIMIT_BAL"].dtype == np.float64
        np.testing.assert_allclose(back["LIMIT_BAL"], frame["LIMIT_BAL"], rtol=1e-5)
        pd.testing.assert_frame_equal(
            back.drop(columns="LIMIT_BAL"), frame.drop(columns="LIMIT_BAL")
        )


@pytest.fixture(scope="module")
def decoded(layout) -> pd.DataFrame:
    """500 rows decoded from uniform random sequences."""
    rng = np.random.default_rng(11)
    return _timegan().from_sequences(rng.uniform(0, 1, (500, 6, 9)), layout)


class TestDecodeArbitraryValues:
    """What the generator will produce: any numbers, not rows of the training table."""

    def test_columns_order_and_dtypes_match_training(self, decoded, toy_df):
        assert list(decoded.columns) == COLUMNS
        assert decoded.dtypes.to_dict() == toy_df.dtypes.to_dict()
        assert len(decoded) == 500
        assert not decoded.isna().any().any()

    def test_static_values_are_valid_categories(self, decoded, toy_df):
        for name, codes in STATIC_CODES.items():
            assert set(decoded[name]) <= codes
            assert set(decoded[name]) <= set(toy_df[name])
        assert set(decoded[TARGET]) <= {0, 1}

    def test_pay_status_is_inside_the_default_codes(self, decoded):
        for name in PAY:
            assert set(decoded[name]) <= DEFAULT_PAY_CODES

    def test_payment_amounts_are_not_negative(self, decoded):
        assert (decoded[PAID] >= 0).all().all()

    def test_no_default_rule_is_broken(self, decoded):
        assert _violations(decoded) == {rule.name: 0.0 for rule in DEFAULT_RULES}

    def test_continuous_values_stay_in_the_training_range(self, decoded, toy_df):
        for name in ("LIMIT_BAL", "AGE"):
            assert decoded[name].between(toy_df[name].min(), toy_df[name].max()).all()
        for group in (BILL, PAID):
            low, high = toy_df[group].min().min(), toy_df[group].max().max()
            assert ((decoded[group] >= low) & (decoded[group] <= high)).all().all()

    def test_values_outside_zero_one_are_clipped(self, layout, toy_df):
        rng = np.random.default_rng(12)
        frame = _timegan().from_sequences(rng.uniform(-3, 4, (200, 6, 9)), layout)
        assert _violations(frame) == {rule.name: 0.0 for rule in DEFAULT_RULES}
        for name in PAY:
            assert set(frame[name]) <= DEFAULT_PAY_CODES
        assert frame["AGE"].between(toy_df["AGE"].min(), toy_df["AGE"].max()).all()

    def test_static_and_target_come_from_the_first_step(self, layout, toy_df):
        array = np.ones((4, 6, 9), dtype=np.float32)
        array[:, 0, :] = 0.0
        frame = _timegan().from_sequences(array, layout)
        for name in [*STATIC, TARGET]:
            assert (frame[name] == toy_df[name].min()).all()
        # The first step is April (suffix 6); the later steps hold the maximum.
        assert (frame["PAY_6"] == toy_df[PAY].min().min()).all()
        assert (frame["PAY_1"] == toy_df[PAY].max().max()).all()
        assert (frame["PAY_AMT6"] == 0).all()
        assert (frame["PAY_AMT1"] == toy_df[PAID].max().max()).all()

    def test_coded_value_snaps_to_the_nearest_code(self, toy_df):
        tg = _timegan()
        frame = toy_df.copy()
        frame["EDUCATION"] = np.where(np.arange(len(frame)) % 2 == 0, 0, 6)
        fitted = tg.fit_layout(frame, TARGET, STATIC, _default_steps())
        education = fitted.features.index("EDUCATION")
        array = np.zeros((3, 6, 9), dtype=np.float32)
        array[:, 0, education] = [0.2, 0.8, 0.5]
        # Valid codes are 0 and 6; the tie at 0.5 (value 3) goes to the lower code.
        assert tg.from_sequences(array, fitted)["EDUCATION"].tolist() == [0, 6, 0]

    def test_zero_rows_gives_an_empty_frame_with_the_columns(self, layout, toy_df):
        frame = _timegan().from_sequences(np.empty((0, 6, 9)), layout)
        assert len(frame) == 0
        assert list(frame.columns) == COLUMNS
        assert frame.dtypes.to_dict() == toy_df.dtypes.to_dict()

    def test_wrong_shape_raises(self, layout):
        with pytest.raises(ValueError, match="shape"):
            _timegan().from_sequences(np.zeros((5, 6, 8)), layout)
        with pytest.raises(ValueError, match="shape"):
            _timegan().from_sequences(np.zeros((5, 9)), layout)

    def test_non_finite_values_raise(self, layout):
        array = np.zeros((5, 6, 9))
        array[2, 3, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _timegan().from_sequences(array, layout)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestFrameValidation:
    def _fit(self, frame, target=TARGET, static=None, steps=None):
        return _timegan().fit_layout(
            frame,
            target,
            STATIC if static is None else static,
            _default_steps() if steps is None else steps,
        )

    def test_target_missing(self, toy_df):
        with pytest.raises(ValueError, match="target_col 'label' is not in train_df"):
            self._fit(toy_df, target="label")

    def test_listed_column_missing(self, toy_df):
        with pytest.raises(ValueError, match=r"not in train_df: \['GHOST'\]"):
            self._fit(toy_df, static=[*STATIC, "GHOST"])

    def test_column_in_both_lists(self, toy_df):
        with pytest.raises(ValueError, match=r"more than once.*\['PAY_1'\]"):
            self._fit(toy_df, static=[*STATIC, "PAY_1"])

    def test_target_listed_as_a_feature(self, toy_df):
        with pytest.raises(ValueError, match="must not be listed"):
            self._fit(toy_df, static=[*STATIC, TARGET])

    def test_uncovered_column_is_named(self, toy_df):
        frame = toy_df.assign(source_row=np.arange(len(toy_df)))
        with pytest.raises(ValueError, match=r"neither.*\['source_row'\]"):
            self._fit(frame)

    def test_is_synthetic_column_is_refused(self, toy_df):
        with pytest.raises(ValueError, match="is_synthetic"):
            self._fit(toy_df.assign(is_synthetic=0))

    def test_missing_values(self, toy_df):
        frame = toy_df.astype({"AGE": "float64"})
        frame.loc[3, "AGE"] = np.nan
        with pytest.raises(ValueError, match=r"missing or non-finite values: \['AGE'\]"):
            self._fit(frame)

    def test_infinite_values(self, toy_df):
        frame = toy_df.astype({"BILL_AMT2": "float64"})
        frame.loc[3, "BILL_AMT2"] = np.inf
        with pytest.raises(ValueError, match=r"non-finite values: \['BILL_AMT2'\]"):
            self._fit(frame)

    def test_non_numeric_column(self, toy_df):
        frame = toy_df.astype({"SEX": "str"})
        with pytest.raises(ValueError, match=r"numeric columns.*\['SEX'\]"):
            self._fit(frame)

    def test_step_lists_of_unequal_length(self, toy_df):
        steps = _default_steps()
        steps[2] = steps[2][:2]
        with pytest.raises(ValueError, match="same number of columns"):
            self._fit(toy_df, steps=steps)

    def test_fewer_than_two_steps(self, toy_df):
        with pytest.raises(ValueError, match="at least 2 steps, got 1"):
            self._fit(toy_df, steps=_default_steps()[:1])

    def test_fewer_than_two_rows(self, toy_df):
        with pytest.raises(ValueError, match="at least 2 rows"):
            self._fit(toy_df.head(1))


class TestArgumentValidation:
    def test_valid_settings_pass(self):
        tg = _timegan()
        for hidden_dim in (24, 48, 64):
            tg._validate_args(1, 2, hidden_dim, "cpu")
        tg._validate_args(300, 128, 48, "auto")

    @pytest.mark.parametrize("hidden_dim", [23, 65, 0, 48.0, True])
    def test_hidden_dim_outside_24_to_64(self, hidden_dim):
        with pytest.raises(ValueError, match="hidden_dim must be a whole number from 24 to 64"):
            _timegan()._validate_args(2, 128, hidden_dim, "cpu")

    @pytest.mark.parametrize("epochs", [0, -1, 2.0, True])
    def test_epochs_below_one(self, epochs):
        with pytest.raises(ValueError, match="epochs"):
            _timegan()._validate_args(epochs, 128, 48, "cpu")

    @pytest.mark.parametrize("batch_size", [1, 0, 64.0])
    def test_batch_size_below_two(self, batch_size):
        with pytest.raises(ValueError, match="batch_size"):
            _timegan()._validate_args(2, batch_size, 48, "cpu")

    @pytest.mark.parametrize("device", ["mps", "gpu", "", None])
    def test_unknown_device(self, device):
        with pytest.raises(ValueError, match="device"):
            _timegan()._validate_args(2, 128, 48, device)


# ---------------------------------------------------------------------------
# Defaults, save and load
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_settings_outside_the_signature(self):
        defaults = _timegan().DEFAULTS
        assert defaults == {"num_layers": 3, "gamma": 1.0, "lr": 1e-3, "noise_dim": None}

    def test_coded_rule_is_the_ctgan_sampling_constant(self):
        from agentic_gan.tools import ctgan_sampling

        assert _timegan().CODED_MAX_VALUES is ctgan_sampling.CODED_MAX_VALUES


class TestSaveLoad:
    def test_round_trip_keeps_weights_and_layout(self, layout, tmp_path):
        import torch
        import torch.nn as nn

        tg = _timegan()
        torch.manual_seed(0)
        networks = [nn.GRU(9, 24, batch_first=True) for _ in range(5)]
        model = tg.TrainedTimeGAN(
            *networks,
            layout=layout,
            target_col=TARGET,
            static_columns=list(STATIC),
            step_columns=_default_steps(),
            columns=list(COLUMNS),
            dtypes=dict(layout.dtypes),
            params={"epochs": 2, "seed": 42},
        )
        path = tmp_path / "nested" / "timegan.pkl"
        tg.save_model(model, path)
        loaded = tg.load_model(path)

        assert isinstance(loaded, tg.TrainedTimeGAN)
        assert loaded.layout == layout
        assert loaded.params == model.params
        assert loaded.step_columns == model.step_columns
        assert loaded.columns == COLUMNS
        for before, after in zip(model.networks, loaded.networks, strict=True):
            for key, value in before.state_dict().items():
                assert torch.equal(value, after.state_dict()[key])
