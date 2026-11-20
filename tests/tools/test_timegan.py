"""Tests for tools/timegan.py.

Step 1: the data layer (step columns, layout, scaling, decoding, validation, save and load).
Step 2: the five networks and the three-phase training loop.
Step 3: the sampler.
The toy table has the shape of the cleaned Default training split without ``source_row``.
"""

from __future__ import annotations

import inspect
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


# ===========================================================================
# Step 2: networks and training
# ===========================================================================

NETWORK_NAMES = ("embedder", "recovery", "supervisor", "generator", "discriminator")
LOSS_COLUMNS = ["Embedder Loss", "Supervised Loss", "Generator Loss", "Discriminator Loss"]


def _train(frame: pd.DataFrame, **kwargs):
    """Train a small TimeGAN quickly: 2 epochs per phase unless told otherwise."""
    settings = {"epochs": 2, "seed": 42}
    settings.update(kwargs)
    return _timegan().train_timegan(frame, TARGET, STATIC, _default_steps(), **settings)


def _weights(model) -> dict[str, object]:
    return {
        f"{name}.{key}": value.clone()
        for name in NETWORK_NAMES
        for key, value in getattr(model, name).state_dict().items()
    }


@pytest.fixture(scope="module")
def trained(toy_df, tmp_path_factory):
    """One 2-epoch model for the read-only tests, and the loss log it wrote."""
    log = tmp_path_factory.mktemp("timegan") / "nested" / "dir" / "losses.csv"
    return _train(toy_df, log_path=log), log


def _track_adam(monkeypatch):
    """Record every Adam optimiser created, in order, and how often each one steps."""
    import torch

    created: list = []
    steps: dict[int, int] = {}
    original_init = torch.optim.Adam.__init__
    original_step = torch.optim.Adam.step

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)
        steps[id(self)] = 0

    def tracked_step(self, closure=None):
        steps[id(self)] += 1
        return original_step(self, closure)

    monkeypatch.setattr(torch.optim.Adam, "__init__", tracked_init)
    monkeypatch.setattr(torch.optim.Adam, "step", tracked_step)
    return created, steps


class TestNetworks:
    def test_shapes_and_ranges(self):
        import torch

        tg = _timegan()
        torch.manual_seed(0)
        x = torch.rand(4, 6, 9)
        hidden = tg.Embedder(9, 48, 3)(x)
        assert hidden.shape == (4, 6, 48)
        assert (hidden > 0).all() and (hidden < 1).all()
        back = tg.Recovery(48, 9, 3)(hidden)
        assert back.shape == (4, 6, 9)
        assert (back > 0).all() and (back < 1).all()
        nxt = tg.Supervisor(48, 3)(hidden)
        assert nxt.shape == (4, 6, 48)
        assert (nxt > 0).all() and (nxt < 1).all()
        made = tg.SequenceGenerator(9, 48, 3)(torch.rand(4, 6, 9))
        assert made.shape == (4, 6, 48)
        assert (made > 0).all() and (made < 1).all()
        assert tg.Discriminator(48, 3)(hidden).shape == (4, 6)

    def test_every_network_is_a_batch_first_gru(self):
        import torch.nn as nn

        tg = _timegan()
        nets = [
            tg.Embedder(9, 24, 3), tg.Recovery(24, 9, 3), tg.Supervisor(24, 3),
            tg.SequenceGenerator(9, 24, 3), tg.Discriminator(24, 3),
        ]
        for net in nets:
            assert isinstance(net.gru, nn.GRU)
            assert net.gru.batch_first
            assert net.gru.hidden_size == 24

    def test_supervisor_has_one_layer_fewer(self):
        tg = _timegan()
        assert tg.Embedder(9, 24, 3).gru.num_layers == 3
        assert tg.Supervisor(24, 3).gru.num_layers == 2
        assert tg.Supervisor(24, 1).gru.num_layers == 1

    def test_discriminator_returns_logits_not_probabilities(self):
        import torch

        net = _timegan().Discriminator(24, 3)
        with torch.no_grad():
            net.linear.bias.fill_(50.0)
            out = net(torch.rand(4, 6, 24))
        assert (out > 1).all()


class TestTrainTimeGAN:
    def test_signature_is_the_one_in_the_issue(self):
        parameters = list(inspect.signature(_timegan().train_timegan).parameters.values())
        assert [p.name for p in parameters] == [
            "train_df", "target_col", "static_columns", "step_columns",
            "epochs", "batch_size", "hidden_dim", "seed", "device", "log_path",
        ]
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in parameters[:4])
        assert all(p.kind is p.KEYWORD_ONLY for p in parameters[4:])
        assert parameters[4].default is inspect.Parameter.empty
        assert {p.name: p.default for p in parameters[5:]} == {
            "batch_size": 128, "hidden_dim": 48, "seed": 42, "device": "cpu", "log_path": None,
        }

    def test_returns_a_trained_model_with_the_five_networks(self, trained):
        tg = _timegan()
        model, _ = trained
        assert isinstance(model, tg.TrainedTimeGAN)
        assert isinstance(model.embedder, tg.Embedder)
        assert isinstance(model.recovery, tg.Recovery)
        assert isinstance(model.supervisor, tg.Supervisor)
        assert isinstance(model.generator, tg.SequenceGenerator)
        assert isinstance(model.discriminator, tg.Discriminator)

    def test_six_loss_rows_in_phase_order(self, trained):
        history = trained[0].loss_history
        assert list(history.columns) == ["Phase", "Epoch", *LOSS_COLUMNS]
        assert history["Phase"].tolist() == [
            "autoencoder", "autoencoder", "supervised", "supervised", "joint", "joint",
        ]
        assert history["Epoch"].tolist() == [1, 2, 1, 2, 1, 2]

    def test_a_loss_a_phase_does_not_compute_is_nan(self, trained):
        history = trained[0].loss_history.set_index(["Phase", "Epoch"])[LOSS_COLUMNS]
        applies = {
            "autoencoder": ["Embedder Loss"],
            "supervised": ["Supervised Loss"],
            "joint": LOSS_COLUMNS,
        }
        for (phase, _), row in history.iterrows():
            for column in LOSS_COLUMNS:
                if column in applies[phase]:
                    assert np.isfinite(row[column]), (phase, column)
                else:
                    assert np.isnan(row[column]), (phase, column)

    def test_log_file_is_written_and_matches_the_history(self, trained):
        model, log = trained
        assert log.exists()
        pd.testing.assert_frame_equal(pd.read_csv(log), model.loss_history)

    def test_one_epoch_gives_three_rows(self, toy_df):
        history = _train(toy_df, epochs=1).loss_history
        assert history["Phase"].tolist() == ["autoencoder", "supervised", "joint"]

    def test_same_seed_gives_identical_history_and_weights(self, toy_df):
        import torch

        first, second = _train(toy_df, seed=5), _train(toy_df, seed=5)
        pd.testing.assert_frame_equal(first.loss_history, second.loss_history)
        a, b = _weights(first), _weights(second)
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), key

    def test_different_seed_gives_different_history_and_weights(self, toy_df):
        import torch

        first, second = _train(toy_df, seed=5), _train(toy_df, seed=6)
        assert not first.loss_history[LOSS_COLUMNS].equals(second.loss_history[LOSS_COLUMNS])
        a, b = _weights(first), _weights(second)
        for name in NETWORK_NAMES:
            keys = [key for key in a if key.startswith(f"{name}.")]
            assert any(not torch.equal(a[key], b[key]) for key in keys), name

    def test_input_frame_is_unchanged(self, toy_df):
        before = toy_df.copy()
        _train(toy_df, epochs=1)
        pd.testing.assert_frame_equal(toy_df, before)

    def test_bookkeeping(self, trained, toy_df, layout):
        model, _ = trained
        assert model.n_train_rows == ROWS
        assert model.train_seconds > 0
        assert model.layout == layout
        assert model.target_col == TARGET
        assert model.static_columns == STATIC
        assert model.step_columns == _default_steps()
        assert model.columns == COLUMNS
        assert model.dtypes == toy_df.dtypes.astype(str).to_dict()
        assert model.params == {
            "epochs": 2, "batch_size": 128, "hidden_dim": 48, "seed": 42, "device": "cpu",
            "num_layers": 3, "noise_dim": 9, "gamma": 1.0, "lr": 1e-3,
            "generator_steps": 2, "discriminator_threshold": 0.15, "steps_per_epoch": 2,
        }

    def test_networks_are_returned_in_eval_mode(self, trained):
        assert not any(network.training for network in trained[0].networks)

    def test_hidden_dim_sets_the_network_width(self, toy_df):
        for hidden_dim in (24, 64):
            model = _train(toy_df, epochs=1, hidden_dim=hidden_dim)
            assert model.embedder.gru.hidden_size == hidden_dim
            assert model.recovery.linear.out_features == 9
            assert model.generator.gru.input_size == 9

    def test_batch_size_larger_than_the_table_still_trains(self, toy_df):
        model = _train(toy_df.head(40), epochs=1, batch_size=256)
        assert model.params["steps_per_epoch"] == 1
        assert len(model.loss_history) == 3

    def test_training_lowers_the_losses_it_minimises(self, toy_df):
        """Seeded, so deterministic: 15 epochs is 30 steps per phase on the toy table."""
        history = _train(toy_df, epochs=15).loss_history.set_index(["Phase", "Epoch"])
        reconstruction = history.loc["autoencoder", "Embedder Loss"]
        next_step = history.loc["supervised", "Supervised Loss"]
        assert reconstruction.iloc[-1] < reconstruction.iloc[0]
        assert next_step.iloc[-1] < next_step.iloc[0]

    def test_each_update_steps_the_right_networks_the_right_number_of_times(
        self, toy_df, monkeypatch
    ):
        created, steps = _track_adam(monkeypatch)
        model = _train(toy_df)  # 300 rows, batch 128: 2 steps per epoch, 2 epochs per phase

        def ids(*names):
            return {id(p) for name in names for p in getattr(model, name).parameters()}

        def held(optimiser):
            return {id(p) for group in optimiser.param_groups for p in group["params"]}

        assert len(created) == 5
        autoencoder, supervisor, generator, embedder, discriminator = created
        assert held(autoencoder) == ids("embedder", "recovery")
        assert held(supervisor) == ids("supervisor")
        assert held(generator) == ids("generator", "supervisor")
        assert held(embedder) == ids("embedder", "recovery")
        assert held(discriminator) == ids("discriminator")
        # Joint phase: per step, two generator updates and two embedder updates, then one
        # discriminator update (its loss starts near 3 * log 2, well above the threshold).
        assert [steps[id(o)] for o in created] == [4, 4, 8, 8, 4]

    def test_discriminator_is_not_updated_at_or_below_the_threshold(self, toy_df, monkeypatch):
        tg = _timegan()
        created, steps = _track_adam(monkeypatch)
        monkeypatch.setattr(tg, "DISCRIMINATOR_THRESHOLD", 1e9)
        model = _train(toy_df)
        assert steps[id(created[4])] == 0
        assert [steps[id(o)] for o in created[:4]] == [4, 4, 8, 8]
        assert np.isfinite(model.loss_history["Discriminator Loss"].iloc[-1])

    @pytest.mark.parametrize(
        ("helper", "phase"),
        [
            ("_reconstruction_loss", "autoencoder"),
            ("_supervised_loss", "supervised"),
            ("_moment_loss", "joint"),
        ],
    )
    def test_non_finite_loss_raises_naming_phase_and_epoch(
        self, toy_df, monkeypatch, helper, phase
    ):
        import torch

        tg = _timegan()

        def nan():
            return torch.tensor(float("nan"), requires_grad=True)

        replacements = {
            "_reconstruction_loss": lambda real, embedder, recovery: (nan(), embedder(real)),
            "_supervised_loss": lambda hidden, supervisor: nan(),
            "_moment_loss": lambda generated, real: nan(),
        }
        monkeypatch.setattr(tg, helper, replacements[helper])
        with pytest.raises(RuntimeError, match=f"diverged in the {phase} phase at epoch 1"):
            _train(toy_df)


class TestTrainArguments:
    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"epochs": 0}, "epochs"),
            ({"epochs": 1.5}, "epochs"),
            ({"batch_size": 1}, "batch_size"),
            ({"hidden_dim": 23}, "hidden_dim must be a whole number from 24 to 64"),
            ({"hidden_dim": 65}, "hidden_dim must be a whole number from 24 to 64"),
            ({"device": "mps"}, "device"),
            ({"seed": "42"}, "seed"),
            ({"seed": 1.0}, "seed"),
        ],
    )
    def test_bad_settings_raise(self, toy_df, settings, message):
        with pytest.raises(ValueError, match=message):
            _train(toy_df, **settings)

    def test_epochs_is_required(self, toy_df):
        with pytest.raises(TypeError, match="epochs"):
            _timegan().train_timegan(toy_df, TARGET, STATIC, _default_steps())

    def test_settings_are_keyword_only(self, toy_df):
        with pytest.raises(TypeError):
            _timegan().train_timegan(toy_df, TARGET, STATIC, _default_steps(), 2)

    def test_frame_errors_reach_the_caller(self, toy_df):
        frame = toy_df.assign(source_row=np.arange(len(toy_df)))
        with pytest.raises(ValueError, match="source_row"):
            _train(frame)
        with pytest.raises(ValueError, match="target_col"):
            _timegan().train_timegan(toy_df, "label", STATIC, _default_steps(), epochs=1)


# ===========================================================================
# Step 3: sampler
# ===========================================================================

IS_SYNTHETIC = "is_synthetic"


@pytest.fixture(scope="module")
def varied(toy_df):
    """A model trained long enough (108 steps per phase) to give varied rows of both classes.

    After 2 epochs every sampled row is the same: an untrained stack of GRUs shrinks the noise
    to almost nothing. The tests that need different rows use this model instead.
    """
    model = _train(toy_df, epochs=12, batch_size=32)
    rows = _timegan().sample_timegan(model, 200, seed=0)
    assert not rows[COLUMNS].duplicated().any()
    assert set(rows[TARGET]) == {0, 1}
    return model


@pytest.fixture(scope="module", params=["two_epochs", "varied"])
def model(request, trained):
    """Validity must hold for the 2-epoch model of the issue and for the varied one."""
    return trained[0] if request.param == "two_epochs" else request.getfixturevalue("varied")


def _sample(model, n=40, **kwargs):
    kwargs.setdefault("seed", 0)
    return _timegan().sample_timegan(model, n, **kwargs)


def _repeat_one_row(monkeypatch, row: pd.DataFrame) -> list[int]:
    """Make every generated row equal ``row``; return the list of row counts requested."""
    requests: list[int] = []

    def generate(model, rows, rng):
        requests.append(rows)
        return pd.concat([row] * rows, ignore_index=True)

    monkeypatch.setattr(_timegan(), "_generate", generate)
    return requests


class TestSampleTimeGAN:
    def test_signature_is_the_one_in_the_issue(self):
        parameters = list(inspect.signature(_timegan().sample_timegan).parameters.values())
        assert [p.name for p in parameters] == [
            "model", "n", "seed", "row_filter", "drop_duplicates_of", "max_rounds",
        ]
        assert all(p.kind is p.KEYWORD_ONLY for p in parameters[2:])
        assert parameters[2].default is inspect.Parameter.empty
        assert {p.name: p.default for p in parameters[3:]} == {
            "row_filter": None, "drop_duplicates_of": None, "max_rounds": 100,
        }

    def test_columns_and_order_equal_the_training_frame(self, model, toy_df):
        out = _sample(model)
        assert list(out.columns) == [*toy_df.columns, IS_SYNTHETIC]

    def test_dtypes_match_the_training_frame(self, model, toy_df):
        out = _sample(model)
        assert out[COLUMNS].dtypes.to_dict() == toy_df.dtypes.to_dict()
        assert out[IS_SYNTHETIC].dtype == np.int64

    def test_returns_n_flagged_rows_without_missing_values(self, model):
        out = _sample(model, 57)
        assert len(out) == 57
        assert out.index.tolist() == list(range(57))
        assert (out[IS_SYNTHETIC] == 1).all()
        assert not out.isna().any().any()

    def test_static_values_are_valid_categories(self, model, toy_df):
        out = _sample(model, 200)
        for name, codes in STATIC_CODES.items():
            assert set(out[name]) <= codes
            assert set(out[name]) <= set(toy_df[name])
        assert set(out[TARGET]) <= {0, 1}

    def test_pay_status_is_inside_the_default_codes(self, model):
        out = _sample(model, 200)
        for name in PAY:
            assert set(out[name]) <= DEFAULT_PAY_CODES

    def test_no_default_rule_is_broken(self, model):
        assert _violations(_sample(model, 200)) == {rule.name: 0.0 for rule in DEFAULT_RULES}

    def test_continuous_values_stay_in_the_training_range(self, model, toy_df):
        out = _sample(model, 200)
        for name in ("LIMIT_BAL", "AGE"):
            assert out[name].between(toy_df[name].min(), toy_df[name].max()).all()
        for group in (BILL, PAID):
            low, high = toy_df[group].min().min(), toy_df[group].max().max()
            assert ((out[group] >= low) & (out[group] <= high)).all().all()

    def test_rows_are_noise_through_generator_supervisor_and_recovery(self, model):
        import torch

        tg = _timegan()
        rng = torch.Generator().manual_seed(8)
        # The first round for n=20 asks for 20 * 1.2 + 10 = 34 rows of uniform noise.
        noise = torch.rand(34, model.layout.n_steps, model.params["noise_dim"], generator=rng)
        with torch.no_grad():
            sequences = model.recovery(model.supervisor(model.generator(noise)))
        expected = tg.from_sequences(sequences.numpy(), model.layout).head(20)
        pd.testing.assert_frame_equal(_sample(model, 20, seed=8)[COLUMNS], expected)

    def test_same_seed_gives_the_same_rows(self, model):
        pd.testing.assert_frame_equal(_sample(model, seed=10), _sample(model, seed=10))

    def test_different_seed_gives_different_rows(self, varied):
        first, second = _sample(varied, seed=10), _sample(varied, seed=11)
        assert not (first.to_numpy() == second.to_numpy()).all(axis=1).all()

    def test_rows_do_not_depend_on_the_global_random_state(self, varied):
        import torch

        torch.manual_seed(1)
        first = _sample(varied, seed=10)
        torch.manual_seed(2)
        pd.testing.assert_frame_equal(first, _sample(varied, seed=10))

    def test_sampling_leaves_the_global_random_state_alone(self, varied):
        import torch

        torch.manual_seed(3)
        expected = torch.rand(4)
        torch.manual_seed(3)
        _sample(varied, seed=10)
        assert torch.equal(torch.rand(4), expected)

    def test_weights_are_unchanged_by_sampling(self, model):
        import torch

        before = _weights(model)
        modes = [network.training for network in model.networks]
        _sample(model, seed=20)
        _sample(model, seed=21)
        after = _weights(model)
        for key, value in before.items():
            assert torch.equal(value, after[key]), key
        assert [network.training for network in model.networks] == modes

    def test_save_then_load_gives_the_same_sample(self, model, tmp_path):
        tg = _timegan()
        expected = _sample(model, seed=30)
        path = tmp_path / "saved" / "timegan.pkl"
        tg.save_model(model, path)
        loaded = tg.load_model(path)
        pd.testing.assert_frame_equal(_sample(loaded, seed=30), expected)
        assert loaded.params == model.params
        pd.testing.assert_frame_equal(loaded.loss_history, model.loss_history)


class TestRowFilter:
    def test_filter_keeping_positives_returns_only_positives(self, varied):
        out = _sample(varied, 50, row_filter=lambda frame: frame[TARGET] == 1)
        assert len(out) == 50
        assert (out[TARGET] == 1).all()

    def test_filter_sees_the_training_columns_without_is_synthetic(self, varied):
        seen: list[list[str]] = []

        def keep_all(frame: pd.DataFrame) -> pd.Series:
            seen.append(list(frame.columns))
            return pd.Series(True, index=frame.index)

        _sample(varied, 10, row_filter=keep_all)
        assert seen == [COLUMNS]

    def test_filter_may_return_a_plain_array(self, varied):
        out = _sample(varied, 20, row_filter=lambda frame: (frame[TARGET] == 0).to_numpy())
        assert len(out) == 20
        assert (out[TARGET] == 0).all()

    def test_filtered_rows_are_the_unfiltered_rows_that_pass(self, varied):
        """The first round draws the same rows with or without a filter."""
        everything = _sample(varied, 34, seed=5)  # the first request for n=20 is 34 rows
        passing = everything[everything[TARGET] == 0].reset_index(drop=True)
        out = _sample(varied, 20, seed=5, row_filter=lambda frame: frame[TARGET] == 0)
        shared = min(len(passing), 20)
        assert shared > 0
        pd.testing.assert_frame_equal(out.head(shared), passing.head(shared))

    def test_impossible_filter_returns_an_empty_frame_with_the_columns(
        self, model, toy_df, caplog
    ):
        import logging

        def impossible(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(False, index=frame.index)

        with caplog.at_level(logging.WARNING, logger="agentic_gan.tools.timegan"):
            out = _sample(model, 50, row_filter=impossible, max_rounds=2)
        assert len(out) == 0
        assert list(out.columns) == [*COLUMNS, IS_SYNTHETIC]
        assert out[COLUMNS].dtypes.to_dict() == toy_df.dtypes.to_dict()
        assert any("Only 0 of 50 valid synthetic rows after 2 rounds" in m for m in caplog.messages)

    def test_request_size_follows_the_keep_rate(self, trained, toy_df, monkeypatch):
        requests = _repeat_one_row(monkeypatch, toy_df.head(1))
        _sample(
            trained[0], 20, row_filter=lambda f: pd.Series(False, index=f.index), max_rounds=2
        )
        # Round 1 assumes every row survives: 20 * 1.2 + 10. Round 2 saw none survive, so the
        # keep rate is floored at 0.02: 20 / 0.02 * 1.2 + 10.
        assert requests == [34, 1210]

    def test_request_size_is_capped(self, trained, toy_df, monkeypatch):
        requests = _repeat_one_row(monkeypatch, toy_df.head(1))
        out = _sample(trained[0], 200_000, max_rounds=1)
        assert requests == [100_000]
        assert len(out) == 100_000


class TestDropDuplicates:
    def test_no_returned_row_equals_a_training_row(self, varied, toy_df):
        out = _sample(varied, 60, seed=4, drop_duplicates_of=toy_df)
        assert len(out) == 60
        assert out[COLUMNS].merge(toy_df, on=COLUMNS, how="inner").empty
        assert not out[COLUMNS].duplicated().any()

    def test_rows_listed_as_duplicates_are_skipped(self, varied):
        """Sampling again while forbidding the first 10 rows returns the rows after them."""
        first = _sample(varied, 30, seed=4)
        forbidden = first.head(10)  # carries is_synthetic too; extra columns are ignored
        again = _sample(varied, 30, seed=4, drop_duplicates_of=forbidden)
        assert len(again) == 30
        assert again[COLUMNS].merge(forbidden[COLUMNS], on=COLUMNS, how="inner").empty
        pd.testing.assert_frame_equal(
            again.head(20), first.iloc[10:30].reset_index(drop=True)
        )

    def test_a_copy_of_a_training_row_is_never_returned(self, trained, toy_df, monkeypatch, caplog):
        import logging

        _repeat_one_row(monkeypatch, toy_df.iloc[[7]])
        with caplog.at_level(logging.WARNING, logger="agentic_gan.tools.timegan"):
            out = _sample(trained[0], 5, drop_duplicates_of=toy_df, max_rounds=3)
        assert len(out) == 0
        assert list(out.columns) == [*COLUMNS, IS_SYNTHETIC]
        assert any("Only 0 of 5" in m for m in caplog.messages)

    def test_duplicates_within_the_sample_are_dropped(self, trained, toy_df, monkeypatch):
        _repeat_one_row(monkeypatch, toy_df.iloc[[7]])
        out = _sample(trained[0], 5, drop_duplicates_of=toy_df.iloc[:0], max_rounds=3)
        assert len(out) == 1
        pd.testing.assert_frame_equal(out[COLUMNS], toy_df.iloc[[7]].reset_index(drop=True))

    def test_duplicates_are_kept_when_no_frame_is_given(self, trained, toy_df, monkeypatch):
        """As in wgan_gp.py: the duplicate check runs only with drop_duplicates_of."""
        _repeat_one_row(monkeypatch, toy_df.iloc[[7]])
        out = _sample(trained[0], 5)
        assert len(out) == 5
        assert out[COLUMNS].duplicated().sum() == 4

    def test_frame_without_the_training_columns_raises(self, trained, toy_df):
        with pytest.raises(ValueError, match=r"drop_duplicates_of is missing columns: \['AGE'\]"):
            _sample(trained[0], 5, drop_duplicates_of=toy_df.drop(columns="AGE"))


class TestSampleArguments:
    @pytest.mark.parametrize("n", [0, -1, 2.5, True, None])
    def test_n_must_be_a_positive_integer(self, trained, n):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            _timegan().sample_timegan(trained[0], n, seed=0)

    @pytest.mark.parametrize("seed", ["1", 1.0, None])
    def test_seed_must_be_an_int(self, trained, seed):
        with pytest.raises(ValueError, match="seed must be an int"):
            _timegan().sample_timegan(trained[0], 5, seed=seed)

    @pytest.mark.parametrize("max_rounds", [0, -3, 1.5])
    def test_max_rounds_must_be_positive(self, trained, max_rounds):
        with pytest.raises(ValueError, match="max_rounds must be a positive integer"):
            _sample(trained[0], 5, max_rounds=max_rounds)

    def test_seed_is_required(self, trained):
        with pytest.raises(TypeError, match="seed"):
            _timegan().sample_timegan(trained[0], 5)
