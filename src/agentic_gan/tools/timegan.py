"""Dataset-agnostic TimeGAN tool: train on a wide table whose rows hold a short sequence.

``train_timegan`` fits a standard TimeGAN (Yoon, Jarrett and van der Schaar, 2019) on the
training split it is given, both classes together: the target is modelled as one more
feature. The first half of this file is the layout that turns the table into sequences and
back; the second half is the five networks and the three training phases.

Layout
------
A wide table stores each entity's history as repeated column groups. Default of Credit Card
has three such groups (repayment status, bill amount, payment amount), each with one column
per month. Here every row becomes a sequence of ``steps`` x ``features`` numbers:

* a *step feature* is one position inside a step, such as "bill amount". All of its columns,
  one per step, share one minimum and maximum, taken over all steps;
* the static columns and the target are repeated at every step as extra features and are read
  back from the first step.

So a feature vector is ``[step features..., static columns..., target]``, and steps are listed
oldest first.

Nothing here names a dataset. ``fit_layout`` learns the column order, dtypes, ranges and valid
codes from the training frame. A feature is *coded* when it is integer-valued with at most
``CODED_MAX_VALUES`` distinct values (the rule ``ctgan_sampling`` uses for SEX or PAY_1); the
target is always coded. Decoding snaps a coded feature to its nearest valid value and clips
the others to the training range, so a decoded row never holds a code or an amount the
training rows did not allow.

Sequences are float32. An integer-valued feature decodes exactly as long as its range is
below 2**24 (about 16.7 million).

Training
--------
Five GRU networks: the embedder maps a sequence to a hidden sequence and the recovery maps it
back; the supervisor predicts the next hidden step; the generator maps uniform noise to a
hidden sequence; the discriminator scores each hidden step as real or generated. Training
runs three phases, each for ``epochs`` epochs:

1. ``autoencoder``: embedder and recovery learn to reconstruct the real sequences;
2. ``supervised``: the supervisor learns the next hidden step, with the embedder frozen;
3. ``joint``: per step, two generator-and-supervisor updates, each followed by an
   embedder-and-recovery update, then one discriminator update, which is skipped while the
   discriminator's loss is at or below ``DISCRIMINATOR_THRESHOLD``.

The loss history has one row per phase and epoch (so ``3 * epochs`` rows) with the epoch mean
of each loss; a loss that a phase does not compute is NaN. ``Embedder Loss`` is the loss the
embedder-and-recovery update minimised, ``Supervised Loss`` the next-step error, ``Generator
Loss`` and ``Discriminator Loss`` the joint-phase totals.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic_gan.tools.ctgan_sampling import CODED_MAX_VALUES

__all__ = [
    "CODED_MAX_VALUES",
    "DEFAULTS",
    "DEVICES",
    "DISCRIMINATOR_THRESHOLD",
    "GENERATOR_STEPS",
    "HIDDEN_DIM_RANGE",
    "LOSS_COLUMNS",
    "PHASES",
    "Discriminator",
    "Embedder",
    "Recovery",
    "SequenceGenerator",
    "SequenceLayout",
    "Supervisor",
    "TrainedTimeGAN",
    "fit_layout",
    "from_sequences",
    "load_model",
    "save_model",
    "step_columns_from_wide",
    "to_sequences",
    "train_timegan",
]

logger = logging.getLogger(__name__)

# Settings that are not part of the public training signature.
DEFAULTS: dict[str, Any] = {
    "num_layers": 3,
    "gamma": 1.0,
    "lr": 1e-3,
    "noise_dim": None,  # None means the same width as the feature vector
}
HIDDEN_DIM_RANGE = (24, 64)
DEVICES = ("cpu", "cuda", "auto")
PHASES = ("autoencoder", "supervised", "joint")
LOSS_COLUMNS = ("Embedder Loss", "Supervised Loss", "Generator Loss", "Discriminator Loss")
GENERATOR_STEPS = 2  # generator (and embedder) updates per discriminator update, joint phase
DISCRIMINATOR_THRESHOLD = 0.15  # the discriminator is updated only while its loss is above this
_SQRT_FLOOR = 1e-8  # keeps the gradient of sqrt(loss) finite if a loss reaches exactly zero


# ---------------------------------------------------------------------------
# Step columns
# ---------------------------------------------------------------------------

def step_columns_from_wide(
    wide: Sequence[str], n_steps: int, *, newest_first: bool
) -> list[list[str]]:
    """Regroup feature-major column names into one list per step, oldest step first.

    ``wide`` lists every column of the first feature, then every column of the second, and
    so on, each feature in the same step order (``PAY_1..6, BILL_AMT1..6, PAY_AMT1..6``).
    ``newest_first`` says which way that order runs: True when the first name of each
    feature is the latest step, as in Default, where suffix 1 is September and 6 is April.
    """
    names = list(wide)
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError(f"n_steps must be a whole number of at least 2, got {n_steps!r}")
    if not names or len(names) % n_steps:
        raise ValueError(
            f"{len(names)} wide columns cannot be split into {n_steps} steps of equal width"
        )
    if len(set(names)) != len(names):
        raise ValueError("wide lists a column more than once")
    width = len(names) // n_steps
    steps = [[names[feature * n_steps + step] for feature in range(width)]
             for step in range(n_steps)]
    return steps[::-1] if newest_first else steps


# ---------------------------------------------------------------------------
# Argument and frame validation
# ---------------------------------------------------------------------------

def _is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_args(epochs: int, batch_size: int, hidden_dim: int, device: str) -> None:
    if not _is_whole(epochs) or epochs < 1:
        raise ValueError(f"epochs must be a whole number of at least 1, got {epochs!r}")
    if not _is_whole(batch_size) or batch_size < 2:
        raise ValueError(f"batch_size must be a whole number of at least 2, got {batch_size!r}")
    low, high = HIDDEN_DIM_RANGE
    if not _is_whole(hidden_dim) or not low <= hidden_dim <= high:
        raise ValueError(
            f"hidden_dim must be a whole number from {low} to {high}, got {hidden_dim!r}"
        )
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")


def _validate_frame(
    train_df: pd.DataFrame,
    target_col: str,
    static_columns: Sequence[str],
    step_columns: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Check the frame against the column lists; return the lists as tuples."""
    static = tuple(static_columns)
    steps = tuple(tuple(step) for step in step_columns)
    if len(steps) < 2:
        raise ValueError(f"step_columns needs at least 2 steps, got {len(steps)}")
    widths = [len(step) for step in steps]
    if len(set(widths)) != 1 or widths[0] < 1:
        raise ValueError(
            f"every step must list the same number of columns (at least 1), got {widths}"
        )
    if train_df.columns.duplicated().any():
        raise ValueError("train_df has duplicate column names")
    if target_col not in train_df.columns:
        raise ValueError(f"target_col {target_col!r} is not in train_df")
    listed = [*static, *(name for step in steps for name in step)]
    if target_col in listed:
        raise ValueError(
            f"target_col {target_col!r} must not be listed in static_columns or step_columns"
        )
    repeated = sorted({name for name in listed if listed.count(name) > 1})
    if repeated:
        raise ValueError(
            f"columns listed more than once in static_columns and step_columns: {repeated}"
        )
    missing = [name for name in listed if name not in train_df.columns]
    if missing:
        raise ValueError(f"listed columns are not in train_df: {missing}")
    covered = {target_col, *listed}
    uncovered = [name for name in train_df.columns if name not in covered]
    if uncovered:
        raise ValueError(
            "train_df has columns that are in neither static_columns nor step_columns: "
            f"{uncovered}. Drop them or list them."
        )
    non_numeric = [
        name for name in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[name])
    ]
    if non_numeric:
        raise ValueError(f"TimeGAN needs numeric columns; encode these first: {non_numeric}")
    unusable = [
        name
        for name in train_df.columns
        if train_df[name].isna().any()
        or not np.isfinite(train_df[name].to_numpy(dtype=np.float64)).all()
    ]
    if unusable:
        raise ValueError(f"columns have missing or non-finite values: {unusable}")
    if len(train_df) < 2:
        raise ValueError(f"train_df needs at least 2 rows, got {len(train_df)}")
    return static, steps


# ---------------------------------------------------------------------------
# Layout: wide table <-> scaled sequences
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequenceLayout:
    """Everything needed to turn the training table into scaled sequences and back.

    The per-feature fields (``features``, ``minimum``, ``maximum``, ``integer_valued``,
    ``valid_values``) follow the feature vector: step features, static columns, target.
    ``valid_values`` holds the sorted codes of a coded feature and None for the others.
    """

    target_col: str
    static_columns: tuple[str, ...]
    step_columns: tuple[tuple[str, ...], ...]  # oldest step first
    columns: tuple[str, ...]  # the training frame's column order, target included
    dtypes: dict[str, str]
    features: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    integer_valued: tuple[bool, ...]
    valid_values: tuple[tuple[float, ...] | None, ...]

    @property
    def n_steps(self) -> int:
        return len(self.step_columns)

    @property
    def n_step_features(self) -> int:
        return len(self.step_columns[0])

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def coded(self) -> tuple[str, ...]:
        """Names of the coded features."""
        return tuple(
            name
            for name, values in zip(self.features, self.valid_values, strict=True)
            if values is not None
        )


def fit_layout(
    train_df: pd.DataFrame,
    target_col: str,
    static_columns: Sequence[str],
    step_columns: Sequence[Sequence[str]],
) -> SequenceLayout:
    """Learn the layout from the training frame. ``train_df`` is not changed."""
    static, steps = _validate_frame(train_df, target_col, static_columns, step_columns)
    width = len(steps[0])
    # One array of training values per feature. A step feature pools its column from every step.
    groups = [
        train_df[[step[position] for step in steps]].to_numpy(dtype=np.float64).ravel()
        for position in range(width)
    ]
    groups += [train_df[name].to_numpy(dtype=np.float64) for name in (*static, target_col)]
    features = (*(f"step_feature_{position}" for position in range(width)), *static, target_col)

    minimum: list[float] = []
    maximum: list[float] = []
    integer_valued: list[bool] = []
    valid_values: list[tuple[float, ...] | None] = []
    for name, values in zip(features, groups, strict=True):
        whole = bool(np.all(values == np.round(values)))
        distinct = np.unique(values)
        coded = name == target_col or (whole and len(distinct) <= CODED_MAX_VALUES)
        minimum.append(float(values.min()))
        maximum.append(float(values.max()))
        integer_valued.append(whole)
        valid_values.append(tuple(float(v) for v in distinct) if coded else None)

    return SequenceLayout(
        target_col=target_col,
        static_columns=static,
        step_columns=steps,
        columns=tuple(train_df.columns),
        dtypes={name: str(dtype) for name, dtype in train_df.dtypes.items()},
        features=features,
        minimum=tuple(minimum),
        maximum=tuple(maximum),
        integer_valued=tuple(integer_valued),
        valid_values=tuple(valid_values),
    )


def _bounds(layout: SequenceLayout) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(layout.minimum, dtype=np.float64),
        np.asarray(layout.maximum, dtype=np.float64),
    )


def to_sequences(df: pd.DataFrame, layout: SequenceLayout) -> np.ndarray:
    """Rows of ``df`` as a float32 array of shape (rows, steps, features), min-max scaled.

    Training rows land in [0, 1]. A feature that is constant in training scales to 0.
    """
    absent = [name for name in layout.columns if name not in df.columns]
    if absent:
        raise ValueError(f"df is missing columns the layout was fitted on: {absent}")
    width, n_static = layout.n_step_features, len(layout.static_columns)
    raw = np.empty((len(df), layout.n_steps, layout.n_features), dtype=np.float64)
    for step, names in enumerate(layout.step_columns):
        raw[:, step, :width] = df[list(names)].to_numpy(dtype=np.float64)
    static = df[list(layout.static_columns)].to_numpy(dtype=np.float64)
    raw[:, :, width:width + n_static] = static[:, None, :]
    raw[:, :, -1] = df[layout.target_col].to_numpy(dtype=np.float64)[:, None]
    low, high = _bounds(layout)
    span = np.where(high > low, high - low, 1.0)
    return ((raw - low) / span).astype(np.float32)


def _snap(values: np.ndarray, valid: Sequence[float]) -> np.ndarray:
    """Each value replaced by the nearest of the sorted ``valid`` codes (ties go down)."""
    codes = np.asarray(valid, dtype=np.float64)
    if len(codes) == 1:
        return np.full(values.shape, codes[0])
    right = np.searchsorted(codes, values).clip(1, len(codes) - 1)
    left = right - 1
    nearer_left = values - codes[left] <= codes[right] - values
    return codes[np.where(nearer_left, left, right)]


def _restore(values: np.ndarray, index: int, layout: SequenceLayout) -> np.ndarray:
    """Unscaled values of feature ``index`` made valid: snapped, or clipped and rounded."""
    valid = layout.valid_values[index]
    if valid is not None:
        return _snap(values, valid)
    low, high = layout.minimum[index], layout.maximum[index]
    if layout.integer_valued[index]:
        values = np.round(values)
    return np.clip(values, low, high)


def from_sequences(array: np.ndarray, layout: SequenceLayout) -> pd.DataFrame:
    """Scaled sequences back to a table with the training frame's columns, order and dtypes.

    Values are clipped to [0, 1] and unscaled. Static columns and the target are read from the
    first step. A coded feature is snapped to its nearest valid value; any other feature is
    clipped to the training range and rounded when the training values were whole numbers.
    """
    data = np.asarray(array, dtype=np.float64)
    expected = (layout.n_steps, layout.n_features)
    if data.ndim != 3 or data.shape[1:] != expected:
        raise ValueError(
            f"expected an array of shape (rows, {expected[0]}, {expected[1]}), got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError("the sequences hold missing or non-finite values")
    low, high = _bounds(layout)
    values = np.clip(data, 0.0, 1.0) * (high - low) + low
    width = layout.n_step_features
    table: dict[str, np.ndarray] = {}
    for step, names in enumerate(layout.step_columns):
        for position, name in enumerate(names):
            table[name] = _restore(values[:, step, position], position, layout)
    for offset, name in enumerate((*layout.static_columns, layout.target_col)):
        table[name] = _restore(values[:, 0, width + offset], width + offset, layout)
    frame = pd.DataFrame(table)[list(layout.columns)]
    return frame.astype(layout.dtypes)


# ---------------------------------------------------------------------------
# Trained model, save and load
# ---------------------------------------------------------------------------

@dataclass
class TrainedTimeGAN:
    """A fitted TimeGAN plus everything needed to sample from it and to audit the run."""

    embedder: nn.Module
    recovery: nn.Module
    supervisor: nn.Module
    generator: nn.Module
    discriminator: nn.Module
    layout: SequenceLayout
    target_col: str
    static_columns: list[str]
    step_columns: list[list[str]]  # oldest step first
    columns: list[str]  # full original column order, target included
    dtypes: dict[str, str]
    params: dict[str, Any]
    loss_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_seconds: float = 0.0
    n_train_rows: int = 0

    @property
    def networks(self) -> tuple[nn.Module, ...]:
        return (
            self.embedder, self.recovery, self.supervisor, self.generator, self.discriminator
        )


def save_model(model: TrainedTimeGAN, path: str | Path) -> None:
    """Pickle the trained networks and their metadata (moved to CPU for portability).

    Only load files you created yourself.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for network in model.networks:
        network.cpu()
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> TrainedTimeGAN:
    """Reload a model written by ``save_model``.  Only load files you created."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _SequenceNet(nn.Module):
    """A GRU over the steps, then one linear layer applied at every step."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, *, squash: bool
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.squash = squash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.linear(out)
        return torch.sigmoid(out) if self.squash else out


class Embedder(_SequenceNet):
    """Real sequence (batch, steps, features) -> hidden sequence, each value in (0, 1)."""

    def __init__(self, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__(n_features, hidden_dim, hidden_dim, num_layers, squash=True)


class Recovery(_SequenceNet):
    """Hidden sequence -> sequence of features, each value in (0, 1) like the scaled data."""

    def __init__(self, hidden_dim: int, n_features: int, num_layers: int) -> None:
        super().__init__(hidden_dim, hidden_dim, n_features, num_layers, squash=True)


class Supervisor(_SequenceNet):
    """Hidden sequence -> the next hidden step at each position. One GRU layer fewer."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1), squash=True)


class SequenceGenerator(_SequenceNet):
    """Noise sequence (batch, steps, noise_dim) -> hidden sequence, each value in (0, 1)."""

    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__(noise_dim, hidden_dim, hidden_dim, num_layers, squash=True)


class Discriminator(_SequenceNet):
    """Hidden sequence -> one logit per step (batch, steps). No sigmoid: the losses take logits."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__(hidden_dim, hidden_dim, 1, num_layers, squash=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Losses (the weights 10, 100 and 0.1 are those of the TimeGAN reference code)
# ---------------------------------------------------------------------------

def _reconstruction_loss(
    real: torch.Tensor, embedder: nn.Module, recovery: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """``10 * sqrt(MSE(X, R(E(X))))`` and the hidden sequence ``E(X)`` it was computed from."""
    hidden = embedder(real)
    error = F.mse_loss(recovery(hidden), real)
    return 10.0 * torch.sqrt(error + _SQRT_FLOOR), hidden


def _supervised_loss(hidden: torch.Tensor, supervisor: nn.Module) -> torch.Tensor:
    """``MSE(H[:, 1:], S(H)[:, :-1])``: how well the supervisor predicts the next hidden step."""
    return F.mse_loss(supervisor(hidden)[:, :-1], hidden[:, 1:])


def _batch_std(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x.var(dim=0, correction=0) + 1e-6)


def _moment_loss(generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Mean gap between the batch standard deviations plus the mean gap between the means."""
    spread = (_batch_std(generated) - _batch_std(real)).abs().mean()
    centre = (generated.mean(dim=0) - real.mean(dim=0)).abs().mean()
    return spread + centre


def _adversarial_loss(logits: torch.Tensor, *, real: bool) -> torch.Tensor:
    """Binary cross-entropy of the discriminator's logits against "real" or "generated"."""
    labels = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _epoch_row(phase: str, epoch: int, **means: float) -> dict[str, Any]:
    """One loss-history row. Raises if any epoch mean is not finite."""
    broken = [name for name, value in means.items() if not np.isfinite(value)]
    if broken:
        raise RuntimeError(
            f"TimeGAN diverged in the {phase} phase at epoch {epoch}: "
            f"non-finite {', '.join(broken)}"
        )
    row: dict[str, Any] = {"Phase": phase, "Epoch": epoch}
    row.update({column: float("nan") for column in LOSS_COLUMNS})
    row.update(means)
    return row


def _autoencoder_phase(embedder, recovery, optimiser, batch, steps, epochs) -> list[dict]:
    rows = []
    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(steps):
            loss, _ = _reconstruction_loss(batch(), embedder, recovery)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        rows.append(_epoch_row("autoencoder", epoch, **{"Embedder Loss": float(np.mean(losses))}))
    return rows


def _supervised_phase(embedder, supervisor, optimiser, batch, steps, epochs) -> list[dict]:
    rows = []
    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(steps):
            with torch.no_grad():  # the embedder is frozen in this phase
                hidden = embedder(batch())
            loss = _supervised_loss(hidden, supervisor)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        rows.append(
            _epoch_row("supervised", epoch, **{"Supervised Loss": float(np.mean(losses))})
        )
    return rows


def _joint_phase(networks, optimisers, batch, noise, steps, epochs, gamma) -> list[dict]:
    embedder, recovery, supervisor, generator, discriminator = networks
    generator_opt, embedder_opt, discriminator_opt = optimisers
    rows = []
    for epoch in range(1, epochs + 1):
        history: dict[str, list[float]] = {column: [] for column in LOSS_COLUMNS}
        for _ in range(steps):
            for _ in range(GENERATOR_STEPS):
                real = batch()

                # Generator and supervisor. The embedder is not updated here, so H is a constant.
                with torch.no_grad():
                    hidden = embedder(real)
                generated_embedding = generator(noise())
                generated_hidden = supervisor(generated_embedding)
                generated = recovery(generated_hidden)
                supervised = _supervised_loss(hidden, supervisor)
                generator_loss = (
                    _adversarial_loss(discriminator(generated_hidden), real=True)
                    + gamma * _adversarial_loss(discriminator(generated_embedding), real=True)
                    + 100.0 * torch.sqrt(supervised + _SQRT_FLOOR)
                    + 100.0 * _moment_loss(generated, real)
                )
                generator_opt.zero_grad()
                generator_loss.backward()
                generator_opt.step()

                # Embedder and recovery, on the same real batch.
                reconstruction, hidden = _reconstruction_loss(real, embedder, recovery)
                embedder_loss = reconstruction + 0.1 * _supervised_loss(hidden, supervisor)
                embedder_opt.zero_grad()
                embedder_loss.backward()
                embedder_opt.step()

                history["Supervised Loss"].append(supervised.item())
                history["Generator Loss"].append(generator_loss.item())
                history["Embedder Loss"].append(embedder_loss.item())

            # Discriminator, on fresh batches. Nothing upstream of it is updated here.
            with torch.no_grad():
                hidden = embedder(batch())
                generated_embedding = generator(noise())
                generated_hidden = supervisor(generated_embedding)
            discriminator_loss = (
                _adversarial_loss(discriminator(hidden), real=True)
                + _adversarial_loss(discriminator(generated_hidden), real=False)
                + gamma * _adversarial_loss(discriminator(generated_embedding), real=False)
            )
            if discriminator_loss.item() > DISCRIMINATOR_THRESHOLD:
                discriminator_opt.zero_grad()
                discriminator_loss.backward()
                discriminator_opt.step()
            history["Discriminator Loss"].append(discriminator_loss.item())

        rows.append(
            _epoch_row(
                "joint", epoch, **{column: float(np.mean(history[column])) for column in history}
            )
        )
    return rows


def train_timegan(
    train_df: pd.DataFrame,
    target_col: str,
    static_columns: Sequence[str],
    step_columns: Sequence[Sequence[str]],
    *,
    epochs: int,
    batch_size: int = 128,
    hidden_dim: int = 48,
    seed: int = 42,
    device: str = "cpu",
    log_path: str | Path | None = None,
) -> TrainedTimeGAN:
    """Fit TimeGAN on every row of ``train_df``, which must be the training split only.

    ``step_columns`` holds one list of column names per step, oldest step first, each with the
    same features in the same order. ``static_columns`` are the per-entity columns. Every
    column of ``train_df`` except the target must appear in exactly one of the two.

    ``epochs`` is the number of epochs of each of the three phases. An epoch is
    ``max(rows // batch_size, 1)`` steps, each on a batch drawn with replacement. The loss
    history (one row per phase and epoch) is returned on the model and written to ``log_path``
    as CSV when given. ``train_df`` is not changed.
    """
    _validate_args(epochs, batch_size, hidden_dim, device)
    if not _is_whole(seed):
        raise ValueError(f"seed must be a whole number, got {seed!r}")
    layout = fit_layout(train_df, target_col, static_columns, step_columns)
    use = "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"

    # Seed before the networks are built: their initial weights come from the global state.
    torch.manual_seed(seed)
    np.random.seed(seed)

    real = torch.tensor(to_sequences(train_df, layout), dtype=torch.float32, device=use)
    n_rows, n_steps, n_features = real.shape
    num_layers = DEFAULTS["num_layers"]
    noise_dim = DEFAULTS["noise_dim"] or n_features
    gamma, lr = DEFAULTS["gamma"], DEFAULTS["lr"]

    embedder = Embedder(n_features, hidden_dim, num_layers).to(use)
    recovery = Recovery(hidden_dim, n_features, num_layers).to(use)
    supervisor = Supervisor(hidden_dim, num_layers).to(use)
    generator = SequenceGenerator(noise_dim, hidden_dim, num_layers).to(use)
    discriminator = Discriminator(hidden_dim, num_layers).to(use)
    networks = (embedder, recovery, supervisor, generator, discriminator)

    def adam(*modules: nn.Module) -> torch.optim.Adam:
        return torch.optim.Adam([p for module in modules for p in module.parameters()], lr=lr)

    # One optimiser per kind of update, as in the reference code. Created in this order.
    autoencoder_opt = adam(embedder, recovery)  # autoencoder phase
    supervisor_opt = adam(supervisor)  # supervised phase
    generator_opt = adam(generator, supervisor)  # joint phase
    embedder_opt = adam(embedder, recovery)  # joint phase
    discriminator_opt = adam(discriminator)  # joint phase

    # Dedicated generator for batch indices and noise; it does not consume the global state.
    rng = torch.Generator(device=use)
    rng.manual_seed(seed)

    def batch() -> torch.Tensor:
        return real[torch.randint(0, n_rows, (batch_size,), generator=rng, device=use)]

    def noise() -> torch.Tensor:
        return torch.rand(batch_size, n_steps, noise_dim, generator=rng, device=use)

    steps = max(n_rows // batch_size, 1)
    for network in networks:
        network.train()
    start = time.perf_counter()
    rows = _autoencoder_phase(embedder, recovery, autoencoder_opt, batch, steps, epochs)
    rows += _supervised_phase(embedder, supervisor, supervisor_opt, batch, steps, epochs)
    rows += _joint_phase(
        networks,
        (generator_opt, embedder_opt, discriminator_opt),
        batch,
        noise,
        steps,
        epochs,
        gamma,
    )
    train_seconds = time.perf_counter() - start
    for network in networks:
        network.eval()

    loss_history = pd.DataFrame(rows, columns=["Phase", "Epoch", *LOSS_COLUMNS])
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        loss_history.to_csv(log_path, index=False)

    final = loss_history.iloc[-1]
    logger.info(
        "TimeGAN trained on %d rows, %d steps x %d features, device=%s in %.1fs; "
        "final G=%.3f D=%.3f E=%.3f",
        n_rows, n_steps, n_features, use, train_seconds,
        final["Generator Loss"], final["Discriminator Loss"], final["Embedder Loss"],
    )

    return TrainedTimeGAN(
        embedder=embedder,
        recovery=recovery,
        supervisor=supervisor,
        generator=generator,
        discriminator=discriminator,
        layout=layout,
        target_col=target_col,
        static_columns=list(layout.static_columns),
        step_columns=[list(step) for step in layout.step_columns],
        columns=list(layout.columns),
        dtypes=dict(layout.dtypes),
        params={
            "epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden_dim,
            "seed": seed, "device": device, "num_layers": num_layers,
            "noise_dim": noise_dim, "gamma": gamma, "lr": lr,
            "generator_steps": GENERATOR_STEPS,
            "discriminator_threshold": DISCRIMINATOR_THRESHOLD,
            "steps_per_epoch": steps,
        },
        loss_history=loss_history,
        train_seconds=train_seconds,
        n_train_rows=n_rows,
    )
