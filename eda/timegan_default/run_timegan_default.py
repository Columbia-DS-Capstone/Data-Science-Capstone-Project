"""Train TimeGAN on the Default training split and write the 10% fill file.

Reads data/interim/default/train_cleaned.csv only. The validation and test splits are not
read by this script.

Each customer is a sequence of 6 months, April (suffix 6) first and September (suffix 1) last.
The fill file holds synthetic positive rows (y == 1) in the same layout as the CTGAN fill files.

Run:
    uv run --no-sync python eda/timegan_default/run_timegan_default.py --epochs 100

Outputs:
    eda/timegan_default/artifacts/timegan_losses.csv   one row per phase and epoch
    eda/timegan_default/artifacts/timegan_default.pkl  the trained model
    eda/timegan_default/artifacts/loss_curves.png      one panel per phase
    eda/timegan_default/artifacts/run_meta.json        settings, timings, counts, checks
    data/interim/default/synthetic/default_timegan_fill10.csv
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.figure  # noqa: E402
import pandas as pd
import torch

from agentic_gan.schemas import fill_rows
from agentic_gan.tools.ctgan_sampling import rule_row_filter
from agentic_gan.tools.quality_fidelity import DEFAULT_RULES, rule_violation_rate
from agentic_gan.tools.selection import DEFAULT_WIDE_TEMPORAL
from agentic_gan.tools.timegan import (
    DEVICES,
    PHASES,
    TrainedTimeGAN,
    sample_timegan,
    save_model,
    step_columns_from_wide,
    train_timegan,
)
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import IS_SYNTHETIC, fill_counts

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["default"]
TRAIN_FILE = ROOT / "data" / "interim" / CONFIG.name / "train_cleaned.csv"
ARTIFACT_DIR = ROOT / "eda" / "timegan_default" / "artifacts"
FILL_FILE = ROOT / "data" / "interim" / CONFIG.name / "synthetic" / "default_timegan_fill10.csv"

ID_COLUMN = "source_row"
TARGET = CONFIG.target
POSITIVE = 1
STATIC_COLUMNS = ("LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE")
N_STEPS = 6
FILL_LEVEL = 0.10
DIAGNOSTIC_ROWS = 5_000

# The loss each phase is there to lower; its last value is the phase's "final loss".
PHASE_LOSSES = {
    "autoencoder": ("Embedder Loss",),
    "supervised": ("Supervised Loss",),
    "joint": ("Generator Loss", "Discriminator Loss", "Embedder Loss", "Supervised Loss"),
}

log = logging.getLogger("run_timegan_default")


def _lib_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _display(path: Path) -> str:
    """``path`` relative to the repository when it is inside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_train() -> pd.DataFrame:
    """The cleaned training split without the row id. Stops if a real row breaks a rule."""
    train = pd.read_csv(TRAIN_FILE)
    if ID_COLUMN in train.columns:
        train = train.drop(columns=ID_COLUMN)
    broken = {rule.name: rule_violation_rate(train, rule) for rule in DEFAULT_RULES}
    if any(rate > 0 for rate in broken.values()):
        raise SystemExit(f"Real training rows break DEFAULT_RULES: {broken}")
    return train


def rule_rates(frame: pd.DataFrame) -> dict[str, float]:
    """Share of rows breaking each Default rule, and the share breaking any of them."""
    rates = {rule.name: rule_violation_rate(frame, rule) for rule in DEFAULT_RULES}
    any_broken = ~rule_row_filter(DEFAULT_RULES)(frame)
    rates["any_rule"] = float(any_broken.mean()) if len(frame) else 0.0
    return rates


def diagnose(model: TrainedTimeGAN, train: pd.DataFrame, seed: int) -> dict:
    """What the generator gives before any filter: class share, rule breaks, variety."""
    sample = sample_timegan(model, DIAGNOSTIC_ROWS, seed=seed)
    rows = sample[model.columns]
    return {
        "rows": len(sample),
        "seed": seed,
        "positive_share": float((sample[TARGET] == POSITIVE).mean()),
        "real_positive_share": float((train[TARGET] == POSITIVE).mean()),
        "rule_violation_rates": rule_rates(rows),
        "unique_row_share": float(len(rows.drop_duplicates()) / len(rows)),
        "exact_train_copies": int(len(rows.merge(train.drop_duplicates(), how="inner"))),
        "distinct_values": {
            name: {"real": int(train[name].nunique()), "synthetic": int(rows[name].nunique())}
            for name in CONFIG.discrete
        },
    }


def counting_filter(
    keep: Callable[[pd.DataFrame], pd.Series], counts: dict[str, int]
) -> Callable[[pd.DataFrame], pd.Series]:
    """``keep`` plus a tally of the rows it saw and the rows it kept, one entry per round."""

    def counted(frame: pd.DataFrame) -> pd.Series:
        mask = keep(frame)
        counts["rounds"] += 1
        counts["drawn"] += len(frame)
        counts["kept"] += int(mask.sum())
        return mask

    return counted


def sample_fill(model: TrainedTimeGAN, train: pd.DataFrame, n: int, seed: int):
    """``n`` positive rows that break no rule and copy no training row, with the tally."""
    no_rule_broken = rule_row_filter(DEFAULT_RULES)

    def keep(frame: pd.DataFrame) -> pd.Series:
        return no_rule_broken(frame) & (frame[TARGET] == POSITIVE)

    counts = {"rounds": 0, "drawn": 0, "kept": 0}
    fill = sample_timegan(
        model, n, seed=seed, row_filter=counting_filter(keep, counts), drop_duplicates_of=train
    )
    return fill, counts


def check_fill_file(path: Path, train: pd.DataFrame, n: int) -> dict[str, bool]:
    """Reread the fill file and check it. Raises if any check fails."""
    fill = pd.read_csv(path)
    if list(fill.columns) != [*train.columns, IS_SYNTHETIC]:
        raise SystemExit(f"Fill file {path} has the wrong columns: {list(fill.columns)}")
    rows = fill[list(train.columns)]
    checks = {
        "row_count_equals_fill_count": len(fill) == n,
        "columns_are_train_columns_plus_is_synthetic": True,
        "dtypes_match_train": rows.dtypes.to_dict() == train.dtypes.to_dict(),
        "every_target_is_positive": bool((fill[TARGET] == POSITIVE).all()),
        "every_is_synthetic_is_1": bool((fill[IS_SYNTHETIC] == 1).all()),
        "no_missing_values": not bool(fill.isna().any().any()),
        "no_exact_copy_of_a_train_row":
            rows.merge(train.drop_duplicates(), how="inner").empty,
        "no_repeated_row": not bool(rows.duplicated().any()),
        "no_default_rule_broken": rule_rates(rows)["any_rule"] == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Fill file {path} failed checks: {failed}")
    return checks


def plot_losses(losses: pd.DataFrame, path: Path, title: str) -> None:
    """Three panels, one per training phase."""
    fig = matplotlib.figure.Figure(figsize=(15, 4.2))
    for index, phase in enumerate(PHASES, start=1):
        ax = fig.add_subplot(1, len(PHASES), index)
        rows = losses[losses["Phase"] == phase]
        for column in PHASE_LOSSES[phase]:
            ax.plot(rows["Epoch"], rows[column], label=column)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(f"{phase.capitalize()} phase")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def final_losses(losses: pd.DataFrame) -> dict[str, dict[str, float]]:
    """The last epoch's losses in each phase."""
    final = {}
    for phase in PHASES:
        last = losses[losses["Phase"] == phase].iloc[-1]
        final[phase] = {column: float(last[column]) for column in PHASE_LOSSES[phase]}
    return final


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, required=True, help="epochs per phase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=list(DEVICES))
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FILL_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load the training split ---
    train = load_train()
    positives = int((train[TARGET] == POSITIVE).sum())
    negatives = int((train[TARGET] != POSITIVE).sum())
    log.info("Training split: %d rows (%d positive, %d negative)", len(train), positives, negatives)

    # --- 2. Sequence layout: month 6 (April) first, month 1 (September) last ---
    step_columns = step_columns_from_wide(DEFAULT_WIDE_TEMPORAL, N_STEPS, newest_first=True)
    log.info("Steps, oldest first: %s", step_columns)

    # --- 3. Train ---
    loss_path = ARTIFACT_DIR / "timegan_losses.csv"
    log.info(
        "Training: %d epochs per phase, batch size %d, hidden size %d, seed %d, device %s",
        args.epochs, args.batch_size, args.hidden_dim, args.seed, args.device,
    )
    start = time.perf_counter()
    model = train_timegan(
        train,
        TARGET,
        list(STATIC_COLUMNS),
        step_columns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        log_path=loss_path,
    )
    wall_time = time.perf_counter() - start
    model_path = ARTIFACT_DIR / "timegan_default.pkl"
    save_model(model, model_path)
    log.info("Training wall time: %.1f s", wall_time)

    losses = pd.read_csv(loss_path)
    final = final_losses(losses)
    for phase, values in final.items():
        log.info("Final losses, %s phase: %s", phase, values)

    # --- 8. Loss curves (drawn now, so they exist even if sampling stops the run) ---
    loss_png = ARTIFACT_DIR / "loss_curves.png"
    plot_losses(
        losses, loss_png, f"TimeGAN on Default: training losses, {args.epochs} epochs per phase"
    )
    log.info("Loss curves saved to %s", loss_png)

    # --- 4. Fill count ---
    n = fill_counts(train[TARGET], (FILL_LEVEL,))[FILL_LEVEL]
    assert n == fill_rows(FILL_LEVEL, positives, negatives), (
        n, fill_rows(FILL_LEVEL, positives, negatives),
    )
    log.info("Fill count at %.0f%%: %d rows", FILL_LEVEL * 100, n)

    # --- 5. Diagnostic: what the generator gives before any filter ---
    diagnostic = diagnose(model, train, seed=args.seed + 1)
    log.info(
        "Unfiltered sample of %d rows: y == 1 share %.4f (real %.4f), rule violation rate %.4f",
        diagnostic["rows"], diagnostic["positive_share"], diagnostic["real_positive_share"],
        diagnostic["rule_violation_rates"]["any_rule"],
    )

    # --- 6. Fill file: positive rows that break no rule and copy no training row ---
    start = time.perf_counter()
    fill, tally = sample_fill(model, train, n, seed=args.seed)
    sampling_seconds = time.perf_counter() - start
    keep_rate = tally["kept"] / max(tally["drawn"], 1)
    log.info(
        "Sampler: %d rows drawn in %d rounds, %d passed the filter (keep rate %.4f), %d returned",
        tally["drawn"], tally["rounds"], tally["kept"], keep_rate, len(fill),
    )
    if len(fill) < n:
        raise SystemExit(
            f"Only {len(fill)} of {n} fill rows. The share of y == 1 in an unfiltered sample is "
            f"{diagnostic['positive_share']:.4f}. No fill file was written."
        )
    fill.to_csv(FILL_FILE, index=False)
    log.info("Fill file written to %s", FILL_FILE)

    # --- 7. Reread the file and check it ---
    checks = check_fill_file(FILL_FILE, train, n)
    for name in checks:
        log.info("PASS %s", name)

    # --- 9. run_meta.json ---
    meta = {
        "settings": {
            "epochs_per_phase": args.epochs,
            "seed": args.seed,
            "device": args.device,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
        },
        "model_params": model.params,
        "data_note": (
            "Trained on data/interim/default/train_cleaned.csv only. "
            "The validation and test splits were not read by this script."
        ),
        "sequence_layout": {
            "steps": N_STEPS,
            "order": "oldest month first: suffix 6 (April 2005) to suffix 1 (September 2005)",
            "step_columns": step_columns,
            "static_columns": list(STATIC_COLUMNS),
            "target": TARGET,
        },
        "wall_time_seconds": round(wall_time, 1),
        "train_seconds_reported_by_model": round(model.train_seconds, 1),
        "torch_threads": torch.get_num_threads(),
        "n_train_rows": len(train),
        "n_positive_rows": positives,
        "n_negative_rows": negatives,
        "fill_level": FILL_LEVEL,
        "fill_count": n,
        "loss_final": final,
        "diagnostic_unfiltered_sample": diagnostic,
        "fill_sampler": {
            "seed": args.seed,
            "filter": "no DEFAULT_RULES broken and y == 1",
            "drop_duplicates_of": "the training split",
            "rounds": tally["rounds"],
            "rows_drawn": tally["drawn"],
            "rows_passing_filter": tally["kept"],
            "keep_rate": keep_rate,
            "rows_returned": len(fill),
            "sampling_seconds": round(sampling_seconds, 2),
        },
        "fill_file": _display(FILL_FILE),
        "fill_file_checks": checks,
        "library_versions": {
            "python": platform.python_version(),
            "torch": _lib_version("torch"),
            "numpy": _lib_version("numpy"),
            "pandas": _lib_version("pandas"),
            "matplotlib": _lib_version("matplotlib"),
        },
    }
    meta_path = ARTIFACT_DIR / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("run_meta.json written to %s", meta_path)
    print(f"\nWall time: {wall_time:.1f} s")
    print(f"Artifacts: {ARTIFACT_DIR}")
    print(f"Fill file: {FILL_FILE}")


if __name__ == "__main__":
    main()
