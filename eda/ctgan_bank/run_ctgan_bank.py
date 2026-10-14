"""Train CTGAN on the UCI Bank Marketing training split and write synthetic subscriber rows.

Input is Raghav's ``X_train`` (eda/baseline-modeling/uci_banking/scripts/uci_preprocessing.py).
``pdays`` is NaN there for customers never contacted before; CTGAN cannot take NaN, so it is
temporarily filled with the original -1 sentinel and restored to NaN after sampling.
Switch the input to Noor's cleaned table (#29) when it is ready and drop the pdays handling.

Run from the repo root:
    uv run python eda/ctgan_bank/run_ctgan_bank.py [--epochs 300]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eda" / "baseline-modeling" / "uci_banking" / "scripts"))

from uci_preprocessing import load_and_preprocess_uci  # noqa: E402

from agentic_gan.tools.ctgan import (  # noqa: E402
    fill_counts,
    sample_synthetic,
    save_model,
    train_ctgan,
)

SYNTHETIC_DIR = ROOT / "data" / "interim" / "bank_marketing" / "synthetic"
ARTIFACT_DIR = ROOT / "eda" / "ctgan_bank" / "artifacts"
PDAYS_SENTINEL = -1


def consistent_rows(df):
    """Bank rules that hold for every real training row but that CTGAN does not enforce.

    ``previously_contacted`` must agree with ``previous`` and ``pdays``, and ``pdays`` is
    either the -1 sentinel or at least 1 day.
    """
    contacted = df["previously_contacted"] == 1
    never = df["pdays"] == PDAYS_SENTINEL
    agrees = (never == ~contacted) & ((df["previous"] > 0) == contacted)
    return agrees & (never | (df["pdays"] >= 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data = load_and_preprocess_uci(seed=args.seed)
    train = data.X_train.copy()
    train["y"] = data.y_train.values

    # Temporary adapter: CTGAN cannot take NaN. previously_contacted keeps the information.
    train_filled = train.fillna({"pdays": PDAYS_SENTINEL})

    real_pass = float(consistent_rows(train_filled).mean())
    assert real_pass == 1.0, f"consistency rules fail on {1 - real_pass:.2%} of real rows"

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = train_ctgan(
        train_filled, "y", epochs=args.epochs, device=args.device, seed=args.seed,
        log_path=ARTIFACT_DIR / "ctgan_losses.csv",
    )
    save_model(model, ARTIFACT_DIR / "ctgan_bank.pkl")

    counts = fill_counts(data.y_train)
    largest = max(counts.values())
    pool = sample_synthetic(model, largest, target_value=1, seed=args.seed,
                            drop_duplicates_of=train_filled, row_filter=consistent_rows)

    # Restore the NaN that was filled in for training: never-contacted customers.
    pool.loc[pool["pdays"] == PDAYS_SENTINEL, "pdays"] = np.nan

    # Nested subsets: the 10% and 25% files are the first rows of the 50% file.
    written = {}
    for fraction, n in counts.items():
        path = SYNTHETIC_DIR / f"bank_ctgan_fill{int(fraction * 100)}.csv"
        pool.head(n).to_csv(path, index=False)
        written[str(fraction)] = {"rows": n, "file": str(path.relative_to(ROOT))}

    meta = {
        "n_train_rows": model.n_train_rows,
        "n_train_positives": int(data.y_train.sum()),
        "params": model.params,
        "train_seconds": round(model.train_seconds, 1),
        "fills": written,
        "pdays_nan_share_real": round(float(train["pdays"].isna().mean()), 4),
        "pdays_nan_share_synthetic": round(float(pool["pdays"].isna().mean()), 4),
    }
    (ARTIFACT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
