"""Train CTGAN on the cleaned UCI Bank Marketing training split and write synthetic subscribers.

Input is cleaned table (data/interim/bank_marketing/train_cleaned.csv, built by
agentic_gan.tools.preprocessing). It has no missing values: ``pdays`` keeps the -1 sentinel
for customers never contacted before, and ``source_row`` (a row ID) is dropped here.
The target and discrete columns come from the shared dataset settings.

Run from the repo root:
    uv run python eda/ctgan_bank/run_ctgan_bank.py [--epochs 300]
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from agentic_gan.tools.ctgan import fill_counts, sample_synthetic, save_model, train_ctgan
from agentic_gan.utils.datasets import DATASETS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DATASETS["bank_marketing"]
TRAIN_FILE = ROOT / "data" / "interim" / CONFIG.name / "train_cleaned.csv"
SYNTHETIC_DIR = ROOT / "data" / "interim" / CONFIG.name / "synthetic"
ARTIFACT_DIR = ROOT / "eda" / "ctgan_bank" / "artifacts"
ID_COLUMN = "source_row"
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

    train = pd.read_csv(TRAIN_FILE).drop(columns=ID_COLUMN)

    real_pass = float(consistent_rows(train).mean())
    assert real_pass == 1.0, f"consistency rules fail on {1 - real_pass:.2%} of real rows"

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = train_ctgan(
        train, CONFIG.target, discrete_columns=list(CONFIG.discrete),
        epochs=args.epochs, device=args.device, seed=args.seed,
        log_path=ARTIFACT_DIR / "ctgan_losses.csv",
    )
    save_model(model, ARTIFACT_DIR / "ctgan_bank.pkl")

    counts = fill_counts(train[CONFIG.target])
    pool = sample_synthetic(model, max(counts.values()), target_value=1, seed=args.seed,
                            drop_duplicates_of=train, row_filter=consistent_rows)

    # Nested subsets: the 10% and 25% files are the first rows of the 50% file.
    written = {}
    for fraction, n in counts.items():
        path = SYNTHETIC_DIR / f"bank_ctgan_fill{int(fraction * 100)}.csv"
        pool.head(n).to_csv(path, index=False)
        written[str(fraction)] = {"rows": n, "file": str(path.relative_to(ROOT))}

    meta = {
        "input_file": str(TRAIN_FILE.relative_to(ROOT)),
        "n_train_rows": model.n_train_rows,
        "n_train_positives": int(train[CONFIG.target].sum()),
        "params": model.params,
        "train_seconds": round(model.train_seconds, 1),
        "fills": written,
        "never_contacted_share_real": round(float((train["pdays"] == PDAYS_SENTINEL).mean()), 4),
        "never_contacted_share_synthetic": round(
            float((pool["pdays"] == PDAYS_SENTINEL).mean()), 4),
    }
    (ARTIFACT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
