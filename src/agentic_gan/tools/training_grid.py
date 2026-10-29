"""Run the Training Agent on a real dataset's files, with every arm, and time it.

Builds a run folder and a ``TrainingRequest`` from the files that already exist (the raw source,
Noor's cleaned splits in ``data/interim/<dataset>/`` and, optionally, CTGAN files), calls
``run_training`` and checks that no synthetic row reached validation or test.

    uv run python -m agentic_gan.tools.training_grid --dataset bank_marketing --seeds 42 \
        --ctgan-prefix data/interim/bank_marketing/synthetic/bank_ctgan_v2_fill

Without ``--ctgan-prefix`` the CTGAN arm is not run (original, cleaned and SMOTENC only).
"""

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

from agentic_gan.agents.training import run_training
from agentic_gan.orchestrator.manifest import RunDir, create_run
from agentic_gan.schemas import (
    ArmFiles,
    RunConfig,
    Status,
    SyntheticSet,
    TrainingOutput,
    TrainingRequest,
    fill_rows,
)
from agentic_gan.tools.preprocessing import load_dataset, normalize, split_dataset
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import IS_SYNTHETIC

ROOT = Path(__file__).resolve().parents[3]
FILL_LEVELS = (0.10, 0.25, 0.50)
PARTS = {"train": "train_path", "validation": "val_path", "test": "test_path"}


def build_request(
    dataset: str,
    seeds: list[int],
    fill_levels: tuple[float, ...] = FILL_LEVELS,
    ctgan_prefix: str | None = None,
    runs_dir: Path = ROOT / "runs",
) -> tuple[TrainingRequest, RunDir]:
    """Create the run folder and the request for one dataset.

    ``ctgan_prefix`` + the fill level as a whole percent + ``.csv`` names each CTGAN file.
    """
    config = DATASETS[dataset]
    if config.name == "ieee_cis":
        raise NotImplementedError("IEEE-CIS needs its local files and a time split; not wired yet")
    source = ROOT / config.file
    cfg = RunConfig(
        dataset_name=dataset,
        dataset_path=source,
        target=config.target,
        positive_class="yes" if dataset == "bank_marketing" else 1,
        seeds=seeds,
        fill_levels=list(fill_levels),
    )
    run = create_run(cfg, runs_dir=runs_dir)

    raw = load_dataset(config, source)
    splits = split_dataset(normalize(raw, config, config.include_duration), config)
    arms = {}
    for arm, tables in (
        ("raw", {part: raw.loc[ids] for part, ids in splits.items()}),
        (
            "baseline",
            {
                part: pd.read_csv(ROOT / "data" / "interim" / dataset / f"{part}_cleaned.csv")
                for part in PARTS
            },
        ),
    ):
        paths = {}
        for part, table in tables.items():
            path = run.subdir("data") / f"{arm}_{part}.csv"
            if arm == "raw":  # the raw table's index is the source row; the cleaned CSV has it
                table.to_csv(path, index_label="source_row")
            else:
                table.to_csv(path, index=False)
            paths[PARTS[part]] = path
        arms[arm] = ArmFiles(**paths)

    accepted = []
    if ctgan_prefix:
        train = pd.read_csv(ROOT / "data" / "interim" / dataset / "train_cleaned.csv")
        positives = int(train[config.target].sum())
        for level in fill_levels:
            file = ROOT / f"{ctgan_prefix}{round(level * 100)}.csv"
            copy = run.subdir("synthetic") / file.name
            shutil.copy(file, copy)
            rows = fill_rows(level, positives, len(train) - positives)
            accepted.append(SyntheticSet(fill_level=level, rows=rows, path=copy))
    return TrainingRequest(config=cfg, **arms, accepted_synthetic=accepted), run


def check_no_synthetic_in_holdouts(result: TrainingOutput) -> dict[str, int]:
    """Count synthetic rows in every saved validation and test prediction file (must be 0)."""
    counts = {"validation": 0, "test": 0}
    for record in result.records:
        if record.status is not Status.SUCCESS:
            continue
        for part, path in (
            ("validation", record.val_predictions_path),
            ("test", record.test_predictions_path),
        ):
            counts[part] += int(pd.read_csv(path)[IS_SYNTHETIC].sum())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bank_marketing", choices=sorted(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--fill-levels", type=float, nargs="+", default=list(FILL_LEVELS))
    parser.add_argument("--ctgan-prefix", help="path prefix of the CTGAN files, up to the level")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    request, run = build_request(
        args.dataset, args.seeds, tuple(args.fill_levels), args.ctgan_prefix, args.runs_dir
    )
    start = time.perf_counter()
    result = run_training(request, run)
    seconds = time.perf_counter() - start
    leaked = check_no_synthetic_in_holdouts(result)

    print(f"status: {result.status.value}  fits: {len(result.records)}  run time: {seconds:.0f}s")
    print(f"synthetic rows in validation/test predictions: {leaked}")
    print(f"results: {result.metrics_path}")
    failed = [(r.arm.value, r.model, r.reason) for r in result.records if r.status is Status.FAILED]
    if failed:
        print("failed fits:", failed[:5])


if __name__ == "__main__":
    main()
