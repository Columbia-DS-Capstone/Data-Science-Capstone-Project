"""One results table and chart across datasets.

Each row is a dataset, arm and fill level: the mean test score over seeds, its seed-to-seed spread,
the paired lift over the reference arm with its paired-bootstrap interval (`lo`, `hi`), and the
reliable-gain verdict from `tools/evaluation.py`. The input is either finished runs'
report/comparisons.csv or the training agent's metrics table (`results/training_all_arms.csv`,
from `tools/training_grid.py`).

With one seed the verdicts come from the bootstrap, which needs each model's saved scores (in the
run folders, which are not committed). `--scores` computes it from them and `--bootstrap-out`
saves the intervals, which are aggregates and safe to commit. `--bootstrap` rebuilds the table
from saved intervals alone, so IEEE-CIS can be bootstrapped where its files are:

    uv run python -m agentic_gan.tools.cross_dataset --metrics results/training_all_arms.csv \
        --scores bank_marketing=runs/<bank> --scores polish=runs/<polish> \
        --bootstrap-out results/bootstrap_bank_polish.csv
    uv run python -m agentic_gan.tools.cross_dataset --metrics results/training_all_arms.csv \
        --bootstrap results/bootstrap_*.csv \
        --out results/cross_dataset.csv --plot results/cross_dataset.png
    uv run python -m agentic_gan.tools.cross_dataset runs/<bank> runs/<polish> \
        --out results/cross_dataset.csv
"""

import argparse
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_gan.tools.evaluation import ARM_ORDER, compare_arms, reliable_gain

COLUMNS = [
    "dataset",
    "arm",
    "fill_level",
    "n_seeds",
    "test_mean",
    "test_sd",
    "lift",
    "lo",
    "hi",
    "verdict",
]
KEY = ["dataset", "arm", "fill_level", "split", "metric"]
BOOT = ["boot_seed", "boot_diff", "boot_lo", "boot_hi", "n_boot"]


# Okabe-Ito hues, in a fixed order per arm (checked for colour-blind separation); each arm also
# has its own marker shape, so colour is never the only cue.
ARM_COLORS = {"original": "#CC79A7", "cleaned": "#E69F00", "smotenc": "#0072B2", "ctgan": "#009E73"}
ARM_MARKERS = {"original": "s", "cleaned": "D", "smotenc": "o", "ctgan": "^"}
DATASET_NAMES = {
    "bank_marketing": "Bank Marketing",
    "default": "Default",
    "polish": "Polish",
    "ieee_cis": "IEEE-CIS",
}
METRIC_TITLES = {"pr_auc": "Test PR-AUC", "f1": "Test F1 (frozen threshold)"}


def comparisons_from_metrics(
    metrics: pd.DataFrame,
    *,
    reference: str = "cleaned",
    models_dir: str | Path | Mapping[str, str | Path] | None = None,
    bootstrap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The evaluation tool's comparison table for a training agent metrics table.

    The bootstrap columns come from the saved scores (`models_dir`, see `bootstrap_table`) or from
    an interval table saved earlier (`bootstrap`, with a `reference` column), not both.
    """
    if models_dir is not None and bootstrap is not None:
        raise ValueError("give the saved scores or a saved bootstrap table, not both")
    table = compare_arms(metrics, reference=reference, models_dir=models_dir)
    if bootstrap is not None:
        saved = bootstrap[bootstrap["reference"].eq(reference)].drop(columns="reference")
        table = table.merge(saved, on=KEY, how="left")
    return table.assign(reference=reference)


def table_from_comparisons(
    comparisons: pd.DataFrame, *, reference: str = "cleaned", metric: str = "pr_auc"
) -> pd.DataFrame:
    """Keep one test-score row per dataset, arm and fill level, with the reliable-gain verdict."""
    chosen = comparisons[comparisons.reference.eq(reference) & comparisons.metric.eq(metric)]
    rows = []
    for (dataset, arm, level), group in chosen.groupby(
        ["dataset", "arm", "fill_level"], sort=False
    ):
        test = group[group.split.eq("test")]
        if test.empty:
            continue
        test = test.iloc[0]
        if arm == reference:
            verdict, lift = "reference", float("nan")
        else:
            verdict = reliable_gain(
                chosen, dataset=dataset, arm=arm, fill_level=level, metric=metric
            ).verdict
            lift = test.diff_mean
        low, high = test.get("boot_lo", np.nan), test.get("boot_hi", np.nan)
        rows.append(
            [
                dataset,
                arm,
                level,
                int(test.n_seeds),
                test["mean"],
                test.sd,
                lift,
                low,
                high,
                verdict,
            ]
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def cross_dataset_table(
    run_dirs: Iterable[Path], *, reference: str = "cleaned", metric: str = "pr_auc"
) -> pd.DataFrame:
    """Stack the runs' comparisons and keep one test-score row per dataset, arm and fill level."""
    comparisons = pd.concat(
        [pd.read_csv(Path(run) / "report" / "comparisons.csv") for run in run_dirs],
        ignore_index=True,
    )
    return table_from_comparisons(comparisons, reference=reference, metric=metric)


def _bars(points: pd.DataFrame):
    """Error bars: the bootstrap interval of the lift where there is one, else the seed spread.

    The interval is drawn around the reference line, so a bar that crosses the dotted line is a
    lift whose interval holds 0.
    """
    if "boot_lo" in points and points["boot_lo"].notna().any():
        below = (points["boot_diff"] - points["boot_lo"]).fillna(0).clip(lower=0)
        above = (points["boot_hi"] - points["boot_diff"]).fillna(0).clip(lower=0)
        return np.vstack([below, above])
    return points["sd"].fillna(0)


def plot_cross_dataset(
    comparisons: pd.DataFrame, path: Path, *, reference: str = "cleaned"
) -> None:
    """Test PR-AUC (top) and F1 (bottom) against fill level, one panel per dataset.

    The original and cleaned arms are single points at 0%; SMOTENC and CTGAN are lines over the
    fill levels. The bars are the paired-bootstrap interval of the lift over the reference arm
    (PR-AUC) where `comparisons` has one, otherwise the seed-to-seed standard deviation. A dotted
    line marks the reference arm, which the synthetic rows are added to.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    test = comparisons[comparisons.split.eq("test")]
    datasets = list(dict.fromkeys(test.dataset))
    interval = "boot_lo" in test and test["boot_lo"].notna().any()
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.7 * len(datasets), 6.6), squeeze=False)
    for row, metric in enumerate(METRIC_TITLES):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            rows = test[test.dataset.eq(dataset) & test.metric.eq(metric)]
            base = rows[rows.arm.eq(reference)]
            if not base.empty:
                ax.axhline(base["mean"].iloc[0], color=ARM_COLORS[reference], ls=":", lw=1.2)
            for arm in ARM_ORDER:
                points = rows[rows.arm.eq(arm)].sort_values("fill_level")
                if points.empty:
                    continue
                single = arm in ("original", "cleaned")
                ax.errorbar(
                    [0.0] * len(points) if single else points.fill_level * 100,
                    points["mean"],
                    yerr=_bars(points),
                    color=ARM_COLORS[arm],
                    marker=ARM_MARKERS[arm],
                    ms=7 if single else 6,
                    lw=0 if single else 1.8,
                    capsize=3,
                    elinewidth=1.5,
                    mec="white",
                    mew=0.8,
                )
            levels = sorted({0.0, *rows["fill_level"].mul(100).tolist()})
            ax.set_xticks(levels, [f"{level:g}%" for level in levels])
            if len(levels) > 4:
                ax.tick_params(axis="x", labelrotation=60, labelsize=8)
            ax.grid(axis="y", color="#d0d0d0", lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                seeds = int(rows["n_seeds"].max()) if not rows.empty else 0
                name = DATASET_NAMES.get(dataset, dataset)
                label = f"{seeds} seed{'s' if seeds != 1 else ''}"
                if interval and not rows["boot_lo"].notna().any():
                    label += ", no interval"  # this dataset's saved scores were not available
                ax.set_title(f"{name} ({label})", fontsize=11)
            if col == 0:
                ax.set_ylabel(METRIC_TITLES[metric])
            if row == 1:
                ax.set_xlabel("Fill level (synthetic rows added)")
    present = [arm for arm in ARM_ORDER if test.arm.eq(arm).any()]
    handles = [
        Line2D(
            [],
            [],
            color=ARM_COLORS[a],
            marker=ARM_MARKERS[a],
            ls="-" if a in ("smotenc", "ctgan") else "",
            ms=7,
            label=a,
        )
        for a in present
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), max(2, 2 * len(datasets))),
        frameon=False,
    )
    bars = (
        f"PR-AUC bars = 95% paired-bootstrap interval of the lift over {reference}, drawn around "
        f"the {reference} line (F1 has none). A verdict needs the interval clear of 0 on "
        "validation as well as test"
        if interval
        else "mean over seeds, bars = standard deviation"
    )
    note = (
        f"Test scores. {bars}.\n"
        f"Dotted line = the {reference} arm that the synthetic rows are added to. "
        "Each panel has its own scale."
    )
    note = textwrap.fill(note, width=max(55, int(fig.get_figwidth() * 18)))
    fig.text(0.5, 0.006, note, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _score_folders(pairs: list[str]) -> dict[str, Path]:
    """`DATASET=RUN_FOLDER` arguments as {dataset: that run's models folder}."""
    folders = {}
    for pair in pairs:
        dataset, _, folder = pair.partition("=")
        if not dataset or not folder:
            raise SystemExit(f"--scores wants DATASET=RUN_FOLDER, not {pair!r}")
        path = Path(folder)
        folders[dataset] = path / "models" if (path / "models").is_dir() else path
    return folders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="*", type=Path, help="finished run folders")
    parser.add_argument("--metrics", type=Path, help="a training agent metrics table instead")
    parser.add_argument("--reference", default="cleaned", help="arm the lifts are measured against")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="use only these model seeds (the one-seed table: 42)"
    )
    parser.add_argument("--out", type=Path, help="write the table here as CSV")
    parser.add_argument("--plot", type=Path, help="write the chart here as PNG (needs --metrics)")
    parser.add_argument(
        "--scores",
        action="append",
        default=[],
        metavar="DATASET=RUN_FOLDER",
        help="a run folder with the saved scores of this dataset's models (repeat per dataset)",
    )
    parser.add_argument(
        "--bootstrap", nargs="+", type=Path, help="bootstrap interval tables saved earlier"
    )
    parser.add_argument(
        "--bootstrap-out", type=Path, help="save the intervals computed from --scores (aggregates)"
    )
    args = parser.parse_args()
    if bool(args.metrics) == bool(args.run_dirs):
        parser.error("give either run folders or --metrics")
    if args.plot and not args.metrics:
        parser.error("--plot needs --metrics")
    if (args.scores or args.bootstrap) and not args.metrics:
        parser.error("--scores and --bootstrap need --metrics")
    if args.scores and args.bootstrap:
        parser.error("give --scores or --bootstrap, not both")
    if args.bootstrap_out and not args.scores:
        parser.error("--bootstrap-out needs --scores")
    if args.metrics:
        saved = (
            pd.concat(
                [pd.read_csv(path) for path in args.bootstrap], ignore_index=True
            ).drop_duplicates([*KEY, "reference"], keep="last")
            if args.bootstrap
            else None
        )
        metrics = pd.read_csv(args.metrics)
        if args.seeds:
            metrics = metrics[metrics["seed"].isin(args.seeds)]
        comparisons = comparisons_from_metrics(
            metrics,
            reference=args.reference,
            models_dir=_score_folders(args.scores) or None,
            bootstrap=saved,
        )
        if args.bootstrap_out:
            computed = comparisons[comparisons["boot_lo"].notna()]
            args.bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
            computed[[*KEY, *BOOT]].assign(reference=args.reference).to_csv(
                args.bootstrap_out, index=False
            )
        table = table_from_comparisons(comparisons, reference=args.reference)
    else:
        table = cross_dataset_table(args.run_dirs, reference=args.reference)
    print(table.round(4).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    if args.plot:
        plot_cross_dataset(comparisons, args.plot, reference=args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
