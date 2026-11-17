"""The cross-dataset table built from runs' comparisons.csv files."""

import sys

import numpy as np
import pandas as pd
import pytest
from test_evaluation import fabricated_run, results

from agentic_gan.tools import cross_dataset
from agentic_gan.tools.cross_dataset import (
    BOOT,
    COLUMNS,
    KEY,
    comparisons_from_metrics,
    cross_dataset_table,
    plot_cross_dataset,
    table_from_comparisons,
)
from agentic_gan.tools.evaluation import compare_arms


def write_run(tmp_path, name, metrics):
    """A run folder whose report/comparisons.csv is what the evaluation agent writes."""
    comparison = compare_arms(metrics, reference="cleaned").assign(reference="cleaned")
    folder = tmp_path / name / "report"
    folder.mkdir(parents=True)
    comparison.to_csv(folder / "comparisons.csv", index=False)
    return tmp_path / name


def test_one_row_per_dataset_arm_and_level_with_the_verdict(tmp_path):
    bank = results()
    other = results().assign(dataset="polish")
    other = other[~other.arm.eq("ctgan")]  # a dataset where the GAN arm was not available
    runs = [write_run(tmp_path, "bank", bank), write_run(tmp_path, "polish", other)]

    table = cross_dataset_table(runs)

    assert list(table.columns) == COLUMNS
    keys = table[["dataset", "arm", "fill_level"]].apply(tuple, axis=1)
    assert not keys.duplicated().any()
    assert set(table.dataset) == {"bank_marketing", "polish"}
    assert not (table.dataset.eq("polish") & table.arm.eq("ctgan")).any()
    bank_ctgan = table.query("dataset == 'bank_marketing' and arm == 'ctgan'")
    row = bank_ctgan[bank_ctgan.fill_level.eq(0.25)].iloc[0]
    # The fixture's CTGAN arm beats original by 0.02 on test, and cleaned equals original.
    assert row.lift == pytest.approx(0.02, abs=0.003)
    assert row.verdict == "reliable gain" and row.n_seeds == 5
    assert table.query("arm == 'cleaned'").verdict.eq("reference").all()
    assert table.query("arm == 'cleaned'").lift.isna().all()
    loss = table.query("dataset == 'bank_marketing' and arm == 'smotenc' and fill_level == 0.5")
    assert loss.verdict.iloc[0] == "reliable loss"
    assert np.isclose(
        table.query("arm == 'original'").test_mean.iloc[0], np.mean([0.48, 0.50, 0.46, 0.49, 0.47])
    )


def test_table_from_a_metrics_table_matches_the_run_folder_route(tmp_path):
    metrics = results()
    from_runs = cross_dataset_table([write_run(tmp_path, "bank", metrics)])
    from_metrics = table_from_comparisons(comparisons_from_metrics(metrics))
    assert list(from_metrics.columns) == COLUMNS
    pd.testing.assert_frame_equal(from_metrics, from_runs)


def test_a_one_seed_dataset_has_no_reliable_verdict():
    metrics = results()
    single = metrics[metrics.seed.eq(42)].assign(dataset="ieee_cis")
    table = table_from_comparisons(comparisons_from_metrics(single))
    verdicts = table[~table.arm.eq("cleaned")].verdict
    assert verdicts.eq("not enough paired seeds").all()


def test_the_chart_has_one_column_per_dataset_and_is_written(tmp_path):
    metrics = pd.concat([results(), results().assign(dataset="polish")], ignore_index=True)
    out = tmp_path / "charts" / "cross.png"
    plot_cross_dataset(comparisons_from_metrics(metrics), out)
    assert out.stat().st_size > 10_000


def test_command_line_writes_the_table_and_the_chart(tmp_path):
    metrics = tmp_path / "metrics.csv"
    results().to_csv(metrics, index=False)
    table, chart = tmp_path / "t.csv", tmp_path / "c.png"
    argv = ["--metrics", str(metrics), "--out", str(table), "--plot", str(chart)]
    old = sys.argv
    sys.argv = ["cross_dataset", *argv]
    try:
        assert cross_dataset.main() == 0
    finally:
        sys.argv = old
    assert list(pd.read_csv(table).columns) == COLUMNS and chart.exists()


# --- one seed: bootstrap intervals (issue 138) ---------------------------------------------------


def one_seed_table(tmp_path, metrics=None, **options):
    """The table for a one-seed run, with the original arm as the base."""
    metrics = fabricated_run(tmp_path) if metrics is None else metrics
    comparisons = comparisons_from_metrics(metrics, reference="original", **options)
    return comparisons, table_from_comparisons(comparisons, reference="original")


def test_one_seed_rows_get_an_interval_and_a_verdict_from_the_saved_scores(tmp_path):
    comparisons, table = one_seed_table(tmp_path, models_dir=tmp_path)

    assert list(table.columns) == COLUMNS and table.n_seeds.max() == 1
    gain = table.query("arm == 'ctgan' and fill_level == 0.25").iloc[0]
    assert gain.verdict == "reliable gain" and 0 < gain.lo < gain.lift < gain.hi
    loss = table.query("arm == 'smotenc' and fill_level == 0.25").iloc[0]
    assert loss.verdict == "reliable loss" and loss.lo < loss.lift < loss.hi < 0
    reference = table.query("arm == 'original'").iloc[0]
    assert reference.verdict == "reference" and np.isnan([reference.lo, reference.hi]).all()


def test_saved_intervals_rebuild_the_same_table_without_the_scores(tmp_path):
    comparisons, table = one_seed_table(tmp_path, models_dir=tmp_path)
    saved = comparisons[comparisons.boot_lo.notna()][[*KEY, *BOOT]].assign(reference="original")

    _, rebuilt = one_seed_table(tmp_path, bootstrap=saved)  # no models_dir: the scores are not read

    pd.testing.assert_frame_equal(rebuilt, table)
    with pytest.raises(ValueError, match="not both"):
        comparisons_from_metrics(
            fabricated_run(tmp_path), reference="original", models_dir=tmp_path, bootstrap=saved
        )


def test_a_dataset_without_saved_scores_keeps_no_verdict(tmp_path):
    bank = fabricated_run(tmp_path)
    metrics = pd.concat([bank, bank.assign(dataset="ieee_cis")], ignore_index=True)

    _, table = one_seed_table(tmp_path, metrics, models_dir={"bank_marketing": tmp_path})

    ieee = table[table.dataset.eq("ieee_cis") & ~table.arm.eq("original")]
    assert ieee.verdict.eq("not enough paired seeds").all() and ieee.lo.isna().all()
    assert table[table.dataset.eq("bank_marketing") & table.arm.eq("ctgan")].lo.notna().all()


def test_the_chart_draws_the_bootstrap_interval_when_there_is_one(tmp_path):
    comparisons, _ = one_seed_table(tmp_path, models_dir=tmp_path)
    out = tmp_path / "interval.png"

    plot_cross_dataset(comparisons, out, reference="original")

    assert out.stat().st_size > 10_000


def test_command_line_saves_the_intervals_and_rebuilds_the_table_from_them(tmp_path):
    metrics, scores = tmp_path / "metrics.csv", tmp_path / "scores"
    scores.mkdir()
    fabricated_run(scores).to_csv(metrics, index=False)
    saved, direct, rebuilt = tmp_path / "boot.csv", tmp_path / "a.csv", tmp_path / "b.csv"

    def run(*argv):
        old = sys.argv
        sys.argv = ["cross_dataset", "--metrics", str(metrics), "--reference", "original", *argv]
        try:
            assert cross_dataset.main() == 0
        finally:
            sys.argv = old

    run("--scores", f"bank_marketing={scores}", "--bootstrap-out", str(saved), "--out", str(direct))
    run("--bootstrap", str(saved), "--out", str(rebuilt))

    assert pd.read_csv(saved).reference.eq("original").all()
    assert "probability" not in pd.read_csv(saved)  # aggregates only: no row-level scores
    pd.testing.assert_frame_equal(pd.read_csv(direct), pd.read_csv(rebuilt))
    assert pd.read_csv(direct).query("arm == 'ctgan' and fill_level == 0.25").lo.iloc[0] > 0


def test_command_line_can_keep_one_seed_of_a_multi_seed_metrics_table(tmp_path):
    metrics = tmp_path / "metrics.csv"
    fabricated_run(tmp_path, seeds=(42, 43)).to_csv(metrics, index=False)
    out = tmp_path / "table.csv"
    old = sys.argv
    sys.argv = ["cross_dataset", "--metrics", str(metrics), "--reference", "original"]
    sys.argv += ["--seeds", "42", "--out", str(out)]
    try:
        assert cross_dataset.main() == 0
    finally:
        sys.argv = old
    assert pd.read_csv(out).n_seeds.eq(1).all()
