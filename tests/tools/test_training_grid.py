from types import SimpleNamespace

import pandas as pd
import pytest

from agentic_gan.schemas import Status
from agentic_gan.tools.training_grid import (
    ROOT,
    build_request,
    check_no_synthetic_in_holdouts,
    save_results,
)


def _record(tmp_path, name, val_flags, test_flags, status=Status.SUCCESS):
    paths = {}
    for part, flags in (("val", val_flags), ("test", test_flags)):
        paths[part] = tmp_path / f"{name}_{part}.csv"
        pd.DataFrame({"is_synthetic": flags}).to_csv(paths[part], index=False)
    return SimpleNamespace(
        status=status, val_predictions_path=paths["val"], test_predictions_path=paths["test"]
    )


def test_holdout_check_counts_synthetic_rows(tmp_path):
    clean = _record(tmp_path, "a", [0, 0, 0], [0, 0])
    leaky = _record(tmp_path, "b", [0, 1, 1], [1, 0])
    assert check_no_synthetic_in_holdouts(SimpleNamespace(records=[clean])) == {
        "validation": 0,
        "test": 0,
    }
    assert check_no_synthetic_in_holdouts(SimpleNamespace(records=[clean, leaky])) == {
        "validation": 2,
        "test": 1,
    }


def test_holdout_check_ignores_records_that_did_not_fit(tmp_path):
    failed = SimpleNamespace(
        status=Status.FAILED, val_predictions_path=None, test_predictions_path=None
    )
    assert check_no_synthetic_in_holdouts(SimpleNamespace(records=[failed])) == {
        "validation": 0,
        "test": 0,
    }


def _result(tmp_path, dataset, rows):
    path = tmp_path / f"{dataset}_metrics.csv"
    pd.DataFrame({"dataset": dataset, "arm": "cleaned", "pr_auc": rows}).to_csv(path, index=False)
    return SimpleNamespace(metrics_path=path, records=[object()] * len(rows))


def test_save_results_replaces_one_dataset_and_keeps_the_others(tmp_path):
    out = tmp_path / "results"
    save_results(_result(tmp_path, "bank", [0.1, 0.2]), "bank", [42], 5.0, out)
    save_results(_result(tmp_path, "polish", [0.8]), "polish", [42, 43], 7.0, out)
    save_results(_result(tmp_path, "bank", [0.3]), "bank", [42], 6.0, out)

    table = pd.read_csv(out / "training_all_arms.csv")
    assert sorted(table["dataset"]) == ["bank", "polish"]
    assert table.loc[table["dataset"] == "bank", "pr_auc"].tolist() == [0.3]

    times = pd.read_csv(out / "training_run_times.csv").set_index("dataset")
    assert times.loc["bank", "seconds"] == 6.0
    assert times.loc["polish", "seeds"] == 2
    assert len(times) == 2


@pytest.mark.skipif(
    (ROOT / "data" / "external" / "train_transaction.csv").exists(),
    reason="the IEEE-CIS files are present",
)
def test_ieee_cis_needs_its_local_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_request("ieee_cis", [42], runs_dir=tmp_path)


def test_fine_grid_uses_nested_prefixes_of_one_pool(tmp_path):
    levels = (0.01, 0.025, 0.05, 0.10, 0.25)
    request, run = build_request(
        "bank_marketing",
        [42],
        levels,
        "data/interim/bank_marketing/synthetic/bank_ctgan_v2_fill",
        tmp_path,
        synthetic_weight=0.25,
    )
    assert request.config.synthetic_weight == 0.25
    pool = pd.read_csv(ROOT / "data/interim/bank_marketing/synthetic/bank_ctgan_v2_fill25.csv")
    for item in request.accepted_synthetic:
        actual = pd.read_csv(run.root / item.path)
        pd.testing.assert_frame_equal(actual, pool.head(item.rows))
    assert "2.5" in request.accepted_synthetic[1].path.name


def test_save_results_keeps_distinct_synthetic_weights(tmp_path):
    out = tmp_path / "weighted_results"
    for weight in (1.0, 0.25, 0.5):
        result = _result(tmp_path, "bank", [weight])
        metrics = pd.read_csv(result.metrics_path).assign(synthetic_weight=weight)
        metrics.to_csv(result.metrics_path, index=False)
        save_results(result, "bank", [42], 5.0, out)
    saved = pd.read_csv(out / "training_all_arms.csv")
    assert set(saved.synthetic_weight) == {1.0, 0.25, 0.5}
    assert len(saved) == 3
    times = pd.read_csv(out / "training_run_times.csv")
    assert set(times.synthetic_weight) == {1.0, 0.25, 0.5}
