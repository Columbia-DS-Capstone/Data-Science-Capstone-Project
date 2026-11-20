"""Evaluation tools: arm comparison, the reliable-gain rule, segment metrics, number check."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import agentic_gan.tools.evaluation as evaluation_module
from agentic_gan.agents.training import _save_metrics, model_artifact_stem
from agentic_gan.schemas import Arm, ModelRecord, Status
from agentic_gan.tools.baselines import pr_auc_score
from agentic_gan.tools.evaluation import (
    BootstrapResult,
    compare_arms,
    load_selected_scores,
    paired_bootstrap,
    reliable_gain,
    segment_metrics,
    verify_report_numbers,
)

SEEDS = [42, 43, 44, 45, 46]
ORIGINAL = {
    "validation": [0.50, 0.52, 0.48, 0.51, 0.49],
    "test": [0.48, 0.50, 0.46, 0.49, 0.47],
}
# Paired difference from the original arm, per seed, by split.
NOISE = [0.002, -0.002, 0.001, -0.001, 0.0]
LIFTS = {
    ("smotenc", 0.25): {  # swings both ways: no lift beyond the noise
        "validation": [0.02, -0.02, 0.03, -0.03, 0.0],
        "test": [0.0, 0.01, -0.01, 0.0, 0.0],
    },
    ("smotenc", 0.5): {  # about -0.05 and -0.04 on every seed
        "validation": [-0.05 + n for n in NOISE],
        "test": [-0.04 + n for n in NOISE],
    },
    ("ctgan", 0.25): {  # about +0.03 and +0.02 on every seed
        "validation": [0.03 + n for n in NOISE],
        "test": [0.02 + n for n in NOISE],
    },
    ("ctgan", 0.5): {  # reliable on validation only
        "validation": [0.03 + n for n in NOISE],
        "test": [0.03, -0.02, 0.04, -0.03, 0.01],
    },
}


def arm_rows(arm, fill_level, scores):
    """Rows as the training agent saves them: two models per seed, one of them selected."""
    rows = []
    for split, values in scores.items():
        for seed, score in zip(SEEDS, values, strict=True):
            for model, selected, pr_auc in (("CatBoost", True, score), ("LightGBM", False, 0.99)):
                rows.append(
                    {
                        "dataset": "bank_marketing",
                        "arm": arm,
                        "fill_level": fill_level,
                        "seed": seed,
                        "model": model,
                        "split": split,
                        "status": "SUCCESS",
                        "selected": selected,
                        "pr_auc": pr_auc,
                        "f1": pr_auc / 2,
                        "precision": pr_auc / 4,
                        "recall": pr_auc * 2,
                        "roc_auc": 0.5 + pr_auc / 2,
                        "accuracy": 0.7 + pr_auc / 4,
                    }
                )
    return rows


def results():
    rows = arm_rows("original", 0.0, ORIGINAL)
    rows += arm_rows("cleaned", 0.0, ORIGINAL)
    for (arm, level), lift in LIFTS.items():
        scores = {
            split: [o + d for o, d in zip(ORIGINAL[split], lift[split], strict=True)]
            for split in lift
        }
        rows += arm_rows(arm, level, scores)
    # A failed fit that carries the selected flag must not count.
    rows.append({**rows[0], "model": "XGBoost", "status": "FAILED", "pr_auc": np.nan})
    return pd.DataFrame(rows)


def pick(table, arm, fill_level, split, metric="pr_auc"):
    row = table[
        table.arm.eq(arm)
        & np.isclose(table.fill_level, fill_level)
        & table.split.eq(split)
        & table.metric.eq(metric)
    ]
    assert len(row) == 1
    return row.iloc[0]


def test_compare_arms_gives_mean_spread_and_paired_difference():
    table = compare_arms(results())

    original = pick(table, "original", 0.0, "validation")
    assert original.n_seeds == 5
    assert original["mean"] == pytest.approx(0.50)
    assert original["sd"] == pytest.approx(np.std(ORIGINAL["validation"], ddof=1))
    assert math.isnan(original.diff_mean) and original.n_paired == 0

    gan = pick(table, "ctgan", 0.25, "validation")
    lift = LIFTS[("ctgan", 0.25)]["validation"]
    assert gan["mean"] == pytest.approx(0.50 + 0.03)
    assert gan.diff_mean == pytest.approx(np.mean(lift))
    assert gan.diff_sd == pytest.approx(np.std(lift, ddof=1))
    assert gan.n_paired == 5
    # The unselected LightGBM rows (0.99) and the failed fit are left out; other metrics follow.
    assert pick(table, "ctgan", 0.25, "test", "f1").diff_mean == pytest.approx(0.02 / 2)
    assert pick(table, "ctgan", 0.25, "test", "recall").diff_mean == pytest.approx(0.02 * 2)
    auroc = pick(table, "ctgan", 0.25, "validation", "roc_auc")
    accuracy = pick(table, "ctgan", 0.25, "test", "accuracy")
    assert auroc["mean"] == pytest.approx(0.765)
    assert auroc.diff_mean == pytest.approx(0.03 / 2)
    assert accuracy["mean"] == pytest.approx(0.825)
    assert accuracy.diff_mean == pytest.approx(0.02 / 4)


def test_compare_arms_orders_arms_then_fill_levels():
    arms = compare_arms(results())[["arm", "fill_level"]].drop_duplicates()
    assert arms.values.tolist() == [
        ["original", 0.0],
        ["cleaned", 0.0],
        ["smotenc", 0.25],
        ["smotenc", 0.5],
        ["ctgan", 0.25],
        ["ctgan", 0.5],
    ]


def test_compare_arms_pairs_only_the_seeds_both_arms_have():
    data = results()
    data = data[~(data.arm.eq("original") & data.seed.eq(44))]
    gan = pick(compare_arms(data), "ctgan", 0.25, "validation")
    assert (gan.n_seeds, gan.n_paired) == (5, 4)
    kept = [i for i, seed in enumerate(SEEDS) if seed != 44]
    assert gan.diff_mean == pytest.approx(
        np.mean([LIFTS[("ctgan", 0.25)]["validation"][i] for i in kept])
    )


def test_compare_arms_rejects_two_selected_models_for_one_seed():
    data = results()
    data.loc[data.model.eq("LightGBM"), "selected"] = True
    with pytest.raises(ValueError, match="more than one selected model"):
        compare_arms(data)
    with pytest.raises(ValueError, match="missing columns"):
        compare_arms(results().drop(columns="split"))


@pytest.mark.parametrize(
    ("arm", "level", "verdict"),
    [
        ("ctgan", 0.25, "reliable gain"),  # lift above the spread on both splits
        ("ctgan", 0.5, "inconclusive"),  # validation only
        ("smotenc", 0.25, "inconclusive"),  # lift inside the noise
        ("smotenc", 0.5, "reliable loss"),  # below minus the spread on both splits
        ("original", 0.0, "not enough paired seeds"),  # nothing to pair with
    ],
)
def test_reliable_gain_needs_a_lift_above_the_spread_on_validation_and_test(arm, level, verdict):
    table = compare_arms(results())
    result = reliable_gain(table, dataset="bank_marketing", arm=arm, fill_level=level)

    assert result.verdict == verdict
    assert result.reliable is (verdict == "reliable gain")
    if arm != "original":
        for split in ("validation", "test"):
            row = pick(table, arm, level, split)
            assert result.lift[split] == pytest.approx(row.diff_mean)
            assert result.spread[split] == pytest.approx(row.diff_sd)
            assert f"{split} lift {row.diff_mean:.4f} vs spread {row.diff_sd:.4f}" in result.reason


def test_reliable_gain_needs_at_least_three_paired_seeds():
    data = results()
    table = compare_arms(data[data.seed.isin(SEEDS[:2])])
    result = reliable_gain(table, dataset="bank_marketing", arm="ctgan", fill_level=0.25)
    assert result.verdict == "not enough paired seeds"


def test_comparison_reads_the_training_agents_own_metrics_file(tmp_path):
    def record(arm, level, seed, pr_auc, model="CatBoost", selected=True):
        scores = {
            "pr_auc": pr_auc,
            "f1": 0.4,
            "precision": 0.5,
            "recall": 0.4,
            "roc_auc": 0.8,
            "accuracy": 0.85,
        }
        return ModelRecord(
            status=Status.SUCCESS,
            arm=arm,
            fill_level=level,
            model=model,
            seed=seed,
            selected=selected,
            threshold=0.2,
            protocol_id="p",
            training_rows=100,
            model_path=Path("m.joblib"),
            val_predictions_path=Path("v.csv"),
            test_predictions_path=Path("t.csv"),
            val_metrics=scores,
            test_metrics=scores,
        )

    records = []
    for seed, base in zip(SEEDS[:3], [0.50, 0.52, 0.48], strict=True):
        records += [record(Arm.RAW, None, seed, base), record(Arm.GAN, 0.25, seed, base + 0.03)]
        records.append(record(Arm.RAW, None, seed, 0.9, model="LightGBM", selected=False))
    path, _ = _save_metrics(records, "bank_marketing", tmp_path)

    table = compare_arms(pd.read_csv(path))
    gan = pick(table, "ctgan", 0.25, "test")
    assert gan.diff_mean == pytest.approx(0.03) and gan.n_paired == 3
    assert pick(table, "ctgan", 0.25, "test", "roc_auc")["mean"] == pytest.approx(0.8)
    assert pick(table, "ctgan", 0.25, "test", "accuracy")["mean"] == pytest.approx(0.85)
    assert reliable_gain(table, dataset="bank_marketing", arm="ctgan", fill_level=0.25).reliable


# --- paired bootstrap (issue 138) --------------------------------------------------------------


def labelled(n=600, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < rate).astype(int), rng


def informative(y, rng, strength):
    """Scores in [0, 1] that put the positives higher the larger `strength` is."""
    return np.clip(rng.random(len(y)) * 0.6 + strength * y, 0, 1)


def test_identical_scores_have_no_difference_and_an_interval_around_zero():
    y, rng = labelled()
    scores = informative(y, rng, 0.3)
    for metric in ("pr_auc", "roc_auc"):
        result = paired_bootstrap(y, scores, scores, metric=metric, n_boot=100)
        assert result == BootstrapResult(diff=0.0, lo=0.0, hi=0.0, n_boot=100)


def test_scores_built_from_the_labels_beat_random_scores_with_a_positive_interval():
    y, rng = labelled()
    good, random = informative(y, rng, 0.4), rng.random(len(y))
    for metric in ("pr_auc", "roc_auc"):
        result = paired_bootstrap(y, random, good, metric=metric, n_boot=200)
        assert 0 < result.lo < result.diff < result.hi
        worse = paired_bootstrap(y, good, random, metric=metric, n_boot=200)
        assert worse.hi < 0 and worse.diff == pytest.approx(-result.diff)


def test_the_same_seed_reproduces_the_result_and_another_seed_moves_the_interval():
    y, rng = labelled()
    base, other = informative(y, rng, 0.2), informative(y, rng, 0.25)
    first = paired_bootstrap(y, base, other, n_boot=100, seed=7)
    assert first == paired_bootstrap(y, base, other, n_boot=100, seed=7)
    again = paired_bootstrap(y, base, other, n_boot=100, seed=8)
    assert again.diff == first.diff and (again.lo, again.hi) != (first.lo, first.hi)


def test_misaligned_or_unusable_inputs_raise():
    y, rng = labelled(n=100)
    a, b = rng.random(100), rng.random(100)
    for args in ((y[:-1], a, b), (y, a[:-1], b), (y, a, b[:-1])):
        with pytest.raises(ValueError, match="aligned"):
            paired_bootstrap(*args)
    with pytest.raises(ValueError, match="both classes"):
        paired_bootstrap(np.zeros(100, dtype=int), a, b)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap(y, np.where(np.arange(100) == 3, np.nan, a), b)
    with pytest.raises(ValueError, match="metric"):
        paired_bootstrap(y, a, b, metric="f1")
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap(y, a, b, alpha=1.5)


def test_class_counts_stay_fixed_in_every_resample(monkeypatch):
    y, rng = labelled(n=200, rate=0.1)
    seen = []
    area = evaluation_module._Ranked.area

    def spy(self, counts, metric):
        weights = counts[:, self.order]
        seen.append(
            ((weights * self.positive).sum(axis=1), (weights * ~self.positive).sum(axis=1), counts)
        )
        return area(self, counts, metric)

    monkeypatch.setattr(evaluation_module._Ranked, "area", spy)
    paired_bootstrap(y, rng.random(200), rng.random(200), n_boot=40)

    positives = np.concatenate([drawn for drawn, _, _ in seen])
    negatives = np.concatenate([drawn for _, drawn, _ in seen])
    assert (positives == y.sum()).all() and (negatives == (1 - y).sum()).all()
    assert len({tuple(row) for row in seen[0][2]}) > 1  # the resamples differ from each other


@pytest.mark.parametrize("metric", ["pr_auc", "roc_auc"])
def test_the_fast_resampling_matches_scoring_every_resample_with_sklearn(metric):
    y, rng = labelled(n=300, rate=0.15)
    base = np.round(informative(y, rng, 0.2), 2)  # rounded, so many scores are tied
    other = np.round(informative(y, rng, 0.35), 2)
    score = pr_auc_score if metric == "pr_auc" else roc_auc_score
    # The same draws as paired_bootstrap: the positives, then the negatives, one resample at a time.
    draws = np.random.default_rng(3)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(60):
        rows = np.concatenate(
            [
                positives[draws.integers(0, len(positives), len(positives))],
                negatives[draws.integers(0, len(negatives), len(negatives))],
            ]
        )
        diffs.append(score(y[rows], other[rows]) - score(y[rows], base[rows]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])

    result = paired_bootstrap(y, base, other, metric=metric, n_boot=60, seed=3)

    assert (result.lo, result.hi) == pytest.approx((lo, hi), abs=1e-12)
    assert result.diff == pytest.approx(score(y, other) - score(y, base))


ARMS = {"original": Arm.RAW, "cleaned": Arm.BASELINE, "smotenc": Arm.SMOTE, "ctgan": Arm.GAN}
EVAL_ROWS = {"validation": 700, "test": 800}
# How informative each arm's scores are on each split (larger means better).
STRENGTH = {
    ("original", 0.0): {"validation": 0.15, "test": 0.15},
    ("ctgan", 0.25): {"validation": 0.30, "test": 0.30},  # better on both splits
    ("smotenc", 0.25): {"validation": 0.02, "test": 0.02},  # worse on both splits
    ("smotenc", 0.5): {"validation": 0.15, "test": 0.15},  # the same quality as the original
    ("ctgan", 0.5): {"validation": 0.30, "test": 0.15},  # better on validation only
}


def fabricated_run(folder, seeds=(42,)):
    """A metrics table and the score files for it, as the training agent saves them."""
    rng = np.random.default_rng(5)
    labels = {split: (rng.random(n) < 0.2).astype(int) for split, n in EVAL_ROWS.items()}
    rows = []
    for seed in seeds:
        for (arm, level), strengths in STRENGTH.items():
            for split, strength in strengths.items():
                y = labels[split]
                probability = informative(y, rng, strength)
                stem = model_artifact_stem(ARMS[arm], level, seed, "CatBoost")
                scores = pd.DataFrame(
                    {"source_row": np.arange(len(y)), "y_true": y, "probability": probability}
                )
                scores.sample(frac=1, random_state=seed).to_csv(
                    folder / f"{stem}_{split}.csv", index=False
                )
                # A second model that lost on validation has no score files: it must not be read.
                for model, selected, pr_auc in (
                    ("CatBoost", True, pr_auc_score(y, probability)),
                    ("LightGBM", False, 0.99),
                ):
                    rows.append(
                        {
                            "dataset": "bank_marketing",
                            "arm": arm,
                            "fill_level": level,
                            "seed": seed,
                            "model": model,
                            "split": split,
                            "status": "SUCCESS",
                            "selected": selected,
                            "pr_auc": pr_auc,
                            "roc_auc": roc_auc_score(y, probability) if selected else 0.99,
                            "f1": 0.3,
                            "accuracy": 0.8,
                            "precision": 0.3,
                            "recall": 0.3,
                        }
                    )
    return pd.DataFrame(rows)


def test_selected_scores_come_from_the_selected_model_sorted_by_source_row(tmp_path):
    metrics = fabricated_run(tmp_path)

    scores = load_selected_scores(
        tmp_path, metrics, arm="ctgan", fill_level=0.25, seed=42, split="test"
    )

    assert list(scores.columns) == ["source_row", "y_true", "probability"]
    assert scores["source_row"].tolist() == list(range(EVAL_ROWS["test"]))
    with pytest.raises(ValueError, match="expected one selected model"):
        load_selected_scores(tmp_path, metrics, arm="ctgan", fill_level=0.75, seed=42, split="test")
    with pytest.raises(ValueError, match="unknown arm"):
        load_selected_scores(tmp_path, metrics, arm="raw", fill_level=0.0, seed=42, split="test")


@pytest.mark.parametrize(
    ("arm", "level", "verdict"),
    [
        ("ctgan", 0.25, "reliable gain"),  # the interval is above 0 on both splits
        ("smotenc", 0.25, "reliable loss"),  # and below 0 on both splits
        ("smotenc", 0.5, "inconclusive"),  # the same quality: the interval holds 0
        ("ctgan", 0.5, "inconclusive"),  # better on validation only
    ],
)
def test_one_seed_verdicts_come_from_the_bootstrap_on_both_splits(tmp_path, arm, level, verdict):
    metrics = fabricated_run(tmp_path)

    table = compare_arms(metrics, models_dir=tmp_path, n_boot=200)
    result = reliable_gain(table, dataset="bank_marketing", arm=arm, fill_level=level)

    assert result.verdict == verdict and result.basis == "paired bootstrap"
    for split in ("validation", "test"):
        row = pick(table, arm, level, split)
        assert row.n_paired == 1 and row.boot_seed == 42 and row.n_boot == 200
        assert row.boot_diff == pytest.approx(row.diff_mean)  # the scores reproduce the metrics
        assert result.lift[split] == pytest.approx(row.diff_mean)
        assert result.interval[split] == pytest.approx((row.boot_lo, row.boot_hi))
    reference = pick(table, "original", 0.0, "test")
    assert pd.isna(reference.boot_lo) and pd.isna(reference.boot_diff)


def test_without_scores_one_seed_still_has_no_verdict(tmp_path):
    table = compare_arms(fabricated_run(tmp_path))
    result = reliable_gain(table, dataset="bank_marketing", arm="ctgan", fill_level=0.25)
    assert result.verdict == "not enough paired seeds"
    assert "boot_lo" not in table


def test_three_or_more_seeds_keep_the_seed_rule_and_add_the_bootstrap_columns(tmp_path):
    metrics = fabricated_run(tmp_path, seeds=(42, 43, 44))

    table = compare_arms(metrics, models_dir=tmp_path, n_boot=100)
    result = reliable_gain(table, dataset="bank_marketing", arm="ctgan", fill_level=0.25)

    row = pick(table, "ctgan", 0.25, "test")
    assert row.n_paired == 3 and row.boot_seed == 42  # the bootstrap uses the lowest seed
    assert result.basis == "seed spread" and result.spread["test"] == pytest.approx(row.diff_sd)
    assert set(result.interval) == {"validation", "test"}


def test_arms_scored_on_different_rows_are_rejected(tmp_path):
    metrics = fabricated_run(tmp_path)
    path = tmp_path / f"{model_artifact_stem(Arm.GAN, 0.25, 42, 'CatBoost')}_test.csv"
    scores = pd.read_csv(path)
    scores["source_row"] += 1
    scores.to_csv(path, index=False)

    with pytest.raises(ValueError, match="different rows"):
        compare_arms(metrics, models_dir=tmp_path, n_boot=50)


# --- segments ----------------------------------------------------------------------------------


def predictions():
    return pd.DataFrame(
        {
            "source_row": range(8),
            "y_true": [1, 1, 0, 0, 1, 0, 0, 0],
            "probability": [0.9, 0.4, 0.2, 0.1, 0.8, 0.7, 0.3, 0.05],
            "prediction": [1, 0, 0, 0, 1, 1, 0, 0],
        }
    )


def test_segment_metrics_scores_each_segment_and_flags_small_ones():
    features = pd.DataFrame({"source_row": range(8), "job": ["admin"] * 4 + ["student"] * 4})
    table = segment_metrics(predictions(), features, "job", min_positives=2).set_index("segment")

    admin, student = table.loc["admin"], table.loc["student"]
    assert (admin.n, admin.positives, bool(admin.small)) == (4, 2, False)
    assert (admin.precision, admin.recall) == (1.0, 0.5)
    assert admin.f1 == pytest.approx(2 / 3)
    assert admin.pr_auc == pytest.approx(1.0)
    assert (student.n, student.positives, bool(student.small)) == (4, 1, True)
    assert (student.precision, student.recall) == (0.5, 1.0)


def test_segment_metrics_bands_numeric_columns_and_skips_missing_ones():
    n = 20
    data = pd.DataFrame(
        {
            "source_row": range(n),
            "y_true": [1, 0] * (n // 2),
            "probability": np.linspace(0.05, 0.95, n),
            "prediction": [0, 1] * (n // 2),
        }
    )
    features = pd.DataFrame({"source_row": range(n), "age": range(20, 20 + n)})
    table = segment_metrics(data, features, "age")
    assert table.n.tolist() == [5, 5, 5, 5]
    assert segment_metrics(data, features, "income").empty


# --- number check ------------------------------------------------------------------------------


@pytest.fixture
def result_files(tmp_path):
    table = pd.DataFrame(
        {
            "arm": ["ctgan", "ctgan"],
            "fill_level": [0.25, 0.5],
            "n_seeds": [5, 5],
            "mean": [0.4812, 0.4701],
            "diff_mean": [0.0524, -0.0515],
        }
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)
    (tmp_path / "counts.json").write_text(
        json.dumps({"test_rows": 8238, "ok": True, "x": [0.123456]})
    )
    return [tmp_path / "comparison.csv", tmp_path / "counts.json"]


def test_numbers_that_are_in_the_results_pass(result_files):
    report = (
        "1. Summary\n"
        "CTGAN v1 at 25% fill reached PR-AUC 0.48 (0.4812 exactly), a lift of 0.052 over the "
        "original arm across 5 seeds; at 0.5 fill it dropped by 0.05 (-0.0515). The test set "
        "has 8,238 rows and pr_auc2 is a column name. Another value is 0.123456."
    )
    check = verify_report_numbers(report, result_files)
    assert check.verified and check.unverified == ()
    assert check.checked == 10


@pytest.mark.parametrize(
    ("report", "unverified"),
    [
        ("PR-AUC reached 0.61 with the CTGAN arm.", ("0.61",)),
        ("It was run on 12 seeds.", ("12",)),  # a whole number must match exactly
        ("A 30% fill level was best.", ("30%",)),
        ("The lift was 0.052, then 0.07, then 0.07.", ("0.07",)),
    ],
)
def test_a_number_that_is_not_in_the_results_fails(result_files, report, unverified):
    check = verify_report_numbers(report, result_files)
    assert not check.verified
    assert check.unverified == unverified


def test_numbers_the_caller_vouches_for_are_allowed(result_files):
    report = "We trained 4 models."
    assert not verify_report_numbers(report, result_files).verified
    assert verify_report_numbers(report, result_files, allowed=[4]).verified


def test_numeric_segment_boundaries_are_grounded_for_report_verification(tmp_path):
    n = 24
    data = pd.DataFrame(
        {
            "source_row": range(n),
            "y_true": [0, 1] * (n // 2),
            "probability": np.linspace(0.05, 0.95, n),
            "prediction": [0, 1] * (n // 2),
        }
    )
    features = pd.DataFrame({"source_row": range(n), "age": range(21, 21 + n)})
    table = segment_metrics(data, features, "age")
    path = tmp_path / "segments.csv"
    table.to_csv(path, index=False)
    assert table.segment_lower_bound.min() == pytest.approx(20.999)
    assert verify_report_numbers("Age band (20.999, 26.75] is observed.", [path]).verified
    assert not verify_report_numbers("Age band (20.998, 26.75] is observed.", [path]).verified
    categorical = segment_metrics(data, features.assign(job="admin"), "job")
    assert categorical.segment_lower_bound.isna().all()


def test_missing_numeric_segment_values_are_not_a_string_nan_group():
    n = 24
    data = pd.DataFrame(
        {
            "source_row": range(n),
            "y_true": [0, 1] * (n // 2),
            "probability": np.linspace(0.05, 0.95, n),
            "prediction": [0, 1] * (n // 2),
        }
    )
    features = pd.DataFrame({"source_row": range(n), "age": [np.nan, *range(21, 21 + n - 1)]})
    table = segment_metrics(data, features, "age")
    assert table.n.sum() == n - 1
    assert "nan" not in table.segment.tolist()


def test_comparisons_require_a_single_synthetic_weight_per_dataset():
    mixed = pd.concat(
        [
            results().assign(synthetic_weight=1.0),
            results().assign(synthetic_weight=0.25),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="one synthetic_weight"):
        compare_arms(mixed, reference="cleaned")
