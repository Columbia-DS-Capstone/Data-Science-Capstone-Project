"""Turn fidelity, privacy, and utility checks into the pipeline quality verdict."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from agentic_gan.orchestrator.decisions import log_decision
from agentic_gan.orchestrator.manifest import RunDir
from agentic_gan.schemas import LevelReport, QualityInput, QualityReport, Stage, Status, Verdict
from agentic_gan.tools.quality_fidelity import RULES, fidelity_checks, schema_thresholds
from agentic_gan.tools.quality_privacy import quality_privacy_checks
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.files import write_text_atomic


def _positive_class(request: QualityInput, train: pd.DataFrame) -> int | str:
    positive = request.config.positive_class
    if train[request.config.target].eq(positive).any():
        return positive
    # Bank preprocessing maps the raw label "yes" to 1 in every cleaned split.
    if positive == "yes" and train[request.config.target].eq(1).any():
        return 1
    raise ValueError(f"positive class {positive!r} is absent from cleaned training data")


def _rules(request: QualityInput):
    available = {rule.name: rule for rule in RULES.get(request.config.dataset_name, ())}
    chosen = [name.removeprefix("rule:") for name in request.config.sanity_rules]
    unknown = set(chosen) - available.keys()
    if unknown:
        raise ValueError(f"Unknown sanity rules: {', '.join(sorted(unknown))}")
    return [available[name] for name in chosen]


def _plot(path, fidelity, diagnostics, level: float) -> None:
    distances = fidelity.columns["distance"].sort_values(ascending=False).head(10)
    figure = Figure(figsize=(10, 9))
    top, middle, bottom = figure.subplots(3, 1)
    top.barh(distances.index[::-1], distances.iloc[::-1])
    top.set_xlabel("Column distance from real train")
    top.set_title(f"Fidelity at {level:.0%} fill")
    middle.bar(
        ["Real validation", "Synthetic"],
        [diagnostics["validation_median"], diagnostics["synthetic_median"]],
    )
    middle.set_ylabel("Median distance to closest real train row")
    bottom.bar(
        ["PR-AUC real", "PR-AUC augmented", "F1 real", "F1 augmented"],
        [
            diagnostics["real_only_pr_auc"],
            diagnostics["augmented_pr_auc"],
            diagnostics["real_only_best_f1"],
            diagnostics["augmented_best_f1"],
        ],
    )
    bottom.set_ylim(0, 1)
    bottom.set_title(
        f"Real vs synthetic AUC: {diagnostics['real_vs_synthetic_auc']:.3f}; "
        f"revealing: {', '.join(diagnostics['revealing_columns']) or 'none'}"
    )
    figure.tight_layout()
    figure.savefig(path)


def run_quality(request: QualityInput, run: RunDir) -> QualityReport:
    """Score each fill level, persist the report and plots, and log the decisions."""
    config = request.config
    target = config.target
    train = pd.read_csv(request.preprocess.baseline.train_path)
    val = pd.read_csv(request.preprocess.baseline.val_path)
    positive = _positive_class(request, train)
    rules = _rules(request)
    discrete = DATASETS[config.dataset_name].discrete if config.dataset_name in DATASETS else ()
    thresholds = schema_thresholds(config.quality_gate, rules)
    reference = train.loc[train[target].eq(positive)]
    attempt = max(run.load_state().attempts(Stage.QUALITY_CHECK), 1)
    levels = []
    plots = []
    decisions = []

    for synthetic_set in request.generator.synthetic_sets:
        synthetic = pd.read_csv(synthetic_set.path)
        if len(synthetic) != synthetic_set.rows:
            raise ValueError(
                f"Synthetic row count differs from declared count at {synthetic_set.fill_level}"
            )
        fidelity = fidelity_checks(
            reference,
            synthetic,
            target=target,
            positive=positive,
            discrete=discrete,
            rules=rules,
            thresholds=thresholds,
        )
        privacy, details = quality_privacy_checks(
            train,
            val,
            synthetic,
            target,
            positive,
            thresholds=config.quality_gate,
            seed=config.split_seed,
        )
        checks = [*fidelity.checks, *privacy]
        level = LevelReport(
            fill_level=synthetic_set.fill_level,
            checks=checks,
            passed=all(check.passed for check in checks if check.hard),
        )
        levels.append(level)
        details["real_vs_synthetic_auc"] = next(
            check.value for check in privacy if check.name == "real_vs_synthetic_auc"
        )
        plot_path = run.subdir("quality") / (
            f"fill_{str(level.fill_level).replace('.', 'p')}_attempt_{attempt}.png"
        )
        _plot(plot_path, fidelity, details, level.fill_level)
        plots.append(plot_path.resolve())
        decisions.append((level, details))

    if not levels:
        raise ValueError("Generator supplied no synthetic fill levels")
    verdict = Verdict.PASS if any(level.passed for level in levels) else Verdict.FAIL
    report_path = run.subdir("quality") / f"quality_report_attempt_{attempt}.json"
    report = QualityReport(
        status=Status.SUCCESS,
        verdict=verdict,
        levels=levels,
        report_path=report_path.resolve(),
        plot_paths=plots,
    )
    write_text_atomic(report_path, report.model_dump_json(indent=2) + "\n")

    for level, details in decisions:
        for check in level.checks:
            log_decision(
                Stage.QUALITY_CHECK.value,
                check.name,
                {
                    "fill_level": level.fill_level,
                    "value": check.value,
                    "threshold": check.threshold,
                    "hard": check.hard,
                },
                "PASS" if check.passed else "FAIL",
                "Quality check result",
                run=run,
            )
        log_decision(
            Stage.QUALITY_CHECK.value,
            "fill_level",
            {
                "fill_level": level.fill_level,
                "revealing_columns": details["revealing_columns"],
                "real_vs_real_auc": details["real_vs_real_auc"],
            },
            "PASS" if level.passed else "FAIL",
            "Hard checks determine the level verdict",
            run=run,
        )
    log_decision(
        Stage.QUALITY_CHECK.value,
        "quality_verdict",
        {"attempt": attempt, "passing_levels": report.passing_levels},
        verdict.value,
        "At least one fill level must pass",
        run=run,
    )
    return report
