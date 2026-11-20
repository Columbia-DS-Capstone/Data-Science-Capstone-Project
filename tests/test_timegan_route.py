"""TimeGAN as the Default generator, with CTGAN when it is off, raises, or fails the gate.

The first CTGAN attempt after TimeGAN trains the Default recipe; the retry ladder follows it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agentic_gan.agents.model_selection import (
    TIMEGAN_DEFAULTS,
    attempt_history,
    run_model_selection,
    settings_key,
)
from agentic_gan.llm import RuleOverrideError
from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.manifest import create_run
from agentic_gan.schemas import (
    ArmFiles,
    CheckResult,
    LevelReport,
    ModelSelectionInput,
    PreprocessOutput,
    QualityReport,
    RunConfig,
    Status,
    Verdict,
    fill_rows,
)
from agentic_gan.tools.ctgan_sampling import recipe_for
from agentic_gan.tools.selection import DEFAULT_WIDE_TEMPORAL
from agentic_gan.utils.synthetic import IS_SYNTHETIC

ROWS = 2000
POSITIVES = 600
NEGATIVES = ROWS - POSITIVES


def _labels(rows: int, positives: int) -> list[int]:
    return [1] * positives + [0] * (rows - positives)


def _config(tmp_path: Path, **overrides) -> RunConfig:
    source = tmp_path / "source.csv"
    source.write_text("y\n1\n")
    fields = {
        "dataset_name": "default",
        "dataset_path": source,
        "target": "y",
        "positive_class": 1,
        "fill_levels": [0.25, 0.50],
    }
    fields.update(overrides)
    return RunConfig(**fields)


def _request(
    tmp_path: Path,
    config: RunConfig,
    frame: pd.DataFrame,
    *,
    attempt: int = 1,
    previous=None,
    run_id: str = "sel",
):
    run = create_run(config, run_id=run_id, runs_dir=tmp_path / "runs")
    train_path = tmp_path / f"{run_id}_train.csv"
    frame.to_csv(train_path, index=False)
    unread = tmp_path / f"{run_id}_unread.csv"
    unread.write_text("not-a-training-table\n")
    profile = tmp_path / f"{run_id}_profile.json"
    profile.write_text("{}\n")
    files = ArmFiles(
        train_path=train_path.resolve(),
        val_path=unread.resolve(),
        test_path=unread.resolve(),
    )
    preprocess = PreprocessOutput(
        status=Status.SUCCESS,
        raw=files,
        baseline=files,
        profile_path=profile.resolve(),
        treatment_log_path=profile.resolve(),
        transformer_path=profile.resolve(),
        transformer_fit_rows=max(len(frame), 1),
        baseline_train_positives=POSITIVES,
        baseline_train_negatives=NEGATIVES,
    )
    request = ModelSelectionInput(
        config=config,
        preprocess=preprocess,
        attempt=attempt,
        previous_report=previous,
    )
    return request, run


def _default_frame() -> pd.DataFrame:
    frame = pd.DataFrame({name: range(ROWS) for name in DEFAULT_WIDE_TEMPORAL})
    frame["SEX"] = 1
    frame["y"] = _labels(ROWS, POSITIVES)
    return frame


def _fail_report(*names):
    checks = [CheckResult(name=name, value=1.0, threshold=0.1, passed=False) for name in names]
    return QualityReport(
        status=Status.SUCCESS,
        verdict=Verdict.FAIL,
        levels=[LevelReport(fill_level=0.25, passed=False, checks=checks)],
        report_path=Path("quality.json"),
    )


def _logged(run, rule: str):
    matched = read_decisions(run).query("rule == @rule")
    assert len(matched) >= 1, f"rule {rule!r} not found in decisions"
    return matched.iloc[-1]


def _ctgan_hyperparameters(run) -> dict:
    decisions = read_decisions(run)
    rows = decisions.query("rule == 'hyperparameters'")
    ctgan_rows = [inputs for inputs in rows.inputs if inputs.get("generator") == "ctgan"]
    assert ctgan_rows, "CTGAN hyperparameters were not logged"
    return ctgan_rows[-1]


class _TimeganStub:
    def __init__(self):
        self.trains: list = []
        self.samples: list = []
        self.error: BaseException | None = None
        self.short_rows: int | None = None
        self.write_partial_fill = False

    def train(self, train_df, target_col, static_columns, step_columns, **kwargs):
        self.trains.append({
            "frame": train_df.copy(),
            "static": list(static_columns),
            "steps": [list(step) for step in step_columns],
        })
        if isinstance(self.error, RuleOverrideError):
            raise self.error
        log_path = kwargs.get("log_path")
        if log_path is not None and self.error is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Phase,Epoch\njoint,1\n")
            pkl_name = path.name.replace("timegan_loss_attempt_", "timegan_attempt_")
            path.with_name(pkl_name.replace(".csv", ".pkl")).write_bytes(b"partial-timegan")
            if self.write_partial_fill:
                attempt = path.stem.replace("timegan_loss_attempt_", "")
                sentinel = path.parent / f"fill_9p99_attempt_{attempt}.csv"
                sentinel.write_text("TIMEGAN_PARTIAL\n")
        if self.error is not None:
            raise self.error
        if log_path is not None:
            Path(log_path).write_text("Phase,Epoch\njoint,1\n")
        return SimpleNamespace(target_col=target_col, columns=list(train_df.columns))

    def save(self, model, path):
        Path(path).write_bytes(b"timegan")

    def sample(self, model, n, *, seed, row_filter=None, drop_duplicates_of=None, **kwargs):
        self.samples.append({
            "n": n,
            "seed": seed,
            "row_filter": row_filter,
            "reference": None if drop_duplicates_of is None else drop_duplicates_of.copy(),
        })
        rows = n if self.short_rows is None else self.short_rows
        columns = list(drop_duplicates_of.columns)
        frame = pd.DataFrame({name: list(range(rows)) for name in columns})
        frame[model.target_col] = 1
        frame[IS_SYNTHETIC] = 1
        return frame


class _CtganStub:
    def __init__(self):
        self.trains: list = []
        self.samples: list = []

    def train(self, train_df, target_col, **kwargs):
        self.trains.append((train_df.copy(), kwargs))
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).write_text("Generator Loss,Discriminator Loss\n0.1,0.2\n")
        return SimpleNamespace(target_col=target_col)

    def save(self, model, path):
        Path(path).write_bytes(b"ctgan")

    def sample(self, model, n, target_value=1, seed=None, drop_duplicates_of=None, **kwargs):
        self.samples.append(n)
        frame = pd.DataFrame({name: [0] * n for name in drop_duplicates_of.columns})
        frame[drop_duplicates_of.columns[-1]] = target_value
        return frame


def _sample_for_fill(ctgan: _CtganStub):
    """Stand in for sample_for_fill.

    The Default recipe rakes EDUCATION, MARRIAGE and PAY_1. This frame has none of those,
    and the agent calls sample_for_fill rather than sample_synthetic.
    """

    def fill(model, train, positive_rows, counts, **kwargs):
        n = max(counts)
        ctgan.samples.append(n)
        frame = pd.DataFrame({name: list(range(n)) for name in train.columns})
        frame[model.target_col] = 1
        return frame

    return fill


@pytest.fixture
def stubs(monkeypatch):
    timegan = _TimeganStub()
    ctgan = _CtganStub()
    monkeypatch.setattr("agentic_gan.tools.timegan.train_timegan", timegan.train)
    monkeypatch.setattr("agentic_gan.tools.timegan.save_model", timegan.save)
    monkeypatch.setattr("agentic_gan.tools.timegan.sample_timegan", timegan.sample)
    monkeypatch.setattr("agentic_gan.tools.ctgan.train_ctgan", ctgan.train)
    monkeypatch.setattr("agentic_gan.tools.ctgan.save_model", ctgan.save)
    monkeypatch.setattr("agentic_gan.tools.ctgan.sample_synthetic", ctgan.sample)
    monkeypatch.setattr(
        "agentic_gan.agents.model_selection.sample_for_fill",
        _sample_for_fill(ctgan),
    )
    return timegan, ctgan


def _assert_default_recipe(inputs: dict) -> None:
    recipe = recipe_for("default")
    assert inputs["epochs"] == recipe.epochs == 300
    assert inputs["batch_size"] == recipe.batch_size == 500
    assert inputs["pac"] == recipe.pac == 5
    assert inputs["discriminator_steps"] == recipe.discriminator_steps == 3
    assert inputs["sampling_mode"] == recipe.sampling_mode == "unconditional"
    assert inputs["rule_filter"] is True
    assert recipe.rule_filter is True
    assert inputs["rake_columns"] == ["SEX", "EDUCATION", "MARRIAGE", "PAY_1"]
    assert inputs["rake_columns"] == list(recipe.rake_columns)


def _rule_probe() -> pd.DataFrame:
    probe = pd.DataFrame({name: [0, 0, 0] for name in DEFAULT_WIDE_TEMPORAL})
    probe["SEX"] = 1
    probe["EDUCATION"] = 1
    probe["MARRIAGE"] = 1
    probe["y"] = [0, 1, 1]
    probe.loc[2, "PAY_1"] = 99
    return probe


def test_default_attempt_1_trains_timegan(tmp_path, stubs):
    timegan, ctgan = stubs
    frame = _default_frame()
    request, run = _request(tmp_path, _config(tmp_path), frame)
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "timegan"
    assert ctgan.trains == []
    assert len(timegan.trains) == 1
    assert len(timegan.samples) == 1

    temporal = _logged(run, "temporal_test")
    assert temporal.outcome == "PASS"
    assert temporal.inputs["generator"] == "timegan"
    fallback = _logged(run, "fallback")
    assert fallback.outcome == "DOCUMENTED"
    rationale = fallback.rationale.lower()
    assert "implemented" in rationale
    assert "not implemented" not in rationale

    settings = _logged(run, "hyperparameters")
    assert settings.outcome == "SET"
    assert settings.inputs["generator"] == "timegan"
    assert settings.inputs["epochs"] == TIMEGAN_DEFAULTS["epochs"] == 100
    assert settings.inputs["batch_size"] == TIMEGAN_DEFAULTS["batch_size"] == 128
    assert settings.inputs["hidden_dim"] == TIMEGAN_DEFAULTS["hidden_dim"] == 48
    assert settings.inputs["device"] == TIMEGAN_DEFAULTS["device"] == "cpu"
    assert settings.inputs["seed"] == 42

    counts = [fill_rows(level, POSITIVES, NEGATIVES) for level in request.config.fill_levels]
    assert timegan.samples[0]["n"] == max(counts)
    assert [item.rows for item in output.synthetic_sets] == counts
    small = pd.read_csv(output.synthetic_sets[0].path)
    large = pd.read_csv(output.synthetic_sets[1].path)
    assert small.reset_index(drop=True).equals(large.head(len(small)).reset_index(drop=True))

    assert timegan.trains[0]["static"] == ["SEX"]
    assert timegan.trains[0]["steps"][0] == ["PAY_6", "BILL_AMT6", "PAY_AMT6"]
    assert timegan.samples[0]["reference"].equals(timegan.trains[0]["frame"])
    kept = timegan.samples[0]["row_filter"](_rule_probe())
    assert kept.tolist() == [False, True, False]


def test_timegan_runtime_error_falls_back_to_ctgan(tmp_path, stubs):
    timegan, ctgan = stubs
    timegan.error = RuntimeError("simulated training failure")
    timegan.write_partial_fill = True
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert len(timegan.trains) == 1
    assert len(ctgan.trains) == 1
    failed = _logged(run, "backup_failed")
    assert failed.outcome == "FALLBACK_CTGAN"
    assert "RuntimeError" in failed.inputs["error"]
    assert "simulated training failure" in failed.inputs["error"]
    folder = run.subdir("synthetic")
    assert not (folder / "timegan_attempt_1.pkl").exists()
    assert not (folder / "timegan_loss_attempt_1.csv").exists()
    assert not (folder / "fill_9p99_attempt_1.csv").exists()
    assert "TIMEGAN_PARTIAL" not in (folder / "fill_0p25_attempt_1.csv").read_text()
    _assert_default_recipe(_ctgan_hyperparameters(run))


def test_failed_gate_trains_ctgan_on_attempt_2(tmp_path, stubs):
    timegan, ctgan = stubs
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    first = run_model_selection(request, run)
    assert first.status is Status.SUCCESS
    assert first.generator == "timegan"
    assert len(timegan.trains) == 1

    # These are TimeGAN's failures. The ladder would widen rake_columns and raise epochs on them.
    timegan_report = _fail_report("worst_column_distance", "mean_column_distance")
    retry = request.updated(attempt=2, previous_report=timegan_report)
    second = run_model_selection(retry, run)

    assert second.status is Status.SUCCESS
    assert second.generator == "ctgan"
    assert len(timegan.trains) == 1
    assert len(ctgan.trains) == 1
    assert _logged(run, "retry_not_applicable").outcome == "NOT_APPLIED"
    decisions = read_decisions(run)
    assert decisions.query("rule == 'timegan_skipped'").empty
    assert decisions.query("rule == 'hyperparameter_retry'").empty
    assert decisions.query("rule == 'retry_unchanged'").empty
    recipe_row = _logged(run, "fallback_recipe")
    assert recipe_row.outcome == "RECIPE"
    assert recipe_row.inputs["attempt"] == 2
    settings = _ctgan_hyperparameters(run)
    assert settings["attempt"] == 2
    _assert_default_recipe(settings)


def test_attempt_3_adjusts_the_attempt_2_recipe(tmp_path, stubs):
    timegan, ctgan = stubs
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    run_model_selection(request, run)
    second = request.updated(attempt=2, previous_report=_fail_report("spearman_diff"))
    run_model_selection(second, run)

    third = request.updated(attempt=3, previous_report=_fail_report("worst_column_distance"))
    # TimeGAN's attempt 1 is not a CTGAN attempt, so the history starts at attempt 2.
    history = attempt_history(third, run)
    assert [record.attempt for record in history] == [2]
    assert history[0].failed_checks == ["worst_column_distance"]
    output = run_model_selection(third, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert len(timegan.trains) == 1
    assert len(ctgan.trains) == 2
    decisions = read_decisions(run)
    assert len(decisions.query("rule == 'fallback_recipe'")) == 1
    assert decisions.query("rule == 'backup_skipped'").empty
    retry_row = _logged(run, "hyperparameter_retry")
    assert retry_row.outcome == "ADJUSTED"
    assert retry_row.inputs["attempt"] == 3
    _assert_default_recipe(retry_row.inputs["previous_settings"])
    settings = _ctgan_hyperparameters(run)
    assert settings["attempt"] == 3
    assert settings_key(settings) != settings_key(retry_row.inputs["previous_settings"])


def test_timegan_disabled_trains_ctgan_recipe(tmp_path, stubs, monkeypatch):
    timegan, ctgan = stubs
    monkeypatch.setattr("agentic_gan.agents.model_selection.TIMEGAN_ENABLED", False)
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert timegan.trains == []
    assert len(ctgan.trains) == 1
    assert _logged(run, "timegan_skipped").outcome == "DISABLED"
    _assert_default_recipe(_ctgan_hyperparameters(run))


def test_rule_override_from_timegan_propagates(tmp_path, stubs):
    timegan, ctgan = stubs
    timegan.error = RuleOverrideError("the LLM changed the generator")
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    with pytest.raises(RuleOverrideError):
        run_model_selection(request, run)
    assert len(timegan.trains) == 1
    assert ctgan.trains == []


def test_timegan_import_error_falls_back_to_ctgan(tmp_path, stubs):
    timegan, ctgan = stubs
    timegan.error = ImportError("torch not available")
    timegan.write_partial_fill = True
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert len(ctgan.trains) == 1
    assert _logged(run, "backup_failed").outcome == "FALLBACK_CTGAN"
    folder = run.subdir("synthetic")
    assert not (folder / "timegan_attempt_1.pkl").exists()
    assert not (folder / "timegan_loss_attempt_1.csv").exists()


def test_short_timegan_sample_falls_back_to_ctgan(tmp_path, stubs):
    timegan, ctgan = stubs
    timegan.short_rows = 1
    request, run = _request(tmp_path, _config(tmp_path), _default_frame())
    output = run_model_selection(request, run)

    assert output.status is Status.SUCCESS
    assert output.generator == "ctgan"
    assert len(timegan.samples) == 1
    assert len(ctgan.trains) == 1
    assert _logged(run, "backup_failed").outcome == "FALLBACK_CTGAN"
