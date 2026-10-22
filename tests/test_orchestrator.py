"""End-to-end control flow with typed placeholder agents."""

from pathlib import Path

from agentic_gan.orchestrator.decisions import read_decisions
from agentic_gan.orchestrator.graph import build_graph, run_pipeline
from agentic_gan.orchestrator.stubs import StubAgent, default_agents
from agentic_gan.schemas import Arm, RunConfig, Stage, Status


def config(tmp_path):
    dataset = tmp_path / "bank.csv"
    dataset.write_text("feature,y\n1,yes\n0,no\n")
    return RunConfig(dataset_name="bank_marketing", dataset_path=dataset,
                     target="y", positive_class="yes")


def test_graph_is_visualizable():
    diagram = build_graph().get_graph().draw_mermaid()
    assert "model_selection" in diagram
    assert "quality_check" in diagram
    assert "partial_report" in diagram


def test_pass_first_time(tmp_path, monkeypatch):
    run_config = config(tmp_path)
    monkeypatch.chdir(tmp_path)
    state, run = run_pipeline(run_config, runs_dir=Path("runs"), agents=default_agents())
    assert state.is_complete
    assert state.gan_available is True
    assert [state.attempts(stage) for stage in Stage] == [1] * 6
    assert run.load_manifest().verify(run) == []
    assert run.decisions_path.exists()
    assert (run.subdir("report") / "final_report.md").exists()


def test_quality_fail_then_pass(tmp_path):
    agents = default_agents()
    agents[Stage.QUALITY_CHECK] = StubAgent(
        Stage.QUALITY_CHECK, failures=1, failure_mode="quality"
    )
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.is_complete
    assert state.gan_available is True
    assert state.attempts(Stage.MODEL_SELECTION) == 2
    assert state.attempts(Stage.QUALITY_CHECK) == 2
    assert read_decisions(run).query("rule == 'quality_gate'").outcome.tolist() == [
        "RETRY_MODEL_SELECTION", "ACCEPT"
    ]
    assert run.load_manifest().verify(run) == []


def test_quality_exhaustion_continues_without_gan(tmp_path):
    agents = default_agents()
    agents[Stage.QUALITY_CHECK] = StubAgent(
        Stage.QUALITY_CHECK, failures=-1, failure_mode="quality"
    )
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.is_complete
    assert state.gan_available is False
    assert state.gan_unavailable_reason
    assert state.attempts(Stage.MODEL_SELECTION) == 3
    assert state.attempts(Stage.QUALITY_CHECK) == 3
    assert state.status(Stage.TRAINING) is Status.SUCCESS
    assert read_decisions(run).query("rule == 'quality_gate'").outcome.iloc[-1] == "NOT_AVAILABLE"
    assert "NOT_AVAILABLE" in (run.subdir("report") / "final_report.md").read_text()
    assert all(entry.agent is not Stage.TRAINING or entry.path.name != "gan.txt"
               for entry in run.load_manifest().entries)
    assert run.load_manifest().verify(run) == []


def test_stage_error_retries_once_then_stops(tmp_path):
    agents = default_agents()
    agents[Stage.PREPROCESSING] = StubAgent(Stage.PREPROCESSING, failures=-1)
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.failed_stage is Stage.PREPROCESSING
    assert state.attempts(Stage.PREPROCESSING) == 2
    assert state.attempts(Stage.MODEL_SELECTION) == 0
    assert (run.subdir("report") / "partial_report.md").exists()
    assert read_decisions(run).query("rule == 'stage_error'").outcome.tolist() == [
        "RETRY", "FAILED"
    ]
    assert run.load_manifest().verify(run) == []


def test_stage_error_then_pass(tmp_path):
    agents = default_agents()
    agents[Stage.TRAINING] = StubAgent(Stage.TRAINING, failures=1)
    state, run = run_pipeline(config(tmp_path), runs_dir=tmp_path / "runs", agents=agents)
    assert state.is_complete
    assert state.attempts(Stage.TRAINING) == 2
    assert any(entry.path.name.startswith(Arm.GAN.value)
               for entry in run.load_manifest().produced_by(Stage.TRAINING))
