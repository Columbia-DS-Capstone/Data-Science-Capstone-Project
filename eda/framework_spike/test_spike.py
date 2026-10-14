import os

import pytest
from benchmark import CASES, aggregate, run_case, usage_counts
from spike import (
    Gate,
    Proposal,
    fake_sdk_model,
    load_local_env,
    run_agents_sdk,
    run_langgraph,
    summary,
)


def test_same_two_step_retry_and_cap():
    for cap, expected_pass in ((1, False), (2, True)):
        graph_state = run_langgraph(max_attempts=cap)
        sdk_state = run_agents_sdk(fake_sdk_model(), max_attempts=cap)
        assert isinstance(graph_state["proposal"], Proposal)
        assert isinstance(sdk_state["proposal"], Proposal)
        assert isinstance(graph_state["gate"], Gate)
        assert isinstance(sdk_state["gate"], Gate)
        graph = summary(graph_state)
        sdk = summary(sdk_state)
        assert graph == sdk
        assert graph["attempts"] == cap
        assert graph["passed"] is expected_pass
        assert len(graph["decisions"]) == 2 * cap


def test_model_cannot_override_gate_rule():
    with pytest.raises(ValueError, match="deterministic quality rule"):
        run_langgraph(judge=lambda _proposal: Gate(passed=True, reason="wrong"), max_attempts=1)


def test_project_cases_and_metric_aggregation():
    rows = [run_case(case, framework) for case in CASES
            for framework in ("langgraph", "agents_sdk")]
    measured = aggregate(rows)
    assert all(row["valid"] and row["error"] is None for row in rows)
    assert all(measured[name]["runs"] == len(CASES) for name in measured)
    assert all(measured[name]["estimated_usd"] is None for name in measured)


def test_usage_requires_provider_counts():
    with pytest.raises(ValueError, match="token usage"):
        usage_counts(None)
    usages = []
    run_agents_sdk(fake_sdk_model(), on_usage=usages.append)
    assert len(usages) == 4
    assert [usage_counts(usage) for usage in usages] == [(1, 1, 0)] * 4


def test_local_env_loads_exports_without_overwriting_shell(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("export GOOGLE_CLOUD_PROJECT='team-project'\n"
                   "GOOGLE_CLOUD_LOCATION=global # comment\n"
                   "VERTEX_MODEL='google/gemini-3.5-flash'\n")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "shell-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_MODEL", raising=False)
    load_local_env(env)
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "shell-project"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "global"
    assert os.environ["VERTEX_MODEL"] == "google/gemini-3.5-flash"


def test_authentication_failure_excludes_both_sides_of_pair():
    good = [dict(run_case(CASES[0], framework), repeat=1)
            for framework in ("langgraph", "agents_sdk")]
    failed_pair = [dict(run_case(CASES[1], framework), repeat=2)
                   for framework in ("langgraph", "agents_sdk")]
    failed_pair[0].update(error="AuthenticationError: expired", valid=False)
    measured = aggregate(good + failed_pair)
    assert measured["langgraph"]["runs"] == 1
    assert measured["agents_sdk"]["runs"] == 1
    assert aggregate(failed_pair)["langgraph"]["runs"] == 0
