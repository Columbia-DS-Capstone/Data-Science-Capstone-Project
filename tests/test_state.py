"""Tests for the run state: transitions, invariants and persistence."""

import json

import pytest
from pydantic import ValidationError

from agentic_gan.orchestrator.state import STATE_FILE, RunState, StageState, StateError
from agentic_gan.schemas import STAGE_ORDER, Stage, Status, VolumeDecision

RUN_ID = "bank_marketing_20261014-101500"


def run_stages(state: RunState, *stages: Stage) -> RunState:
    """Run each stage to SUCCESS, in the order given."""
    for stage in stages:
        state = state.start(stage).finish(stage, Status.SUCCESS)
    return state


def run_all(state: RunState | None = None) -> RunState:
    return run_stages(state or RunState.new(RUN_ID), *STAGE_ORDER)


# --- a fresh state ----------------------------------------------------------------------------


def test_new_state_has_every_stage_not_started():
    state = RunState.new(RUN_ID)
    assert list(state.stages) == list(STAGE_ORDER)
    for stage in STAGE_ORDER:
        assert state.status(stage) is None
        assert state.attempts(stage) == 0
        assert not state.is_done(stage)
    assert not state.is_complete
    assert state.failed_stage is None
    assert state.gan_available is None
    assert state.volume_decision is None


# --- stage transitions ------------------------------------------------------------------------


def test_a_full_run_finishes_with_one_attempt_per_stage():
    state = run_all()
    assert state.is_complete
    assert [state.attempts(s) for s in STAGE_ORDER] == [1] * 6


def test_changes_return_a_new_state_and_leave_the_old_one_alone():
    before = RunState.new(RUN_ID)
    after = before.start(Stage.SPLIT)
    assert before.attempts(Stage.SPLIT) == 0
    assert after.attempts(Stage.SPLIT) == 1
    assert after.stages[Stage.SPLIT].running


def test_a_stage_cannot_start_before_the_earlier_ones_are_done():
    with pytest.raises(StateError, match="earlier stages are not done: split"):
        RunState.new(RUN_ID).start(Stage.PREPROCESSING)
    running = RunState.new(RUN_ID).start(Stage.SPLIT)
    with pytest.raises(StateError, match="split"):
        running.start(Stage.PREPROCESSING)


def test_a_running_stage_cannot_start_again():
    state = RunState.new(RUN_ID).start(Stage.SPLIT)
    with pytest.raises(StateError, match="split: already running"):
        state.start(Stage.SPLIT)


def test_a_stage_that_is_not_running_cannot_finish():
    with pytest.raises(StateError, match="not running"):
        RunState.new(RUN_ID).finish(Stage.SPLIT, Status.SUCCESS)


def test_failed_needs_a_reason_and_is_final():
    state = run_stages(RunState.new(RUN_ID), Stage.SPLIT).start(Stage.PREPROCESSING)
    with pytest.raises(ValidationError, match="needs a reason"):
        state.finish(Stage.PREPROCESSING, Status.FAILED)
    failed = state.finish(Stage.PREPROCESSING, Status.FAILED, "schema check failed twice")
    assert failed.failed_stage is Stage.PREPROCESSING
    assert not failed.is_complete
    with pytest.raises(StateError, match="cannot start a stage that is FAILED"):
        failed.start(Stage.PREPROCESSING)


def test_an_error_retry_counts_attempts():
    state = run_stages(RunState.new(RUN_ID), Stage.SPLIT).start(Stage.PREPROCESSING)
    state = state.finish(Stage.PREPROCESSING, Status.RETRY, "timeout")
    assert state.status(Stage.PREPROCESSING) is Status.RETRY
    assert state.attempts(Stage.PREPROCESSING) == 1
    assert not state.is_done(Stage.PREPROCESSING)
    state = state.start(Stage.PREPROCESSING).finish(Stage.PREPROCESSING, Status.SUCCESS)
    assert state.attempts(Stage.PREPROCESSING) == 2
    assert state.is_done(Stage.PREPROCESSING)


def test_retry_needs_a_reason():
    state = RunState.new(RUN_ID).start(Stage.SPLIT)
    with pytest.raises(ValidationError, match="needs a reason"):
        state.finish(Stage.SPLIT, Status.RETRY)


def test_a_finished_stage_can_run_again_for_the_quality_gate_loop():
    state = run_stages(
        RunState.new(RUN_ID),
        Stage.SPLIT,
        Stage.PREPROCESSING,
        Stage.MODEL_SELECTION,
        Stage.QUALITY_CHECK,
    )
    state = run_stages(state, Stage.MODEL_SELECTION, Stage.QUALITY_CHECK)
    assert state.attempts(Stage.MODEL_SELECTION) == 2
    assert state.attempts(Stage.QUALITY_CHECK) == 2
    assert state.is_done(Stage.QUALITY_CHECK)


def test_a_stage_that_never_ran_can_be_skipped_and_unblocks_the_next():
    state = run_stages(RunState.new(RUN_ID), Stage.SPLIT, Stage.PREPROCESSING)
    state = state.skip(Stage.MODEL_SELECTION, "dataset has fewer than min_rows rows")
    state = state.skip(Stage.QUALITY_CHECK, "no synthetic data to check")
    state = run_stages(state, Stage.TRAINING, Stage.EVALUATION)
    assert state.is_complete
    assert state.attempts(Stage.MODEL_SELECTION) == 0


def test_skip_needs_a_reason():
    state = run_stages(RunState.new(RUN_ID), Stage.SPLIT, Stage.PREPROCESSING)
    with pytest.raises(ValidationError, match="needs a reason"):
        state.skip(Stage.MODEL_SELECTION, "")


def test_a_started_stage_cannot_be_skipped_and_a_skipped_one_cannot_start():
    state = run_stages(RunState.new(RUN_ID), Stage.SPLIT, Stage.PREPROCESSING)
    with pytest.raises(StateError, match="has not started"):
        state.start(Stage.MODEL_SELECTION).skip(Stage.MODEL_SELECTION, "late")
    skipped = state.skip(Stage.MODEL_SELECTION, "too small")
    with pytest.raises(StateError, match="cannot start a stage that is SKIPPED"):
        skipped.start(Stage.MODEL_SELECTION)


def test_is_done_is_true_only_for_success_and_skipped():
    state = RunState.new(RUN_ID)
    assert not state.is_done(Stage.SPLIT)
    assert not state.start(Stage.SPLIT).is_done(Stage.SPLIT)
    assert state.start(Stage.SPLIT).finish(Stage.SPLIT, Status.SUCCESS).is_done(Stage.SPLIT)


# --- stage invariants -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"running": True, "attempts": 1, "status": Status.SUCCESS},
        {"running": True, "attempts": 0},
        {"attempts": 2},
        {"status": Status.SUCCESS, "attempts": 0},
        {"status": Status.FAILED, "attempts": 1},
        {"attempts": -1},
    ],
    ids=[
        "running with a status",
        "running with no attempt",
        "attempts but no status",
        "success with no attempt",
        "failed with no reason",
        "negative attempts",
    ],
)
def test_stage_state_rejects_impossible_combinations(fields):
    with pytest.raises(ValidationError):
        StageState(**fields)


def test_stage_state_allows_the_normal_combinations():
    StageState()
    StageState(running=True, attempts=1)
    StageState(status=Status.SUCCESS, attempts=1)
    StageState(status=Status.SKIPPED, reason="too small")


# --- run-level choices ------------------------------------------------------------------------


def test_gan_can_be_marked_unavailable_with_a_reason():
    state = RunState.new(RUN_ID).with_gan(False, "quality gate failed after 3 attempts")
    assert state.gan_available is False
    assert state.gan_unavailable_reason == "quality gate failed after 3 attempts"


def test_gan_available_has_no_reason():
    assert RunState.new(RUN_ID).with_gan(True).gan_unavailable_reason is None


def test_gan_off_needs_a_reason_and_a_reason_needs_gan_off():
    with pytest.raises(ValidationError, match="gan_unavailable_reason"):
        RunState.new(RUN_ID).with_gan(False)
    with pytest.raises(ValidationError, match="gan_unavailable_reason"):
        RunState.new(RUN_ID).with_gan(True, "not needed")


def test_volume_decision_is_stored():
    decision = VolumeDecision(level=0.25, reason="smallest level with a reliable lift")
    assert RunState.new(RUN_ID).with_volume_decision(decision).volume_decision == decision


def test_state_must_hold_every_stage():
    state = RunState.new(RUN_ID)
    partial = {s: v for s, v in state.stages.items() if s is not Stage.EVALUATION}
    with pytest.raises(ValidationError, match="every stage"):
        RunState(run_id=RUN_ID, stages=partial)


@pytest.mark.parametrize("run_id", ["", "../escape", "a/b", "with space", ".hidden", "x" * 101])
def test_run_id_must_be_a_safe_folder_name(run_id):
    with pytest.raises(ValidationError):
        RunState.new(run_id)


def test_run_state_is_immutable():
    with pytest.raises(ValidationError):
        RunState.new(RUN_ID).gan_available = True


# --- saving -----------------------------------------------------------------------------------


def test_state_round_trips_through_json():
    state = run_all().with_gan(True).with_volume_decision(VolumeDecision(level=0.0, reason="x"))
    assert RunState.model_validate_json(state.model_dump_json()) == state


def test_save_and_load(tmp_path):
    state = run_stages(RunState.new(RUN_ID), Stage.SPLIT).start(Stage.PREPROCESSING)
    path = tmp_path / "runs" / RUN_ID / STATE_FILE
    state.save(path)
    assert RunState.load(path) == state


def test_saved_file_names_stages_by_value(tmp_path):
    path = tmp_path / STATE_FILE
    RunState.new(RUN_ID).save(path)
    assert list(json.loads(path.read_text())["stages"]) == [s.value for s in STAGE_ORDER]


def test_saving_again_replaces_the_file(tmp_path):
    path = tmp_path / STATE_FILE
    RunState.new(RUN_ID).save(path)
    later = RunState.new(RUN_ID).start(Stage.SPLIT)
    later.save(path)
    assert RunState.load(path) == later


def test_loading_a_corrupt_file_is_an_error(tmp_path):
    path = tmp_path / STATE_FILE
    path.write_text('{"run_id": "x", "stages": {}')
    with pytest.raises(ValidationError):
        RunState.load(path)
