"""Run state: where each stage stands, and what the run has decided so far.

`RunState` is immutable. Every change returns a new, validated copy, so it works as a LangGraph
state value or in plain Python, and it is saved as runs/<run_id>/state.json.

`StageState.attempts` counts every invocation of a stage, including retries after an error. It is
not the quality-gate attempt number that `ModelSelectionInput.attempt` carries.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from agentic_gan.schemas import STAGE_ORDER, RunId, Schema, Stage, Status, VolumeDecision
from agentic_gan.utils.files import write_text_atomic

STATE_FILE = "state.json"
_NEEDS_REASON = (Status.FAILED, Status.SKIPPED, Status.RETRY)


class StateError(ValueError):
    """A change the run state does not allow, such as starting a stage that is already running."""


class StageState(Schema):
    """One stage. status is None until its first attempt finishes."""

    status: Status | None = None
    attempts: int = Field(0, ge=0)
    running: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.running:
            if self.status is not None or self.attempts < 1:
                raise ValueError("a running stage has no status yet and at least one attempt")
        elif self.status is None:
            if self.attempts:
                raise ValueError("a stage with attempts is running or has a status")
        elif self.status is not Status.SKIPPED and self.attempts < 1:
            raise ValueError(f"status {self.status} needs at least one attempt")
        if self.status in _NEEDS_REASON and not self.reason:
            raise ValueError(f"status {self.status} needs a reason")
        return self

    def start(self) -> Self:
        """Begin an attempt. A finished stage may run again; FAILED and SKIPPED are final."""
        if self.running:
            raise StateError("already running")
        if self.status in (Status.FAILED, Status.SKIPPED):
            raise StateError(f"cannot start a stage that is {self.status}")
        return self.updated(status=None, running=True, attempts=self.attempts + 1, reason=None)

    def finish(self, status: Status, reason: str | None = None) -> Self:
        if not self.running:
            raise StateError("cannot finish a stage that is not running")
        return self.updated(status=status, running=False, reason=reason)

    def skip(self, reason: str) -> Self:
        """Mark a stage that never ran, such as quality_check when no generator was chosen."""
        if self.running or self.status is not None or self.attempts:
            raise StateError("only a stage that has not started can be skipped")
        return self.updated(status=Status.SKIPPED, reason=reason)


class RunState(Schema):
    """Status and attempts per stage, plus the run-level choices that later stages read."""

    run_id: RunId
    stages: dict[Stage, StageState]
    # None until the quality gate has settled. False needs a reason, True and None do not.
    gan_available: bool | None = None
    gan_unavailable_reason: str | None = None
    # None until the volume rule has run. level 0.0 means real data only.
    volume_decision: VolumeDecision | None = None

    @classmethod
    def new(cls, run_id: str) -> Self:
        return cls(run_id=run_id, stages={stage: StageState() for stage in STAGE_ORDER})

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.stages) != set(Stage):
            raise ValueError("stages must hold every stage exactly once")
        if (self.gan_available is False) != bool(self.gan_unavailable_reason):
            raise ValueError("gan_unavailable_reason is required when, and only when, gan is off")
        return self

    # -- reading

    def status(self, stage: Stage) -> Status | None:
        return self.stages[stage].status

    def attempts(self, stage: Stage) -> int:
        return self.stages[stage].attempts

    def is_done(self, stage: Stage) -> bool:
        """SUCCESS, or SKIPPED with a reason. Both let the next stage start."""
        return self.status(stage) in (Status.SUCCESS, Status.SKIPPED)

    @property
    def is_complete(self) -> bool:
        return all(self.is_done(stage) for stage in STAGE_ORDER)

    @property
    def failed_stage(self) -> Stage | None:
        return next((s for s in STAGE_ORDER if self.status(s) is Status.FAILED), None)

    # -- changing (each returns a new RunState)

    def start(self, stage: Stage) -> Self:
        self._check_order(stage)
        return self._change(stage, StageState.start)

    def finish(self, stage: Stage, status: Status, reason: str | None = None) -> Self:
        return self._change(stage, lambda s: s.finish(status, reason))

    def skip(self, stage: Stage, reason: str) -> Self:
        self._check_order(stage)
        return self._change(stage, lambda s: s.skip(reason))

    def with_gan(self, available: bool, reason: str | None = None) -> Self:
        return self.updated(gan_available=available, gan_unavailable_reason=reason)

    def with_volume_decision(self, decision: VolumeDecision) -> Self:
        return self.updated(volume_decision=decision)

    def _check_order(self, stage: Stage) -> None:
        earlier = STAGE_ORDER[: STAGE_ORDER.index(stage)]
        waiting = [s.value for s in earlier if not self.is_done(s)]
        if waiting:
            raise StateError(f"{stage}: earlier stages are not done: {', '.join(waiting)}")

    def _change(self, stage: Stage, change: Callable[[StageState], StageState]) -> Self:
        try:
            new = change(self.stages[stage])
        except StateError as exc:
            raise StateError(f"{stage}: {exc}") from None
        return self.updated(stages={**self.stages, stage: new})

    # -- saving

    def save(self, path: Path) -> None:
        write_text_atomic(Path(path), self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
