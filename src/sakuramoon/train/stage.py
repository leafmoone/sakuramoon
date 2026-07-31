"""Explicit manual stage transitions and growth progress."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sakuramoon.checkpoint.schema import GrowthCheckpointState, RawCheckpointState
from sakuramoon.model.growth import (
    active_slot_ids,
    growth_ramp_updates,
    half_cosine_growth_alpha,
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    predecessor: str | None
    world_size: int
    depth: int
    resolution: int


STAGE_SPECS: dict[str, StageSpec] = {
    "S0": StageSpec("S0", None, 1, 16, 256),
    "S1": StageSpec("S1", "S0", 4, 16, 256),
    "G1": StageSpec("G1", "S1", 4, 20, 256),
    "S2": StageSpec("S2", "G1", 4, 20, 512),
    "G2": StageSpec("G2", "S2", 4, 24, 512),
    "S3": StageSpec("S3", "G2", 4, 24, 512),
    "H1": StageSpec("H1", "S3", 4, 24, 768),
    "H2": StageSpec("H2", "H1", 4, 24, 1024),
}


class ForcedCheckpoint(StrEnum):
    PRE_TRANSITION = "pre-transition"
    POST_TRANSITION = "post-transition"
    RAMP_MIDPOINT = "ramp-midpoint"
    RAMP_END = "ramp-end"


@dataclass(frozen=True, slots=True)
class StageTransitionRequest:
    source_stage: str
    target_stage: str
    source_checkpoint: Path
    source_checkpoint_id: str
    planned_updates: int
    manual_approval: bool

    def __post_init__(self) -> None:
        source = stage_spec(self.source_stage)
        target = stage_spec(self.target_stage)
        if target.predecessor != source.name:
            raise ValueError("target stage does not have the requested unique predecessor")
        if target.name in {"H1", "H2"}:
            raise ValueError("H1/H2 transitions remain disabled pending separate approval")
        changed = sum(
            left != right
            for left, right in (
                (source.world_size, target.world_size),
                (source.depth, target.depth),
                (source.resolution, target.resolution),
            )
        )
        if changed != 1:
            raise ValueError("a stage transition must change exactly one primary axis")
        if (
            not self.source_checkpoint_id
            or self.source_checkpoint.is_symlink()
            or not self.source_checkpoint.is_dir()
        ):
            raise ValueError("transition requires an existing source checkpoint directory")
        if type(self.planned_updates) is not int or self.planned_updates <= 0:
            raise ValueError("planned updates must be a positive integer")
        if self.manual_approval is not True:
            raise ValueError("stage transition requires explicit manual approval")

    @property
    def is_growth(self) -> bool:
        return stage_spec(self.source_stage).depth != stage_spec(self.target_stage).depth

    @property
    def ramp_updates(self) -> int:
        return growth_ramp_updates(self.planned_updates) if self.is_growth else 0

    def forced_checkpoints(self) -> tuple[ForcedCheckpoint, ...]:
        if not self.is_growth:
            return (ForcedCheckpoint.PRE_TRANSITION, ForcedCheckpoint.POST_TRANSITION)
        return (
            ForcedCheckpoint.PRE_TRANSITION,
            ForcedCheckpoint.POST_TRANSITION,
            ForcedCheckpoint.RAMP_MIDPOINT,
            ForcedCheckpoint.RAMP_END,
        )


@dataclass(frozen=True, slots=True)
class StageReadiness:
    data_exposure_complete: bool
    flops_and_updates_complete: bool
    recent_updates_stable: bool
    performance_gate_passed: bool
    recovery_gate_passed: bool

    def __post_init__(self) -> None:
        if not all(
            type(value) is bool
            for value in (
                self.data_exposure_complete,
                self.flops_and_updates_complete,
                self.recent_updates_stable,
                self.performance_gate_passed,
                self.recovery_gate_passed,
            )
        ):
            raise TypeError("stage readiness gates must be explicit booleans")

    @property
    def stage_ready(self) -> bool:
        return all(
            (
                self.data_exposure_complete,
                self.flops_and_updates_complete,
                self.recent_updates_stable,
                self.performance_gate_passed,
                self.recovery_gate_passed,
            )
        )


@dataclass(frozen=True, slots=True)
class GrowthProgress:
    start_successful_update: int
    ramp_updates: int

    def __post_init__(self) -> None:
        if type(self.start_successful_update) is not int or self.start_successful_update < 0:
            raise ValueError("growth start update must be a nonnegative integer")
        if type(self.ramp_updates) is not int or not 1000 <= self.ramp_updates <= 5000:
            raise ValueError("growth ramp must contain 1,000-5,000 updates")

    @classmethod
    def from_checkpoint(cls, state: RawCheckpointState) -> GrowthProgress:
        start = state.growth.ramp_start_successful_update
        updates = state.growth.ramp_updates
        if start is None or updates is None:
            raise ValueError("checkpoint does not contain active growth progress")
        return cls(start_successful_update=start, ramp_updates=updates)

    def elapsed(self, successful_update: int) -> int:
        if type(successful_update) is not int or successful_update < self.start_successful_update:
            raise ValueError("successful update precedes the growth start")
        return successful_update - self.start_successful_update

    def alpha(self, successful_update: int) -> float:
        return half_cosine_growth_alpha(
            self.elapsed(successful_update), self.ramp_updates
        )

    def forced_checkpoint(self, successful_update: int) -> ForcedCheckpoint | None:
        elapsed = self.elapsed(successful_update)
        if elapsed == 0:
            return ForcedCheckpoint.POST_TRANSITION
        if elapsed == self.ramp_updates // 2:
            return ForcedCheckpoint.RAMP_MIDPOINT
        if elapsed == self.ramp_updates:
            return ForcedCheckpoint.RAMP_END
        return None


def stage_spec(name: str) -> StageSpec:
    try:
        return STAGE_SPECS[name]
    except KeyError as error:
        raise ValueError("stage name is not in the approved transition graph") from error


def validate_checkpoint_stage(state: RawCheckpointState, expected: StageSpec) -> None:
    if state.growth.active_slot_ids != active_slot_ids(expected.depth):
        raise ValueError("source checkpoint slots do not match the source stage")
    if (
        state.growth.stage != expected.name
        or state.growth.world_size != expected.world_size
        or state.growth.resolution != expected.resolution
    ):
        raise ValueError("source checkpoint axes do not match the source stage")


def transition_checkpoint_state(
    source: RawCheckpointState,
    request: StageTransitionRequest,
) -> RawCheckpointState:
    source_spec = stage_spec(request.source_stage)
    target_spec = stage_spec(request.target_stage)
    validate_checkpoint_stage(source, source_spec)
    if source.growth.alpha != 1.0:
        raise ValueError("a stage transition requires a completed growth ramp")
    return RawCheckpointState(
        trainer=source.trainer,
        growth=GrowthCheckpointState(
            active_slot_ids=active_slot_ids(target_spec.depth),
            alpha=0.0 if request.is_growth else 1.0,
            stage=target_spec.name,
            world_size=target_spec.world_size,
            resolution=target_spec.resolution,
            ramp_start_successful_update=(
                source.trainer.successful_updates if request.is_growth else None
            ),
            ramp_updates=request.ramp_updates if request.is_growth else None,
        ),
    )


__all__ = [
    "STAGE_SPECS",
    "ForcedCheckpoint",
    "GrowthProgress",
    "StageReadiness",
    "StageSpec",
    "StageTransitionRequest",
    "stage_spec",
    "transition_checkpoint_state",
    "validate_checkpoint_stage",
]
