"""Growth progress over the successful-update coordinate.

The historical stage graph (S0/S1/G1/S2/G2/S3/H1/H2) is gone: run axes
(depth, resolution, world size, budget) come from the resolved config and
from checkpoint state, never from a code-level stage table.  What remains
here is the growth-ramp bookkeeping that the training loop needs: the
in-flight ramp anchor, the half-cosine alpha, and the ramp's forced
checkpoint points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sakuramoon.checkpoint.schema import (
    CheckpointReason,
    GrowthCheckpointState,
    RawCheckpointState,
)
from sakuramoon.model.growth import half_cosine_growth_alpha


class ForcedCheckpoint(StrEnum):
    PRE_TRANSITION = "pre-transition"
    POST_TRANSITION = "post-transition"
    RAMP_MIDPOINT = "ramp-midpoint"
    RAMP_END = "ramp-end"


_FORCED_CHECKPOINT_REASONS: dict[ForcedCheckpoint, CheckpointReason] = {
    ForcedCheckpoint.PRE_TRANSITION: CheckpointReason.PRE_TRANSITION,
    ForcedCheckpoint.POST_TRANSITION: CheckpointReason.POST_TRANSITION,
    ForcedCheckpoint.RAMP_MIDPOINT: CheckpointReason.RAMP_MIDPOINT,
    ForcedCheckpoint.RAMP_END: CheckpointReason.RAMP_END,
}


def checkpoint_reason(forced: ForcedCheckpoint) -> CheckpointReason:
    """Map a growth event to the scheduler's exact enum type."""

    if type(forced) is not ForcedCheckpoint:
        raise TypeError("forced checkpoint must use the event enum")
    return _FORCED_CHECKPOINT_REASONS[forced]


@dataclass(frozen=True, slots=True)
class GrowthProgress:
    """An in-flight growth ramp anchored at a successful update."""

    start_successful_update: int
    ramp_updates: int

    def __post_init__(self) -> None:
        if type(self.start_successful_update) is not int or self.start_successful_update < 0:
            raise ValueError("growth start update must be a nonnegative integer")
        if type(self.ramp_updates) is not int or self.ramp_updates <= 0:
            raise ValueError("growth ramp must be a positive update count")

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


def canonical_growth_alpha(
    growth: GrowthCheckpointState, successful_update: int
) -> float:
    """Return the alpha for a persisted successful-update edge."""

    if type(successful_update) is not int or successful_update < 0:
        raise ValueError("successful update must be a nonnegative integer")
    if growth.ramp_start_successful_update is None:
        return 1.0
    assert growth.ramp_updates is not None
    if successful_update < growth.ramp_start_successful_update:
        raise ValueError("successful update precedes growth ramp origin")
    return half_cosine_growth_alpha(
        successful_update - growth.ramp_start_successful_update,
        growth.ramp_updates,
    )


__all__ = [
    "ForcedCheckpoint",
    "GrowthProgress",
    "canonical_growth_alpha",
    "checkpoint_reason",
]
