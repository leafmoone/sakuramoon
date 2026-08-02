"""Successful-update schedulers used by the single-GPU training loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sakuramoon.checkpoint.policy import (
    FORCED_CHECKPOINT_REASONS,
    CheckpointCadence,
    CheckpointReason,
)


@dataclass(frozen=True, slots=True)
class CheckpointDecision:
    """A checkpoint decision with its durable wall-clock audit timestamp."""

    successful_update: int
    wall_clock_unix_seconds: float
    reason: CheckpointReason

    def __post_init__(self) -> None:
        if (
            type(self.successful_update) is not int
            or self.successful_update < 0
            or type(self.wall_clock_unix_seconds) is not float
            or self.wall_clock_unix_seconds < 0.0
            or type(self.reason) is not CheckpointReason
        ):
            raise ValueError("checkpoint decision is invalid")


class CheckpointScheduler:
    """Resolve update/forced cadence and commit only after a durable save.

    ``due`` is side-effect free.  Callers must invoke ``committed`` only after
    the checkpoint callback returns successfully; a failed callback therefore
    leaves the previous update and wall-clock audit anchors intact for diagnostics
    or an explicit retry. Elapsed wall time is never a checkpoint trigger.
    """

    def __init__(
        self,
        cadence: CheckpointCadence,
        *,
        clock: Callable[[], float] = time.time,
        forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    ) -> None:
        if type(cadence) is not CheckpointCadence:
            raise TypeError("cadence must be a CheckpointCadence")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if forced_checkpoint is not None and not callable(forced_checkpoint):
            raise TypeError("forced_checkpoint must be callable")
        self._cadence = cadence
        self._clock = clock
        self._forced_checkpoint = forced_checkpoint

    @property
    def cadence(self) -> CheckpointCadence:
        return self._cadence

    def due(self, successful_update: int) -> CheckpointDecision | None:
        """Return the next due reason without advancing the cadence state."""

        now = self._clock()
        if type(now) is not float:
            raise TypeError("checkpoint clock must return a float")
        forced = (
            self._forced_checkpoint(successful_update)
            if self._forced_checkpoint is not None
            else None
        )
        if forced is not None and forced not in FORCED_CHECKPOINT_REASONS:
            raise ValueError("forced checkpoint reason is not allowed")
        reason = self._cadence.due(
            successful_update=successful_update,
            wall_clock_unix_seconds=now,
            forced=forced,
        )
        if reason is None:
            return None
        return CheckpointDecision(successful_update, now, reason)

    def proposed_cadence(self, decision: CheckpointDecision) -> CheckpointCadence:
        if type(decision) is not CheckpointDecision:
            raise TypeError("decision must be a CheckpointDecision")
        return self._cadence.committed(
            successful_update=decision.successful_update,
            wall_clock_unix_seconds=decision.wall_clock_unix_seconds,
            reason=decision.reason,
        )

    def committed(self, decision: CheckpointDecision) -> None:
        """Advance cadence after the callback durably published ``decision``."""

        if type(decision) is not CheckpointDecision:
            raise TypeError("decision must be a CheckpointDecision")
        if decision.successful_update < self._cadence.last_successful_update:
            raise ValueError("checkpoint decision precedes the committed cadence")
        self._cadence = self._cadence.committed(
            successful_update=decision.successful_update,
            wall_clock_unix_seconds=decision.wall_clock_unix_seconds,
            reason=decision.reason,
        )


__all__ = ["CheckpointDecision", "CheckpointScheduler"]
