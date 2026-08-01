"""Successful-update-driven single-GPU training loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sakuramoon.checkpoint.policy import (
    FORCED_CHECKPOINT_REASONS,
    CheckpointCadence,
    CheckpointReason,
)
from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.scheduler import CheckpointScheduler
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    StepOptimizer,
)


@dataclass(frozen=True, slots=True)
class LoopResult:
    state: SingleGpuUpdateState
    checkpoint_updates: tuple[int, ...]
    cadence: CheckpointCadence | None = None


class SingleGpuTrainingLoop[BatchT]:
    """Advance scheduler and checkpoint cadence only after successful updates."""

    def __init__(
        self,
        *,
        module: nn.Module,
        optimizer: StepOptimizer,
        loss_fn: Callable[[BatchT], torch.Tensor],
        accumulation_steps: int,
        target_successful_updates: int,
        checkpoint_every_successful_updates: int,
        scheduler_step: Callable[[int], None],
        checkpoint: Callable[[int], None],
        diagnostic_root: Path,
        failure_id: Callable[[str, SingleGpuUpdateState], str],
        state: SingleGpuUpdateState,
        cadence: CheckpointCadence | None = None,
        clock: Callable[[], float] | None = None,
        forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
        checkpoint_event: Callable[[int, CheckpointReason], None] | None = None,
        checkpoint_cadence_event: Callable[
            [int, CheckpointReason, CheckpointCadence], None
        ]
        | None = None,
    ) -> None:
        if (
            type(target_successful_updates) is not int
            or target_successful_updates <= state.successful_updates
            or type(checkpoint_every_successful_updates) is not int
            or checkpoint_every_successful_updates <= 0
        ):
            raise ValueError("loop update targets and checkpoint cadence are invalid")
        self.module = module
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accumulation_steps = accumulation_steps
        self.target_successful_updates = target_successful_updates
        self.checkpoint_every_successful_updates = (
            checkpoint_every_successful_updates
        )
        self.scheduler_step = scheduler_step
        self.checkpoint = checkpoint
        self.diagnostic_root = diagnostic_root
        self.failure_id = failure_id
        self.state: SingleGpuUpdateState = state
        if cadence is not None and cadence.every_successful_updates != (
            checkpoint_every_successful_updates
        ):
            raise ValueError(
                "checkpoint cadence and update cadence must use the same interval"
            )
        if checkpoint_event is not None and not callable(checkpoint_event):
            raise TypeError("checkpoint_event must be callable")
        if checkpoint_cadence_event is not None and not callable(
            checkpoint_cadence_event
        ):
            raise TypeError("checkpoint_cadence_event must be callable")
        if cadence is None and clock is not None:
            raise ValueError("clock requires an explicit checkpoint cadence")
        self._checkpoint_event = checkpoint_event
        self._checkpoint_cadence_event = checkpoint_cadence_event
        self._checkpoint_scheduler = (
            CheckpointScheduler(
                cadence,
                clock=clock if clock is not None else time.time,
                forced_checkpoint=forced_checkpoint,
            )
            if cadence is not None
            else None
        )
        self._forced_checkpoint = forced_checkpoint

    def _diagnose(self, phase: str, exc: Exception) -> None:
        snapshot = FailureSnapshot(
            failure_id=self.failure_id(phase, self.state),
            phase=phase,
            error_type=type(exc).__name__,
            attempted_updates=self.state.attempted_updates,
            successful_updates=self.state.successful_updates,
            effective_samples=self.state.effective_samples,
        )
        try:
            write_failure_bundle(self.diagnostic_root, snapshot)
        except Exception as diagnostic_exc:  # noqa: BLE001 - failure boundary
            raise ExceptionGroup(
                "training failed and diagnostic publication failed",
                [exc, diagnostic_exc],
            ) from None

    def run(self, batches: Iterable[BatchT]) -> LoopResult:
        iterator = iter(batches)
        checkpoint_updates: list[int] = []
        checkpoint_scheduler = self._checkpoint_scheduler
        while self.state.successful_updates < self.target_successful_updates:
            step = SingleGpuStep(
                self.module,
                self.optimizer,
                accumulation_steps=self.accumulation_steps,
                state=self.state,
            )
            try:
                for _ in range(self.accumulation_steps):
                    batch = next(iterator)
                    step.backward(self.loss_fn(batch))
                update = step.finish_update()
            except Exception as exc:
                try:
                    step.abort()
                except Exception as cleanup_exc:  # noqa: BLE001 - cleanup boundary
                    self.state = step.state
                    self._diagnose("update_cleanup", cleanup_exc)
                    raise ExceptionGroup(
                        "training update and gradient cleanup both failed",
                        [exc, cleanup_exc],
                    ) from None
                self.state = step.state
                self._diagnose("update", exc)
                raise
            self.state = update.state
            try:
                decision = None
                decision_reason: CheckpointReason | None = None
                if checkpoint_scheduler is not None:
                    decision = checkpoint_scheduler.due(
                        self.state.successful_updates
                    )
                else:
                    forced = (
                        self._forced_checkpoint(self.state.successful_updates)
                        if self._forced_checkpoint is not None
                        else None
                    )
                    if forced is not None and forced not in FORCED_CHECKPOINT_REASONS:
                        raise ValueError("forced checkpoint reason is not allowed")
                    if forced is not None:
                        decision_reason = forced
                    elif (
                        self.state.successful_updates
                        % self.checkpoint_every_successful_updates
                        == 0
                    ):
                        decision_reason = CheckpointReason.UPDATE_CADENCE
                    else:
                        decision_reason = None
                    if (
                        decision_reason is CheckpointReason.PRE_DECAY
                        and self._checkpoint_event is not None
                    ):
                        self._checkpoint_event(
                            self.state.successful_updates, decision_reason
                        )
                        checkpoint_updates.append(self.state.successful_updates)
                    elif decision_reason is CheckpointReason.PRE_DECAY:
                        self.checkpoint(self.state.successful_updates)
                        checkpoint_updates.append(self.state.successful_updates)

                pre_decay = (
                    decision is not None
                    and decision.reason is CheckpointReason.PRE_DECAY
                )
                if (
                    decision is not None
                    and decision.reason is CheckpointReason.PRE_DECAY
                ):
                    assert checkpoint_scheduler is not None
                    proposed = checkpoint_scheduler.proposed_cadence(decision)
                    if self._checkpoint_cadence_event is not None:
                        self._checkpoint_cadence_event(
                            decision.successful_update, decision.reason, proposed
                        )
                    elif self._checkpoint_event is not None:
                        self._checkpoint_event(
                            decision.successful_update, decision.reason
                        )
                    else:
                        self.checkpoint(decision.successful_update)
                    checkpoint_scheduler.committed(decision)
                    checkpoint_updates.append(decision.successful_update)

                self.scheduler_step(self.state.successful_updates)

                if checkpoint_scheduler is None:
                    if (
                        decision_reason is not None
                        and decision_reason is not CheckpointReason.PRE_DECAY
                    ):
                        if self._checkpoint_cadence_event is not None:
                            raise RuntimeError(
                                "cadence callback requires a checkpoint scheduler"
                            )
                        if self._checkpoint_event is not None:
                            self._checkpoint_event(
                                self.state.successful_updates, decision_reason
                            )
                        else:
                            self.checkpoint(self.state.successful_updates)
                        checkpoint_updates.append(self.state.successful_updates)
                elif decision is not None and not pre_decay:
                    if self._checkpoint_cadence_event is not None:
                        proposed = checkpoint_scheduler.proposed_cadence(decision)
                        self._checkpoint_cadence_event(
                            decision.successful_update, decision.reason, proposed
                        )
                    elif self._checkpoint_event is not None:
                        self._checkpoint_event(
                            decision.successful_update, decision.reason
                        )
                    else:
                        self.checkpoint(decision.successful_update)
                    checkpoint_scheduler.committed(decision)
                    checkpoint_updates.append(decision.successful_update)
            except Exception as exc:
                self._diagnose("post_update", exc)
                raise
        return LoopResult(
            self.state,
            tuple(checkpoint_updates),
            checkpoint_scheduler.cadence if checkpoint_scheduler is not None else None,
        )


__all__ = ["LoopResult", "SingleGpuTrainingLoop"]
