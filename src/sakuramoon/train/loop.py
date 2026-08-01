"""Successful-update-driven single-GPU training loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import torch
from torch import nn

from sakuramoon.checkpoint.policy import (
    FORCED_CHECKPOINT_REASONS,
    CheckpointCadence,
    CheckpointReason,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.scheduler import CheckpointScheduler
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateResult,
    SingleGpuUpdateState,
    StepOptimizer,
)


@dataclass(frozen=True, slots=True)
class LoopResult:
    state: SingleGpuUpdateState
    checkpoint_updates: tuple[int, ...]
    cadence: CheckpointCadence | None = None


@dataclass(frozen=True, slots=True)
class SuccessfulLoopObservation:
    update: SingleGpuUpdateResult
    checkpoint_reason: CheckpointReason | None
    data_wait_seconds: float
    checkpoint_seconds: float
    phase_timer: PhaseTimer | None


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
            [SingleGpuUpdateState, CheckpointReason, CheckpointCadence], None
        ]
        | None = None,
        phase_timer: PhaseTimer | None = None,
        update_started: Callable[[PhaseTimer | None], None] | None = None,
        successful_update_observer: Callable[[SuccessfulLoopObservation], None]
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
        self.checkpoint_every_successful_updates = checkpoint_every_successful_updates
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
        self._phase_timer = phase_timer
        self._update_started = update_started
        self._successful_update_observer = successful_update_observer

    def _diagnose(self, phase: str, exc: BaseException) -> None:
        snapshot = FailureSnapshot(
            failure_id=self.failure_id(phase, self.state),
            phase=phase,
            error_type=type(exc).__name__,
            attempted_updates=self.state.attempted_updates,
            successful_updates=self.state.successful_updates,
            effective_samples=self.state.effective_samples,
        )
        write_failure_bundle(self.diagnostic_root, snapshot)

    def _raise_with_diagnostics(
        self,
        phase: str,
        primary: BaseException,
        *additional: BaseException,
    ) -> NoReturn:
        errors = [primary, *additional]
        try:
            self._diagnose(phase, primary)
        except BaseException as diagnostic_error:  # noqa: BLE001
            errors.append(diagnostic_error)
        if len(errors) == 1:
            raise primary
        raise BaseExceptionGroup(
            "training failed at a guarded boundary", errors
        ) from None

    @staticmethod
    def _record(timer: PhaseTimer | None, phase: str) -> AbstractContextManager[None]:
        if timer is None:
            return nullcontext()
        return timer.record(phase)

    def run(self, batches: Iterable[BatchT]) -> LoopResult:
        iterator = iter(batches)
        checkpoint_updates: list[int] = []
        checkpoint_scheduler = self._checkpoint_scheduler
        while self.state.successful_updates < self.target_successful_updates:
            phase_timer = (
                PhaseTimer(device=self._phase_timer.device)
                if self._phase_timer is not None
                else None
            )
            if self._update_started is not None:
                self._update_started(phase_timer)
            data_wait_seconds = 0.0
            checkpoint_seconds = 0.0

            @contextmanager
            def checkpoint_wall() -> Generator[None]:
                nonlocal checkpoint_seconds
                started = time.perf_counter_ns()
                try:
                    yield
                finally:
                    checkpoint_seconds += (
                        time.perf_counter_ns() - started
                    ) / 1_000_000_000.0

            step = SingleGpuStep(
                self.module,
                self.optimizer,
                accumulation_steps=self.accumulation_steps,
                state=self.state,
            )
            try:
                for _ in range(self.accumulation_steps):
                    data_started = time.perf_counter_ns()
                    batch = next(iterator)
                    data_wait_seconds += (
                        time.perf_counter_ns() - data_started
                    ) / 1_000_000_000.0
                    per_sample_loss = self.loss_fn(batch)
                    with self._record(phase_timer, "backward"):
                        step.backward(per_sample_loss)
                update = step.finish_update(phase_timer=phase_timer)
            except BaseException as exc:  # noqa: BLE001
                cleanup_error: BaseException | None = None
                try:
                    step.abort()
                except BaseException as error:  # noqa: BLE001
                    cleanup_error = error
                self.state = step.state
                if cleanup_error is None:
                    self._raise_with_diagnostics("update", exc)
                assert cleanup_error is not None
                self._raise_with_diagnostics("update", exc, cleanup_error)
            self.state = update.state
            try:
                decision = None
                decision_reason: CheckpointReason | None = None
                completed_checkpoint_reason: CheckpointReason | None = None
                if checkpoint_scheduler is not None:
                    decision = checkpoint_scheduler.due(self.state.successful_updates)
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
                        with checkpoint_wall():
                            self._checkpoint_event(
                                self.state.successful_updates, decision_reason
                            )
                        checkpoint_updates.append(self.state.successful_updates)
                        completed_checkpoint_reason = decision_reason
                    elif decision_reason is CheckpointReason.PRE_DECAY:
                        with checkpoint_wall():
                            self.checkpoint(self.state.successful_updates)
                        checkpoint_updates.append(self.state.successful_updates)
                        completed_checkpoint_reason = decision_reason

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
                        with checkpoint_wall():
                            self._checkpoint_cadence_event(
                                self.state, decision.reason, proposed
                            )
                    elif self._checkpoint_event is not None:
                        with checkpoint_wall():
                            self._checkpoint_event(
                                decision.successful_update, decision.reason
                            )
                    else:
                        with checkpoint_wall():
                            self.checkpoint(decision.successful_update)
                    checkpoint_scheduler.committed(decision)
                    checkpoint_updates.append(decision.successful_update)
                    completed_checkpoint_reason = decision.reason

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
                            with checkpoint_wall():
                                self._checkpoint_event(
                                    self.state.successful_updates, decision_reason
                                )
                        else:
                            with checkpoint_wall():
                                self.checkpoint(self.state.successful_updates)
                        checkpoint_updates.append(self.state.successful_updates)
                        completed_checkpoint_reason = decision_reason
                elif decision is not None and not pre_decay:
                    if self._checkpoint_cadence_event is not None:
                        proposed = checkpoint_scheduler.proposed_cadence(decision)
                        with checkpoint_wall():
                            self._checkpoint_cadence_event(
                                self.state, decision.reason, proposed
                            )
                    elif self._checkpoint_event is not None:
                        with checkpoint_wall():
                            self._checkpoint_event(
                                decision.successful_update, decision.reason
                            )
                    else:
                        with checkpoint_wall():
                            self.checkpoint(decision.successful_update)
                    checkpoint_scheduler.committed(decision)
                    checkpoint_updates.append(decision.successful_update)
                    completed_checkpoint_reason = decision.reason
                if self._successful_update_observer is not None:
                    self._successful_update_observer(
                        SuccessfulLoopObservation(
                            update,
                            completed_checkpoint_reason,
                            data_wait_seconds,
                            checkpoint_seconds,
                            phase_timer,
                        )
                    )
            except BaseException as exc:  # noqa: BLE001
                self._raise_with_diagnostics("post_update", exc)
        return LoopResult(
            self.state,
            tuple(checkpoint_updates),
            checkpoint_scheduler.cadence if checkpoint_scheduler is not None else None,
        )


__all__ = ["LoopResult", "SingleGpuTrainingLoop", "SuccessfulLoopObservation"]
