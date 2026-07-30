"""Successful-update-driven single-GPU training loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    StepOptimizer,
)


@dataclass(frozen=True, slots=True)
class LoopResult:
    state: SingleGpuUpdateState
    checkpoint_updates: tuple[int, ...]


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
                self.scheduler_step(self.state.successful_updates)
                if (
                    self.state.successful_updates
                    % self.checkpoint_every_successful_updates
                    == 0
                ):
                    self.checkpoint(self.state.successful_updates)
                    checkpoint_updates.append(self.state.successful_updates)
            except Exception as exc:
                self._diagnose("post_update", exc)
                raise
        return LoopResult(self.state, tuple(checkpoint_updates))


__all__ = ["LoopResult", "SingleGpuTrainingLoop"]
