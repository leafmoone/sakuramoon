"""Successful-update adapter connecting the training step to the benchmark harness."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from sakuramoon.telemetry.profiler import BenchmarkObservation, StepPayload
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.step import SingleGpuStep, SingleGpuUpdateState, StepOptimizer


@dataclass(frozen=True, slots=True)
class MeasuredMicrobatch:
    per_sample_loss: torch.Tensor
    image_tokens: int
    text_tokens: int
    dit_flops: int
    sample_ids: tuple[str, ...]
    shape_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.per_sample_loss.ndim != 1 or self.per_sample_loss.numel() <= 0:
            raise ValueError("benchmark loss must be a nonempty per-sample vector")
        if (
            len(self.sample_ids) != self.per_sample_loss.numel()
            or len(self.shape_keys) != self.per_sample_loss.numel()
        ):
            raise ValueError("benchmark observation count must match per-sample loss")
        for name in ("image_tokens", "text_tokens", "dit_flops"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class SingleGpuStepBenchmarkAdapter[BatchT]:
    """Execute actual successful optimizer updates and expose measured facts only."""

    def __init__(
        self,
        *,
        module: nn.Module,
        optimizer: StepOptimizer,
        batches: Iterator[BatchT],
        measure_microbatch: Callable[[BatchT, PhaseTimer], MeasuredMicrobatch],
        accumulation_steps: int,
        state: SingleGpuUpdateState,
        scheduler_step: Callable[[int], None],
        checkpoint_every_successful_updates: int,
        checkpoint: Callable[[int], tuple[Path, ...]],
    ) -> None:
        if type(accumulation_steps) is not int or accumulation_steps <= 0:
            raise ValueError("benchmark accumulation must be positive")
        if (
            type(checkpoint_every_successful_updates) is not int
            or checkpoint_every_successful_updates <= 0
        ):
            raise ValueError("benchmark checkpoint cadence must be positive")
        self.module = module
        self.optimizer = optimizer
        self.batches = batches
        self.measure_microbatch = measure_microbatch
        self.accumulation_steps = accumulation_steps
        self.state = state
        self.scheduler_step = scheduler_step
        self.checkpoint_every_successful_updates = (
            checkpoint_every_successful_updates
        )
        self.checkpoint = checkpoint

    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        del measured
        if update != self.state.successful_updates + 1:
            raise ValueError("benchmark update is not contiguous with training state")
        parameter = next(self.module.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        timer = PhaseTimer(device=device)
        samples = 0
        image_tokens = 0
        text_tokens = 0
        dit_flops = 0
        sample_ids: list[str] = []
        shape_keys: list[str] = []
        host_phase_seconds: dict[str, float] = {}
        step = SingleGpuStep(
            self.module,
            self.optimizer,
            accumulation_steps=self.accumulation_steps,
            state=self.state,
        )
        try:
            for _ in range(self.accumulation_steps):
                data_started = time.perf_counter_ns()
                with timer.record("data"):
                    batch = next(self.batches)
                host_phase_seconds["data"] = host_phase_seconds.get("data", 0.0) + (
                    time.perf_counter_ns() - data_started
                ) / 1_000_000_000.0
                phase_counts_before = {
                    phase: timer.recorded_count(phase)
                    for phase in ("dit_forward", "loss")
                }
                measurement = self.measure_microbatch(batch, timer)
                if any(
                    timer.recorded_count(phase) != count + 1
                    for phase, count in phase_counts_before.items()
                ):
                    raise RuntimeError(
                        "benchmark measurement must time DiT forward and loss separately"
                    )
                with timer.record("backward"):
                    step.backward(measurement.per_sample_loss)
                samples += measurement.per_sample_loss.numel()
                image_tokens += measurement.image_tokens
                text_tokens += measurement.text_tokens
                dit_flops += measurement.dit_flops
                sample_ids.extend(measurement.sample_ids)
                shape_keys.extend(measurement.shape_keys)
            result = step.finish_update(phase_timer=timer)
        except Exception:
            step.abort()
            self.state = step.state
            raise
        self.state = result.state
        if self.state.successful_updates != update:
            raise RuntimeError("training step did not produce the requested successful update")
        self.scheduler_step(update)
        checkpoint_paths: tuple[Path, ...] = ()
        if update % self.checkpoint_every_successful_updates == 0:
            checkpoint_started = time.perf_counter_ns()
            with timer.record("checkpoint"):
                checkpoint_paths = self.checkpoint(update)
            host_phase_seconds["checkpoint"] = (
                time.perf_counter_ns() - checkpoint_started
            ) / 1_000_000_000.0
        return StepPayload(
            update,
            timer,
            samples,
            image_tokens,
            text_tokens,
            dit_flops,
            BenchmarkObservation(tuple(sample_ids), tuple(shape_keys)),
            checkpoint_paths,
            host_phase_seconds,
        )


__all__ = ["MeasuredMicrobatch", "SingleGpuStepBenchmarkAdapter"]
