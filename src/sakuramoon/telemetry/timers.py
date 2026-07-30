"""Monotonic CPU and asynchronous CUDA-event phase timing."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from sakuramoon.telemetry.metrics import TIMING_PHASES


@dataclass(frozen=True, slots=True)
class _PendingCudaTiming:
    phase: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class PhaseTimer:
    """Record phases without synchronizing the device at phase boundaries."""

    def __init__(self, *, device: torch.device) -> None:
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("phase timer supports only CPU or CUDA devices")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA phase timing requested without CUDA")
        self.device = device
        self._seconds: dict[str, list[float]] = defaultdict(list)
        self._pending: list[_PendingCudaTiming] = []

    @contextmanager
    def record(self, phase: str) -> Generator[None]:
        if phase not in TIMING_PHASES:
            raise ValueError(f"unknown timing phase: {phase}")
        if self.device.type == "cpu":
            started = time.perf_counter_ns()
            try:
                yield
            finally:
                self._seconds[phase].append(
                    (time.perf_counter_ns() - started) / 1_000_000_000.0
                )
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append(_PendingCudaTiming(phase, start, end))

    def collect_ready(self) -> dict[str, float]:
        """Collect completed CUDA pairs with `query`, never `synchronize`."""

        remaining: list[_PendingCudaTiming] = []
        for pending in self._pending:
            if pending.end.query():
                self._seconds[pending.phase].append(
                    pending.start.elapsed_time(pending.end)  # pyright: ignore[reportUnknownMemberType]
                    / 1000.0
                )
            else:
                remaining.append(pending)
        self._pending = remaining
        return {
            phase: sum(durations)
            for phase, durations in sorted(self._seconds.items())
        }

    @property
    def pending_cuda_pairs(self) -> int:
        return len(self._pending)


__all__ = ["PhaseTimer"]
