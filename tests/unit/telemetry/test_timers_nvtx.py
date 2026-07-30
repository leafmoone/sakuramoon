from __future__ import annotations

import time

import pytest
import torch

from sakuramoon.telemetry.nvtx import nvtx_range
from sakuramoon.telemetry.timers import PhaseTimer


def test_cpu_timer_uses_monotonic_clock_and_rejects_unknown_phase() -> None:
    timer = PhaseTimer(device=torch.device("cpu"))

    with timer.record("data"):
        time.sleep(0.001)

    durations = timer.collect_ready()
    assert durations["data"] >= 0.001
    assert timer.pending_cuda_pairs == 0
    with pytest.raises(ValueError, match="unknown"), timer.record("not-a-phase"):
        pass


def test_nvtx_range_is_balanced_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def push(name: str) -> int:
        calls.append(name)
        return 0

    def pop() -> int:
        calls.append("pop")
        return 0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", pop)

    with pytest.raises(RuntimeError, match="synthetic"), nvtx_range(
        "optimizer", enabled=True
    ):
        raise RuntimeError("synthetic")

    assert calls == ["sakuramoon:optimizer", "pop"]


def test_disabled_nvtx_does_not_probe_or_push(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("disabled NVTX must not probe CUDA"),
    )

    with nvtx_range("data", enabled=False):
        pass
