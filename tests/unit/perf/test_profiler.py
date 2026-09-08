"""Unit tests for the P0 profiler helpers (enum-robust probe, validation)."""

from __future__ import annotations

import types

import pytest
import torch

from sakuramoon.perf import profiler as perf_profiler
from sakuramoon.perf.profiler import (
    DEFAULT_PROFILER_UPDATES,
    unavailable_snapshot,
    validate_profiler_updates,
)


class _FakeEvent:
    def __init__(self, device_type: object, device_time: float = 0.0) -> None:
        self.device_type = device_type
        self.self_device_time_total = device_time


def _cuda_enum():
    device_type = getattr(torch.autograd, "DeviceType", None)
    assert device_type is not None
    return device_type.CUDA


def _cpu_enum():
    device_type = getattr(torch.autograd, "DeviceType", None)
    assert device_type is not None
    return device_type.CPU


class TestIsCudaDeviceEvent:
    """§6: robust device-type detection; CPU must never read as CUDA."""

    def test_real_torch_cuda_enum(self) -> None:
        assert perf_profiler.is_cuda_device_event(_FakeEvent(_cuda_enum()))

    def test_real_torch_cpu_enum_is_not_cuda(self) -> None:
        assert not perf_profiler.is_cuda_device_event(_FakeEvent(_cpu_enum()))

    def test_string_forms(self) -> None:
        assert perf_profiler.is_cuda_device_event(_FakeEvent("CUDA"))
        assert perf_profiler.is_cuda_device_event(_FakeEvent("DeviceType.CUDA"))
        assert perf_profiler.is_cuda_device_event(_FakeEvent("cuda"))

    def test_cpu_string_forms_are_not_cuda(self) -> None:
        assert not perf_profiler.is_cuda_device_event(_FakeEvent("CPU"))
        assert not perf_profiler.is_cuda_device_event(_FakeEvent("DeviceType.CPU"))
        assert not perf_profiler.is_cuda_device_event(_FakeEvent("cpu"))

    def test_missing_attribute_is_not_cuda(self) -> None:
        assert not perf_profiler.is_cuda_device_event(types.SimpleNamespace())

    def test_none_device_type_is_not_cuda(self) -> None:
        assert not perf_profiler.is_cuda_device_event(_FakeEvent(None))


class TestValidateProfilerUpdates:
    """§10-H: profiler_updates validation (0 rejected, default 1)."""

    def test_default_is_one(self) -> None:
        assert DEFAULT_PROFILER_UPDATES == 1
        assert validate_profiler_updates(DEFAULT_PROFILER_UPDATES) == 1

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="profiler_updates"):
            validate_profiler_updates(0)

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="profiler_updates"):
            validate_profiler_updates(-3)

    def test_non_int_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="profiler_updates"):
            validate_profiler_updates("1")  # type: ignore[arg-type]

    def test_positive_is_accepted(self) -> None:
        assert validate_profiler_updates(3) == 3


class TestUnavailableSnapshot:
    def test_unavailable_shape(self) -> None:
        snapshot = unavailable_snapshot()
        assert snapshot.device_trace_available is False
        assert snapshot.top_by_self_device_time == ()
        assert snapshot.top_by_total_device_time == ()
        assert snapshot.trace_path is None
        payload = snapshot.to_dict()
        assert payload["profiler_updates"] == 1
        assert payload["top_by_self_device_time"] == []
        assert payload["top_by_total_device_time"] == []


class TestCaptureUnavailableExecutesNoWorkload:
    """§7: an unavailable probe must not run the scratch workload at all."""

    def test_capture_without_cuda_skips_workload(self, monkeypatch) -> None:
        called = {"workload": False}

        def workload() -> None:
            called["workload"] = True

        # Force the probe to report unavailable (this unit suite has no
        # CUDA on most runners; make it deterministic either way).
        monkeypatch.setattr(
            perf_profiler, "profiler_device_trace_available", lambda: False
        )
        snapshot = perf_profiler.capture_profiler_window(
            workload,
            output_root=None,
            rank=0,  # type: ignore[arg-type]
        )
        assert snapshot.device_trace_available is False
        assert called["workload"] is False
