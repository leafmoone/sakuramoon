"""Fail-closed end-to-end benchmark, hotspot, comparison, and trace contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

import torch

from sakuramoon.telemetry.metrics import CORE_TIMING_PHASES, TIMING_PHASES
from sakuramoon.telemetry.timers import PhaseTimer

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _finite_nonnegative(name: str, value: float, *, positive: bool = False) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or (positive and value == 0.0)
    ):
        boundary = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {boundary} float")


def _nonnegative_int(name: str, value: int, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} has an invalid integer value")


def stream_sha256(path: Path) -> str:
    """Hash a regular file without loading a profiler trace into RAM."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("hashed artifact must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    kind: Literal["candidate", "final"]
    warmup_updates: int
    measured_updates: int
    starting_successful_update: int
    checkpoint_every_updates: int

    def __post_init__(self) -> None:
        if self.kind not in ("candidate", "final"):
            raise ValueError("benchmark kind is invalid")
        if self.warmup_updates != 100:
            raise ValueError("benchmark requires exactly 100 warmup updates")
        if self.kind == "candidate" and self.measured_updates != 500:
            raise ValueError("candidate benchmark requires 500 measured updates")
        if self.kind == "final" and self.measured_updates < 1000:
            raise ValueError("final benchmark requires at least 1,000 measured updates")
        _nonnegative_int("starting_successful_update", self.starting_successful_update)
        _nonnegative_int(
            "checkpoint_every_updates", self.checkpoint_every_updates, positive=True
        )
        first_measured = self.starting_successful_update + self.warmup_updates + 1
        last_measured = first_measured + self.measured_updates - 1
        first_checkpoint = (
            (first_measured + self.checkpoint_every_updates - 1)
            // self.checkpoint_every_updates
            * self.checkpoint_every_updates
        )
        if first_checkpoint > last_measured:
            raise ValueError(
                "benchmark measured window must include a configured checkpoint cadence"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkWorkloadIdentity:
    normalized_config_sha256: str
    checkpoint_id: str
    checkpoint_sha256: str
    data_sequence_sha256: str
    shape_distribution_sha256: str
    software_lock_sha256: str
    hardware_id: str
    local_batch: int
    accumulation: int
    world_size: int

    def __post_init__(self) -> None:
        for name in (
            "normalized_config_sha256",
            "checkpoint_sha256",
            "data_sequence_sha256",
            "shape_distribution_sha256",
            "software_lock_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be SHA-256")
        if not self.checkpoint_id or not self.hardware_id:
            raise ValueError("workload identity strings must not be empty")
        for name in ("local_batch", "accumulation", "world_size"):
            _nonnegative_int(name, getattr(self, name), positive=True)


@dataclass(frozen=True, slots=True)
class BenchmarkVariant:
    name: str
    resolved_config_sha256: str
    source_commit: str
    build_sha256: str
    backend: str
    enabled_features: tuple[str, ...]
    changed_config_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None or not self.backend:
            raise ValueError("benchmark variant name/backend is invalid")
        if _SHA256.fullmatch(self.resolved_config_sha256) is None:
            raise ValueError("variant resolved config must be SHA-256")
        if _SHA256.fullmatch(self.build_sha256) is None:
            raise ValueError("variant build identity must be SHA-256")
        if _COMMIT.fullmatch(self.source_commit) is None:
            raise ValueError("variant source commit must be a full commit SHA")
        for name, values in (
            ("enabled_features", self.enabled_features),
            ("changed_config_keys", self.changed_config_keys),
        ):
            if (
                type(values) is not tuple
                or any(type(value) is not str or not value for value in values)
                or len(set(values)) != len(values)
                or tuple(sorted(values)) != values
            ):
                raise ValueError(f"{name} must be a sorted unique tuple")


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    workload: BenchmarkWorkloadIdentity
    variant: BenchmarkVariant

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    sample_ids: tuple[str, ...]
    shape_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.sample_ids
            or len(self.sample_ids) != len(self.shape_keys)
            or any(type(value) is not str or not value for value in self.sample_ids)
            or any(type(value) is not str or not value for value in self.shape_keys)
        ):
            raise ValueError("benchmark observation must bind every sample ID and shape")


def _workload_observation_line(
    successful_update: int,
    values: tuple[str, ...],
    *,
    field: Literal["sample_ids", "shape_keys"],
) -> bytes:
    _nonnegative_int("successful_update", successful_update, positive=True)
    return (
        json.dumps(
            {"successful_update": successful_update, field: list(values)},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def canonical_workload_artifact_bytes(
    observations: Sequence[tuple[int, BenchmarkObservation]],
    *,
    kind: Literal["data_sequence", "shape_distribution"],
) -> bytes:
    """Encode the exact runtime observation stream hashed by the harness."""

    field: Literal["sample_ids", "shape_keys"] = (
        "sample_ids" if kind == "data_sequence" else "shape_keys"
    )
    return b"".join(
        _workload_observation_line(update, getattr(observation, field), field=field)
        for update, observation in observations
    )


@dataclass(frozen=True, slots=True)
class StepPayload:
    """Successful-update facts; the harness supplies timing and memory itself."""

    successful_update: int
    phase_timer: PhaseTimer
    samples: int
    image_tokens: int
    text_tokens: int
    dit_flops: int
    observation: BenchmarkObservation
    checkpoint_paths: tuple[Path, ...]
    host_phase_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        _nonnegative_int("successful_update", self.successful_update, positive=True)
        for name in ("samples", "image_tokens", "text_tokens", "dit_flops"):
            _nonnegative_int(name, getattr(self, name), positive=True)
        if self.samples != len(self.observation.sample_ids):
            raise ValueError("benchmark observation count differs from samples")
        if type(self.phase_timer) is not PhaseTimer:
            raise TypeError("successful update requires a PhaseTimer")
        if (
            type(self.checkpoint_paths) is not tuple
            or len(set(self.checkpoint_paths)) != len(self.checkpoint_paths)
        ):
            raise ValueError("checkpoint paths must be a unique tuple")
        if not set(self.host_phase_seconds).issubset(TIMING_PHASES):
            raise ValueError("host phase timing contains an unknown phase")
        for phase, seconds in self.host_phase_seconds.items():
            _finite_nonnegative(f"host_phase_seconds.{phase}", seconds)
        object.__setattr__(
            self,
            "host_phase_seconds",
            MappingProxyType(dict(self.host_phase_seconds)),
        )


class BenchmarkStepAdapter(Protocol):
    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload: ...


@dataclass(frozen=True, slots=True)
class CompileCounters:
    compile_count: int
    recompile_count: int
    fallback_count: int

    def __post_init__(self) -> None:
        for name in ("compile_count", "recompile_count", "fallback_count"):
            _nonnegative_int(name, getattr(self, name))


class CompileCounterProbe(Protocol):
    def snapshot(self) -> CompileCounters: ...


@dataclass(frozen=True, slots=True)
class CompileWindowEvidence:
    before: CompileCounters
    after_warmup: CompileCounters
    after_measured: CompileCounters
    warmup_seconds: float

    def __post_init__(self) -> None:
        _finite_nonnegative("warmup_seconds", self.warmup_seconds, positive=True)
        for name in ("compile_count", "recompile_count", "fallback_count"):
            values = (
                getattr(self.before, name),
                getattr(self.after_warmup, name),
                getattr(self.after_measured, name),
            )
            if values != tuple(sorted(values)):
                raise ValueError("compile counters must be monotonic")

    @property
    def measured_recompiles(self) -> int:
        return self.after_measured.recompile_count - self.after_warmup.recompile_count

    @property
    def measured_fallbacks(self) -> int:
        return self.after_measured.fallback_count - self.after_warmup.fallback_count

    @property
    def measured_compiles(self) -> int:
        return self.after_measured.compile_count - self.after_warmup.compile_count


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    measured_update: int
    successful_update: int
    step_seconds: float
    cuda_span_seconds: float
    phase_seconds: Mapping[str, float]
    samples: int
    image_tokens: int
    text_tokens: int
    dit_flops: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    host_rss_bytes: int
    pinned_ram_bytes: int
    host_swap_bytes: int
    checkpoint_bytes: int
    checkpoint_seconds: float

    def __post_init__(self) -> None:
        for name in ("measured_update", "successful_update"):
            _nonnegative_int(name, getattr(self, name), positive=True)
        _finite_nonnegative("step_seconds", self.step_seconds, positive=True)
        _finite_nonnegative("cuda_span_seconds", self.cuda_span_seconds)
        if self.cuda_span_seconds > self.step_seconds * 1.05 + 0.001:
            raise ValueError("CUDA span materially exceeds measured wall time")
        for name in ("samples", "image_tokens", "text_tokens", "dit_flops"):
            _nonnegative_int(name, getattr(self, name), positive=True)
        for name in (
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
            "host_rss_bytes",
            "pinned_ram_bytes",
            "host_swap_bytes",
            "checkpoint_bytes",
        ):
            _nonnegative_int(name, getattr(self, name))
        _finite_nonnegative("checkpoint_seconds", self.checkpoint_seconds)
        phases = set(self.phase_seconds)
        if not set(CORE_TIMING_PHASES).issubset(phases) or not phases.issubset(
            TIMING_PHASES
        ):
            raise ValueError("benchmark phases are incomplete or unknown")
        for phase, value in self.phase_seconds.items():
            _finite_nonnegative(f"phase_seconds.{phase}", value)
            if value > self.step_seconds * 1.05:
                raise ValueError("an individual phase exceeds measured wall time")
        object.__setattr__(
            self, "phase_seconds", MappingProxyType(dict(self.phase_seconds))
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_seconds": self.checkpoint_seconds,
            "cuda_span_seconds": self.cuda_span_seconds,
            "dit_flops": self.dit_flops,
            "host_rss_bytes": self.host_rss_bytes,
            "host_swap_bytes": self.host_swap_bytes,
            "image_tokens": self.image_tokens,
            "measured_update": self.measured_update,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "phase_seconds": dict(self.phase_seconds),
            "pinned_ram_bytes": self.pinned_ram_bytes,
            "samples": self.samples,
            "step_seconds": self.step_seconds,
            "successful_update": self.successful_update,
            "text_tokens": self.text_tokens,
        }


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    sampled_updates: int
    trace_wall_seconds: float
    gpu_active_seconds: float
    gpu_idle_seconds: float
    gpu_unattributed_seconds: float
    kernel_launches: int
    kernel_gap_seconds: float
    nccl_seconds: float
    kernel_group_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        _nonnegative_int("sampled_updates", self.sampled_updates, positive=True)
        for name in (
            "trace_wall_seconds",
            "gpu_active_seconds",
            "gpu_idle_seconds",
            "gpu_unattributed_seconds",
            "kernel_gap_seconds",
            "nccl_seconds",
        ):
            _finite_nonnegative(
                name, getattr(self, name), positive=name == "trace_wall_seconds"
            )
        _nonnegative_int("kernel_launches", self.kernel_launches)
        attributed = (
            self.gpu_active_seconds
            + self.gpu_idle_seconds
            + self.gpu_unattributed_seconds
        )
        if not math.isclose(attributed, self.trace_wall_seconds, rel_tol=1e-6):
            raise ValueError("GPU active/idle/unattributed time must cover trace wall time")
        if self.kernel_gap_seconds > self.trace_wall_seconds:
            raise ValueError("kernel gaps exceed trace wall time")
        for name, value in self.kernel_group_seconds.items():
            if _NAME.fullmatch(name) is None:
                raise ValueError("kernel group name is invalid")
            _finite_nonnegative(f"kernel_group_seconds.{name}", value)
        if sum(self.kernel_group_seconds.values()) > self.gpu_active_seconds * 1.000001:
            raise ValueError("kernel group time exceeds GPU active time")
        object.__setattr__(
            self,
            "kernel_group_seconds",
            MappingProxyType(dict(self.kernel_group_seconds)),
        )


@dataclass(frozen=True, slots=True)
class PytorchTracePlan:
    """Bounded trace configuration for the first updates of a measured window."""

    path: Path
    benchmark_identity_sha256: str
    sampled_updates: int
    record_shapes: bool
    profile_memory: bool
    require_cuda_activity: bool
    kernel_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict[str, tuple[str, ...]]()
    )

    def __post_init__(self) -> None:
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError("profiler trace already exists")
        if _SHA256.fullmatch(self.benchmark_identity_sha256) is None:
            raise ValueError("trace benchmark identity is invalid")
        _nonnegative_int("sampled_updates", self.sampled_updates, positive=True)
        if (
            type(self.record_shapes) is not bool
            or type(self.profile_memory) is not bool
            or type(self.require_cuda_activity) is not bool
        ):
            raise TypeError("profiler trace switches must be booleans")
        groups = dict(self.kernel_groups)
        patterns: set[str] = set()
        for name, members in groups.items():
            if _NAME.fullmatch(name) is None:
                raise ValueError("kernel group name is invalid")
            if (
                type(members) is not tuple
                or not members
                or any(type(member) is not str or not member for member in members)
                or len(set(members)) != len(members)
            ):
                raise ValueError("kernel group patterns are invalid")
            overlap = patterns.intersection(members)
            if overlap:
                raise ValueError("kernel group patterns must be globally unique")
            patterns.update(members)
        object.__setattr__(self, "kernel_groups", MappingProxyType(groups))


@dataclass(frozen=True, slots=True)
class CapturedTrace:
    entry: TraceIndexEntry
    metrics: TraceMetrics


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    samples: tuple[BenchmarkSample, ...]
    compile: CompileWindowEvidence
    trace: CapturedTrace
    data_sequence_sha256: str
    shape_distribution_sha256: str
    measured_window_seconds: float

    def __post_init__(self) -> None:
        _finite_nonnegative(
            "measured_window_seconds", self.measured_window_seconds, positive=True
        )


def _process_memory_status(pid: int) -> tuple[int, int, int]:
    values = {"VmRSS": 0, "VmHWM": 0, "VmPin": 0, "VmSwap": 0}
    try:
        with Path(f"/proc/{pid}/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in values:
                    parts = line.split()
                    if len(parts) != 3 or parts[2] != "kB":
                        raise ValueError("unexpected /proc memory unit")
                    values[key] = int(parts[1]) * 1024
    except OSError:
        raise RuntimeError("host memory accounting is unavailable") from None
    return max(values["VmRSS"], values["VmHWM"]), values["VmPin"], values["VmSwap"]


def _process_children(pid: int) -> tuple[int, ...]:
    try:
        payload = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return ()
    except OSError:
        raise RuntimeError("process-tree memory accounting is unavailable") from None
    try:
        return tuple(int(value) for value in payload.split())
    except ValueError:
        raise RuntimeError("process-tree child list is invalid") from None


class _ProcessTreeMemoryPeakSampler:
    """Track process-tree RSS high-water and sampled pinned/swap window peaks."""

    def __init__(self, *, interval_seconds: float = 0.01) -> None:
        self._root_pid = os.getpid()
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peaks = [0, 0, 0]
        self._error: RuntimeError | None = None

    def _sample(self) -> None:
        pending = [self._root_pid]
        seen: set[int] = set()
        totals = [0, 0, 0]
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                status = _process_memory_status(pid)
            except RuntimeError:
                if pid == self._root_pid:
                    raise
                continue
            for index, value in enumerate(status):
                totals[index] += value
            pending.extend(_process_children(pid))
        self._peaks = [max(old, new) for old, new in zip(self._peaks, totals, strict=True)]

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._interval_seconds):
                self._sample()
        except RuntimeError as exc:
            self._error = exc
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory sampler was already started")
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="sakuramoon-benchmark-memory",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> tuple[int, int, int]:
        thread = self._thread
        if thread is None:
            raise RuntimeError("memory sampler was not started")
        self._stop.set()
        thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("process-tree memory sampler failed") from self._error
        return self._peaks[0], self._peaks[1], self._peaks[2]


def _checkpoint_bytes(paths: tuple[Path, ...]) -> int:
    total = 0
    for path in paths:
        if path.is_symlink() or not path.exists():
            raise ValueError("checkpoint observation path is missing or a symlink")
        items: Sequence[Path] = (path,) if path.is_file() else tuple(path.rglob("*"))
        for item in items:
            if item.is_symlink():
                raise ValueError("checkpoint observation contains a symlink")
            if item.is_file():
                total += item.stat().st_size
            elif not item.is_dir():
                raise ValueError("checkpoint observation contains a special file")
    return total


class DisabledCompileCounterProbe:
    """Explicit probe for configs where regional compile is schema-disabled."""

    def snapshot(self) -> CompileCounters:
        return CompileCounters(0, 0, 0)


def _trace_span(event: object) -> tuple[float, float] | None:
    if not isinstance(event, dict):
        return None
    mapping = cast(dict[str, object], event)
    if mapping.get("ph") != "X":
        return None
    timestamp = mapping.get("ts")
    duration = mapping.get("dur")
    if (
        not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(timestamp))
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        return None
    start = float(timestamp) / 1_000_000.0
    return start, start + float(duration) / 1_000_000.0


def _interval_union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _derive_trace_metrics(
    path: Path,
    *,
    sampled_successful_updates: tuple[int, ...],
    kernel_groups: Mapping[str, tuple[str, ...]],
    require_cuda_activity: bool,
) -> TraceMetrics:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded: object = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("profiler trace is not valid JSON") from None
    if not isinstance(loaded, dict):
        raise TypeError("profiler trace omits traceEvents")
    document = cast(dict[str, object], loaded)
    raw_events = document.get("traceEvents")
    if not isinstance(raw_events, list):
        raise TypeError("profiler trace omits traceEvents")
    events = cast(list[object], raw_events)
    outer_name = (
        f"benchmark_trace_window:{sampled_successful_updates[0]}:"
        f"{sampled_successful_updates[-1]}"
    )
    outer_spans = [
        span
        for event in events
        if isinstance(event, dict)
        and cast(dict[str, object], event).get("name") == outer_name
        if (span := _trace_span(cast(object, event))) is not None
    ]
    if len(outer_spans) != 1:
        raise ValueError("profiler trace does not contain one benchmark trace window")
    window_start, window_end = outer_spans[0]
    trace_wall = window_end - window_start
    _finite_nonnegative("trace_wall_seconds", trace_wall, positive=True)
    expected_markers = {
        f"benchmark_measured_update:{index};successful_update:{successful}"
        for index, successful in enumerate(sampled_successful_updates, start=1)
    }
    observed_markers = {
        str(cast(dict[str, object], event).get("name"))
        for event in events
        if isinstance(event, dict)
        and cast(dict[str, object], event).get("name") in expected_markers
    }
    if observed_markers != expected_markers:
        raise ValueError("profiler trace omits measured successful-update markers")

    kernel_intervals: list[tuple[float, float]] = []
    active_device_intervals: list[tuple[float, float]] = []
    unknown_device_intervals: list[tuple[float, float]] = []
    grouped_intervals: dict[str, list[tuple[float, float]]] = {
        name: [] for name in kernel_groups
    }
    kernel_launches = 0
    nccl_intervals: list[tuple[float, float]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        mapping = cast(dict[str, object], event)
        category = mapping.get("cat")
        if type(category) is not str:
            continue
        category_tokens = {
            token.strip().lower() for token in category.split(",") if token.strip()
        }
        is_kernel = any("kernel" in token for token in category_tokens)
        is_governed_transfer = bool(
            category_tokens.intersection({"gpu_memcpy", "gpu_memset"})
        )
        is_unknown_device_work = (
            not is_kernel
            and not is_governed_transfer
            and any(token.startswith("gpu_") for token in category_tokens)
        )
        if not (is_kernel or is_governed_transfer or is_unknown_device_work):
            continue
        span = _trace_span(mapping)
        if span is None:
            continue
        start = max(span[0], window_start)
        end = min(span[1], window_end)
        if end <= start:
            continue
        clipped = (start, end)
        if is_unknown_device_work:
            unknown_device_intervals.append(clipped)
        else:
            active_device_intervals.append(clipped)
        if not is_kernel:
            continue
        kernel_intervals.append(clipped)
        kernel_launches += 1
        name = mapping.get("name")
        kernel_name = name if type(name) is str else ""
        if "nccl" in kernel_name.lower():
            nccl_intervals.append(clipped)
        matched_groups = [
            group
            for group, patterns in kernel_groups.items()
            if any(pattern in kernel_name for pattern in patterns)
        ]
        if len(matched_groups) > 1:
            raise ValueError("a traced kernel matches multiple declared groups")
        if matched_groups:
            grouped_intervals[matched_groups[0]].append(clipped)

    gpu_active = _interval_union_seconds(active_device_intervals)
    observed_device = _interval_union_seconds(
        (*active_device_intervals, *unknown_device_intervals)
    )
    gpu_unattributed = max(0.0, observed_device - gpu_active)
    gpu_idle = max(0.0, trace_wall - observed_device)
    ordered_kernel_intervals = sorted(kernel_intervals)
    kernel_gap = 0.0
    if ordered_kernel_intervals:
        previous_end = ordered_kernel_intervals[0][1]
        for start, end in ordered_kernel_intervals[1:]:
            if start > previous_end:
                kernel_gap += start - previous_end
            previous_end = max(previous_end, end)
    group_seconds = {
        name: _interval_union_seconds(intervals)
        for name, intervals in grouped_intervals.items()
        if intervals
    }
    if require_cuda_activity and not (
        active_device_intervals or unknown_device_intervals
    ):
        raise ValueError("CUDA profiler trace omits device activity")
    return TraceMetrics(
        sampled_updates=len(sampled_successful_updates),
        trace_wall_seconds=trace_wall,
        gpu_active_seconds=gpu_active,
        gpu_idle_seconds=gpu_idle,
        gpu_unattributed_seconds=gpu_unattributed,
        kernel_launches=kernel_launches,
        kernel_gap_seconds=kernel_gap,
        nccl_seconds=_interval_union_seconds(nccl_intervals),
        kernel_group_seconds=group_seconds,
    )


def _publish_captured_trace(
    temporary: Path,
    plan: PytorchTracePlan,
    sampled_successful_updates: tuple[int, ...],
) -> CapturedTrace:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    metrics = _derive_trace_metrics(
        temporary,
        sampled_successful_updates=sampled_successful_updates,
        kernel_groups=plan.kernel_groups,
        require_cuda_activity=plan.require_cuda_activity,
    )
    os.link(temporary, plan.path, follow_symlinks=False)
    directory_fd = os.open(plan.path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    entry = TraceIndexEntry(
        "pytorch_profiler",
        plan.path,
        stream_sha256(plan.path),
        plan.benchmark_identity_sha256,
        1,
        plan.sampled_updates,
        sampled_successful_updates[0],
        sampled_successful_updates[-1],
        None,
        None,
    )
    return CapturedTrace(entry, metrics)


@dataclass(frozen=True, slots=True)
class _PendingMeasurement:
    measured_update: int
    payload: StepPayload
    host_seconds: float
    start_event: torch.cuda.Event | None
    end_event: torch.cuda.Event | None
    checkpoint_bytes: int


def _measure_update(
    adapter: BenchmarkStepAdapter,
    update: int,
    *,
    measured_update: int,
    measured: bool,
) -> _PendingMeasurement:
    cuda = torch.cuda.is_available()
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    if cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    started = time.perf_counter_ns()
    payload = adapter.run_successful_update(update, measured=measured)
    if cuda:
        assert end_event is not None and start_event is not None
        end_event.record()
    host_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
    checkpoint_bytes = _checkpoint_bytes(payload.checkpoint_paths)
    return _PendingMeasurement(
        measured_update,
        payload,
        host_seconds,
        start_event,
        end_event,
        checkpoint_bytes,
    )


def run_benchmark(
    plan: BenchmarkPlan,
    adapter: BenchmarkStepAdapter,
    *,
    compile_probe: CompileCounterProbe,
    trace_plan: PytorchTracePlan,
    expected_data_sequence_sha256: str,
    expected_shape_distribution_sha256: str,
) -> BenchmarkRun:
    if trace_plan.sampled_updates > plan.measured_updates:
        raise ValueError("profile trace window exceeds measured benchmark updates")
    for name, value in (
        ("expected_data_sequence_sha256", expected_data_sequence_sha256),
        ("expected_shape_distribution_sha256", expected_shape_distribution_sha256),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be SHA-256")
    cuda = torch.cuda.is_available()
    if trace_plan.require_cuda_activity and not cuda:
        raise RuntimeError("CUDA trace was required but CUDA is unavailable")
    device = torch.cuda.current_device() if cuda else None
    trace_plan.path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=trace_plan.path.parent,
        prefix=f".{trace_plan.path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary_trace = Path(temporary_name)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        acc_events=True,
        record_shapes=trace_plan.record_shapes,
        profile_memory=trace_plan.profile_memory,
    )
    profiler_started = False
    trace_window: torch.profiler.record_function | None = None
    trace_capture_completed = False
    trace_exported = False
    measurement_completed = False
    data_sequence_digest = hashlib.sha256()
    shape_distribution_digest = hashlib.sha256()
    measured_window_seconds: float | None = None

    def observe(payload: StepPayload) -> None:
        data_sequence_digest.update(
            _workload_observation_line(
                payload.successful_update,
                payload.observation.sample_ids,
                field="sample_ids",
            )
        )
        shape_distribution_digest.update(
            _workload_observation_line(
                payload.successful_update,
                payload.observation.shape_keys,
                field="shape_keys",
            )
        )

    if cuda:
        torch.cuda.synchronize(device)
    before = compile_probe.snapshot()
    warmup_started = time.perf_counter_ns()
    try:
        for offset in range(1, plan.warmup_updates + 1):
            successful_update = plan.starting_successful_update + offset
            payload = adapter.run_successful_update(successful_update, measured=False)
            if payload.successful_update != successful_update:
                raise ValueError("warmup did not complete the expected successful update")
            observe(payload)
        if cuda:
            torch.cuda.synchronize(device)
        warmup_seconds = (
            time.perf_counter_ns() - warmup_started
        ) / 1_000_000_000.0
        after_warmup = compile_probe.snapshot()
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        pending: list[_PendingMeasurement] = []
        memory_sampler = _ProcessTreeMemoryPeakSampler()
        memory_sampler.start()
        first_measured_successful = (
            plan.starting_successful_update + plan.warmup_updates + 1
        )
        sampled_successful_updates = tuple(
            range(
                first_measured_successful,
                first_measured_successful + trace_plan.sampled_updates,
            )
        )
        try:
            if cuda:
                torch.cuda.synchronize(device)
            measured_window_started = time.perf_counter_ns()
            for measured_update in range(1, plan.measured_updates + 1):
                successful_update = first_measured_successful + measured_update - 1
                tracing = measured_update <= trace_plan.sampled_updates
                if measured_update == 1:
                    profiler.start()
                    profiler_started = True
                    trace_window = torch.profiler.record_function(
                        f"benchmark_trace_window:{sampled_successful_updates[0]}:"
                        f"{sampled_successful_updates[-1]}"
                    )
                    trace_window.__enter__()
                if tracing:
                    with torch.profiler.record_function(
                        f"benchmark_measured_update:{measured_update};"
                        f"successful_update:{successful_update}"
                    ):
                        measurement = _measure_update(
                            adapter,
                            successful_update,
                            measured_update=measured_update,
                            measured=True,
                        )
                    profiler.step()
                else:
                    measurement = _measure_update(
                        adapter,
                        successful_update,
                        measured_update=measured_update,
                        measured=True,
                    )
                if measurement.payload.successful_update != successful_update:
                    raise ValueError("measured workload returned the wrong successful update")
                observe(measurement.payload)
                pending.append(measurement)
                if measured_update == trace_plan.sampled_updates:
                    if cuda:
                        torch.cuda.synchronize(device)
                    assert trace_window is not None
                    trace_window.__exit__(None, None, None)
                    trace_window = None
                    profiler.stop()
                    profiler_started = False
                    trace_capture_completed = True
            if cuda:
                torch.cuda.synchronize(device)
            measured_window_seconds = (
                time.perf_counter_ns() - measured_window_started
            ) / 1_000_000_000.0
        finally:
            host_rss_bytes, pinned_ram_bytes, host_swap_bytes = memory_sampler.stop()
        if not trace_capture_completed:
            raise RuntimeError("measured profiler trace capture did not complete")
        profiler.export_chrome_trace(temporary_trace.as_posix())
        trace_exported = True
        observed_data_sequence_sha256 = data_sequence_digest.hexdigest()
        observed_shape_distribution_sha256 = shape_distribution_digest.hexdigest()
        if observed_data_sequence_sha256 != expected_data_sequence_sha256:
            raise RuntimeError("actual benchmark data sequence differs from identity")
        if observed_shape_distribution_sha256 != expected_shape_distribution_sha256:
            raise RuntimeError("actual benchmark shape distribution differs from identity")
        measurement_completed = True
    finally:
        if trace_window is not None:
            trace_window.__exit__(None, None, None)
        if profiler_started:
            profiler.stop()
        if not measurement_completed:
            temporary_trace.unlink(missing_ok=True)
    if not trace_exported:
        raise RuntimeError("measured profiler trace was not exported")
    try:
        allocated = torch.cuda.max_memory_allocated(device) if cuda else 0
        reserved = torch.cuda.max_memory_reserved(device) if cuda else 0
        samples: list[BenchmarkSample] = []
        for measurement in pending:
            start_event = measurement.start_event
            end_event = measurement.end_event
            cuda_span = (
                float(start_event.elapsed_time(end_event)) / 1000.0  # pyright: ignore[reportUnknownMemberType]
                if start_event is not None and end_event is not None
                else 0.0
            )
            step_seconds = max(measurement.host_seconds, cuda_span)
            payload = measurement.payload
            observed_phases = payload.phase_timer.collect_ready()
            if payload.phase_timer.pending_cuda_pairs:
                raise RuntimeError("CUDA phase timings remain unresolved after window sync")
            phase_seconds: dict[str, float] = dict.fromkeys(CORE_TIMING_PHASES, 0.0)
            phase_seconds.update(observed_phases)
            for phase, host_seconds in payload.host_phase_seconds.items():
                phase_seconds[phase] = max(phase_seconds.get(phase, 0.0), host_seconds)
            scheduled_checkpoint = (
                payload.successful_update % plan.checkpoint_every_updates == 0
            )
            checkpoint_seconds = phase_seconds.get("checkpoint", 0.0)
            if scheduled_checkpoint and (
                measurement.checkpoint_bytes == 0
                or checkpoint_seconds <= 0.0
            ):
                raise RuntimeError(
                    "measured checkpoint cadence produced no timed checkpoint artifact"
                )
            if not scheduled_checkpoint and (
                payload.checkpoint_paths
                or measurement.checkpoint_bytes != 0
                or checkpoint_seconds != 0.0
            ):
                raise RuntimeError(
                    "measured workload produced a checkpoint outside configured cadence"
                )
            samples.append(
                BenchmarkSample(
                    measurement.measured_update,
                    payload.successful_update,
                    step_seconds,
                    cuda_span,
                    phase_seconds,
                    payload.samples,
                    payload.image_tokens,
                    payload.text_tokens,
                    payload.dit_flops,
                    allocated,
                    reserved,
                    host_rss_bytes,
                    pinned_ram_bytes,
                    host_swap_bytes,
                    measurement.checkpoint_bytes,
                    checkpoint_seconds,
                )
            )
        after_measured = compile_probe.snapshot()
        compile_evidence = CompileWindowEvidence(
            before, after_warmup, after_measured, warmup_seconds
        )
        if (
            compile_evidence.measured_compiles
            or compile_evidence.measured_recompiles
            or compile_evidence.measured_fallbacks
        ):
            raise RuntimeError(
                "measured benchmark window compiled, recompiled, or fell back"
            )
        captured_trace = _publish_captured_trace(
            temporary_trace, trace_plan, sampled_successful_updates
        )
        return BenchmarkRun(
            tuple(samples),
            compile_evidence,
            captured_trace,
            observed_data_sequence_sha256,
            observed_shape_distribution_sha256,
            measured_window_seconds,
        )
    finally:
        temporary_trace.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    identity: BenchmarkIdentity
    plan: BenchmarkPlan
    profiler_trace_sha256: str
    measured_compile_count: int
    measured_recompile_count: int
    measured_fallback_count: int
    observed_data_sequence_sha256: str
    observed_shape_distribution_sha256: str
    sample_count: int
    raw_samples_sha256: str
    total_measured_seconds: float
    step_p50_seconds: float
    step_p95_seconds: float
    step_p99_seconds: float
    phase_share: Mapping[str, float]
    samples_per_second: float
    image_tokens_per_second: float
    text_tokens_per_second: float
    dit_flops_per_second: float
    gpu_active_share: float
    gpu_idle_share: float
    gpu_unattributed_share: float
    kernel_launches_per_update: float
    kernel_gap_share: float
    kernel_group_share: Mapping[str, float]
    nccl_share: float
    max_cuda_allocated_bytes: int
    max_cuda_reserved_bytes: int
    max_host_rss_bytes: int
    max_pinned_ram_bytes: int
    max_host_swap_bytes: int
    checkpoint_bytes: int
    checkpoint_amortized_share: float


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _samples_sha256(samples: tuple[BenchmarkSample, ...]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                sample.as_mapping(), sort_keys=True, separators=(",", ":")
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_benchmark(
    identity: BenchmarkIdentity,
    plan: BenchmarkPlan,
    run: BenchmarkRun,
) -> BenchmarkReport:
    samples = run.samples
    trace = run.trace.metrics
    if len(samples) != plan.measured_updates:
        raise ValueError("benchmark sample count differs from the plan")
    if tuple(sample.measured_update for sample in samples) != tuple(
        range(1, plan.measured_updates + 1)
    ):
        raise ValueError("measured update identities are not contiguous")
    if trace.sampled_updates > plan.measured_updates:
        raise ValueError("trace sample exceeds measured benchmark window")
    if run.data_sequence_sha256 != identity.workload.data_sequence_sha256:
        raise ValueError("benchmark data sequence differs from workload identity")
    if run.shape_distribution_sha256 != identity.workload.shape_distribution_sha256:
        raise ValueError("benchmark shape distribution differs from workload identity")
    total_seconds = run.measured_window_seconds
    phase_totals = {
        phase: sum(sample.phase_seconds.get(phase, 0.0) for sample in samples)
        for phase in sorted(TIMING_PHASES)
    }
    steps = tuple(sample.step_seconds for sample in samples)
    if max(steps) > total_seconds * 1.05 + 0.001:
        raise ValueError("an update duration exceeds the measured benchmark window")
    return BenchmarkReport(
        identity=identity,
        plan=plan,
        profiler_trace_sha256=run.trace.entry.sha256,
        measured_compile_count=run.compile.measured_compiles,
        measured_recompile_count=run.compile.measured_recompiles,
        measured_fallback_count=run.compile.measured_fallbacks,
        observed_data_sequence_sha256=run.data_sequence_sha256,
        observed_shape_distribution_sha256=run.shape_distribution_sha256,
        sample_count=len(samples),
        raw_samples_sha256=_samples_sha256(samples),
        total_measured_seconds=total_seconds,
        step_p50_seconds=float(statistics.median(steps)),
        step_p95_seconds=_nearest_rank(steps, 0.95),
        step_p99_seconds=_nearest_rank(steps, 0.99),
        phase_share=MappingProxyType(
            {phase: value / total_seconds for phase, value in phase_totals.items()}
        ),
        samples_per_second=sum(sample.samples for sample in samples) / total_seconds,
        image_tokens_per_second=sum(sample.image_tokens for sample in samples)
        / total_seconds,
        text_tokens_per_second=sum(sample.text_tokens for sample in samples)
        / total_seconds,
        dit_flops_per_second=sum(sample.dit_flops for sample in samples) / total_seconds,
        gpu_active_share=trace.gpu_active_seconds / trace.trace_wall_seconds,
        gpu_idle_share=trace.gpu_idle_seconds / trace.trace_wall_seconds,
        gpu_unattributed_share=trace.gpu_unattributed_seconds
        / trace.trace_wall_seconds,
        kernel_launches_per_update=trace.kernel_launches / trace.sampled_updates,
        kernel_gap_share=trace.kernel_gap_seconds / trace.trace_wall_seconds,
        kernel_group_share=MappingProxyType(
            {
                name: seconds / trace.trace_wall_seconds
                for name, seconds in trace.kernel_group_seconds.items()
            }
        ),
        nccl_share=trace.nccl_seconds / trace.trace_wall_seconds,
        max_cuda_allocated_bytes=max(
            sample.peak_cuda_allocated_bytes for sample in samples
        ),
        max_cuda_reserved_bytes=max(
            sample.peak_cuda_reserved_bytes for sample in samples
        ),
        max_host_rss_bytes=max(sample.host_rss_bytes for sample in samples),
        max_pinned_ram_bytes=max(sample.pinned_ram_bytes for sample in samples),
        max_host_swap_bytes=max(sample.host_swap_bytes for sample in samples),
        checkpoint_bytes=sum(sample.checkpoint_bytes for sample in samples),
        checkpoint_amortized_share=sum(
            sample.checkpoint_seconds for sample in samples
        )
        / total_seconds,
    )


def _strict_json_object(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("compile evidence contains duplicate fields")
            document[key] = value
        return document

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded: object = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("compile evidence is not valid JSON") from None
    if not isinstance(loaded, dict):
        raise TypeError("compile evidence must be a JSON object")
    return cast(dict[str, object], loaded)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: Path
    sha256: str
    kind: Literal["correctness", "ddp", "resume"]
    world_size: int = field(init=False)
    checkpoint_id: str = field(init=False)
    resolved_config_sha256: str = field(init=False)
    source_commit: str = field(init=False)
    build_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.sha256) is None or stream_sha256(self.path) != self.sha256:
            raise ValueError("evidence artifact content differs from its SHA-256")
        document = _strict_json_object(self.path)
        expected_keys = {
            "build_sha256",
            "checkpoint_id",
            "kind",
            "resolved_config_sha256",
            "schema_version",
            "source_commit",
            "status",
            "world_size",
        }
        if set(document) != expected_keys:
            raise ValueError("compile evidence has unknown or missing fields")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != 1
            or type(document["kind"]) is not str
            or document["kind"] != self.kind
            or type(document["status"]) is not str
            or document["status"] != "passed"
        ):
            raise ValueError("compile evidence is not a passing artifact of the expected kind")
        world_size = document["world_size"]
        checkpoint_id = document["checkpoint_id"]
        resolved_config = document["resolved_config_sha256"]
        source_commit = document["source_commit"]
        build_sha256 = document["build_sha256"]
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise TypeError("evidence world size is invalid")
        _nonnegative_int("artifact world_size", world_size, positive=True)
        if type(checkpoint_id) is not str or not checkpoint_id:
            raise ValueError("evidence checkpoint identity is empty")
        if type(resolved_config) is not str or _SHA256.fullmatch(resolved_config) is None:
            raise ValueError("evidence resolved config identity is invalid")
        if type(source_commit) is not str or _COMMIT.fullmatch(source_commit) is None:
            raise ValueError("evidence source commit identity is invalid")
        if type(build_sha256) is not str or _SHA256.fullmatch(build_sha256) is None:
            raise ValueError("evidence build identity is invalid")
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "resolved_config_sha256", resolved_config)
        object.__setattr__(self, "source_commit", source_commit)
        object.__setattr__(self, "build_sha256", build_sha256)


@dataclass(frozen=True, slots=True)
class RegionalCompileEvidence:
    correctness: ArtifactReference | None
    ddp: ArtifactReference | None
    resume: ArtifactReference | None

    def complete_for(self, identity: BenchmarkIdentity) -> bool:
        variant = identity.variant
        workload = identity.workload
        references = (self.correctness, self.ddp, self.resume)
        return (
            all(reference is not None for reference in references)
            and all(
                reference is not None
                and reference.checkpoint_id == workload.checkpoint_id
                and reference.resolved_config_sha256
                == variant.resolved_config_sha256
                and reference.source_commit == variant.source_commit
                and reference.build_sha256 == variant.build_sha256
                for reference in references
            )
            and self.correctness is not None
            and self.ddp is not None
            and self.resume is not None
            and self.correctness.kind == "correctness"
            and self.ddp.kind == "ddp"
            and self.resume.kind == "resume"
            and workload.world_size == 4
            and self.correctness.world_size == 4
            and self.ddp.world_size == 4
            and self.resume.world_size == 4
        )


@dataclass(frozen=True, slots=True)
class ResourceIncreaseDisclosure:
    max_extra_bytes: int
    rationale: str

    def __post_init__(self) -> None:
        _nonnegative_int("max_extra_bytes", self.max_extra_bytes)
        if self.max_extra_bytes > 0 and not self.rationale.strip():
            raise ValueError("a memory increase requires a retention rationale")


@dataclass(frozen=True, slots=True)
class ComparisonPolicy:
    minimum_end_to_end_gain_percent: float
    max_p95_regression_percent: float
    max_p99_regression_percent: float
    cuda_allocated: ResourceIncreaseDisclosure
    cuda_reserved: ResourceIncreaseDisclosure
    host_rss: ResourceIncreaseDisclosure
    pinned_ram: ResourceIncreaseDisclosure
    host_swap: ResourceIncreaseDisclosure

    def __post_init__(self) -> None:
        _finite_nonnegative(
            "minimum_end_to_end_gain_percent",
            self.minimum_end_to_end_gain_percent,
            positive=True,
        )
        if self.minimum_end_to_end_gain_percent < 3.0:
            raise ValueError("regional compile gain gate must be at least 3 percent")
        for name in ("max_p95_regression_percent", "max_p99_regression_percent"):
            _finite_nonnegative(name, getattr(self, name))
        if self.host_swap.max_extra_bytes != 0:
            raise ValueError("benchmark comparison policy must require zero host swap")


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    throughput_gain_percent: float
    p95_regression_percent: float
    p99_regression_percent: float
    extra_cuda_allocated_bytes: int
    extra_cuda_reserved_bytes: int
    extra_host_rss_bytes: int
    extra_pinned_ram_bytes: int
    extra_host_swap_bytes: int
    regional_compile_allowed: bool


def _percent_change(after: float, baseline: float) -> float:
    return (after / baseline - 1.0) * 100.0


def compare_benchmarks(
    baseline: BenchmarkReport,
    after: BenchmarkReport,
    *,
    policy: ComparisonPolicy,
    regional_compile: RegionalCompileEvidence,
) -> BenchmarkComparison:
    if baseline.identity.workload != after.identity.workload or baseline.plan != after.plan:
        raise ValueError("before/after benchmark workload identities or plans differ")
    before_variant = baseline.identity.variant
    after_variant = after.identity.variant
    if before_variant.changed_config_keys != after_variant.changed_config_keys:
        raise ValueError("before/after variants disclose different config changes")
    if not set(before_variant.enabled_features).issubset(after_variant.enabled_features):
        raise ValueError("after variant removes an enabled feature")
    if (
        before_variant.backend != after_variant.backend
        and "kernels.attention_backend" not in after_variant.changed_config_keys
    ):
        raise ValueError("backend drift is not an explicitly disclosed variant")
    compile_transition = (
        "regional_compile" not in before_variant.enabled_features
        and "regional_compile" in after_variant.enabled_features
    )
    expected_compile_features = set(before_variant.enabled_features) | {
        "regional_compile"
    }
    if compile_transition and (
        after_variant.changed_config_keys != ("compile.regional_enabled",)
        or before_variant.backend != after_variant.backend
        or set(after_variant.enabled_features) != expected_compile_features
    ):
        raise ValueError(
            "regional compile comparison must isolate the compile feature and config key"
        )
    if baseline.max_host_swap_bytes != 0 or after.max_host_swap_bytes != 0:
        raise ValueError("benchmark comparison requires zero host swap")
    resources = (
        (
            after.max_cuda_allocated_bytes - baseline.max_cuda_allocated_bytes,
            policy.cuda_allocated,
        ),
        (
            after.max_cuda_reserved_bytes - baseline.max_cuda_reserved_bytes,
            policy.cuda_reserved,
        ),
        (after.max_host_rss_bytes - baseline.max_host_rss_bytes, policy.host_rss),
        (
            after.max_pinned_ram_bytes - baseline.max_pinned_ram_bytes,
            policy.pinned_ram,
        ),
        (
            after.max_host_swap_bytes - baseline.max_host_swap_bytes,
            policy.host_swap,
        ),
    )
    if any(extra > disclosure.max_extra_bytes for extra, disclosure in resources):
        raise ValueError("after benchmark uses an undisclosed memory increase")
    gain = _percent_change(after.samples_per_second, baseline.samples_per_second)
    p95 = _percent_change(after.step_p95_seconds, baseline.step_p95_seconds)
    p99 = _percent_change(after.step_p99_seconds, baseline.step_p99_seconds)
    compile_variant = compile_transition
    return BenchmarkComparison(
        throughput_gain_percent=gain,
        p95_regression_percent=p95,
        p99_regression_percent=p99,
        extra_cuda_allocated_bytes=resources[0][0],
        extra_cuda_reserved_bytes=resources[1][0],
        extra_host_rss_bytes=resources[2][0],
        extra_pinned_ram_bytes=resources[3][0],
        extra_host_swap_bytes=resources[4][0],
        regional_compile_allowed=(
            compile_variant
            and after.measured_compile_count == 0
            and after.measured_recompile_count == 0
            and after.measured_fallback_count == 0
            and gain >= policy.minimum_end_to_end_gain_percent
            and p95 <= policy.max_p95_regression_percent
            and p99 <= policy.max_p99_regression_percent
            and regional_compile.complete_for(after.identity)
        ),
    )


@dataclass(frozen=True, slots=True)
class KernelGroupHotspot:
    name: str
    kernels: tuple[str, ...]
    cumulative_share: float

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.name) is None:
            raise ValueError("kernel hotspot name is invalid")
        if (
            not self.kernels
            or any(not kernel.strip() for kernel in self.kernels)
            or len(set(self.kernels)) != len(self.kernels)
        ):
            raise ValueError("kernel hotspot members are invalid")
        _finite_nonnegative("cumulative_share", self.cumulative_share)
        if self.cumulative_share > 1.0:
            raise ValueError("kernel hotspot share exceeds one")


@dataclass(frozen=True, slots=True)
class HotspotDecisions:
    optimized_phases: tuple[str, ...]
    phase_retention_rationales: Mapping[str, str]
    kernel_groups: tuple[KernelGroupHotspot, ...]
    optimized_kernel_groups: tuple[str, ...]
    kernel_retention_rationales: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(set(self.optimized_phases)) != len(self.optimized_phases):
            raise ValueError("optimized phase decisions are duplicated")
        groups = tuple(group.name for group in self.kernel_groups)
        if len(set(groups)) != len(groups):
            raise ValueError("kernel hotspot groups are duplicated")
        unknown_phases = (
            set(self.optimized_phases) | set(self.phase_retention_rationales)
        ) - TIMING_PHASES
        if unknown_phases:
            raise ValueError("hotspot decision names an unknown phase")
        unknown_groups = (
            set(self.optimized_kernel_groups) | set(self.kernel_retention_rationales)
        ) - set(groups)
        if unknown_groups:
            raise ValueError("hotspot decision names an unknown kernel group")
        object.__setattr__(
            self,
            "phase_retention_rationales",
            MappingProxyType(dict(self.phase_retention_rationales)),
        )
        object.__setattr__(
            self,
            "kernel_retention_rationales",
            MappingProxyType(dict(self.kernel_retention_rationales)),
        )


def validate_hotspot_decisions(
    report: BenchmarkReport,
    decisions: HotspotDecisions,
) -> None:
    decision_groups = {group.name: group for group in decisions.kernel_groups}
    if set(decision_groups) != set(report.kernel_group_share):
        raise ValueError("kernel hotspot decisions differ from measured trace groups")
    for name, share in report.kernel_group_share.items():
        if not math.isclose(
            decision_groups[name].cumulative_share, share, rel_tol=1e-9
        ):
            raise ValueError("kernel hotspot share differs from measured trace")
    for phase, share in report.phase_share.items():
        if share > 0.05 and phase not in decisions.optimized_phases:
            rationale = decisions.phase_retention_rationales.get(phase)
            if rationale is None or not rationale.strip():
                raise ValueError(f"hot phase lacks optimization or rationale: {phase}")
    for group in decisions.kernel_groups:
        if group.cumulative_share > 0.05 and group.name not in decisions.optimized_kernel_groups:
            rationale = decisions.kernel_retention_rationales.get(group.name)
            if rationale is None or not rationale.strip():
                raise ValueError(
                    f"hot kernel group lacks optimization or rationale: {group.name}"
                )


@dataclass(frozen=True, slots=True)
class TraceIndexEntry:
    tool: Literal["pytorch_profiler", "nsys", "ncu"]
    path: Path
    sha256: str
    benchmark_identity_sha256: str
    first_measured_update: int
    last_measured_update: int
    first_successful_update: int
    last_successful_update: int
    hotspot_name: str | None
    hotspot_rationale: str | None

    def __post_init__(self) -> None:
        if self.tool not in ("pytorch_profiler", "nsys", "ncu"):
            raise ValueError("trace tool is invalid")
        if _SHA256.fullmatch(self.sha256) is None or stream_sha256(self.path) != self.sha256:
            raise ValueError("trace content differs from its SHA-256")
        if _SHA256.fullmatch(self.benchmark_identity_sha256) is None:
            raise ValueError("trace benchmark identity is invalid")
        if (
            type(self.first_measured_update) is not int
            or type(self.last_measured_update) is not int
            or self.first_measured_update <= 0
            or self.last_measured_update < self.first_measured_update
        ):
            raise ValueError("trace update range is invalid")
        if (
            type(self.first_successful_update) is not int
            or type(self.last_successful_update) is not int
            or self.first_successful_update <= 0
            or self.last_successful_update < self.first_successful_update
            or self.last_successful_update - self.first_successful_update
            != self.last_measured_update - self.first_measured_update
        ):
            raise ValueError("trace successful-update range is invalid")
        if self.tool == "ncu" and (
            self.hotspot_name is None
            or self.hotspot_rationale is None
            or not self.hotspot_rationale.strip()
        ):
            raise ValueError("Nsight Compute requires a proven-hotspot reference")


@dataclass(frozen=True, slots=True)
class ExternalTraceSmoke:
    tool: Literal["nsys", "ncu"]
    path: Path
    sha256: str


def validate_trace_index(
    entries: tuple[TraceIndexEntry, ...],
    *,
    plan: BenchmarkPlan,
    profile_trace_updates: int,
    benchmark_identity_sha256: str,
    hotspots: HotspotDecisions,
) -> None:
    _nonnegative_int("profile_trace_updates", profile_trace_updates, positive=True)
    if profile_trace_updates > plan.measured_updates:
        raise ValueError("profile trace window exceeds measured benchmark updates")
    del hotspots
    for entry in entries:
        if entry.tool in {"nsys", "ncu"}:
            raise ValueError(
                "formal external trace indexing requires a verified marker importer"
            )
        if entry.benchmark_identity_sha256 != benchmark_identity_sha256:
            raise ValueError("trace index identity differs from benchmark")
        if entry.last_measured_update > plan.measured_updates:
            raise ValueError("trace range exceeds measured benchmark window")
        expected_first_successful = (
            plan.starting_successful_update
            + plan.warmup_updates
            + entry.first_measured_update
        )
        expected_last_successful = (
            plan.starting_successful_update
            + plan.warmup_updates
            + entry.last_measured_update
        )
        if (
            entry.first_successful_update != expected_first_successful
            or entry.last_successful_update != expected_last_successful
        ):
            raise ValueError("trace successful-update range differs from benchmark")
        count = entry.last_measured_update - entry.first_measured_update + 1
        if entry.tool in {"pytorch_profiler", "nsys"} and count != profile_trace_updates:
            raise ValueError("profiler trace count differs from runtime config")


def capture_external_trace_smoke(
    tool: Literal["nsys", "ncu"],
    command: tuple[str, ...],
    *,
    output: Path,
) -> ExternalTraceSmoke:
    """Exercise an Nsight collector without creating formal benchmark evidence."""

    if not command or Path(command[0]).name != tool:
        raise ValueError("external trace command does not match the declared tool")
    expected_suffix = ".nsys-rep" if tool == "nsys" else ".ncu-rep"
    if not output.name.endswith(expected_suffix):
        raise ValueError(f"{tool} trace output must end with {expected_suffix}")
    if output.exists() or output.is_symlink():
        raise FileExistsError("external trace output already exists")
    if sum(argument.count("{output}") for argument in command) != 1:
        raise ValueError("external trace command requires one {output} placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    generated = Path(f"{temporary}{expected_suffix}")
    try:
        resolved_command = tuple(
            argument.replace("{output}", temporary.as_posix()) for argument in command
        )
        completed = subprocess.run(resolved_command, check=False, shell=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{tool} trace collection failed")
        if not generated.is_file() or generated.is_symlink():
            raise RuntimeError(f"{tool} did not publish the requested trace")
        with generated.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(generated, output, follow_symlinks=False)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
        generated.unlink(missing_ok=True)
    return ExternalTraceSmoke(tool, output, stream_sha256(output))


__all__ = [
    "ArtifactReference",
    "BenchmarkComparison",
    "BenchmarkIdentity",
    "BenchmarkObservation",
    "BenchmarkPlan",
    "BenchmarkReport",
    "BenchmarkRun",
    "BenchmarkSample",
    "BenchmarkStepAdapter",
    "BenchmarkVariant",
    "BenchmarkWorkloadIdentity",
    "CapturedTrace",
    "ComparisonPolicy",
    "CompileCounterProbe",
    "CompileCounters",
    "CompileWindowEvidence",
    "DisabledCompileCounterProbe",
    "ExternalTraceSmoke",
    "HotspotDecisions",
    "KernelGroupHotspot",
    "PytorchTracePlan",
    "RegionalCompileEvidence",
    "ResourceIncreaseDisclosure",
    "StepPayload",
    "TraceIndexEntry",
    "TraceMetrics",
    "canonical_workload_artifact_bytes",
    "capture_external_trace_smoke",
    "compare_benchmarks",
    "run_benchmark",
    "stream_sha256",
    "summarize_benchmark",
    "validate_hotspot_decisions",
    "validate_trace_index",
]
