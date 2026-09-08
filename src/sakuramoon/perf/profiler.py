"""Optional torch.profiler snapshot helpers (P0, display-only heuristics).

The profiler is STRICTLY opt-in.  When supported by the DTK/PyTorch
stack it captures a small representative window after warmup and produces
a normalized top-N operator summary.  Operator categories are heuristic
string matches for DISPLAY ONLY — the raw operator names are always kept
and the heuristics are never used as correctness facts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    from torch import profiler as _torch_profiler
except ImportError:  # torch.profiler is optional at import time
    _torch_profiler = None  # type: ignore[assignment]


def profiler_supported() -> bool:
    """Whether torch.profiler is importable on this stack."""

    return _torch_profiler is not None


def profiler_device_trace_available() -> bool:
    """Cheap probe: does a tiny CUDA profile actually yield device events?

    Returns False (UNAVAILABLE) on any exception instead of guessing.
    """

    if _torch_profiler is None:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        left = torch.randn(64, 64, device="cuda")
        right = torch.randn(64, 64, device="cuda")
        with _torch_profiler.profile(
            activities=[_torch_profiler.ProfilerActivity.CUDA]
        ) as prof:
            _ = left @ right
            torch.cuda.synchronize()
        events: list[Any] = cast(list[Any], prof.events())
        if not events:
            return False
        device_events = [
            event
            for event in events
            if str(getattr(event, "device_type", "")).lower() == "cuda"
        ]
        return len(device_events) > 0
    except Exception:  # noqa: BLE001 - the probe must degrade to UNAVAILABLE on any failure
        return False


_CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("flash_attention", re.compile(r"flash|varlen|sdpa|attention", re.IGNORECASE)),
    (
        "allreduce_distributed",
        re.compile(
            r"allreduce|all_reduce|reduce_scatter|broadcast|nccl|ccl",
            re.IGNORECASE,
        ),
    ),
    (
        "memcpy_h2d",
        re.compile(r"memcpy|h2d|d2h|copy_?kernel|batchedcopy", re.IGNORECASE),
    ),
    (
        "gemm_linear",
        re.compile(r"gemm|gemv|linear|matmul|\bmm\b|bmm|sgemm|hgemm", re.IGNORECASE),
    ),
    (
        "rmsnorm_norm",
        re.compile(r"rmsnorm|layer_?norm|group_?norm|norm", re.IGNORECASE),
    ),
    ("rope_trig", re.compile(r"rope|rotary|trig|sincos", re.IGNORECASE)),
    (
        "swiglu_activation",
        re.compile(r"silu|swiglu|gelu|sigmoid|relu|gelutanh", re.IGNORECASE),
    ),
    ("optimizer", re.compile(r"adamw|cmuon|foreach|_fused|optimizer", re.IGNORECASE)),
    (
        "elementwise",
        re.compile(
            r"elementwise|add|mul|sub|div|cat|slice|select|index|fill|zero|clone|triu|tril",
            re.IGNORECASE,
        ),
    ),
)


def display_category(operator_name: str) -> str:
    """Heuristic display category (NOT a correctness fact)."""

    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(operator_name):
            return category
    return "other"


@dataclass(frozen=True, slots=True)
class ProfilerOperatorRow:
    """One normalized operator row from key_averages()."""

    name: str
    display_category: str
    count: int
    self_device_time_us: float
    total_device_time_us: float
    cpu_time_total_us: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_category": self.display_category,
            "count": self.count,
            "self_device_time_us": self.self_device_time_us,
            "total_device_time_us": self.total_device_time_us,
            "cpu_time_total_us": self.cpu_time_total_us,
        }


@dataclass(frozen=True, slots=True)
class ProfilerSnapshot:
    """The normalized top-N operator summary of one captured window."""

    top_by_self_device_time: tuple[ProfilerOperatorRow, ...]
    top_by_total_device_time: tuple[ProfilerOperatorRow, ...]
    trace_path: str | None
    device_trace_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "device_trace_available": self.device_trace_available,
            "trace_path": self.trace_path,
            "top_by_self_device_time": [
                row.to_dict() for row in self.top_by_self_device_time
            ],
            "top_by_total_device_time": [
                row.to_dict() for row in self.top_by_total_device_time
            ],
        }


def _event_time(event: Any, *attributes: str) -> float:
    for attribute in attributes:
        value = getattr(event, attribute, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def capture_profiler_window(
    workload: Callable[[], None],
    *,
    output_root: Path,
    rank: int,
    top_n: int = 20,
    export_trace: bool = False,
) -> ProfilerSnapshot:
    """Run ``workload`` under torch.profiler and summarize device operators.

    The window is whatever the caller passes (the benchmark harness passes
    the measured-update stage only, after warmup).
    """

    if _torch_profiler is None:
        raise RuntimeError("torch.profiler is not supported on this stack")
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n must be a positive int")

    import torch

    if not profiler_device_trace_available():
        # The stack's activity tracer produced no device events in the probe.
        # Capturing the full workload in that state has been observed to
        # hang the DTK runtime (CPU spin, device idle), so the P0 contract
        # degrades to UNAVAILABLE and runs the workload WITHOUT a profiler
        # context instead of risking the machine.
        workload()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return ProfilerSnapshot(
            top_by_self_device_time=(),
            top_by_total_device_time=(),
            trace_path=None,
            device_trace_available=False,
        )

    with _torch_profiler.profile(
        activities=[
            _torch_profiler.ProfilerActivity.CPU,
            _torch_profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        workload()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    rows: list[ProfilerOperatorRow] = []
    for event in cast(list[Any], prof.key_averages()):
        self_device = _event_time(event, "self_device_time_total", "self_device_time")
        total_device = _event_time(event, "device_time_total", "device_time")
        if self_device <= 0.0 and total_device <= 0.0:
            continue
        name = str(getattr(event, "key", "") or "").strip()
        if not name or name.startswith("unnamed"):
            continue
        rows.append(
            ProfilerOperatorRow(
                name=name,
                display_category=display_category(name),
                count=int(getattr(event, "count", 0) or 0),
                self_device_time_us=self_device,
                total_device_time_us=total_device,
                cpu_time_total_us=_event_time(event, "cpu_time_total"),
            )
        )

    device_trace_available = any(row.self_device_time_us > 0.0 for row in rows)
    by_self = tuple(
        sorted(rows, key=lambda row: row.self_device_time_us, reverse=True)[:top_n]
    )
    by_total = tuple(
        sorted(rows, key=lambda row: row.total_device_time_us, reverse=True)[:top_n]
    )

    trace_path: str | None = None
    if export_trace:
        output_root.mkdir(parents=True, exist_ok=True)
        trace_file = output_root / f"profiler-trace-rank{rank}.json"
        prof.export_chrome_trace(str(trace_file))
        trace_path = str(trace_file)

    return ProfilerSnapshot(
        top_by_self_device_time=by_self,
        top_by_total_device_time=by_total,
        trace_path=trace_path,
        device_trace_available=device_trace_available,
    )


def write_snapshot_json(snapshot: ProfilerSnapshot, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = snapshot.to_dict()
    destination.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def render_snapshot_markdown(snapshot: ProfilerSnapshot, limit: int = 20) -> str:
    """A compact Markdown table for the top operators (display categories)."""

    if not snapshot.device_trace_available:
        return (
            "# Profiler summary\n\n"
            "PROFILER_DEVICE_TRACE = UNAVAILABLE on this stack; no kernel-level "
            "rows were invented.\n"
        )
    lines = [
        "# Profiler summary",
        "",
        f"Device trace available: {'yes' if snapshot.device_trace_available else 'no'}",
        f"Trace file: {snapshot.trace_path or '(not exported)'}",
        "",
        "Top operators by self device time (display categories are heuristic only):",
        "",
        "| # | operator | category (display) | count | self device (ms) | total device (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(snapshot.top_by_self_device_time[:limit], start=1):
        lines.append(
            f"| {index} | `{row.name}` | {row.display_category} | {row.count} "
            f"| {row.self_device_time_us / 1000.0:.3f} | {row.total_device_time_us / 1000.0:.3f} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ProfilerOperatorRow",
    "ProfilerSnapshot",
    "capture_profiler_window",
    "display_category",
    "profiler_device_trace_available",
    "profiler_supported",
    "render_snapshot_markdown",
    "write_snapshot_json",
]
