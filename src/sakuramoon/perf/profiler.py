"""Optional torch.profiler snapshot helpers (P0, display-only heuristics).

The profiler is STRICTLY opt-in and runs as a SEPARATE scratch stage
AFTER the canonical measured baseline (never inside it).  When supported
by the stack it captures exactly ``profiler_updates`` logical updates
(default 1, canonical P0) and produces a normalized top-N operator
summary.  Operator categories are heuristic string matches for DISPLAY
ONLY — the raw operator names are always kept and the heuristics are
never used as correctness facts.

Device-trace availability is decided by a tiny matmul probe that must
yield at least one real CUDA device event with positive device time.
Importing ``torch.profiler`` alone never counts as "available".  When
the probe reports unavailable, NO extra workload is executed (a prior
ungated full-workload capture was observed to hang the DTK runtime).
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

DEFAULT_PROFILER_UPDATES = 1


def validate_profiler_updates(value: object) -> int:
    """``profiler_updates`` must be a positive int (0 and negatives rejected)."""

    if type(value) is not int:
        raise ValueError("profiler_updates must be an int")
    if value <= 0:
        raise ValueError("profiler_updates must be a positive int (got 0)")
    return value


def profiler_supported() -> bool:
    """Whether torch.profiler is importable on this stack."""

    return _torch_profiler is not None


def _torch_device_type_enum() -> Any:
    try:
        import torch

        device_type = getattr(torch.autograd, "DeviceType", None)
        if device_type is None:
            return None
        cuda = getattr(device_type, "CUDA", None)
        return cuda if cuda is not None else device_type
    except Exception:  # noqa: BLE001 - detection must degrade, never raise
        return None


def is_cuda_device_event(event: Any) -> bool:
    """Robust device-type classification for profiler events.

    Prefers direct enum comparison against ``torch.autograd.DeviceType.CUDA``
    when the build exposes it; falls back to parsing string representations
    (``CUDA``, ``DeviceType.CUDA``, ``cuda``).  CPU events (enum or
    string) are NEVER classified as CUDA, and an absent ``device_type``
    attribute is not CUDA.
    """

    device_type = getattr(event, "device_type", None)
    if device_type is None:
        return False
    cuda = _torch_device_type_enum()
    if cuda is not None and device_type == cuda:
        return True
    text = str(device_type).strip()
    tail = text.rsplit(".", 1)[-1]
    return tail.upper() == "CUDA"


def profiler_device_trace_available() -> bool:
    """Cheap probe: does a tiny CUDA profile yield a real device event?

    Requires at least one CUDA device event with POSITIVE device time.
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
        cuda_events = [event for event in events if is_cuda_device_event(event)]
        return any(
            _event_time(event, "self_device_time_total", "self_device_time") > 0.0
            for event in cuda_events
        )
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
    profiler_updates: int = DEFAULT_PROFILER_UPDATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "device_trace_available": self.device_trace_available,
            "profiler_updates": self.profiler_updates,
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


def unavailable_snapshot(
    profiler_updates: int = DEFAULT_PROFILER_UPDATES,
) -> ProfilerSnapshot:
    """The honest UNAVAILABLE record (no workload was executed)."""

    return ProfilerSnapshot(
        top_by_self_device_time=(),
        top_by_total_device_time=(),
        trace_path=None,
        device_trace_available=False,
        profiler_updates=profiler_updates,
    )


def capture_profiler_window(
    workload: Callable[[], None],
    *,
    output_root: Path,
    rank: int,
    top_n: int = 20,
    export_trace: bool = False,
    profiler_updates: int = DEFAULT_PROFILER_UPDATES,
) -> ProfilerSnapshot:
    """Run a SEPARATE scratch workload under torch.profiler and summarize it.

    The caller passes the scratch profiler stage (exactly
    ``profiler_updates`` logical updates, post-baseline).  The canonical
    measured baseline never runs under a profiler context.
    """

    if _torch_profiler is None:
        raise RuntimeError("torch.profiler is not supported on this stack")
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n must be a positive int")
    validate_profiler_updates(profiler_updates)

    if not profiler_device_trace_available():
        # The stack's activity tracer produced no real device event in the
        # probe.  A prior ungated full-workload capture in this state was
        # observed to hang the DTK runtime (CPU spin, device idle), so the
        # P0 contract emits the UNAVAILABLE record and executes NO extra
        # workload at all.
        return unavailable_snapshot(profiler_updates)

    with _torch_profiler.profile(
        activities=[
            _torch_profiler.ProfilerActivity.CPU,
            _torch_profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        workload()
        import torch

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
        profiler_updates=profiler_updates,
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
            "rows were invented and no extra workload was executed for the "
            f"profiler stage ({snapshot.profiler_updates} update(s) requested).\n"
        )
    lines = [
        "# Profiler summary",
        "",
        (
            "Mode: SEPARATE scratch stage after the measured baseline "
            f"({snapshot.profiler_updates} logical update(s), default 1)."
        ),
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
    "DEFAULT_PROFILER_UPDATES",
    "ProfilerOperatorRow",
    "ProfilerSnapshot",
    "capture_profiler_window",
    "display_category",
    "is_cuda_device_event",
    "profiler_device_trace_available",
    "profiler_supported",
    "render_snapshot_markdown",
    "unavailable_snapshot",
    "validate_profiler_updates",
    "write_snapshot_json",
]
