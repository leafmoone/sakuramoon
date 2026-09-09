"""Compile / persistent-cache observability (P1-R1B).

Benchmark-only module.  NO import from this module may be added to normal
production training paths: it exists to MEASURE torch.compile startup /
restart / steady-state behavior and persistent-cache benefit without
changing training math or the steady-state production shape.

Concretely it provides:

* :class:`CompileEnvironment` — an auditable runtime identity for one
  compile/cache run (torch / HIP / DTK / HCU / Triton / backend / mode /
  dynamic / shape / cache root / cache env / git SHA), so cache-reuse
  evidence is never attributed to the wrong runtime.
* :class:`CompileCacheSnapshot` — a filesystem BEFORE/AFTER snapshot of a
  cache root (existence, file count, total bytes, per-top-level-subdir
  bytes).  Filesystem evidence is SUPPORTING only: it is never treated as
  proof of a cache hit.
* :class:`CompileStartupSample` — the OUTSIDE-the-measured-window startup
  timeline (process start -> assembly -> model / encoder / DDP / compile
  -> warmup -> measured window), including each warmup update's wall time
  and time-to-first-successful-update / time-to-measured-window.
* :class:`CompileDiagnosticSummary` — classified graph-break and recompile
  counts (expected vs unexpected, warmup-window vs measured-window).
* Pure, torch-free functions: cache-identity sanitization, cache-env
  construction, cache-benefit classification, steady-state regression
  gate, environment-bracket drift, graph-break / recompile classification,
  and a tolerant compiler-log diagnostic parser.

The steady-state schema-2 logical-update timing (``PerformanceSample``) is
untouched by this module; startup is a separate additive concern recorded
outside the measured window.  Stage boundaries that cannot be isolated
honestly are recorded ABSENT (never a fake 0.0).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Cache identity / environment
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def sanitize_cache_identity(*parts: str | float | None) -> str:
    """Build a stable, filesystem-safe cache identity from runtime facts.

    Each part is lower-cased and every run of non-alphanumeric characters
    collapses to a single ``-``.  Empty parts are skipped.  The result is a
    single path-safe token (no path separators, no leading/trailing
    hyphens).  An all-empty input yields ``"unknown"``.

    GO P1-R1B: the identity is derived from torch version, HIP runtime, DTK
    identity, HCU model, compile backend, compile mode, and the dynamic
    flag.  The git SHA is deliberately NOT part of the top-level directory
    (PyTorch keys compiled code by content), but every report records it.
    """

    cleaned_parts: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip().lower()
        if text:
            cleaned_parts.append(text)
    joined = "-".join(cleaned_parts)
    collapsed = _NON_ALNUM.sub("-", joined).strip("-")
    collapsed = re.sub(r"-{2,}", "-", collapsed)
    return collapsed or "unknown"


def build_cache_env(
    cache_root: str | Path,
    *,
    fx_graph_cache: bool,
    aot_autograd_cache: bool = False,
) -> dict[str, str]:
    """Construct the persistent-cache environment for a benchmark subprocess.

    The primary mechanism is the local on-disk Inductor cache rooted at
    ``cache_root`` via ``TORCHINDUCTOR_CACHE_DIR``.  Triton is left to
    Inductor (it places its subcache under ``<root>/triton/<device>``);
    ``TRITON_CACHE_DIR`` is NOT set here.

    ``aot_autograd_cache`` is rejected fail-closed unless
    ``fx_graph_cache`` is also enabled: the AOTAutograd cache depends on the
    FX graph cache, so enabling the former without the latter is a
    configuration error, not a capability to emulate.

    No remote/Redis cache, no unsafe guard skipping, no compiler-error
    suppression, no force-eager, and no cross-runtime reuse forcing is set.
    """

    if aot_autograd_cache and not fx_graph_cache:
        raise ValueError(
            "aot_autograd_cache requires fx_graph_cache: the AOTAutograd "
            "cache depends on the FX graph cache (refusing to set "
            "TORCHINDUCTOR_AUTOGRAD_CACHE without TORCHINDUCTOR_FX_GRAPH_CACHE)"
        )
    env: dict[str, str] = {"TORCHINDUCTOR_CACHE_DIR": str(cache_root)}
    if fx_graph_cache:
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    if aot_autograd_cache:
        env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    return env


@dataclass(frozen=True, slots=True)
class CompileEnvironment:
    """Auditable runtime identity for one compile/cache benchmark run.

    Every field is a fact recorded from the actual runtime (never guessed).
    ``None`` is allowed for optional software versions that cannot be
    discovered honestly (rendered as JSON null).
    """

    torch_version: str
    hip_version: str | None
    dtk_identity: str
    hcu_model: str
    triton_version: str | None
    compile_backend: str
    compile_mode: str
    compile_dynamic: bool
    world_size: int
    resolution: int
    local_batch: int
    accumulation: int
    cache_root: str
    git_sha: str
    cache_env: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
            "dtk_identity": self.dtk_identity,
            "hcu_model": self.hcu_model,
            "triton_version": self.triton_version,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "compile_dynamic": self.compile_dynamic,
            "world_size": self.world_size,
            "resolution": self.resolution,
            "local_batch": self.local_batch,
            "accumulation": self.accumulation,
            "cache_root": self.cache_root,
            "git_sha": self.git_sha,
            "cache_env": {
                key: value
                for key, value in (_split_env_item(item) for item in self.cache_env)
            },
        }


def _split_env_item(item: str) -> tuple[str, str]:
    key, sep, value = item.partition("=")
    if not sep:
        return item, ""
    return key, value


# ---------------------------------------------------------------------------
# Cache snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompileCacheSnapshot:
    """A point-in-time filesystem snapshot of one cache root.

    Filesystem evidence is supporting evidence only — a file existing does
    not by itself prove a cache hit.  ``subdir_bytes`` breaks the total
    down by the FIRST path component under the root (``fxgraph``,
    ``aotautograd``, ``triton``, ``autotune``, ``cache``, ``locks``, ...),
    with files directly under the root grouped under ``"(root)"``.
    """

    root: str
    exists: bool
    file_count: int
    total_bytes: int
    subdir_bytes: dict[str, int]
    captured_at: str

    @classmethod
    def capture(
        cls, root: str | Path, *, captured_at: str | None = None
    ) -> CompileCacheSnapshot:
        root_path = Path(root)
        exists = root_path.is_dir()
        file_count = 0
        total_bytes = 0
        subdir_bytes: dict[str, int] = {}
        if exists:
            for dirpath, _dirnames, filenames in os.walk(root_path):
                base = Path(dirpath)
                for name in filenames:
                    file_path = base / name
                    try:
                        size = file_path.stat().st_size
                    except OSError:
                        size = 0
                    file_count += 1
                    total_bytes += size
                    try:
                        rel = file_path.relative_to(root_path)
                        top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
                    except ValueError:
                        top = "(root)"
                    subdir_bytes[top] = subdir_bytes.get(top, 0) + size
        return cls(
            root=str(root_path),
            exists=exists,
            file_count=file_count,
            total_bytes=total_bytes,
            subdir_bytes=dict(sorted(subdir_bytes.items())),
            captured_at=captured_at or datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "exists": self.exists,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "subdir_bytes": dict(self.subdir_bytes),
            "captured_at": self.captured_at,
        }


# ---------------------------------------------------------------------------
# Startup timeline
# ---------------------------------------------------------------------------

#: Canonical startup mark names, in the order they are expected to occur.
#: ``process_start`` is the epoch for the whole timeline.  A mark that did
#: not occur (e.g. no DDP in single-rank mode, or compile disabled) is simply
#: absent — never faked.
STARTUP_MARKS: tuple[str, ...] = (
    "process_start",
    "assembly_start",
    "config_loaded",
    "accelerator_ready",
    "model_built",
    "optimizer_built",
    "encoders_loaded",
    "ddp_prepared",
    "compile_installed",
    "runtime_built",
    "batches_pregenerated",
    "assembly_end",
    "warmup_stage_start",
    "first_update_complete",
    "warmup_stage_end",
    "measured_stage_start",
)

_STARTUP_SCHEMA = "startup-v1"


def _delta(marks: dict[str, float], start: str, end: str) -> float | None:
    """Seconds between two marks, or None if either mark is absent."""

    if start not in marks or end not in marks:
        return None
    value = marks[end] - marks[start]
    return max(value, 0.0)


@dataclass(frozen=True, slots=True)
class CompileStartupSample:
    """The outside-the-measured-window startup timeline for one rank.

    All ``*_seconds`` headline fields are wall-clock seconds.  A headline
    field that cannot be isolated honestly (its bounding marks did not both
    occur) is ``None`` and is omitted from :meth:`to_dict` — no fake zero.
    """

    process_start_to_assembly_start_seconds: float | None
    assembly_wall_seconds: float | None
    model_build_wall_seconds: float | None
    optimizer_build_wall_seconds: float | None
    encoder_load_wall_seconds: float | None
    ddp_prepare_wall_seconds: float | None
    compile_install_wall_seconds: float | None
    warmup_total_wall_seconds: float | None
    warmup_update_walls_seconds: tuple[float, ...]
    time_to_first_successful_update_seconds: float | None
    time_to_measured_window_seconds: float | None
    marks: dict[str, float] = field(default_factory=dict[str, float])

    @classmethod
    def from_marks(
        cls,
        marks: Mapping[str, float],
        *,
        warmup_update_walls: Sequence[float] = (),
    ) -> CompileStartupSample:
        """Build a startup sample from ABSOLUTE ``time.perf_counter()`` marks.

        ``marks`` maps mark name -> absolute ``perf_counter`` value.  Only
        marks actually recorded are expected.  The result is expressed
        relative to ``process_start``.  Raises :class:`ValueError` if
        ``process_start`` is missing or any recorded mark is before it.
        """

        if "process_start" not in marks:
            raise ValueError("startup marks must include 'process_start'")
        origin = marks["process_start"]
        for name, value in marks.items():
            if type(value) is not float:
                raise ValueError(f"startup mark {name!r} must be a float")
            if value < origin - 1e-6:
                raise ValueError(f"startup mark {name!r} precedes process_start")
        relative = {name: max(0.0, value - origin) for name, value in marks.items()}
        for wall in warmup_update_walls:
            if type(wall) is not float or not math.isfinite(wall) or wall <= 0.0:
                raise ValueError(
                    "each warmup update wall must be a finite positive float"
                )

        pick = partial(_delta, relative)

        # compile_install is bounded by whichever preceding mark occurred:
        # DDP prepare in multi-rank, else encoder load (single-rank).
        compile_start = (
            "ddp_prepared" if "ddp_prepared" in relative else "encoders_loaded"
        )
        return cls(
            process_start_to_assembly_start_seconds=pick(
                "process_start", "assembly_start"
            ),
            assembly_wall_seconds=pick("assembly_start", "assembly_end"),
            model_build_wall_seconds=pick("accelerator_ready", "model_built"),
            optimizer_build_wall_seconds=pick("model_built", "optimizer_built"),
            encoder_load_wall_seconds=pick("optimizer_built", "encoders_loaded"),
            ddp_prepare_wall_seconds=pick("encoders_loaded", "ddp_prepared"),
            compile_install_wall_seconds=pick(compile_start, "compile_installed"),
            warmup_total_wall_seconds=pick("warmup_stage_start", "warmup_stage_end"),
            warmup_update_walls_seconds=tuple(warmup_update_walls),
            time_to_first_successful_update_seconds=pick(
                "process_start", "first_update_complete"
            ),
            time_to_measured_window_seconds=pick(
                "process_start", "measured_stage_start"
            ),
            marks=dict(sorted(relative.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema": _STARTUP_SCHEMA}
        optional: dict[str, float | None] = {
            "process_start_to_assembly_start_s": self.process_start_to_assembly_start_seconds,
            "assembly_wall_s": self.assembly_wall_seconds,
            "model_build_wall_s": self.model_build_wall_seconds,
            "optimizer_build_wall_s": self.optimizer_build_wall_seconds,
            "encoder_load_wall_s": self.encoder_load_wall_seconds,
            "ddp_prepare_wall_s": self.ddp_prepare_wall_seconds,
            "compile_install_wall_s": self.compile_install_wall_seconds,
            "warmup_total_wall_s": self.warmup_total_wall_seconds,
            "time_to_first_successful_update_s": self.time_to_first_successful_update_seconds,
            "time_to_measured_window_s": self.time_to_measured_window_seconds,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
        payload["warmup_update_walls_s"] = list(self.warmup_update_walls_seconds)
        payload["marks"] = dict(self.marks)
        return payload


# ---------------------------------------------------------------------------
# Diagnostic summary + classification
# ---------------------------------------------------------------------------

EXPECTED_BREAK = "expected"
UNEXPECTED_BREAK = "unexpected"

# Display-only markers that indicate an INTENTIONAL eager boundary (the DAS
# FA2 torchdynamo-disable boundary and other deliberately-eager external /
# custom operators).  Classification is evidence only; it never alters code.
_EXPECTED_BREAK_MARKERS: tuple[str, ...] = (
    "fa2",
    "fa4",
    "varlen_attention",
    "flash_attn",
    "flash-attn",
    "das",
    "_torchdynamo_disable",
    "torchdynamo_disable",
    "torch._dynamo.disable",
    "torch.compiler.disable",
    "compiler.disable",
    "dynamo.disable",
    "explicit eager",
    "intentionally eager",
)


def classify_graph_break(
    reason: str = "", *, file: str = "", function: str = ""
) -> str:
    """Classify ONE graph break as expected or unexpected (display only).

    A break is *expected* when its reason / file / function text names the
    explicit DAS FA2 ``torchdynamo-disable`` boundary or another
    intentionally-eager external / custom operator.  Everything else
    (scalar extraction, unsupported mutation, print/logging inside a
    compiled region, unsupported dynamic shape, accidental hook, ...) is
    *unexpected*.  The caller must retain the original text regardless.
    """

    haystack = " ".join(part for part in (reason, file, function) if part).lower()
    for marker in _EXPECTED_BREAK_MARKERS:
        if marker in haystack:
            return EXPECTED_BREAK
    return UNEXPECTED_BREAK


@dataclass(frozen=True, slots=True)
class CompileDiagnosticSummary:
    """Classified graph-break / recompile evidence for one diagnostic run.

    Counts are authoritative when derived from in-process compiler
    counters; the expected/unexpected split and per-reason text come from
    the retained compiler log.  ``measured_window_recompiles`` is the
    canonical signal we WANT to see as 0 — it is reported exactly, never
    fabricated.
    """

    graph_breaks_total: int
    graph_breaks_expected: int
    graph_breaks_unexpected: int
    recompiles_total: int
    recompiles_warmup: int
    recompiles_measured: int
    top_recompile_reason: str | None
    dynamic_shape_notes: str | None
    raw_log_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_breaks_total": self.graph_breaks_total,
            "graph_breaks_expected": self.graph_breaks_expected,
            "graph_breaks_unexpected": self.graph_breaks_unexpected,
            "recompiles_total": self.recompiles_total,
            "recompiles_warmup": self.recompiles_warmup,
            "recompiles_measured": self.recompiles_measured,
            "top_recompile_reason": self.top_recompile_reason,
            "dynamic_shape_notes": self.dynamic_shape_notes,
            "raw_log_preserved": self.raw_log_preserved,
        }


# ---------------------------------------------------------------------------
# Tolerant compiler-log diagnostic parser
# ---------------------------------------------------------------------------

_GRAPH_BREAK_LOCATION_RE = re.compile(
    r"graph\s+break\s*(\(.*?\)\s*)?in\s+", re.IGNORECASE
)
_GRAPH_BREAK_REASON_RE = re.compile(r"graph\s+break\s+reason\s*[:=]", re.IGNORECASE)
_GRAPH_BREAK_NOISE_RE = re.compile(
    r"for more details about this graph break", re.IGNORECASE
)
_BREAK_PATH_RE = re.compile(r"at\s+(\S+?):(\d+)")
_RECOMPILE_RE = re.compile(r"recompil\w*|re-evaluat\w*|re-evaluated", re.IGNORECASE)
# Inductor max-autotune lines say "... precompiling for N choices": the
# substring "recompil" inside "precompiling" is NOT a dynamo recompile.
_AUTOTUNE_NOISE_RE = re.compile(
    r"precompiling for|autotune benchmarking", re.IGNORECASE
)
_DYNAMIC_RE = re.compile(r"dynamic|specializ\w*|guard\b", re.IGNORECASE)
MEASURED_BANNER_RE = re.compile(r"\[bench rank\d+\] measuring \d+ logical updates")


def parse_compiler_log(text: str) -> dict[str, Any]:
    """Tolerantly parse a ``TORCH_LOGS`` compiler log for evidence.

    Returns a dict with "graph_breaks", "recompiles" and
    "dynamic_notes".  Graph-break EVENTS: a "Graph break ... in user
    code at <file>:<line>" location line starts one event; an adjacent
    (<= 2 line gap) "Graph Break Reason:" line attaches to it; a
    standalone reason line is its own event.  The "For more details
    about this graph break ..." URL line is noise and never counted.
    Inductor autotune "precompiling for N choices" lines are NOT
    recompiles.  The exact 2.9/DTK line format varies; this parser is
    deliberately loose and always retains the ORIGINAL line (GO
    P1-R1B §10).  Evidence/display only; it never drives a code change.
    """

    graph_breaks: list[dict[str, Any]] = []
    recompiles: list[dict[str, Any]] = []
    dynamic_notes: list[str] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        if not line.strip() or _GRAPH_BREAK_NOISE_RE.search(line):
            continue
        if _GRAPH_BREAK_LOCATION_RE.search(line):
            path = _BREAK_PATH_RE.search(line)
            event: dict[str, Any] = {
                "line": line,
                "line_no": line_no,
                "kind": "location",
                "reason": "",
                "file": path.group(1) if path else "",
            }
            graph_breaks.append(event)
        elif _GRAPH_BREAK_REASON_RE.search(line):
            tail = line[
                _GRAPH_BREAK_REASON_RE.search(line).end() :  # type: ignore[union-attr]
            ].strip(" :|-\u2013\u2014\t")
            if (
                graph_breaks
                and graph_breaks[-1]["kind"] == "location"
                and line_no - int(graph_breaks[-1]["line_no"]) <= 2
            ):
                graph_breaks[-1]["reason"] = tail
            else:
                graph_breaks.append(
                    {
                        "line": line,
                        "line_no": line_no,
                        "kind": "reason",
                        "reason": tail,
                        "file": "",
                    }
                )
        if _RECOMPILE_RE.search(line) and not _AUTOTUNE_NOISE_RE.search(line):
            tail = line[
                _RECOMPILE_RE.search(line).end() :  # type: ignore[union-attr]
            ].strip(" :|-\t")
            recompiles.append({"line": line, "line_no": line_no, "reason": tail})
        if _DYNAMIC_RE.search(line):
            dynamic_notes.append(line)
    return {
        "graph_breaks": graph_breaks,
        "recompiles": recompiles,
        "dynamic_notes": dynamic_notes,
    }


def split_lines_at_measured_banner(lines: Sequence[str]) -> dict[str, Any]:
    """Split log lines at the benchmark measured-stage banner.

    Returns ``{"found": bool, "before": [line_no, ...], "measured":
    [line_no, ...]}`` with 1-based line numbers.  ``before`` is
    everything up to (but not including) the banner; ``measured`` is
    the banner line and everything after it.  When the banner is
    absent, ``found`` is False and ``measured`` is EMPTY — the measured
    window is undefined, never fabricated.
    """

    before: list[int] = []
    measured: list[int] = []
    found = False
    for line_no, line in enumerate(lines, start=1):
        if not found and MEASURED_BANNER_RE.search(line):
            found = True
        (measured if found else before).append(line_no)
    if not found:
        return {"found": False, "before": before, "measured": []}
    return {"found": True, "before": before, "measured": measured}


def classify_graph_breaks(
    parsed: dict[str, Any], *, line_nos: set[int] | None = None
) -> tuple[int, int]:
    """Split parsed graph breaks into (expected, unexpected) counts.

    ``line_nos`` optionally restricts the count to a set of 1-based
    line numbers (e.g. the measured window).
    """

    expected = 0
    unexpected = 0
    for entry in parsed.get("graph_breaks", []):
        if line_nos is not None and int(entry.get("line_no", -1)) not in line_nos:
            continue
        if (
            classify_graph_break(entry.get("reason", ""), file=entry.get("file", ""))
            == EXPECTED_BREAK
        ):
            expected += 1
        else:
            unexpected += 1
    return expected, unexpected


def top_recompile_reason(parsed: dict[str, Any]) -> str | None:
    """Most common recompile reason (original text), or None if none."""

    reasons: dict[str, int] = {}
    for entry in parsed.get("recompiles", []):
        reason = entry.get("reason", "").strip()
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    if not reasons:
        return None
    return max(sorted(reasons), key=lambda key: (reasons[key], key))


def split_recompiles_by_window(
    recompiles_total: int, *, measured_window_recompiles: int
) -> tuple[int, int]:
    """Split total recompiles into (warmup, measured) given the measured count.

    The measured-window count is the authoritative in-process number; the
    warmup-window count is the remainder (clamped at zero).
    """

    if recompiles_total < 0 or measured_window_recompiles < 0:
        raise ValueError("recompile counts must be nonnegative")
    if measured_window_recompiles > recompiles_total:
        raise ValueError(
            "measured-window recompiles cannot exceed the total recompile count"
        )
    return recompiles_total - measured_window_recompiles, measured_window_recompiles


# ---------------------------------------------------------------------------
# Cold/warm math, benefit classification, bracket drift, regression gate
# ---------------------------------------------------------------------------


def pct_improvement(baseline: float, candidate: float) -> float:
    """Percent improvement where a SMALLER candidate is better.

    ``baseline`` is the reference (e.g. cold), ``candidate`` the value being
    judged (e.g. cache reuse).  Positive means the candidate is faster /
    cheaper.  A non-positive baseline yields 0.0 (no meaningful ratio).
    """

    if baseline <= 0.0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def classify_cache_benefit(improvement_pct: float, saving_seconds: float) -> str:
    """Classify persistent-cache RESTART-latency benefit (GO P1-R1B §20).

    Applied to ``time_to_first_successful_update``:

    * NEUTRAL: improvement < 10% AND absolute saving < 15 s.
    * USEFUL:  improvement >= 10% OR absolute saving >= 15 s.
    * STRONG:  improvement >= 25% OR absolute saving >= 60 s.
    """

    if not math.isfinite(improvement_pct) or not math.isfinite(saving_seconds):
        return "NEUTRAL"
    if improvement_pct >= 25.0 or saving_seconds >= 60.0:
        return "STRONG"
    if improvement_pct >= 10.0 or saving_seconds >= 15.0:
        return "USEFUL"
    return "NEUTRAL"


def environment_bracket_drift_pct(a: float, b: float) -> float:
    """Environment-bracket drift between two controls: |a-b| / mean(a,b) * 100.

    A non-positive operand yields NaN (the drift is undefined), so callers
    must not treat it as a passing 0.0.
    """

    if a <= 0.0 or b <= 0.0:
        return float("nan")
    return abs(a - b) / ((a + b) / 2.0) * 100.0


def steady_state_regression_pass(p50_delta_pct: float, memory_delta_pct: float) -> bool:
    """Steady-state regression gate for cache REUSE vs the cold bracket (GO §21).

    ``p50_delta_pct`` is the reuse-vs-cold p50 change in percent (positive =
    reuse is SLOWER).  ``memory_delta_pct`` is the peak-memory change in
    percent (positive = reuse uses more).  The gate passes when steady state
    is neutral-to-better: p50 regression <= 2% and no unexplained >5% memory
    increase.
    """

    if not math.isfinite(p50_delta_pct) or not math.isfinite(memory_delta_pct):
        return False
    return p50_delta_pct <= 2.0 and memory_delta_pct <= 5.0


# ---------------------------------------------------------------------------
# In-process compiler counters (lazy torch import; GPU/benchmark context)
# ---------------------------------------------------------------------------


def capture_dynamo_counters() -> dict[str, Any]:
    """Capture the in-process torch.compile counters, defensively.

    Returns a plain dict of the counter keys that exist in THIS runtime
    build (never guesses a missing key).  Importing torch happens lazily so
    the pure core of this module stays importable in GPU-less unit tests.
    The authoritative restart-window recompile count is the DELTA of the
    ``recompiles``-style counter between the warmup and measured window
    boundaries; the caller captures this twice and subtracts.
    """

    try:
        from torch._dynamo.utils import counters
    except (ImportError, AttributeError):  # build-specific
        return {"available": False}

    def _flatten(node: Any, prefix: str = "") -> dict[str, int]:
        out: dict[str, int] = {}
        if isinstance(node, dict):
            mapping: dict[Any, Any] = cast("dict[Any, Any]", node)
            for key, value in mapping.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[path] = int(value)
                elif isinstance(value, dict):
                    out.update(_flatten(value, path))
        return out

    flat: dict[str, int] = {}
    for key in ("stats", "warnings", "exceptions", "backends"):
        if key in counters:
            flat.update(_flatten(counters[key], key))
    return {"available": True, "counters": flat}


__all__ = [
    "EXPECTED_BREAK",
    "STARTUP_MARKS",
    "UNEXPECTED_BREAK",
    "CompileCacheSnapshot",
    "CompileDiagnosticSummary",
    "CompileEnvironment",
    "CompileStartupSample",
    "build_cache_env",
    "capture_dynamo_counters",
    "classify_cache_benefit",
    "classify_graph_break",
    "classify_graph_breaks",
    "environment_bracket_drift_pct",
    "parse_compiler_log",
    "pct_improvement",
    "sanitize_cache_identity",
    "split_lines_at_measured_banner",
    "split_recompiles_by_window",
    "steady_state_regression_pass",
    "top_recompile_reason",
]
