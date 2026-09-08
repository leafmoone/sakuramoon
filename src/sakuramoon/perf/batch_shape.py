"""Microbatch / gradient-accumulation shape utilities (P1-R1A sweep).

The sweep compares candidate factor pairs of the SAME effective global
batch (canonical 256px G1: local_batch 20, accumulation 20, world 2 ->
global batch 800; 400 samples per rank per logical update).

``benchmark_batch_shape_config`` derives a candidate shape IN MEMORY from
the loaded production config: no TOML is written and no canonical config
file is modified.  The global-batch derivation FAILS CLOSED when it
disagrees with the sweep's required global batch, so a wrong factor pair
can never silently change the effective training batch.

Every helper in this module is pure (no torch, no I/O): the sweep
orchestration, its classification and its math are unit-testable
without GPUs or models.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sakuramoon.config.schema import RuntimeConfig

GIB = 2**30

# Canonical sweep workload (P1-R1A): the 2-GPU G1 production shape.
CANONICAL_WORLD_SIZE = 2
CANONICAL_GLOBAL_BATCH = 800
CANONICAL_SAMPLES_PER_RANK_UPDATE = 400
SWEEP_MEASURED_UPDATES = 10

# Memory safety classes for the 64-GiB benchmark HCUs.  A future
# production recommendation is SAFE only inside both envelopes (with no
# OOM and a stable 10-update measured window); beyond either envelope it
# is TIGHT; OOM / nonfinite / instability is UNSAFE.
MEMORY_SAFE_PEAK_ALLOCATED_GIB = 54.0
MEMORY_SAFE_PEAK_RESERVED_GIB = 60.0
MEMORY_CLASSES = ("SAFE", "TIGHT", "UNSAFE")

# C2 (40x10) exploratory safety gate: it runs only if C1 (25x16) stayed
# inside BOTH envelopes measured on the same session.
C2_GATE_PEAK_ALLOCATED_GIB = 48.0
C2_GATE_PEAK_RESERVED_GIB = 56.0

# Baseline drift gate: A0/A1 p50 relative difference, <= 2% of the mean.
BASELINE_DRIFT_LIMIT_PCT = 2.0

# Speed classification thresholds (whole-step p50 improvement).
SPEED_USEFUL_PCT = 2.0
SPEED_STRONG_PCT = 5.0
SPEED_CLASSES = ("NEUTRAL", "USEFUL", "STRONG")

# Candidate statuses (never hidden, always serialized).
STATUS_PASS = "PASS"
STATUS_OOM = "OOM"
STATUS_SKIPPED_SAFETY_GATE = "SKIPPED_SAFETY_GATE"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_ERROR = "ERROR"
CANDIDATE_STATUSES = (
    STATUS_PASS,
    STATUS_OOM,
    STATUS_SKIPPED_SAFETY_GATE,
    STATUS_TIMEOUT,
    STATUS_ERROR,
)

_OOM_LOG_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "hip oom",
    "oom: ",
    "cannot allocate",
)

_SHAPE_SPEC_RE = re.compile(r"^(\d+):(\d+)$")


@dataclass(frozen=True, slots=True)
class BatchShapeCandidate:
    """One sweep slot: a factor pair plus its run-order label.

    ``dir_name`` is the per-shape scratch directory (the two 20x20
    bracket runs need distinct names: ``lb20-acc20-a`` / ``lb20-acc20-b``).
    """

    label: str
    local_batch: int
    accumulation: int
    dir_name: str


#: Canonical P1-R1A run order (A0 -> S -> C1 -> C2 -> A1).  C2 is
#: safety-gated from C1's measured peaks at execution time.
CANONICAL_SWEEP: tuple[BatchShapeCandidate, ...] = (
    BatchShapeCandidate("A0", 20, 20, "lb20-acc20-a"),
    BatchShapeCandidate("S", 16, 25, "lb16-acc25"),
    BatchShapeCandidate("C1", 25, 16, "lb25-acc16"),
    BatchShapeCandidate("C2", 40, 10, "lb40-acc10"),
    BatchShapeCandidate("A1", 20, 20, "lb20-acc20-b"),
)


def benchmark_batch_shape_config(
    config: RuntimeConfig,
    local_batch: int,
    accumulation: int,
    *,
    expected_global_batch: int | None = None,
) -> RuntimeConfig:
    """Derive a benchmark-only config with the given microbatch shape.

    world_size, model, resolution, optimizer and the LR rule are left
    untouched; ``global_batch`` is re-derived as
    ``local_batch * accumulation * world_size`` and the derivation FAILS
    CLOSED when it disagrees with ``expected_global_batch`` (the canonical
    sweep requires 800).  Pure in-memory ``model_copy``: no config file
    is read or written.
    """

    if type(local_batch) is not int or local_batch <= 0:
        raise ValueError("local_batch must be a positive int")
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("accumulation must be a positive int")
    if expected_global_batch is not None and (
        type(expected_global_batch) is not int or expected_global_batch <= 0
    ):
        raise ValueError("expected_global_batch must be a positive int")
    train = config.train
    world_size = config.distributed.world_size
    global_batch = local_batch * accumulation * world_size
    if expected_global_batch is not None and global_batch != expected_global_batch:
        raise ValueError(
            f"batch shape local_batch={local_batch} accumulation={accumulation} "
            f"world_size={world_size} gives global batch {global_batch}, "
            f"expected {expected_global_batch}"
        )
    if local_batch == train.local_batch and accumulation == train.accumulation:
        return config
    return config.model_copy(
        update={
            "train": train.model_copy(
                update={
                    "local_batch": local_batch,
                    "accumulation": accumulation,
                    "global_batch": global_batch,
                }
            )
        }
    )


def logical_update_sample_count(local_batch: int, accumulation: int) -> int:
    """Samples per logical update (shape-invariant when the product is)."""

    if type(local_batch) is not int or local_batch <= 0:
        raise ValueError("local_batch must be a positive int")
    if type(accumulation) is not int or accumulation <= 0:
        raise ValueError("accumulation must be a positive int")
    return local_batch * accumulation


def parse_shape_spec(spec: str) -> tuple[int, int]:
    """Parse ``"<local_batch>:<accumulation>"`` into positive ints."""

    match = _SHAPE_SPEC_RE.fullmatch(spec.strip())
    if match is None:
        raise ValueError(
            "batch shape spec must be '<local_batch>:<accumulation>' "
            f"without spaces: {spec!r}"
        )
    local_batch, accumulation = int(match.group(1)), int(match.group(2))
    if local_batch <= 0 or accumulation <= 0:
        raise ValueError("batch shape components must be positive")
    return local_batch, accumulation


def median_pair(a: float, b: float) -> float:
    """Median of exactly two values (the A0/A1 bracket reference)."""

    if a <= 0.0 or b <= 0.0:
        raise ValueError("median_pair values must be positive")
    return (a + b) / 2.0


def baseline_drift_pct(p50_a: float, p50_b: float) -> float:
    """|p50_a - p50_b| / mean(p50_a, p50_b) * 100 (GO §12 drift metric)."""

    if p50_a <= 0.0 or p50_b <= 0.0:
        raise ValueError("baseline p50 values must be positive")
    return abs(p50_b - p50_a) / ((p50_a + p50_b) / 2.0) * 100.0


def is_environment_stable(p50_a: float, p50_b: float) -> bool:
    """True when the A0/A1 drift is within the 2% stability gate."""

    return baseline_drift_pct(p50_a, p50_b) <= BASELINE_DRIFT_LIMIT_PCT


def speedup_pct(reference_p50: float, candidate_p50: float) -> float:
    """Whole-step p50 improvement vs the reference, in percent.

    Positive means the candidate is FASTER (smaller p50).
    """

    if reference_p50 <= 0.0 or candidate_p50 <= 0.0:
        raise ValueError("p50 values must be positive")
    return (reference_p50 - candidate_p50) / reference_p50 * 100.0


def speed_class(improvement_pct: float) -> str:
    """NEUTRAL < 2% <= USEFUL < 5% <= STRONG (whole-step p50)."""

    if improvement_pct >= SPEED_STRONG_PCT:
        return "STRONG"
    if improvement_pct >= SPEED_USEFUL_PCT:
        return "USEFUL"
    return "NEUTRAL"


def classify_memory_safety(
    peak_allocated_gib: float,
    peak_reserved_gib: float,
    *,
    oom: bool = False,
    measured_updates: int = SWEEP_MEASURED_UPDATES,
) -> str:
    """SAFE / TIGHT / UNSAFE for a candidate's measured memory envelope."""

    if oom or measured_updates < SWEEP_MEASURED_UPDATES:
        return "UNSAFE"
    if (
        peak_allocated_gib > MEMORY_SAFE_PEAK_ALLOCATED_GIB
        or peak_reserved_gib > MEMORY_SAFE_PEAK_RESERVED_GIB
    ):
        return "TIGHT"
    return "SAFE"


def high_candidate_gate_passed(
    peak_allocated_gib: float, peak_reserved_gib: float
) -> bool:
    """C2 gate from C1's measured peaks (both envelopes inclusive)."""

    return (
        peak_allocated_gib <= C2_GATE_PEAK_ALLOCATED_GIB
        and peak_reserved_gib <= C2_GATE_PEAK_RESERVED_GIB
    )


def oom_marker_found(log_text: str) -> str | None:
    """First OOM marker present in the log (case-insensitive), if any."""

    lowered = log_text.lower()
    for marker in _OOM_LOG_MARKERS:
        if marker in lowered:
            return marker
    return None


def classify_candidate_status(
    exit_code: int, log_text: str, summary_found: bool
) -> str:
    """Classify one finished (not timed-out) candidate subprocess."""

    if exit_code == 0 and summary_found:
        return STATUS_PASS
    if oom_marker_found(log_text) is not None:
        return STATUS_OOM
    return STATUS_ERROR


def summarize_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the sweep report fields from one candidate's summary JSON.

    The payload is the rank-0 aggregated baseline JSON (schema 2) written
    by ``benchmark_training_perf.py``.
    """

    memory_per_rank = payload["memory"]["per_rank"]
    peak_allocated = {
        int(rank): block["peak_allocated_bytes"] / GIB
        for rank, block in memory_per_rank.items()
    }
    peak_reserved = {
        int(rank): block["peak_reserved_bytes"] / GIB
        for rank, block in memory_per_rank.items()
    }
    phases = {name: block["mean"] for name, block in payload["phase_seconds"].items()}
    return {
        "p50_s": payload["global_step_seconds"]["p50"],
        "mean_s": payload["global_step_seconds"]["mean"],
        "p95_s": payload["global_step_seconds"]["p95"],
        "samples_per_s": payload["global_samples_per_second"],
        "rank_skew_pct": payload.get("rank_step_skew_pct", 0.0),
        "measured_updates": payload["measured_iterations"],
        "world_size": payload["world_size"],
        "phases_s": phases,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "max_peak_allocated_gib": max(peak_allocated.values()),
        "max_peak_reserved_gib": max(peak_reserved.values()),
    }


def rank_best_safe(
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Pick the best SAFE candidate by the GO §21 ranking.

    Primary: highest global samples/s; tie-breaks: lower p50, lower p95,
    lower max peak allocated, lower rank skew.  Returns ``(dir_name,
    reason)`` or ``None`` when no entry qualifies (entries are the
    already-filtered PASS + SAFE candidate summaries).
    """

    if not entries:
        return None
    names = sorted(entries)
    first = names[0]
    first_entry = entries[first]
    best_name = first
    best_key = (
        -float(first_entry["samples_per_s"]),
        float(first_entry["p50_s"]),
        float(first_entry["p95_s"]),
        float(first_entry["max_peak_allocated_gib"]),
        float(first_entry["rank_skew_pct"]),
        first,
    )
    for name in names[1:]:
        entry = entries[name]
        key = (
            -float(entry["samples_per_s"]),
            float(entry["p50_s"]),
            float(entry["p95_s"]),
            float(entry["max_peak_allocated_gib"]),
            float(entry["rank_skew_pct"]),
            name,
        )
        if key < best_key:
            best_name = name
            best_key = key
    winner = entries[best_name]
    reason = (
        f"highest global throughput among SAFE PASS candidates "
        f"({winner['samples_per_s']:.2f} samples/s, p50 {winner['p50_s']:.3f} s)"
    )
    return best_name, reason


__all__ = [
    "BASELINE_DRIFT_LIMIT_PCT",
    "C2_GATE_PEAK_ALLOCATED_GIB",
    "C2_GATE_PEAK_RESERVED_GIB",
    "CANDIDATE_STATUSES",
    "CANONICAL_GLOBAL_BATCH",
    "CANONICAL_SAMPLES_PER_RANK_UPDATE",
    "CANONICAL_SWEEP",
    "CANONICAL_WORLD_SIZE",
    "GIB",
    "MEMORY_CLASSES",
    "MEMORY_SAFE_PEAK_ALLOCATED_GIB",
    "MEMORY_SAFE_PEAK_RESERVED_GIB",
    "SPEED_CLASSES",
    "SPEED_STRONG_PCT",
    "SPEED_USEFUL_PCT",
    "STATUS_ERROR",
    "STATUS_OOM",
    "STATUS_PASS",
    "STATUS_SKIPPED_SAFETY_GATE",
    "STATUS_TIMEOUT",
    "SWEEP_MEASURED_UPDATES",
    "BatchShapeCandidate",
    "baseline_drift_pct",
    "benchmark_batch_shape_config",
    "classify_candidate_status",
    "classify_memory_safety",
    "high_candidate_gate_passed",
    "is_environment_stable",
    "logical_update_sample_count",
    "median_pair",
    "oom_marker_found",
    "parse_shape_spec",
    "rank_best_safe",
    "speed_class",
    "speedup_pct",
    "summarize_candidate",
]
