"""Aggregation of measured logical-update samples (P0 observatory).

Pure-Python, deterministic statistics over :class:`PerformanceSample`
records.  Percentiles use linear interpolation between closest ranks
(the "linear" method, identical to ``numpy.percentile``'s default).
Standard deviation is the sample standard deviation (n-1); with a single
sample it is reported as 0.0.

Distributed global-step semantics (schema 2, exact):

* Every rank must contribute the SAME set of logical-update identities
  (missing, duplicate or mismatched update ids fail closed).
* For each aligned update ``u`` the global wall is the SLOWEST rank:
  ``global_wall[u] = max(rank0.wall[u], rank1.wall[u], ...)``.
* ``global_step_seconds`` is the canonical statistic block over the
  aligned ``global_wall`` values — the real distributed logical-update
  critical path, NOT a pooled all-rank wall distribution and NOT
  max-of-means.
* Throughput uses actual summed cross-rank volumes over summed aligned
  global walls: ``rate = sum(global_volume[u]) / sum(global_wall[u])``.
  Ranks are NOT assumed to carry equal per-update token counts.
* ``per_rank_step_seconds`` stays rank-local.
* Phase timing blocks are pooled rank-local component statistics; a
  phase's ``share_of_step`` uses the MEAN aligned global wall as the
  denominator.  Phase means are not claimed to decompose one exact
  global critical path when overlap exists.

Memory semantics (exact): per-rank ``peak_allocated_bytes`` /
``peak_reserved_bytes`` are the max over the measured window of the true
per-update ``torch.cuda.max_memory_*`` counters (each update's peak
window is reset immediately before that update); ``final_*`` are the
current post-update counts of the last measured update.  Current values
are never reported as peaks.

* ``dit_forward_matmul_tflops_per_second`` is the DiT FORWARD matmul FLOP
  counter divided by the measured ``dit_forward`` phase seconds.  It is
  NOT MFU and NOT a whole-step training FLOPS model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from sakuramoon.perf.sample import PerformanceSample

SUMMARY_SCHEMA_VERSION = 2

_STAT_KEYS: tuple[str, ...] = (
    "mean",
    "p50",
    "p90",
    "p95",
    "min",
    "max",
    "stddev",
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""

    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile q must be in [0, 100]")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence is undefined")
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    fraction = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def step_stats(wall_seconds: Sequence[float]) -> dict[str, float]:
    """The canonical step-time statistic block for a sample set."""

    values = [float(value) for value in wall_seconds]
    if not values:
        raise ValueError("step_stats requires at least one sample")
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50.0),
        "p90": percentile(values, 90.0),
        "p95": percentile(values, 95.0),
        "min": min(values),
        "max": max(values),
        "stddev": _stddev(values),
    }


def rank_step_skew_pct(rank_mean_wall: Mapping[int, float]) -> float:
    """Relative skew between the fastest and slowest rank mean step wall.

    ``0.0`` for a single rank or perfectly symmetric ranks.
    """

    values = [float(value) for value in rank_mean_wall.values()]
    if not values:
        raise ValueError("rank skew requires at least one rank")
    fastest = min(values)
    slowest = max(values)
    if slowest <= 0.0:
        raise ValueError("rank mean step wall must be positive")
    return (slowest - fastest) / slowest * 100.0


def _align_updates(
    rank_samples: Mapping[int, Sequence[PerformanceSample]],
) -> tuple[list[int], dict[int, dict[int, PerformanceSample]]]:
    """Align every rank by logical-update identity.

    Returns ``(sorted_update_ids, {rank: {update: sample}})``.  Fails
    closed on an empty rank, a duplicate update id within a rank, or a
    mismatched update-id set between ranks.
    """

    per_rank_map: dict[int, dict[int, PerformanceSample]] = {}
    reference: set[int] | None = None
    for rank in sorted(rank_samples):
        mapping: dict[int, PerformanceSample] = {}
        for sample in rank_samples[rank]:
            if sample.update in mapping:
                raise ValueError(
                    f"rank {rank} reports duplicate update {sample.update}"
                )
            mapping[sample.update] = sample
        if not mapping:
            raise ValueError(
                "every rank must contribute the same nonzero set of measured updates"
            )
        identity = set(mapping)
        if reference is None:
            reference = identity
        elif identity != reference:
            missing = sorted(reference - identity)
            extra = sorted(identity - reference)
            raise ValueError(
                "rank update identities differ from the reference rank "
                f"(missing: {missing}, extra: {extra})"
            )
        per_rank_map[int(rank)] = mapping
    return sorted(cast(set[int], reference)), per_rank_map


def _phase_pool(samples: Sequence[PerformanceSample]) -> dict[str, list[float]]:
    pool: dict[str, list[float]] = {}
    for sample in samples:
        for name, seconds in sample.phases.items():
            pool.setdefault(name, []).append(float(seconds))
    return pool


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Aggregated, JSON-serializable baseline summary for one run."""

    world_size: int
    measured_iterations: int
    warmup_iterations: int
    per_rank_step_seconds: dict[int, dict[str, float]]
    global_step_seconds: dict[str, float]
    global_samples_per_second: float
    image_tokens_per_second: float
    text_tokens_per_second: float
    phase_seconds: dict[str, dict[str, float | None]]
    memory: dict[str, Any]
    dit: dict[str, Any]
    schema_version: int = SUMMARY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "world_size": self.world_size,
            "measured_iterations": self.measured_iterations,
            "warmup_iterations": self.warmup_iterations,
            "per_rank_step_seconds": {
                int(rank): dict(block)
                for rank, block in sorted(self.per_rank_step_seconds.items())
            },
            "global_step_seconds": dict(self.global_step_seconds),
            "global_samples_per_second": self.global_samples_per_second,
            "image_tokens_per_second": self.image_tokens_per_second,
            "text_tokens_per_second": self.text_tokens_per_second,
            "phase_seconds": {
                name: dict(self.phase_seconds[name])
                for name in sorted(self.phase_seconds)
            },
            "memory": self.memory,
            "dit": self.dit,
        }


def summarize(
    rank_samples: Mapping[int, Sequence[PerformanceSample]],
    *,
    warmup_iterations: int,
    world_size: int,
) -> PerformanceSummary:
    """Aggregate per-rank measured samples into one summary.

    All ranks must contribute the exact same logical-update identity
    set; per-update global walls take the slowest rank, and throughput
    volumes are summed across ranks per update.
    """

    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive int")
    if type(warmup_iterations) is not int or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a nonnegative int")
    if not rank_samples:
        raise ValueError("summarize requires at least one rank")

    updates, per_rank_map = _align_updates(rank_samples)
    measured = len(updates)
    ranks = sorted(per_rank_map)

    # Per-update aligned global walls: the slowest rank bounds the update.
    global_walls = [
        max(per_rank_map[rank][update].wall_seconds for rank in ranks)
        for update in updates
    ]
    global_step = step_stats(global_walls)
    global_wall_sum = sum(global_walls)
    # Mean aligned global wall: the documented share_of_step denominator.
    global_step_mean = sum(global_walls) / len(global_walls)

    # Rank-local distributions, in aligned update order.
    per_rank: dict[int, dict[str, float]] = {}
    rank_means: dict[int, float] = {}
    for rank in ranks:
        walls = [per_rank_map[rank][update].wall_seconds for update in updates]
        per_rank[rank] = step_stats(walls)
        rank_means[rank] = sum(walls) / len(walls)

    # Actual summed cross-rank volumes over the measured window.
    total_samples = 0
    total_image_tokens = 0
    total_text_tokens = 0
    for update in updates:
        for rank in ranks:
            sample = per_rank_map[rank][update]
            total_samples += sample.samples
            total_image_tokens += sample.image_tokens
            total_text_tokens += sample.text_tokens
    if global_wall_sum <= 0.0:
        raise ValueError("summed aligned global walls must be positive")
    global_samples_per_second = total_samples / global_wall_sum
    image_tokens_per_second = total_image_tokens / global_wall_sum
    text_tokens_per_second = total_text_tokens / global_wall_sum

    all_samples = [per_rank_map[rank][update] for rank in ranks for update in updates]
    phase_seconds: dict[str, dict[str, float | None]] = {}
    pool = _phase_pool(all_samples)
    total_samples_count = len(all_samples)
    for name in sorted(pool):
        values = pool[name]
        if len(values) != total_samples_count:
            share: float | None = None
        else:
            share = (sum(values) / len(values)) / global_step_mean
        phase_seconds[name] = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 50.0),
            "p95": percentile(values, 95.0),
            "share_of_step": share,
        }

    memory: dict[str, Any] = {"per_rank": {}}
    max_peak_allocated = 0
    max_peak_reserved = 0
    for rank in ranks:
        rows = [per_rank_map[rank][update] for update in updates]
        peak_allocated = max(sample.peak_memory_allocated_bytes for sample in rows)
        peak_reserved = max(sample.peak_memory_reserved_bytes for sample in rows)
        memory["per_rank"][rank] = {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "final_allocated_bytes": rows[-1].memory_allocated_bytes,
            "final_reserved_bytes": rows[-1].memory_reserved_bytes,
        }
        max_peak_allocated = max(max_peak_allocated, peak_allocated)
        max_peak_reserved = max(max_peak_reserved, peak_reserved)
    memory["max_across_ranks"] = {
        "peak_allocated_bytes": max_peak_allocated,
        "peak_reserved_bytes": max_peak_reserved,
    }

    dit_per_rank: dict[int, Any] = {}
    dit_tflops: list[float] = []
    for rank in ranks:
        rows = [per_rank_map[rank][update] for update in updates]
        flops = sum(sample.dit_forward_matmul_flops for sample in rows)
        seconds = sum(sample.phases.get("dit_forward", 0.0) for sample in rows)
        rate = (flops / seconds / 1e12) if seconds > 0.0 else 0.0
        dit_per_rank[rank] = {
            "measured_forward_matmul_flops": flops,
            "dit_forward_seconds": seconds,
            "dit_forward_matmul_tflops_per_second": rate,
        }
        dit_tflops.append(rate)
    dit: dict[str, Any] = {
        "measured_forward_matmul_flops": sum(
            row["measured_forward_matmul_flops"] for row in dit_per_rank.values()
        ),
        "per_rank": dit_per_rank,
        "max_across_ranks_tflops_per_second": max(dit_tflops) if dit_tflops else 0.0,
    }

    return PerformanceSummary(
        world_size=world_size,
        measured_iterations=measured,
        warmup_iterations=warmup_iterations,
        per_rank_step_seconds=per_rank,
        global_step_seconds=global_step,
        global_samples_per_second=global_samples_per_second,
        image_tokens_per_second=image_tokens_per_second,
        text_tokens_per_second=text_tokens_per_second,
        phase_seconds=phase_seconds,
        memory=memory,
        dit=dit,
    )


__all__ = [
    "SUMMARY_SCHEMA_VERSION",
    "PerformanceSummary",
    "percentile",
    "rank_step_skew_pct",
    "step_stats",
    "summarize",
]
