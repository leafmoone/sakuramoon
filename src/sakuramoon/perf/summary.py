"""Aggregation of measured logical-update samples (P0 observatory).

Pure-Python, deterministic statistics over :class:`PerformanceSample`
records.  Percentiles use linear interpolation between closest ranks
(the "linear" method, identical to ``numpy.percentile``'s default).
Standard deviation is the sample standard deviation (n-1); with a single
sample it is reported as 0.0.

Throughput and DiT matmul rate conventions (documented, exact):

* the global step time is the MAX across ranks of the per-rank mean step
  wall — distributed ranks are lock-step at the allreduce boundary, so
  the slowest rank bounds the update;
* ``dit_forward_matmul_tflops_per_second`` is the DiT FORWARD matmul FLOP
  counter divided by the measured ``dit_forward`` phase seconds.  It is
  NOT MFU and NOT a whole-step training FLOPS model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sakuramoon.perf.sample import PerformanceSample

SUMMARY_SCHEMA_VERSION = 1

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

    All ranks must contribute the same number of measured iterations.
    Phases are pooled across ranks for the distribution statistics; a
    phase's ``share_of_step`` is defined only when the phase was measured
    in EVERY sample (otherwise null).
    """

    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive int")
    if type(warmup_iterations) is not int or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a nonnegative int")
    if not rank_samples:
        raise ValueError("summarize requires at least one rank")
    first_rank = next(iter(rank_samples))
    lengths = {rank: len(samples) for rank, samples in rank_samples.items()}
    if len(set(lengths.values())) != 1 or lengths[first_rank] == 0:
        raise ValueError("every rank must contribute the same nonzero iteration count")
    measured = lengths[first_rank]

    all_samples = [
        sample for rank in sorted(rank_samples) for sample in rank_samples[rank]
    ]

    per_rank: dict[int, dict[str, float]] = {}
    rank_means: dict[int, float] = {}
    for rank in sorted(rank_samples):
        walls = [sample.wall_seconds for sample in rank_samples[rank]]
        per_rank[int(rank)] = step_stats(walls)
        rank_means[int(rank)] = sum(walls) / len(walls)

    global_step = step_stats([sample.wall_seconds for sample in all_samples])
    global_step_time = max(rank_means.values())
    # Every rank carries the same local batch/accumulation, so the global
    # per-update volume is rank 0's local volume times the world size.
    samples_per_update = all_samples[0].samples * world_size
    image_tokens_per_update = all_samples[0].image_tokens * world_size
    text_tokens_per_update = all_samples[0].text_tokens * world_size
    global_samples_per_second = samples_per_update / global_step_time
    image_tokens_per_second = image_tokens_per_update / global_step_time
    text_tokens_per_second = text_tokens_per_update / global_step_time

    phase_seconds: dict[str, dict[str, float | None]] = {}
    pool = _phase_pool(all_samples)
    total_samples = len(all_samples)
    for name in sorted(pool):
        values = pool[name]
        if len(values) != total_samples:
            share: float | None = None
        else:
            share = (sum(values) / len(values)) / global_step_time
        phase_seconds[name] = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 50.0),
            "p95": percentile(values, 95.0),
            "share_of_step": share,
        }

    memory: dict[str, Any] = {"per_rank": {}}
    max_allocated = 0
    max_reserved = 0
    for rank in sorted(rank_samples):
        rows = rank_samples[rank]
        allocated = max(sample.memory_allocated_bytes for sample in rows)
        reserved = max(sample.memory_reserved_bytes for sample in rows)
        memory["per_rank"][int(rank)] = {
            "max_allocated_bytes": allocated,
            "max_reserved_bytes": reserved,
            "final_allocated_bytes": rows[-1].memory_allocated_bytes,
            "final_reserved_bytes": rows[-1].memory_reserved_bytes,
        }
        max_allocated = max(max_allocated, allocated)
        max_reserved = max(max_reserved, reserved)
    memory["max_across_ranks"] = {
        "max_allocated_bytes": max_allocated,
        "max_reserved_bytes": max_reserved,
    }

    dit_per_rank: dict[int, Any] = {}
    dit_tflops: list[float] = []
    for rank in sorted(rank_samples):
        rows = rank_samples[rank]
        flops = sum(sample.dit_forward_matmul_flops for sample in rows)
        seconds = sum(sample.phases.get("dit_forward", 0.0) for sample in rows)
        rate = (flops / seconds / 1e12) if seconds > 0.0 else 0.0
        dit_per_rank[int(rank)] = {
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
