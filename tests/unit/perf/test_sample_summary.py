"""Unit tests for P0 performance samples and summary aggregation."""

from __future__ import annotations

import json
import math

import pytest

from sakuramoon.perf.sample import PerformanceSample
from sakuramoon.perf.summary import (
    percentile,
    rank_step_skew_pct,
    step_stats,
    summarize,
)


def _sample(
    update: int = 1,
    wall: float = 10.0,
    samples: int = 40,
    image_tokens: int = 10240,
    text_tokens: int = 8000,
    flops: int = 1_000_000_000_000,
    phases: dict[str, float] | None = None,
    allocated: int = 1 << 30,
    reserved: int = 2 << 30,
) -> PerformanceSample:
    if phases is None:
        phases = {
            "h2d": 0.05,
            "qwen": 1.0,
            "vae": 0.5,
            "dit_forward": 3.0,
            "loss": 0.2,
            "backward": 4.0,
            "optimizer": 1.2,
        }
    return PerformanceSample(
        update=update,
        wall_seconds=wall,
        samples=samples,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        dit_forward_matmul_flops=flops,
        phases=phases,
        memory_allocated_bytes=allocated,
        memory_reserved_bytes=reserved,
    )


class TestSampleValidation:
    def test_rejects_zero_and_negative_wall(self) -> None:
        with pytest.raises(ValueError, match="wall_seconds"):
            _sample(wall=0.0)
        with pytest.raises(ValueError, match="wall_seconds"):
            _sample(wall=-1.0)

    def test_rejects_nonfinite_wall(self) -> None:
        with pytest.raises(ValueError, match="wall_seconds"):
            _sample(wall=float("nan"))
        with pytest.raises(ValueError, match="wall_seconds"):
            _sample(wall=float("inf"))

    def test_rejects_non_float_wall(self) -> None:
        with pytest.raises(ValueError, match="wall_seconds"):
            _sample(wall=10)  # type: ignore[arg-type]

    def test_rejects_zero_samples(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            _sample(samples=0)

    def test_rejects_negative_token_counts(self) -> None:
        with pytest.raises(ValueError, match="image_tokens"):
            _sample(image_tokens=-1)

    def test_rejects_unknown_phase(self) -> None:
        with pytest.raises(ValueError, match="unknown performance phase"):
            _sample(phases={"not_a_phase": 1.0})

    def test_rejects_negative_phase_seconds(self) -> None:
        with pytest.raises(ValueError, match="phases"):
            _sample(phases={"loss": -0.1})

    def test_rejects_missing_update(self) -> None:
        with pytest.raises(ValueError, match="update"):
            _sample(update=0)

    def test_rejects_bad_memory_fields(self) -> None:
        with pytest.raises(ValueError, match="memory_allocated_bytes"):
            _sample(allocated=-1)

    def test_empty_phases_are_allowed(self) -> None:
        sample = _sample(phases={})
        assert sample.phases == {}


class TestJsonRoundTrip:
    def test_round_trip_is_exact(self) -> None:
        original = _sample(update=7, wall=12.25)
        payload = original.to_dict()
        encoded = json.dumps(payload)
        restored = PerformanceSample.from_dict(json.loads(encoded))
        assert restored == original
        assert restored.to_dict() == payload

    def test_round_trip_preserves_phase_order_independence(self) -> None:
        original = _sample()
        restored = PerformanceSample.from_dict(original.to_dict())
        assert set(restored.phases) == set(original.phases)

    def test_from_dict_rejects_unknown_keys(self) -> None:
        payload = _sample().to_dict()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="payload keys"):
            PerformanceSample.from_dict(payload)

    def test_from_dict_rejects_missing_keys(self) -> None:
        payload = _sample().to_dict()
        del payload["wall_seconds"]
        with pytest.raises(ValueError, match="payload keys"):
            PerformanceSample.from_dict(payload)


class TestPercentile:
    def test_linear_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 10.0
        assert percentile(values, 50) == 5.5
        assert percentile(values, 90) == 9.1
        assert percentile(values, 25) == 3.25

    def test_single_value(self) -> None:
        assert percentile([42.0], 95) == 42.0

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)

    def test_out_of_range_q_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            percentile([1.0], 101)


class TestStepStats:
    def test_full_stat_block(self) -> None:
        stats = step_stats([1.0, 2.0, 3.0, 4.0])
        assert set(stats) == {"mean", "p50", "p90", "p95", "min", "max", "stddev"}
        assert stats["mean"] == 2.5
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert math.isclose(stats["stddev"], 1.2909944487358056, rel_tol=1e-12)

    def test_single_sample_stddev_is_zero(self) -> None:
        assert step_stats([3.0])["stddev"] == 0.0

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            step_stats([])


class TestRankSkew:
    def test_symmetric_ranks_are_zero(self) -> None:
        assert rank_step_skew_pct({0: 10.0, 1: 10.0}) == 0.0

    def test_single_rank_is_zero(self) -> None:
        assert rank_step_skew_pct({0: 10.0}) == 0.0

    def test_skew_is_relative_to_slowest(self) -> None:
        assert math.isclose(rank_step_skew_pct({0: 9.0, 1: 10.0}), 10.0)

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one rank"):
            rank_step_skew_pct({})


class TestSummarize:
    def _two_rank_samples(self) -> dict[int, list[PerformanceSample]]:
        rank0 = [_sample(update=i, wall=10.0) for i in range(1, 5)]
        rank1 = [_sample(update=i, wall=12.0) for i in range(1, 5)]
        # Rank 1 lacks the vae phase in one sample: share must become null.
        rank1[2] = PerformanceSample(
            update=3,
            wall_seconds=12.0,
            samples=40,
            image_tokens=10240,
            text_tokens=8000,
            dit_forward_matmul_flops=1_000_000_000_000,
            phases={
                "h2d": 0.05,
                "qwen": 1.0,
                "dit_forward": 3.0,
                "loss": 0.2,
                "backward": 4.0,
                "optimizer": 1.2,
            },
            memory_allocated_bytes=1 << 30,
            memory_reserved_bytes=2 << 30,
        )
        return {0: rank0, 1: rank1}

    def test_summary_counts_and_throughput(self) -> None:
        summary = summarize(self._two_rank_samples(), warmup_iterations=5, world_size=2)
        assert summary.measured_iterations == 4
        assert summary.warmup_iterations == 5
        assert summary.world_size == 2
        # Global step time = max rank mean (12.0); 40 samples/update/rank.
        assert summary.global_samples_per_second == pytest.approx(80.0 / 12.0)
        assert summary.image_tokens_per_second == pytest.approx(20480.0 / 12.0)
        assert summary.text_tokens_per_second == pytest.approx(16000.0 / 12.0)
        assert summary.per_rank_step_seconds[0]["mean"] == 10.0
        assert summary.per_rank_step_seconds[1]["mean"] == 12.0

    def test_phase_share_defined_only_when_measured_everywhere(self) -> None:
        summary = summarize(self._two_rank_samples(), warmup_iterations=5, world_size=2)
        qwen = summary.phase_seconds["qwen"]
        assert qwen["share_of_step"] is not None
        assert qwen["share_of_step"] == pytest.approx(1.0 / 12.0)
        assert summary.phase_seconds["vae"]["share_of_step"] is None
        assert summary.phase_seconds["vae"]["mean"] == pytest.approx(0.5)

    def test_memory_block(self) -> None:
        summary = summarize(self._two_rank_samples(), warmup_iterations=5, world_size=2)
        per_rank = summary.memory["per_rank"]
        assert per_rank[0]["max_allocated_bytes"] == 1 << 30
        assert per_rank[1]["final_reserved_bytes"] == 2 << 30
        top = summary.memory["max_across_ranks"]
        assert top["max_allocated_bytes"] == 1 << 30
        assert top["max_reserved_bytes"] == 2 << 30

    def test_dit_matmul_rate(self) -> None:
        summary = summarize(self._two_rank_samples(), warmup_iterations=5, world_size=2)
        dit = summary.dit
        # 4 updates x 1e12 flops x 2 ranks
        assert dit["measured_forward_matmul_flops"] == 8_000_000_000_000
        per_rank = dit["per_rank"]
        assert per_rank[0]["dit_forward_seconds"] == pytest.approx(12.0)
        assert per_rank[0]["dit_forward_matmul_tflops_per_second"] == pytest.approx(
            4_000_000_000_000 / 12.0 / 1e12
        )
        assert (
            dit["max_across_ranks_tflops_per_second"]
            == per_rank[0]["dit_forward_matmul_tflops_per_second"]
        )

    def test_mismatched_iteration_counts_are_rejected(self) -> None:
        samples = self._two_rank_samples()
        del samples[1][0]
        with pytest.raises(ValueError, match="same nonzero iteration count"):
            summarize(samples, warmup_iterations=5, world_size=2)

    def test_empty_ranks_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one rank"):
            summarize({}, warmup_iterations=5, world_size=1)

    def test_to_dict_is_json_serializable(self) -> None:
        summary = summarize(self._two_rank_samples(), warmup_iterations=5, world_size=2)
        encoded = json.dumps(summary.to_dict(), sort_keys=True)
        assert "global_samples_per_second" in encoded

    def test_invalid_world_size_is_rejected(self) -> None:
        samples = {0: [_sample()]}
        with pytest.raises(ValueError, match="world_size"):
            summarize(samples, warmup_iterations=0, world_size=0)

    def test_invalid_warmup_is_rejected(self) -> None:
        samples = {0: [_sample()]}
        with pytest.raises(ValueError, match="warmup_iterations"):
            summarize(samples, warmup_iterations=-1, world_size=1)
