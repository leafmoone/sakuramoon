"""Unit tests for P0 performance samples and summary aggregation.

Covers the measurement-contract closure: true per-update peak memory
fields (schema 2), aligned distributed global-step semantics
(slowest-rank per update, fail-closed identity checks), and actual
summed cross-rank volume rates.
"""

from __future__ import annotations

import json

import pytest

from sakuramoon.perf.sample import (
    PERFORMANCE_SCHEMA_VERSION,
    PerformanceSample,
)
from sakuramoon.perf.summary import (
    SUMMARY_SCHEMA_VERSION,
    percentile,
    rank_step_skew_pct,
    step_stats,
    summarize,
)

PHASES = {
    "h2d": 0.05,
    "qwen": 1.0,
    "vae": 0.5,
    "dit_forward": 3.0,
    "loss": 0.2,
    "backward": 4.0,
    "optimizer": 1.2,
}


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
    peak_allocated: int = 3 << 30,
    peak_reserved: int = 4 << 30,
) -> PerformanceSample:
    if phases is None:
        phases = dict(PHASES)
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
        peak_memory_allocated_bytes=peak_allocated,
        peak_memory_reserved_bytes=peak_reserved,
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
        with pytest.raises(ValueError, match="peak_memory_allocated_bytes"):
            _sample(peak_allocated=-1)

    def test_rejects_peak_below_final_allocation(self) -> None:
        # A window peak can never be smaller than the current (final)
        # allocation of that same window.
        with pytest.raises(ValueError, match="peak_memory_allocated_bytes"):
            _sample(allocated=3 << 30, peak_allocated=1 << 30)
        with pytest.raises(ValueError, match="peak_memory_reserved_bytes"):
            _sample(reserved=5 << 30, peak_reserved=1 << 30)

    def test_peak_equal_to_final_is_allowed(self) -> None:
        sample = _sample(allocated=1 << 30, peak_allocated=1 << 30)
        assert sample.peak_memory_allocated_bytes == sample.memory_allocated_bytes

    def test_empty_phases_are_allowed(self) -> None:
        sample = _sample(phases={})
        assert sample.phases == {}


class TestJsonRoundTrip:
    def test_round_trip_is_exact_including_peak_memory(self) -> None:
        original = _sample(
            update=7,
            wall=12.25,
            peak_allocated=5 << 30,
            peak_reserved=6 << 30,
        )
        payload = original.to_dict()
        assert payload["schema_version"] == PERFORMANCE_SCHEMA_VERSION == 2
        assert payload["peak_memory_allocated_bytes"] == 5 << 30
        assert payload["peak_memory_reserved_bytes"] == 6 << 30
        encoded = json.dumps(payload)
        restored = PerformanceSample.from_dict(json.loads(encoded))
        assert restored == original
        assert restored.to_dict() == payload

    def test_round_trip_preserves_phase_order_independence(self) -> None:
        original = _sample()
        restored = PerformanceSample.from_dict(original.to_dict())
        assert set(restored.phases) == set(original.phases)

    def test_from_dict_rejects_schema_version_one_payload(self) -> None:
        payload = _sample().to_dict()
        legacy = {
            key: value
            for key, value in payload.items()
            if key
            not in (
                "peak_memory_allocated_bytes",
                "peak_memory_reserved_bytes",
            )
        }
        legacy["schema_version"] = 1
        with pytest.raises(ValueError):
            PerformanceSample.from_dict(legacy)

    def test_from_dict_rejects_extra_keys(self) -> None:
        payload = _sample().to_dict()
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="keys differ"):
            PerformanceSample.from_dict(payload)


class TestStepStats:
    def test_percentile_linear_interpolation(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]
        assert percentile(values, 0.0) == 10.0
        assert percentile(values, 100.0) == 40.0
        assert percentile(values, 50.0) == 25.0
        assert percentile(values, 90.0) == 37.0

    def test_step_stats_block(self) -> None:
        stats = step_stats([10.0, 20.0])
        assert stats["mean"] == 15.0
        assert stats["p50"] == 15.0
        assert stats["min"] == 10.0
        assert stats["max"] == 20.0

    def test_step_stats_single_sample(self) -> None:
        stats = step_stats([7.5])
        assert stats["mean"] == 7.5
        assert stats["stddev"] == 0.0

    def test_step_stats_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            step_stats([])

    def test_rank_skew(self) -> None:
        assert rank_step_skew_pct({0: 15.0}) == 0.0
        assert rank_step_skew_pct({0: 10.0, 1: 20.0}) == pytest.approx(50.0)
        with pytest.raises(ValueError, match="at least one rank"):
            rank_step_skew_pct({})


class TestAlignedGlobalStep:
    """§3: per-update slowest-rank walls, fail-closed identity checks."""

    def test_alternating_straggler_is_not_max_of_means(self) -> None:
        # rank0 [10, 20], rank1 [20, 10]:
        # aligned global walls = [20, 20] -> mean 20, NOT the pooled
        # distribution mean (15) and NOT max-of-means confusion.
        rank0 = [_sample(update=1, wall=10.0), _sample(update=2, wall=20.0)]
        rank1 = [_sample(update=1, wall=20.0), _sample(update=2, wall=10.0)]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        assert summary.global_step_seconds["mean"] == pytest.approx(20.0)
        assert summary.global_step_seconds["p50"] == pytest.approx(20.0)
        assert summary.global_step_seconds["min"] == pytest.approx(20.0)
        assert summary.global_step_seconds["max"] == pytest.approx(20.0)
        assert summary.global_step_seconds["mean"] != pytest.approx(15.0)
        # Rank-local distributions are preserved as-is.
        assert summary.per_rank_step_seconds[0]["mean"] == pytest.approx(15.0)
        assert summary.per_rank_step_seconds[1]["mean"] == pytest.approx(15.0)

    def test_global_wall_is_max_per_update(self) -> None:
        rank0 = [_sample(update=1, wall=5.0), _sample(update=2, wall=9.0)]
        rank1 = [_sample(update=1, wall=7.0), _sample(update=2, wall=3.0)]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        assert summary.global_step_seconds["min"] == pytest.approx(7.0)
        assert summary.global_step_seconds["max"] == pytest.approx(9.0)

    def test_single_rank_global_equals_rank_local(self) -> None:
        rank0 = [_sample(update=1, wall=4.0), _sample(update=2, wall=6.0)]
        summary = summarize({0: rank0}, warmup_iterations=0, world_size=1)
        assert summary.global_step_seconds == summary.per_rank_step_seconds[0]

    def test_missing_update_fails_closed(self) -> None:
        rank0 = [_sample(update=1, wall=5.0), _sample(update=2, wall=7.0)]
        rank1 = [_sample(update=2, wall=7.0)]
        with pytest.raises(ValueError, match="identities differ"):
            summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)

    def test_duplicate_update_fails_closed(self) -> None:
        rank0 = [_sample(update=1, wall=5.0)]
        rank1 = [
            _sample(update=1, wall=5.0),
            _sample(update=1, wall=6.0),
        ]
        with pytest.raises(ValueError, match="duplicate update"):
            summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)

    def test_extra_update_fails_closed(self) -> None:
        rank0 = [_sample(update=1, wall=5.0)]
        rank1 = [
            _sample(update=1, wall=5.0),
            _sample(update=2, wall=6.0),
        ]
        with pytest.raises(ValueError, match="identities differ"):
            summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)

    def test_empty_rank_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="at least one rank"):
            summarize({}, warmup_iterations=5, world_size=1)

    def test_ranks_in_any_order_align_by_update_id(self) -> None:
        # Feed ranks with shuffled per-rank ordering; alignment is by id.
        rank0 = [_sample(update=2, wall=6.0), _sample(update=1, wall=4.0)]
        rank1 = [_sample(update=1, wall=5.0), _sample(update=2, wall=3.0)]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        assert summary.measured_iterations == 2
        # Aligned walls: update1 max(4,5)=5, update2 max(6,3)=6.
        assert summary.global_step_seconds["mean"] == pytest.approx(5.5)


class TestCrossRankVolumes:
    """§4: actual summed cross-rank volumes; unequal rank tokens are legal."""

    def test_differing_rank_text_tokens_aggregate_exactly(self) -> None:
        # rank0: 8000 text tokens/update, rank1: 12000/update (unequal!),
        # aligned walls: update1 max(10,20)=20, update2 max(20,10)=20.
        rank0 = [
            _sample(update=1, wall=10.0, text_tokens=8000),
            _sample(update=2, wall=20.0, text_tokens=8000),
        ]
        rank1 = [
            _sample(update=1, wall=20.0, text_tokens=12000),
            _sample(update=2, wall=10.0, text_tokens=12000),
        ]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        # Exact: (8000+12000) * 2 updates / (20 + 20) seconds.
        assert summary.text_tokens_per_second == pytest.approx(20_000 * 2 / 40.0)
        # NOT the invalid rank0 * world_size formulation:
        assert summary.text_tokens_per_second != pytest.approx(8000 * 2 * 2 / 40.0)

    def test_samples_and_image_tokens_use_actual_sums(self) -> None:
        rank0 = [
            _sample(update=1, wall=10.0, samples=40, image_tokens=10240),
        ]
        rank1 = [
            _sample(update=1, wall=12.0, samples=36, image_tokens=9216),
        ]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        # Global wall = 12 (slowest rank); volumes are actual sums.
        assert summary.global_samples_per_second == pytest.approx((40 + 36) / 12.0)
        assert summary.image_tokens_per_second == pytest.approx((10240 + 9216) / 12.0)

    def test_throughput_denominator_is_summed_aligned_walls(self) -> None:
        rank0 = [
            _sample(update=1, wall=10.0, samples=40),
            _sample(update=2, wall=20.0, samples=40),
        ]
        summary = summarize({0: rank0}, warmup_iterations=0, world_size=1)
        assert summary.global_samples_per_second == pytest.approx(80 / 30.0)


class TestPhaseShareDenominator:
    def test_share_uses_mean_aligned_global_wall(self) -> None:
        # Aligned walls: max(10,20)=20 and max(20,10)=20 -> mean 20.
        rank0 = [_sample(update=1, wall=10.0), _sample(update=2, wall=20.0)]
        rank1 = [_sample(update=1, wall=20.0), _sample(update=2, wall=10.0)]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        dit = summary.phase_seconds["dit_forward"]
        dit_share = dit["share_of_step"]
        dit_mean = dit["mean"]
        assert dit_share is not None
        assert dit_mean is not None
        assert dit_share == pytest.approx(dit_mean / 20.0)

    def test_share_is_none_when_phase_not_measured_everywhere(self) -> None:
        rank0 = [_sample(update=1, wall=10.0)]
        rank1 = [
            _sample(
                update=1,
                wall=12.0,
                phases={
                    name: seconds for name, seconds in PHASES.items() if name != "qwen"
                },
            ),
        ]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        assert summary.phase_seconds["qwen"]["share_of_step"] is None


class TestMemoryBlock:
    def test_memory_block_reports_true_peaks_and_finals(self) -> None:
        rank0 = [
            _sample(
                update=1,
                allocated=1 << 30,
                reserved=2 << 30,
                peak_allocated=3 << 30,
                peak_reserved=4 << 30,
            ),
            _sample(
                update=2,
                allocated=2 << 30,
                reserved=3 << 30,
                peak_allocated=5 << 30,
                peak_reserved=6 << 30,
            ),
        ]
        rank1 = [
            _sample(
                update=1,
                allocated=1 << 30,
                reserved=2 << 30,
                peak_allocated=9 << 30,
                peak_reserved=8 << 30,
            ),
            _sample(
                update=2,
                allocated=1 << 30,
                reserved=2 << 30,
                peak_allocated=7 << 30,
                peak_reserved=10 << 30,
            ),
        ]
        summary = summarize({0: rank0, 1: rank1}, warmup_iterations=0, world_size=2)
        memory = summary.memory
        assert memory["per_rank"][0] == {
            "peak_allocated_bytes": 5 << 30,
            "peak_reserved_bytes": 6 << 30,
            "final_allocated_bytes": 2 << 30,
            "final_reserved_bytes": 3 << 30,
        }
        assert memory["per_rank"][1] == {
            "peak_allocated_bytes": 9 << 30,
            "peak_reserved_bytes": 10 << 30,
            "final_allocated_bytes": 1 << 30,
            "final_reserved_bytes": 2 << 30,
        }
        assert memory["max_across_ranks"] == {
            "peak_allocated_bytes": 9 << 30,
            "peak_reserved_bytes": 10 << 30,
        }

    def test_to_dict_is_json_serializable_and_schema_two(self) -> None:
        summary = summarize({0: [_sample()]}, warmup_iterations=5, world_size=1)
        payload = summary.to_dict()
        assert payload["schema_version"] == SUMMARY_SCHEMA_VERSION == 2
        encoded = json.dumps(payload, sort_keys=True)
        assert "peak_allocated_bytes" in encoded
        assert "final_allocated_bytes" in encoded


class TestSummaryValidation:
    def test_invalid_world_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            summarize({0: [_sample()]}, warmup_iterations=0, world_size=0)

    def test_invalid_warmup_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup_iterations"):
            summarize({0: [_sample()]}, warmup_iterations=-1, world_size=1)
