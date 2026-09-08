"""Unit tests for the P1-R1A batch-shape sweep (pure, no GPU, no model).

Covers the benchmark-only in-memory batch-shape override, the canonical
global-batch derivation (fail-closed), the shape-invariant synthetic
sample population, and the sweep orchestration math (statuses, speedup,
drift, memory safety, candidate parsing, summary parsing, ranking).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sakuramoon.config import load_config
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.perf.batch_shape import (
    BASELINE_DRIFT_LIMIT_PCT,
    CANONICAL_SWEEP,
    SPEED_STRONG_PCT,
    SPEED_USEFUL_PCT,
    STATUS_ERROR,
    STATUS_OOM,
    STATUS_PASS,
    STATUS_SKIPPED_SAFETY_GATE,
    STATUS_TIMEOUT,
    BatchShapeCandidate,
    baseline_drift_pct,
    benchmark_batch_shape_config,
    classify_candidate_status,
    classify_memory_safety,
    high_candidate_gate_passed,
    is_environment_stable,
    logical_update_sample_count,
    median_pair,
    oom_marker_found,
    parse_shape_spec,
    rank_best_safe,
    speed_class,
    speedup_pct,
    summarize_candidate,
)
from sakuramoon.perf.synthetic import logical_update_identities

REPOSITORY_ROOT = Path(__file__).parents[3]

_PLACEHOLDER_ENVIRONMENT = {
    "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
    "WANDB_API_KEY": "synthetic-wandb-secret",
}


def _load_g1_config() -> RuntimeConfig:
    return load_config(
        Path("train_g1_cmuon_production.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment=dict(_PLACEHOLDER_ENVIRONMENT),
    ).config


@pytest.fixture(scope="module")
def g1_config() -> Iterator[RuntimeConfig]:
    """The CANONICAL 256px G1 production config (20x20x2 -> 800)."""

    yield _load_g1_config()


class TestCanonicalConfigPremise:
    def test_canonical_shape_is_20x20x2_global_800(self, g1_config) -> None:
        train = g1_config.train
        distributed = g1_config.distributed
        assert distributed.world_size == 2
        assert train.local_batch == 20
        assert train.accumulation == 20
        assert train.resolution == 256
        assert train.global_batch == 800
        assert logical_update_sample_count(train.local_batch, train.accumulation) == 400


class TestBenchmarkBatchShapeConfig:
    def test_candidate_shape_preserves_everything_except_shape(
        self, g1_config: RuntimeConfig
    ) -> None:
        derived = benchmark_batch_shape_config(
            g1_config, 25, 16, expected_global_batch=800
        )
        assert derived.train.local_batch == 25
        assert derived.train.accumulation == 16
        assert derived.train.global_batch == 800
        # Unchanged: topology, model, resolution, optimizer, LR rule.
        assert derived.distributed == g1_config.distributed
        assert derived.model == g1_config.model
        assert derived.optimizer == g1_config.optimizer
        assert derived.scheduler == g1_config.scheduler
        assert derived.train.resolution == g1_config.train.resolution
        assert derived.train.max_updates == g1_config.train.max_updates

    def test_all_canonical_factor_pairs_derive_800(self, g1_config) -> None:
        for local_batch, accumulation in ((16, 25), (20, 20), (25, 16), (40, 10)):
            derived = benchmark_batch_shape_config(
                g1_config, local_batch, accumulation, expected_global_batch=800
            )
            assert derived.train.global_batch == 800
            assert logical_update_sample_count(local_batch, accumulation) == 400

    def test_non_800_canonical_candidate_is_rejected(self, g1_config) -> None:
        # 20x10 x world 2 = 400 != 800: must fail closed.
        with pytest.raises(ValueError, match="global batch"):
            benchmark_batch_shape_config(g1_config, 20, 10, expected_global_batch=800)
        # 50x8 x world 2 = 800: allowed by the derivation itself
        # (out of the R1A matrix, but a legal factor pair).
        derived = benchmark_batch_shape_config(
            g1_config, 50, 8, expected_global_batch=800
        )
        assert derived.train.global_batch == 800
        # A wrong expected value is rejected even for the base shape.
        with pytest.raises(ValueError, match="expected 760"):
            benchmark_batch_shape_config(g1_config, 20, 20, expected_global_batch=760)

    def test_positive_int_validation(self, g1_config) -> None:
        with pytest.raises(ValueError, match="local_batch"):
            benchmark_batch_shape_config(g1_config, 0, 40)
        with pytest.raises(ValueError, match="local_batch"):
            benchmark_batch_shape_config(g1_config, -4, 40)
        with pytest.raises(ValueError, match="local_batch"):
            benchmark_batch_shape_config(g1_config, 20.0, 40)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="accumulation"):
            benchmark_batch_shape_config(g1_config, 20, 0)
        with pytest.raises(ValueError, match="accumulation"):
            benchmark_batch_shape_config(g1_config, 20, -1)
        with pytest.raises(ValueError, match="expected_global_batch"):
            benchmark_batch_shape_config(
                g1_config,
                20,
                20,
                expected_global_batch=800.0,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="expected_global_batch"):
            benchmark_batch_shape_config(g1_config, 20, 20, expected_global_batch=0)

    def test_noop_shape_returns_same_config(self, g1_config) -> None:
        assert (
            benchmark_batch_shape_config(g1_config, 20, 20, expected_global_batch=800)
            is g1_config
        )


class TestLogicalUpdateIdentities:
    CANDIDATES = ((16, 25), (20, 20), (25, 16), (40, 10))

    def test_same_population_for_every_candidate_factor_pair(self) -> None:
        # warmup 5 + measured 10 -> 15 total updates; the measured stage
        # covers updates 5..14.
        reference: dict[int, set[int]] | None = None
        for local_batch, accumulation in self.CANDIDATES:
            for update in range(15):
                identities = logical_update_identities(
                    local_batch, accumulation, update, total_updates=15
                )
                expected = tuple(range(update * 400, (update + 1) * 400))
                assert len(identities) == 400
                assert identities == expected
                if reference is None:
                    reference = {}
                if update not in reference:
                    reference[update] = set(identities)
                assert set(identities) == reference[update]
        assert reference is not None and len(reference) == 15

    def test_measured_window_identities_match_across_shapes(self) -> None:
        populations = {
            (local_batch, accumulation): tuple(
                sorted(
                    logical_update_identities(
                        local_batch, accumulation, update, total_updates=15
                    )
                )
                for update in range(5, 15)
            )
            for local_batch, accumulation in self.CANDIDATES
        }
        for key, value in populations.items():
            assert value == populations[(20, 20)], f"shape {key} differs"

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="local_batch"):
            logical_update_identities(0, 20, 0, 15)
        with pytest.raises(ValueError, match="accumulation"):
            logical_update_identities(20, 0, 0, 15)
        with pytest.raises(ValueError, match="update"):
            logical_update_identities(20, 20, -1, 15)
        with pytest.raises(ValueError, match="total_updates"):
            logical_update_identities(20, 20, 0, 0)


def _candidate_summary(
    *,
    p50: float = 15.0,
    samples_per_s: float = 53.0,
    peak_allocated: float = 36.0,
    peak_reserved: float = 45.0,
    measured_updates: int = 10,
    skew: float = 0.01,
) -> dict[str, Any]:
    """A minimal schema-2 candidate summary payload (field names exact)."""

    per_rank = {
        str(rank): {
            "peak_allocated_bytes": peak_allocated * 2**30,
            "peak_reserved_bytes": peak_reserved * 2**30,
            "final_allocated_bytes": 9.7 * 2**30,
            "final_reserved_bytes": peak_reserved * 2**30,
        }
        for rank in (0, 1)
    }
    return {
        "schema_version": 2,
        "world_size": 2,
        "measured_iterations": measured_updates,
        "warmup_iterations": 5,
        "per_rank_step_seconds": {
            str(rank): {
                "mean": p50,
                "p50": p50,
                "p90": p50,
                "p95": p50 * 1.01,
                "min": p50,
                "max": p50,
                "stddev": 0.0,
            }
            for rank in (0, 1)
        },
        "global_step_seconds": {
            "mean": p50,
            "p50": p50,
            "p90": p50,
            "p95": p50 * 1.01,
            "min": p50,
            "max": p50,
            "stddev": 0.0,
        },
        "global_samples_per_second": samples_per_s,
        "image_tokens_per_second": samples_per_s * 256,
        "text_tokens_per_second": samples_per_s * 128,
        "phase_seconds": {
            name: {"mean": value, "p50": value, "p95": value, "share_of_step": 0.1}
            for name, value in {
                "qwen": 1.0,
                "vae": 0.5,
                "dit_forward": 3.0,
                "backward": 4.0,
                "optimizer": 1.2,
            }.items()
        },
        "memory": {
            "per_rank": per_rank,
            "max_across_ranks": {
                "peak_allocated_bytes": peak_allocated * 2**30,
                "peak_reserved_bytes": peak_reserved * 2**30,
            },
        },
        "dit": {"max_across_ranks_tflops_per_second": 10.0},
        "rank_step_skew_pct": skew,
    }


class TestSweepMath:
    def test_median_pair(self) -> None:
        assert median_pair(1.0, 3.0) == 2.0
        assert median_pair(2.0, 2.0) == 2.0
        with pytest.raises(ValueError, match="positive"):
            median_pair(0.0, 1.0)

    def test_baseline_drift(self) -> None:
        assert baseline_drift_pct(15.0, 15.0) == 0.0
        assert baseline_drift_pct(15.0, 15.5) == pytest.approx(0.5 / 15.25 * 100.0)
        # The 2% gate boundary (inclusive).
        a, b = 15.0, 15.3
        assert baseline_drift_pct(a, b) < BASELINE_DRIFT_LIMIT_PCT
        assert is_environment_stable(a, b) is True
        a, b = 15.0, 15.304
        assert baseline_drift_pct(a, b) > BASELINE_DRIFT_LIMIT_PCT
        assert is_environment_stable(a, b) is False
        with pytest.raises(ValueError, match="positive"):
            baseline_drift_pct(-1.0, 15.0)

    def test_speedup(self) -> None:
        assert speedup_pct(15.0, 14.1) == pytest.approx(6.0)
        assert speedup_pct(15.0, 15.0) == 0.0
        assert speedup_pct(15.0, 16.0) == pytest.approx(-6.6667, rel=1e-4)
        with pytest.raises(ValueError, match="positive"):
            speedup_pct(0.0, 15.0)

    def test_speed_class_boundaries(self) -> None:
        assert speed_class(SPEED_USEFUL_PCT - 0.001) == "NEUTRAL"
        assert speed_class(SPEED_USEFUL_PCT) == "USEFUL"
        assert speed_class(SPEED_STRONG_PCT - 0.001) == "USEFUL"
        assert speed_class(SPEED_STRONG_PCT) == "STRONG"
        assert speed_class(-5.0) == "NEUTRAL"

    def test_memory_safety_classification(self) -> None:
        assert classify_memory_safety(54.0, 60.0) == "SAFE"
        assert classify_memory_safety(50.0, 58.0) == "SAFE"
        assert classify_memory_safety(54.001, 60.0) == "TIGHT"
        assert classify_memory_safety(54.0, 60.001) == "TIGHT"
        assert classify_memory_safety(30.0, 30.0, oom=True) == "UNSAFE"
        assert classify_memory_safety(30.0, 30.0, measured_updates=9) == "UNSAFE"

    def test_high_candidate_gate(self) -> None:
        assert high_candidate_gate_passed(48.0, 56.0) is True
        assert high_candidate_gate_passed(47.9, 55.9) is True
        assert high_candidate_gate_passed(48.0001, 56.0) is False
        assert high_candidate_gate_passed(48.0, 56.0001) is False

    def test_oom_marker(self) -> None:
        assert (
            oom_marker_found("torch.OutOfMemoryError: CUDA out of memory") is not None
        )
        assert oom_marker_found("RuntimeError: collective mismatch") is None

    def test_candidate_status(self) -> None:
        assert classify_candidate_status(0, "", True) == STATUS_PASS
        assert classify_candidate_status(0, "", False) == STATUS_ERROR
        assert (
            classify_candidate_status(
                1, "Traceback ... torch.OutOfMemoryError: out of memory", False
            )
            == STATUS_OOM
        )
        assert (
            classify_candidate_status(1, "Traceback ... RuntimeError: boom", False)
            == STATUS_ERROR
        )


class TestSweepCandidateParsing:
    def test_parse_shape_spec(self) -> None:
        assert parse_shape_spec("25:16") == (25, 16)
        assert parse_shape_spec(" 40:10 ") == (40, 10)
        for bad in ("25x16", "25:16:2", ":16", "25:", "", "a:b"):
            with pytest.raises(ValueError, match="shape spec"):
                parse_shape_spec(bad)
        with pytest.raises(ValueError, match="positive"):
            parse_shape_spec("0:10")

    def test_canonical_sweep_table(self) -> None:
        assert [c.label for c in CANONICAL_SWEEP] == ["A0", "S", "C1", "C2", "A1"]
        assert [(c.local_batch, c.accumulation) for c in CANONICAL_SWEEP] == [
            (20, 20),
            (16, 25),
            (25, 16),
            (40, 10),
            (20, 20),
        ]
        dir_names = [c.dir_name for c in CANONICAL_SWEEP]
        assert len(set(dir_names)) == 5  # the two 20x20 slots are distinct


class TestSummarizeCandidateAndSerialization:
    def test_summarize_candidate_extracts_report_fields(self) -> None:
        payload = _candidate_summary(
            p50=15.636, samples_per_s=51.26, peak_allocated=36.12, peak_reserved=45.7
        )
        summary = summarize_candidate(payload)
        assert summary["p50_s"] == 15.636
        assert summary["p95_s"] == pytest.approx(15.636 * 1.01)
        assert summary["samples_per_s"] == 51.26
        assert summary["measured_updates"] == 10
        assert summary["world_size"] == 2
        assert summary["rank_skew_pct"] == 0.01
        assert summary["peak_allocated_gib"] == {
            0: pytest.approx(36.12),
            1: pytest.approx(36.12),
        }
        assert summary["max_peak_reserved_gib"] == pytest.approx(45.7)
        assert summary["phases_s"]["qwen"] == 1.0
        assert summary["phases_s"]["dit_forward"] == 3.0

    def test_status_serialization_roundtrip(self) -> None:
        entries = {
            "lb20-acc20-a": {"status": STATUS_PASS},
            "lb16-acc25": {"status": STATUS_OOM},
            "lb25-acc16": {"status": STATUS_SKIPPED_SAFETY_GATE},
            "lb40-acc10": {"status": STATUS_TIMEOUT},
            "lb20-acc20-b": {"status": STATUS_ERROR},
        }
        payload = json.loads(json.dumps(entries))
        assert payload == entries
        assert all(
            value["status"]
            in (
                STATUS_PASS,
                STATUS_OOM,
                STATUS_SKIPPED_SAFETY_GATE,
                STATUS_TIMEOUT,
                STATUS_ERROR,
            )
            for value in entries.values()
        )


class TestRankBestSafe:
    def _entry(self, samples_per_s: float, p50: float = 15.0) -> dict[str, Any]:
        return {
            "samples_per_s": samples_per_s,
            "p50_s": p50,
            "p95_s": p50 * 1.01,
            "max_peak_allocated_gib": 36.0,
            "rank_skew_pct": 0.01,
        }

    def test_highest_throughput_wins(self) -> None:
        entries = {
            "lb25-acc16": self._entry(51.0),
            "lb40-acc10": self._entry(53.5),
            "lb20-acc20-a": self._entry(51.2),
        }
        pick = rank_best_safe(entries)
        assert pick is not None and pick[0] == "lb40-acc10"

    def test_tie_breaks_on_p50(self) -> None:
        entries = {
            "a": self._entry(51.0, p50=15.2),
            "b": self._entry(51.0, p50=15.0),
        }
        pick = rank_best_safe(entries)
        assert pick is not None and pick[0] == "b"

    def test_empty_is_none(self) -> None:
        assert rank_best_safe({}) is None

    def test_candidate_dataclass_is_immutable(self) -> None:
        candidate = BatchShapeCandidate("X", 10, 40, "lb10-acc40")
        with pytest.raises(AttributeError):
            candidate.local_batch = 5  # type: ignore[misc]
