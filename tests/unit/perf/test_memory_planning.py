"""Focused tests for the R1C memory-planning pure core."""

from __future__ import annotations

import math

import pytest

from sakuramoon.perf.memory_planning import (
    CANDIDATE,
    DEFAULT_MEMORY_POOL,
    MEMORY_PLANNING_ENV,
    MEMORY_POOL_ENV,
    REJECT,
    STRONG,
    MemoryPlanningDecision,
    build_memory_planning_env,
    census_record_to_dict,
    classify_improvement,
    decide_memory_planning,
    environment_drift_pct,
    environment_stable,
    improvement_pct,
    memory_planning_enabled,
    parse_memory_planning_value,
    reference_p50,
)


def test_parse_memory_planning_value_accepts_common_forms() -> None:
    assert parse_memory_planning_value("1") is True
    assert parse_memory_planning_value("TRUE") is True
    assert parse_memory_planning_value(" yes ") is True
    assert parse_memory_planning_value("on") is True
    assert parse_memory_planning_value("0") is False
    assert parse_memory_planning_value("false") is False
    assert parse_memory_planning_value("No") is False
    assert parse_memory_planning_value("off") is False


def test_parse_memory_planning_value_fails_closed_on_garbage() -> None:
    for raw in ("", "2", "enabled", "intermediates", "true1"):
        with pytest.raises(ValueError, match="invalid"):
            parse_memory_planning_value(raw)


def test_memory_planning_enabled_defaults_off_when_unset() -> None:
    assert memory_planning_enabled({}) is False
    assert memory_planning_enabled({MEMORY_PLANNING_ENV: ""}) is False
    assert memory_planning_enabled({MEMORY_PLANNING_ENV: "0"}) is False
    assert memory_planning_enabled({MEMORY_PLANNING_ENV: "1"}) is True


def test_build_env_disabled_is_reference_mode_no_keys_added() -> None:
    base = {"PATH": "/usr/bin"}
    env = build_memory_planning_env(enabled=False, base_env=base)
    assert env == base
    assert MEMORY_PLANNING_ENV not in env
    assert MEMORY_POOL_ENV not in env


def test_build_env_enabled_sets_flag_and_default_pool() -> None:
    env = build_memory_planning_env(enabled=True)
    assert env[MEMORY_PLANNING_ENV] == "1"
    assert env[MEMORY_POOL_ENV] == DEFAULT_MEMORY_POOL


def test_build_env_explicit_contradiction_fails_closed() -> None:
    base = {MEMORY_PLANNING_ENV: "1"}
    with pytest.raises(ValueError, match="contradicts"):
        build_memory_planning_env(enabled=False, base_env=base)
    base = {MEMORY_PLANNING_ENV: "0"}
    with pytest.raises(ValueError, match="contradicts"):
        build_memory_planning_env(enabled=True, base_env=base)


def test_build_env_non_default_pool_fails_closed() -> None:
    base = {MEMORY_POOL_ENV: "activation"}
    with pytest.raises(ValueError, match="must not sweep"):
        build_memory_planning_env(enabled=True, base_env=base)
    # Disabled mode never touches the pool.
    env = build_memory_planning_env(enabled=False, base_env=base)
    assert env[MEMORY_POOL_ENV] == "activation"


def test_reference_p50_is_mean_and_validates() -> None:
    assert reference_p50(10.0, 12.0) == pytest.approx(11.0)
    with pytest.raises(ValueError):
        reference_p50(math.nan, 10.0)
    with pytest.raises(ValueError):
        reference_p50(10.0, math.inf)


def test_environment_drift_and_stability_gate() -> None:
    drift = environment_drift_pct(10.0, 10.2)
    assert drift == pytest.approx(1.9801980198, rel=1e-6)
    assert environment_stable(10.0, 10.2) is True
    assert environment_stable(10.0, 10.3) is False
    with pytest.raises(ValueError):
        environment_drift_pct(0.0, 1.0)


def test_improvement_pct_sign_and_validity() -> None:
    assert improvement_pct(100.0, 98.0) == pytest.approx(2.0)
    assert improvement_pct(100.0, 102.0) == pytest.approx(-2.0)
    with pytest.raises(ValueError):
        improvement_pct(0.0, 1.0)
    with pytest.raises(ValueError):
        improvement_pct(math.nan, 1.0)


def test_classify_improvement_boundaries() -> None:
    assert classify_improvement(0.999) == REJECT
    assert classify_improvement(1.0) == CANDIDATE
    assert classify_improvement(1.999) == CANDIDATE
    assert classify_improvement(2.0) == STRONG
    assert classify_improvement(-5.0) == REJECT
    with pytest.raises(ValueError):
        classify_improvement(math.nan)


def _decision_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "m0_p50_s": 14.7,
        "m1_p50_s": 14.75,
        "reference_p50_s": 15.0,
        "m0_p95_s": 15.4,
        "reference_p95_s": 15.5,
        "m0_peak_allocated_gib": 36.0,
        "reference_peak_allocated_gib": 36.5,
        "m0_peak_reserved_gib": 40.0,
        "reference_peak_reserved_gib": 40.5,
    }
    base.update(overrides)
    return base


def test_decide_accepts_strong_candidate_with_clean_memory() -> None:
    decision = decide_memory_planning(**_decision_kwargs())  # type: ignore[arg-type]
    assert isinstance(decision, MemoryPlanningDecision)
    assert decision.accepted is True
    assert decision.improvement_class == STRONG
    assert decision.reasons == ()
    # median of (2.0%, 1.667%) -> the upper half for two samples
    assert decision.median_improvement_pct == pytest.approx(2.0)


def test_decide_rejects_sub_threshold_improvement() -> None:
    decision = decide_memory_planning(
        **_decision_kwargs(m0_p50_s=14.99, m1_p50_s=14.98)  # type: ignore[arg-type]
    )
    assert decision.accepted is False
    assert decision.improvement_class == REJECT


def test_decide_requires_m1_repeat_for_candidate_threshold() -> None:
    decision = decide_memory_planning(
        **_decision_kwargs(m0_p50_s=14.7, m1_p50_s=None)  # type: ignore[arg-type]
    )
    assert decision.accepted is False
    assert any("M1 repeat missing" in r for r in decision.reasons)


def test_decide_rejects_p95_regression() -> None:
    decision = decide_memory_planning(
        **_decision_kwargs(m0_p50_s=14.7, m0_p95_s=15.9)  # type: ignore[arg-type]  # p95 >2% above 15.5
    )
    assert decision.accepted is False
    assert any("p95 regression" in r for r in decision.reasons)


def test_decide_rejects_memory_regression_above_5_percent() -> None:
    decision = decide_memory_planning(
        **_decision_kwargs(
            m0_p50_s=14.7,  # type: ignore[arg-type]
            m0_peak_allocated_gib=38.5,  # ~5.48% above 36.5
            reference_peak_allocated_gib=36.5,
        )
    )
    assert decision.accepted is False
    assert any("peak_allocated" in r for r in decision.reasons)


def test_decide_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        decide_memory_planning(**_decision_kwargs(m0_p50_s=math.nan))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        decide_memory_planning(**_decision_kwargs(m1_p50_s=math.nan))  # type: ignore[arg-type]


def _census_record() -> dict[str, object]:
    return {
        "stage_base": "deadbeef",
        "workload": {"config": "train_g1_cmuon_production.toml"},
        "candidates": [
            {
                "id": "F1",
                "path": "src/sakuramoon/model/block.py",
                "op": "index_select + cast",
                "frequency": "per block",
                "classification": "K1 reserved",
            }
        ],
        "r1c_b_conclusion": "NONE in-scope",
    }


def test_census_record_round_trip() -> None:
    record = census_record_to_dict(_census_record())
    assert record["stage_base"] == "deadbeef"
    assert isinstance(record["candidates"], list)


def test_census_record_validation() -> None:
    record = _census_record()
    del record["r1c_b_conclusion"]
    with pytest.raises(ValueError, match="r1c_b_conclusion"):
        census_record_to_dict(record)
    record = _census_record()
    entry = record["candidates"][0]
    assert isinstance(entry, dict)
    del entry["op"]
    with pytest.raises(ValueError, match="op"):
        census_record_to_dict(record)
    with pytest.raises(TypeError):
        census_record_to_dict(["not", "a", "mapping"])  # type: ignore[arg-type]
