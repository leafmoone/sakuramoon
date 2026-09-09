"""Memory-planning experiment support for the packed training benchmark.

Pure (torch-free) core used by the P1-R1C allocation round:

* environment construction for the existing Inductor memory planner flag
  (``TORCHINDUCTOR_MEMORY_PLANNING``), default-off / reference mode,
  fail-closed validation of explicit values;
* environment-bracket drift math (A0/A1 steady p50 control comparison);
* candidate improvement classification (REJECT / CANDIDATE / STRONG) and the
  R1C acceptance decision (GO section 12);
* allocation-census record validation/serialization.

No benchmark code and no production runtime code imports torch here; every
function is deterministic and unit-testable without a device.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

MEMORY_PLANNING_ENV = "TORCHINDUCTOR_MEMORY_PLANNING"
MEMORY_POOL_ENV = "TORCHINDUCTOR_MEMORY_POOL"
DEFAULT_MEMORY_POOL = "intermediates"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

REJECT = "REJECT"
CANDIDATE = "CANDIDATE"
STRONG = "STRONG"

#: Whole-step p50 improvement (percent) at/above which a candidate becomes a
#: CANDIDATE (GO section 12).
CANDIDATE_IMPROVEMENT_PCT = 1.0
#: Whole-step p50 improvement (percent) at/above which a candidate is STRONG.
STRONG_IMPROVEMENT_PCT = 2.0
#: A0/A1 steady-p50 drift (percent) at/above which the environment is not
#: considered stable for a performance verdict (GO section 11).
ENVIRONMENT_STABLE_MAX_PCT = 2.0
#: Memory regression limit (percent above baseline) for acceptance.
MEMORY_REGRESSION_LIMIT_PCT = 5.0
#: p95 regression limit (percent above baseline) for acceptance.
P95_REGRESSION_LIMIT_PCT = 2.0

_CENSUS_REQUIRED_KEYS = (
    "stage_base",
    "workload",
    "candidates",
    "r1c_b_conclusion",
)
_CANDIDATE_REQUIRED_KEYS = (
    "id",
    "path",
    "op",
    "frequency",
    "classification",
)


def parse_memory_planning_value(raw: str) -> bool:
    """Parse an explicit memory-planning env value; fail closed on garbage."""

    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"invalid {MEMORY_PLANNING_ENV} value: {raw!r} "
        "(expected one of 1/0, true/false, yes/no, on/off)"
    )


def memory_planning_enabled(
    env: Mapping[str, str],
) -> bool:
    """Effective flag from an environment mapping. Unset -> False (default-off)."""

    raw = env.get(MEMORY_PLANNING_ENV)
    if raw is None or raw.strip() == "":
        return False
    return parse_memory_planning_value(raw)


def build_memory_planning_env(
    *,
    enabled: bool,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the benchmark-subprocess env for one candidate.

    * disabled (reference mode): the flag is left UNSET so the run is
      bit-identical to the unmodified default behavior;
    * enabled: the flag is set to "1" and the memory pool is pinned to the
      PyTorch default ("intermediates") explicitly, so the experiment is
      self-documenting and reproducible;
    * an explicit pre-existing value that contradicts ``enabled`` raises
      (fail closed) instead of being silently overwritten.
    """

    env: dict[str, str] = dict(base_env or {})
    existing = env.get(MEMORY_PLANNING_ENV)
    if (
        existing is not None
        and existing.strip() != ""
        and memory_planning_enabled(env) != enabled
    ):
        raise ValueError(
            f"{MEMORY_PLANNING_ENV}={existing!r} contradicts "
            f"enabled={enabled}; remove it or align the flag"
        )
    if enabled:
        env[MEMORY_PLANNING_ENV] = "1"
        pool = env.get(MEMORY_POOL_ENV, DEFAULT_MEMORY_POOL)
        if pool != DEFAULT_MEMORY_POOL:
            raise ValueError(
                f"{MEMORY_POOL_ENV}={pool!r} is not the default "
                f"{DEFAULT_MEMORY_POOL!r}; the R1C experiment must not sweep "
                "memory-pool modes"
            )
        env[MEMORY_POOL_ENV] = DEFAULT_MEMORY_POOL
    return env


def reference_p50(a0_p50_s: float, a1_p50_s: float) -> float:
    """Baseline reference: the MEAN of the two same-session controls (A0/A1).

    The mean (not the min) is used deliberately: no faster-control
    cherry-picking (GO section 11).
    """

    for name, value in (("a0", a0_p50_s), ("a1", a1_p50_s)):
        if not _finite(value):
            raise ValueError(f"{name}_p50_s must be finite, got {value!r}")
    return (a0_p50_s + a1_p50_s) / 2.0


def environment_drift_pct(a0_p50_s: float, a1_p50_s: float) -> float:
    """Relative spread between the two controls, in percent (always >= 0)."""

    for name, value in (("a0", a0_p50_s), ("a1", a1_p50_s)):
        if not _finite(value) or value <= 0.0:
            raise ValueError(f"{name}_p50_s must be finite and positive")
    reference = (a0_p50_s + a1_p50_s) / 2.0
    return abs(a0_p50_s - a1_p50_s) / reference * 100.0


def environment_stable(
    a0_p50_s: float,
    a1_p50_s: float,
    max_pct: float = ENVIRONMENT_STABLE_MAX_PCT,
) -> bool:
    return environment_drift_pct(a0_p50_s, a1_p50_s) <= max_pct


def improvement_pct(baseline_p50_s: float, candidate_p50_s: float) -> float:
    """Whole-step p50 improvement in percent (positive = candidate faster)."""

    if not _finite(baseline_p50_s) or not _finite(candidate_p50_s):
        raise ValueError("p50 values must be finite")
    if baseline_p50_s <= 0.0:
        raise ValueError("baseline p50 must be positive")
    return (baseline_p50_s - candidate_p50_s) / baseline_p50_s * 100.0


def classify_improvement(improvement: float) -> str:
    """REJECT (<1%), CANDIDATE (>=1%), STRONG (>=2%) (GO section 12)."""

    if not _finite(improvement):
        raise ValueError(f"improvement must be finite, got {improvement!r}")
    if improvement >= STRONG_IMPROVEMENT_PCT:
        return STRONG
    if improvement >= CANDIDATE_IMPROVEMENT_PCT:
        return CANDIDATE
    return REJECT


@dataclass(frozen=True)
class MemoryPlanningDecision:
    accepted: bool
    improvement_class: str
    median_improvement_pct: float
    reasons: tuple[str, ...]


def decide_memory_planning(
    *,
    m0_p50_s: float,
    m1_p50_s: float | None,
    reference_p50_s: float,
    m0_p95_s: float,
    reference_p95_s: float,
    m0_peak_allocated_gib: float,
    reference_peak_allocated_gib: float,
    m0_peak_reserved_gib: float,
    reference_peak_reserved_gib: float,
) -> MemoryPlanningDecision:
    """Apply the GO section 12 acceptance rules to one measured candidate.

    ``m1_p50_s`` is None when the M1 repeat has not been run (required as soon
    as M0 improvement reaches the candidate threshold).
    """

    reasons: list[str] = []
    m0_improvement = improvement_pct(reference_p50_s, m0_p50_s)
    class_ = classify_improvement(m0_improvement)
    candidates = (m0_p50_s,) + ((m1_p50_s,) if m1_p50_s is not None else ())
    if class_ != REJECT and m1_p50_s is None:
        reasons.append("M1 repeat missing although M0 reached the candidate threshold")
    median_improvement = max(
        0.0,
        sorted(improvement_pct(reference_p50_s, value) for value in candidates)[
            len(candidates) // 2
        ],
    )
    if median_improvement < CANDIDATE_IMPROVEMENT_PCT:
        reasons.append(
            f"median improvement {median_improvement:.3f}% < "
            f"{CANDIDATE_IMPROVEMENT_PCT:.1f}%"
        )
    if not _finite(m0_p95_s) or not _finite(reference_p95_s):
        reasons.append("p95 values must be finite")
    elif reference_p95_s > 0.0:
        p95_delta_pct = (m0_p95_s - reference_p95_s) / reference_p95_s * 100.0
        if p95_delta_pct > P95_REGRESSION_LIMIT_PCT:
            reasons.append(
                f"p95 regression {p95_delta_pct:.3f}% > {P95_REGRESSION_LIMIT_PCT:.1f}%"
            )
    for name, value, baseline in (
        ("peak_allocated", m0_peak_allocated_gib, reference_peak_allocated_gib),
        ("peak_reserved", m0_peak_reserved_gib, reference_peak_reserved_gib),
    ):
        if not _finite(value) or not _finite(baseline) or baseline <= 0.0:
            reasons.append(f"{name} values must be finite and baseline positive")
            continue
        delta_pct = (value - baseline) / baseline * 100.0
        if delta_pct > MEMORY_REGRESSION_LIMIT_PCT:
            reasons.append(
                f"{name} regression {delta_pct:.3f}% > "
                f"{MEMORY_REGRESSION_LIMIT_PCT:.1f}%"
            )
    accepted = class_ != REJECT and m1_p50_s is not None and not reasons
    return MemoryPlanningDecision(
        accepted=accepted,
        improvement_class=class_,
        median_improvement_pct=median_improvement,
        reasons=tuple(reasons),
    )


def census_record_to_dict(record: object) -> dict[str, object]:
    """Validate and return a plain-dict copy of an allocation-census record."""

    if not isinstance(record, Mapping):
        raise TypeError("census record must be a mapping")
    rec = cast("Mapping[str, object]", record)
    for key in _CENSUS_REQUIRED_KEYS:
        if key not in rec:
            raise ValueError(f"census record is missing required key {key!r}")
    raw_candidates = rec["candidates"]
    if not isinstance(raw_candidates, (list, tuple)):
        raise TypeError("census 'candidates' must be a list")
    # Typed boundary: the census builder always emits Mapping entries here.
    candidates = cast("list[Mapping[str, object]]", raw_candidates)
    for entry in candidates:
        for key in _CANDIDATE_REQUIRED_KEYS:
            if key not in entry:
                raise ValueError(f"census candidate is missing required key {key!r}")
    return dict(rec)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


__all__ = [
    "CANDIDATE",
    "CANDIDATE_IMPROVEMENT_PCT",
    "DEFAULT_MEMORY_POOL",
    "ENVIRONMENT_STABLE_MAX_PCT",
    "MEMORY_PLANNING_ENV",
    "MEMORY_POOL_ENV",
    "MEMORY_REGRESSION_LIMIT_PCT",
    "P95_REGRESSION_LIMIT_PCT",
    "REJECT",
    "STRONG",
    "STRONG_IMPROVEMENT_PCT",
    "MemoryPlanningDecision",
    "build_memory_planning_env",
    "census_record_to_dict",
    "classify_improvement",
    "decide_memory_planning",
    "environment_drift_pct",
    "environment_stable",
    "improvement_pct",
    "memory_planning_enabled",
    "parse_memory_planning_value",
    "reference_p50",
]
