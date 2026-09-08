"""Opt-in performance observatory (P0 baseline measurement).

Standalone measurement code for the canonical training-performance
baseline.  Nothing in this package is imported by the production training
hot path; production timing keeps using :mod:`sakuramoon.telemetry`
unchanged.  See ``docs/performance-benchmarking.md``.
"""

from sakuramoon.perf.fingerprint import (
    SCHEMA_VERSION,
    RuntimeFingerprint,
    capture_runtime_fingerprint,
)
from sakuramoon.perf.sample import PERFORMANCE_SCHEMA_VERSION, PerformanceSample
from sakuramoon.perf.summary import (
    SUMMARY_SCHEMA_VERSION,
    PerformanceSummary,
    percentile,
    rank_step_skew_pct,
    summarize,
)

__all__ = [
    "PERFORMANCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "PerformanceSample",
    "PerformanceSummary",
    "RuntimeFingerprint",
    "capture_runtime_fingerprint",
    "percentile",
    "rank_step_skew_pct",
    "summarize",
]
