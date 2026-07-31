"""SakuraMoon command-line entry points."""

from sakuramoon.cli.benchmark import (
    benchmark_plan,
    build_benchmark_identity,
    comparison_policy,
    run_configured_benchmark,
    write_benchmark_report,
    write_benchmark_samples,
    write_comparison_report,
    write_trace_index,
)
from sakuramoon.cli.eval import build_evaluation_jobs, write_evaluation_job
from sakuramoon.cli.fault_injection import main as fault_injection_main

__all__ = [
    "benchmark_plan",
    "build_benchmark_identity",
    "build_evaluation_jobs",
    "comparison_policy",
    "fault_injection_main",
    "run_configured_benchmark",
    "write_benchmark_report",
    "write_benchmark_samples",
    "write_comparison_report",
    "write_evaluation_job",
    "write_trace_index",
]
