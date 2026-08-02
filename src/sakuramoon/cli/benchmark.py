"""Config-bound benchmark execution and immutable evidence publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

from sakuramoon.config.resolve import resolved_config_sha256
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.telemetry.profiler import (
    ArtifactReference,
    BenchmarkComparison,
    BenchmarkIdentity,
    BenchmarkPlan,
    BenchmarkReport,
    BenchmarkRun,
    BenchmarkStepAdapter,
    BenchmarkVariant,
    BenchmarkWorkloadIdentity,
    ComparisonPolicy,
    CompileCounterProbe,
    CompileWindowEvidence,
    HotspotDecisions,
    PytorchTracePlan,
    RegionalCompileEvidence,
    ResourceIncreaseDisclosure,
    TraceIndexEntry,
    compare_benchmarks,
    run_benchmark,
    stream_sha256,
    summarize_benchmark,
    validate_hotspot_decisions,
    validate_trace_index,
)

_ALLOWED_VARIANT_KEYS = frozenset(
    {"compile.regional_enabled", "kernels.attention_backend"}
)


def benchmark_plan(config: RuntimeConfig) -> BenchmarkPlan:
    return BenchmarkPlan(
        kind=config.benchmark.kind,
        warmup_updates=config.benchmark.warmup_updates,
        measured_updates=config.benchmark.measured_updates,
        starting_successful_update=config.benchmark.starting_successful_update,
        checkpoint_every_updates=config.checkpoint.full_every_updates,
    )


def _replace_path(payload: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, dict):
            raise TypeError(f"variant config key does not exist: {path}")
        current = cast(dict[str, Any], value)
    if parts[-1] not in current:
        raise ValueError(f"variant config key does not exist: {path}")
    current[parts[-1]] = "<BENCHMARK_VARIANT>"


def _normalized_config_sha256(
    config: RuntimeConfig, changed_config_keys: tuple[str, ...]
) -> str:
    if not set(changed_config_keys).issubset(_ALLOWED_VARIANT_KEYS):
        raise ValueError("benchmark variant changes a protected workload config key")
    payload = config.model_dump(mode="python", by_alias=True)
    for path in changed_config_keys:
        _replace_path(payload, path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_benchmark_identity(
    config: RuntimeConfig,
    *,
    checkpoint_id: str,
    checkpoint_artifact: Path,
    data_sequence_artifact: Path,
    shape_distribution_artifact: Path,
    software_lock_artifact: Path,
    hardware_id: str,
    source_commit: str,
    build_artifact: Path,
    variant_name: str,
    changed_config_keys: tuple[str, ...],
    enabled_features: tuple[str, ...],
) -> BenchmarkIdentity:
    """Construct identity from validated config and actual local artifacts."""

    changed = tuple(sorted(changed_config_keys))
    features = tuple(sorted(enabled_features))
    return BenchmarkIdentity(
        workload=BenchmarkWorkloadIdentity(
            normalized_config_sha256=_normalized_config_sha256(config, changed),
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=stream_sha256(checkpoint_artifact),
            data_sequence_sha256=stream_sha256(data_sequence_artifact),
            shape_distribution_sha256=stream_sha256(shape_distribution_artifact),
            software_lock_sha256=stream_sha256(software_lock_artifact),
            hardware_id=hardware_id,
            local_batch=config.stage.local_batch,
            accumulation=config.stage.accumulation,
            world_size=config.distributed.world_size,
        ),
        variant=BenchmarkVariant(
            name=variant_name,
            resolved_config_sha256=resolved_config_sha256(config),
            source_commit=source_commit,
            build_sha256=stream_sha256(build_artifact),
            backend=config.kernels.attention_backend,
            enabled_features=features,
            changed_config_keys=changed,
        ),
    )


def run_configured_benchmark(
    config: RuntimeConfig,
    identity: BenchmarkIdentity,
    adapter: BenchmarkStepAdapter,
    *,
    compile_probe: CompileCounterProbe,
    trace_output: Path,
    trace_kernel_groups: Mapping[str, tuple[str, ...]],
    require_cuda_activity: bool,
) -> tuple[BenchmarkRun, BenchmarkReport]:
    """Execute the configured warmup/measured windows through a real-step adapter."""

    if identity.variant.resolved_config_sha256 != resolved_config_sha256(config):
        raise ValueError("benchmark identity differs from the actual resolved config")
    workload = identity.workload
    if (
        workload.local_batch != config.stage.local_batch
        or workload.accumulation != config.stage.accumulation
        or workload.world_size != config.distributed.world_size
        or identity.variant.backend != config.kernels.attention_backend
    ):
        raise ValueError("benchmark identity differs from runtime topology")
    plan = benchmark_plan(config)
    trace_plan = PytorchTracePlan(
        path=trace_output,
        benchmark_identity_sha256=identity.sha256,
        sampled_updates=config.benchmark.profile_trace_updates,
        record_shapes=config.profiling.record_shapes,
        profile_memory=config.profiling.profile_memory,
        require_cuda_activity=require_cuda_activity,
        kernel_groups=trace_kernel_groups,
    )
    run = run_benchmark(
        plan,
        adapter,
        compile_probe=compile_probe,
        trace_plan=trace_plan,
        expected_data_sequence_sha256=identity.workload.data_sequence_sha256,
        expected_shape_distribution_sha256=identity.workload.shape_distribution_sha256,
    )
    report = summarize_benchmark(identity, plan, run)
    return run, report


def comparison_policy(
    config: RuntimeConfig,
    *,
    max_p95_regression_percent: float,
    max_p99_regression_percent: float,
    cuda_allocated: ResourceIncreaseDisclosure,
    cuda_reserved: ResourceIncreaseDisclosure,
    host_rss: ResourceIncreaseDisclosure,
    pinned_ram: ResourceIncreaseDisclosure,
    host_swap: ResourceIncreaseDisclosure,
) -> ComparisonPolicy:
    return ComparisonPolicy(
        config.compile.minimum_end_to_end_gain_percent,
        max_p95_regression_percent,
        max_p99_regression_percent,
        cuda_allocated,
        cuda_reserved,
        host_rss,
        pinned_ram,
        host_swap,
    )


def _identity_mapping(identity: BenchmarkIdentity) -> dict[str, object]:
    return asdict(identity)


def _report_mapping(report: BenchmarkReport) -> dict[str, object]:
    payload = {field.name: getattr(report, field.name) for field in fields(report)}
    payload["identity"] = _identity_mapping(report.identity)
    payload["kernel_group_share"] = dict(report.kernel_group_share)
    payload["plan"] = asdict(report.plan)
    payload["phase_share"] = dict(report.phase_share)
    payload["schema_version"] = 1
    return payload


def _hotspot_mapping(decisions: HotspotDecisions) -> dict[str, object]:
    return {
        "kernel_groups": [asdict(group) for group in decisions.kernel_groups],
        "kernel_retention_rationales": dict(
            decisions.kernel_retention_rationales
        ),
        "optimized_kernel_groups": list(decisions.optimized_kernel_groups),
        "optimized_phases": list(decisions.optimized_phases),
        "phase_retention_rationales": dict(decisions.phase_retention_rationales),
    }


def _trace_mapping(entry: TraceIndexEntry) -> dict[str, object]:
    return {
        "benchmark_identity_sha256": entry.benchmark_identity_sha256,
        "first_measured_update": entry.first_measured_update,
        "first_successful_update": entry.first_successful_update,
        "hotspot_name": entry.hotspot_name,
        "hotspot_rationale": entry.hotspot_rationale,
        "last_measured_update": entry.last_measured_update,
        "last_successful_update": entry.last_successful_update,
        "path": entry.path.as_posix(),
        "sha256": entry.sha256,
        "tool": entry.tool,
    }


def _validate_report_profiler_trace(
    report: BenchmarkReport, entries: tuple[TraceIndexEntry, ...]
) -> None:
    profiler_entries = tuple(entry for entry in entries if entry.tool == "pytorch_profiler")
    if len(profiler_entries) != 1:
        raise ValueError("benchmark report requires exactly one measured PyTorch trace")
    if profiler_entries[0].sha256 != report.profiler_trace_sha256:
        raise ValueError("benchmark report metrics differ from the indexed profiler trace")


def _artifact_reference_mapping(reference: ArtifactReference | None) -> object:
    if reference is None:
        return None
    return {
        "build_sha256": reference.build_sha256,
        "checkpoint_id": reference.checkpoint_id,
        "kind": reference.kind,
        "path": reference.path.as_posix(),
        "resolved_config_sha256": reference.resolved_config_sha256,
        "sha256": reference.sha256,
        "source_commit": reference.source_commit,
        "world_size": reference.world_size,
    }


def _publish_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("benchmark evidence already exists")
    body = (
        json.dumps(dict(payload), allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_benchmark_samples(
    path: Path, report: BenchmarkReport, run: BenchmarkRun
) -> None:
    lines = [
        json.dumps(sample.as_mapping(), sort_keys=True, separators=(",", ":"))
        for sample in run.samples
    ]
    raw = ("\n".join(lines) + "\n").encode()
    if hashlib.sha256(raw).hexdigest() != report.raw_samples_sha256:
        raise ValueError("raw benchmark samples differ from report identity")
    _publish_json(
        path,
        {
            "benchmark_identity_sha256": report.identity.sha256,
            "raw_samples_sha256": report.raw_samples_sha256,
            "samples": [sample.as_mapping() for sample in run.samples],
            "schema_version": 1,
        },
    )


def write_trace_index(
    path: Path,
    entries: tuple[TraceIndexEntry, ...],
    *,
    report: BenchmarkReport,
    profile_trace_updates: int,
    hotspots: HotspotDecisions,
) -> None:
    _validate_report_profiler_trace(report, entries)
    for entry in entries:
        if stream_sha256(entry.path) != entry.sha256:
            raise ValueError("trace was modified after index construction")
    validate_trace_index(
        entries,
        plan=report.plan,
        profile_trace_updates=profile_trace_updates,
        benchmark_identity_sha256=report.identity.sha256,
        hotspots=hotspots,
    )
    _publish_json(
        path,
        {
            "benchmark_identity_sha256": report.identity.sha256,
            "entries": [_trace_mapping(entry) for entry in entries],
            "profile_trace_updates": profile_trace_updates,
            "schema_version": 1,
        },
    )


def write_benchmark_report(
    path: Path,
    report: BenchmarkReport,
    *,
    compile_evidence: CompileWindowEvidence,
    hotspots: HotspotDecisions,
    traces: tuple[TraceIndexEntry, ...],
    profile_trace_updates: int,
) -> None:
    compile_counts = (
        compile_evidence.measured_compiles,
        compile_evidence.measured_recompiles,
        compile_evidence.measured_fallbacks,
    )
    report_counts = (
        report.measured_compile_count,
        report.measured_recompile_count,
        report.measured_fallback_count,
    )
    if compile_counts != report_counts:
        raise ValueError("compile evidence counters differ from benchmark report")
    validate_hotspot_decisions(report, hotspots)
    _validate_report_profiler_trace(report, traces)
    for entry in traces:
        if stream_sha256(entry.path) != entry.sha256:
            raise ValueError("trace was modified before report publication")
    validate_trace_index(
        traces,
        plan=report.plan,
        profile_trace_updates=profile_trace_updates,
        benchmark_identity_sha256=report.identity.sha256,
        hotspots=hotspots,
    )
    _publish_json(
        path,
        {
            "benchmark_identity_sha256": report.identity.sha256,
            "compile_window": asdict(compile_evidence),
            "hotspot_decisions": _hotspot_mapping(hotspots),
            "report": _report_mapping(report),
            "schema_version": 1,
            "traces": [_trace_mapping(entry) for entry in traces],
        },
    )


def write_comparison_report(
    path: Path,
    baseline: BenchmarkReport,
    after: BenchmarkReport,
    *,
    policy: ComparisonPolicy,
    compile_evidence: RegionalCompileEvidence,
) -> BenchmarkComparison:
    comparison = compare_benchmarks(
        baseline,
        after,
        policy=policy,
        regional_compile=compile_evidence,
    )
    evidence = {
        "correctness": _artifact_reference_mapping(compile_evidence.correctness),
        "ddp": _artifact_reference_mapping(compile_evidence.ddp),
        "resume": _artifact_reference_mapping(compile_evidence.resume),
    }
    _publish_json(
        path,
        {
            "after_identity_sha256": after.identity.sha256,
            "baseline_identity_sha256": baseline.identity.sha256,
            "comparison": asdict(comparison),
            "policy": asdict(policy),
            "regional_compile_evidence": evidence,
            "schema_version": 1,
        },
    )
    return comparison


__all__ = [
    "benchmark_plan",
    "build_benchmark_identity",
    "comparison_policy",
    "run_configured_benchmark",
    "write_benchmark_report",
    "write_benchmark_samples",
    "write_comparison_report",
    "write_trace_index",
]
