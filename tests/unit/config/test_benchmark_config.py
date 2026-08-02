from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from sakuramoon.cli.benchmark import (
    benchmark_plan,
    build_benchmark_identity,
    run_configured_benchmark,
    write_benchmark_report,
    write_benchmark_samples,
    write_comparison_report,
    write_trace_index,
)
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.telemetry.profiler import (
    BenchmarkObservation,
    BenchmarkVariant,
    ComparisonPolicy,
    DisabledCompileCounterProbe,
    HotspotDecisions,
    RegionalCompileEvidence,
    ResourceIncreaseDisclosure,
    StepPayload,
    canonical_workload_artifact_bytes,
)
from sakuramoon.telemetry.timers import PhaseTimer


class _Adapter:
    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        del measured
        timer = PhaseTimer(device=torch.device("cpu"))
        checkpoint_paths: tuple[Path, ...] = ()
        if update % 1000 == 0:
            with timer.record("checkpoint"):
                checkpoint_paths = (Path(__file__),)
        return StepPayload(
            update,
            timer,
            1,
            256,
            128,
            1000,
            BenchmarkObservation((f"sample-{update}",), ("bucket-256",)),
            checkpoint_paths,
            {},
        )


def _identity(config: RuntimeConfig, tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"kind": "checkpoint"}) + "\n")
    plan = benchmark_plan(config)
    observations = tuple(
        (
            update,
            BenchmarkObservation((f"sample-{update}",), ("bucket-256",)),
        )
        for update in range(
            plan.starting_successful_update + 1,
            plan.starting_successful_update
            + plan.warmup_updates
            + plan.measured_updates
            + 1,
        )
    )
    data_sequence = tmp_path / "data-sequence.jsonl"
    data_sequence.write_bytes(
        canonical_workload_artifact_bytes(observations, kind="data_sequence")
    )
    shape_distribution = tmp_path / "shape-distribution.jsonl"
    shape_distribution.write_bytes(
        canonical_workload_artifact_bytes(observations, kind="shape_distribution")
    )
    root = Path(__file__).parents[3]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return build_benchmark_identity(
        config,
        checkpoint_id="checkpoint-contract",
        checkpoint_artifact=checkpoint,
        data_sequence_artifact=data_sequence,
        shape_distribution_artifact=shape_distribution,
        software_lock_artifact=root / "uv.lock",
        hardware_id="cpu-contract",
        source_commit=source_commit,
        build_artifact=root / "pyproject.toml",
        variant_name="baseline",
        changed_config_keys=(),
        enabled_features=("dit",),
    )


def _run(config: RuntimeConfig, tmp_path: Path):
    return run_configured_benchmark(
        config,
        _identity(config, tmp_path),
        _Adapter(),
        compile_probe=DisabledCompileCounterProbe(),
        trace_output=tmp_path / "measured-trace.json",
        trace_kernel_groups={},
        require_cuda_activity=False,
    )


def test_runtime_config_drives_harness_and_complete_report(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    plan = benchmark_plan(config)
    run, report = _run(config, tmp_path)
    destination = tmp_path / "benchmark.json"
    leftover = tmp_path / ".benchmark.json.tmp"
    leftover.write_text("belongs-to-another-writer\n")

    write_benchmark_report(
        destination,
        report,
        compile_evidence=run.compile,
        hotspots=HotspotDecisions((), {}, (), (), {}),
        traces=(run.trace.entry,),
        profile_trace_updates=config.benchmark.profile_trace_updates,
    )

    document = json.loads(destination.read_text())
    assert plan.warmup_updates == 100
    assert plan.measured_updates == 500
    assert document["report"]["checkpoint_bytes"] > 0
    assert document["report"]["checkpoint_amortized_share"] > 0.0
    assert run.measured_window_seconds > 0.0
    assert (
        document["report"]["total_measured_seconds"]
        == run.measured_window_seconds
        == report.total_measured_seconds
    )
    assert document["report"]["measured_compile_count"] == 0
    assert document["report"]["measured_recompile_count"] == 0
    assert document["report"]["measured_fallback_count"] == 0
    assert (
        document["report"]["observed_data_sequence_sha256"]
        == report.identity.workload.data_sequence_sha256
    )
    assert (
        document["report"]["observed_shape_distribution_sha256"]
        == report.identity.workload.shape_distribution_sha256
    )
    assert document["traces"][0]["first_successful_update"] == 1000
    assert document["traces"][0]["last_successful_update"] == 1004
    assert document["report"]["raw_samples_sha256"] == report.raw_samples_sha256
    assert document["compile_window"]["after_measured"]["recompile_count"] == 0
    assert leftover.read_text() == "belongs-to-another-writer\n"

    samples_path = tmp_path / "samples.json"
    write_benchmark_samples(samples_path, report, run)
    samples_document = json.loads(samples_path.read_text())
    assert len(samples_document["samples"]) == 500
    assert samples_document["raw_samples_sha256"] == report.raw_samples_sha256


def test_concurrent_report_writers_publish_once_without_cross_cleanup(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    run, report = _run(config, tmp_path)
    destination = tmp_path / "race.json"

    def publish() -> str:
        try:
            write_benchmark_report(
                destination,
                report,
                compile_evidence=run.compile,
                hotspots=HotspotDecisions((), {}, (), (), {}),
                traces=(run.trace.entry,),
                profile_trace_updates=config.benchmark.profile_trace_updates,
            )
        except FileExistsError:
            return "lost"
        return "published"

    def publish_index(_: int) -> str:
        return publish()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(publish_index, range(2)))

    assert outcomes == ["lost", "published"]
    assert destination.is_file()
    assert list(tmp_path.glob(".race.json.*.tmp")) == []


def test_report_writer_rejects_compile_evidence_counter_mismatch(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    run, report = _run(config, tmp_path)

    with pytest.raises(ValueError, match="compile evidence counters differ"):
        write_benchmark_report(
            tmp_path / "mismatched-compile.json",
            replace(report, measured_compile_count=1),
            compile_evidence=run.compile,
            hotspots=HotspotDecisions((), {}, (), (), {}),
            traces=(run.trace.entry,),
            profile_trace_updates=config.benchmark.profile_trace_updates,
        )


def test_trace_index_rechecks_content_before_publication(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    run, report = _run(config, tmp_path)
    entry = run.trace.entry
    hotspots = HotspotDecisions((), {}, (), (), {})
    write_trace_index(
        tmp_path / "trace-index.json",
        (entry,),
        report=report,
        profile_trace_updates=5,
        hotspots=hotspots,
    )

    entry.path.write_text("tampered\n")
    with pytest.raises(ValueError, match="modified"):
        write_trace_index(
            tmp_path / "tampered-index.json",
            (entry,),
            report=report,
            profile_trace_updates=5,
            hotspots=hotspots,
        )


def test_comparison_artifact_persists_policy_result_and_compile_gate(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    _, baseline = _run(config, tmp_path)
    baseline_variant = replace(
        baseline.identity.variant,
        changed_config_keys=("compile.regional_enabled",),
    )
    baseline = replace(
        baseline, identity=replace(baseline.identity, variant=baseline_variant)
    )
    after = replace(
        baseline,
        identity=replace(
            baseline.identity,
            variant=BenchmarkVariant(
                "compile-on-contract",
                "9" * 64,
                "8" * 40,
                "7" * 64,
                baseline.identity.variant.backend,
                ("dit", "regional_compile"),
                ("compile.regional_enabled",),
            ),
        ),
    )
    zero = ResourceIncreaseDisclosure(0, "")
    policy = ComparisonPolicy(3.0, 0.0, 0.0, zero, zero, zero, zero, zero)
    destination = tmp_path / "comparison.json"

    result = write_comparison_report(
        destination,
        baseline,
        after,
        policy=policy,
        compile_evidence=RegionalCompileEvidence(None, None, None),
    )

    document = json.loads(destination.read_text())
    assert result.regional_compile_allowed is False
    assert document["policy"]["minimum_end_to_end_gain_percent"] == 3.0
    assert document["regional_compile_evidence"]["ddp"] is None
    assert document["comparison"]["regional_compile_allowed"] is False
    assert document["comparison"]["extra_host_swap_bytes"] == 0
    assert document["policy"]["host_swap"]["max_extra_bytes"] == 0


def test_runtime_config_rejects_short_final_or_compile_enable(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["benchmark"].update({"kind": "final", "measured_updates": 999})
    with pytest.raises(ValidationError, match="1,000"):
        RuntimeConfig.model_validate(valid_payload)

    valid_payload["benchmark"].update({"kind": "candidate", "measured_updates": 500})
    valid_payload["compile"]["regional_enabled"] = True
    with pytest.raises(ValidationError, match="False"):
        RuntimeConfig.model_validate(valid_payload)


def test_runtime_config_requires_measured_checkpoint_cadence(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["benchmark"]["starting_successful_update"] = 0
    with pytest.raises(ValidationError, match="checkpoint cadence"):
        RuntimeConfig.model_validate(valid_payload)


def test_identity_builder_rejects_protected_workload_changes(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("checkpoint\n")
    with pytest.raises(ValueError, match="protected"):
        build_benchmark_identity(
            config,
            checkpoint_id="checkpoint",
            checkpoint_artifact=checkpoint,
            data_sequence_artifact=checkpoint,
            shape_distribution_artifact=checkpoint,
            software_lock_artifact=checkpoint,
            hardware_id="cpu",
            source_commit="1" * 40,
            build_artifact=checkpoint,
            variant_name="invalid",
            changed_config_keys=("stage.local_batch",),
            enabled_features=("dit",),
        )
