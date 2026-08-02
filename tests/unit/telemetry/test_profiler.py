from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import sakuramoon.telemetry.profiler as profiler_module
from sakuramoon.telemetry.metrics import CORE_TIMING_PHASES
from sakuramoon.telemetry.profiler import (
    ArtifactReference,
    BenchmarkIdentity,
    BenchmarkObservation,
    BenchmarkPlan,
    BenchmarkReport,
    BenchmarkRun,
    BenchmarkSample,
    BenchmarkVariant,
    BenchmarkWorkloadIdentity,
    CapturedTrace,
    ComparisonPolicy,
    CompileCounters,
    CompileWindowEvidence,
    HotspotDecisions,
    KernelGroupHotspot,
    PytorchTracePlan,
    RegionalCompileEvidence,
    ResourceIncreaseDisclosure,
    StepPayload,
    TraceIndexEntry,
    TraceMetrics,
    canonical_workload_artifact_bytes,
    capture_external_trace_smoke,
    compare_benchmarks,
    run_benchmark,
    stream_sha256,
    summarize_benchmark,
    validate_hotspot_decisions,
    validate_trace_index,
)
from sakuramoon.telemetry.timers import PhaseTimer


def _identity(
    *,
    variant: str = "baseline",
    changed: tuple[str, ...] = (),
    features: tuple[str, ...] = ("dit",),
    world_size: int = 1,
) -> BenchmarkIdentity:
    return BenchmarkIdentity(
        BenchmarkWorkloadIdentity(
            "1" * 64,
            "checkpoint-1",
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            "rtx-5090-0",
            1,
            2,
            world_size,
        ),
        BenchmarkVariant(
            variant,
            "6" * 64,
            "7" * 40,
            "8" * 64,
            "native",
            features,
            changed,
        ),
    )


def _phases(*, scale: float = 1.0) -> dict[str, float]:
    phases: dict[str, float] = dict.fromkeys(CORE_TIMING_PHASES, 0.0)
    phases.update(
        {
            "data": 0.00003 * scale,
            "dit_forward": 0.00004 * scale,
            "optimizer": 0.00001 * scale,
            "h2d": 0.000005 * scale,
        }
    )
    return phases


class _Adapter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        self.calls.append((update, measured))
        timer = PhaseTimer(device=torch.device("cpu"))
        checkpoint_paths: tuple[Path, ...] = ()
        if update % 1000 == 0:
            with timer.record("checkpoint"):
                checkpoint_paths = (Path(__file__),)
        return StepPayload(
            update,
            timer,
            2,
            1024,
            128,
            10000,
            _observation(update),
            checkpoint_paths,
            {},
        )


class _Probe:
    def __init__(self, snapshots: list[CompileCounters]) -> None:
        self.snapshots = iter(snapshots)

    def snapshot(self) -> CompileCounters:
        return next(self.snapshots)


class _MissingCheckpointAdapter(_Adapter):
    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        payload = super().run_successful_update(update, measured=measured)
        if update % 1000 == 0:
            return dataclasses.replace(payload, checkpoint_paths=())
        return payload


class _UnexpectedCheckpointAdapter(_Adapter):
    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        payload = super().run_successful_update(update, measured=measured)
        if update == 1250:
            with payload.phase_timer.record("checkpoint"):
                return dataclasses.replace(payload, checkpoint_paths=(Path(__file__),))
        return payload


class _DriftedObservationAdapter(_Adapter):
    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        payload = super().run_successful_update(update, measured=measured)
        if update == 1250:
            return dataclasses.replace(
                payload,
                observation=BenchmarkObservation(
                    ("different-sample-0", "different-sample-1"),
                    payload.observation.shape_keys,
                ),
            )
        return payload


def _sample(
    update: int,
    *,
    scale: float = 1.0,
    extra_reserved: int = 0,
    host_swap: int = 0,
) -> BenchmarkSample:
    return BenchmarkSample(
        update,
        100 + update,
        (0.0001 + update / 1_000_000_000.0) * scale,
        0.00008 * scale,
        _phases(scale=scale),
        2,
        1024,
        128,
        10000,
        1000,
        2000 + extra_reserved,
        3000,
        400,
        host_swap,
        0,
        0.0,
    )


def _observation(update: int) -> BenchmarkObservation:
    return BenchmarkObservation(
        (f"sample-{update}-0", f"sample-{update}-1"),
        ("bucket-512", "bucket-512"),
    )


def _workload_hashes(plan: BenchmarkPlan) -> tuple[str, str]:
    observations = tuple(
        (update, _observation(update))
        for update in range(
            plan.starting_successful_update + 1,
            plan.starting_successful_update
            + plan.warmup_updates
            + plan.measured_updates
            + 1,
        )
    )
    data = canonical_workload_artifact_bytes(observations, kind="data_sequence")
    shapes = canonical_workload_artifact_bytes(
        observations, kind="shape_distribution"
    )
    return hashlib.sha256(data).hexdigest(), hashlib.sha256(shapes).hexdigest()


def _trace() -> TraceMetrics:
    return TraceMetrics(
        5,
        0.0005,
        0.00035,
        0.0001,
        0.00005,
        100,
        0.00005,
        0.0,
        {"fused-small": 0.00003},
    )


def _run(
    samples: tuple[BenchmarkSample, ...],
    *,
    measured_window_seconds: float | None = None,
) -> BenchmarkRun:
    zero = CompileCounters(0, 0, 0)
    path = Path(__file__)
    trace = CapturedTrace(
        TraceIndexEntry(
            "pytorch_profiler",
            path,
            stream_sha256(path),
            _identity().sha256,
            1,
            5,
            1000,
            1004,
            None,
            None,
        ),
        _trace(),
    )
    return BenchmarkRun(
        samples,
        CompileWindowEvidence(zero, zero, zero, 0.01),
        trace,
        "3" * 64,
        "4" * 64,
        sum(sample.step_seconds for sample in samples)
        if measured_window_seconds is None
        else measured_window_seconds,
    )


def _zero_probe() -> _Probe:
    zero = CompileCounters(0, 0, 0)
    return _Probe([zero, zero, zero])


def _plan() -> BenchmarkPlan:
    return BenchmarkPlan("candidate", 100, 500, 899, 1000)


def _trace_plan(tmp_path: Path) -> PytorchTracePlan:
    return PytorchTracePlan(
        tmp_path / "trace.json", _identity().sha256, 5, True, True, False
    )


def _run_observed_benchmark(
    plan: BenchmarkPlan,
    adapter: _Adapter,
    *,
    compile_probe: _Probe,
    trace_plan: PytorchTracePlan,
) -> BenchmarkRun:
    data_sha256, shape_sha256 = _workload_hashes(plan)
    return run_benchmark(
        plan,
        adapter,
        compile_probe=compile_probe,
        trace_plan=trace_plan,
        expected_data_sequence_sha256=data_sha256,
        expected_shape_distribution_sha256=shape_sha256,
    )


def test_candidate_runner_measures_100_warmup_and_500_successful_updates(
    tmp_path: Path,
) -> None:
    plan = _plan()
    adapter = _Adapter()

    run = _run_observed_benchmark(
        plan,
        adapter,
        compile_probe=_zero_probe(),
        trace_plan=_trace_plan(tmp_path),
    )

    assert adapter.calls[:2] == [(900, False), (901, False)]
    assert adapter.calls[99:102] == [(999, False), (1000, True), (1001, True)]
    assert adapter.calls[-1] == (1499, True)
    assert len(run.samples) == 500
    assert run.samples[0].successful_update == 1000
    assert run.samples[-1].successful_update == 1499
    assert run.samples[0].host_rss_bytes > 0
    assert len({sample.host_rss_bytes for sample in run.samples}) == 1
    assert run.trace.metrics.sampled_updates == 5
    assert run.trace.entry.sha256 == stream_sha256(run.trace.entry.path)


def test_trace_serialization_is_outside_the_measured_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock_ns = 0

    def perf_counter_ns() -> int:
        nonlocal clock_ns
        clock_ns += 1_000_000
        return clock_ns

    class FakeProfiler:
        def start(self) -> None:
            pass

        def step(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def export_chrome_trace(self, path: str) -> None:
            nonlocal clock_ns
            clock_ns += 10_000_000_000
            events: list[dict[str, object]] = [
                {
                    "name": "benchmark_trace_window:1000:1004",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 0,
                    "dur": 1_000_000,
                }
            ]
            events.extend(
                {
                    "name": (
                        f"benchmark_measured_update:{index};"
                        f"successful_update:{999 + index}"
                    ),
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": index,
                    "dur": 1,
                }
                for index in range(1, 6)
            )
            Path(path).write_text(
                json.dumps({"traceEvents": events}), encoding="utf-8"
            )

    monkeypatch.setattr(profiler_module.time, "perf_counter_ns", perf_counter_ns)
    monkeypatch.setattr(
        torch.profiler, "profile", lambda **_kwargs: FakeProfiler()
    )

    run = _run_observed_benchmark(
        _plan(),
        _Adapter(),
        compile_probe=_zero_probe(),
        trace_plan=_trace_plan(tmp_path),
    )

    assert run.measured_window_seconds < 10.0


def test_measured_recompile_or_fallback_hard_fails(tmp_path: Path) -> None:
    probe = _Probe(
        [CompileCounters(0, 0, 0), CompileCounters(1, 0, 0), CompileCounters(1, 1, 0)]
    )
    with pytest.raises(RuntimeError, match="recompiled"):
        _run_observed_benchmark(
            _plan(),
            _Adapter(),
            compile_probe=probe,
            trace_plan=_trace_plan(tmp_path),
        )


@pytest.mark.parametrize(
    "after_measured",
    [CompileCounters(2, 0, 0), CompileCounters(1, 1, 0), CompileCounters(1, 0, 1)],
)
def test_any_measured_compile_counter_growth_hard_fails(
    tmp_path: Path, after_measured: CompileCounters
) -> None:
    probe = _Probe(
        [CompileCounters(0, 0, 0), CompileCounters(1, 0, 0), after_measured]
    )
    with pytest.raises(RuntimeError, match="compiled, recompiled, or fell back"):
        _run_observed_benchmark(
            _plan(),
            _Adapter(),
            compile_probe=probe,
            trace_plan=_trace_plan(tmp_path),
        )
    assert not (tmp_path / "trace.json").exists()
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []


def test_measured_checkpoint_cadence_requires_real_timed_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="checkpoint cadence"):
        _run_observed_benchmark(
            _plan(),
            _MissingCheckpointAdapter(),
            compile_probe=_zero_probe(),
            trace_plan=_trace_plan(tmp_path),
        )
    assert not (tmp_path / "trace.json").exists()
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []


def test_measured_workload_rejects_checkpoint_outside_configured_cadence(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="outside configured cadence"):
        _run_observed_benchmark(
            _plan(),
            _UnexpectedCheckpointAdapter(),
            compile_probe=_zero_probe(),
            trace_plan=_trace_plan(tmp_path),
        )
    assert not (tmp_path / "trace.json").exists()
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []


def test_actual_observation_stream_must_match_workload_identity(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="data sequence differs"):
        _run_observed_benchmark(
            _plan(),
            _DriftedObservationAdapter(),
            compile_probe=_zero_probe(),
            trace_plan=_trace_plan(tmp_path),
        )
    assert not (tmp_path / "trace.json").exists()
    assert list(tmp_path.glob(".trace.json.*.tmp")) == []


@pytest.mark.parametrize(
    "plan",
    [
        ("candidate", 99, 500, 899, 1000),
        ("candidate", 100, 499, 899, 1000),
        ("final", 100, 999, 899, 1000),
        ("candidate", 100, 500, 0, 1000),
    ],
)
def test_plan_rejects_short_measurement_windows(
    plan: tuple[str, int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        BenchmarkPlan(*plan)  # pyright: ignore[reportArgumentType]


def test_summary_supports_overlapped_core_and_detailed_phases() -> None:
    plan = _plan()
    samples = tuple(_sample(update) for update in range(1, 501))
    report = summarize_benchmark(_identity(), plan, _run(samples))

    assert report.sample_count == 500
    assert report.step_p99_seconds >= report.step_p50_seconds > 0.0
    assert report.phase_share["h2d"] > 0.0
    assert report.raw_samples_sha256 == hashlib.sha256(
        b"".join(
            __import__("json").dumps(sample.as_mapping(), sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for sample in samples
        )
    ).hexdigest()


def test_summary_uses_synchronized_window_for_aggregate_rates_and_shares() -> None:
    plan = _plan()
    samples = tuple(_sample(update) for update in range(1, 501))
    checkpoint_seconds = 0.00001
    first_phases = dict(samples[0].phase_seconds)
    first_phases["checkpoint"] = checkpoint_seconds
    samples = (
        dataclasses.replace(
            samples[0],
            phase_seconds=first_phases,
            checkpoint_bytes=17,
            checkpoint_seconds=checkpoint_seconds,
        ),
        *samples[1:],
    )
    per_update_sum = sum(sample.step_seconds for sample in samples)
    synchronized_window = per_update_sum * 0.75

    report = summarize_benchmark(
        _identity(),
        plan,
        _run(samples, measured_window_seconds=synchronized_window),
    )

    assert report.total_measured_seconds == pytest.approx(synchronized_window)
    assert report.samples_per_second == pytest.approx(
        sum(sample.samples for sample in samples) / synchronized_window
    )
    assert report.image_tokens_per_second == pytest.approx(
        sum(sample.image_tokens for sample in samples) / synchronized_window
    )
    assert report.phase_share["checkpoint"] == pytest.approx(
        checkpoint_seconds / synchronized_window
    )
    assert report.checkpoint_amortized_share == pytest.approx(
        checkpoint_seconds / synchronized_window
    )
    assert report.step_p50_seconds == pytest.approx(
        float(statistics.median(sample.step_seconds for sample in samples))
    )


@pytest.mark.parametrize("elapsed", [0.0, float("nan"), float("inf")])
def test_benchmark_run_rejects_invalid_synchronized_window(elapsed: float) -> None:
    samples = tuple(_sample(update) for update in range(1, 501))
    with pytest.raises(ValueError, match="measured_window_seconds"):
        _run(samples, measured_window_seconds=elapsed)


def test_summary_rejects_update_longer_than_synchronized_window() -> None:
    samples = (
        _sample(1, scale=20.0),
        *tuple(_sample(update) for update in range(2, 501)),
    )
    with pytest.raises(ValueError, match="update duration"):
        summarize_benchmark(
            _identity(),
            _plan(),
            _run(samples, measured_window_seconds=0.00001),
        )


def _write_synthetic_trace(path: Path, device_events: list[dict[str, object]]) -> None:
    events: list[dict[str, object]] = [
        {
            "name": "benchmark_trace_window:1000:1000",
            "cat": "user_annotation",
            "ph": "X",
            "ts": 0,
            "dur": 1_000_000,
        },
        {
            "name": "benchmark_measured_update:1;successful_update:1000",
            "cat": "user_annotation",
            "ph": "X",
            "ts": 1,
            "dur": 1,
        },
        *device_events,
    ]
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def test_trace_metrics_include_overlapping_kernel_copy_and_memset_activity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.json"
    _write_synthetic_trace(
        path,
        [
            {
                "name": "fused_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 100_000,
                "dur": 200_000,
            },
            {
                "name": "copy",
                "cat": "gpu_memcpy",
                "ph": "X",
                "ts": 250_000,
                "dur": 200_000,
            },
            {
                "name": "zero",
                "cat": "gpu_memset",
                "ph": "X",
                "ts": 500_000,
                "dur": 100_000,
            },
        ],
    )

    metrics = profiler_module._derive_trace_metrics(  # pyright: ignore[reportPrivateUsage]
        path,
        sampled_successful_updates=(1000,),
        kernel_groups={"fused": ("fused_kernel",)},
        require_cuda_activity=True,
    )

    assert metrics.gpu_active_seconds == pytest.approx(0.45)
    assert metrics.gpu_idle_seconds == pytest.approx(0.55)
    assert metrics.gpu_unattributed_seconds == 0.0
    assert metrics.kernel_launches == 1
    assert metrics.kernel_group_seconds == {"fused": pytest.approx(0.2)}


def test_trace_metrics_classify_unknown_gpu_work_as_unattributed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.json"
    _write_synthetic_trace(
        path,
        [
            {
                "name": "kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 100_000,
                "dur": 350_000,
            },
            {
                "name": "unknown",
                "cat": "gpu_custom",
                "ph": "X",
                "ts": 400_000,
                "dur": 300_000,
            },
        ],
    )

    metrics = profiler_module._derive_trace_metrics(  # pyright: ignore[reportPrivateUsage]
        path,
        sampled_successful_updates=(1000,),
        kernel_groups={},
        require_cuda_activity=True,
    )

    assert metrics.gpu_active_seconds == pytest.approx(0.35)
    assert metrics.gpu_unattributed_seconds == pytest.approx(0.25)
    assert metrics.gpu_idle_seconds == pytest.approx(0.4)
    assert metrics.kernel_launches == 1


def test_coarse_and_fusible_kernel_groups_require_decisions() -> None:
    report = summarize_benchmark(
        _identity(),
        _plan(),
        _run(tuple(_sample(update) for update in range(1, 501))),
    )
    group = KernelGroupHotspot("fused-small", ("a", "b", "c"), 0.06)
    missing = HotspotDecisions((), {}, (group,), (), {})
    with pytest.raises(ValueError, match="phase"):
        validate_hotspot_decisions(report, missing)

    phases = tuple(
        sorted(phase for phase, share in report.phase_share.items() if share > 0.05)
    )
    with pytest.raises(ValueError, match="kernel group"):
        validate_hotspot_decisions(
            report, HotspotDecisions(phases, {}, (group,), (), {})
        )
    validate_hotspot_decisions(
        report,
        HotspotDecisions(
            phases,
            {},
            (group,),
            (),
            {"fused-small": "launch latency is hidden by the critical path"},
        ),
    )


def _policy(*, extra_reserved: int = 0) -> ComparisonPolicy:
    zero = ResourceIncreaseDisclosure(0, "")
    return ComparisonPolicy(
        3.0,
        0.0,
        0.0,
        zero,
        ResourceIncreaseDisclosure(extra_reserved, "declared workspace" if extra_reserved else ""),
        zero,
        zero,
        zero,
    )


def _report(
    identity: BenchmarkIdentity,
    *,
    scale: float,
    extra_reserved: int = 0,
    host_swap: int = 0,
) -> BenchmarkReport:
    return summarize_benchmark(
        identity,
        _plan(),
        _run(
            tuple(
                _sample(
                    update,
                    scale=scale,
                    extra_reserved=extra_reserved,
                    host_swap=host_swap,
                )
                for update in range(1, 501)
            )
        ),
    )


def _regional_compile_evidence(
    tmp_path: Path,
    identity: BenchmarkIdentity,
    *,
    world_sizes: tuple[int, int, int] = (4, 4, 4),
) -> RegionalCompileEvidence:
    variant = identity.variant
    workload = identity.workload

    def evidence(kind: str, world_size: int) -> ArtifactReference:
        path = tmp_path / f"{kind}.json"
        path.write_text(
            json.dumps(
                {
                    "build_sha256": variant.build_sha256,
                    "checkpoint_id": workload.checkpoint_id,
                    "kind": kind,
                    "resolved_config_sha256": variant.resolved_config_sha256,
                    "schema_version": 1,
                    "source_commit": variant.source_commit,
                    "status": "passed",
                    "world_size": world_size,
                },
                sort_keys=True,
            )
            + "\n"
        )
        return ArtifactReference(
            path,
            stream_sha256(path),
            kind,  # pyright: ignore[reportArgumentType]
        )

    return RegionalCompileEvidence(
        *(evidence(kind, world_size) for kind, world_size in zip(
            ("correctness", "ddp", "resume"), world_sizes, strict=True
        ))
    )


def test_comparison_allows_disclosed_variant_but_single_gpu_never_enables_compile() -> None:
    changed = ("compile.regional_enabled",)
    baseline = _report(_identity(changed=changed), scale=1.0)
    after = _report(
        _identity(variant="compile-on", changed=changed, features=("dit", "regional_compile")),
        scale=0.96,
        extra_reserved=100,
    )
    with pytest.raises(ValueError, match="undisclosed"):
        compare_benchmarks(
            baseline,
            after,
            policy=_policy(),
            regional_compile=RegionalCompileEvidence(None, None, None),
        )
    comparison = compare_benchmarks(
        baseline,
        after,
        policy=_policy(extra_reserved=100),
        regional_compile=RegionalCompileEvidence(None, None, None),
    )
    assert comparison.throughput_gain_percent > 3.0
    assert comparison.regional_compile_allowed is False


def test_compile_gate_requires_hash_bound_four_gpu_evidence(tmp_path: Path) -> None:
    changed = ("compile.regional_enabled",)
    baseline = _report(_identity(changed=changed, world_size=4), scale=1.0)
    after = _report(
        _identity(
            variant="compile-on",
            changed=changed,
            features=("dit", "regional_compile"),
            world_size=4,
        ),
        scale=0.96,
    )
    comparison = compare_benchmarks(
        baseline,
        after,
        policy=_policy(),
        regional_compile=_regional_compile_evidence(tmp_path, after.identity),
    )
    assert comparison.regional_compile_allowed is True


def test_compile_gate_rejects_one_gpu_correctness_and_resume_evidence(
    tmp_path: Path,
) -> None:
    changed = ("compile.regional_enabled",)
    baseline = _report(_identity(changed=changed), scale=1.0)
    after = _report(
        _identity(
            variant="compile-on",
            changed=changed,
            features=("dit", "regional_compile"),
        ),
        scale=0.96,
    )
    comparison = compare_benchmarks(
        baseline,
        after,
        policy=_policy(),
        regional_compile=_regional_compile_evidence(
            tmp_path,
            after.identity,
            world_sizes=(1, 4, 1),
        ),
    )
    assert comparison.regional_compile_allowed is False


def test_compile_gate_rejects_variant_without_compile_feature(tmp_path: Path) -> None:
    changed = ("compile.regional_enabled",)
    baseline = _report(_identity(changed=changed, world_size=4), scale=1.0)
    after = _report(
        _identity(variant="compile-claimed", changed=changed, world_size=4),
        scale=0.96,
    )

    comparison = compare_benchmarks(
        baseline,
        after,
        policy=_policy(),
        regional_compile=_regional_compile_evidence(tmp_path, after.identity),
    )
    assert comparison.regional_compile_allowed is False


def test_compile_gate_rejects_combined_backend_and_compile_change() -> None:
    changed = ("compile.regional_enabled", "kernels.attention_backend")
    baseline_identity = _identity(changed=changed, world_size=4)
    baseline_identity = dataclasses.replace(
        baseline_identity,
        variant=dataclasses.replace(
            baseline_identity.variant, backend="dense_sdpa_reference"
        ),
    )
    after_identity = _identity(
        variant="compile-and-backend",
        changed=changed,
        features=("dit", "regional_compile"),
        world_size=4,
    )
    after_identity = dataclasses.replace(
        after_identity,
        variant=dataclasses.replace(after_identity.variant, backend="fa4_varlen"),
    )

    with pytest.raises(ValueError, match="must isolate"):
        compare_benchmarks(
            _report(baseline_identity, scale=1.0),
            _report(after_identity, scale=0.9),
            policy=_policy(),
            regional_compile=RegionalCompileEvidence(None, None, None),
        )


@pytest.mark.parametrize(("baseline_swap", "after_swap"), [(1, 0), (0, 1)])
def test_comparison_rejects_any_host_swap(baseline_swap: int, after_swap: int) -> None:
    with pytest.raises(ValueError, match="zero host swap"):
        compare_benchmarks(
            _report(_identity(), scale=1.0, host_swap=baseline_swap),
            _report(_identity(variant="after"), scale=0.9, host_swap=after_swap),
            policy=_policy(),
            regional_compile=RegionalCompileEvidence(None, None, None),
        )


def test_comparison_policy_rejects_host_swap_allowance() -> None:
    zero = ResourceIncreaseDisclosure(0, "")
    with pytest.raises(ValueError, match="zero host swap"):
        ComparisonPolicy(
            3.0,
            0.0,
            0.0,
            zero,
            zero,
            zero,
            zero,
            ResourceIncreaseDisclosure(1, "not permitted"),
        )


def test_compile_artifact_rejects_non_exact_schema_types(tmp_path: Path) -> None:
    path = tmp_path / "ddp.json"
    path.write_text(
        json.dumps(
            {
                "build_sha256": "8" * 64,
                "checkpoint_id": "checkpoint-1",
                "kind": "ddp",
                "resolved_config_sha256": "6" * 64,
                "schema_version": True,
                "source_commit": "7" * 40,
                "status": "passed",
                "world_size": 4,
            }
        )
    )
    with pytest.raises(ValueError, match="passing artifact"):
        ArtifactReference(path, stream_sha256(path), "ddp")


@pytest.mark.parametrize(
    ("tool", "suffix"), [("nsys", ".nsys-rep"), ("ncu", ".ncu-rep")]
)
def test_external_trace_collector_smoke_cannot_claim_benchmark_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    suffix: str,
) -> None:
    def run(command: tuple[str, ...], *, check: bool, shell: bool):
        assert check is False
        assert shell is False
        base = next(argument for argument in command if argument.startswith("/") and ".tmp" in argument)
        Path(f"{base}{suffix}").write_bytes(b"trace")

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("sakuramoon.telemetry.profiler.subprocess.run", run)
    output = tmp_path / f"trace{suffix}"
    artifact = capture_external_trace_smoke(
        tool,  # pyright: ignore[reportArgumentType]
        (tool, "profile", "{output}"),
        output=output,
    )
    assert artifact.path == output
    assert artifact.tool == tool
    assert artifact.sha256 == stream_sha256(output)
    assert output.read_bytes() == b"trace"


def test_trace_index_streams_hash_and_binds_range_and_hotspot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.json"
    trace.write_bytes(b"trace")
    digest = stream_sha256(trace)

    def reject_read_bytes(_: Path) -> bytes:
        raise AssertionError("whole-file read")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    identity_sha = _identity().sha256
    profiler = TraceIndexEntry(
        "pytorch_profiler", trace, digest, identity_sha, 1, 5, 1000, 1004, None, None
    )
    ncu = TraceIndexEntry(
        "ncu", trace, digest, identity_sha, 1, 1, 1000, 1000, "fused", "measured at 6%"
    )
    hotspots = HotspotDecisions((), {}, (KernelGroupHotspot("fused", ("a", "b"), 0.06),), (), {"fused": "retained"})

    validate_trace_index(
        (profiler,),
        plan=_plan(),
        profile_trace_updates=5,
        benchmark_identity_sha256=identity_sha,
        hotspots=hotspots,
    )
    with pytest.raises(ValueError, match="verified marker importer"):
        validate_trace_index(
            (profiler, ncu),
            plan=_plan(),
            profile_trace_updates=5,
            benchmark_identity_sha256=identity_sha,
            hotspots=hotspots,
        )
    with pytest.raises(ValueError, match="successful-update"):
        validate_trace_index(
            (
                dataclasses.replace(
                    profiler,
                    first_successful_update=999,
                    last_successful_update=1003,
                ),
            ),
            plan=_plan(),
            profile_trace_updates=5,
            benchmark_identity_sha256=identity_sha,
            hotspots=hotspots,
        )
    with pytest.raises(ValueError, match="count"):
        validate_trace_index(
            (profiler,),
            plan=_plan(),
            profile_trace_updates=6,
            benchmark_identity_sha256=identity_sha,
            hotspots=hotspots,
        )


def test_comparison_rejects_workload_drift() -> None:
    baseline = _report(_identity(), scale=1.0)
    drifted_identity = dataclasses.replace(
        _identity(variant="after"),
        workload=dataclasses.replace(_identity().workload, checkpoint_id="other"),
    )
    with pytest.raises(ValueError, match="workload"):
        compare_benchmarks(
            baseline,
            _report(drifted_identity, scale=0.9),
            policy=_policy(),
            regional_compile=RegionalCompileEvidence(None, None, None),
        )


def test_report_phase_mapping_is_immutable() -> None:
    report = _report(_identity(), scale=1.0)
    assert isinstance(report.phase_share, MappingProxyType)
