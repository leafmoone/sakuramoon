from __future__ import annotations

import hashlib
from itertools import repeat
from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.telemetry.profiler import (
    BenchmarkObservation,
    BenchmarkPlan,
    DisabledCompileCounterProbe,
    PytorchTracePlan,
    StepPayload,
    canonical_workload_artifact_bytes,
    run_benchmark,
    stream_sha256,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.benchmark import (
    MeasuredMicrobatch,
    SingleGpuStepBenchmarkAdapter,
)
from sakuramoon.train.step import SingleGpuUpdateState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class _MatmulAdapter:
    def __init__(self, checkpoint_path: Path) -> None:
        self.left = torch.randn((128, 128), device="cuda")
        self.right = torch.randn((128, 128), device="cuda")
        self.checkpoint_path = checkpoint_path

    def run_successful_update(self, update: int, *, measured: bool) -> StepPayload:
        del measured
        output = torch.mm(self.left, self.right)
        assert bool(torch.isfinite(output).all().item())
        timer = PhaseTimer(device=torch.device("cuda"))
        checkpoint_paths: tuple[Path, ...] = ()
        if update % 1000 == 0:
            with timer.record("checkpoint"):
                self.checkpoint_path.write_bytes(b"synthetic-checkpoint")
                checkpoint_paths = (self.checkpoint_path,)
        return StepPayload(
            update,
            timer,
            1,
            128,
            1,
            2 * 128**3,
            BenchmarkObservation((f"matmul-{update}",), ("128x128",)),
            checkpoint_paths,
            {},
        )


def test_synthetic_cuda_candidate_mechanics_runs_100_plus_500(
    tmp_path: Path,
) -> None:
    plan = BenchmarkPlan("candidate", 100, 500, 899, 1000)
    observations = tuple(
        (
            update,
            BenchmarkObservation((f"matmul-{update}",), ("128x128",)),
        )
        for update in range(900, 1500)
    )
    run = run_benchmark(
        plan,
        _MatmulAdapter(tmp_path / "matmul-checkpoint.bin"),
        compile_probe=DisabledCompileCounterProbe(),
        trace_plan=PytorchTracePlan(
            tmp_path / "matmul-trace.json", "1" * 64, 5, True, True, True
        ),
        expected_data_sequence_sha256=hashlib.sha256(
            canonical_workload_artifact_bytes(observations, kind="data_sequence")
        ).hexdigest(),
        expected_shape_distribution_sha256=hashlib.sha256(
            canonical_workload_artifact_bytes(
                observations, kind="shape_distribution"
            )
        ).hexdigest(),
    )

    assert len(run.samples) == 500
    assert run.samples[0].successful_update == 1000
    assert run.samples[-1].successful_update == 1499
    assert min(sample.step_seconds for sample in run.samples) > 0.0
    assert max(sample.peak_cuda_reserved_bytes for sample in run.samples) > 0
    assert run.samples[0].checkpoint_bytes > 0
    assert run.trace.metrics.kernel_launches > 0


def test_real_cuda_successful_update_adapter_writes_bound_trace_in_measured_window(
    tmp_path: Path,
) -> None:
    module = nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    inputs = torch.randn((2, 64), device="cuda", dtype=torch.bfloat16)

    def measure(batch: torch.Tensor, timer: PhaseTimer) -> MeasuredMicrobatch:
        with timer.record("dit_forward"):
            prediction = module(batch)
        with timer.record("loss"):
            loss = prediction.float().square().mean(dim=1)
        return MeasuredMicrobatch(
            loss,
            128,
            2,
            2 * 2 * 64 * 64,
            ("linear-0", "linear-1"),
            ("2x64", "2x64"),
        )

    checkpoint_path = tmp_path / "adapter-checkpoint.bin"

    def checkpoint(update: int) -> tuple[Path, ...]:
        checkpoint_path.write_text(str(update), encoding="ascii")
        return (checkpoint_path,)

    adapter = SingleGpuStepBenchmarkAdapter[torch.Tensor](
        module=module,
        optimizer=optimizer,  # pyright: ignore[reportArgumentType]
        batches=iter(repeat(inputs)),
        measure_microbatch=measure,
        accumulation_steps=1,
        state=SingleGpuUpdateState(899, 899, 0),
        scheduler_step=lambda update: None,
        checkpoint_every_successful_updates=1000,
        checkpoint=checkpoint,
    )
    plan = BenchmarkPlan("candidate", 100, 500, 899, 1000)
    observations = tuple(
        (
            update,
            BenchmarkObservation(
                ("linear-0", "linear-1"),
                ("2x64", "2x64"),
            ),
        )
        for update in range(900, 1500)
    )
    run = run_benchmark(
        plan,
        adapter,
        compile_probe=DisabledCompileCounterProbe(),
        trace_plan=PytorchTracePlan(
            tmp_path / "trace.json", "1" * 64, 5, True, True, True
        ),
        expected_data_sequence_sha256=hashlib.sha256(
            canonical_workload_artifact_bytes(observations, kind="data_sequence")
        ).hexdigest(),
        expected_shape_distribution_sha256=hashlib.sha256(
            canonical_workload_artifact_bytes(
                observations, kind="shape_distribution"
            )
        ).hexdigest(),
    )
    trace = run.trace.entry

    assert trace.path.stat().st_size > 0
    assert trace.sha256 == stream_sha256(trace.path)
    assert trace.first_measured_update == 1
    assert trace.last_measured_update == 5
    assert run.samples[0].successful_update == 1000
    assert run.samples[0].checkpoint_bytes == len("1000")
    assert adapter.state.successful_updates == 1499
