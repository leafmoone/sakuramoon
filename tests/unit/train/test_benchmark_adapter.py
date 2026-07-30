from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.benchmark import (
    MeasuredMicrobatch,
    SingleGpuStepBenchmarkAdapter,
)
from sakuramoon.train.step import SingleGpuUpdateState


class _SgdAdapter:
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(parameters, lr=0.01)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def test_adapter_executes_actual_updates_and_returns_checkpoint_paths(
    tmp_path: Path,
) -> None:
    module = nn.Linear(2, 1, bias=False)
    optimizer = _SgdAdapter(list(module.parameters()))
    batches = iter(torch.ones(6, 1, 2))
    scheduler: list[int] = []

    def measure(batch: torch.Tensor, timer: PhaseTimer) -> MeasuredMicrobatch:
        with timer.record("dit_forward"):
            prediction = module(batch)
        with timer.record("loss"):
            loss = prediction.float().square().flatten(1).mean(1)
        return MeasuredMicrobatch(
            loss,
            2,
            1,
            4,
            ("sample",),
            ("1x2",),
        )

    def checkpoint(update: int) -> tuple[Path, ...]:
        path = tmp_path / f"checkpoint-{update}.bin"
        path.write_bytes(b"checkpoint")
        return (path,)

    adapter = SingleGpuStepBenchmarkAdapter[torch.Tensor](
        module=module,
        optimizer=optimizer,
        batches=batches,
        measure_microbatch=measure,
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
        scheduler_step=scheduler.append,
        checkpoint_every_successful_updates=2,
        checkpoint=checkpoint,
    )

    first = adapter.run_successful_update(1, measured=False)
    second = adapter.run_successful_update(2, measured=True)
    third = adapter.run_successful_update(3, measured=True)

    assert first.samples == second.samples == third.samples == 2
    assert second.checkpoint_paths == (tmp_path / "checkpoint-2.bin",)
    phases = second.phase_timer.collect_ready()
    assert phases["checkpoint"] > 0.0
    assert all(
        phases[phase] > 0.0
        for phase in ("dit_forward", "loss", "clip", "optimizer", "zero_grad")
    )
    assert third.checkpoint_paths == ()
    assert adapter.state == SingleGpuUpdateState(3, 3, 6)
    assert scheduler == [1, 2, 3]
    assert all(parameter.grad is None for parameter in module.parameters())


def test_adapter_rejects_noncontiguous_update() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    adapter = SingleGpuStepBenchmarkAdapter[torch.Tensor](
        module=nn.ParameterList([parameter]),
        optimizer=_SgdAdapter([parameter]),
        batches=iter((torch.tensor(1.0),)),
        measure_microbatch=lambda batch, timer: MeasuredMicrobatch(
            (parameter * batch).reshape(1),
            1,
            1,
            1,
            ("sample",),
            ("scalar",),
        ),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
        scheduler_step=lambda update: None,
        checkpoint_every_successful_updates=10,
        checkpoint=lambda update: (),
    )

    with pytest.raises(ValueError, match="contiguous"):
        adapter.run_successful_update(2, measured=True)
