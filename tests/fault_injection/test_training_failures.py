from __future__ import annotations

import errno
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

import sakuramoon.train.loop as loop_module
from sakuramoon.train.loop import SingleGpuTrainingLoop
from sakuramoon.train.step import SingleGpuStep, SingleGpuUpdateState


class _SgdAdapter:
    def __init__(self, parameter: nn.Parameter) -> None:
        self.optimizer = torch.optim.SGD([parameter], lr=0.1)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def test_finite_loss_with_nonfinite_gradient_does_not_update_or_continue(
    tmp_path: Path,
) -> None:
    del tmp_path
    parameter = nn.Parameter(torch.tensor(1.0))

    def poison_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, float("nan"))

    parameter.register_hook(poison_gradient)  # pyright: ignore[reportUnknownMemberType]
    optimizer = _SgdAdapter(parameter)
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        optimizer,
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    before = parameter.detach().clone()
    step.backward(parameter.square().reshape(1))

    with pytest.raises(FloatingPointError, match="nonfinite"):
        step.finish_update()

    torch.testing.assert_close(parameter, before, atol=0, rtol=0)
    assert parameter.grad is None
    assert step.state == SingleGpuUpdateState(1, 0, 0)
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


def test_diagnostic_enospc_preserves_original_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = _SgdAdapter(parameter)

    def fail_diagnostic(*_args: object, **_kwargs: object) -> Path:
        raise OSError(errno.ENOSPC, "injected diagnostic disk full")

    monkeypatch.setattr(loop_module, "write_failure_bundle", fail_diagnostic)
    loop = SingleGpuTrainingLoop[torch.Tensor](
        module=nn.ParameterList([parameter]),
        optimizer=optimizer,
        loss_fn=lambda batch: (parameter * batch).reshape(1),
        accumulation_steps=1,
        target_successful_updates=1,
        checkpoint_every_successful_updates=1,
        scheduler_step=lambda update: None,
        checkpoint=lambda update: None,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: "enospc",
        state=SingleGpuUpdateState.initial(),
    )

    with pytest.raises(ExceptionGroup) as captured:
        loop.run((torch.tensor(float("nan")),))

    assert [type(error) for error in captured.value.exceptions] == [
        FloatingPointError,
        OSError,
    ]
    assert cast(OSError, captured.value.exceptions[1]).errno == errno.ENOSPC
    assert loop.state == SingleGpuUpdateState(1, 0, 0)
