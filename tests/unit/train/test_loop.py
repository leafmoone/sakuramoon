from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.train.loop import SingleGpuTrainingLoop
from sakuramoon.train.step import SingleGpuUpdateState


class _SgdAdapter:
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(parameters, lr=0.01)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def test_loop_drives_scheduler_and_checkpoint_only_by_successful_update(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = _SgdAdapter(list(model.parameters()))
    scheduler_updates: list[int] = []
    checkpoint_updates: list[int] = []

    def loss_fn(batch: torch.Tensor) -> torch.Tensor:
        return model(batch).float().square().flatten(1).mean(1)

    loop = SingleGpuTrainingLoop[torch.Tensor](
        module=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accumulation_steps=2,
        target_successful_updates=3,
        checkpoint_every_successful_updates=2,
        scheduler_step=scheduler_updates.append,
        checkpoint=checkpoint_updates.append,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
        state=SingleGpuUpdateState.initial(),
    )

    result = loop.run(torch.ones(6, 1, 2))

    assert result.state == SingleGpuUpdateState(3, 3, 6)
    assert scheduler_updates == [1, 2, 3]
    assert checkpoint_updates == [2]
    assert result.checkpoint_updates == (2,)
    assert not (tmp_path / "diagnostics").exists()


def test_loop_nonfinite_failure_writes_bundle_without_advancing_success(
    tmp_path: Path,
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    module = nn.ParameterList([parameter])
    optimizer = _SgdAdapter([parameter])
    scheduler_updates: list[int] = []

    def loss_fn(batch: torch.Tensor) -> torch.Tensor:
        return (parameter * batch).reshape(1)

    loop = SingleGpuTrainingLoop[torch.Tensor](
        module=module,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accumulation_steps=1,
        target_successful_updates=1,
        checkpoint_every_successful_updates=1,
        scheduler_step=scheduler_updates.append,
        checkpoint=lambda update: None,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
        state=SingleGpuUpdateState.initial(),
    )

    with pytest.raises(FloatingPointError, match="nonfinite"):
        loop.run((torch.tensor(float("nan")),))

    assert loop.state == SingleGpuUpdateState(1, 0, 0)
    assert scheduler_updates == []
    bundle = tmp_path / "diagnostics/update-1"
    assert (bundle / "COMPLETE").is_file()
    payload = json.loads((bundle / "failure.json").read_text())
    assert payload == {
        "attempted_updates": 1,
        "effective_samples": 0,
        "error_type": "FloatingPointError",
        "failure_id": "update-1",
        "phase": "update",
        "successful_updates": 0,
    }


def test_post_update_failure_preserves_successful_counter_and_stops(
    tmp_path: Path,
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    module = nn.ParameterList([parameter])
    optimizer = _SgdAdapter([parameter])

    def fail_checkpoint(update: int) -> None:
        raise OSError(f"synthetic checkpoint failure at {update}")

    loop = SingleGpuTrainingLoop[torch.Tensor](
        module=module,
        optimizer=optimizer,
        loss_fn=lambda batch: (parameter * batch).square().reshape(1),
        accumulation_steps=1,
        target_successful_updates=2,
        checkpoint_every_successful_updates=1,
        scheduler_step=lambda update: None,
        checkpoint=fail_checkpoint,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: f"{phase}-{state.successful_updates}",
        state=SingleGpuUpdateState.initial(),
    )

    with pytest.raises(OSError, match="checkpoint failure"):
        loop.run((torch.tensor(1.0), torch.tensor(1.0)))

    assert loop.state == SingleGpuUpdateState(1, 1, 1)
    assert (tmp_path / "diagnostics/post_update-1/COMPLETE").is_file()


@pytest.mark.parametrize("failure", ["input", "loss"])
def test_interrupted_accumulation_clears_gradients_and_counts_attempt(
    tmp_path: Path, failure: str
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    module = nn.ParameterList([parameter])
    optimizer = _SgdAdapter([parameter])

    def batches() -> object:
        yield torch.tensor(1.0)
        if failure == "input":
            raise OSError("synthetic input failure")
        yield torch.tensor(float("inf"))

    def loss_fn(batch: torch.Tensor) -> torch.Tensor:
        if failure == "loss" and not bool(torch.isfinite(batch).item()):
            raise RuntimeError("synthetic loss failure")
        return (parameter * batch).square().reshape(1)

    loop = SingleGpuTrainingLoop[torch.Tensor](
        module=module,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accumulation_steps=2,
        target_successful_updates=1,
        checkpoint_every_successful_updates=1,
        scheduler_step=lambda update: None,
        checkpoint=lambda update: None,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
        state=SingleGpuUpdateState.initial(),
    )

    expected = OSError if failure == "input" else RuntimeError
    with pytest.raises(expected):
        loop.run(batches())  # pyright: ignore[reportArgumentType]

    assert loop.state == SingleGpuUpdateState(1, 0, 0)
    assert parameter.grad is None
    assert (tmp_path / "diagnostics/update-1/COMPLETE").is_file()


def test_diagnostic_publication_failure_preserves_training_error(
    tmp_path: Path,
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    loop = SingleGpuTrainingLoop[torch.Tensor](
        module=nn.ParameterList([parameter]),
        optimizer=_SgdAdapter([parameter]),
        loss_fn=lambda batch: (parameter * batch).reshape(1),
        accumulation_steps=1,
        target_successful_updates=1,
        checkpoint_every_successful_updates=1,
        scheduler_step=lambda update: None,
        checkpoint=lambda update: None,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: "same-id",
        state=SingleGpuUpdateState.initial(),
    )
    (tmp_path / "diagnostics" / "same-id").mkdir(parents=True)

    with pytest.raises(ExceptionGroup) as captured:
        loop.run((torch.tensor(float("nan")),))

    assert [type(exc) for exc in captured.value.exceptions] == [
        FloatingPointError,
        FileExistsError,
    ]
    assert loop.state == SingleGpuUpdateState(1, 0, 0)
