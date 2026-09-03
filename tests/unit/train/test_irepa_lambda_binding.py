"""Phase-4 lambda binding: runtime setter contract and loop wiring.

The lambda is a pure function of the successful-update number (tested at the
objective level in tests/unit/objective/test_irepa_objective.py).  This file
covers the two runtime boundaries around it: the ``set_irepa_weight``
binder on the batch runtime and the loop's DDP-transparent detection of the
projector child.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.objective.irepa import IRepaLambdaSchedule, irepa_weight_for_update
from sakuramoon.train.loop import SingleGpuTrainingLoop
from sakuramoon.train.runtime import SingleGpuBatchRuntime
from sakuramoon.train.step import SingleGpuUpdateState


def _bare_runtime(teacher: object | None) -> SingleGpuBatchRuntime:
    """A runtime instance without the CUDA-only __init__ (setter tests only)."""

    runtime = SingleGpuBatchRuntime.__new__(SingleGpuBatchRuntime)
    runtime.irepa_teacher = teacher
    runtime.irepa_weight = 0.0
    return runtime


def _schedule() -> IRepaLambdaSchedule:
    return IRepaLambdaSchedule(
        start_successful_update=100,
        target_weight=0.5,
        ramp_in_updates=10,
        ramp_out_after_updates=None,
        ramp_out_updates=10,
    )


def test_set_irepa_weight_requires_an_installed_teacher() -> None:
    runtime = _bare_runtime(teacher=None)
    with pytest.raises(ValueError, match="installed teacher"):
        runtime.set_irepa_weight(0.0)
    assert runtime.irepa_weight == 0.0


def test_set_irepa_weight_accepts_finite_nonnegative_floats() -> None:
    runtime = _bare_runtime(teacher=object())
    for value in (0.0, 0.25, 1e-07, 3.5):
        runtime.set_irepa_weight(value)
        assert runtime.irepa_weight == value


def test_set_irepa_weight_rejects_invalid_values() -> None:
    runtime = _bare_runtime(teacher=object())
    for value in (-0.1, float("nan"), float("inf"), 1, True, "0.5"):
        with pytest.raises(ValueError, match="finite nonnegative float"):
            runtime.set_irepa_weight(value)  # type: ignore[arg-type]
    assert runtime.irepa_weight == 0.0


def test_weight_rebinding_is_resume_stable_across_instances() -> None:
    # a resume rebuilds a fresh runtime; rebinding the same successful-
    # update number must reproduce the same lambda exactly
    schedule = _schedule()
    updates = (99, 100, 105, 110, 150)
    first = _bare_runtime(teacher=object())
    second = _bare_runtime(teacher=object())
    for update in updates:
        first.set_irepa_weight(schedule.weight_for_update(update))
        second.set_irepa_weight(schedule.weight_for_update(update))
        assert first.irepa_weight == second.irepa_weight
        assert first.irepa_weight == pytest.approx(
            schedule.weight_for_update(update)
        )


def test_failed_update_retry_rebinds_the_same_lambda() -> None:
    # a failed update does not advance the successful-update counter, so the
    # retry binds the identical lambda (pure function of the same number)
    schedule = _schedule()
    pending_update = 107
    weight_before_failure = irepa_weight_for_update(
        successful_update=pending_update,
        start_successful_update=schedule.start_successful_update,
        target_weight=schedule.target_weight,
        ramp_in_updates=schedule.ramp_in_updates,
        ramp_out_after_updates=schedule.ramp_out_after_updates,
        ramp_out_updates=schedule.ramp_out_updates,
    )
    runtime = _bare_runtime(teacher=object())
    runtime.set_irepa_weight(weight_before_failure)
    # ... update fails; the loop retries the same successful update number
    retry_weight = schedule.weight_for_update(pending_update)
    runtime.set_irepa_weight(retry_weight)
    assert runtime.irepa_weight == weight_before_failure


class _ProjectorCarryingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.irepa_alignment = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _PlainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.weight)


def _run_loop(
    module: nn.Module, tmp_path, observed: list[float]
) -> list[float]:
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    def record(observation) -> None:
        observed.append(observation.update.irepa_projector_grad_norm)

    def loss_fn(batch) -> torch.Tensor:
        x, y = batch
        main = (module(x) - y).square().mean(dim=1)
        if hasattr(module, "irepa_alignment"):
            # the projector is trained through the alignment term, exactly
            # as the composite trains irepa_alignment.*
            return main + 0.5 * module.irepa_alignment(x).square().mean(dim=1)
        return main

    loop = SingleGpuTrainingLoop(
        module=module,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accumulation_steps=1,
        target_successful_updates=2,
        checkpoint_every_successful_updates=100,
        scheduler_step=lambda _update: None,
        checkpoint=lambda _update: None,
        diagnostic_root=tmp_path,
        failure_id=lambda _phase, _state: "irepa-lambda-test",
        state=SingleGpuUpdateState.initial(),
        successful_update_observer=record,
    )
    batches = (
        (torch.randn(3, 4), torch.randn(3, 4)),
        (torch.randn(3, 4), torch.randn(3, 4)),
    )
    result = loop.run(batches)
    assert result.state.successful_updates == 2
    return observed


def test_training_loop_computes_projector_grad_norm_when_child_present(
    tmp_path,
) -> None:
    torch.manual_seed(11)
    observed: list[float] = []
    _run_loop(_ProjectorCarryingModel(), tmp_path, observed)
    assert len(observed) == 2
    # the projector child is trained, so its grad norm is real, not the 0.0
    # default
    assert all(norm > 0.0 for norm in observed)


def test_training_loop_keeps_zero_norm_without_the_projector_child(tmp_path) -> None:
    torch.manual_seed(11)
    observed: list[float] = []
    _run_loop(_PlainModel(), tmp_path, observed)
    assert observed == [0.0, 0.0]


def test_schedule_delegation_matches_the_pure_function() -> None:
    schedule = _schedule()
    for update in (99, 100, 105, 110, 111):
        assert schedule.weight_for_update(update) == irepa_weight_for_update(
            successful_update=update,
            start_successful_update=schedule.start_successful_update,
            target_weight=schedule.target_weight,
            ramp_in_updates=schedule.ramp_in_updates,
            ramp_out_after_updates=schedule.ramp_out_after_updates,
            ramp_out_updates=schedule.ramp_out_updates,
        )
