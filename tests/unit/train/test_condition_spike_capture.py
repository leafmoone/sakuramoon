"""Optional experiment spike capture, without changing the stable clipping path."""
# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

import sakuramoon.train.step as step_module
from sakuramoon.config.schema import ConditionOptimizerConfig
from sakuramoon.optim.clip import ClipResult, clip_grad_norm_groups_fp32
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    _ConditionSpikeCapture,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Exact canonical identities, tiny synthetic tensors (no model assets).
        for name in sorted(_ConditionSpikeCapture._names):
            owner: nn.Module = self
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(owner, part):
                    owner.add_module(part, nn.Module())
                owner = owner.get_submodule(part)
            owner.register_parameter(parts[-1], nn.Parameter(torch.ones(1)))
        self.rest = nn.Parameter(torch.ones(1))


class _Optimizer:
    def step(self) -> None:
        pass

    def zero_grad(self, *, set_to_none: bool) -> None:
        pass


def _policy() -> ConditionOptimizerConfig:
    return ConditionOptimizerConfig(
        parameter_prefixes=("condition_tokens",), independent_clipping=True,
        clip_norm=1.0, rest_clip_norm=2.0,
    )


def _named(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple((name, value) for name, value in model.named_parameters() if name.startswith("condition_tokens."))


def _capture(path: Path, model: nn.Module) -> _ConditionSpikeCapture:
    named = _named(model)
    capture = _ConditionSpikeCapture.create(str(path), named, tuple(p for _, p in named), _policy())
    assert capture is not None
    return capture


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_observer_gets_exact_results_before_either_apply(threshold: float) -> None:
    condition, rest = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    condition.grad, rest.grad = torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0])
    seen: list[tuple[ClipResult, ClipResult]] = []
    def observe(first: ClipResult, second: ClipResult) -> None:
        assert torch.equal(cast(torch.Tensor, condition.grad), torch.tensor([3.0, 4.0]))
        assert torch.equal(cast(torch.Tensor, rest.grad), torch.tensor([0.0, 12.0]))
        assert first.pre_clip_norm.item() == 5.0 and second.pre_clip_norm.item() == 12.0
        seen.append((first, second))
    result = clip_grad_norm_groups_fp32(
        (condition,), (rest,), condition_max_norm=threshold, rest_max_norm=2.0,
        pre_clip_observer=observe,
    )
    assert len(seen) == 1 and seen[0][0] is result[0] and seen[0][1] is result[1]
    torch.testing.assert_close(condition.grad, torch.tensor([3.0, 4.0]) * (0.2 if threshold else 1.0))
    torch.testing.assert_close(rest.grad, torch.tensor([0.0, 2.0]))


@pytest.mark.parametrize("bad_partition", [0, 1])
def test_nonfinite_never_observed_or_partially_clipped(bad_partition: int) -> None:
    parameters = (nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1)))
    for parameter in parameters:
        parameter.grad = torch.tensor([4.0])
    parameters[bad_partition].grad = torch.tensor([float("inf")])
    before = [cast(torch.Tensor, parameter.grad).clone() for parameter in parameters]
    def forbidden(_first: ClipResult, _second: ClipResult) -> None:
        pytest.fail("observer ran before both finite checks")
    with pytest.raises(FloatingPointError):
        clip_grad_norm_groups_fp32(parameters[:1], parameters[1:], condition_max_norm=1.0, rest_max_norm=1.0, pre_clip_observer=forbidden)
    assert all(torch.equal(cast(torch.Tensor, parameter.grad), value) for parameter, value in zip(parameters, before, strict=True))


def test_observer_exception_aborts_without_modification() -> None:
    parameters = (nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1)))
    for parameter in parameters:
        parameter.grad = torch.tensor([4.0])
    def fail(_first: ClipResult, _second: ClipResult) -> None:
        raise RuntimeError("observer failed")
    with pytest.raises(RuntimeError, match="observer failed"):
        clip_grad_norm_groups_fp32(parameters[:1], parameters[1:], condition_max_norm=1.0, rest_max_norm=1.0, pre_clip_observer=fail)
    assert all(cast(torch.Tensor, parameter.grad).item() == 4.0 for parameter in parameters)


def test_no_env_does_not_create_or_scan_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAKURAMOON_CONDITION_SPIKE_LOG", raising=False)
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("disabled capture performed work")
    monkeypatch.setattr(_ConditionSpikeCapture, "create", forbidden)
    model = _Model()
    step = SingleGpuStep(model, _Optimizer(), accumulation_steps=1, state=SingleGpuUpdateState.initial(), condition_optimizer=_policy(), condition_grad_share=True)
    step.backward(sum((p.sum() for p in model.parameters()), torch.zeros(())).expand(2))
    result = step.finish_update()
    assert step._condition_spike_capture is None
    assert not hasattr(model, "_sakuramoon_condition_spike_capture")
    assert result.condition_grad_norm_pre_clip is not None
    assert result.condition_grad_norm_post_clip is not None


def test_capture_baseline_spike_cap_scalar_payload_and_parameter_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    model = _Model()
    destination = tmp_path / "external.jsonl"
    capture = _capture(destination, model)
    parameters = tuple(p for _, p in _named(model))
    original_state = tuple(model.state_dict())
    for iteration in range(25):
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 0.01 if iteration == 1 else 3.0)
        model.rest.grad = torch.tensor([4.0])
        clip_grad_norm_groups_fp32(parameters, (model.rest,), condition_max_norm=1.0, rest_max_norm=2.0, pre_clip_observer=partial(capture.observe, threshold=1.0, attempted_update=iteration + 1, last_successful_update=iteration))
    events = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(events) == 17 and events[0]["kind"] == "baseline"
    assert all(event["kind"] == "spike" for event in events[1:])
    assert capture.spikes == 16 and not capture.enabled
    assert events[0]["attempted_update"] == 1 and events[0]["last_successful_update"] == 0
    assert events[0]["phase"] == "post_sample_mean_pre_clip"
    rows = events[0]["parameters"]
    assert {row["fqn"] for row in rows} == _ConditionSpikeCapture._names
    assert all(row["l2_fp32"] == 3.0 and row["max_abs"] == 3.0 and row["finite"] for row in rows)
    assert sum(row["square_energy_share"] for row in rows) == pytest.approx(1.0)
    assert all(value is None or isinstance(value, (str, bool, int, float)) for row in rows for value in row.values())
    assert tuple(model.state_dict()) == original_state
    assert all(left is right for (_, left), right in zip(capture.named, parameters, strict=True))


def test_step_cache_persists_and_observes_sample_mean_before_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    path = tmp_path / "step.jsonl"
    monkeypatch.setenv("SAKURAMOON_CONDITION_SPIKE_LOG", str(path))
    model = _Model()
    first = SingleGpuStep(model, _Optimizer(), accumulation_steps=1, state=SingleGpuUpdateState.initial(), condition_optimizer=_policy())
    first.backward(sum((p.sum() * 3 for p in model.parameters()), torch.zeros(())).expand(4))
    result = first.finish_update()
    second = SingleGpuStep(model, _Optimizer(), accumulation_steps=1, state=result.state, condition_optimizer=_policy())
    assert first._condition_spike_capture is second._condition_spike_capture
    rows = json.loads(path.read_text())["parameters"]
    assert all(row["l2_fp32"] == 3.0 for row in rows)
    assert result.condition_clip is not None and result.condition_clip.post_clip_norm.item() == pytest.approx(1.0)


@pytest.mark.parametrize("case", ["rank1", "bad_rank", "identity", "policy", "relative", "internal"])
def test_capture_fails_closed_for_inapplicable_identity_or_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    monkeypatch.setenv("RANK", "1" if case == "rank1" else "00" if case == "bad_rank" else "0")
    model = _Model()
    named = _named(model)
    parameters = tuple(p for _, p in named)
    policy = _policy()
    destination = str(tmp_path / "outside.jsonl")
    if case == "identity":
        parameters = tuple(reversed(parameters))
    elif case == "policy":
        policy = None
    elif case == "relative":
        destination = "local.jsonl"
    elif case == "internal":
        destination = str(Path(step_module.__file__).resolve().parents[3] / "blocked.jsonl")
    assert _ConditionSpikeCapture.create(destination, named, parameters, policy) is None
    assert not (tmp_path / "outside.jsonl").exists()


def test_capture_io_failure_disables_without_changing_clipping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    model = _Model()
    capture = _capture(tmp_path / "missing-parent" / "capture.jsonl", model)
    parameters = tuple(p for _, p in _named(model))
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 3.0)
    model.rest.grad = torch.tensor([4.0])
    first, second = clip_grad_norm_groups_fp32(parameters, (model.rest,), condition_max_norm=1.0, rest_max_norm=2.0, pre_clip_observer=lambda first, second: capture.observe(first, second, threshold=1.0, attempted_update=1, last_successful_update=0))
    assert not capture.enabled
    assert first.post_clip_norm.item() == pytest.approx(1.0) and second.post_clip_norm.item() == pytest.approx(2.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_none_and_noop_observer_have_bitwise_identical_clipping(dtype: torch.dtype) -> None:
    reference = (nn.Parameter(torch.zeros(3, dtype=dtype)), nn.Parameter(torch.zeros(2, dtype=dtype)))
    candidate = (nn.Parameter(torch.zeros(3, dtype=dtype)), nn.Parameter(torch.zeros(2, dtype=dtype)))
    for pair in (reference, candidate):
        pair[0].grad = torch.tensor([3.0, -2.0, 1.25], dtype=dtype)
        pair[1].grad = torch.tensor([2.5, -4.25], dtype=dtype)
    expected = clip_grad_norm_groups_fp32(reference[:1], reference[1:], condition_max_norm=0.7, rest_max_norm=1.3)
    actual = clip_grad_norm_groups_fp32(candidate[:1], candidate[1:], condition_max_norm=0.7, rest_max_norm=1.3, pre_clip_observer=lambda _first, _second: None)
    for before, after in zip(reference, candidate, strict=True):
        assert torch.equal(cast(torch.Tensor, before.grad), cast(torch.Tensor, after.grad))
    for before, after in zip(expected, actual, strict=True):
        assert torch.equal(before.pre_clip_norm, after.pre_clip_norm)
        assert torch.equal(before.coefficient, after.coefficient)
        assert torch.equal(before.post_clip_norm, after.post_clip_norm)
