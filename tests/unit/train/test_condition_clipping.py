from __future__ import annotations

import copy
import math
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from sakuramoon.config.schema import ConditionOptimizerConfig, OptimizerConfig
from sakuramoon.optim.clip import clip_grad_norm_fp32, clip_grad_norm_groups_fp32
from sakuramoon.train.loop import SingleGpuTrainingLoop, SuccessfulLoopObservation
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateResult,
    SingleGpuUpdateState,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.condition_tokens = nn.Linear(1, 1, bias=False)
        self.condition_tokens_extra = nn.Parameter(torch.ones(1))
        self.irepa_alignment = nn.Parameter(torch.ones(1))
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        with torch.no_grad():
            self.condition_tokens.weight.fill_(1.0)

    def loss(self, condition: float, rest: float, samples: int) -> torch.Tensor:
        return (
            condition * self.condition_tokens.weight.sum()
            + rest * self.condition_tokens_extra.sum()
            + 0.0 * self.irepa_alignment.sum()
        ).expand(samples)


class _Optimizer:
    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9)
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _policy(condition: float = 1.0, rest: float = 2.0) -> ConditionOptimizerConfig:
    return ConditionOptimizerConfig(
        parameter_prefixes=("condition_tokens",),
        independent_clipping=True,
        clip_norm=condition,
        rest_clip_norm=rest,
    )


@pytest.mark.parametrize("threshold", [0.0, 0.25, 1.0, 2.5, 10.0])
def test_global_thresholds_and_zero_disable(threshold: float) -> None:
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    result = clip_grad_norm_fp32((parameter,), max_norm=threshold)
    expected = min(5.0, threshold) if threshold else 5.0
    torch.testing.assert_close(result.pre_clip_norm, torch.tensor(5.0))
    torch.testing.assert_close(result.post_clip_norm, torch.tensor(expected))
    torch.testing.assert_close(parameter.grad.square().sum().sqrt(), torch.tensor(expected))


@pytest.mark.parametrize("threshold", [0.0, 0.25, 2.5])
def test_step_global_clip_norm_is_forwarded(threshold: float) -> None:
    model = _Model()
    step = SingleGpuStep(
        model, _Optimizer(model.parameters()), accumulation_steps=1,
        state=SingleGpuUpdateState.initial(), global_clip_norm=threshold,
    )
    step.backward(model.loss(3.0, 4.0, 2))
    result = step.finish_update()
    assert result.clip.pre_clip_norm.item() == 5.0
    assert math.isclose(result.clip.post_clip_norm.item(), threshold or 5.0, rel_tol=1e-6)


def test_disabled_global_clipping_still_rejects_nonfinite() -> None:
    parameter = nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError, match="nonfinite"):
        clip_grad_norm_fp32((parameter,), max_norm=0.0)


def test_all_remaining_trainables_including_irepa_share_rest_threshold() -> None:
    model = _Model()
    step = SingleGpuStep(
        model, _Optimizer(model.parameters()), accumulation_steps=1,
        state=SingleGpuUpdateState.initial(), condition_optimizer=_policy(),
        irepa_projector=True,
    )
    step.backward(model.loss(3.0, 4.0, 2) + 3.0 * model.irepa_alignment.sum())
    result = step.finish_update()
    assert result.rest_clip is not None
    assert result.rest_clip.pre_clip_norm.item() == 5.0
    assert result.rest_clip.post_clip_norm.item() == 2.0
    assert result.irepa_projector_grad_norm == 0.0  # canonical projector is a submodule
    torch.testing.assert_close(model.condition_tokens_extra, torch.tensor([0.84]))
    torch.testing.assert_close(model.irepa_alignment, torch.tensor([0.88]))


@pytest.mark.parametrize("norm,threshold,expected", [(1e-14, 1e-15, 0.1), (1e-14, 2e-14, 1.0), (0.0, 1e-15, 1.0)])
def test_tiny_positive_threshold_uses_true_norm(norm: float, threshold: float, expected: float) -> None:
    parameter = nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([norm])
    result = clip_grad_norm_fp32((parameter,), max_norm=threshold)
    assert math.isclose(result.pre_clip_norm.item(), norm, rel_tol=1e-6)
    assert math.isclose(result.coefficient.item(), expected, rel_tol=1e-6)
    assert math.isclose(parameter.grad.item(), norm * expected, rel_tol=1e-6)
    assert math.isclose(result.post_clip_norm.item(), norm * expected, rel_tol=1e-6)
    assert (result.coefficient.item() < 1.0) == (expected < 1.0)


@pytest.mark.parametrize("kind", ["none", "zero", "empty"])
def test_tiny_threshold_on_absent_or_zero_group_is_not_clipped(kind: str) -> None:
    condition, rest = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    if kind == "zero":
        condition.grad = torch.zeros(1)
    rest.grad = torch.tensor([1e-14])
    first, second = clip_grad_norm_groups_fp32(
        () if kind == "empty" else (condition,), (rest,),
        condition_max_norm=1e-15, rest_max_norm=1e-15,
    )
    assert first.pre_clip_norm.item() == first.post_clip_norm.item() == 0.0
    assert first.coefficient.item() == 1.0
    assert math.isclose(second.coefficient.item(), 0.1, rel_tol=1e-6)
    assert math.isclose(rest.grad.item(), 1e-15, rel_tol=1e-6)


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf"), True])
def test_invalid_thresholds_do_not_modify_gradients(threshold: float) -> None:
    parameter = nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones(1)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        clip_grad_norm_fp32((parameter,), max_norm=threshold)
    assert parameter.grad.item() == 1.0


@pytest.mark.parametrize("bad_partition", ["condition", "rest"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 3.0e38])
def test_both_groups_checked_before_either_is_mutated(bad_partition: str, bad: float) -> None:
    condition, rest = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    condition.grad = torch.tensor([bad if bad_partition == "condition" else 10.0])
    rest.grad = torch.tensor([bad if bad_partition == "rest" else 10.0])
    before = (condition.grad.clone(), rest.grad.clone())
    with pytest.raises(FloatingPointError, match="nonfinite"):
        clip_grad_norm_groups_fp32(
            (condition,), (rest,), condition_max_norm=1.0, rest_max_norm=0.0,
        )
    torch.testing.assert_close(condition.grad, before[0], equal_nan=True)
    torch.testing.assert_close(rest.grad, before[1], equal_nan=True)


def test_group_none_zero_and_empty_rest() -> None:
    condition, rest = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    rest.grad = torch.zeros(1)
    first, second = clip_grad_norm_groups_fp32(
        (condition,), (rest,), condition_max_norm=0.0, rest_max_norm=0.5,
    )
    assert condition.grad is None
    assert first.pre_clip_norm.item() == second.pre_clip_norm.item() == 0.0
    assert first.coefficient.item() == second.coefficient.item() == 1.0
    first, second = clip_grad_norm_groups_fp32(
        (condition,), (), condition_max_norm=0.5, rest_max_norm=0.5,
    )
    assert first.post_clip_norm.item() == second.post_clip_norm.item() == 0.0
    with pytest.raises(RuntimeError, match="empty gradient"):
        clip_grad_norm_fp32((condition,), max_norm=1.0)


def test_independent_clipping_after_common_sample_mean_and_all_rest() -> None:
    model = _Model()
    optimizer = _Optimizer(model.parameters())
    step = SingleGpuStep(
        model, optimizer, accumulation_steps=2, state=SingleGpuUpdateState.initial(),
        global_clip_norm=0.01, condition_optimizer=_policy(), irepa_projector=True,
    )
    step.backward(model.loss(3.0, 4.0, 1))
    step.backward(model.loss(3.0, 4.0, 3))
    result = step.finish_update()
    assert result.condition_clip is not None and result.rest_clip is not None
    assert result.condition_clip.pre_clip_norm.item() == 3.0
    assert result.rest_clip.pre_clip_norm.item() == 4.0
    assert result.condition_clip.post_clip_norm.item() == 1.0
    assert result.rest_clip.post_clip_norm.item() == 2.0
    assert result.condition_encoder_grad_norm.item() == 3.0
    assert result.irepa_projector_grad_norm == 0.0
    assert result.clip.pre_clip_norm.item() == 5.0
    assert math.isclose(result.clip.post_clip_norm.item(), 5**0.5, rel_tol=1e-6)
    assert math.isclose(result.clip.coefficient.item(), 5**0.5/5, rel_tol=1e-6)
    assert result.clip.post_clip_norm.item() > 1.0
    torch.testing.assert_close(model.condition_tokens.weight, torch.tensor([[0.9]]))
    torch.testing.assert_close(model.condition_tokens_extra, torch.tensor([0.8]))
    assert model.frozen.item() == model.irepa_alignment.item() == 1.0
    assert result.state == SingleGpuUpdateState(1, 1, 4)
    assert optimizer.steps == 1


@pytest.mark.parametrize("unused", ["condition", "rest", "both"])
def test_step_accepts_unused_partition_gradients(unused: str) -> None:
    model = _Model()
    optimizer = _Optimizer(model.parameters())
    step = SingleGpuStep(
        model, optimizer, accumulation_steps=1, state=SingleGpuUpdateState.initial(),
        condition_optimizer=_policy(),
    )
    step.backward(model.loss(0.0, 0.0, 1))
    if unused in ("condition", "both"):
        model.condition_tokens.weight.grad = None
    if unused in ("rest", "both"):
        model.condition_tokens_extra.grad = model.irepa_alignment.grad = None
    result = step.finish_update()
    assert result.condition_clip is not None and result.rest_clip is not None
    assert result.condition_clip.pre_clip_norm.item() == result.rest_clip.pre_clip_norm.item() == 0.0
    assert result.clip.coefficient.item() == 1.0
    assert optimizer.steps == 1


@pytest.mark.parametrize("bad_partition", ["condition", "rest"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 3.0e38])
def test_step_nonfinite_fails_and_poison_clears(bad_partition: str, bad: float) -> None:
    model = _Model()
    optimizer = _Optimizer(model.parameters())
    step = SingleGpuStep(
        model, optimizer, accumulation_steps=1, state=SingleGpuUpdateState.initial(),
        condition_optimizer=_policy(),
    )
    step.backward(model.loss(3.0, 4.0, 2))
    condition_gradient = model.condition_tokens.weight.grad
    assert condition_gradient is not None
    if bad_partition == "rest":
        model.condition_tokens_extra.grad = torch.tensor([bad])
    else:
        model.condition_tokens.weight.grad = torch.tensor([[bad]])
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError, match="nonfinite"):
        step.finish_update()
    if not math.isfinite(bad):
        assert condition_gradient.item() == 6.0
    assert optimizer.steps == 0
    assert step.state == SingleGpuUpdateState(1, 0, 0)
    assert step.detection_phase == "clip"
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, before[name])
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(RuntimeError, match="failed update"):
        step.finish_update()


@pytest.mark.parametrize("prefix", ["missing", "frozen", "condition_tokens.missing"])
def test_unmatched_or_frozen_condition_fails_at_construction(prefix: str) -> None:
    model = _Model()
    policy = _policy().model_copy(update={"parameter_prefixes": (prefix,)})
    with pytest.raises(ValueError, match="no trainable parameters"):
        SingleGpuStep(
            model, _Optimizer(model.parameters()), accumulation_steps=1,
            state=SingleGpuUpdateState.initial(), condition_optimizer=policy,
        )


def test_disabled_condition_is_exact_legacy_including_momentum_and_state() -> None:
    legacy, disabled = _Model(), _Model()
    optimizers = [_Optimizer(model.parameters()) for model in (legacy, disabled)]
    steps = [
        SingleGpuStep(
            model, optimizer, accumulation_steps=1, state=SingleGpuUpdateState.initial(),
            condition_optimizer=policy,
        )
        for model, optimizer, policy in zip(
            (legacy, disabled), optimizers, (None, ConditionOptimizerConfig()), strict=True,
        )
    ]
    for _ in range(3):
        results: list[SingleGpuUpdateResult] = []
        for model, step in zip((legacy, disabled), steps, strict=True):
            step.backward(model.loss(3.0, 4.0, 3))
            results.append(step.finish_update())
        assert results[0].state == results[1].state
        assert results[1].condition_clip is results[1].rest_clip is None
        for attribute in ("pre_clip_norm", "post_clip_norm", "coefficient"):
            assert torch.equal(getattr(results[0].clip, attribute), getattr(results[1].clip, attribute))
        for first, second in zip(legacy.parameters(), disabled.parameters(), strict=True):
            assert torch.equal(first, second)
            first_state = optimizers[0].optimizer.state[first]
            second_state = optimizers[1].optimizer.state[second]
            assert first_state.keys() == second_state.keys()
            for name in first_state:
                assert torch.equal(first_state[name], second_state[name])


@pytest.mark.parametrize("controls", [
    {"independent_clipping": True},
    {"independent_lr": True},
    {"parameter_prefixes": ["condition_tokens"]},
    {"clip_norm": 1.0},
    {"rest_clip_norm": 1.0},
    {"lr_multiplier": 2.0},
    {"independent_lr": True, "clip_norm": 1.0, "parameter_prefixes": ["condition_tokens"]},
])
def test_schema_rejects_missing_or_inactive_knobs(controls: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ConditionOptimizerConfig.model_validate(controls)


@pytest.mark.parametrize("prefixes", [
    [""], [" condition_tokens"], ["condition_tokens."], ["condition_tokens..weight"],
    ["condition_tokens.*"], ["module.condition_tokens"], ["_orig_mod.condition_tokens"],
    ["condition_tokens/weight"], ["condition_tokens", "condition_tokens"],
    ["condition_tokens", "condition_tokens.weight"],
])
def test_schema_rejects_noncanonical_duplicate_overlapping_prefixes(prefixes: list[str]) -> None:
    with pytest.raises(ValueError):
        ConditionOptimizerConfig.model_validate({
            "parameter_prefixes": prefixes, "independent_lr": True,
        })


@pytest.mark.parametrize("knob", ["clip_norm", "rest_clip_norm", "lr_multiplier"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_schema_rejects_invalid_numeric_controls(knob: str, value: object) -> None:
    controls = _policy().model_dump()
    controls["independent_lr"] = True
    controls[knob] = value
    with pytest.raises((ValueError, TypeError)):
        ConditionOptimizerConfig.model_validate(controls)


@pytest.mark.parametrize("condition_threshold,rest_threshold", [(0.43, 0.91), (0.0, 2.7), (1.9, 0.0)])
def test_arbitrary_independent_thresholds(condition_threshold: float, rest_threshold: float) -> None:
    model = _Model()
    optimizer = _Optimizer(model.parameters())
    step = SingleGpuStep(
        model, optimizer, accumulation_steps=1, state=SingleGpuUpdateState.initial(),
        condition_optimizer=_policy(condition_threshold, rest_threshold),
    )
    step.backward(model.loss(3.0, 4.0, 3))
    result = step.finish_update()
    assert result.condition_clip is not None and result.rest_clip is not None
    assert math.isclose(result.condition_clip.post_clip_norm.item(), min(3.0, condition_threshold) if condition_threshold else 3.0, rel_tol=1e-6)
    assert math.isclose(result.rest_clip.post_clip_norm.item(), min(4.0, rest_threshold) if rest_threshold else 4.0, rel_tol=1e-6)


@pytest.mark.parametrize("threshold,policy", [(0.0, None), (0.43, None), (0.001, _policy(0.43, 0.91))])
def test_loop_forwards_clipping_controls(tmp_path: Path, threshold: float, policy: ConditionOptimizerConfig | None) -> None:
    model = _Model()
    optimizer = _Optimizer(model.parameters())
    observations: list[SuccessfulLoopObservation] = []
    loop = SingleGpuTrainingLoop[None](
        module=model, optimizer=optimizer,
        loss_fn=lambda _batch, _microbatch: model.loss(3.0, 4.0, 3), accumulation_steps=1,
        target_successful_updates=1, checkpoint_every_successful_updates=100,
        scheduler_step=lambda _update: None, checkpoint=lambda _update: None,
        diagnostic_root=tmp_path, failure_id=lambda _phase, _state: "cpu-test",
        state=SingleGpuUpdateState.initial(),
        global_clip_norm=threshold, condition_optimizer=policy,
        successful_update_observer=observations.append,
    )
    loop.run([None])
    result = observations[0].update
    if policy is None:
        assert result.condition_clip is None
        assert math.isclose(result.clip.post_clip_norm.item(), threshold or 5.0, rel_tol=1e-6)
    else:
        assert result.condition_clip is not None and result.rest_clip is not None
        assert math.isclose(result.condition_clip.post_clip_norm.item(), 0.43, rel_tol=1e-6)
        assert math.isclose(result.rest_clip.post_clip_norm.item(), 0.91, rel_tol=1e-6)


@pytest.mark.parametrize("policy", [None, _policy(0.43, 0.91)])
def test_perf_harness_forwards_clipping_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy: ConditionOptimizerConfig | None,
) -> None:
    from types import SimpleNamespace

    from sakuramoon.perf import harness

    captured: dict[str, object] = {}
    state = SingleGpuUpdateState.initial()

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, _batches: object) -> SimpleNamespace:
            return SimpleNamespace(state=state)

    monkeypatch.setattr(harness, "SingleGpuTrainingLoop", FakeLoop)
    config = SimpleNamespace(
        train=SimpleNamespace(accumulation=1),
        gradient=SimpleNamespace(clip_norm=0.37),
        optimizer=SimpleNamespace(condition=policy),
    )
    assembly = cast(harness.BenchmarkAssembly, SimpleNamespace(
        config=config, device=torch.device("cpu"), runtime=object(),
        module=_Model(), optimizer=object(), scheduler=object(),
        world_size=1, backward=None, no_sync=None, batches=(None,),
    ))
    harness.run_benchmark_stage(
        assembly, state=state, target_successful_updates=1, diagnostic_root=tmp_path,
    )
    assert captured["global_clip_norm"] == 0.37
    assert captured["condition_optimizer"] is policy


def test_schema_toml_condition_table_and_b_only() -> None:
    document = tomllib.loads('''
[optimizer]
name = "torchao_adamw8bit"
base_lr = 0.001
reference_batch = 1
lr_scaling = "none"
betas = [0.9, 0.95]
eps = 0.00000001
block_size = 256
bf16_stochastic_round = false
matrix_weight_decay = 0.0
sensitive_weight_decay = 0.0
[optimizer.condition]
parameter_prefixes = ["condition_tokens", "dit.conditioner.0"]
independent_clipping = true
clip_norm = 0
rest_clip_norm = 2.75
independent_lr = true
lr_multiplier = 0.25
''')
    config = OptimizerConfig.model_validate(document["optimizer"])
    assert config.condition is not None
    assert config.condition.clip_norm == 0.0
    assert config.condition.rest_clip_norm == 2.75
    assert config.condition.lr_multiplier == 0.25
    b_only = ConditionOptimizerConfig.model_validate({
        "parameter_prefixes": ["condition_tokens"], "independent_lr": True,
    })
    assert b_only.clip_norm is b_only.rest_clip_norm is None
    assert OptimizerConfig.model_validate({
        **cast(dict[str, object], document["optimizer"]), "condition": None,
    }).condition is None
