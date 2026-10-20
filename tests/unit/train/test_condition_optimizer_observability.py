"""End-to-end observability of the merged condition optimizer paths.

Every assertion is against real CPU optimizer state, gradients and update
trajectories rather than config plumbing:

* global vs independent clipping produces real norms, coefficients and
  weight deltas through a real ``torch.optim`` optimizer;
* the ``condition_grad_share`` diagnostic side (``condition_tokens.*`` plus
  ``dit.conditioner.*``) is strictly wider than the optimizer partition
  (``condition_tokens`` only), and both are checked against exact norms;
* diagnostics never change weights, clip results or optimizer state;
* the loop-level cadence callback toggles the diagnostics per update, and a
  failing microbatch loss through the new two-argument ``loss_fn`` still
  propagates instead of being swallowed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.config.schema import ConditionOptimizerConfig
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.condition import split_condition_lr_groups
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG
from sakuramoon.train.condition_diagnostics_stream import (
    build_condition_diagnostic_record,
)
from sakuramoon.train.loop import SingleGpuTrainingLoop, SuccessfulLoopObservation
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateResult,
    SingleGpuUpdateState,
)

SAMPLES = 3


class _Conditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.weight = nn.Parameter(torch.ones(1))


class _DiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.conditioner = _Conditioner()


class _ObsModel(nn.Module):
    """Sample-mean scaled gradients of exactly (3, 2, 4).

    The per-sample loss is expanded and summed, so the backward gradient
    carries the sample count that ``finish_update`` divides out: the net
    gradient equals the per-sample coefficients.

    ``dit.conditioner.*`` sits on the diagnostic side only: the optimizer
    partition prefix ``condition_tokens`` does not match it, while
    ``condition_tokens`` sits on both sides.  ``rest`` sits on neither.
    """

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.condition_tokens = nn.Linear(1, 1, bias=False)
        self.dit = _DiT()
        self.rest = nn.Parameter(torch.ones(1))
        with torch.no_grad():
            self.condition_tokens.weight.fill_(1.0)

    def loss(self, samples: int) -> torch.Tensor:
        return (
            3.0 * self.condition_tokens.weight.sum()
            + 2.0 * self.dit.conditioner.weight.sum()
            + 4.0 * self.rest.sum()
        ).expand(samples)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


class _SGD:
    """Real CPU optimizer, dev-harness style."""

    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(parameters, lr=0.1)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _clipping_policy(condition_threshold: float, rest_threshold: float) -> ConditionOptimizerConfig:
    return ConditionOptimizerConfig(
        parameter_prefixes=("condition_tokens",),
        independent_clipping=True,
        clip_norm=condition_threshold,
        rest_clip_norm=rest_threshold,
    )


def _diagnostic_union() -> float:
    # Scaled union norm over condition side only: sqrt(3^2 + 2^2).
    return math.sqrt(13.0)


# Minimal static activation facts for the record contract (base keys only,
# no slots): finite-or-null RMS facts plus integer sample counts.
_TELEMETRY_FACTS: dict[str, float | int | None] = {
    "active_samples": 8,
    "null_samples": 0,
    "condition_token_rms_active_pre_norm": 1.0,
    "condition_token_rms_active_post_norm": 1.0,
    "condition_token_rms_null_pre_norm": None,
    "condition_token_rms_null_post_norm": None,
    "condition_global_norm_weight_rms": 1.0,
    "base_hidden_rms": 1.0,
    "condition_residual_rms": None,
}


def _assert_telemetry_record(
    update: SingleGpuUpdateResult,
    *,
    condition_pre: float,
    condition_post: float,
    total_pre: float,
    total_post: float,
) -> None:
    """Check the recorder record against static expected norms.

    The share is the stream's accepted definition: the norm ratio
    condition/total (not an energy-squared ratio), derived here from the
    static norms rather than re-deriving any clip math.
    """
    record = build_condition_diagnostic_record(
        successful_update=1,
        rank=0,
        world_size=1,
        update=update,
        activation_facts=dict(_TELEMETRY_FACTS),
        slot_ids=(),
    )
    assert record["condition_grad_norm_pre_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        condition_pre
    )
    assert record["condition_grad_norm_post_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        condition_post
    )
    assert record["total_grad_norm_pre_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        total_pre
    )
    assert record["total_grad_norm_post_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        total_post
    )
    assert record["condition_grad_share_pre_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        condition_pre / total_pre
    )
    assert record["condition_grad_share_post_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        condition_post / total_post
    )


def _assert_telemetry_record_zero(update: SingleGpuUpdateResult) -> None:
    record = build_condition_diagnostic_record(
        successful_update=1,
        rank=0,
        world_size=1,
        update=update,
        activation_facts=dict(_TELEMETRY_FACTS),
        slot_ids=(),
    )
    assert record["condition_grad_norm_pre_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        0.0
    )
    assert record["total_grad_norm_pre_clip"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        0.0
    )
    # A zero total norm has no defined share: the contract emits null.
    assert record["condition_grad_share_pre_clip"] is None
    assert record["condition_grad_share_post_clip"] is None


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_global_clipping_observed_end_to_end(threshold: float) -> None:
    model = _ObsModel()
    step = SingleGpuStep(
        model,
        _SGD(model.parameters()),
        accumulation_steps=3,
        state=SingleGpuUpdateState.initial(),
        global_clip_norm=threshold,
        condition_grad_share=True,
    )
    for _ in range(3):
        step.backward(model.loss(SAMPLES))
    result = step.finish_update()

    pre = math.sqrt(29.0)
    post = threshold if threshold else pre
    coefficient = post / pre
    torch.testing.assert_close(result.clip.pre_clip_norm, torch.tensor(pre))
    torch.testing.assert_close(result.clip.post_clip_norm, torch.tensor(post))
    torch.testing.assert_close(result.clip.coefficient, torch.tensor(coefficient))
    assert result.condition_clip is None
    assert result.rest_clip is None
    # One common rescale over every gradient, then the real SGD update.
    torch.testing.assert_close(
        model.condition_tokens.weight.squeeze(),
        torch.tensor(1.0 - 0.1 * 3.0 * coefficient),
    )
    torch.testing.assert_close(
        model.dit.conditioner.weight.squeeze(),
        torch.tensor(1.0 - 0.1 * 2.0 * coefficient),
    )
    torch.testing.assert_close(
        model.rest.squeeze(), torch.tensor(1.0 - 0.1 * 4.0 * coefficient)
    )
    # Diagnostics observe the same rescale on the condition side.
    assert result.condition_grad_norm_pre_clip is not None
    torch.testing.assert_close(
        result.condition_grad_norm_pre_clip, torch.tensor(_diagnostic_union())
    )
    assert result.condition_grad_norm_post_clip is not None
    torch.testing.assert_close(
        result.condition_grad_norm_post_clip,
        torch.tensor(_diagnostic_union() * coefficient),
    )
    # Recorder telemetry: the accepted norm-ratio share of the condition side.
    _assert_telemetry_record(
        result,
        condition_pre=_diagnostic_union(),
        condition_post=_diagnostic_union() * coefficient,
        total_pre=pre,
        total_post=post,
    )


@pytest.mark.parametrize(
    ("condition_threshold", "rest_threshold"),
    [(0.5, 2.0), (0.0, 0.0), (0.0, 2.0), (0.5, 0.0)],
)
def test_independent_clipping_observed_end_to_end(
    condition_threshold: float, rest_threshold: float,
) -> None:
    model = _ObsModel()
    step = SingleGpuStep(
        model,
        _SGD(model.parameters()),
        accumulation_steps=3,
        state=SingleGpuUpdateState.initial(),
        condition_optimizer=_clipping_policy(condition_threshold, rest_threshold),
        condition_grad_share=True,
    )
    for _ in range(3):
        step.backward(model.loss(SAMPLES))
    result = step.finish_update()

    # Scaled norms: condition group 3.0; rest group (conditioner + rest)
    # sqrt(20).  A zero threshold disables clipping for that group.
    condition_pre = 3.0
    rest_pre = math.sqrt(20.0)
    condition_post = (
        min(condition_threshold, condition_pre) if condition_threshold else condition_pre
    )
    rest_post = (
        min(rest_threshold, rest_pre) if rest_threshold else rest_pre
    )
    condition_coefficient = condition_post / condition_pre
    rest_coefficient = rest_post / rest_pre
    assert result.condition_clip is not None
    torch.testing.assert_close(
        result.condition_clip.pre_clip_norm, torch.tensor(condition_pre)
    )
    torch.testing.assert_close(
        result.condition_clip.post_clip_norm, torch.tensor(condition_post)
    )
    torch.testing.assert_close(
        result.condition_clip.coefficient, torch.tensor(condition_coefficient)
    )
    assert result.rest_clip is not None
    torch.testing.assert_close(result.rest_clip.pre_clip_norm, torch.tensor(rest_pre))
    torch.testing.assert_close(
        result.rest_clip.post_clip_norm, torch.tensor(rest_post)
    )
    torch.testing.assert_close(
        result.rest_clip.coefficient, torch.tensor(rest_coefficient)
    )
    # Summary ratio: hypot of the two group results.
    torch.testing.assert_close(result.clip.pre_clip_norm, torch.tensor(math.sqrt(29.0)))
    torch.testing.assert_close(
        result.clip.post_clip_norm,
        torch.tensor(math.hypot(condition_post, rest_post)),
    )
    # Real per-group updates through the actual SGD step.
    torch.testing.assert_close(
        model.condition_tokens.weight.squeeze(),
        torch.tensor(1.0 - 0.1 * 3.0 * condition_coefficient),
    )
    torch.testing.assert_close(
        model.dit.conditioner.weight.squeeze(),
        torch.tensor(1.0 - 0.1 * 2.0 * rest_coefficient),
    )
    torch.testing.assert_close(
        model.rest.squeeze(), torch.tensor(1.0 - 0.1 * 4.0 * rest_coefficient)
    )
    # The diagnostic side spans both groups.
    assert result.condition_grad_norm_pre_clip is not None
    torch.testing.assert_close(
        result.condition_grad_norm_pre_clip, torch.tensor(_diagnostic_union())
    )
    condition_post_side = math.hypot(
        3.0 * condition_coefficient, 2.0 * rest_coefficient
    )
    assert result.condition_grad_norm_post_clip is not None
    torch.testing.assert_close(
        result.condition_grad_norm_post_clip,
        torch.tensor(condition_post_side),
    )
    # Recorder telemetry: the accepted norm-ratio share of the condition side.
    _assert_telemetry_record(
        result,
        condition_pre=_diagnostic_union(),
        condition_post=condition_post_side,
        total_pre=math.sqrt(29.0),
        total_post=math.hypot(condition_post, rest_post),
    )


def test_zero_gradients_with_independent_clipping_do_not_divide() -> None:
    model = _ObsModel()
    step = SingleGpuStep(
        model,
        _SGD(model.parameters()),
        accumulation_steps=3,
        state=SingleGpuUpdateState.initial(),
        condition_optimizer=_clipping_policy(0.5, 2.0),
        condition_grad_share=True,
    )
    for _ in range(3):
        zero_loss = (
            0.0
            * (
                model.condition_tokens.weight.sum()
                + model.dit.conditioner.weight.sum()
                + model.rest.sum()
            )
        ).expand(SAMPLES)
        step.backward(zero_loss)
    result = step.finish_update()
    assert result.condition_clip is not None and result.rest_clip is not None
    for clip in (result.clip, result.condition_clip, result.rest_clip):
        torch.testing.assert_close(clip.pre_clip_norm, torch.tensor(0.0))
        torch.testing.assert_close(clip.post_clip_norm, torch.tensor(0.0))
        torch.testing.assert_close(clip.coefficient, torch.tensor(1.0))
    _assert_telemetry_record_zero(result)
    assert result.condition_grad_norm_pre_clip is not None
    torch.testing.assert_close(
        result.condition_grad_norm_pre_clip, torch.tensor(0.0)
    )
    torch.testing.assert_close(
        result.condition_grad_norm_post_clip, torch.tensor(0.0)
    )


def test_independent_lr_only_keeps_global_clip_and_trajectory() -> None:
    legacy, lr_only = _ObsModel(), _ObsModel()
    policies: tuple[ConditionOptimizerConfig | None, ConditionOptimizerConfig | None] = (
        None,
        ConditionOptimizerConfig(
            parameter_prefixes=("condition_tokens",),
            independent_lr=True,
            lr_multiplier=1.5,
        ),
    )
    steps = [
        SingleGpuStep(
            model,
            _SGD(model.parameters()),
            accumulation_steps=3,
            state=SingleGpuUpdateState.initial(),
            global_clip_norm=0.5,
            condition_optimizer=policy,
            condition_grad_share=True,
        )
        for model, policy in zip((legacy, lr_only), policies, strict=True)
    ]
    results: list[tuple[SingleGpuUpdateResult, SingleGpuUpdateResult]] = []
    for _ in range(3):
        for model, step in zip((legacy, lr_only), steps, strict=True):
            for _microbatch in range(3):
                step.backward(model.loss(SAMPLES))
        results.append((steps[0].finish_update(), steps[1].finish_update()))
    for first, second in zip(legacy.parameters(), lr_only.parameters(), strict=True):
        assert torch.equal(first, second)
    legacy_result, lr_result = results[-1]
    assert lr_result.condition_clip is None
    assert lr_result.rest_clip is None
    torch.testing.assert_close(
        lr_result.clip.pre_clip_norm, legacy_result.clip.pre_clip_norm
    )
    torch.testing.assert_close(
        lr_result.clip.post_clip_norm, legacy_result.clip.post_clip_norm
    )
    torch.testing.assert_close(
        lr_result.clip.coefficient, legacy_result.clip.coefficient
    )
    # Recorder telemetry on the LR-only path: norms measured, no clip split.
    # Gradients stay (3, 2, 4) per update, so the expected norms are static.
    _assert_telemetry_record(
        lr_result,
        condition_pre=_diagnostic_union(),
        condition_post=_diagnostic_union() * (0.5 / math.sqrt(29.0)),
        total_pre=math.sqrt(29.0),
        total_post=0.5,
    )


def test_production_condition_lr_split_drives_real_cpu_adamw_updates() -> None:
    """Real production group construction and a real FP32 CPU AdamW step.

    The groups are split by the production ``split_condition_lr_groups``
    over an ``IsolatedAdamW8bit`` wrapper (never hand-built), then the
    real inner AdamW performs the update.
    """
    condition = nn.Parameter(torch.full((4,), 1.0))
    rest = nn.Parameter(torch.full((4,), 1.0))
    audit = ParameterAudit(
        (
            ParameterSpec(
                "condition_tokens.weight", condition, "matrix_decay", 0.0
            ),
            ParameterSpec("rest.layer.weight", rest, "matrix_decay", 0.0),
        )
    )
    groups = [
        {
            "params": [condition, rest],
            "param_names": ["condition_tokens.weight", "rest.layer.weight"],
            "group_name": "matrix_decay",
            "weight_decay": 0.0,
        }
    ]
    inner = torch.optim.AdamW(groups, lr=0.01, betas=(0.9, 0.95), eps=1e-8)
    # Production CPU harness convention: the SR RNG carries a dummy state;
    # CPU verification drives the real inner AdamW step directly.
    rng = StochasticRoundingRNG(
        torch.device("cuda:0"), torch.arange(16, dtype=torch.uint8)
    )
    optimizer = IsolatedAdamW8bit(inner, audit, rng)
    policy = ConditionOptimizerConfig(
        parameter_prefixes=("condition_tokens",),
        independent_lr=True,
        lr_multiplier=1.5,
    )
    split_condition_lr_groups(optimizer, policy)
    split_groups = optimizer.optimizer.param_groups
    assert [group.get("condition_lr_group") for group in split_groups] == [
        True,
        False,
    ]
    # The production scheduling formula: condition groups get the scheduled
    # base times the condition multiplier.
    for group in split_groups:
        group["lr"] = (
            0.01
            * (
                policy.lr_multiplier
                if group.get("condition_lr_group") is True
                else 1.0
            )
        )
    condition.grad = torch.full((4,), 0.5)
    rest.grad = torch.full((4,), 0.5)
    optimizer.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    # AdamW first step with uniform positive grads: the update is lr
    # (m_hat / sqrt(v_hat) ~= 1 before eps), so each parameter moves by its
    # own group learning rate.
    torch.testing.assert_close(
        condition, torch.full((4,), 1.0 - 0.015), rtol=1e-5, atol=1e-7
    )
    torch.testing.assert_close(
        rest, torch.full((4,), 1.0 - 0.01), rtol=1e-5, atol=1e-7
    )


def test_diagnostic_prefixes_are_wider_than_optimizer_partition() -> None:
    model = _ObsModel()
    step = SingleGpuStep(
        model,
        _SGD(model.parameters()),
        accumulation_steps=3,
        state=SingleGpuUpdateState.initial(),
        condition_optimizer=_clipping_policy(0.5, 2.0),
        condition_grad_share=True,
    )
    for _ in range(3):
        step.backward(model.loss(SAMPLES))
    result = step.finish_update()
    # Diagnostic union includes dit.conditioner.* (scaled grad 2):
    # sqrt(3^2 + 2^2), not the optimizer group's 3.0 and not the full
    # sqrt(29) union with rest.
    assert result.condition_grad_norm_pre_clip is not None
    torch.testing.assert_close(
        result.condition_grad_norm_pre_clip, torch.tensor(_diagnostic_union())
    )
    assert result.condition_clip is not None
    torch.testing.assert_close(
        result.condition_clip.pre_clip_norm, torch.tensor(3.0)
    )


def test_diagnostics_do_not_change_updates_or_optimizer_state() -> None:
    with_diag, without_diag = _ObsModel(), _ObsModel()
    optimizers = [
        _SGD(model.parameters()) for model in (with_diag, without_diag)
    ]
    steps = [
        SingleGpuStep(
            model,
            optimizer,
            accumulation_steps=3,
            state=SingleGpuUpdateState.initial(),
            global_clip_norm=0.5,
            condition_optimizer=_clipping_policy(0.5, 2.0),
            condition_grad_share=enabled,
        )
        for model, optimizer, enabled in zip(
            (with_diag, without_diag), optimizers, (True, False), strict=True
        )
    ]
    results: list[tuple[SingleGpuUpdateResult, SingleGpuUpdateResult]] = []
    for _ in range(3):
        for model, step in zip((with_diag, without_diag), steps, strict=True):
            for _microbatch in range(3):
                step.backward(model.loss(SAMPLES))
        results.append((steps[0].finish_update(), steps[1].finish_update()))
    with_diag_result, without_diag_result = results[-1]
    assert with_diag_result.condition_grad_norm_pre_clip is not None
    assert without_diag_result.condition_grad_norm_pre_clip is None
    assert with_diag_result.condition_grad_norm_post_clip is not None
    assert without_diag_result.condition_grad_norm_post_clip is None
    parameter_pairs = list(
        zip(with_diag.parameters(), without_diag.parameters(), strict=True)
    )
    assert parameter_pairs
    for first, second in parameter_pairs:
        assert torch.equal(first, second)
    first, second = parameter_pairs[0]
    first_state = optimizers[0].optimizer.state[first]
    second_state = optimizers[1].optimizer.state[second]
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name])
    for attribute in ("pre_clip_norm", "post_clip_norm", "coefficient"):
        assert torch.equal(
            getattr(with_diag_result.clip, attribute),
            getattr(without_diag_result.clip, attribute),
        )
    assert with_diag_result.state == without_diag_result.state


def test_loop_cadence_callback_toggles_diagnostics_per_update(tmp_path: Path) -> None:
    model = _ObsModel()
    observations: list[SuccessfulLoopObservation] = []
    cadence = {1: True, 2: False, 3: True}
    loop = SingleGpuTrainingLoop[None](
        module=model,
        optimizer=_SGD(model.parameters()),
        loss_fn=lambda _batch, _microbatch: model.loss(SAMPLES),
        accumulation_steps=1,
        target_successful_updates=3,
        checkpoint_every_successful_updates=99,
        scheduler_step=lambda _update: None,
        checkpoint=lambda _update: None,
        diagnostic_root=tmp_path,
        failure_id=lambda phase, _state: f"failure-{phase}",
        state=SingleGpuUpdateState.initial(),
        condition_grad_share=lambda update: cadence[update],
        successful_update_observer=observations.append,
    )
    loop.run([None, None, None])
    assert [
        observation.update.condition_grad_norm_pre_clip is not None
        for observation in observations
    ] == [True, False, True]
    assert [
        observation.update.condition_grad_norm_post_clip is not None
        for observation in observations
    ] == [True, False, True]
    first = observations[0].update.condition_grad_norm_pre_clip
    torch.testing.assert_close(first, torch.tensor(_diagnostic_union()))
    third = observations[2].update.condition_grad_norm_pre_clip
    torch.testing.assert_close(
        third, torch.tensor(_diagnostic_union()), rtol=1e-5, atol=1e-6
    )


def test_loop_microbatch_loss_failure_propagates(tmp_path: Path) -> None:
    model = _ObsModel()

    def loss_fn(_batch: None, microbatch: int) -> torch.Tensor:
        if microbatch == 1:
            raise RuntimeError("injected microbatch failure")
        return model.loss(1)

    loop = SingleGpuTrainingLoop[None](
        module=model,
        optimizer=_SGD(model.parameters()),
        loss_fn=loss_fn,
        accumulation_steps=2,
        target_successful_updates=1,
        checkpoint_every_successful_updates=99,
        scheduler_step=lambda _update: None,
        checkpoint=lambda _update: None,
        diagnostic_root=tmp_path,
        failure_id=lambda phase, _state: f"failure-{phase}",
        state=SingleGpuUpdateState.initial(),
        condition_grad_share=True,
    )
    with pytest.raises(RuntimeError, match="injected microbatch failure"):
        loop.run([None, None])
