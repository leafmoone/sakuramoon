from __future__ import annotations

import pytest
import torch
from torch import nn

import sakuramoon.train.step as step_module
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.optim.clip import clip_grad_norm_fp32
from sakuramoon.optim.groups import audit_trainable_parameters
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    TrainableComposite,
)


class _SgdAdapter:
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(parameters, lr=0.1)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _per_sample(model: nn.Linear, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (model(inputs) - targets).float().square().flatten(1).mean(1)


def _production_composite() -> TrainableComposite:
    return TrainableComposite(
        dit=PackedDiT(
            depth=16,
            input_channels=128,
            hidden_size=2560,
            intermediate_size=6912,
            q_heads=20,
            kv_heads=5,
            head_dim=128,
            rope_nope_dim=32,
            rope_y_dim=48,
            rope_x_dim=48,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=5120,
            out_channels=128,
            condition_token_count=8,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        ),
        text=TextConditioner(
            input_size=2048,
            adapter_size=1024,
            output_size=2560,
            groups=8,
            attention_heads=16,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
        condition_tokens=ConditionTokenEncoder(
            input_size=2048,
            hidden_size=1024,
            intermediate_size=2048,
            output_size=2560,
            token_count=8,
            attention_heads=16,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
    )


def test_production_composite_locks_trainable_fqn_boundary() -> None:
    with torch.device("meta"):
        composite = _production_composite()
    audit = audit_trainable_parameters(
        composite,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )

    assert {name for name, _module in composite.named_children()} == {
        "dit",
        "text",
        "condition_tokens",
    }
    assert all(
        "qwen" not in type(module).__name__.lower()
        and "vae" not in type(module).__name__.lower()
        for module in composite.modules()
    )
    assert len(audit.specs) == 241
    assert len(audit.decay) == 126
    assert len(audit.sensitive) == 115


def test_unequal_microbatches_match_merged_sample_mean_update() -> None:
    torch.manual_seed(1501)  # pyright: ignore[reportUnknownMemberType]
    accumulated = nn.Linear(2, 1, bias=False)
    merged = nn.Linear(2, 1, bias=False)
    merged.load_state_dict(accumulated.state_dict())
    inputs = torch.tensor([[0.1, 0.2], [0.3, -0.2], [0.0, 0.4], [-0.1, 0.2]])
    targets = torch.tensor([[0.0], [0.1], [-0.2], [0.2]])

    step = SingleGpuStep(
        accumulated,
        _SgdAdapter(list(accumulated.parameters())),
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward(_per_sample(accumulated, inputs[:1], targets[:1]))
    step.backward(_per_sample(accumulated, inputs[1:], targets[1:]))
    result = step.finish_update()

    merged_loss = _per_sample(merged, inputs, targets)
    merged_loss.mean().backward()  # pyright: ignore[reportUnknownMemberType]
    merged_optimizer = _SgdAdapter(list(merged.parameters()))
    merged_clip = clip_grad_norm_fp32(merged.parameters(), max_norm=1.0)
    merged_optimizer.step()
    merged_optimizer.zero_grad(set_to_none=True)

    torch.testing.assert_close(accumulated.weight, merged.weight, atol=0, rtol=0)
    torch.testing.assert_close(
        result.mean_loss,
        merged_loss.mean(),
        atol=1e-8,
        rtol=1e-7,
    )
    torch.testing.assert_close(result.clip.pre_clip_norm, merged_clip.pre_clip_norm)
    assert result.microbatches == 2
    assert result.effective_samples == 4
    assert result.state == SingleGpuUpdateState(1, 1, 4)
    assert step.pending_samples == 0 and step.pending_microbatches == 0
    assert all(parameter.grad is None for parameter in accumulated.parameters())


def test_update_uses_per_sample_not_element_weighting() -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    module = nn.ParameterList([parameter])
    step = SingleGpuStep(
        module,
        _SgdAdapter([parameter]),
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
    )
    first = (parameter - torch.tensor([1.0])).square().mean().reshape(1)
    second = torch.stack(
        (
            (parameter - torch.tensor([3.0, 3.0, 3.0])).square().mean(),
            (parameter - torch.tensor([5.0, 5.0])).square().mean(),
        )
    )

    step.backward(first)
    step.backward(second)
    result = step.finish_update()

    torch.testing.assert_close(result.mean_loss, torch.tensor((1.0 + 9.0 + 25.0) / 3.0))
    assert result.effective_samples == 3


@pytest.mark.parametrize(
    ("loss", "error"),
    [
        (torch.empty(0, requires_grad=True), ValueError),
        (torch.ones(1, 1, requires_grad=True), ValueError),
        (torch.ones(1, dtype=torch.bfloat16, requires_grad=True), TypeError),
        (torch.ones(1), ValueError),
    ],
)
def test_invalid_per_sample_loss_fails_before_state_changes(
    loss: torch.Tensor, error: type[Exception]
) -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        _SgdAdapter([parameter]),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )

    with pytest.raises(error):
        step.backward(loss)

    assert step.pending_samples == 0
    assert step.state == SingleGpuUpdateState.initial()
    assert parameter.grad is None


def test_microbatch_count_is_exact_and_repeated_finish_fails() -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        _SgdAdapter([parameter]),
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward((parameter - 1.0).square().reshape(1))
    with pytest.raises(RuntimeError, match="exactly"):
        step.finish_update()
    step.backward((parameter - 2.0).square().reshape(1))
    with pytest.raises(RuntimeError, match="more microbatches"):
        step.backward((parameter - 3.0).square().reshape(1))
    step.finish_update()
    with pytest.raises(RuntimeError, match="exactly"):
        step.finish_update()


def test_failed_optimizer_attempt_is_counted_and_cannot_continue() -> None:
    class _FailingOptimizer(_SgdAdapter):
        def step(self) -> None:
            raise RuntimeError("step failed")

    parameter = nn.Parameter(torch.tensor(0.0))
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        _FailingOptimizer([parameter]),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward((parameter - 1.0).square().reshape(1))

    with pytest.raises(RuntimeError, match="step failed"):
        step.finish_update()

    assert step.state == SingleGpuUpdateState(1, 0, 0)
    assert step.detection_phase == "optimizer"
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


def test_completion_failure_prevents_successful_update_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        _SgdAdapter([parameter]),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward((parameter - 1.0).square().reshape(1))

    def fail_completion(_device: torch.device) -> None:
        raise RuntimeError("device completion failed")

    monkeypatch.setattr(step_module, "_complete_device_work", fail_completion)
    with pytest.raises(RuntimeError, match="completion failed"):
        step.finish_update()

    assert step.state == SingleGpuUpdateState(1, 0, 0)
    assert step.detection_phase == "device_completion"
    assert parameter.grad is None
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


def test_post_step_zero_grad_failure_preserves_successful_mutation_state() -> None:
    class _FailingCleanupOptimizer(_SgdAdapter):
        def zero_grad(self, *, set_to_none: bool) -> None:
            del set_to_none
            raise RuntimeError("cleanup failed")

    parameter = nn.Parameter(torch.tensor(0.0))
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        _FailingCleanupOptimizer([parameter]),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward((parameter - 1.0).square().reshape(1))

    with pytest.raises(RuntimeError, match="cleanup failed"):
        step.finish_update()

    assert parameter.item() != 0.0
    assert step.state == SingleGpuUpdateState(1, 1, 1)
    assert step.detection_phase == "zero_grad"
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


@pytest.mark.parametrize("failure", ["nonfinite", "backward"])
def test_loss_failure_and_zero_grad_failure_are_both_preserved(failure: str) -> None:
    class _FailingCleanupOptimizer(_SgdAdapter):
        def zero_grad(self, *, set_to_none: bool) -> None:
            del set_to_none
            raise OSError("cleanup failed")

    parameter = nn.Parameter(torch.tensor(1.0))
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        _FailingCleanupOptimizer([parameter]),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    if failure == "nonfinite":
        loss = (parameter * torch.tensor(float("nan"))).reshape(1)
    else:
        loss = parameter.clone().reshape(1)

        def fail_backward(_gradient: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("backward failed")

        _hook = loss.register_hook(  # pyright: ignore[reportUnknownMemberType]
            fail_backward
        )

    with pytest.raises(BaseExceptionGroup) as captured:
        step.backward(loss)

    expected = FloatingPointError if failure == "nonfinite" else RuntimeError
    assert [type(error) for error in captured.value.exceptions] == [expected, OSError]
    assert step.state == SingleGpuUpdateState(1, 0, 0)
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


def test_nonfinite_loss_aborts_attempt_and_clears_gradients() -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = _SgdAdapter([parameter])
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        optimizer,
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward((parameter - 1.0).square().reshape(1))

    with pytest.raises(FloatingPointError, match="nonfinite"):
        step.backward((parameter * torch.tensor(float("nan"))).reshape(1))

    assert step.state == SingleGpuUpdateState(1, 0, 0)
    assert parameter.grad is None
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


def test_external_abort_is_idempotent_and_clears_pending_gradients() -> None:
    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = _SgdAdapter([parameter])
    step = SingleGpuStep(
        nn.ParameterList([parameter]),
        optimizer,
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward((parameter - 1.0).square().reshape(1))

    step.abort()
    step.abort()

    assert step.state == SingleGpuUpdateState(1, 0, 0)
    assert parameter.grad is None
    with pytest.raises(RuntimeError, match="cannot continue"):
        step.finish_update()


def test_update_state_rejects_inconsistent_counters() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        SingleGpuUpdateState(attempted_updates=0, successful_updates=1, effective_samples=0)
