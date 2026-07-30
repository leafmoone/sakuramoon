from __future__ import annotations

import io
import json
from typing import cast

import pytest
import torch
from torch import nn

from sakuramoon.optim.adamw8bit import build_adamw8bit

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class _OptimizerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Linear(
            64, 64, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.sensitive = nn.Parameter(torch.ones(64, device="cuda"))
        self.lazy = nn.Linear(
            64, 64, bias=False, device="cuda", dtype=torch.bfloat16
        )


class _CanaryModule(nn.Module):
    def __init__(self, matrix_dtype: torch.dtype) -> None:
        super().__init__()
        self.matrix = nn.Linear(
            64,
            64,
            bias=False,
            device="cuda",
            dtype=matrix_dtype,
        )
        self.sensitive = nn.Parameter(torch.ones(64, device="cuda"))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.matrix(inputs.to(self.matrix.weight.dtype)).float()
        return projected * self.sensitive


def _optimizer(module: nn.Module, seed: int = 901):
    return build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=seed,
    )


def _assert_parameter_state_equal(
    first: dict[str, object], second: dict[str, object]
) -> None:
    for key in ("step", "exp_avg", "exp_avg_sq"):
        first_value = cast(torch.Tensor, first[key])
        second_value = cast(torch.Tensor, second[key])
        if type(first_value).__name__ == "OptimState8bit":
            assert type(second_value).__name__ == "OptimState8bit"
            for field in ("codes", "qmap", "scale"):
                torch.testing.assert_close(
                    getattr(first_value, field),
                    getattr(second_value, field),
                    atol=0,
                    rtol=0,
                )
        else:
            torch.testing.assert_close(first_value, second_value, atol=0, rtol=0)


def test_torchao_state_is_lazy_quantized_and_training_rng_isolated() -> None:
    torch.manual_seed(902)  # pyright: ignore[reportUnknownMemberType]
    module = _OptimizerModule()
    optimizer = _optimizer(module)
    training_state = torch.cuda.get_rng_state()
    expected_random = torch.rand(8, device="cuda")
    torch.cuda.set_rng_state(training_state)
    sr_before = optimizer.sr_rng.state.clone()
    matrix_before = module.matrix.weight.detach().clone()
    module.matrix.weight.grad = torch.ones_like(module.matrix.weight)
    module.sensitive.grad = torch.ones_like(module.sensitive)

    optimizer.step()
    actual_random = torch.rand(8, device="cuda")

    torch.testing.assert_close(actual_random, expected_random, atol=0, rtol=0)
    assert not torch.equal(sr_before, optimizer.sr_rng.state)
    assert not torch.equal(matrix_before, module.matrix.weight)
    assert module.lazy.weight not in optimizer.optimizer.state
    state_audit = {spec.name: spec for spec in optimizer.audit_state()}
    assert state_audit["matrix.weight"].state_class == "OptimState8bit"
    assert state_audit["matrix.weight"].block_size == 256
    assert state_audit["matrix.weight"].state_bytes == 10_372
    assert state_audit["matrix.weight"].step == 1
    assert state_audit["sensitive"].state_class == "Tensor"
    assert state_audit["sensitive"].state_bytes == 516
    assert not state_audit["lazy.weight"].initialized
    assert optimizer.optimizer.param_groups[0]["param_names"] == [
        "lazy.weight",
        "matrix.weight",
    ]
    assert optimizer.optimizer.param_groups[1]["param_names"] == ["sensitive"]
    optimizer.zero_grad(set_to_none=True)
    assert module.matrix.weight.grad is None and module.sensitive.grad is None


def test_nonfinite_gradient_does_not_advance_any_state() -> None:
    module = _OptimizerModule()
    optimizer = _optimizer(module)
    module.matrix.weight.grad = torch.full_like(
        module.matrix.weight, float("nan")
    )
    parameter_before = module.matrix.weight.detach().clone()
    sr_before = optimizer.sr_rng.state.clone()

    with pytest.raises(FloatingPointError, match="nonfinite"):
        optimizer.step()

    torch.testing.assert_close(module.matrix.weight, parameter_before, equal_nan=True)
    assert torch.equal(optimizer.sr_rng.state, sr_before)
    assert module.matrix.weight not in optimizer.optimizer.state


def test_optimizer_state_round_trip_matches_next_step_bitwise() -> None:
    first_module = _OptimizerModule()
    first = _optimizer(first_module, seed=903)
    first_module.matrix.weight.grad = torch.ones_like(first_module.matrix.weight)
    first_module.sensitive.grad = torch.ones_like(first_module.sensitive)
    first.step()
    checkpoint = io.BytesIO()
    torch.save(
        {"model": first_module.state_dict(), "optimizer": first.state_dict()},
        checkpoint,
    )
    checkpoint.seek(0)
    restored = cast(dict[str, object], torch.load(checkpoint, weights_only=False))

    second_module = _OptimizerModule()
    second = _optimizer(second_module, seed=904)
    second_module.load_state_dict(
        cast(dict[str, torch.Tensor], restored["model"])
    )
    second.load_state_dict(cast(dict[str, object], restored["optimizer"]))

    assert torch.equal(second.sr_rng.state, first.sr_rng.state)
    assert second.audit_state() == first.audit_state()
    for module in (first_module, second_module):
        module.matrix.weight.grad = torch.full_like(module.matrix.weight, 0.25)
        module.sensitive.grad = torch.full_like(module.sensitive, -0.5)
    first.step()
    second.step()

    for first_parameter, second_parameter in zip(
        first_module.parameters(), second_module.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, atol=0, rtol=0)
        first_state = first.optimizer.state.get(first_parameter)
        second_state = second.optimizer.state.get(second_parameter)
        if not first_state:
            assert not second_state
        else:
            assert second_state is not None
            _assert_parameter_state_equal(first_state, second_state)
    assert torch.equal(second.sr_rng.state, first.sr_rng.state)
    assert second.audit_state() == first.audit_state()


def test_sr_rng_load_rejects_wrong_dtype_and_shape() -> None:
    optimizer = _optimizer(_OptimizerModule(), seed=905)
    state = optimizer.sr_rng.state_dict()

    wrong_dtype = dict(state)
    wrong_dtype["state"] = optimizer.sr_rng.state.float()
    with pytest.raises(TypeError, match="CPU uint8"):
        optimizer.sr_rng.load_state_dict(wrong_dtype)

    wrong_shape = dict(state)
    wrong_shape["state"] = optimizer.sr_rng.state[:-1]
    with pytest.raises(TypeError, match="expected shape"):
        optimizer.sr_rng.load_state_dict(wrong_shape)


def test_small_bf16_matrix_cannot_fall_back_to_unquantized_state() -> None:
    module = nn.Linear(
        32,
        32,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="non-quantized"):
        _optimizer(module, seed=906)


def test_one_thousand_step_mixed_sr_canary_tracks_validation_ema() -> None:
    torch.manual_seed(907)  # pyright: ignore[reportUnknownMemberType]
    mixed = _CanaryModule(torch.bfloat16)
    reference = _CanaryModule(torch.float32)
    with torch.no_grad():
        reference.matrix.weight.copy_(mixed.matrix.weight.float())
        reference.sensitive.copy_(mixed.sensitive)
    mixed_optimizer = _optimizer(mixed, seed=908)
    reference_optimizer = torch.optim.AdamW(
        [
            {"params": [reference.matrix.weight], "weight_decay": 0.01},
            {"params": [reference.sensitive], "weight_decay": 0.0},
        ],
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    train_inputs = torch.randn(16, 8, 64, device="cuda", dtype=torch.bfloat16)
    train_targets = torch.randn(16, 8, 64, device="cuda", dtype=torch.float32)
    validation_inputs = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    validation_target = torch.randn(16, 64, device="cuda", dtype=torch.float32)

    mixed_ema = 0.0
    reference_ema = 0.0
    ema_decay = 0.99
    ratios: list[float] = []
    for step in range(1000):
        inputs = train_inputs[step % train_inputs.shape[0]]
        target = train_targets[step % train_targets.shape[0]]
        mixed_loss = (mixed(inputs) - target).square().mean()
        reference_loss = (reference(inputs) - target).square().mean()
        assert torch.isfinite(mixed_loss) and torch.isfinite(reference_loss)
        mixed_loss.backward()  # pyright: ignore[reportUnknownMemberType]
        reference_loss.backward()  # pyright: ignore[reportUnknownMemberType]
        mixed_optimizer.step()
        reference_optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        mixed_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)  # pyright: ignore[reportUnknownMemberType]

        with torch.no_grad():
            mixed_validation = (
                mixed(validation_inputs) - validation_target
            ).square().mean()
            reference_validation = (
                reference(validation_inputs) - validation_target
            ).square().mean()
        assert torch.isfinite(mixed_validation) and torch.isfinite(reference_validation)
        mixed_value = mixed_validation.item()
        reference_value = reference_validation.item()
        if step == 0:
            mixed_ema = mixed_value
            reference_ema = reference_value
        else:
            mixed_ema = ema_decay * mixed_ema + (1.0 - ema_decay) * mixed_value
            reference_ema = (
                ema_decay * reference_ema
                + (1.0 - ema_decay) * reference_value
            )
        ratios.append(mixed_ema / reference_ema)

    assert mixed_ema <= 1.03 * reference_ema
    assert max(ratios[500:]) <= 1.03
    state_audit = {spec.name: spec for spec in mixed_optimizer.audit_state()}
    assert state_audit["matrix.weight"].step == 1000
    assert state_audit["sensitive"].step == 1000
    mixed_state = mixed_optimizer.optimizer.state[mixed.matrix.weight]
    reference_state = reference_optimizer.state[reference.matrix.weight]
    assert mixed_state["step"].item() == 1000.0
    assert reference_state["step"].item() == 1000.0
    print(
        json.dumps(
            {
                "final_validation_ema_ratio": mixed_ema / reference_ema,
                "max_validation_ema_ratio_steps_501_1000": max(ratios[500:]),
                "mixed_validation_ema": mixed_ema,
                "reference_validation_ema": reference_ema,
            },
            sort_keys=True,
        )
    )
