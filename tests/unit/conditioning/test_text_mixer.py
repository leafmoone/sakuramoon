from __future__ import annotations

import inspect

import pytest
import torch

from sakuramoon.conditioning.text_mixer import TextConditioner


def _conditioner() -> TextConditioner:
    return TextConditioner(
        input_size=16,
        adapter_size=16,
        output_size=24,
        groups=4,
        attention_heads=4,
        norm_eps=1e-6,
        mix_gate_init=0.0,
        layer_scale_init=1.0,
        projection_bias=False,
        linear_dtype=torch.float32,
        sensitive_dtype=torch.float32,
    )


def test_gathers_only_main_tokens_and_zeroes_padding() -> None:
    module = _conditioner()
    states = torch.randn(2, 6, 7, 16)
    indices = torch.tensor([[0, 2, 4], [1, -1, -1]])
    mask = torch.tensor([[True, True, True], [True, False, False]])

    output = module(states, indices, mask)
    changed = states.clone()
    changed[:, 5] += 1000.0
    output_changed = module(changed, indices, mask)

    torch.testing.assert_close(output.tokens, output_changed.tokens)
    assert output.tokens.shape == (2, 3, 24)
    assert torch.count_nonzero(output.tokens[1, 1:]) == 0


def test_inactive_indices_are_sanitized_before_gather() -> None:
    module = _conditioner()
    states = torch.randn(1, 4, 7, 16)
    mask = torch.tensor([[True, True, False, False]])
    baseline = module(
        states,
        torch.tensor([[0, 3, -1, -1]]),
        mask,
    )
    inactive_out_of_range = module(
        states,
        torch.tensor([[0, 3, torch.iinfo(torch.long).max, -10_000]]),
        mask,
    )

    torch.testing.assert_close(baseline.tokens, inactive_out_of_range.tokens)
    torch.testing.assert_close(
        baseline.layer_weights,
        inactive_out_of_range.layer_weights,
    )


@pytest.mark.parametrize("invalid_index", [-1, 4])
def test_active_out_of_range_indices_still_fail(invalid_index: int) -> None:
    module = _conditioner()

    with pytest.raises(ValueError, match="active main token index"):
        module(
            torch.randn(1, 4, 7, 16),
            torch.tensor([[0, invalid_index]]),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_production_constructor_locks_decided_architecture_and_precision() -> None:
    signature = inspect.signature(TextConditioner.for_production)
    assert tuple(signature.parameters) == (
        "attention_heads",
        "mix_gate_init",
        "layer_scale_init",
        "projection_bias",
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )

    with torch.device("meta"):
        module = TextConditioner.for_production(
            attention_heads=8,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
        )

    assert module.artifact_config() == {
        "adapter_size": 1024,
        "attention_heads": 8,
        "groups": 8,
        "input_size": 2048,
        "layer_scale_init": 1.0,
        "linear_dtype": "bfloat16",
        "mix_gate_init": 0.0,
        "norm_eps": 1e-6,
        "output_size": 2560,
        "projection_bias": False,
        "sensitive_dtype": "float32",
    }


def test_zero_initialized_gate_starts_uniform_and_gradients_stay_in_adapter() -> None:
    module = _conditioner()
    states = torch.randn(1, 3, 7, 16, requires_grad=True)
    output = module(
        states,
        torch.tensor([[0, 1, 2]]),
        torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(
        output.layer_weights,
        torch.full_like(output.layer_weights, 1.0 / 7.0),
    )
    output.tokens.square().mean().backward()
    assert states.grad is None
    assert module.output_projection.weight.grad is not None


def test_attention_is_bidirectional_and_padding_is_not_visible() -> None:
    module = _conditioner()
    states = torch.randn(1, 3, 7, 16)
    indices = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[True, True, False]])
    baseline = module(states, indices, mask).tokens

    future_changed = states.clone()
    future_changed[:, 1] += 4.0
    future_output = module(future_changed, indices, mask).tokens
    padded_changed = states.clone()
    padded_changed[:, 2] += 4000.0
    padded_output = module(padded_changed, indices, mask).tokens

    assert not torch.allclose(baseline[:, 0], future_output[:, 0])
    torch.testing.assert_close(baseline, padded_output)
