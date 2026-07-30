from __future__ import annotations

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
