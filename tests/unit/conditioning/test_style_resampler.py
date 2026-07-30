from __future__ import annotations

import torch

from sakuramoon.conditioning.style_resampler import StyleResampler


def _resampler() -> StyleResampler:
    return StyleResampler(
        input_size=16,
        hidden_size=12,
        intermediate_size=24,
        output_size=20,
        query_count=4,
        attention_heads=3,
        norm_eps=1e-6,
        init_std=0.02,
        projection_bias=False,
        linear_dtype=torch.float32,
        sensitive_dtype=torch.float32,
    )


def test_artist_tokens_produce_four_independent_slots() -> None:
    module = _resampler()
    states = torch.randn(2, 5, 7, 16)
    output = module(
        states,
        torch.tensor([[3, 4], [1, -1]]),
        torch.tensor([[True, True], [True, False]]),
        torch.zeros(2, dtype=torch.bool),
    )

    assert output.tokens.shape == (2, 4, 20)
    assert output.mask.all()
    assert not torch.equal(output.tokens[:, 0], output.tokens[:, 1])


def test_missing_dropout_and_all_condition_share_learned_null_tokens() -> None:
    module = _resampler()
    states = torch.randn(3, 4, 7, 16)
    output = module(
        states,
        torch.tensor([[-1], [2], [1]]),
        torch.tensor([[False], [True], [True]]),
        torch.tensor([False, True, True]),
    )

    expected = module.null_tokens.unsqueeze(0).expand(3, -1, -1)
    torch.testing.assert_close(output.tokens, expected)
    assert output.mask.all()


def test_style_gathers_only_artist_span_and_detaches_qwen() -> None:
    module = _resampler()
    states = torch.randn(1, 4, 7, 16, requires_grad=True)
    indices = torch.tensor([[2]])
    mask = torch.tensor([[True]])
    use_null = torch.tensor([False])
    baseline = module(states, indices, mask, use_null).tokens

    changed = states.detach().clone()
    changed[:, :2] += 1000.0
    changed_output = module(changed, indices, mask, use_null).tokens
    torch.testing.assert_close(baseline, changed_output)

    baseline.square().mean().backward()
    assert states.grad is None
    assert module.output_projection.weight.grad is not None


def test_autocast_active_and_null_outputs_share_input_dtype() -> None:
    module = _resampler()
    states = torch.randn(2, 3, 7, 16, dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(
            states,
            torch.tensor([[1], [-1]]),
            torch.tensor([[True], [False]]),
            torch.zeros(2, dtype=torch.bool),
        )

    assert output.tokens.dtype == torch.bfloat16
    torch.testing.assert_close(output.tokens[1], module.null_tokens.to(torch.bfloat16))
