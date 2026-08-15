from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.style_resampler import (
    StyleConditionEncoder,
    StyleResampler,
)


def _encoder() -> StyleConditionEncoder:
    return StyleConditionEncoder(
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


def test_condition_tokens_produce_four_independent_slots() -> None:
    module = _encoder()
    states = torch.randn(2, 5, 7, 16)
    output = module(
        states,
        torch.tensor([[3, 4], [1, -1]]),
        torch.tensor([[True, True], [True, False]]),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([0, 1]),
    )

    assert output.tokens.shape == (2, 4, 20)
    assert output.mask.all()
    assert not torch.equal(output.tokens[:, 0], output.tokens[:, 1])


def test_missing_dropout_and_all_condition_share_learned_null_tokens() -> None:
    module = _encoder()
    states = torch.randn(3, 4, 7, 16)
    output = module(
        states,
        torch.tensor([[-1], [2], [1]]),
        torch.tensor([[False], [True], [True]]),
        torch.tensor([False, True, True]),
        torch.empty(0, dtype=torch.long),
    )

    expected = module.null_tokens.unsqueeze(0).expand(3, -1, -1)
    torch.testing.assert_close(output.tokens, expected)
    assert output.mask.all()


def test_mixed_batch_projects_only_active_samples() -> None:
    module = _encoder()
    states = torch.randn(3, 4, 7, 16)
    projected_batch_sizes: list[int] = []

    def record_projected_batch(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        projected_batch_sizes.append(inputs[0].shape[0])

    handle = module.input_projection.register_forward_pre_hook(record_projected_batch)
    try:
        output = module(
            states,
            torch.tensor([[2, 999], [999, 999], [1, 999]]),
            torch.tensor(
                [[True, False], [True, True], [True, False]],
                dtype=torch.bool,
            ),
            torch.tensor([False, True, True]),
            torch.tensor([0]),
        )
    finally:
        handle.remove()

    assert projected_batch_sizes == [1]
    expected_null = module.null_tokens.to(output.tokens.dtype)
    torch.testing.assert_close(output.tokens[1], expected_null)
    torch.testing.assert_close(output.tokens[2], expected_null)


def test_inactive_large_index_does_not_reach_gather() -> None:
    module = _encoder()
    states = torch.randn(1, 4, 7, 16)
    mask = torch.tensor([[True, False]])

    baseline = module(
        states,
        torch.tensor([[2, -1]]),
        mask,
        torch.tensor([False]),
        torch.tensor([0]),
    )
    with_large_inactive_index = module(
        states,
        torch.tensor([[2, 10000]]),
        mask,
        torch.tensor([False]),
        torch.tensor([0]),
    )

    torch.testing.assert_close(
        with_large_inactive_index.tokens,
        baseline.tokens,
        atol=0,
        rtol=0,
    )


def test_style_gathers_only_condition_span_and_detaches_qwen() -> None:
    module = _encoder()
    states = torch.randn(1, 4, 7, 16, requires_grad=True)
    indices = torch.tensor([[2]])
    mask = torch.tensor([[True]])
    use_null = torch.tensor([False])

    changed = states.detach().clone()
    changed[:, :2] += 1000.0
    active_samples = torch.tensor([0])
    baseline = module(states, indices, mask, use_null, active_samples).tokens

    changed_output = module(
        changed, indices, mask, use_null, active_samples
    ).tokens
    torch.testing.assert_close(baseline, changed_output)

    baseline.square().mean().backward()
    assert states.grad is None
    assert module.output_projection.weight.grad is not None


@pytest.mark.parametrize(
    "active_samples",
    [
        torch.empty(0, dtype=torch.long),
        torch.tensor([1]),
        torch.tensor([0, 0]),
    ],
)
def test_invalid_active_sample_plan_fails(active_samples: torch.Tensor) -> None:
    module = _encoder()

    with pytest.raises(ValueError, match="sample/index plan"):
        module(
            torch.randn(2, 3, 7, 16),
            torch.tensor([[1], [-1]]),
            torch.tensor([[True], [False]]),
            torch.tensor([False, True]),
            active_samples,
        )


@pytest.mark.parametrize("invalid_index", [-1, 3])
def test_active_condition_index_outside_qwen_sequence_fails(
    invalid_index: int,
) -> None:
    module = _encoder()

    with pytest.raises(ValueError, match="sample/index plan"):
        module(
            torch.randn(1, 3, 7, 16),
            torch.tensor([[invalid_index]]),
            torch.tensor([[True]]),
            torch.tensor([False]),
            torch.tensor([0]),
        )


def test_autocast_active_and_null_outputs_share_input_dtype() -> None:
    module = _encoder()
    states = torch.randn(2, 3, 7, 16, dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(
            states,
            torch.tensor([[1], [-1]]),
            torch.tensor([[True], [False]]),
            torch.zeros(2, dtype=torch.bool),
            torch.tensor([0]),
        )

    assert output.tokens.dtype == torch.bfloat16
    torch.testing.assert_close(output.tokens[1], module.null_tokens.to(torch.bfloat16))


def test_legacy_style_resampler_alias_preserves_exact_class_and_state_keys() -> None:
    assert StyleResampler is StyleConditionEncoder
    current = _encoder()
    legacy = StyleResampler(
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

    expected_keys = (
        "layer_embedding",
        "queries",
        "null_tokens",
        "shared_norm.weight",
        "input_projection.weight",
        "cross_attention.in_proj_weight",
        "cross_attention.out_proj.weight",
        "style_mlp.norm.weight",
        "style_mlp.gate.weight",
        "style_mlp.up.weight",
        "style_mlp.down.weight",
        "output_projection.weight",
    )
    assert tuple(current.state_dict()) == expected_keys
    assert tuple(legacy.state_dict()) == expected_keys
