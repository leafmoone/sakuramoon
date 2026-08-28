from __future__ import annotations

import torch

from sakuramoon.conditioning.global_condition import BlockModulation
from sakuramoon.model.block import DiTBlock
from sakuramoon.model.growth import packed_growth_alpha


def _block() -> DiTBlock:
    return DiTBlock(
        hidden_size=8,
        intermediate_size=16,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        rope_nope_dim=0,
        rope_y_dim=2,
        rope_x_dim=2,
        rope_position_scale=1.0,
        rope_theta=10.0,
        norm_eps=1e-6,
        linear_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
    )


def test_attention_and_mlp_growth_are_both_gated_and_trainable() -> None:
    torch.manual_seed(77)
    block = _block()
    tokens = torch.randn(1, 3, 8)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    attention_mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    coordinates = torch.zeros(1, 3, 2, dtype=torch.float32)
    modulation = BlockModulation(
        attention_scale=torch.zeros(1, 8),
        attention_shift=torch.zeros(1, 8),
        attention_gate=torch.ones(1, 8),
        mlp_scale=torch.zeros(1, 8),
        mlp_shift=torch.zeros(1, 8),
        mlp_gate=torch.ones(1, 8),
    )

    base = block(
        tokens,
        token_mask,
        attention_mask,
        coordinates,
        modulation,
        attention_growth=0.0,
        mlp_growth=0.0,
    )
    attention_only = block(
        tokens,
        token_mask,
        attention_mask,
        coordinates,
        modulation,
        attention_growth=1.0,
        mlp_growth=0.0,
    )
    mlp_only = block(
        tokens,
        token_mask,
        attention_mask,
        coordinates,
        modulation,
        attention_growth=0.0,
        mlp_growth=1.0,
    )
    both = block(
        tokens,
        token_mask,
        attention_mask,
        coordinates,
        modulation,
        attention_growth=1.0,
        mlp_growth=1.0,
    )

    assert torch.equal(base, tokens)
    assert not torch.equal(attention_only, base)
    assert not torch.equal(mlp_only, base)
    assert not torch.equal(attention_only, mlp_only)
    assert not torch.equal(both, attention_only)
    both.square().mean().backward()
    assert block.attention.q_proj.weight.grad is not None
    assert block.mlp.in_proj.weight.grad is not None
    assert torch.count_nonzero(block.attention.q_proj.weight.grad) > 0
    assert torch.count_nonzero(block.mlp.in_proj.weight.grad) > 0


def test_packed_growth_alpha_does_not_specialize_each_ramp_value() -> None:
    compile_count = 0

    def backend(
        graph: torch.fx.GraphModule,
        _example_inputs: list[torch.Tensor],
    ) -> torch.fx.GraphModule:
        nonlocal compile_count
        compile_count += 1
        return graph

    def apply_growth(tokens: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return tokens * alpha

    reference = torch.ones(4, dtype=torch.float32)
    compiled = torch.compile(apply_growth, backend=backend, dynamic=True)
    for index in range(12):
        alpha = packed_growth_alpha(20, index / 11.0, reference)
        expected = reference * (index / 11.0)
        assert torch.equal(compiled(reference, alpha), expected)

    assert compile_count == 1


def test_packed_growth_alpha_is_fail_closed() -> None:
    reference = torch.ones(4, dtype=torch.float32)
    for invalid in (-0.1, 1.1):
        try:
            packed_growth_alpha(20, invalid, reference)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range packed growth alpha was accepted")

    try:
        packed_growth_alpha(20, 0.5, torch.ones(4, dtype=torch.int64))
    except TypeError:
        pass
    else:
        raise AssertionError("integer packed growth reference was accepted")
