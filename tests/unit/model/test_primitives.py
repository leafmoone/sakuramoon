from __future__ import annotations

import torch
import torch.nn.functional as F

from sakuramoon.conditioning.global_condition import BlockModulation
from sakuramoon.model.attention import DenseGQAAttention, dense_attention_mask
from sakuramoon.model.block import DiTBlock
from sakuramoon.model.mlp import SwiGLU
from sakuramoon.model.norm import RMSNorm


def _attention() -> DenseGQAAttention:
    return DenseGQAAttention(
        hidden_size=8,
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
        dropout=0.0,
    )


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


def _modulation(
    batch: int,
    hidden: int,
    *,
    attention_gate: float,
    mlp_gate: float,
) -> BlockModulation:
    zero = torch.zeros(batch, hidden)
    return BlockModulation(
        attention_scale=zero,
        attention_shift=zero,
        attention_gate=torch.full_like(zero, attention_gate),
        mlp_scale=zero,
        mlp_shift=zero,
        mlp_gate=torch.full_like(zero, mlp_gate),
    )


def test_rms_norm_accumulates_in_fp32_and_returns_input_dtype() -> None:
    module = RMSNorm(8, 1e-6)
    tokens = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    output = module(tokens)
    expected = tokens.float() * torch.rsqrt(
        tokens.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )

    assert module.weight.dtype == torch.float32
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output.float(), expected, atol=0.01, rtol=0.01)


def test_dense_mask_blocks_both_padding_queries_and_keys() -> None:
    token_mask = torch.tensor([[True, True, False]])

    mask = dense_attention_mask(token_mask)

    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, :2, :2].all()
    assert not mask[0, 0, 2].any()
    assert not mask[0, 0, :, 2].any()


def test_attention_matches_native_gqa_content_gate_formula() -> None:
    module = _attention()
    tokens = torch.randn(1, 3, 8)
    coordinates = torch.zeros(1, 3, 2)
    mask = dense_attention_mask(torch.ones(1, 3, dtype=torch.bool))

    output = module(tokens, mask, coordinates)
    query = module.q_proj(tokens).view(3, 2, 4)
    key = module.k_proj(tokens).view(3, 1, 4)
    query, key = module.qk_rope(query, key, coordinates.flatten(0, 1))
    value = module.v_proj(tokens).view(1, 3, 1, 4).transpose(1, 2)
    attended = F.scaled_dot_product_attention(
        query.view(1, 3, 2, 4).transpose(1, 2),
        key.view(1, 3, 1, 4).transpose(1, 2),
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )
    attended = attended.transpose(1, 2).reshape(1, 3, 8)
    expected = module.out_proj(attended * torch.sigmoid(module.content_gate(tokens)))

    assert module.k_proj.out_features == 4
    assert module.v_proj.out_features == 4
    torch.testing.assert_close(output, expected)


def test_attention_padding_cannot_affect_valid_tokens() -> None:
    module = _attention()
    tokens = torch.randn(1, 3, 8)
    changed = tokens.clone()
    changed[:, 2] = 1000.0
    coordinates = torch.zeros(1, 3, 2)
    mask = dense_attention_mask(torch.tensor([[True, True, False]]))

    output = module(tokens, mask, coordinates)
    changed_output = module(changed, mask, coordinates)

    torch.testing.assert_close(output[:, :2], changed_output[:, :2])
    assert torch.count_nonzero(output[:, 2]) == 0
    assert torch.count_nonzero(changed_output[:, 2]) == 0


def test_swiglu_matches_explicit_formula_without_bias() -> None:
    module = SwiGLU(
        hidden_size=8,
        intermediate_size=16,
        linear_dtype=torch.float32,
        projection_bias=False,
        dropout=0.0,
    )
    tokens = torch.randn(2, 3, 8)

    output = module(tokens)
    gate, up = module.in_proj(tokens).chunk(2, dim=-1)
    expected = module.down_proj(F.silu(gate) * up)

    assert module.in_proj.bias is None
    assert module.down_proj.bias is None
    torch.testing.assert_close(output, expected)


def test_block_uses_raw_condition_gates_and_clears_padding() -> None:
    module = _block()
    tokens = torch.randn(1, 3, 8)
    token_mask = torch.tensor([[True, True, False]])
    coordinates = torch.zeros(1, 3, 2)
    modulation = _modulation(1, 8, attention_gate=2.0, mlp_gate=3.0)

    output = module(
        tokens,
        token_mask,
        dense_attention_mask(token_mask),
        coordinates,
        modulation,
        attention_growth=1.0,
        mlp_growth=1.0,
    )
    base = tokens * token_mask.unsqueeze(-1)
    attention_input = module.attention_norm(base)
    attention_output = module.attention(
        attention_input,
        dense_attention_mask(token_mask),
        coordinates,
    )
    after_attention = (base + 2.0 * attention_output) * token_mask.unsqueeze(-1)
    expected = after_attention + 3.0 * module.mlp(module.mlp_norm(after_attention))
    expected = expected * token_mask.unsqueeze(-1)

    torch.testing.assert_close(output, expected)
    assert torch.count_nonzero(output[:, 2]) == 0


def test_zero_condition_gates_make_block_identity_with_finite_gradients() -> None:
    module = _block()
    tokens = torch.randn(2, 3, 8, requires_grad=True)
    token_mask = torch.ones(2, 3, dtype=torch.bool)
    coordinates = torch.zeros(2, 3, 2)
    modulation = _modulation(2, 8, attention_gate=0.0, mlp_gate=0.0)

    output = module(
        tokens,
        token_mask,
        dense_attention_mask(token_mask),
        coordinates,
        modulation,
        attention_growth=1.0,
        mlp_growth=1.0,
    )
    output.sum().backward()

    torch.testing.assert_close(output, tokens)
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_zero_growth_makes_nonzero_condition_gates_an_identity() -> None:
    module = _block()
    tokens = torch.randn(1, 3, 8)
    token_mask = torch.ones(1, 3, dtype=torch.bool)
    coordinates = torch.zeros(1, 3, 2)
    modulation = _modulation(1, 8, attention_gate=2.0, mlp_gate=3.0)

    output = module(
        tokens,
        token_mask,
        dense_attention_mask(token_mask),
        coordinates,
        modulation,
        attention_growth=0.0,
        mlp_growth=0.0,
    )

    torch.testing.assert_close(output, tokens)


def test_production_block_parameter_count_and_policy() -> None:
    module = DiTBlock(
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
        linear_dtype=torch.bfloat16,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
    )

    assert sum(parameter.numel() for parameter in module.parameters()) == 76_027_136
    linears = (
        module.attention.q_proj,
        module.attention.k_proj,
        module.attention.v_proj,
        module.attention.content_gate,
        module.attention.out_proj,
        module.mlp.in_proj,
        module.mlp.down_proj,
    )
    assert all(linear.bias is None for linear in linears)
    assert not any("layer_scale" in name for name, _ in module.named_parameters())
