from __future__ import annotations

import pytest
import torch

from sakuramoon.model.attention import (
    FA4VarlenGQAAttention,
    ValidatedCuSeqlens,
    fa4_varlen_attention,
)


def _production_attention() -> FA4VarlenGQAAttention:
    return FA4VarlenGQAAttention(
        hidden_size=2560,
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
        dropout=0.0,
    )


def test_fa4_attention_keeps_native_five_head_kv_projections() -> None:
    module = _production_attention()

    assert module.q_proj.out_features == 20 * 128
    assert module.k_proj.out_features == 5 * 128
    assert module.v_proj.out_features == 5 * 128
    assert sum(parameter.numel() for parameter in module.parameters()) == 22_937_856
    assert all(
        projection.bias is None
        for projection in (
            module.q_proj,
            module.k_proj,
            module.v_proj,
            module.content_gate,
            module.out_proj,
        )
    )


def test_fa4_attention_rejects_nonproduction_shape() -> None:
    with pytest.raises(ValueError, match="locked"):
        FA4VarlenGQAAttention(
            hidden_size=128,
            q_heads=1,
            kv_heads=1,
            head_dim=128,
            rope_nope_dim=32,
            rope_y_dim=48,
            rope_x_dim=48,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            linear_dtype=torch.bfloat16,
            projection_bias=False,
            dropout=0.0,
        )


def test_fa4_core_rejects_cpu_instead_of_falling_back() -> None:
    query = torch.zeros(2, 20, 128, dtype=torch.bfloat16)
    key = torch.zeros(2, 5, 128, dtype=torch.bfloat16)
    value = torch.zeros(2, 5, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    boundaries = ValidatedCuSeqlens(cu_seqlens, 2, 2, 1)

    with pytest.raises(ValueError, match="CUDA"):
        fa4_varlen_attention(query, key, value, boundaries)
