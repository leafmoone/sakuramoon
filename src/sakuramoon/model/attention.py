"""Dense SDPA reference attention with native grouped-query heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sakuramoon.conditioning.rope import QKRoPE2D


def dense_attention_mask(token_mask: torch.Tensor) -> torch.Tensor:
    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a boolean [B,L] tensor")
    return token_mask[:, None, :, None] & token_mask[:, None, None, :]


class DenseGQAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        rope_nope_dim: int,
        rope_y_dim: int,
        rope_x_dim: int,
        rope_position_scale: float,
        rope_theta: float,
        norm_eps: float,
        linear_dtype: torch.dtype,
        projection_bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            hidden_size <= 0
            or q_heads <= 0
            or kv_heads <= 0
            or head_dim <= 0
            or q_heads * head_dim != hidden_size
            or q_heads % kv_heads
        ):
            raise ValueError("attention dimensions violate native GQA")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if projection_bias or dropout != 0.0:
            raise ValueError("DiT attention requires bias=false and dropout=0")
        self.hidden_size = hidden_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden_size,
            q_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            kv_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            kv_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.content_gate = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            dtype=linear_dtype,
        )
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            dtype=linear_dtype,
        )
        self.qk_rope = QKRoPE2D(
            head_dim=head_dim,
            nope_dim=rope_nope_dim,
            y_dim=rope_y_dim,
            x_dim=rope_x_dim,
            position_scale=rope_position_scale,
            theta=rope_theta,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [B,L,hidden_size]")
        batch, length, _ = tokens.shape
        if attention_mask.shape != (batch, 1, length, length):
            raise ValueError("attention_mask must have shape [B,1,L,L]")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean with True meaning allowed")
        if coordinates.shape != (batch, length, 2) or coordinates.dtype != torch.float32:
            raise ValueError("coordinates must be FP32 with shape [B,L,2]")

        query = self.q_proj(tokens).view(
            batch,
            length,
            self.q_heads,
            self.head_dim,
        )
        key = self.k_proj(tokens).view(
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        )
        value = self.v_proj(tokens).view(
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        )
        query, key = self.qk_rope(
            query.flatten(0, 1),
            key.flatten(0, 1),
            coordinates.flatten(0, 1),
        )
        query = (
            query.view(batch, length, self.q_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        key = (
            key.view(batch, length, self.kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value = value.transpose(1, 2).contiguous()
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.hidden_size)
        gated = attended * torch.sigmoid(self.content_gate(tokens))
        output = self.out_proj(gated)
        valid_queries = attention_mask.any(dim=-1).squeeze(1)
        return output * valid_queries.unsqueeze(-1)


__all__ = ["DenseGQAAttention", "dense_attention_mask"]
