"""One non-causal attention-only refinement block."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sakuramoon.conditioning.norm import FP32RMSNorm


class BidirectionalAttentionOnly(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        norm_eps: float,
        layer_scale_init: float,
        projection_bias: bool,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.pre_norm = FP32RMSNorm(hidden_size, norm_eps)
        self.q_norm = FP32RMSNorm(self.head_dim, norm_eps)
        self.k_norm = FP32RMSNorm(self.head_dim, norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=projection_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=projection_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=projection_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=projection_bias)
        self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("tokens must be [B,L,D] with a boolean [B,L] mask")
        batch, length, hidden = tokens.shape
        base = tokens * mask.unsqueeze(-1)
        normalized = self.pre_norm(base)

        def project(layer: nn.Linear, tensor: torch.Tensor) -> torch.Tensor:
            return layer(tensor).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(project(self.q_proj, normalized))
        key = self.k_norm(project(self.k_proj, normalized))
        value = project(self.v_proj, normalized)
        allowed_keys = mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_keys,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        output = base + self.layer_scale.to(base.dtype) * self.out_proj(attended)
        return output * mask.unsqueeze(-1)


__all__ = ["BidirectionalAttentionOnly"]
