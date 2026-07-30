"""Bias-free SwiGLU feed-forward layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        linear_dtype: torch.dtype,
        projection_bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("SwiGLU dimensions must be positive")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if projection_bias or dropout != 0.0:
            raise ValueError("DiT SwiGLU requires bias=false and dropout=0")
        self.in_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            dtype=linear_dtype,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=linear_dtype,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate, up = self.in_proj(tokens).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


__all__ = ["SwiGLU"]
