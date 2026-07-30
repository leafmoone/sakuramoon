"""Small normalization primitives shared by conditioning modules."""

from __future__ import annotations

import torch
from torch import nn


class FP32RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        if size <= 0 or eps <= 0.0:
            raise ValueError("RMSNorm size and epsilon must be positive")
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        source_dtype = tensor.dtype
        normalized = tensor.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(source_dtype) * self.weight.to(source_dtype)


__all__ = ["FP32RMSNorm"]
