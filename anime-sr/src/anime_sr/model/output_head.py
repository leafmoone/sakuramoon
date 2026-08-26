"""Output head (plan §7.7): 64x64 trunk features -> 128-channel velocity.

Frozen by §7.7:

    RMSNorm -> 3x3 Conv (384 -> 384) -> SiLU -> 3x3 Conv (384 -> 128)

The last convolution is zero-init (weight and bias), so at init v-hat = 0
and the one-step model outputs  z-hr = z_lr + r0 (plan §7.7: "at init the
model is equivalent to emitting the aligned LQ latent directly", Faithful
sigma=0).
"""

from __future__ import annotations

import torch
from torch import nn

from anime_sr.model.window_attention import RMSNorm2d

__all__ = ["OutputHead"]


class OutputHead(nn.Module):
    """§7.7: (B, 384, 64, 64) -> (B, 128, 64, 64) velocity field (zero-init exit)."""

    def __init__(self, in_dim: int = 384, out_dim: int = 128) -> None:
        super().__init__()
        self.norm = RMSNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim, H, W) -> (B, out_dim, H, W)."""
        B, C, H, W = x.shape
        h = self.norm(x.reshape(B, C, H * W).transpose(1, 2))  # RMSNorm per token (B, N, C)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        h = self.act(self.conv1(h))
        return self.conv2(h)
