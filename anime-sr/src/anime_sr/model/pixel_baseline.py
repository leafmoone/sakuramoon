"""M2 deterministic Pixel Baseline (plan §M2, step 7).

A lightweight NAFNet-like U-Net in *pixel space*: LQ (B,3,H,W) -> HR
(B,3,4H,4W). Purpose per the plan: validate data + degradation, establish
the fidelity floor, and decide whether the 128M flow model earns its cost.
No flow sampling, no attention, no EMA tricks — a plain CNN whose only
non-linearity budget is the NAF (non-local affine feature) blocks.

Frozen band (plan §M2): 5M-10M parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NAFBlock(nn.Module):
    """NAFNet-style block: GN + dw 7x7 (non-local) + two 1x1 affine transforms."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pw1 = nn.Conv2d(dim, dim, 1)
        self.pw2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.pw1(h))
        h = F.gelu(self.dw(h))
        h = self.pw2(h)
        return x + h


def _naf(depth: int, dim: int) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(dim) for _ in range(depth)])


class PixelBaseline(nn.Module):
    """4x pixel-space SR U-Net with NAF blocks (5M-10M params, plan §M2).

    Layout (B = base channels): 256^2 grid at B, 128^2 at 2B, 64^2 at 4B
    (encoder), mirror decoder with 2x interpolates + skip connections, and
    a single 4x PixelShuffle head so the output is exactly 4x the input.
    """

    def __init__(self, base_channels: int = 160, depth: int = 2) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError(f"base_channels {base_channels} must be divisible by 8 (GroupNorm)")
        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels
        self.stem = nn.Conv2d(3, c1, 3, padding=1)
        self.enc1 = _naf(depth, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = _naf(depth, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = _naf(depth + 1, c3)
        self.up1 = nn.Conv2d(c3 + c2, c2, 3, padding=1)  # skip concat from e2
        self.dec1 = _naf(depth, c2)
        self.up2 = nn.Conv2d(c2 + c1, c1, 3, padding=1)  # skip concat from e1
        self.dec2 = _naf(depth, c1)
        self.head = nn.Conv2d(c1, 3 * 4 * 4, 3, padding=1)  # 4x PixelShuffle

    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        """lq: (B, 3, H, W) with H, W multiples of 16 -> (B, 3, 4H, 4W)."""
        if lq.dim() != 4 or lq.shape[1] != 3:
            raise ValueError(f"lq must be (B, 3, H, W), got {tuple(lq.shape)}")
        if lq.shape[-1] % 16 or lq.shape[-2] % 16:
            raise ValueError(f"lq spatial size must be a multiple of 16, got {lq.shape[-2:]}")
        e1 = F.silu(self.stem(lq))
        e1 = self.enc1(e1)
        d1 = F.silu(self.down1(e1))
        e2 = self.enc2(d1)
        d2 = F.silu(self.down2(e2))
        m = self.mid(d2)
        u1 = F.interpolate(m, scale_factor=2, mode="nearest")
        u1 = torch.cat([u1, e2], dim=1)
        u1 = F.silu(self.up1(u1))
        u1 = self.dec1(u1)
        u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u2 = torch.cat([u2, e1], dim=1)
        u2 = F.silu(self.up2(u2))
        u2 = self.dec2(u2)
        out = F.pixel_shuffle(self.head(u2), 4)
        return out

    @classmethod
    def n_params(cls, base_channels: int = 160, depth: int = 2) -> int:
        """Parameter count of the default-config construction."""
        m = cls(base_channels, depth)
        return sum(p.numel() for p in m.parameters())
