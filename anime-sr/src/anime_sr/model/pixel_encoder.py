"""Pixel Condition Encoder (plan §6.2, §6.3, §7.5).

Encodes the 256x256 LQ image into per-scale conditioning features for the
U-Flow trunk:

    RGB 256x256
      Stem 3x3 Conv (3 -> 48) + 2 blocks (48ch)
      -> Downsample -> 2 blocks (96ch) @128
      -> Downsample -> 3 blocks (128ch) @64
      -> Downsample -> 3 blocks (192ch) @32
      -> Downsample -> 4 blocks (256ch) @16

Each PixelConditionBlock (plan §6.2, frozen):
    LayerNorm2d -> depthwise 3x3 -> pointwise expand x2 -> SiLU
    -> pointwise projection -> residual

Downsample between stages: PixelUnshuffle(2) + 1x1 projection (same
convention as the U-Flow trunk, plan §7.4, "preserve local information").

Outputs (plan §6.3):
    p256/p128  optional decoder grounding (v1 core: unused, §9.1)
    p64        U-Flow 64x64 input condition
    p32        U-Flow encoder 32x32 condition
    p16        bottleneck condition
    gap16      GAP(p16) global degradation/content summary

The plan's 9-12M estimate for this module is a loose one; the structural
spec is authoritative (plan §15.1). Actual parameter count is measured at
build time and recorded in design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "PixelConditionBlock",
    "PixelConditionEncoder",
    "PixelConditionOutputs",
]


class PixelConditionBlock(nn.Module):
    """LN2d -> dw3x3 -> pw expand 2x -> SiLU -> pw project -> residual (§6.2)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.expand = nn.Conv2d(dim, 2 * dim, 1)
        self.act = nn.SiLU()
        self.project = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x.reshape(B, C, H * W).transpose(1, 2))  # (B, N, C): LN per token
        h = h.transpose(1, 2).reshape(B, C, H, W)
        h = self.dw(h)
        h = self.project(self.act(self.expand(h)))
        return h + x


def _downsample(dim_in: int, dim_out: int) -> nn.Sequential:
    """PixelUnshuffle(2) + 1x1 projection (plan §7.4 convention)."""
    return nn.Sequential(
        nn.PixelUnshuffle(2),
        nn.Conv2d(4 * dim_in, dim_out, 1),
    )


@dataclass(frozen=True)
class PixelConditionOutputs:
    """Per-scale conditioning features (all BHWC channels-first tensors)."""

    p256: torch.Tensor  # (B, 48, 256, 256)
    p128: torch.Tensor  # (B, 96, 128, 128)
    p64: torch.Tensor  # (B, 128, 64, 64)
    p32: torch.Tensor  # (B, 192, 32, 32)
    p16: torch.Tensor  # (B, 256, 16, 16)
    gap16: torch.Tensor  # (B, 256)

    def v1_subset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(p64, p32, p16, gap16): what the v1 U-Flow trunk consumes (§6.3)."""
        return self.p64, self.p32, self.p16, self.gap16


class PixelConditionEncoder(nn.Module):
    """§6.2 pixel condition encoder: 256x256 RGB -> five scales + global summary."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.stem = nn.Conv2d(in_channels, 48, 3, padding=1)
        self.blocks_s256 = nn.Sequential(PixelConditionBlock(48), PixelConditionBlock(48))
        self.down_256 = _downsample(48, 96)
        self.blocks_s128 = nn.Sequential(PixelConditionBlock(96), PixelConditionBlock(96))
        self.down_128 = _downsample(96, 128)
        self.blocks_s64 = nn.Sequential(*(PixelConditionBlock(128) for _ in range(3)))
        self.down_64 = _downsample(128, 192)
        self.blocks_s32 = nn.Sequential(*(PixelConditionBlock(192) for _ in range(3)))
        self.down_32 = _downsample(192, 256)
        self.blocks_s16 = nn.Sequential(*(PixelConditionBlock(256) for _ in range(4)))

    def forward(self, x: torch.Tensor) -> PixelConditionOutputs:
        """x: (B, 3, 256, 256) -> PixelConditionOutputs."""
        _, C, H, W = x.shape
        if (C, H, W) != (3, 256, 256):
            raise ValueError(f"expected (B, 3, 256, 256), got {tuple(x.shape)}")
        h = self.blocks_s256(self.stem(x))
        p256 = h
        h = self.blocks_s128(self.down_256(h))
        p128 = h
        h = self.blocks_s64(self.down_128(h))
        p64 = h
        h = self.blocks_s32(self.down_64(h))
        p32 = h
        h = self.blocks_s16(self.down_32(h))
        p16 = h
        gap16 = p16.mean(dim=(2, 3))
        return PixelConditionOutputs(
            p256=p256, p128=p128, p64=p64, p32=p32, p16=p16, gap16=gap16
        )
