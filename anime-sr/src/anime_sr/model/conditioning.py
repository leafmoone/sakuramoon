"""Global conditioning: timestep + sigma + GAP(p16) -> five stage FiLMs (plan §7.6).

Frozen by §7.6:

    g = MLP( sinusoidal(t) || sinusoidal(sigma) || projection(GAP(p16)) )

``g`` drives five stage-level FiLM groups (Encoder64, Encoder32,
Bottleneck16, Decoder32, Decoder64). Each stage shares ONE projection
``g -> (scale, shift)`` (zero-init, so FiLM is identity at init); no per-block
large conditioner is allowed (§7.6: "each stage shares one projection, do not
build a large per-block conditioner"). Small per-block biases are permitted by
the plan but v1 does not add them.

FiLM form:  y = (1 + scale) * x + shift, applied after the block RMSNorm
(plan §7.3 "stage FiLM" slot).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "GlobalConditioner",
    "StageFilms",
    "TimestepEmbedding",
]

# Per-stage FiLM group names, in trunk order (§7.6).
STAGE_ORDER = ("enc64", "enc32", "bottleneck16", "dec32", "dec64")


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """values: (B,) float -> (B, dim) sinusoidal features (base 10000)."""
    if values.dim() != 1:
        raise ValueError(f"values must be 1-D, got shape {tuple(values.shape)}")
    half = dim // 2
    device = values.device
    # inv_freq[i] = 10000 ** (-2 i / dim)  (RoPE-style geometric series)
    inv_freq = 10000.0 ** (-2.0 * torch.arange(half, dtype=torch.float32, device=device) / dim)
    ang = values.to(torch.float32).unsqueeze(1) * inv_freq
    return torch.cat([ang.sin(), ang.cos()], dim=1)


@dataclass(frozen=True)
class StageFilms:
    """Per-stage FiLM pairs (B, 2*dim_stage): [scale | shift], trunk order.

    (enc64, enc32, bottleneck16, dec32, dec64) matching §7.6.
    """

    enc64: torch.Tensor
    enc32: torch.Tensor
    bottleneck16: torch.Tensor
    dec32: torch.Tensor
    dec64: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (self.enc64, self.enc32, self.bottleneck16, self.dec32, self.dec64)


class TimestepEmbedding(nn.Module):
    """Sinusoidal t (or sigma) embedding: (B,) -> (B, emb_dim)."""

    def __init__(self, freq_dim: int = 256, emb_dim: int = 512) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.net = nn.Sequential(
            nn.Linear(freq_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(values, self.freq_dim))


class GlobalConditioner(nn.Module):
    """§7.6: t + sigma + GAP(p16) -> g -> five stage FiLM projections (zero-init)."""

    def __init__(
        self,
        gap_dim: int = 256,
        freq_dim: int = 256,
        emb_dim: int = 512,
        g_dim: int = 768,
        stage_dims: tuple[int, ...] = (384, 512, 768, 512, 384),
    ) -> None:
        super().__init__()
        if len(stage_dims) != 5:
            raise ValueError(f"stage_dims must have 5 entries, got {len(stage_dims)}")
        self.t_embed = TimestepEmbedding(freq_dim, emb_dim)
        self.sigma_embed = TimestepEmbedding(freq_dim, emb_dim)
        self.gap_proj = nn.Linear(gap_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, g_dim),
        )
        films = [nn.Linear(g_dim, 2 * d, bias=True) for d in stage_dims]
        self.stage_films = nn.ModuleList(films)
        # Zero-init: at init every FiLM is the identity (scale=0, shift=0).
        for film in films:
            nn.init.zeros_(film.weight)
            if film.bias is not None:
                nn.init.zeros_(film.bias)

    def forward(
        self,
        t: torch.Tensor,
        sigma: torch.Tensor,
        gap16: torch.Tensor,
    ) -> StageFilms:
        """t/sigma: (B,) in [0,1]; gap16: (B, 256). Returns per-stage (B, 2*dim) pairs."""
        g = self.mlp(
            torch.cat([self.t_embed(t), self.sigma_embed(sigma), self.gap_proj(gap16)], dim=1)
        )
        return StageFilms(*[film(g) for film in self.stage_films])

    # Overrides nn.Module.apply(fn) on purpose: the conditioner is called as
    # cond.apply(stage, x, films) and is never fed through module.apply.
    def apply(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        stage: str,
        x: torch.Tensor,
        films: StageFilms,
    ) -> torch.Tensor:
        """Modulate normalized features: (1 + scale) * x + shift (identity at init).

        x: (B, N, dim) token layout or (B, dim, H, W) spatial layout.
        """
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}, expected one of {STAGE_ORDER}")
        pair = getattr(films, stage)  # (B, 2*dim)
        scale, shift = pair.chunk(2, dim=1)  # each (B, dim)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        elif x.dim() == 4:
            scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
            shift = shift.view(shift.shape[0], shift.shape[1], 1, 1)
        else:
            raise ValueError(f"x must be 3-D (B,N,dim) or 4-D (B,C,H,W), got {x.dim()}-D")
        return (1.0 + scale) * x + shift
