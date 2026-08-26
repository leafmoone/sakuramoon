"""U-Flow SR trunk assembly (plan §7.1, §7.4, §7.5) + full AnimeSR model (§8).

Residual-coordinate U-Flow over the frozen Mage-VAE latent (128ch, 16x).
Built from the frozen config specs (anime_sr.config.schema):

    AnimeSRModel(ModelSpec)
      ├── PixelConditionEncoder(PixelEncoderSpec)      §6.2/§6.3
      └── UFlowSR(UFlowSpec, OutputHeadSpec)          §7
            ├── input projection P_r / P_z / P_p64     §7.5
            ├── 5 stages (enc64, enc32, b16, dec32, dec64), 28 blocks §7.2
            ├── down/up/skip connectors                §7.4
            ├── GlobalConditioner (5 stage FiLMs)      §7.6
            └── OutputHead (zero-init exit)            §7.7

Inputs (plan §7.5), all at 64x64 latent resolution:

    h64 = P_r(r_t) + P_z(z_lr) + g64 * P_p64(p64)

* r_t  : residual path at time t, (B, 128, 64, 64) (plan §5)
* z_lr : LQ latent anchor E_Mage(Bicubic4x(LQ)), (B, 128, 64, 64) (§4.3)
* p64/p32/p16 + gap16 : PixelConditionEncoder outputs (§6.2, §6.3)
* t, sigma : per-sample scalar conditions (§5.6 sigma mix)

Pixel projections are small-init (normal(0, 0.01)); P_z is NOT zero-init
(plan §7.5). Gates g64/g32/g16 are learnable sigmoid scalars
(param init 0 -> sigmoid(0) = 0.5).

Tiled inference (§17.2, InferenceSpec.rope_absolute_coordinates):
``offset`` is the tile origin in input-grid (64x64 latent) units, propagated
to deeper stages by integer stride division.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from anime_sr.config.schema import ModelSpec, OutputHeadSpec, UFlowSpec
from anime_sr.model.conditioning import GlobalConditioner, StageFilms
from anime_sr.model.output_head import OutputHead
from anime_sr.model.pixel_encoder import (
    PixelConditionEncoder,
    PixelConditionOutputs,
)
from anime_sr.model.restoration_block import RestorationBlock

__all__ = [
    "AnimeSRModel",
    "UFlowSR",
    "count_parameters",
]


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _small_init(conv: nn.Conv2d) -> None:
    """Plan §7.5: pixel projection small-init."""
    with torch.no_grad():
        nn.init.normal_(conv.weight, 0.0, 0.01)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)


def _window_pattern(depth: int) -> list[bool]:
    """High-res stages: normal, shifted, normal, shifted ... (plan §7.3)."""
    return [i % 2 == 1 for i in range(depth)]


class UFlowSR(nn.Module):
    """Config-driven U-Flow trunk (plan §7).

    The frozen base spec (UFlowSpec defaults) is the Base-128M trunk of
    §7.2; smoke configs (config/smoke.toml) override stages for a small
    structure/correctness canary. Hardware never changes this structure
    (plan §15.1) — only config-driven specs select it.
    """

    def __init__(self, spec: UFlowSpec, head_spec: OutputHeadSpec) -> None:
        super().__init__()
        spec.validate_stage_geometry()
        stages = spec.stages
        d = [s.dim for s in stages]
        grids = [s.grid for s in stages]
        g0 = grids[0]
        if any(g0 % g for g in grids):
            raise ValueError(f"input grid {g0} must be a multiple of every stage grid {grids}")
        for s in stages:
            if s.grid % 8:
                raise ValueError(f"stage grid {s.grid} must be divisible by the 8x8 window")

        self.stage_names = ("enc64", "enc32", "bottleneck16", "dec32", "dec64")
        self.grids = grids
        self.stage_grids: dict[str, int] = dict(zip(self.stage_names, grids))
        self.strides = [g0 // g for g in grids]
        self.head_dim = spec.head_dim

        # ---- input projection (§7.5): 128 -> d[0] at the input grid ----
        latent_ch = head_spec.dim_out  # velocity channels == latent channels (128)
        self.proj_r = nn.Conv2d(latent_ch, d[0], 1)
        self.proj_z = nn.Conv2d(latent_ch, d[0], 1)  # z_lr path: NOT zero-init
        self.proj_p64 = nn.Conv2d(128, d[0], 1)
        _small_init(self.proj_p64)
        self.gate_64 = nn.Parameter(torch.zeros(1))
        self.gate_32 = nn.Parameter(torch.zeros(1))
        self.gate_16 = nn.Parameter(torch.zeros(1))

        # ---- stage-wise pixel injection (1x1, small-init): §7.5 ----
        # p64/p32/p16 channel counts come from the pixel encoder spec
        # (stage_channels [48, 96, 128, 192, 256] -> scales 256..16).
        self.proj_p32 = nn.Conv2d(192, d[1], 1)
        _small_init(self.proj_p32)
        self.proj_p16 = nn.Conv2d(256, d[2], 1)
        _small_init(self.proj_p16)

        # ---- down / up / skip connectors (§7.4) ----
        self.down = nn.ModuleList(
            nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * d[i], d[i + 1], 1))
            for i in range(2)  # 64->32, 32->16
        )
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(d[2], 4 * d[1], 1), nn.PixelShuffle(2)),  # 16->32
                nn.Sequential(nn.Conv2d(d[1], 4 * d[0], 1), nn.PixelShuffle(2)),  # 32->64
            ]
        )
        self.skip = nn.ModuleList(
            nn.Conv2d(d[i] + d[i], d[i], 1) for i in (1, 0)  # at 32, then 64: cat(dec, enc)
        )

        # ---- restoration blocks (§7.2 / §7.3) ----
        self.stages = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        RestorationBlock(
                            s.dim,
                            s.q_heads,
                            s.kv_heads,
                            s.ffn,
                            global_attention=s.attention == "global",
                            head_dim=spec.head_dim,
                            layerscale_init=spec.layerscale_init,
                            qk_norm=spec.qk_norm,
                        )
                        for _ in range(s.depth)
                    ]
                )
                for name, s in zip(self.stage_names, stages)
            }
        )
        self.stage_shifts = {
            name: (
                [False] * s.depth
                if s.attention == "global"
                else _window_pattern(s.depth)
            )
            for name, s in zip(self.stage_names, stages)
        }

        # ---- conditioning (§7.6) + output head (§7.7) ----
        self.conditioner = GlobalConditioner(
            gap_dim=256, stage_dims=tuple(d)
        )
        self.head = OutputHead(head_spec.dim_in, head_spec.dim_out)
        if head_spec.dim_in != d[-1]:
            raise ValueError(
                f"output head dim_in {head_spec.dim_in} must equal the 64x64 stage dim {d[-1]}"
            )

    # ------------------------------------------------------------------
    def _run_stage(
        self,
        h: torch.Tensor,
        stage: str,
        films: StageFilms,
        offset: tuple[int, int],
    ) -> torch.Tensor:
        """Run one stage's blocks over (B, C, H, W) features (row-major tokens)."""
        H = W = self.stage_grids[stage]
        Bx, C, Hx, Wx = h.shape
        if (Hx, Wx) != (H, W):
            raise ValueError(f"{stage} expects grid {H}x{W}, got {Hx}x{Wx}")
        tokens = h.reshape(Bx, C, Hx * Wx).permute(0, 2, 1)  # (B, N, C)
        s = self.strides[self.stage_names.index(stage)]
        ox, oy = offset
        o = (ox // s, oy // s)  # tile origin in this stage's token units
        for blk, shift in zip(cast(nn.ModuleList, self.stages[stage]), self.stage_shifts[stage]):
            tokens = blk(
                tokens, H, W, stage, films, self.conditioner, shift=shift, offset=o
            )
        return tokens.permute(0, 2, 1).reshape(Bx, C, Hx, Wx)

    def forward(
        self,
        r_t: torch.Tensor,
        z_lr: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        p64: torch.Tensor | None = None,
        p32: torch.Tensor | None = None,
        p16: torch.Tensor | None = None,
        gap16: torch.Tensor | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """Predict residual velocity v-hat, (B, 128, grid0, grid0) (plan §5.4).

        r_t / z_lr: (B, 128, 64, 64); t / sigma: (B,); pixel features may be
        None (zero conditioning, e.g. smoke tests without the pixel encoder).
        """
        B = r_t.shape[0]
        device, dtype = r_t.device, r_t.dtype

        def zeros(ch: int, h: int, w: int) -> torch.Tensor:
            return torch.zeros((B, ch, h, w), device=device, dtype=dtype)

        g0 = self.grids[0]
        g1 = self.grids[1]
        g2 = self.grids[2]
        p64 = p64 if p64 is not None else zeros(128, g0, g0)
        p32 = p32 if p32 is not None else zeros(192, g1, g1)
        p16 = p16 if p16 is not None else zeros(256, g2, g2)
        gap16 = gap16 if gap16 is not None else torch.zeros((B, 256), device=device, dtype=dtype)

        films = self.conditioner(t, sigma, gap16)

        # ---- input fusion (§7.5) ----
        h64 = (
            self.proj_r(r_t)
            + self.proj_z(z_lr)
            + torch.sigmoid(self.gate_64) * self.proj_p64(p64)
        )
        h64 = self._run_stage(h64, "enc64", films, offset)

        # ---- encoder path ----
        h32 = self.down[0](h64)
        h32 = h32 + torch.sigmoid(self.gate_32) * self.proj_p32(p32)
        h32 = self._run_stage(h32, "enc32", films, offset)
        skip_32 = h32

        h16 = self.down[1](h32)
        h16 = h16 + torch.sigmoid(self.gate_16) * self.proj_p16(p16)
        h16 = self._run_stage(h16, "bottleneck16", films, offset)

        # ---- decoder path (U-shaped skips, §7.4) ----
        h = self.skip[0](torch.cat([self.up[0](h16), skip_32], dim=1))
        h = self._run_stage(h, "dec32", films, offset)

        h = self.skip[1](torch.cat([self.up[1](h), h64], dim=1))
        h = self._run_stage(h, "dec64", films, offset)

        return self.head(h)  # (B, 128, g0, g0)


class AnimeSRModel(nn.Module):
    """Full model (plan §8): PixelConditionEncoder + U-Flow trunk.

    The frozen Mage-VAE is NOT part of this module (see anime_sr.vae).
    Forward: residual path -> velocity; the caller applies the flow sampler
    (anime_sr.flow) to get z-hr = z_lr + delta-hat.
    """

    def __init__(self, model_spec: ModelSpec | None = None) -> None:
        super().__init__()
        spec = model_spec if model_spec is not None else ModelSpec()
        spec.validate_structure()
        self.model_spec = spec
        self.pixel_encoder = PixelConditionEncoder(in_channels=spec.pixel_encoder.in_channels)
        self.trunk = UFlowSR(spec.uflow, spec.output_head)

    def forward(
        self,
        r_t: torch.Tensor,
        z_lr: torch.Tensor,
        lq_rgb: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        offset: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """r_t/z_lr: (B,128,64,64); lq_rgb: (B,3,256,256) -> v-hat (B,128,64,64)."""
        pc: PixelConditionOutputs = self.pixel_encoder(lq_rgb)
        p64, p32, p16, gap16 = pc.v1_subset()
        return self.trunk(r_t, z_lr, t, sigma, p64, p32, p16, gap16, offset=offset)

    @property
    def num_params(self) -> int:
        return count_parameters(self)
