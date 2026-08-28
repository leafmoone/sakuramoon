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

Dynamic U-Flow (option A). The input latent grid (H, W) is chosen by the
caller per bucket (512/768/1024 HR -> 32/48/64, plus non-square crops); the
five stage grids follow as H/2, W/2 (enc32), H/4, W/4 (bottleneck), then the
decoder mirrors. Windows that do not tile an 8-multiple stage grid are
zero-padded to the next 8-multiple with a key-validity mask (clipped
neighbourhood, no wrap); exact-tile grids run the un-padded path unchanged.

Inputs (plan §7.5), at the caller's input latent grid (H, W):

    h_in = P_r(r_t) + P_z(z_lr) + g64 * P_p64(p64)

* r_t  : residual path at time t, (B, 128, H, W) (plan §5)
* z_lr : LQ latent anchor E_Mage(Bicubic4x(LQ)), (B, 128, H, W) (§4.3)
* p64/p32/p16 + gap16 : PixelConditionEncoder outputs at H, H/2, H/4 (§6.2, §6.3)
* t, sigma : per-sample scalar conditions (§5.6 sigma mix)

Pixel projections are small-init (normal(0, 0.01)); P_z is NOT zero-init
(plan §7.5). Gates g64/g32/g16 are learnable sigmoid scalars
(param init 0 -> sigmoid(0) = 0.5).

Tiled inference (§17.2, InferenceSpec.rope_absolute_coordinates):
``offset`` is the tile origin in input-grid units, propagated to deeper
stages by integer stride division.
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
    "apply_pixel_zero_init",
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


#: The four pixel-path weights that the Phase I-P transition zeroes (weights
#: only — trained biases and learned gates are kept). See
#: :func:`apply_pixel_zero_init` for why this is bit-identical to the
#: trunk-only model.
_PIXEL_ZERO_INIT_PARAMS = (
    "proj_p64.weight",
    "proj_p32.weight",
    "proj_p16.weight",
    "conditioner.gap_proj.weight",
)


def apply_pixel_zero_init(trunk: UFlowSR) -> list[str]:
    """Zero the pixel-path weights of a trunk (Phase I-P transition).

    A trunk-only (M4-L0) checkpoint never receives pixel gradients: the
    pixel projections were fed zeros, so their *weights* stay at
    small-init while their *biases* train. Zeroing exactly the four
    weights in :data:`_PIXEL_ZERO_INIT_PARAMS` (keeping the loaded biases
    and the learned gates) makes every pixel term input-independent:

    * ``proj_p*(p') = 0 * p' + b_old`` and ``sigmoid(gate) * b_old`` are
      identical for any pixel features (IEEE-exact: ``0 * finite == 0``),
      the same terms the old model produced feeding zero features;
    * ``gap_proj(gap') = 0 * gap' + b_gap_old`` — the old model fed
      ``gap16 = 0``, so its term was the same constant ``b_gap_old``.

    The result: a full AnimeSRModel with a loaded trunk-only checkpoint +
    this zero-init is bit-identical to the checkpoint's behavior at step 0,
    and the pixel encoder can then train from exactly that state.

    Returns the parameter names zeroed (always ``_PIXEL_ZERO_INIT_PARAMS``;
    a mismatch means the trunk structure changed and the zero-init set must
    be re-derived). Idempotent.
    """
    zeroed: list[str] = []
    for name, param in trunk.named_parameters():
        if name in _PIXEL_ZERO_INIT_PARAMS:
            with torch.no_grad():
                param.zero_()
            zeroed.append(name)
    if set(zeroed) != set(_PIXEL_ZERO_INIT_PARAMS):
        raise RuntimeError(
            f"pixel zero-init mismatch: found {sorted(zeroed)}, "
            f"expected {sorted(_PIXEL_ZERO_INIT_PARAMS)}"
        )
    return zeroed


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
        # Dynamic U-Flow (option A): stage grids are derived from the INPUT
        # latent grid at forward time (stride 1/2/4/2/1), so one trunk serves
        # 512/768/1024 (latent 32/48/64) and non-square buckets. Stage windows
        # are padded to an 8-multiple at runtime when a stage grid is not
        # 8-divisible (e.g. 768 HR -> 48/24/12: bottleneck 12x12 is global).
        self.strides = [s.stride for s in stages]  # [1, 2, 4, 2, 1]

        self.stage_names = ("enc64", "enc32", "bottleneck16", "dec32", "dec64")
        self.stage_strides: dict[str, int] = dict(zip(self.stage_names, self.strides))
        self.head_dim = spec.head_dim

        # ---- input projection (§7.5): 128 -> d[0] at the input grid ----
        latent_ch = head_spec.dim_out  # velocity channels == latent channels (128)
        self.latent_ch = latent_ch
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
        """Run one stage's blocks over (B, C, H, W) features (row-major tokens).

        The stage grid is read from the incoming feature map (dynamic U-Flow):
        each stage's (H, W) is whatever the down/up connectors produced, so
        non-square and 768-class grids flow through unchanged.
        """
        Bx, C, H, W = h.shape
        tokens = h.reshape(Bx, C, H * W).permute(0, 2, 1)  # (B, N, C)
        s = self.stage_strides[stage]
        ox, oy = offset
        o = (ox // s, oy // s)  # tile origin in this stage's token units
        for blk, shift in zip(cast(nn.ModuleList, self.stages[stage]), self.stage_shifts[stage]):
            tokens = blk(
                tokens, H, W, stage, films, self.conditioner, shift=shift, offset=o
            )
        return tokens.permute(0, 2, 1).reshape(Bx, C, H, W)

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
        """Predict residual velocity v-hat, (B, 128, H, W) (plan §5.4).

        r_t / z_lr: (B, 128, H, W) with H, W >= 8 and 4-divisible (the stage
        strides 1/2/4 must divide the input grid); t / sigma: (B,); pixel
        features may be None (zero conditioning, e.g. smoke tests without the
        pixel encoder). H != W (non-square buckets) is supported.
        """
        B, C, H, W = r_t.shape
        if C != self.latent_ch:
            raise ValueError(f"r_t must have {self.latent_ch} channels, got {C}")
        if z_lr.shape != r_t.shape:
            raise ValueError(f"z_lr shape {tuple(z_lr.shape)} must equal r_t shape {r_t.shape}")
        if H < 8 or W < 8:
            raise ValueError(f"input latent grid {H}x{W} must be at least 8x8")
        if H % 4 or W % 4:
            raise ValueError(
                f"input latent grid {H}x{W} must be 4-divisible (stage strides 1/2/4)"
            )
        device, dtype = r_t.device, r_t.dtype

        def zeros(ch: int, h: int, w: int) -> torch.Tensor:
            return torch.zeros((B, ch, h, w), device=device, dtype=dtype)

        p64 = p64 if p64 is not None else zeros(128, H, W)
        p32 = p32 if p32 is not None else zeros(192, H // 2, W // 2)
        p16 = p16 if p16 is not None else zeros(256, H // 4, W // 4)
        gap16 = gap16 if gap16 is not None else torch.zeros((B, 256), device=device, dtype=dtype)
        if p64.shape != (B, 128, H, W):
            raise ValueError(f"p64 shape {tuple(p64.shape)} must be ({B}, 128, {H}, {W})")
        if p32.shape != (B, 192, H // 2, W // 2):
            raise ValueError(f"p32 shape {tuple(p32.shape)} must be ({B}, 192, {H // 2}, {W // 2})")
        if p16.shape != (B, 256, H // 4, W // 4):
            raise ValueError(f"p16 shape {tuple(p16.shape)} must be ({B}, 256, {H // 4}, {W // 4})")

        films = self.conditioner(t, sigma, gap16)

        # ---- input fusion (§7.5) ----
        h_in = (
            self.proj_r(r_t)
            + self.proj_z(z_lr)
            + torch.sigmoid(self.gate_64) * self.proj_p64(p64)
        )
        h_in = self._run_stage(h_in, "enc64", films, offset)

        # ---- encoder path ----
        h32 = self.down[0](h_in)
        h32 = h32 + torch.sigmoid(self.gate_32) * self.proj_p32(p32)
        h32 = self._run_stage(h32, "enc32", films, offset)
        skip_32 = h32

        h16 = self.down[1](h32)
        h16 = h16 + torch.sigmoid(self.gate_16) * self.proj_p16(p16)
        h16 = self._run_stage(h16, "bottleneck16", films, offset)

        # ---- decoder path (U-shaped skips, §7.4) ----
        h = self.skip[0](torch.cat([self.up[0](h16), skip_32], dim=1))
        h = self._run_stage(h, "dec32", films, offset)

        h = self.skip[1](torch.cat([self.up[1](h), h_in], dim=1))
        h = self._run_stage(h, "dec64", films, offset)

        return self.head(h)  # (B, 128, H, W)


class AnimeSRModel(nn.Module):
    """Full model (plan §8): PixelConditionEncoder + U-Flow trunk.

    The frozen Mage-VAE is NOT part of this module (see anime_sr.vae).
    Forward: residual path -> velocity; the caller applies the flow sampler
    (anime_sr.flow) to get z-hr = z_lr + delta-hat.
    """

    def __init__(
        self,
        model_spec: ModelSpec | None = None,
        zero_init_pixel: bool = False,
    ) -> None:
        super().__init__()
        spec = model_spec if model_spec is not None else ModelSpec()
        spec.validate_structure()
        self.model_spec = spec
        self.pixel_encoder = PixelConditionEncoder(in_channels=spec.pixel_encoder.in_channels)
        self.trunk = UFlowSR(spec.uflow, spec.output_head)
        if zero_init_pixel:
            apply_pixel_zero_init(self.trunk)

    def forward(
        self,
        r_t: torch.Tensor,
        z_lr: torch.Tensor,
        lq_rgb: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        offset: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """r_t/z_lr: (B,128,H,W); lq_rgb: (B,3,4H,4W) -> v-hat (B,128,H,W)."""
        pc: PixelConditionOutputs = self.pixel_encoder(lq_rgb)
        p64, p32, p16, gap16 = pc.v1_subset()
        return self.trunk(r_t, z_lr, t, sigma, p64, p32, p16, gap16, offset=offset)

    @property
    def num_params(self) -> int:
        return count_parameters(self)
