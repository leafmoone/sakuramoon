"""Deterministic anime degradation chain (plan §11, docs/degradation-v1.md).

One training exposure = one profile-determined chain of operators applied to
the HR crop, producing the LQ (exactly HR/4). Every parameter is derived
from the exposure seed (plan §11.5:: ``H(global_seed, sample_id,
data_cycle, exposure_index)``) via a single CPU ``torch.Generator`` in a
fixed draw order, so a resume reproduces every pixel.

Contract (data-contract §1 / degradation-v1 §4):
- fp32 throughout, output clamped to [-1, 1], no NaN/Inf;
- operators are torch-native (GPU when the input is on GPU);
- ``apply_degradation`` returns the LQ plus a ``DegradationParams`` record
  for per-batch telemetry (profile histogram + parameter means).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from anime_sr.config.schema import Config

# profile -> parameter ranges. Ranges absent here keep the mild defaults.
_PROFILE_BLUR: dict[str, str | None] = {
    "P0_clean": "none",
    "P1_mild_web": "mild",
    "P2_normal_web": "normal",
    "P3_anime_codec": "mild",
    "P4_severe": "severe",
}
_PROFILE_NOISE: dict[str, str] = {
    "P0_clean": "none",
    "P1_mild_web": "mild",
    "P2_normal_web": "normal",
    "P3_anime_codec": "mild",
    "P4_severe": "severe",
}
_PROFILE_JPEG: dict[str, str] = {
    "P0_clean": "none",
    "P1_mild_web": "mild",
    "P2_normal_web": "normal",
    "P3_anime_codec": "severe",  # codec bank territory: heavy blockiness
    "P4_severe": "severe",
}


def exposure_seed(global_seed: int, sample_id: str, data_cycle: int, exposure_index: int) -> int:
    """plan §11.5: H(global_seed, sample_id, data_cycle, exposure_index).

    blake2b-8 → 64-bit int; platform-independent (stdlib only).
    """
    msg = f"{global_seed}|{sample_id}|{data_cycle}|{exposure_index}".encode()
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class DegradationParams:
    """All sampled parameters of one exposure (telemetry record, §11.4)."""

    profile: str
    blur_kind: str  # none|iso|aniso|motion|sinc
    blur_sigma: float  # HR pixels
    blur_angle_deg: float  # aniso/motion
    motion_len: float  # motion blur length (px)
    downsample_filter: str  # area|bicubic|bilinear|nearest
    anti_alias: bool
    noise_kind: str  # none|gaussian|poisson|chroma
    noise_sigma: float  # 0-255 units (noise-v1 convention)
    jpeg_quality: float  # 0-100, 0 = off (P0)
    block_quantize: bool  # 8x8 blocky luma (codec approximation)
    chroma_subsample: bool
    posterize_levels: int  # 0 = off
    banding_levels: int  # 0 = off
    dither: bool
    gamma: float  # 1.0 = off
    unsharp_amount: float  # 0 = off

    def to_dict(self) -> dict:
        return asdict(self)


def _range(table: dict[str, list[float]], level: str) -> tuple[float, float]:
    lo, hi = table[level]
    return float(lo), float(hi)


def _u(rng: torch.Generator, lo: float, hi: float) -> float:
    if hi <= lo:
        return float(lo)
    return float(torch.rand((), generator=rng, dtype=torch.float32).item() * (hi - lo) + lo)


def _pick(rng: torch.Generator, options: list[str]) -> str:
    i = int(torch.randint(len(options), (1,), generator=rng).item())
    return options[i]


def sample_params(seed: int, cfg: Config) -> DegradationParams:
    """Deterministic parameter draw for one exposure (fixed draw order)."""
    rng = torch.Generator().manual_seed(seed % (2**32))
    # 1) profile (config weights, normalized)
    profiles = list(cfg.degradation.profiles.keys())
    weights = torch.tensor([cfg.degradation.profiles[p] for p in profiles], dtype=torch.float32)
    w = weights / weights.sum()
    profile = profiles[int(torch.multinomial(w, 1, generator=rng).item())]

    blur_level = _PROFILE_BLUR[profile]
    noise_level = _PROFILE_NOISE[profile]
    jpeg_level = _PROFILE_JPEG[profile]

    # 2) blur kind: per-profile availability (P0 none; P3/P4 add sinc)
    if blur_level in (None, "none"):
        blur_kind = "none"
        blur_sigma = 0.0
        angle = 0.0
        motion = 0.0
    else:
        kinds = ["none", "iso", "aniso", "motion", "sinc"]
        blur_kind = _pick(rng, kinds) if profile not in ("P1_mild_web",) else _pick(rng, ["none", "iso", "aniso"])
        lo, hi = _range(cfg.degradation.blur_sigma, blur_level)
        blur_sigma = _u(rng, lo, hi)
        angle = _u(rng, 0.0, 180.0)
        motion = _u(rng, 5.0, 12.0)

    # 3) downsample
    downsample_filter = _pick(rng, ["area", "bicubic", "bilinear", "nearest"])
    anti_alias = bool(torch.randint(0, 2, (1,), generator=rng).item() == 1)

    # 4) noise
    if noise_level == "none":
        noise_kind, noise_sigma = "none", 0.0
    else:
        noise_kind = _pick(rng, ["gaussian", "poisson", "chroma"])
        lo, hi = _range(cfg.degradation.gaussian_noise, noise_level)
        noise_sigma = _u(rng, lo, hi)

    # 5) jpeg
    if jpeg_level == "none":
        jpeg_quality, block_quantize, chroma_sub = 0.0, False, False
    else:
        lo, hi = _range(cfg.degradation.jpeg_quality, jpeg_level)
        jpeg_quality = _u(rng, lo, hi)
        block_quantize = profile in ("P3_anime_codec", "P4_severe")
        chroma_sub = profile in ("P3_anime_codec", "P4_severe") or bool(
            torch.randint(0, 2, (1,), generator=rng).item() == 1
        )

    # 6) color/depth ops (P4 only, degradation-v1 §2)
    if profile == "P4_severe":
        posterize_levels = int(torch.randint(0, 3, (1,), generator=rng).item()) * 16  # 0/16/32
        banding_levels = int(torch.randint(0, 3, (1,), generator=rng).item()) * 12
        dither = bool(torch.randint(0, 2, (1,), generator=rng).item() == 1)
        gamma = _u(rng, 0.92, 1.08)
        unsharp_amount = _u(rng, 0.0, 0.3)
    else:
        posterize_levels = 0
        banding_levels = 0
        dither = False
        gamma = 1.0
        unsharp_amount = 0.0

    return DegradationParams(
        profile=profile,
        blur_kind=blur_kind,
        blur_sigma=round(blur_sigma, 4),
        blur_angle_deg=round(angle, 2),
        motion_len=round(motion, 2),
        downsample_filter=downsample_filter,
        anti_alias=anti_alias,
        noise_kind=noise_kind,
        noise_sigma=round(noise_sigma, 4),
        jpeg_quality=round(jpeg_quality, 2),
        block_quantize=block_quantize,
        chroma_subsample=chroma_sub,
        posterize_levels=posterize_levels,
        banding_levels=banding_levels,
        dither=dither,
        gamma=round(gamma, 4),
        unsharp_amount=round(unsharp_amount, 4),
    )


# ---------------------------------------------------------------------------
# kernels (pure python → torch; deterministic)
# ---------------------------------------------------------------------------
def _gaussian_kernel_1d(sigma: float, length: int | None = None) -> torch.Tensor:
    if length is None:
        length = 2 * max(1, math.ceil(3.0 * sigma)) + 1
    length |= 1  # force odd
    half = length // 2
    x = torch.arange(length, dtype=torch.float32) - half
    k = torch.exp(-(x.square() / (2.0 * max(sigma, 1e-3) ** 2)))
    return (k / k.sum()).view(1, 1, length, 1)


def _blur_kernel(p: DegradationParams) -> torch.Tensor | None:
    """2D blur kernel [1,1,k,k] (or None)."""
    if p.blur_kind == "none" or p.blur_sigma <= 0:
        return None
    sigma = p.blur_sigma
    if p.blur_kind == "iso":
        k1 = _gaussian_kernel_1d(sigma)
        return k1 @ k1.transpose(-1, -2)
    if p.blur_kind == "aniso":
        # rotated 1D gaussian sampled on a square grid (deterministic)
        k1 = _gaussian_kernel_1d(sigma * 2.0)
        length = k1.size(2)  # k1 is [1, 1, L, 1]
        half = length // 2
        ang = math.radians(p.blur_angle_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
        x = (i - half) * ca + (j - half) * sa
        y = -(i - half) * sa + (j - half) * ca
        r2 = x.square() + y.square()
        k2 = torch.exp(-r2 / (2.0 * max(sigma, 1e-3) ** 2))
        return (k2 / k2.sum()).view(1, 1, length, length)
    if p.blur_kind == "motion":
        # line kernel: projection on the motion direction (Gaussian across)
        length = max(3, round(p.motion_len) | 1)
        half = length // 2
        ang = math.radians(p.blur_angle_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        i, j = torch.meshgrid(torch.arange(length), torch.arange(length), indexing="ij")
        proj = (i - half) * ca + (j - half) * sa
        cross = (j - half) * ca - (i - half) * sa
        width = max(1.0, length * 0.15)
        k2 = torch.exp(-cross.square() / (2.0 * width**2)) * (proj.abs() <= length / 2)
        return (k2 / k2.sum()).view(1, 1, length, length)
    if p.blur_kind == "sinc":
        # truncated-sinc low-pass (ring artifacts); cutoff ≈ 1/(2σ) cycles/px
        f = 1.0 / (2.0 * max(sigma, 0.15))
        length = 2 * max(1, math.ceil(4.0 / f)) * 2 + 1
        half = length // 2
        x = torch.arange(length, dtype=torch.float32) - half
        k1 = torch.where(x.abs() < 1e-6, 1.0 / (2.0 * f), torch.sin(math.pi * x / f) / (math.pi * x))
        k1 = torch.where(k1.abs() > 1e-4, k1, torch.zeros_like(k1))
        k1 = k1 / k1.sum()
        return k1.view(1, 1, length, 1) @ k1.view(1, 1, length, 1).transpose(-1, -2)
    raise ValueError(f"unknown blur kind {p.blur_kind}")


def _conv2d_sym(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Same-boundary per-channel conv (blur / unsharp). ``kernel`` is
    [1,1,k,k]; with groups=C each channel is convolved independently."""
    pad = kernel.size(-1) // 2
    k = kernel.expand(x.size(1), 1, kernel.size(-2), kernel.size(-1))
    return F.conv2d(x, k.to(x.device, x.dtype), padding=pad, groups=x.size(1))


def _noise_like(x: torch.Tensor, kind: str, sigma_255: float, rng: torch.Generator) -> torch.Tensor:
    """Gaussian/Poisson/chroma noise; sigma in 0-255 units → [-1,1] units."""
    if kind == "none" or sigma_255 <= 0:
        return torch.zeros_like(x)
    scale = sigma_255 * (2.0 / 255.0)
    if kind == "gaussian":
        n = torch.randn(x.shape, generator=rng, dtype=torch.float32) * scale
        return n.to(x.device)
    if kind == "poisson":  # shot approximation: var ∝ signal (x >= 0 part)
        mu = (x.clamp_min(0.0) + 0.5).cpu()  # shift out of [-1,0]
        n = torch.randn(x.shape, generator=rng, dtype=torch.float32) * scale * mu.sqrt()
        return n.to(x.device)
    if kind == "chroma":  # noise on Cb/Cr only (luma channel stays clean)
        n = torch.randn(x.shape, generator=rng, dtype=torch.float32) * scale
        n[:, 0] = 0.0  # suppress the luma channel
        return n.to(x.device)
    raise ValueError(f"unknown noise kind {kind}")


def _yuv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
    y = (x * w.view(3, 1, 1)).sum(dim=1, keepdim=True)
    cb = (x[:, 0:1] - y) / 0.701
    cr = (x[:, 2:3] - y) / 0.886
    return y, cb, cr


def _uv_y(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    return torch.cat([cb * 0.701 + y, y, cr * 0.886 + y], dim=1)


def _jpeg_approx(x: torch.Tensor, quality: float, block_quantize: bool, chroma_sub: bool) -> torch.Tensor:
    """Fast online JPEG approximation (degradation-v1 §2): chroma 4:2:0
    subsampling + block luma quantization (full-DCT offline codec bank is
    plan step 6; this is the <5%-of-step-time stand-in)."""
    if quality >= 100.0:
        return x
    y, cb, cr = _yuv(x)
    if chroma_sub:
        for p in (cb, cr):
            avg = F.avg_pool2d(p, 2)
            p.data = F.interpolate(avg, size=x.shape[-2:], mode="nearest")
    if block_quantize:
        # step grows as quality drops: q=100 → 1, q=30 → ~16 (0-255 units)
        step = max(1.0, (100.0 - quality) * 0.22) * (2.0 / 255.0)
        h, w = y.shape[-2:]
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        if h8 and w8:
            core = y[..., :h8, :w8]
            core_q = torch.round(core.view(y.shape[0], 1, h8 // 8, 8, w8 // 8, 8) / step) * step
            y = torch.cat([core_q.view(y.shape[0], 1, h8, w8), y[..., h8:, :].clone()], dim=-2)
            y = torch.cat([y[..., :w8], y[..., w8:].clone()], dim=-1)
    if quality < 60:  # mild smearing on low quality
        y = F.conv2d(y, torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0, padding=1)
    return _uv_y(y, cb, cr)


def _posterize(x: torch.Tensor, levels: int) -> torch.Tensor:
    if levels <= 0:
        return x
    return torch.round(x * levels) / levels


def _gamma(x: torch.Tensor, g: float) -> torch.Tensor:
    if abs(g - 1.0) < 1e-3:
        return x
    # first-order Taylor about 1: x + (g-1) * x * |x|  (stable, deterministic)
    return x + (g - 1.0) * x * x.abs()


def apply_degradation(
    hr: torch.Tensor,
    p: DegradationParams,
    *,
    noise_seed: int,
    dither_seed: int,
) -> tuple[torch.Tensor, DegradationParams]:
    """Apply the exposure chain: HR [3,H,W] fp32 [-1,1] → LQ [3,H/4,W/4].

    H/W must be multiples of 64 (data-contract §1). ``noise_seed`` /
    ``dither_seed`` are derived from the exposure seed (plan §11.5) so the
    stochastic operators reproduce bit-exact on resume.
    """
    if hr.dtype != torch.float32:
        hr = hr.float()
    if hr.dim() != 3 or hr.size(0) != 3:
        raise ValueError(f"apply_degradation expects an unbatched [3, H, W] fp32 tensor, got {tuple(hr.shape)}")
    h, w = hr.shape[-2:]
    if h % 64 or w % 64:
        raise ValueError(f"HR {w}x{h} must be multiples of 64 (data-contract §1)")
    lq_shape = (3, h // 4, w // 4)

    # all torch ops below (interpolate/conv2d/avg_pool2d) are 4D; the public
    # contract stays per-image [3, H, W] → work on a batch dim of 1.
    x = hr.unsqueeze(0)

    # 1) blur at HR resolution
    kernel = _blur_kernel(p)
    if kernel is not None:
        x = _conv2d_sym(x, kernel)

    # 2) downsample exactly 4x (area is low-pass by construction; torch only
    #    honours the antialias flag for bilinear/bicubic)
    mode = p.downsample_filter
    aa = p.anti_alias and mode in ("bilinear", "bicubic")
    lq = F.interpolate(x, size=lq_shape[1:], mode=mode, antialias=aa)

    # 3) noise (LQ resolution, deterministic from the exposure seed)
    if p.noise_kind != "none":
        nrng = torch.Generator().manual_seed(noise_seed)
        lq = lq + _noise_like(lq, p.noise_kind, p.noise_sigma, nrng)

    # 4) jpeg approximation
    if p.jpeg_quality > 0:
        lq = _jpeg_approx(lq, p.jpeg_quality, p.block_quantize, p.chroma_subsample)

    # 5) color/depth ops
    if p.posterize_levels:
        lq = _posterize(lq, p.posterize_levels)
    if p.banding_levels:
        lq = _posterize(lq, p.banding_levels)
        if p.dither:
            d = torch.randint(0, 2, lq.shape, generator=torch.Generator().manual_seed(dither_seed))
            lq = lq + (d - 0.5) * (2.0 / max(p.banding_levels, 2))
    if abs(p.gamma - 1.0) > 1e-3:
        lq = _gamma(lq, p.gamma)
    if p.unsharp_amount > 0:
        k = torch.tensor([[-0.2, -0.4, -0.2], [-0.4, 3.6, -0.4], [-0.2, -0.4, -0.2]], device=lq.device, dtype=torch.float32)
        k = k.view(1, 1, 3, 3) * p.unsharp_amount
        lq = lq + F.conv2d(lq, k.expand(lq.size(1), 1, 3, 3), padding=1, groups=lq.size(1))

    # NaN/Inf guard + clamp (degradation-v1 §4)
    lq = torch.where(torch.isfinite(lq), lq, torch.zeros_like(lq))
    return lq.squeeze(0).clamp(-1.0, 1.0), p


# ---------------------------------------------------------------------------
# batch entry point (dataset side)
# ---------------------------------------------------------------------------
def degrade_hr(
    hr: torch.Tensor,
    cfg: Config,
    *,
    global_seed: int,
    sample_id: str,
    data_cycle: int,
    exposure_index: int,
) -> tuple[torch.Tensor, DegradationParams]:
    """Sample + apply in one call (deterministic exposure, plan §11.5).

    The exposure seed (64-bit blake2b) is folded into two 32-bit stream
    seeds (noise, dither) so ``apply_degradation`` stays a pure function.
    """
    seed = exposure_seed(global_seed, sample_id, data_cycle, exposure_index)
    params = sample_params(seed, cfg)
    noise_seed = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFF
    dither_seed = (seed ^ 0x517CC1B727220A95) & 0xFFFFFFFF
    return apply_degradation(hr, params, noise_seed=noise_seed, dither_seed=dither_seed)
