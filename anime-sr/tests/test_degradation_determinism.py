"""Degradation determinism + correctness (degradation-v1 §4, plan §11.5).

Guarantees under test:
  * exposure seed is stable and input-sensitive (resume-safe);
  * same seed + same HR → bit-identical LQ (run-to-run and cross-process);
  * LQ is exactly HR/4, finite, and clamped to [-1, 1];
  * P0 (clean) equals a plain downsample — no hidden damage;
  * every profile runs end-to-end without NaN/Inf.
"""

from __future__ import annotations

import torch
from anime_sr.config.schema import Config
from anime_sr.data.degradation import (
    DegradationParams,
    apply_degradation,
    degrade_hr,
    exposure_seed,
    sample_params,
)

CFG = Config()


def _hr(seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # a smooth-ish HR crop so blur/jpg have structure (64x64 = smallest bucket)
    x = (torch.rand(3, 64, 64, generator=g) * 2.0 - 1.0).clamp(-1.0, 1.0)
    return x.contiguous()


def test_exposure_seed_stable_and_sensitive() -> None:
    a = exposure_seed(1, "img-42", 3, 7)
    b = exposure_seed(1, "img-42", 3, 7)
    c = exposure_seed(1, "img-42", 3, 8)
    assert a == b
    assert a != c
    assert 0 <= a < 2**64


def test_sample_params_deterministic() -> None:
    s = exposure_seed(0, "s", 0, 0)
    p1 = sample_params(s, CFG)
    p2 = sample_params(s, CFG)
    assert p1 == p2
    s2 = exposure_seed(0, "s", 0, 1)
    assert sample_params(s, CFG) != sample_params(s2, CFG)


def test_apply_bit_identical_across_runs() -> None:
    hr = _hr()
    seed = exposure_seed(42, "img", 0, 0)
    p = sample_params(seed, CFG)
    lq_a, _ = apply_degradation(hr, p, noise_seed=(seed ^ 1) & 0xFFFFFFFF, dither_seed=(seed ^ 2) & 0xFFFFFFFF)
    lq_b, _ = apply_degradation(hr, p, noise_seed=(seed ^ 1) & 0xFFFFFFFF, dither_seed=(seed ^ 2) & 0xFFFFFFFF)
    assert torch.equal(lq_a, lq_b)  # bit-exact, not just allclose


def test_lq_shape_range_finite() -> None:
    hr = _hr()
    out, _ = degrade_hr(hr, CFG, global_seed=1, sample_id="x", data_cycle=0, exposure_index=0)
    assert out.shape == (3, 16, 16)
    assert torch.isfinite(out).all()
    assert (out.abs() <= 1.0 + 1e-6).all()


def test_p0_is_clean_downsample() -> None:
    """P0_clean must be a pure 4x downsample (no extra damage)."""
    hr = _hr()
    # force a P0 exposure: find a seed whose sampled profile is P0
    for i in range(200):
        seed = exposure_seed(999, "p0", 0, i)
        p = sample_params(seed, CFG)
        if p.profile == "P0_clean":
            break
    else:
        raise AssertionError("no P0 sample in 200 draws (weights ~10%)")
    assert p.blur_kind == "none" and p.noise_kind == "none" and p.jpeg_quality == 0.0
    out, _ = apply_degradation(hr, p, noise_seed=0, dither_seed=0)
    # reference: same downsample filter, same antialias flag (torch: bilinear/bicubic only)
    import torch.nn.functional as F

    aa = p.anti_alias and p.downsample_filter in ("bilinear", "bicubic")
    ref = F.interpolate(hr.unsqueeze(0), size=(16, 16), mode=p.downsample_filter, antialias=aa).squeeze(0)
    assert torch.allclose(out, ref, atol=1e-6), "P0 must be an undamaged downsample"


def test_all_profiles_run_clean() -> None:
    seen: set[str] = set()
    for i in range(400):
        seed = exposure_seed(1234, f"p{i}", 0, 0)
        p = sample_params(seed, CFG)
        hr = _hr(seed % 50)
        out, _ = apply_degradation(hr, p, noise_seed=seed & 0xFFFFFFFF, dither_seed=(seed ^ 5) & 0xFFFFFFFF)
        assert torch.isfinite(out).all()
        assert out.shape == (3, 16, 16)
        seen.add(p.profile)
    assert seen == {"P0_clean", "P1_mild_web", "P2_normal_web", "P3_anime_codec", "P4_severe"}


def test_different_exposure_differs() -> None:
    hr = _hr()
    o1, _ = degrade_hr(hr, CFG, global_seed=1, sample_id="a", data_cycle=0, exposure_index=0)
    o2, _ = degrade_hr(hr, CFG, global_seed=1, sample_id="a", data_cycle=0, exposure_index=1)
    # not guaranteed for every pair, but 2 different exposures should differ
    # overwhelmingly; use a noise-heavy check via params
    s1 = exposure_seed(1, "a", 0, 0)
    s2 = exposure_seed(1, "a", 0, 1)
    p1, p2 = sample_params(s1, CFG), sample_params(s2, CFG)
    assert (p1.noise_sigma, p1.jpeg_quality, p1.downsample_filter) != (p2.noise_sigma, p2.jpeg_quality, p2.downsample_filter) or not torch.equal(o1, o2)


def _p(profile: str) -> DegradationParams:
    base = sample_params(exposure_seed(5, profile, 0, 0), CFG)
    d = base.to_dict()
    d["profile"] = profile
    return DegradationParams(**d)


def test_forced_profiles_smoke() -> None:
    hr = _hr(21)
    for prof in ("P0_clean", "P1_mild_web", "P2_normal_web", "P3_anime_codec", "P4_severe"):
        p = _p(prof)
        out, _ = apply_degradation(hr, p, noise_seed=99, dither_seed=99)
        assert out.shape == (3, 16, 16) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# P0 color-space regressions: YCbCr inverse + chroma noise (degradation-v1 §2)
# ---------------------------------------------------------------------------
def test_uv_y_round_trip() -> None:
    """RGB -> YCbCr -> RGB must recover the input (exact up to fp32 eps).

    The color helpers operate on the 4D batched layout (1, 3, H, W) used by
    the degradation pipeline (apply_degradation unsqueezes per-image tensors)."""
    from anime_sr.data.degradation import _uv_y, _yuv

    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3)) * 2.0 - 1.0
    y, cb, cr = _yuv(x)
    back = _uv_y(y, cb, cr)
    assert back.shape == x.shape
    assert torch.allclose(back, x, atol=1e-5), (
        f"round-trip max err {(back - x).abs().max().item()}"
    )
    # the middle channel must be the reconstructed green, not the luma
    assert not torch.allclose(back[:, 1], y.squeeze(1))


def test_chroma_noise_preserves_luma() -> None:
    """Chroma noise carries zero luma: adding the returned RGB-space noise
    term leaves the Y component of the image exactly unchanged, while the
    Cb/Cr of (x + n) moves."""
    from anime_sr.data.degradation import _noise_like, _yuv

    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(4)) * 2.0 - 1.0
    rng = torch.Generator().manual_seed(5)
    n = _noise_like(x, "chroma", sigma_255=5.0, rng=rng)
    assert n.shape == x.shape and torch.isfinite(n).all()
    assert n.abs().max().item() > 0.0  # the noise is not empty
    y_in, cb_in, cr_in = _yuv(x)
    y_out, cb_out, cr_out = _yuv(x + n)
    assert torch.allclose(y_out, y_in, atol=1e-6), "luma must stay clean"
    assert (cb_out != cb_in).any() or (cr_out != cr_in).any()  # chroma moved
    # the noise term itself has zero luma
    assert _yuv(n)[0].abs().max().item() < 1e-6
    # deterministic per seed
    n2 = _noise_like(x, "chroma", sigma_255=5.0, rng=torch.Generator().manual_seed(5))
    assert torch.equal(n, n2)


# ---------------------------------------------------------------------------
# P1 ⑤ producer perf: separable blur + reflect boundaries
# ---------------------------------------------------------------------------
def test_separable_blur_matches_2d_outer_product() -> None:
    """The 1D two-pass blur is the outer-product 2D conv (fp32 ULPs only)."""
    from anime_sr.data.degradation import (
        _conv2d_separable,
        _conv2d_sym,
        _gaussian_kernel_1d,
    )

    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(9)) * 2.0 - 1.0
    k1 = _gaussian_kernel_1d(1.2)  # [1, 1, 13, 1]
    k2 = k1 @ k1.transpose(-1, -2)
    ref = _conv2d_sym(x, k2)  # same reflect boundaries, 2D outer-product kernel
    out = _conv2d_separable(x, k1)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5), (
        f"separable vs 2D max err {(out - ref).abs().max().item()}"
    )


def test_reflect_blur_constant_invariant() -> None:
    """Reflect boundaries: blurring a flat field stays flat. (Zero padding
    would dip at the crop edges — the exact artifact P1 ⑤ removes.)"""
    from anime_sr.data.degradation import _conv2d_separable, _gaussian_kernel_1d

    c = 0.37
    x = torch.full((1, 3, 64, 64), c, dtype=torch.float32)
    out = _conv2d_separable(x, _gaussian_kernel_1d(2.0))
    assert (out - c).abs().max().item() < 1e-6, "flat field must stay flat"


def test_separable_kernel_sum_is_one() -> None:
    """Gaussian/sinc 1D kernels are normalized (DC gain 1)."""
    from anime_sr.data.degradation import _gaussian_kernel_1d

    for sigma in (0.1, 0.6, 1.2, 2.0):
        assert abs(_gaussian_kernel_1d(sigma).sum().item() - 1.0) < 1e-6
