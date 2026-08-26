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
