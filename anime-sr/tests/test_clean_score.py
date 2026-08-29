"""§10.5 clean score: deterministic heuristics + frozen read-only sidecar
(plan §10.5; P1-4 2026-08-29 made the sidecar read-only at training time —
scores are precomputed offline by cli/clean_score_precompute). The P1-4
blockiness index fix, horizontal-edge coverage, gate and report are in
test_clean_score_gate.py."""

import json

import torch
import torch.nn.functional as F
from anime_sr.data.clean_score import (
    CLEAN_SCORE_CACHE_NAME,
    CleanScoreCache,
    compute_clean_score,
)


def _base(n: int = 64) -> torch.Tensor:
    """Smooth multi-frequency field in [-1, 1] — the 'clean' reference."""
    x = torch.linspace(0.0, 1.0, n)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    img = (
        torch.sin(xx * 2 * 3.14159265 * 1.0) * 0.5
        + torch.sin(yy * 2 * 3.14159265 * 3.0 + xx * 5.0) * 0.3
        + torch.sin((xx + yy) * 2 * 3.14159265 * 7.0) * 0.2
    )
    img = (img + 1.0) * 0.5  # [0,1]
    img = img * 2.0 - 1.0  # [-1,1]
    return img.unsqueeze(0).repeat(3, 1, 1).contiguous()


def _blockify(img: torch.Tensor, block: int = 8, amp: float = 0.3) -> torch.Tensor:
    h, w = img.shape[-2:]
    rows = (torch.arange(h) // block) % 2
    cols = (torch.arange(w) // block) % 2
    pattern = (rows[:, None].float() + 0.5 * cols[None, :].float() - 0.75) * amp
    out = img.clone()
    out += pattern[None].expand_as(out)
    return out.clamp(-1.0, 1.0)


def _upscaled(img: torch.Tensor) -> torch.Tensor:
    """Simulate an upscaled (soft) image: 2x down then 2x bilinear up."""
    h, w = img.shape[-2:]
    out = []
    for c in range(img.shape[0]):
        d = F.interpolate(
            img[c].unsqueeze(0).unsqueeze(0), size=(h // 2, w // 2), mode="bilinear"
        ).squeeze()
        u = F.interpolate(
            d.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear"
        ).squeeze()
        out.append(u)
    return torch.stack(out).clamp(-1.0, 1.0)


def test_clean_score_bounded_and_deterministic() -> None:
    base = _base()
    a = compute_clean_score(base)
    b = compute_clean_score(base)
    assert a == b  # exact determinism (no RNG in the path)
    assert 0.0 <= a <= 1.0


def test_clean_score_discriminates_degradation_modes() -> None:
    base = _base()
    s_clean = compute_clean_score(base)
    s_blocky = compute_clean_score(_blockify(base))
    s_up = compute_clean_score(_upscaled(base))
    s_noisy = compute_clean_score(base + torch.randn_like(base) * 0.06)
    # every degradation mode must score below the clean reference
    assert s_blocky < s_clean, (s_blocky, s_clean)
    assert s_up < s_clean, (s_up, s_clean)
    assert s_noisy < s_clean, (s_noisy, s_clean)


def test_clean_score_rejects_bad_shape() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_clean_score(torch.randn(4, 8, 8))


def test_clean_score_cache_read_only(tmp_path) -> None:
    """P1-4: the training-time cache is READ-ONLY — it never computes or
    appends (the offline precompute CLI is the single writer). Lookups hit
    the sidecar, miss with None, and leave the file byte-identical."""
    sidecar = tmp_path / CLEAN_SCORE_CACHE_NAME
    s1 = compute_clean_score(_base())
    sidecar.write_text(
        json.dumps({"sample_id": "s1", "score": round(s1, 6)}) + "\n",
        encoding="utf-8",
    )
    before = sidecar.read_bytes()

    cache = CleanScoreCache(tmp_path)
    assert cache.get("s1") == round(s1, 6)  # hit
    assert cache.get("missing") is None  # miss -> None (not a compute+append)
    assert len(cache) == 1

    # a fresh instance (a "new process") reads the same frozen rows
    cache2 = CleanScoreCache(tmp_path)
    assert cache2.get("s1") == round(s1, 6)
    assert len(cache2) == 1
    assert sidecar.read_bytes() == before, "read-only sidecar was modified"
