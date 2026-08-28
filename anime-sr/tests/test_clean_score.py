"""§10.5 clean score: lazy, cached, deterministic (plan §10.5, P1 ③)."""

import torch
import torch.nn.functional as F
from anime_sr.data.clean_score import CleanScoreCache, compute_clean_score


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


def test_clean_score_cache_roundtrip(tmp_path) -> None:
    cache = CleanScoreCache(tmp_path)
    s1 = cache.get("s1", _base())
    assert len(cache) == 1
    lines = (tmp_path / "clean-score-v1.jsonl").read_text().splitlines()
    assert len(lines) == 1

    # second read of the same id: served from the read-through dict, no append
    s2 = cache.get("s1", torch.zeros(3, 4, 4))
    assert s2 == s1
    lines = (tmp_path / "clean-score-v1.jsonl").read_text().splitlines()
    assert len(lines) == 1  # unchanged

    # a fresh process (new instance) loads the cache and hits, not recomputes
    cache2 = CleanScoreCache(tmp_path)
    assert cache2.get("s1", torch.zeros(3, 4, 4)) == s1
    assert len(cache2) == 1

    # a new id is computed and appended exactly once
    s3 = cache2.get("s2", _base())
    assert len(cache2) == 2
    lines = (tmp_path / "clean-score-v1.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert s3 == s1  # same pixels -> same score
