"""M2 pixel baseline: architecture band, determinism, and the training loop.

Reuses the real-shard helpers from test_codec_bank (build_index + webp
extraction) so the loop test runs on genuinely decoded webp data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anime_sr.config.schema import Config
from anime_sr.model.pixel_baseline import PixelBaseline
from anime_sr.train.pixel_baseline import run_pixel_baseline
from test_codec_bank import _extract_webp, _make_real_shard


def test_param_count_in_plan_band() -> None:
    n = PixelBaseline.n_params()  # base 160, depth 2
    assert 5_000_000 <= n <= 10_000_000, f"default config must land in the 5M-10M band, got {n}"


def test_forward_shapes_and_grads() -> None:
    m = PixelBaseline(96, 2)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape == (2, 3, 256, 256)
    y.mean().backward()
    assert all(p.grad is not None for p in m.parameters())
    with pytest.raises(ValueError, match="multiple of 16"):
        m(torch.randn(1, 3, 50, 50))
    with pytest.raises(ValueError, match="B, 3, H, W"):
        m(torch.randn(1, 4, 64, 64))


def test_deterministic_forward() -> None:
    m = PixelBaseline(96, 2).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b)


def test_training_loop_smoke(tmp_path: Path) -> None:
    shard = _make_real_shard(tmp_path)
    index_dir = tmp_path / "index"
    from anime_sr.data.index import build_index

    build_index([str(shard)], Config(), index_dir)
    webp_dir = tmp_path / "webp"
    _extract_webp(shard, webp_dir)

    c = Config()
    c.pixel_baseline.iterations = 2
    c.pixel_baseline.batch_size = 2
    c.pixel_baseline.save_every_steps = 1
    c.pixel_baseline.val_every_steps = 1
    c.optimizer.lr = 1e-4
    out = tmp_path / "pb"

    final = run_pixel_baseline(c, index_dir=index_dir, webp_dir=webp_dir, out_dir=out, bucket_hr=512)
    assert final == 2
    ckpt = out / "latest.pt"
    assert ckpt.is_file()
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["step"] == 2

    # resume: continue to 4 steps, same deterministic stream
    c2 = Config()
    c2.pixel_baseline.iterations = 4
    c2.pixel_baseline.batch_size = 2
    c2.pixel_baseline.save_every_steps = 1
    c2.pixel_baseline.val_every_steps = 0
    c2.optimizer.lr = 1e-4
    final2 = run_pixel_baseline(
        c2,
        index_dir=index_dir,
        webp_dir=webp_dir,
        out_dir=out,
        bucket_hr=512,
        start_step=2,
        resume=ckpt,
    )
    assert final2 == 4
