"""Bucket alignment + deterministic crop boxes (data-contract §1)."""

from __future__ import annotations

import pytest
from anime_sr.config.schema import Config
from anime_sr.data.buckets import Bucket, assert_aligned, check_buckets, crop_box

CFG = Config()


def test_check_buckets_default_table() -> None:
    bs = check_buckets(CFG)
    assert bs == [Bucket(128, 512), Bucket(192, 768), Bucket(256, 1024)]
    assert [b.latent for b in bs] == [32, 48, 64]
    for b in bs:
        assert b.hr // 4 == b.lq


def test_crop_box_deterministic_and_in_range() -> None:
    for seed in (0, 1, 12345, 2**63):
        x, y = crop_box(2048, 1536, 1024, seed)
        assert 0 <= x <= 1024 and 0 <= y <= 512
        assert (x, y) == crop_box(2048, 1536, 1024, seed)


def test_crop_box_exact_image() -> None:
    assert crop_box(1024, 1024, 1024, 7) == (0, 0)


def test_crop_box_too_small_raises() -> None:
    with pytest.raises(ValueError):
        crop_box(800, 800, 1024, 1)


def test_crop_box_spreads_offsets() -> None:
    pts = {crop_box(4096, 4096, 1024, i) for i in range(64)}
    assert len(pts) > 8  # no degenerate single-offset behavior


def test_assert_aligned() -> None:
    assert_aligned(1024, 1024, 256, 256)
    with pytest.raises(ValueError):
        assert_aligned(1000, 1000, 256, 256)  # HR not a 64-multiple
    with pytest.raises(ValueError):
        assert_aligned(1024, 1024, 250, 250)  # LQ not a 16-multiple
