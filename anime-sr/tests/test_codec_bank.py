"""Codec bank (§11.4): variant sampling, runtime consumer, pipeline wiring.

The ffmpeg encode path is exercised only when an ffmpeg binary is present
(server side); everything else is pure-python and runs everywhere.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from anime_sr.config.schema import Config
from anime_sr.data.codec_bank import (
    ENCODE_PROFILES,
    CodecBank,
    CodecVariant,
    _int_from_seed,
    encode_variant,
    sample_variants,
)
from anime_sr.data.index import build_index
from anime_sr.data.pipeline import SRDataset
from PIL import Image

CFG = Config()


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# variant sampling
# ---------------------------------------------------------------------------
def test_sample_variants_deterministic() -> None:
    a = sample_variants(1234, 2)
    b = sample_variants(1234, 2)
    assert a == b
    c = sample_variants(1235, 2)
    assert a != c


def test_sample_variants_axes_supported() -> None:
    # every sampled variant must only use axes its codec supports
    for seed in (1, 7, 99, 424242):
        for j in range(8):
            v = sample_variants(seed, j + 1)[j]
            _enc, _ext, _q, _qrange, chroma_ok, range_ok = ENCODE_PROFILES[v.codec]
            if v.pix_fmt == "yuv422p":
                assert chroma_ok, f"{v.codec} does not support 4:2:2"
            if v.range_mismatch:
                assert range_ok, f"{v.codec} does not support range mismatch"
            assert v.passes in (1, 2)
            assert 0 <= v.quality


def test_variant_id_stable() -> None:
    v = CodecVariant("h264", "yuv422p", True, 2, 30)
    assert v.variant_id == CodecVariant("h264", "yuv422p", True, 2, 30).variant_id
    assert v.variant_id != CodecVariant("h264", "yuv422p", False, 2, 30).variant_id


def test_int_from_seed_distribution() -> None:
    draws = [_int_from_seed(i, 10_000, "t") for i in range(10_000)]
    hits = sum(1 for d in draws if d < 1_500)
    assert 0.12 <= hits / len(draws) <= 0.18  # ~15% within a wide band


# ---------------------------------------------------------------------------
# runtime consumer
# ---------------------------------------------------------------------------
def _fake_bank(root: Path, sids: dict[str, list[tuple[str, int, int, int]]]) -> Path:
    """root + index-v1.json + variants/<vid>.bin (w,h,bytes from the tuple)."""
    (root / "variants").mkdir(parents=True, exist_ok=True)
    samples: dict[str, list[dict]] = {}
    for sid, entries in sids.items():
        samples[sid] = []
        for vid, w, h, val in entries:
            p = root / "variants" / f"{vid}.bin"
            p.write_bytes(bytes([val]) * (w * h * 3))
            samples[sid].append(
                {
                    "variant_id": vid,
                    "codec": "h264",
                    "pix_fmt": "yuv420p",
                    "range_mismatch": False,
                    "passes": 1,
                    "quality": 30,
                    "lq_w": w,
                    "lq_h": h,
                    "bytes": w * h * 3,
                }
            )
    (root / "index-v1.json").write_text(json.dumps({"version": 1, "samples": samples}))
    return root


def test_codec_bank_lookup(tmp_path: Path) -> None:
    _fake_bank(tmp_path, {"a": [("v1", 32, 32, 100)], "b": [("v2", 16, 16, 50)]})
    bank = CodecBank(tmp_path)
    assert len(bank) == 2
    arr = bank.variants_for("a", seed=7)[0]
    assert arr.shape == (32, 32, 3) and arr.dtype == np.uint8
    assert (arr == 100).all()
    # deterministic pick: same seed → same array; different seed may differ
    assert all((x == y).all() for x, y in zip(bank.variants_for("a", seed=7), bank.variants_for("a", seed=7)))
    with pytest.raises(KeyError):
        bank.variants_for("zzz", 1)
    # missing index
    with pytest.raises(FileNotFoundError):
        CodecBank(tmp_path / "nope")


def test_codec_bank_size_mismatch_guard(tmp_path: Path) -> None:
    _fake_bank(tmp_path, {"a": [("v1", 16, 16, 100)]})
    (tmp_path / "variants" / "v1.bin").write_bytes(b"x" * 999)  # corrupt
    with pytest.raises(RuntimeError, match="size mismatch"):
        CodecBank(tmp_path).variants_for("a", 1)


def test_bank_fraction_hit_bounds() -> None:
    bank = CodecBank.__new__(CodecBank)
    assert not any(bank.bank_fraction_hit("s", i, 0.0) for i in range(200))
    assert all(bank.bank_fraction_hit("s", i, 1.0) for i in range(200))
    hits = sum(1 for i in range(5000) if bank.bank_fraction_hit(f"s{i}", i, 0.15))
    assert 0.11 <= hits / 5000 <= 0.19


# ---------------------------------------------------------------------------
# encode path (ffmpeg-gated)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_encode_variant_roundtrip(tmp_path: Path) -> None:
    crop = (np.random.default_rng(0).integers(0, 255, (64, 64, 3))).astype(np.uint8)
    for codec in ("webp", "h264", "h265", "av1", "mpeg4", "avif"):
        _enc, _ext, _qflag, (q_lo, q_hi), _c, _r = ENCODE_PROFILES[codec]
        v = CodecVariant(codec, "yuv420p", False, 1, (q_lo + q_hi) // 2)
        raw, n = encode_variant(crop, v, 16, 16, tmp_path / codec)
        assert n == 16 * 16 * 3
        arr = np.fromfile(str(raw), dtype=np.uint8, count=n).reshape(16, 16, 3)
        assert np.isfinite(arr.astype(float)).all()


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg not available")
def test_encode_variant_double_pass(tmp_path: Path) -> None:
    crop = (np.random.default_rng(1).integers(0, 255, (64, 64, 3))).astype(np.uint8)
    v = CodecVariant("h264", "yuv422p", True, 2, 30)
    _, n = encode_variant(crop, v, 16, 16, tmp_path / "p2")
    assert n == 16 * 16 * 3


# ---------------------------------------------------------------------------
# pipeline wiring: bank substitution in SRDataset.fetch
# ---------------------------------------------------------------------------
def _make_real_shard(tmp_path: Path) -> Path:
    """A shard tar whose webp members are *real* (decodable) images."""
    cases = [
        (2001, 700, 600, "sfw"),
        (2002, 900, 800, "sfw"),
        (2003, 700, 600, "sfw"),
        (2004, 900, 800, "sfw"),
        (2005, 700, 600, "sfw"),
        (2006, 900, 800, "sfw"),
        (2007, 700, 600, "sfw"),
        (2008, 900, 800, "sfw"),
        (2009, 700, 600, "sfw"),
        (2010, 900, 800, "sfw"),
    ]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for cid, w, h, nsfw in cases:
            stem = f"danbooru/5.9/1_2024/{cid}"
            img_bytes = io.BytesIO()
            rng = np.random.default_rng(cid)
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(img_bytes, format="WEBP")
            img_bytes = img_bytes.getvalue()
            meta = json.dumps(
                {
                    "schema_version": 1,
                    "id": cid,
                    "source": {"dataset": "danbooru", "release": "1_2024", "original_path": f"1_2024/{cid}.webp"},
                    "image": {"format": "webp", "width": w, "height": h},
                    "nsfw": nsfw,
                    "tags": {"general": ["anime"]},
                }
            )
            t1 = tarfile.TarInfo(f"{stem}.webp")
            t1.size = len(img_bytes)
            tf.addfile(t1, io.BytesIO(img_bytes))
            t2 = tarfile.TarInfo(f"{stem}.json")
            t2.size = len(meta)
            tf.addfile(t2, io.BytesIO(meta.encode()))
    p = tmp_path / "shard-000000.tar"
    p.write_bytes(buf.getvalue())
    return p


def _extract_webp(shard: Path, webp_dir: Path) -> None:
    with tarfile.open(shard) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".webp"):
                img_id = m.name.rsplit("/", 1)[-1]
                d = webp_dir / "shard-000000.tar"
                d.mkdir(parents=True, exist_ok=True)
                fobj = tf.extractfile(m)
                if fobj is not None:
                    (d / img_id).write_bytes(fobj.read())


def test_pipeline_bank_substitution(tmp_path: Path) -> None:
    """frac = 0.20 (frozen-range edge): per-sample hits get the bank LQ,
    misses fall back to the synthetic chain; both are deterministic."""
    import torch
    from anime_sr.data.degradation import exposure_seed

    shard = _make_real_shard(tmp_path)
    index_dir = tmp_path / "index"
    build_index([str(shard)], CFG, index_dir)
    webp_dir = tmp_path / "webp"
    _extract_webp(shard, webp_dir)

    # constant-200 LQ for every case (128x128 @ bucket 512)
    bank = CodecBank(_fake_bank(tmp_path / "bank", {str(cid): [("vb1", 128, 128, 200)] for cid, *_ in _make_cases()}))

    c = Config()
    c.degradation.codec_bank_fraction = 0.20
    ds = SRDataset(index_dir, webp_dir, c, bucket_hr=512, split="train", bank=bank)
    expected = 200.0 * (2.0 / 255.0) - 1.0

    n_hit = 0
    for i, m in enumerate(ds.samples):
        _, lq, _ = ds.fetch(i)
        assert lq.shape == (3, 128, 128) and torch.isfinite(lq).all()
        seed = exposure_seed(ds.global_seed, m.sample_id, 0, 0)
        if bank.bank_fraction_hit(m.sample_id, seed, c.degradation.codec_bank_fraction):
            assert (lq - expected).abs().max() < 1e-6, f"hit sample {m.sample_id} must use the bank LQ"
            n_hit += 1
        else:
            assert (lq - expected).abs().max() > 1e-3, f"miss sample {m.sample_id} must use the synthetic chain"
    assert n_hit > 0, "with 10 train samples at 20% fraction at least one must hit"
    # determinism across fetches
    i0 = next(i for i, m in enumerate(ds.samples) if bank.bank_fraction_hit(m.sample_id, exposure_seed(ds.global_seed, m.sample_id, 0, 0), 0.20))
    _, lq_a, _ = ds.fetch(i0)
    _, lq_b, _ = ds.fetch(i0)
    assert torch.equal(lq_a, lq_b)


def _make_cases() -> list[tuple[int, int, int, str]]:
    return [
        (2001, 700, 600, "sfw"),
        (2002, 900, 800, "sfw"),
        (2003, 700, 600, "sfw"),
        (2004, 900, 800, "sfw"),
        (2005, 700, 600, "sfw"),
        (2006, 900, 800, "sfw"),
        (2007, 700, 600, "sfw"),
        (2008, 900, 800, "sfw"),
        (2009, 700, 600, "sfw"),
        (2010, 900, 800, "sfw"),
    ]


def test_pipeline_fraction_range_guard(tmp_path: Path) -> None:
    """codec_bank_fraction outside the frozen §11.4 range must fail loudly."""
    shard = _make_real_shard(tmp_path)
    index_dir = tmp_path / "index"
    build_index([str(shard)], CFG, index_dir)
    webp_dir = tmp_path / "webp"
    _extract_webp(shard, webp_dir)
    bank = CodecBank(_fake_bank(tmp_path / "bank", {"2001": [("vb1", 128, 128, 200)]}))
    c = Config()
    c.degradation.codec_bank_fraction = 0.5  # outside [0.10, 0.20]
    with pytest.raises(ValueError, match="codec_bank_fraction"):
        SRDataset(index_dir, webp_dir, c, bucket_hr=512, bank=bank)
