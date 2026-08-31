"""Tar-direct streaming path (2026-09-01): single-pass shard scan with
member coordinates, in-tar decode bit-exactness, and the window oracle.

Contract under test:
  * scan_shard_full pairs each json meta with its webp member in ONE pass
    and records (offset, size) + the REAL webp-header pixel size (the
    danbooru meta dimensions are the original post, not the shipped webp);
  * SRDataset(tar_dir=...) decodes the member at (offset, size) and yields
    tensors BIT-EXACT with the extracted-webp path (identical bytes);
  * window_shards is the demand oracle of the streaming data plane: the
    shards a step interval touches across all DDP ranks, matching the
    trainer's §11.5 slot arithmetic exactly.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from anime_sr.config.schema import Config
from anime_sr.data.index import (
    build_index,
    build_index_from_records,
    scan_shard,
    scan_shard_full,
)
from anime_sr.data.pipeline import SRDataset
from anime_sr.data.pool_sampler import SlotMap
from anime_sr.data.stream import latent_sample_index, window_shards
from PIL import Image

CFG = Config()

# Meta dimensions are DOUBLE the shipped webp (the danbooru-v2 trap: meta =
# original post size, shipped webp = resized). The SHIPPED sizes (meta//2)
# must stay train-eligible for bucket 512 — hence meta >= 1024 on both
# sides (700x600 / 900x800 shipped, the codec_bank fixture's eligible set).
_CASES = [
    (2001, 1400, 1200, "sfw"),
    (2002, 1800, 1600, "sfw"),
    (2003, 1400, 1200, "sfw"),
    (2004, 1800, 1600, "sfw"),
    (2005, 1400, 1200, "sfw"),
]


def _make_shard(tmp_path: Path, flat: str = "shard-1_2024-000000.tar") -> Path:
    """A shard tar with real (decodable) webp members + paired json meta.

    Meta dimensions deliberately DIFFER from the webp dimensions (the
    danbooru-v2 trap: meta = original post size, shipped webp = resized).
    """
    buf = io.BytesIO()
    webp_bytes: dict[str, bytes] = {}
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for cid, meta_w, meta_h, nsfw in _CASES:
            stem = f"danbooru/5.9/1_2024/{cid}"
            # shipped webp: 1/2 of the meta size (the "resized variant")
            img_bytes = io.BytesIO()
            rng = np.random.default_rng(cid)
            Image.fromarray(
                rng.integers(0, 255, (meta_h // 2, meta_w // 2, 3), dtype=np.uint8)
            ).save(img_bytes, format="WEBP")
            img_bytes = img_bytes.getvalue()
            webp_bytes[stem] = img_bytes
            meta = json.dumps(
                {
                    "schema_version": 1,
                    "id": cid,
                    "source": {
                        "dataset": "danbooru",
                        "release": "1_2024",
                        "original_path": f"1_2024/{cid}.webp",
                    },
                    "image": {"format": "webp", "width": meta_w, "height": meta_h},
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
    p = tmp_path / flat
    p.write_bytes(buf.getvalue())
    return p


def _extract_webp(shard: Path, webp_dir: Path) -> None:
    with tarfile.open(shard) as tf:
        for m in tf.getmembers():
            if m.name.endswith(".webp"):
                img_id = m.name.rsplit("/", 1)[-1]
                d = webp_dir / shard.name
                d.mkdir(parents=True, exist_ok=True)
                fobj = tf.extractfile(m)
                if fobj is not None:
                    (d / img_id).write_bytes(fobj.read())


# ---------------------------------------------------------------------------
# scan_shard_full: coordinates + real sizes, one pass
# ---------------------------------------------------------------------------
def test_scan_shard_full_records_coordinates(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    records, summary = scan_shard_full(shard, "shard-1_2024-000000.tar")
    assert summary.shard == "shard-1_2024-000000.tar"
    assert len(records) == len(_CASES)
    assert summary.n_webp_missing == 0
    assert summary.n_webp_bad_header == 0
    assert summary.n_unresolved == 0

    with tarfile.open(shard) as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        bodies = {
            name: tf.extractfile(m).read()
            for name, m in members.items()
            if m.isfile() and name.endswith(".webp")
        }
    for rec, (cid, meta_w, meta_h, _nsfw) in zip(records, _CASES):
        stem = f"danbooru/5.9/1_2024/{cid}"
        member = members[f"{stem}.webp"]
        # read coordinates point at the member body
        assert rec.webp_offset == member.offset_data
        assert rec.webp_size == member.size
        with open(shard, "rb") as fh:
            fh.seek(rec.webp_offset)
            body = fh.read(rec.webp_size)
        assert body == bodies[f"{stem}.webp"]
        # real pixel size = the SHIPPED webp (meta/2), NOT the meta dims
        img = Image.open(io.BytesIO(body))
        assert (img.width, img.height) == (meta_w // 2, meta_h // 2)
        assert (rec.width, rec.height) == (img.width, img.height)
        assert (rec.width, rec.height) != (meta_w, meta_h)
        assert rec.shard == "shard-1_2024-000000.tar"


def test_scan_shard_full_counts_missing_webp(tmp_path: Path) -> None:
    """A meta without its webp member is counted, not silently dropped."""
    shard = _make_shard(tmp_path)
    buf = io.BytesIO()
    with tarfile.open(shard) as src, tarfile.open(fileobj=buf, mode="w") as tf:
        for m in src.getmembers():
            # drop ONLY the 2005 webp member; its json meta stays
            if m.name != "danbooru/5.9/1_2024/2005.webp":
                tf.addfile(m, src.extractfile(m) if m.isfile() else None)
    p = tmp_path / "shard-1_2024-000001.tar"
    p.write_bytes(buf.getvalue())
    records, summary = scan_shard_full(p, "shard-1_2024-000001.tar")
    assert summary.n_webp_missing == 1
    # the record survives (coords 0) so the coverage gate can fail loudly
    assert any(r.webp_offset == 0 for r in records)


def test_scan_shard_full_matches_scan_shard_meta(tmp_path: Path) -> None:
    """Same meta coverage as the legacy scan (only sizes/coords differ)."""
    shard = _make_shard(tmp_path)
    legacy, _ = scan_shard(shard)
    full, _ = scan_shard_full(shard, "shard-1_2024-000000.tar")
    assert {r.sample_id for r in legacy} == {r.sample_id for r in full}


# ---------------------------------------------------------------------------
# build_index over scanned records: rows carry coordinates
# ---------------------------------------------------------------------------
def test_build_index_from_records_keeps_coordinates(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    records, summary = scan_shard_full(shard, "shard-1_2024-000000.tar")
    out = tmp_path / "index"
    report = build_index_from_records(records, [summary], CFG, out)
    assert report["n_webp_missing"] == 0
    assert report["n_webp_bad_header"] == 0

    # rows expose the coordinates (the SRDataset tar-mode gate reads them)
    path = out / "sr-eligibility-v1.jsonl"
    if not path.is_file():
        path = out / "sr-eligibility-v1.parquet"
    assert path.is_file()
    from anime_sr.data.index import iter_index

    rows = list(iter_index(path))
    assert rows, "no eligibility rows written"
    for row in rows:
        assert int(row["webp_offset"]) > 0
        assert int(row["webp_size"]) > 0


# ---------------------------------------------------------------------------
# SRDataset tar mode: bit-exact decode vs the extracted-webp path
# ---------------------------------------------------------------------------
def test_tar_decode_bitexact_with_file_decode(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    index_dir = tmp_path / "index"
    build_index(
        [str(shard)],
        CFG,
        index_dir,
        scan=lambda sp: scan_shard_full(sp, "shard-1_2024-000000.tar"),
    )
    webp_dir = tmp_path / "webp"
    _extract_webp(shard, webp_dir)

    ds_file = SRDataset(index_dir, webp_dir, CFG, bucket_hr=512, split="train")
    ds_tar = SRDataset(
        index_dir, webp_dir, CFG, bucket_hr=512, split="train",
        tar_dir=tmp_path / "pinned",
    )
    # the pinned tar sits where the index ``shard`` column says
    (tmp_path / "pinned").mkdir()
    (tmp_path / "pinned" / "shard-1_2024-000000.tar").write_bytes(shard.read_bytes())

    assert len(ds_tar) == len(ds_file)
    for a, b in zip(ds_file.samples, ds_tar.samples):
        assert (a.sample_id, a.width, a.height) == (b.sample_id, b.width, b.height)
        assert b.offset > 0 and b.size > 0
        ta, st_a = ds_file.decode_hr_timed(a)
        tb, st_b = ds_tar.decode_hr_timed(b)
        assert torch.equal(ta, tb), f"tar decode not bit-exact for {a.sample_id}"
        assert set(st_a) == {"shard", "decode"} and set(st_b) == {"shard", "decode"}


def test_tar_mode_rejects_index_without_coordinates(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    index_dir = tmp_path / "index"
    build_index([str(shard)], CFG, index_dir)  # legacy scan: no coordinates
    webp_dir = tmp_path / "webp"
    _extract_webp(shard, webp_dir)
    with pytest.raises(RuntimeError, match="tar-direct coordinates"):
        SRDataset(
            index_dir, webp_dir, CFG, bucket_hr=512, split="train",
            tar_dir=tmp_path / "pinned",
        )


def test_tar_mode_missing_pin_fails_loud(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    index_dir = tmp_path / "index"
    build_index(
        [str(shard)],
        CFG,
        index_dir,
        scan=lambda sp: scan_shard_full(sp, "shard-1_2024-000000.tar"),
    )
    webp_dir = tmp_path / "webp"
    _extract_webp(shard, webp_dir)
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    # NOTE: no tar inside — the window driver never pinned it
    ds = SRDataset(
        index_dir, webp_dir, CFG, bucket_hr=512, split="train",
        tar_dir=pinned,
    )
    with pytest.raises(FileNotFoundError, match="window driver did not pin"):
        ds.decode_hr(ds.samples[0])


# ---------------------------------------------------------------------------
# window oracle: the demand set of a step interval (all ranks)
# ---------------------------------------------------------------------------
def test_window_shards_matches_bruteforce() -> None:
    n = 7
    shards = [f"S{i % 3}" for i in range(n)]  # 3 distinct shards, 7 samples
    cfg = Config()
    cfg.sampling.enabled = False  # legacy order: slot_map[slot] = slot % n
    slot_map = SlotMap(n, None, cfg, list(range(n)), salt="42")
    bs, world = 3, 2
    start, end = 0, 11

    got = window_shards(
        start, end, bs=bs, world=world, n=n, slot_map=slot_map, shards=shards
    )
    want: list[str] = []
    for s in range(start, end):
        for r in range(world):
            for i in range(bs):
                j = slot_map[latent_sample_index(s, r, i, bs, world, n)]
                if shards[j] not in want:
                    want.append(shards[j])
    assert got == want
    assert len(got) == len(set(got))  # deduplicated
    assert set(got) == set(shards)  # 11 steps x 6 samples covers all


def test_window_shards_pool_stream() -> None:
    """Same oracle with the P1 pool sampler ENABLED (permutation order)."""
    n = 12
    shards = [f"S{i % 4}" for i in range(n)]
    members = {
        "priority": [i for i in range(n) if i % 3 == 0],
        "regular": [i for i in range(n) if i % 3 == 1],
        "aux": [i for i in range(n) if i % 3 == 2],
    }
    cfg = Config()
    cfg.sampling.enabled = True
    slot_map = SlotMap(n, members, cfg, list(range(n)), salt="42")
    bs, world = 2, 2
    got = window_shards(
        3, 9, bs=bs, world=world, n=n, slot_map=slot_map, shards=shards
    )
    want: list[str] = []
    for s in range(3, 9):
        for r in range(world):
            for i in range(bs):
                j = slot_map[latent_sample_index(s, r, i, bs, world, n)]
                if shards[j] not in want:
                    want.append(shards[j])
    assert got == want


def test_window_shards_disjoint_steps_are_covered() -> None:
    """Every slot in the interval maps to a shard in the window (no gaps)."""
    n = 50
    shards = [f"S{i % 5}" for i in range(n)]
    cfg = Config()
    cfg.sampling.enabled = True
    members = {
        "priority": list(range(0, n, 3)),
        "regular": list(range(1, n, 3)),
        "aux": list(range(2, n, 3)),
    }
    slot_map = SlotMap(n, members, cfg, list(range(n)), salt="7")
    bs, world = 8, 2
    win = set(
        window_shards(100, 103, bs=bs, world=world, n=n, slot_map=slot_map, shards=shards)
    )
    for s in range(100, 103):
        for r in range(world):
            for i in range(bs):
                j = slot_map[latent_sample_index(s, r, i, bs, world, n)]
                assert shards[j] in win
