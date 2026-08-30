"""Unit tests for tools/analyze_sr_hr_quality_labels.py (pure rules only).

The tool is a read-only stats script; its pool predicates, ordinals, crop
proxy, determinism helpers and the tar-streaming scan are covered here with
synthetic fixtures (no network, no production data).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "analyze_sr_hr_quality_labels.py"
_spec = importlib.util.spec_from_file_location("analyze_sr_hr_quality_labels", TOOL)
assert _spec is not None and _spec.loader is not None
az = importlib.util.module_from_spec(_spec)
sys.modules["analyze_sr_hr_quality_labels"] = az
_spec.loader.exec_module(az)

# the pool predicates / stats helpers now live in the shared library module
from anime_sr.config.schema import Config
from anime_sr.data.index import MetaRecord, is_validation
from anime_sr.data.sr_clean_pools import (
    _crop_bucket_hist,
    _crop_positions,
    _det_sample,
    _pctl,
    _spearman,
    in_p0,
    in_p1,
    in_p2,
    in_p3,
    in_p4,
    quality_ordinal,
)


def _meta(
    sid: str,
    *,
    tier: str = "masterpiece",
    comp: str = "polished",
    cls: str = "illustration",
    corrupted: bool = False,
    tags: tuple[str, ...] = (),
):
    return az.RawMeta(
        sample_id=sid,
        shard="shard-000000.tar",
        width_meta=1024,
        height_meta=1024,
        nsfw="sfw",
        year=2024,
        quality_tier=tier,
        completeness=comp,
        classification=cls,
        ai_corrupted=corrupted,
        tags_general=tags,
    )


# ---------------------------------------------------------------------------
# quality ordinal
# ---------------------------------------------------------------------------
def test_quality_ordinal_full_scale() -> None:
    expected = {"masterpiece": 6, "best": 5, "great": 4, "good": 3, "normal": 2, "low": 1, "worst": 0}
    for k, v in expected.items():
        assert quality_ordinal(k) == v
    assert quality_ordinal("") == -1
    assert quality_ordinal("mystery") == -1


# ---------------------------------------------------------------------------
# pool predicates
# ---------------------------------------------------------------------------
def test_pool_p0_priority() -> None:
    for tier in ("masterpiece", "best", "great"):
        for cls in ("illustration", "bangumi", "comic"):
            m = _meta("1", tier=tier, comp="polished", cls=cls)
            assert in_p0(m), (tier, cls)
            assert in_p1(m) and in_p2(m)
    # tier good/normal/low/worst -> not P0
    for tier in ("good", "normal", "low", "worst", ""):
        m = _meta("2", tier=tier, comp="polished", cls="illustration")
        assert not in_p0(m), tier
    # non-polished or 3d / not_painting -> not P0
    assert not in_p0(_meta("3", comp="rough", cls="illustration"))
    assert not in_p0(_meta("4", comp="monochrome", cls="illustration"))
    assert not in_p0(_meta("5", comp="polished", cls="3d"))
    assert not in_p0(_meta("6", comp="polished", cls="not_painting"))


def test_pool_p1_sr_clean_v1() -> None:
    for tier in ("masterpiece", "best", "great", "good", "normal"):
        m = _meta("a", tier=tier, comp="polished", cls="bangumi")
        assert in_p1(m), tier
    for tier in ("low", "worst", ""):
        assert not in_p1(_meta("b", tier=tier, comp="polished", cls="bangumi")), tier
    assert not in_p1(_meta("c", tier="good", comp="rough", cls="bangumi"))
    assert not in_p1(_meta("d", tier="good", comp="polished", cls="3d"))


def test_pool_p2_wide_any_tier() -> None:
    for tier in ("masterpiece", "good", "low", "worst", ""):
        assert in_p2(_meta("e", tier=tier, comp="polished", cls="comic")), tier
    assert not in_p2(_meta("f", tier="good", comp="monochrome", cls="comic"))


def test_pool_p3_lineart() -> None:
    assert in_p3(_meta("g", comp="monochrome", cls="illustration"))
    assert in_p3(_meta("h", comp="polished", cls="comic"))
    assert not in_p3(_meta("i", comp="rough", cls="illustration"))
    assert not in_p3(_meta("j", comp="monochrome", cls="not_painting"))
    assert not in_p3(_meta("k", comp="polished", cls="comic", corrupted=True))


def test_pool_p4_rough_only() -> None:
    assert in_p4(_meta("l", tier="good", comp="rough", cls="illustration"))
    assert not in_p4(_meta("m", tier="good", comp="polished", cls="illustration"))
    assert not in_p4(_meta("n", tier="good", comp="monochrome", cls="illustration"))


# ---------------------------------------------------------------------------
# crop proxy
# ---------------------------------------------------------------------------
def test_crop_positions_square_and_wide() -> None:
    assert _crop_positions(1024, 1024, 1024) == (1, 1)
    assert _crop_positions(2048, 1024, 1024) == (1025, 17)
    assert _crop_positions(4096, 4096, 1024) == (3073 * 3073, 49 * 49)
    # wide landscape 2961x1024: single vertical strip, 31 stride-64 columns
    assert _crop_positions(2961, 1024, 1024) == (1938, 31)


def test_crop_bucket_hist_thresholds() -> None:
    hist = _crop_bucket_hist([1, 2, 5, 17, 65, 257])
    assert hist["==1"] == 1
    assert hist[">1"] == 5
    assert hist[">4"] == 4
    assert hist[">16"] == 3
    assert hist[">64"] == 2
    assert hist[">256"] == 1


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def test_pctl_linear_interpolation() -> None:
    vals = list(range(1, 11))
    assert _pctl(vals, 50) == 5.5
    assert _pctl(vals, 10) == 1.9
    assert _pctl(vals, 90) == 9.1
    assert _pctl(vals, 0) == 1
    assert _pctl(vals, 100) == 10
    assert _pctl([7.0], 50) == 7.0
    assert _pctl([], 50) is None


def test_spearman_perfect_and_degenerate() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = _spearman(xs, [10.0, 9.0, 8.0, 7.0, 6.0])
    assert r is not None and abs(r - (-1.0)) < 1e-12
    r2 = _spearman(xs, [2.0, 4.0, 6.0, 8.0, 10.0])
    assert r2 is not None and abs(r2 - 1.0) < 1e-12
    assert _spearman(xs, [3.0, 3.0, 3.0, 3.0, 3.0]) is None
    assert _spearman([1.0], [1.0]) is None


def test_det_sample_deterministic_and_bounded() -> None:
    ids = [str(i) for i in range(1000)]
    a = _det_sample(ids, 200, 42)
    b = _det_sample(list(reversed(ids)), 200, 42)
    assert a == b  # sorted-input contract: independent of input order
    assert len(a) == 200
    assert len(set(a)) == 200
    assert all(x in set(ids) for x in a)
    small = ["9", "10", "1"]
    assert _det_sample(small, 100, 42) == ["1", "10", "9"]


# ---------------------------------------------------------------------------
# tar streaming scan (synthetic shard)
# ---------------------------------------------------------------------------
def _vp8x_header(w: int, h: int) -> bytes:
    hdr = bytearray(64)
    hdr[0:4] = b"RIFF"
    hdr[8:12] = b"WEBP"
    hdr[12:16] = b"VP8X"
    hdr[16:20] = (10).to_bytes(4, "little")
    hdr[24:27] = (w - 1).to_bytes(3, "little")
    hdr[27:30] = (h - 1).to_bytes(3, "little")
    return bytes(hdr)


def _make_shard(path: Path) -> None:
    docs = {
        "101": {"id": "101", "image": {"width": 1200, "height": 800}, "nsfw": "sfw",
                "source": {"release": "1_2024"}, "quality": "good",
                "anime_completeness": "polished", "anime_classification": "bangumi",
                "tags": {"general": ["a", "b"]}},
        "102": {"id": "102", "image": {"width": 1410, "height": 2048}, "nsfw": "nsfw",
                "source": {"release": "1_2024"}, "quality": "worst",
                "anime_completeness": "polished", "anime_classification": "illustration",
                "tags": {"general": []}},
        "103": {"id": "103", "image": {"width": None, "height": None}, "nsfw": "sfw",
                "source": {"release": "1_2024"}, "tags": {"general": []}},
    }
    # member names carry the repo path prefix, exactly like the v2 shards
    prefix = "danbooru/5.9/1_2024/"
    with tarfile.open(path, "w") as tf:
        for sid, doc in docs.items():
            payload = json.dumps(doc).encode()
            info = tarfile.TarInfo(prefix + f"{sid}.json")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        for sid, (w, h) in (("101", (1200, 800)), ("102", (1410, 2048))):
            payload = _vp8x_header(w, h)
            info = tarfile.TarInfo(prefix + f"{sid}.webp")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        bad = b"not json at all {"
        info = tarfile.TarInfo("104.json")
        info.size = len(bad)
        tf.addfile(info, io.BytesIO(bad))


def test_scan_shard_parses_meta_and_webp_headers(tmp_path: Path) -> None:
    shard = tmp_path / "shard-000000.tar"
    _make_shard(shard)
    name, metas, sizes, n_unresolved = az._scan_shard(str(shard))
    assert name == "shard-000000.tar"
    assert n_unresolved == 2  # 103 null dims + 104 malformed -> both dropped
    assert set(metas) == {"101", "102"}
    m101 = metas["101"]
    assert (m101.quality_tier, m101.completeness, m101.classification) == ("good", "polished", "bangumi")
    assert m101.ai_corrupted is False
    assert m101.tags_general == ("a", "b")
    assert (m101.width_meta, m101.height_meta) == (1200, 800)
    m102 = metas["102"]
    assert m102.nsfw == "nsfw"
    assert (m102.quality_tier, m102.completeness, m102.classification) == ("worst", "polished", "illustration")
    assert sizes == {"101": (1200, 800), "102": (1410, 2048)}
    # pooled predicates on the scanned records
    assert in_p1(m101) and not in_p0(m101)  # good tier
    assert not in_p1(m102)  # worst tier


# ---------------------------------------------------------------------------
# full-repo population selection (no frozen table)
# ---------------------------------------------------------------------------
def test_select_full_repo_population_rederives_from_labels() -> None:
    cfg = Config()
    metas = {
        # eligible (1024 bucket fits the SHIPPED webp dims)
        "301": az.RawMeta("301", "shard-000000.tar", 1500, 1200, "sfw", 2024, "masterpiece", "polished", "illustration", False, ()),
        # meta dims are >=1024 but the shipped webp is 900x800 -> must be EXCLUDED
        # (proves webp-corrected dims are used, not the original-post dims)
        "302": az.RawMeta("302", "shard-000000.tar", 2000, 1500, "sfw", 2024, "masterpiece", "polished", "illustration", False, ()),
        # hard excludes
        "303": az.RawMeta("303", "shard-000000.tar", 2000, 2000, "nsfw", 2024, "masterpiece", "polished", "illustration", False, ()),
        "304": az.RawMeta("304", "shard-000000.tar", 2000, 2000, "sfw", 2024, "masterpiece", "polished", "illustration", True, ()),
        "305": az.RawMeta("305", "shard-000000.tar", 2000, 2000, "sfw", 2024, "masterpiece", "polished", "not_painting", False, ()),
        # rough -> aux pool, still train-eligible
        "306": az.RawMeta("306", "shard-000000.tar", 2000, 2000, "sfw", 2024, "masterpiece", "rough", "illustration", False, ()),
        # no webp header available -> falls back to meta dims (1600x1400) -> eligible
        "307": az.RawMeta("307", "shard-000000.tar", 1600, 1400, "sfw", 2024, "masterpiece", "polished", "illustration", False, ()),
    }
    sizes = {
        "301": (1500, 1200),
        "302": (900, 800),
        "303": (2000, 2000),
        "304": (2000, 2000),
        "305": (2000, 2000),
        "306": (2000, 2000),
        # 307 intentionally absent
    }
    train, nsfw_dist, n_val = az.select_full_repo_population(metas, sizes, cfg, 1024)
    assert dict(nsfw_dist) == {"sfw": 6, "nsfw": 1}
    for sid in ("302", "303", "304", "305"):
        assert sid not in train, sid
    # expected train/validation split from the pipeline's own is_validation
    eligible = {"301", "306", "307"}
    expected_val: set[str] = set()
    for sid in eligible:
        m = metas[sid]
        w, h = sizes.get(sid, (m.width_meta, m.height_meta))
        rec = MetaRecord(
            sample_id=m.sample_id,
            shard=m.shard,
            rel_path="",
            width=w,
            height=h,
            nsfw=m.nsfw,
            year=m.year,
            quality="",
            tags_general=m.tags_general,
            quality_tier=m.quality_tier,
            anime_completeness=m.completeness,
            anime_classification=m.classification,
            ai_corrupted=m.ai_corrupted,
        )
        if is_validation(rec, cfg):
            expected_val.add(sid)
    assert train == eligible - expected_val
    assert n_val == len(expected_val)
    assert len(train) + n_val == len(eligible)
