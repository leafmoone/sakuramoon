"""M1 index: scan, eligibility funnel, validation split, artifacts (plan §10).

Builds a tiny in-memory shard tar (danbooru-v2 meta schema) and checks the
full ``build_index`` funnel end to end.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from anime_sr.config.schema import Config
from anime_sr.data.index import (
    build_index,
    evaluate_eligibility,
    is_validation,
    iter_index,
    scan_shard,
    webp_header_size,
)

CFG = Config()


def _meta(
    id_: int,
    w: int | None,
    h: int | None,
    nsfw: str = "sfw",
    general: list[str] | None = None,
    tier: str = "",
    completeness: str = "",
    classification: str = "",
    ai_corrupted: bool = False,
) -> str:
    doc = {
        "schema_version": 1,
        "id": id_,
        "source": {"dataset": "danbooru", "release": "1_2024", "original_path": f"1_2024/{id_}.webp"},
        "image": {"format": "webp", "width": w, "height": h},
        "nsfw": nsfw,
        "tags": {"general": general or ["anime", "solo_female"]},
    }
    if tier:
        doc["quality"] = tier
    if completeness:
        doc["anime_completeness"] = completeness
    if classification:
        doc["anime_classification"] = classification
    if ai_corrupted:
        doc["ai_image_corrupted"] = "corrupted"
    return json.dumps(doc)


def _make_shard(tmp_path: Path, name: str = "shard-000000.tar") -> Path:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        cases = [
            # (id, w, h, nsfw, general, tier, completeness, classification, ai)
            (1001, 2000, 1500, "sfw", ["anime"], "masterpiece", "polished", "illustration", False),  # priority, all buckets
            (1002, 900, 800, "sfw", None, "normal", "polished", "illustration", False),  # regular tier, 512/768 only
            (1003, 400, 300, "sfw", None, "normal", "polished", "illustration", False),  # too small
            (1004, 1600, 1200, "nsfw", None, "normal", "polished", "illustration", False),  # nsfw hard exclude
            (1005, 1600, 1200, "sfw", ["watermark"], "normal", "polished", "illustration", False),  # hard-tag
            (1006, 3000, 2000, "sfw", None, "good", "monochrome", "illustration", False),  # aux pool (completeness)
            (1007, None, None, "sfw", None, "normal", "polished", "illustration", False),  # unresolved dims
            (1008, 1600, 1200, "sfw", None, "masterpiece", "polished", "illustration", True),  # ai-corrupted hard reject
            (1009, 1600, 1200, "sfw", None, "masterpiece", "polished", "not_painting", False),  # not_painting hard reject
        ]
        for cid, w, h, nsfw, general, tier, completeness, classification, ai in cases:
            stem = f"danbooru/5.9/1_2024/{cid}"
            webp_bytes = b"webp-bytes"
            json_bytes = _meta(cid, w, h, nsfw, general, tier, completeness, classification, ai).encode()
            t_webp = tarfile.TarInfo(f"{stem}.webp")
            t_webp.size = len(webp_bytes)  # addfile only writes content when size > 0
            tf.addfile(t_webp, io.BytesIO(webp_bytes))
            t_json = tarfile.TarInfo(f"{stem}.json")
            t_json.size = len(json_bytes)
            tf.addfile(t_json, io.BytesIO(json_bytes))
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


def test_scan_shard(tmp_path: Path) -> None:
    p = _make_shard(tmp_path)
    records, summary = scan_shard(p, progress=False)
    assert len(records) == 8  # 1007 (null dims) is dropped as unresolved
    assert summary.n_images == 8 and summary.n_sfw == 7
    assert summary.n_unresolved == 1
    r1 = next(r for r in records if r.sample_id == "1001")
    assert r1.rel_path == "danbooru/5.9/1_2024/1001.webp"
    assert r1.width == 2000 and r1.year == 2024 and r1.nsfw == "sfw"


def test_eligibility_funnel(tmp_path: Path) -> None:
    records, _ = scan_shard(_make_shard(tmp_path), progress=False)
    from anime_sr.data.index import classify_quality

    by_id = {r.sample_id: classify_quality(r, CFG) for r in records}
    e1 = evaluate_eligibility(by_id["1001"], CFG)
    assert e1.eligible_train and e1.eligible_buckets == (512, 768, 1024)
    e2 = evaluate_eligibility(by_id["1002"], CFG)
    assert e2.eligible_buckets == (512, 768)
    e3 = evaluate_eligibility(by_id["1003"], CFG)
    assert not e3.eligible_train and "size:small" in e3.reasons
    e4 = evaluate_eligibility(by_id["1004"], CFG)
    assert not e4.eligible_train and any(r.startswith("nsfw:") for r in e4.reasons)
    e5 = evaluate_eligibility(by_id["1005"], CFG)
    assert not e5.eligible_train and any(r.startswith("hard-tag:") for r in e5.reasons)
    # danbooru-field hard rejects (P1 ②)
    e8 = evaluate_eligibility(by_id["1008"], CFG)
    assert not e8.eligible_train and "ai_image_corrupted" in e8.reasons
    e9 = evaluate_eligibility(by_id["1009"], CFG)
    assert not e9.eligible_train and any(r.startswith("hard-class:not_painting") for r in e9.reasons)
    # pools from danbooru fields, not tag heuristics
    assert by_id["1001"].quality == "priority"  # masterpiece + polished + illustration
    assert by_id["1002"].quality == "regular"  # "normal" tier is not a priority tier
    assert by_id["1006"].quality == "aux"  # monochrome completeness


def test_build_index_artifacts(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    out = tmp_path / "index"
    report = build_index([str(shard)], CFG, out)
    assert report["n_images"] == 8
    assert report["n_unresolved"] == 1
    assert report["n_eligible_train"] == 3  # 1001, 1002, 1006 (1008/1009 hard-rejected)
    # reason codes recorded per image (P1 ②)
    assert report["n_by_reason"]["ai_image_corrupted"] == 1
    assert report["n_by_reason"]["hard-class:not_painting"] == 1
    # eligibility table readable back (parquet or jsonl)
    rows = list(iter_index(report["paths"]["eligibility"]))
    assert len(rows) == 8
    by_id = {r["sample_id"]: r for r in rows}
    # §10.3 row fields: danbooru tier + pool + anime fields + lazy clean-score
    assert by_id["1001"]["quality"] == "masterpiece"
    assert by_id["1001"]["sampling_pool"] == "priority"
    assert by_id["1001"]["anime_completeness"] == "polished"
    assert by_id["1001"]["anime_classification"] == "illustration"
    assert by_id["1008"]["ai_corrupted"] is True
    assert by_id["1002"]["ai_corrupted"] is False
    assert by_id["1001"]["clean_score"] is None
    ids = {r["sample_id"] for r in rows if r["is_validation"]}
    # validation doc: structural zero overlap
    val_doc = json.loads((out / "sr-validation-v1.json").read_text())
    assert val_doc["zero_overlap"] is True
    assert val_doc["n_train"] + val_doc["n_validation"] == 8
    # split determinism
    records, _ = scan_shard(shard, progress=False)
    assert all(is_validation(r, CFG) == (r.sample_id in val_doc["validation_ids"]) for r in records)
    assert ids == set(val_doc["validation_ids"])


def test_validation_split_zero_overlap(tmp_path: Path) -> None:
    records, _ = scan_shard(_make_shard(tmp_path), progress=False)
    from anime_sr.data.index import split_index

    train, val = split_index(records, CFG)
    assert not ({r.sample_id for r in train} & {r.sample_id for r in val})


def test_webp_header_size() -> None:
    def u32(n: int) -> bytes:
        return n.to_bytes(4, "little")

    def u24(n: int) -> bytes:
        return n.to_bytes(3, "little")

    # VP8 (lossy): frame tag @20, start code @23, w @26, h @28 (14-bit LE)
    vp8 = b"RIFF" + u32(30) + b"WEBP" + b"VP8 " + u32(16) + b"\x9d\x01\x2a" + b"\x0d\x0a\x87" + u32(64)[0:2] + u32(32)[0:2]
    assert webp_header_size(vp8) == (64, 32)
    # VP8L (lossless): signature @20, 32-bit field @21:
    # bits [3] = signature, [17:4] = width-1, [31:18] = height-1
    v = (99 << 4) + (49 << 18)
    vp8l = b"RIFF" + u32(26) + b"WEBP" + b"VP8L" + u32(12) + b"\x2f" + v.to_bytes(4, "little")
    assert webp_header_size(vp8l) == (100, 50)
    # VP8X (extended): flags @20, w-1 @24 (24-bit LE), h-1 @27
    vp8x = b"RIFF" + u32(30) + b"WEBP" + b"VP8X" + u32(16) + b"\x00" * 4 + u24(1279) + u24(719)
    assert webp_header_size(vp8x) == (1280, 720)
    # not a webp / too short
    assert webp_header_size(b"NOTA" + b"\x00" * 30) is None
    assert webp_header_size(b"RIFFWEBP" + b"\x00" * 8) is None
    # .part-style zero-byte header
    assert webp_header_size(b"") is None


def test_build_index_size_overrides(tmp_path: Path) -> None:
    """Meta dims describe the original post; overrides carry shipped-webp dims."""
    shard = _make_shard(tmp_path)
    out = tmp_path / "idx"
    overrides = {"1001": (700, 600), "1003": (512, 512), "9999": (9, 9)}  # 9999 not indexed
    report = build_index([str(shard)], CFG, out, size_overrides=overrides)
    assert report["n_size_corrected"] == 2
    assert report["n_size_missing"] == 6  # 8 indexed, 2 overridden
    rows = {r["sample_id"]: r for r in iter_index(report["paths"]["eligibility"])}
    # 1001: 2000x1500 meta -> 700x600 actual: min 600 → 512 bucket only
    assert (rows["1001"]["width"], rows["1001"]["height"]) == (700, 600)
    assert rows["1001"]["eligible_buckets"] == [512]
    # 1003: 400x300 meta (size:small) -> 512x512 actual → now eligible for 512
    assert rows["1003"]["eligible_train"] is True
    assert rows["1003"]["eligible_buckets"] == [512]
    # 1002 untouched by the override map
    assert (rows["1002"]["width"], rows["1002"]["height"]) == (900, 800)
    assert rows["1002"]["eligible_buckets"] == [512, 768]
    # funnel: 1001, 1002, 1003 (newly eligible), 1006 → 4
    assert report["n_eligible_train"] == 4
    # validation split unaffected by size overrides (id-based)
    val_doc = json.loads((out / "sr-validation-v1.json").read_text())
    assert val_doc["n_train"] + val_doc["n_validation"] == 8
