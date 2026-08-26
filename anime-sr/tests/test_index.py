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
)

CFG = Config()


def _meta(id_: int, w: int | None, h: int | None, nsfw: str = "sfw", general: list[str] | None = None) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "id": id_,
            "source": {"dataset": "danbooru", "release": "1_2024", "original_path": f"1_2024/{id_}.webp"},
            "image": {"format": "webp", "width": w, "height": h},
            "nsfw": nsfw,
            "tags": {"general": general or ["anime", "solo_female"]},
        }
    )


def _make_shard(tmp_path: Path, name: str = "shard-000000.tar") -> Path:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        cases: list[tuple[int, int | None, int | None, str, list[str] | None]] = [
            (1001, 2000, 1500, "sfw", ["masterpiece", "illustration"]),  # priority, all buckets
            (1002, 900, 800, "sfw", None),  # regular, 512/768 only
            (1003, 400, 300, "sfw", None),  # too small for any bucket
            (1004, 1600, 1200, "nsfw", None),  # nsfw hard exclude
            (1005, 1600, 1200, "sfw", ["watermark"]),  # hard-tag exclude
            (1006, 3000, 2000, "sfw", ["monochrome"]),  # aux pool
            (1007, None, None, "sfw", None),  # unresolved dims (JSON null) → dropped
        ]
        for cid, w, h, nsfw, general in cases:
            stem = f"danbooru/5.9/1_2024/{cid}"
            webp_bytes = b"webp-bytes"
            json_bytes = _meta(cid, w, h, nsfw, general).encode()
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
    assert len(records) == 6  # 1007 (null dims) is dropped as unresolved
    assert summary.n_images == 6 and summary.n_sfw == 5
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
    assert by_id["1001"].quality == "priority"
    assert by_id["1006"].quality == "aux"


def test_build_index_artifacts(tmp_path: Path) -> None:
    shard = _make_shard(tmp_path)
    out = tmp_path / "index"
    report = build_index([str(shard)], CFG, out)
    assert report["n_images"] == 6
    assert report["n_unresolved"] == 1
    assert report["n_eligible_train"] == 3  # 1001, 1002, 1006
    # eligibility table readable back (parquet or jsonl)
    rows = list(iter_index(report["paths"]["eligibility"]))
    assert len(rows) == 6
    ids = {r["sample_id"] for r in rows if r["is_validation"]}
    # validation doc: structural zero overlap
    val_doc = json.loads((out / "sr-validation-v1.json").read_text())
    assert val_doc["zero_overlap"] is True
    assert val_doc["n_train"] + val_doc["n_validation"] == 6
    # split determinism
    records, _ = scan_shard(shard, progress=False)
    assert all(is_validation(r, CFG) == (r.sample_id in val_doc["validation_ids"]) for r in records)
    assert ids == set(val_doc["validation_ids"])


def test_validation_split_zero_overlap(tmp_path: Path) -> None:
    records, _ = scan_shard(_make_shard(tmp_path), progress=False)
    from anime_sr.data.index import split_index

    train, val = split_index(records, CFG)
    assert not ({r.sample_id for r in train} & {r.sample_id for r in val})
