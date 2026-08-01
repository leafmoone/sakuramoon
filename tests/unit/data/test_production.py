from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sakuramoon.data.production import (
    ProductionDataError,
    adapt_modelscope_metadata,
    parse_modelscope_caption_fields,
)
from sakuramoon.data.validation import (
    VALIDATION_SAMPLE_COUNT,
    ValidationManifestError,
    load_validation_manifest_ids,
)


def _real_row() -> dict[str, object]:
    return {
        "id": 71,
        "image": {"width": 832, "height": 1216},
        "captions": {"nl2": "A blue-haired character.", "nl3": ""},
        "multicaptions": {"vibes": "soft light"},
        "tags": {
            "character": ["alice"],
            "copyright": ["original"],
            "general": ["blue_hair", "dress"],
            "artist": ["artist_name"],
        },
        "dropout": {"candidate_tags": ["blue_hair"]},
        "nsfw": "safe",
    }


def _validation_payload(*, duplicate_last: bool = False) -> bytes:
    rows: list[str] = []
    for sample_id in range(1, VALIDATION_SAMPLE_COUNT + 1):
        row_id = (
            VALIDATION_SAMPLE_COUNT - 1
            if duplicate_last and sample_id == VALIDATION_SAMPLE_COUNT
            else sample_id
        )
        rows.append(
            json.dumps(
                {
                    "aspect_bucket": "square",
                    "caption_available": sample_id % 2 == 0,
                    "id": row_id,
                    "release": "1_2024",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return ("\n".join(rows) + "\n").encode()


def _write_manifest(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_governed_modelscope_adapter_and_caption_parser() -> None:
    raw = _real_row()

    adapted = adapt_modelscope_metadata(raw)
    fields = parse_modelscope_caption_fields(raw)

    assert adapted == {
        "id": 71,
        "width": 832,
        "height": 1216,
        "caption_available": True,
    }
    assert tuple(tag.text for tag in fields.nsfw) == ("safe",)
    assert tuple(tag.text for tag in fields.character) == ("alice",)
    assert tuple(tag.text for tag in fields.general) == ("blue_hair", "dress")
    assert tuple(tag.text for tag in fields.artists) == ("artist_name",)
    assert fields.candidate_tags == frozenset({"blue_hair"})
    assert fields.nl.long_names is None and fields.nl.long_no_names is None
    assert fields.nl.short_vibes == "soft light"
    assert fields.nl.nl2 == "A blue-haired character."
    assert fields.nl.nl3 is None


def test_governed_modelscope_parser_rejects_schema_drift() -> None:
    missing_nested = _real_row()
    missing_nested.pop("image")
    with pytest.raises(ProductionDataError, match="image must be an object"):
        adapt_modelscope_metadata(missing_nested)

    bad_tags = _real_row()
    tags = bad_tags["tags"]
    assert isinstance(tags, dict)
    tags["artist"] = "not-a-list"
    with pytest.raises(ProductionDataError, match=r"tags\.artist must be a list"):
        parse_modelscope_caption_fields(bad_tags)

    bad_candidate = _real_row()
    dropout = bad_candidate["dropout"]
    assert isinstance(dropout, dict)
    dropout["candidate_tags"] = ["valid", 2]
    with pytest.raises(ProductionDataError, match="only strings"):
        parse_modelscope_caption_fields(bad_candidate)


def test_strict_validation_manifest_loader_accepts_exact_canonical_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_manifest.jsonl"
    digest = _write_manifest(path, _validation_payload())

    ids = load_validation_manifest_ids(
        path,
        expected_sha256=digest,
        expected_count=VALIDATION_SAMPLE_COUNT,
    )

    assert len(ids) == VALIDATION_SAMPLE_COUNT
    assert min(ids) == 1 and max(ids) == VALIDATION_SAMPLE_COUNT


def test_strict_validation_manifest_loader_rejects_hash_and_identity_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_manifest.jsonl"
    _write_manifest(path, _validation_payload())
    with pytest.raises(ValidationManifestError, match="SHA-256"):
        load_validation_manifest_ids(
            path,
            expected_sha256="0" * 64,
            expected_count=VALIDATION_SAMPLE_COUNT,
        )

    duplicate_digest = _write_manifest(path, _validation_payload(duplicate_last=True))
    with pytest.raises(ValidationManifestError, match="sorted and globally unique"):
        load_validation_manifest_ids(
            path,
            expected_sha256=duplicate_digest,
            expected_count=VALIDATION_SAMPLE_COUNT,
        )

    with pytest.raises(ValidationManifestError, match="settings"):
        load_validation_manifest_ids(
            path,
            expected_sha256=duplicate_digest,
            expected_count=VALIDATION_SAMPLE_COUNT - 1,
        )


def test_strict_validation_manifest_loader_rejects_noncanonical_and_symlink(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_manifest.jsonl"
    payload = _validation_payload().replace(b'"id":1', b'"id": 1', 1)
    digest = _write_manifest(path, payload)
    with pytest.raises(ValidationManifestError, match="not canonical"):
        load_validation_manifest_ids(
            path,
            expected_sha256=digest,
            expected_count=VALIDATION_SAMPLE_COUNT,
        )

    canonical = tmp_path / "canonical.jsonl"
    canonical_digest = _write_manifest(canonical, _validation_payload())
    link = tmp_path / "linked.jsonl"
    link.symlink_to(canonical)
    with pytest.raises(ValidationManifestError, match="settings"):
        load_validation_manifest_ids(
            link,
            expected_sha256=canonical_digest,
            expected_count=VALIDATION_SAMPLE_COUNT,
        )
