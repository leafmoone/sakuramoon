from __future__ import annotations

import pytest

from sakuramoon.data.metadata import (
    MetadataError,
    MetadataFieldMapping,
    MetadataRecord,
    OperationalMetadataRecord,
    parse_metadata,
    parse_shard_metadata,
    scan_duplicate_ids,
)


def _raw(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 17,
        "release": "release-a",
        "width": 768,
        "height": 512,
        "caption_available": True,
        "future_caption_fields": {"tags": ["synthetic"]},
    }
    payload.update(changes)
    return payload


def test_parse_metadata_reads_only_minimum_contract_and_preserves_raw() -> None:
    raw = _raw()

    record = parse_metadata(raw)

    assert record == MetadataRecord(
        id=17,
        release="release-a",
        width=768,
        height=512,
        caption_available=True,
        raw=raw,
    )
    assert record.raw["future_caption_fields"] == {"tags": ["synthetic"]}
    assert record.raw is not raw


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", 0, "positive integer"),
        ("id", True, "positive integer"),
        ("release", "", "non-empty string"),
        ("release", " release-a ", "non-empty string"),
        ("width", 0, "positive integer"),
        ("height", 1.5, "positive integer"),
        ("caption_available", 1, "boolean"),
    ],
)
def test_parse_metadata_rejects_invalid_required_field(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(MetadataError, match=message):
        parse_metadata(_raw(**{field: value}))


def test_parse_metadata_reports_missing_fields() -> None:
    raw = _raw()
    del raw["width"]
    del raw["caption_available"]

    with pytest.raises(MetadataError, match="width, caption_available"):
        parse_metadata(raw)


def test_duplicate_id_report_counts_extra_occurrences() -> None:
    records = tuple(
        parse_metadata(_raw(id=sample_id)) for sample_id in (1, 2, 2, 3, 3, 3)
    )

    report = scan_duplicate_ids(records)

    assert report.total_records == 6
    assert report.unique_ids == 3
    assert report.duplicate_ids == (2, 3)
    assert report.duplicate_occurrences == 3
    assert report.has_duplicates is True


def test_unique_id_report_is_empty() -> None:
    records = tuple(parse_metadata(_raw(id=sample_id)) for sample_id in (1, 2, 3))
    report = scan_duplicate_ids(records)
    assert report.duplicate_ids == ()
    assert report.duplicate_occurrences == 0
    assert report.has_duplicates is False


def test_explicit_field_mapping_uses_only_real_row_fields() -> None:
    fields = MetadataFieldMapping(
        id_field="logical_id",
        width_field="image_width",
        height_field="image_height",
        caption_available_field="has_caption",
    )
    raw: dict[str, object] = {
        "logical_id": 23,
        "image_width": 1024,
        "image_height": 768,
        "has_caption": True,
        "release": "untrusted-raw-release",
        "caption_payload": {"nl": "synthetic"},
    }

    record = parse_shard_metadata(raw, fields=fields)

    assert record == OperationalMetadataRecord(
        id=23,
        width=1024,
        height=768,
        caption_available=True,
        raw=raw,
    )


def test_explicit_field_mapping_rejects_missing_or_invalid_values() -> None:
    fields = MetadataFieldMapping("id", "w", "h", "caption")
    with pytest.raises(MetadataError, match="missing mapped fields: caption"):
        parse_shard_metadata(
            {"id": 1, "w": 512, "h": 512},
            fields=fields,
        )
    with pytest.raises(MetadataError, match="boolean"):
        parse_shard_metadata(
            {"id": 1, "w": 512, "h": 512, "caption": 1},
            fields=fields,
        )


@pytest.mark.parametrize(
    "values",
    [
        ("", "w", "h", "caption"),
        (" id", "w", "h", "caption"),
        ("id", "id", "h", "caption"),
    ],
)
def test_field_mapping_has_no_empty_whitespace_or_duplicate_keys(
    values: tuple[str, str, str, str],
) -> None:
    with pytest.raises(MetadataError, match="mapping keys"):
        MetadataFieldMapping(*values)
