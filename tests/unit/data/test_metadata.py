from __future__ import annotations

import pytest

from sakuramoon.data.metadata import (
    MetadataError,
    MetadataRecord,
    parse_metadata,
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
