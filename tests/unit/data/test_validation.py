from __future__ import annotations

import json

import pytest

from sakuramoon.data.metadata import MetadataRecord
from sakuramoon.data.validation import (
    VALIDATION_SAMPLE_COUNT,
    ValidationSelectionError,
    exclude_validation_records,
    select_validation_records,
    validation_manifest_bytes,
)


def _record(sample_id: int) -> MetadataRecord:
    wide = sample_id % 2 == 0
    return MetadataRecord(
        id=sample_id,
        release="release-a" if sample_id % 4 < 2 else "release-b",
        width=768 if wide else 512,
        height=512 if wide else 768,
        caption_available=sample_id % 8 < 4,
        raw={"id": sample_id},
    )


def _records(count: int = 2400) -> tuple[MetadataRecord, ...]:
    return tuple(_record(sample_id) for sample_id in range(1, count + 1))


def _bucket(width: int, height: int) -> str:
    return "wide" if width > height else "tall"


def test_selection_is_exact_unique_stratified_and_order_independent() -> None:
    records = _records()

    first = select_validation_records(records, seed=1234, aspect_bucket=_bucket)
    reordered = select_validation_records(
        tuple(reversed(records)), seed=1234, aspect_bucket=_bucket
    )

    assert first == reordered
    assert len(first.entries) == VALIDATION_SAMPLE_COUNT
    assert len(first.ids) == VALIDATION_SAMPLE_COUNT
    counts = first.stratum_counts()
    assert len(counts) == 8
    assert set(counts.values()) == {250}


def test_selection_seed_changes_members_not_stratum_counts() -> None:
    records = _records()
    first = select_validation_records(records, seed=1, aspect_bucket=_bucket)
    second = select_validation_records(records, seed=2, aspect_bucket=_bucket)

    assert first.ids != second.ids
    assert first.stratum_counts() == second.stratum_counts()


def test_selection_allocates_unequal_strata_by_available_population() -> None:
    records = list(_records(2100))
    for index in range(1, 101):
        record = records[index - 1]
        records[index - 1] = MetadataRecord(
            id=record.id,
            release="small-release",
            width=record.width,
            height=record.height,
            caption_available=record.caption_available,
            raw=record.raw,
        )

    selection = select_validation_records(
        tuple(records), seed=3, aspect_bucket=_bucket
    )

    assert len(selection.entries) == VALIDATION_SAMPLE_COUNT
    assert all(count > 0 for count in selection.stratum_counts().values())


def test_selection_rejects_duplicate_or_insufficient_ids() -> None:
    with pytest.raises(ValidationSelectionError, match="not enough"):
        select_validation_records(_records(1999), seed=1, aspect_bucket=_bucket)

    duplicate = _records(2000) + (_record(1),)
    with pytest.raises(ValidationSelectionError, match="duplicate"):
        select_validation_records(duplicate, seed=1, aspect_bucket=_bucket)


@pytest.mark.parametrize("bucket", ["", " ", 1])
def test_selection_rejects_invalid_bucket_resolver_result(bucket: object) -> None:
    with pytest.raises(ValidationSelectionError, match="invalid key"):
        select_validation_records(
            _records(2000),
            seed=1,
            aspect_bucket=lambda width, height: bucket,  # type: ignore[return-value]
        )


def test_training_exclusion_removes_all_validation_ids() -> None:
    records = _records()
    selection = select_validation_records(records, seed=11, aspect_bucket=_bucket)

    training, report = exclude_validation_records(records, selection.ids)

    assert report.input_records == len(records)
    assert report.output_records == len(records) - VALIDATION_SAMPLE_COUNT
    assert report.excluded_records == VALIDATION_SAMPLE_COUNT
    assert report.encountered_validation_ids == selection.ids
    assert not any(record.id in selection.ids for record in training)


def test_validation_manifest_jsonl_is_deterministic() -> None:
    selection = select_validation_records(_records(), seed=99, aspect_bucket=_bucket)

    first = validation_manifest_bytes(selection)
    second = validation_manifest_bytes(selection)

    assert first == second
    assert first.endswith(b"\n")
    lines = first.splitlines()
    assert len(lines) == VALIDATION_SAMPLE_COUNT
    parsed = json.loads(lines[0])
    assert set(parsed) == {"aspect_bucket", "caption_available", "id", "release"}
    assert [json.loads(line)["id"] for line in lines] == sorted(selection.ids)
