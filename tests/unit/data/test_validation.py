from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pytest

from sakuramoon.data.metadata import MetadataRecord
from sakuramoon.data.validation import (
    VALIDATION_SAMPLE_COUNT,
    ValidationBundleExistsError,
    ValidationEntry,
    ValidationPublicationError,
    ValidationSelection,
    ValidationSelectionError,
    ValidationShardMember,
    ValidationShardSample,
    ValidationStratum,
    exclude_validation_records,
    select_validation_records,
    validation_manifest_bytes,
    write_validation_bundle,
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


def test_selection_type_rejects_noncanonical_cardinality_ids_and_seed() -> None:
    stratum = ValidationStratum("release-a", "wide", True)
    entries = tuple(
        ValidationEntry(sample_id, stratum)
        for sample_id in range(1, VALIDATION_SAMPLE_COUNT + 1)
    )
    with pytest.raises(ValidationSelectionError, match="exactly 2,000"):
        ValidationSelection(entries=entries[:-1], seed=1)
    with pytest.raises(ValidationSelectionError, match="sorted positive"):
        ValidationSelection(entries=tuple(reversed(entries)), seed=1)
    with pytest.raises(ValidationSelectionError, match="seed"):
        ValidationSelection(entries=entries, seed=True)


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


def _validation_samples(
    ids: tuple[int, ...],
) -> tuple[ValidationShardSample, ...]:
    return tuple(
        ValidationShardSample(
            id=sample_id,
            members=(
                ValidationShardMember(
                    suffix="json",
                    payload=json.dumps(
                        {"id": sample_id}, separators=(",", ":")
                    ).encode(),
                ),
            ),
        )
        for sample_id in ids
    )


def test_validation_bundle_is_deterministic_exact_and_no_clobber(
    tmp_path: Path,
) -> None:
    selection = select_validation_records(_records(), seed=99, aspect_bucket=_bucket)
    ordered_ids = tuple(entry.id for entry in selection.entries)
    samples = _validation_samples(ordered_ids)

    first = write_validation_bundle(selection, samples, tmp_path / "first")
    second = write_validation_bundle(selection, samples, tmp_path / "second")

    assert first.manifest_path.read_bytes() == validation_manifest_bytes(selection)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.shard_path.read_bytes() == second.shard_path.read_bytes()
    assert first.shard_sha256 == second.shard_sha256
    assert first.shard_bytes == second.shard_bytes > 0
    with tarfile.open(first.shard_path, mode="r:") as archive:
        members = archive.getmembers()
        assert len(members) == VALIDATION_SAMPLE_COUNT
        assert tuple(member.name for member in members) == tuple(
            f"{sample_id}.json" for sample_id in ordered_ids
        )
        assert all(member.mtime == 0 for member in members)
        first_payload = archive.extractfile(members[0])
        assert first_payload is not None
        assert json.load(first_payload) == {"id": ordered_ids[0]}

    with pytest.raises(ValidationBundleExistsError, match="already exists"):
        write_validation_bundle(selection, samples, first.root)
    assert first.shard_path.read_bytes() == second.shard_path.read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "extra", "out_of_order"])
def test_validation_bundle_rejects_id_drift_and_cleans_temporary_directory(
    tmp_path: Path,
    mutation: str,
) -> None:
    selection = select_validation_records(_records(), seed=17, aspect_bucket=_bucket)
    ids = [entry.id for entry in selection.entries]
    if mutation == "missing":
        ids.pop()
    elif mutation == "extra":
        ids.append(max(ids) + 1)
    else:
        ids[0], ids[1] = ids[1], ids[0]

    destination = tmp_path / "validation"
    with pytest.raises(ValidationSelectionError, match="validation shard"):
        write_validation_bundle(
            selection,
            _validation_samples(tuple(ids)),
            destination,
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".validation.*.tmp"))


def test_validation_bundle_rolls_back_final_when_parent_fsync_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection = select_validation_records(_records(), seed=17, aspect_bucket=_bucket)
    ids = tuple(entry.id for entry in selection.entries)
    destination = tmp_path / "validation"
    real_fsync = os.fsync
    calls = 0

    def fail_parent_fsync(file_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls >= 4:
            raise OSError("injected parent fsync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    with pytest.raises(ValidationPublicationError, match="could not be published"):
        write_validation_bundle(selection, _validation_samples(ids), destination)

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".validation.*.tmp"))


def test_validation_shard_members_are_nonempty_sorted_unique_and_safe() -> None:
    with pytest.raises(ValidationSelectionError, match="must have members"):
        ValidationShardSample(id=1, members=())
    with pytest.raises(ValidationSelectionError, match="sorted and unique"):
        ValidationShardSample(
            id=1,
            members=(
                ValidationShardMember("json", b"{}"),
                ValidationShardMember("jpg", b"x"),
            ),
        )
    with pytest.raises(ValidationSelectionError, match="suffix is invalid"):
        ValidationShardMember("../json", b"{}")
