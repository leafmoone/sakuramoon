"""Deterministic validation selection and training exclusion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sakuramoon.data.metadata import MetadataRecord, scan_duplicate_ids

VALIDATION_SAMPLE_COUNT = 2000
AspectBucketResolver = Callable[[int, int], str]


class ValidationSelectionError(ValueError):
    """The metadata cannot produce the locked validation selection."""


@dataclass(frozen=True, order=True)
class ValidationStratum:
    release: str
    aspect_bucket: str
    caption_available: bool


@dataclass(frozen=True)
class ValidationEntry:
    id: int
    stratum: ValidationStratum


@dataclass(frozen=True)
class ValidationSelection:
    entries: tuple[ValidationEntry, ...]
    seed: int

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(entry.id for entry in self.entries)

    def stratum_counts(self) -> dict[ValidationStratum, int]:
        counts: dict[ValidationStratum, int] = {}
        for entry in self.entries:
            counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
        return counts


@dataclass(frozen=True)
class TrainingExclusionReport:
    input_records: int
    output_records: int
    excluded_records: int
    encountered_validation_ids: frozenset[int]


def _stable_rank(seed: int, stratum: ValidationStratum, sample_id: int) -> bytes:
    payload = (
        f"{seed}\0{stratum.release}\0{stratum.aspect_bucket}\0"
        f"{int(stratum.caption_available)}\0{sample_id}"
    ).encode()
    return hashlib.sha256(payload).digest()


def _validated_bucket_key(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValidationSelectionError("aspect bucket resolver returned an invalid key")
    return value


def _allocate_counts(
    sizes: dict[ValidationStratum, int], total: int
) -> dict[ValidationStratum, int]:
    """Give every stratum one sample, then allocate remaining slots by capacity."""

    strata = tuple(sorted(sizes))
    if len(strata) > total:
        raise ValidationSelectionError(
            "validation sample count is smaller than the number of strata"
        )
    allocation = {stratum: 1 for stratum in strata}
    remaining = total - len(strata)
    capacities = {stratum: sizes[stratum] - 1 for stratum in strata}
    capacity_total = sum(capacities.values())
    if remaining > capacity_total:
        raise ValidationSelectionError("not enough unique metadata IDs for validation")
    if remaining == 0:
        return allocation

    used = 0
    remainders: list[tuple[int, ValidationStratum]] = []
    for stratum in strata:
        numerator = remaining * capacities[stratum]
        extra, remainder = divmod(numerator, capacity_total)
        allocation[stratum] += extra
        used += extra
        remainders.append((remainder, stratum))
    for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : remaining - used
    ]:
        allocation[stratum] += 1
    return allocation


def select_validation_records(
    records: tuple[MetadataRecord, ...],
    *,
    seed: int,
    aspect_bucket: AspectBucketResolver,
) -> ValidationSelection:
    """Select exactly 2,000 unique IDs across the locked three-part strata."""

    duplicate_report = scan_duplicate_ids(records)
    if duplicate_report.has_duplicates:
        raise ValidationSelectionError("metadata contains duplicate logical IDs")
    if len(records) < VALIDATION_SAMPLE_COUNT:
        raise ValidationSelectionError("not enough unique metadata IDs for validation")

    grouped: dict[ValidationStratum, list[MetadataRecord]] = {}
    for record in records:
        bucket = _validated_bucket_key(aspect_bucket(record.width, record.height))
        stratum = ValidationStratum(
            release=record.release,
            aspect_bucket=bucket,
            caption_available=record.caption_available,
        )
        grouped.setdefault(stratum, []).append(record)

    allocation = _allocate_counts(
        {stratum: len(candidates) for stratum, candidates in grouped.items()},
        VALIDATION_SAMPLE_COUNT,
    )
    selected: list[ValidationEntry] = []
    for stratum in sorted(grouped):
        candidates = sorted(
            grouped[stratum],
            key=lambda record: (_stable_rank(seed, stratum, record.id), record.id),
        )
        selected.extend(
            ValidationEntry(record.id, stratum)
            for record in candidates[: allocation[stratum]]
        )
    selected.sort(key=lambda entry: entry.id)
    result = ValidationSelection(entries=tuple(selected), seed=seed)
    if len(result.entries) != VALIDATION_SAMPLE_COUNT or len(result.ids) != len(
        result.entries
    ):
        raise ValidationSelectionError("validation selection is not exactly 2,000 unique IDs")
    return result


def exclude_validation_records(
    records: Iterable[MetadataRecord], validation_ids: frozenset[int]
) -> tuple[tuple[MetadataRecord, ...], TrainingExclusionReport]:
    """Remove validation IDs before downstream shuffle or sample processing."""

    training: list[MetadataRecord] = []
    encountered: set[int] = set()
    total = 0
    for record in records:
        total += 1
        if record.id in validation_ids:
            encountered.add(record.id)
        else:
            training.append(record)
    report = TrainingExclusionReport(
        input_records=total,
        output_records=len(training),
        excluded_records=total - len(training),
        encountered_validation_ids=frozenset(encountered),
    )
    return tuple(training), report


def validation_manifest_bytes(selection: ValidationSelection) -> bytes:
    """Serialize the deterministic validation selection as canonical JSONL."""

    lines: list[str] = []
    for entry in selection.entries:
        payload = {
            "aspect_bucket": entry.stratum.aspect_bucket,
            "caption_available": entry.stratum.caption_available,
            "id": entry.id,
            "release": entry.stratum.release,
        }
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()
