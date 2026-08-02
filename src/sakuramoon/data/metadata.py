"""Minimal metadata contract used before image and caption processing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class MetadataError(ValueError):
    """A metadata row does not satisfy the fields required by the data pipeline."""


def _is_valid_field_name(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


@dataclass(frozen=True)
class MetadataRecord:
    """Fields needed for validation isolation plus the original metadata mapping."""

    id: int
    release: str
    width: int
    height: int
    caption_available: bool
    raw: Mapping[str, object]


@dataclass(frozen=True)
class OperationalMetadataRecord:
    """Training fields available without inventing a dataset release identity."""

    id: int
    width: int
    height: int
    caption_available: bool
    raw: Mapping[str, object]


@dataclass(frozen=True)
class MetadataFieldMapping:
    """Explicit raw metadata keys; production values have no code defaults."""

    id_field: str
    width_field: str
    height_field: str
    caption_available_field: str

    def __post_init__(self) -> None:
        values = (
            self.id_field,
            self.width_field,
            self.height_field,
            self.caption_available_field,
        )
        if any(not _is_valid_field_name(value) for value in values):
            raise MetadataError("metadata field mapping keys must be non-empty strings")
        if len(set(values)) != len(values):
            raise MetadataError("metadata field mapping keys must be unique")


@dataclass(frozen=True)
class DuplicateIdReport:
    total_records: int
    unique_ids: int
    duplicate_ids: tuple[int, ...]
    duplicate_occurrences: int

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicate_ids)


def parse_metadata(raw: Mapping[str, object]) -> MetadataRecord:
    """Parse only the fields whose names and types are currently defined."""

    required = ("id", "release", "width", "height", "caption_available")
    missing = tuple(name for name in required if name not in raw)
    if missing:
        raise MetadataError(f"metadata is missing required fields: {', '.join(missing)}")

    sample_id = raw["id"]
    release = raw["release"]
    width = raw["width"]
    height = raw["height"]
    caption_available = raw["caption_available"]
    if type(sample_id) is not int or sample_id <= 0:
        raise MetadataError("metadata id must be a positive integer")
    if (
        not isinstance(release, str)
        or not release
        or release != release.strip()
    ):
        raise MetadataError("metadata release must be a non-empty string")
    if type(width) is not int or width <= 0:
        raise MetadataError("metadata width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise MetadataError("metadata height must be a positive integer")
    if type(caption_available) is not bool:
        raise MetadataError("metadata caption_available must be a boolean")
    return MetadataRecord(
        id=sample_id,
        release=release,
        width=width,
        height=height,
        caption_available=caption_available,
        raw=dict(raw),
    )


def parse_shard_metadata(
    raw: Mapping[str, object],
    *,
    fields: MetadataFieldMapping,
) -> OperationalMetadataRecord:
    """Map only operational fields directly present in one metadata row."""

    mapped_names = (
        fields.id_field,
        fields.width_field,
        fields.height_field,
        fields.caption_available_field,
    )
    missing = tuple(name for name in mapped_names if name not in raw)
    if missing:
        raise MetadataError(
            f"metadata is missing mapped fields: {', '.join(missing)}"
        )
    sample_id = raw[fields.id_field]
    width = raw[fields.width_field]
    height = raw[fields.height_field]
    caption_available = raw[fields.caption_available_field]
    if type(sample_id) is not int or sample_id <= 0:
        raise MetadataError("metadata id must be a positive integer")
    if type(width) is not int or width <= 0:
        raise MetadataError("metadata width must be a positive integer")
    if type(height) is not int or height <= 0:
        raise MetadataError("metadata height must be a positive integer")
    if type(caption_available) is not bool:
        raise MetadataError("metadata caption_available must be a boolean")
    return OperationalMetadataRecord(
        id=sample_id,
        width=width,
        height=height,
        caption_available=caption_available,
        raw=dict(raw),
    )


def scan_duplicate_ids(records: tuple[MetadataRecord, ...]) -> DuplicateIdReport:
    """Summarize repeated logical IDs without building an external index."""

    counts: dict[int, int] = {}
    for record in records:
        counts[record.id] = counts.get(record.id, 0) + 1
    duplicate_ids = tuple(sorted(sample_id for sample_id, count in counts.items() if count > 1))
    return DuplicateIdReport(
        total_records=len(records),
        unique_ids=len(counts),
        duplicate_ids=duplicate_ids,
        duplicate_occurrences=sum(count - 1 for count in counts.values() if count > 1),
    )
