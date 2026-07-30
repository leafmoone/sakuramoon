"""Minimal metadata contract used before image and caption processing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class MetadataError(ValueError):
    """A metadata row does not satisfy the fields required by the data pipeline."""


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
    if not isinstance(release, str) or not release.strip():
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
