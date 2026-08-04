"""Minimal metadata fields actually consumed by training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class MetadataError(ValueError):
    """A sample has no usable training identity."""


@dataclass(frozen=True)
class OperationalMetadataRecord:
    id: int
    raw: Mapping[str, object]


@dataclass(frozen=True)
class MetadataFieldMapping:
    id_field: str

    def __post_init__(self) -> None:
        if (
            type(self.id_field) is not str
            or not self.id_field
            or self.id_field != self.id_field.strip()
        ):
            raise MetadataError("metadata id field must be a non-empty string")


def parse_shard_metadata(
    raw: Mapping[str, object],
    *,
    fields: MetadataFieldMapping,
) -> OperationalMetadataRecord:
    """Read the stable sample ID; image dimensions come from the decoded image."""

    if fields.id_field not in raw:
        raise MetadataError(f"metadata is missing mapped field: {fields.id_field}")
    sample_id = raw[fields.id_field]
    if type(sample_id) is not int or sample_id <= 0:
        raise MetadataError("metadata id must be a positive integer")
    return OperationalMetadataRecord(id=sample_id, raw=dict(raw))


__all__ = [
    "MetadataError",
    "MetadataFieldMapping",
    "OperationalMetadataRecord",
    "parse_shard_metadata",
]
