"""Minimal metadata fields actually consumed by training."""

from __future__ import annotations

import hashlib
import re
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


def _derive_sample_id(sample_key: str) -> int:
    """Stable positive integer identity for samples whose metadata id is empty.

    Preferred: trailing decimal run of the tar member key
    (e.g. "a2_filter_4429173" -> 4429173).
    Fallback: first 53 bits of sha256(key) + 1 (stable, 2**53 space).
    2026-08-31: Option-B fallback for the 2D annotation shards (a2/bg) whose
    published json carries an empty string id; danbooru samples keep their
    real integer id and are unaffected.
    """
    match = re.search(r"(\d+)$", sample_key)
    if match is not None:
        value = int(match.group(1))
        if 0 < value < 2**53:
            return value
    return int.from_bytes(hashlib.sha256(sample_key.encode("utf-8")).digest()[:8], "big") % (2**53 - 1) + 1


def parse_shard_metadata(
    raw: Mapping[str, object],
    *,
    fields: MetadataFieldMapping,
    sample_key: str | None = None,
) -> OperationalMetadataRecord:
    """Read the stable sample ID; image dimensions come from the decoded image."""

    if fields.id_field not in raw:
        raise MetadataError(f"metadata is missing mapped field: {fields.id_field}")
    sample_id = raw[fields.id_field]
    if type(sample_id) is not int or sample_id <= 0:
        if sample_key is None:
            raise MetadataError("metadata id must be a positive integer")
        sample_id = _derive_sample_id(sample_key)
    return OperationalMetadataRecord(id=sample_id, raw=dict(raw))


__all__ = [
    "MetadataError",
    "MetadataFieldMapping",
    "OperationalMetadataRecord",
    "parse_shard_metadata",
]
