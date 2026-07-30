"""Dataset manifest for the remote ModelScope WebDataset shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from sakuramoon.config.schema import DataSourceConfig

DATASET_REPO_ID = "leafmoone/webdataset_danbooru"

Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=512)]
PositiveInt = Annotated[int, Field(gt=0)]


def _tuple_from_toml(value: object) -> object:
    return tuple(cast(list[object], value)) if type(value) is list else value


class DatasetManifestError(ValueError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def is_safe_shard_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and value == value.strip()
        and "\\" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and value.casefold().endswith((".tar", ".tar.gz", ".tgz"))
    )


class DatasetSourceIdentity(StrictModel):
    repo_id: Literal["leafmoone/webdataset_danbooru"]
    revision: Commit
    license_id: NonEmpty
    access_terms: NonEmpty


class ShardRecord(StrictModel):
    path: NonEmpty
    release: NonEmpty
    bytes: PositiveInt
    sha256: Sha256
    samples: PositiveInt

    @model_validator(mode="after")
    def validate_path(self) -> ShardRecord:
        if not is_safe_shard_path(self.path):
            raise ValueError("path must be a normalized relative WebDataset tar path")
        return self


class ManifestAggregates(StrictModel):
    shards: PositiveInt
    bytes: PositiveInt
    samples: PositiveInt


class DatasetManifest(StrictModel):
    schema_version: Literal[1]
    source: DatasetSourceIdentity
    shards: Annotated[tuple[ShardRecord, ...], BeforeValidator(_tuple_from_toml)]
    aggregates: ManifestAggregates

    @model_validator(mode="after")
    def validate_inventory(self) -> DatasetManifest:
        paths = tuple(shard.path for shard in self.shards)
        if not paths:
            raise ValueError("dataset manifest must contain at least one shard")
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("dataset shard paths must be sorted and unique")
        expected = ManifestAggregates(
            shards=len(self.shards),
            bytes=sum(shard.bytes for shard in self.shards),
            samples=sum(shard.samples for shard in self.shards),
        )
        if self.aggregates != expected:
            raise ValueError("dataset manifest aggregates do not match shard records")
        return self

    @classmethod
    def from_shards(
        cls,
        source: DatasetSourceIdentity,
        shards: tuple[ShardRecord, ...],
    ) -> DatasetManifest:
        ordered = tuple(sorted(shards, key=lambda item: item.path))
        return cls(
            schema_version=1,
            source=source,
            shards=ordered,
            aggregates=ManifestAggregates(
                shards=len(ordered),
                bytes=sum(item.bytes for item in ordered),
                samples=sum(item.samples for item in ordered),
            ),
        )

    def shard(self, path: str) -> ShardRecord:
        for shard in self.shards:
            if shard.path == path:
                return shard
        raise DatasetManifestError(f"unknown dataset shard: {path}")


def canonical_manifest_bytes(manifest: DatasetManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def manifest_sha256(manifest: DatasetManifest) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetManifestError("dataset manifest contains a duplicate JSON key")
        result[key] = value
    return result


def parse_dataset_manifest_bytes(payload: bytes) -> DatasetManifest:
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        manifest = DatasetManifest.model_validate(parsed, strict=True)
    except DatasetManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
        raise DatasetManifestError("dataset manifest is invalid") from None
    if payload != canonical_manifest_bytes(manifest):
        raise DatasetManifestError("dataset manifest is not canonically encoded")
    return manifest


def load_dataset_manifest(
    path: Path,
    expected_sha256: str,
    source: DataSourceConfig,
) -> DatasetManifest:
    try:
        payload = path.read_bytes()
    except OSError:
        raise DatasetManifestError("dataset manifest could not be read") from None
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DatasetManifestError("dataset manifest SHA-256 does not match config")
    manifest = parse_dataset_manifest_bytes(payload)
    if (
        manifest.source.repo_id != source.repo_id
        or manifest.source.revision != source.revision
    ):
        raise DatasetManifestError("dataset manifest source does not match config")
    return manifest


def write_dataset_manifest(manifest: DatasetManifest, destination: Path) -> str:
    payload = canonical_manifest_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise DatasetManifestError("dataset manifest could not be written") from None
    return hashlib.sha256(payload).hexdigest()
