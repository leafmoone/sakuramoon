"""Dataset manifest for the remote ModelScope WebDataset shards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


class DatasetManifestPublicationError(DatasetManifestError):
    pass


class DatasetManifestExistsError(DatasetManifestPublicationError):
    pass


class ManifestBuildInventoryError(DatasetManifestError):
    pass


class RemoteManifestBuildError(DatasetManifestError):
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


class ShardBuildRecord(StrictModel):
    path: NonEmpty
    release: NonEmpty
    samples: PositiveInt

    @model_validator(mode="after")
    def validate_path(self) -> ShardBuildRecord:
        if not is_safe_shard_path(self.path):
            raise ValueError("path must be a normalized relative WebDataset tar path")
        return self


class RemoteShardRecord(StrictModel):
    path: NonEmpty
    bytes: PositiveInt
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> RemoteShardRecord:
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


class ManifestBuildInventory(StrictModel):
    schema_version: Literal[1]
    source: DatasetSourceIdentity
    shards: Annotated[tuple[ShardBuildRecord, ...], BeforeValidator(_tuple_from_toml)]

    @model_validator(mode="after")
    def validate_inventory(self) -> ManifestBuildInventory:
        paths = tuple(shard.path for shard in self.shards)
        if not paths:
            raise ValueError("manifest build inventory must contain at least one shard")
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("manifest build inventory paths must be sorted and unique")
        return self


def canonical_manifest_bytes(manifest: DatasetManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def manifest_sha256(manifest: DatasetManifest) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def canonical_build_inventory_bytes(inventory: ManifestBuildInventory) -> bytes:
    payload = inventory.model_dump(mode="json")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


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


def parse_manifest_build_inventory_bytes(payload: bytes) -> ManifestBuildInventory:
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        inventory = ManifestBuildInventory.model_validate(parsed, strict=True)
    except DatasetManifestError:
        raise ManifestBuildInventoryError("manifest build inventory is invalid") from None
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
        raise ManifestBuildInventoryError("manifest build inventory is invalid") from None
    if payload != canonical_build_inventory_bytes(inventory):
        raise ManifestBuildInventoryError(
            "manifest build inventory is not canonically encoded"
        )
    return inventory


def load_manifest_build_inventory(
    path: Path,
    expected_sha256: str,
    source: DataSourceConfig,
) -> ManifestBuildInventory:
    try:
        payload = path.read_bytes()
    except OSError:
        raise ManifestBuildInventoryError(
            "manifest build inventory could not be read"
        ) from None
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ManifestBuildInventoryError(
            "manifest build inventory SHA-256 does not match"
        )
    inventory = parse_manifest_build_inventory_bytes(payload)
    if (
        inventory.source.repo_id != source.repo_id
        or inventory.source.revision != source.revision
    ):
        raise ManifestBuildInventoryError(
            "manifest build inventory source does not match config"
        )
    return inventory


def build_dataset_manifest(
    inventory: ManifestBuildInventory,
    remote_shards: tuple[RemoteShardRecord, ...],
) -> DatasetManifest:
    remote_by_path = {shard.path: shard for shard in remote_shards}
    if len(remote_by_path) != len(remote_shards):
        raise RemoteManifestBuildError("remote inventory contains duplicate shard paths")
    expected_paths = {shard.path for shard in inventory.shards}
    if set(remote_by_path) != expected_paths:
        raise RemoteManifestBuildError(
            "remote inventory paths differ from build inventory"
        )
    return DatasetManifest.from_shards(
        inventory.source,
        tuple(
            ShardRecord(
                path=shard.path,
                release=shard.release,
                bytes=remote_by_path[shard.path].bytes,
                sha256=remote_by_path[shard.path].sha256,
                samples=shard.samples,
            )
            for shard in inventory.shards
        ),
    )


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
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise DatasetManifestExistsError(
                "dataset manifest destination already exists"
            ) from None
        temporary.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except DatasetManifestExistsError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise DatasetManifestPublicationError(
            "dataset manifest could not be written"
        ) from None
    return hashlib.sha256(payload).hexdigest()
