"""Operational manifest for the ModelScope WebDataset shards."""

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
DATASET_REVISION = "master"

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
    """The configured mutable branch selector, not an immutable commit identity."""

    repo_id: Literal["leafmoone/webdataset_danbooru"]
    revision: Literal["master"]


class ShardRecord(StrictModel):
    """Facts returned by the upstream listing and used for download verification."""

    path: NonEmpty
    bytes: PositiveInt
    upstream_sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> ShardRecord:
        if not is_safe_shard_path(self.path):
            raise ValueError("path must be a normalized relative WebDataset tar path")
        return self


class RemoteShardRecord(StrictModel):
    path: NonEmpty
    bytes: PositiveInt
    upstream_sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> RemoteShardRecord:
        if not is_safe_shard_path(self.path):
            raise ValueError("path must be a normalized relative WebDataset tar path")
        return self


class ManifestAggregates(StrictModel):
    shards: PositiveInt
    bytes: PositiveInt


def _manifest_identity_bytes(
    source: DatasetSourceIdentity,
    shards: tuple[ShardRecord, ...],
    aggregates: ManifestAggregates,
) -> bytes:
    payload = {
        "aggregates": aggregates.model_dump(mode="json"),
        "schema_version": 2,
        "shards": [item.model_dump(mode="json") for item in shards],
        "source": source.model_dump(mode="json"),
    }
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _manifest_id(
    source: DatasetSourceIdentity,
    shards: tuple[ShardRecord, ...],
    aggregates: ManifestAggregates,
) -> str:
    return hashlib.sha256(
        _manifest_identity_bytes(source, shards, aggregates)
    ).hexdigest()


class DatasetManifest(StrictModel):
    schema_version: Literal[2]
    manifest_id: Sha256
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
        )
        if self.aggregates != expected:
            raise ValueError("dataset manifest aggregates do not match shard records")
        if self.manifest_id != _manifest_id(self.source, self.shards, expected):
            raise ValueError("dataset manifest_id does not match its canonical content")
        return self

    @classmethod
    def from_shards(
        cls,
        source: DatasetSourceIdentity,
        shards: tuple[ShardRecord, ...],
    ) -> DatasetManifest:
        ordered = tuple(sorted(shards, key=lambda item: item.path))
        aggregates = ManifestAggregates(
            shards=len(ordered),
            bytes=sum(item.bytes for item in ordered),
        )
        return cls(
            schema_version=2,
            manifest_id=_manifest_id(source, ordered, aggregates),
            source=source,
            shards=ordered,
            aggregates=aggregates,
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
    """Compatibility name for the internal manifest identity, never user input."""

    return manifest.manifest_id


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


def source_identity(source: DataSourceConfig) -> DatasetSourceIdentity:
    try:
        return DatasetSourceIdentity(
            repo_id=source.repo_id,
            revision=source.revision,
        )
    except ValidationError:
        raise DatasetManifestError("dataset source config is invalid") from None


def build_dataset_manifest(
    source: DatasetSourceIdentity,
    remote_shards: tuple[RemoteShardRecord, ...],
) -> DatasetManifest:
    remote_by_path = {shard.path: shard for shard in remote_shards}
    if not remote_shards:
        raise RemoteManifestBuildError("remote inventory contains no WebDataset shards")
    if len(remote_by_path) != len(remote_shards):
        raise RemoteManifestBuildError("remote inventory contains duplicate shard paths")
    return DatasetManifest.from_shards(
        source,
        tuple(
            ShardRecord(
                path=shard.path,
                bytes=shard.bytes,
                upstream_sha256=shard.upstream_sha256,
            )
            for shard in remote_shards
        ),
    )


def load_dataset_manifest(
    path: Path,
    source: DataSourceConfig,
) -> DatasetManifest:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        payload = path.read_bytes()
    except OSError:
        raise DatasetManifestError("dataset manifest could not be read") from None
    manifest = parse_dataset_manifest_bytes(payload)
    if manifest.source != source_identity(source):
        raise DatasetManifestError("dataset manifest source does not match config")
    return manifest


def write_dataset_manifest(manifest: DatasetManifest, destination: Path) -> str:
    payload = canonical_manifest_bytes(manifest)
    temporary: Path | None = None
    published = False
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
            published = True
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
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    except OSError:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if published:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                directory_fd = os.open(
                    destination.parent, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        raise DatasetManifestPublicationError(
            "dataset manifest could not be written"
        ) from None
    return manifest.manifest_id
