from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import DataSourceConfig
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetManifestExistsError,
    DatasetSourceIdentity,
    ManifestAggregates,
    ManifestBuildInventory,
    ManifestBuildInventoryError,
    RemoteManifestBuildError,
    RemoteShardRecord,
    ShardBuildRecord,
    ShardRecord,
    build_dataset_manifest,
    canonical_build_inventory_bytes,
    canonical_manifest_bytes,
    load_dataset_manifest,
    load_manifest_build_inventory,
    manifest_sha256,
    parse_dataset_manifest_bytes,
    parse_manifest_build_inventory_bytes,
    write_dataset_manifest,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _source() -> DatasetSourceIdentity:
    return DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision=REVISION,
        license_id="source-license",
        access_terms="source-access-terms",
    )


def _record(
    path: str = "release-a/000001.tar", content: bytes = b"shard-one"
) -> ShardRecord:
    return ShardRecord(
        path=path,
        release="release-a",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        samples=17,
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest.from_shards(_source(), (_record(),))


def _build_inventory(
    *,
    path: str = "release-a/000001.tar",
    release: str = "explicit-release",
    samples: int = 17,
) -> ManifestBuildInventory:
    return ManifestBuildInventory(
        schema_version=1,
        source=_source(),
        shards=(ShardBuildRecord(path=path, release=release, samples=samples),),
    )


def _config(revision: str = REVISION) -> DataSourceConfig:
    return DataSourceConfig(
        repo_id="leafmoone/webdataset_danbooru", revision=revision
    )


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def test_manifest_round_trip_write_and_load(tmp_path: Path) -> None:
    manifest = _manifest()
    payload = canonical_manifest_bytes(manifest)
    path = tmp_path / "manifest.json"

    digest = write_dataset_manifest(manifest, path)

    assert parse_dataset_manifest_bytes(payload) == manifest
    assert digest == manifest_sha256(manifest) == hashlib.sha256(payload).hexdigest()
    assert load_dataset_manifest(path, digest, _config()) == manifest
    assert not path.with_name(".manifest.json.tmp").exists()


def test_from_shards_sorts_and_computes_aggregates() -> None:
    first = _record("release-a/000001.tar", b"one")
    second = _record("release-b/000002.tar", b"two")

    manifest = DatasetManifest.from_shards(_source(), (second, first))

    assert tuple(item.path for item in manifest.shards) == (first.path, second.path)
    assert manifest.aggregates == ManifestAggregates(
        shards=2,
        bytes=first.bytes + second.bytes,
        samples=first.samples + second.samples,
    )


@pytest.mark.parametrize("revision", ["main", "A" * 40, "0" * 39, "0" * 41])
def test_source_requires_commit_revision(revision: str) -> None:
    with pytest.raises(ValidationError, match="revision"):
        DatasetSourceIdentity(
            repo_id="leafmoone/webdataset_danbooru",
            revision=revision,
            license_id="license",
            access_terms="terms",
        )


def test_source_repository_is_fixed() -> None:
    with pytest.raises(ValidationError, match="repo_id"):
        DatasetSourceIdentity(
            repo_id="another/dataset",  # type: ignore[arg-type]
            revision=REVISION,
            license_id="license",
            access_terms="terms",
        )


@pytest.mark.parametrize(
    "path",
    [
        "../shard.tar",
        "/absolute/shard.tar",
        "release\\shard.tar",
        "release//shard.tar",
        "release/./shard.tar",
        "release/shard.zip",
    ],
)
def test_shard_path_is_normalized_relative_tar(path: str) -> None:
    with pytest.raises(ValidationError, match="path"):
        _record(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("bytes", 0), ("bytes", True), ("sha256", "A" * 64), ("samples", 0)],
)
def test_shard_fields_are_strict(field: str, value: object) -> None:
    payload = _record().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError, match=field):
        ShardRecord.model_validate(payload, strict=True)


def test_duplicate_or_unsorted_shards_are_rejected() -> None:
    first = _record("release-a/000001.tar", b"one")
    second = _record("release-b/000002.tar", b"two")
    aggregates = ManifestAggregates(
        shards=2,
        bytes=first.bytes + second.bytes,
        samples=first.samples + second.samples,
    )
    with pytest.raises(ValidationError, match="sorted and unique"):
        DatasetManifest(
            schema_version=1,
            source=_source(),
            shards=(second, first),
            aggregates=aggregates,
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        DatasetManifest(
            schema_version=1,
            source=_source(),
            shards=(first, first),
            aggregates=ManifestAggregates(
                shards=2, bytes=first.bytes * 2, samples=first.samples * 2
            ),
        )


def test_aggregate_mismatch_is_rejected() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["aggregates"]["samples"] += 1
    with pytest.raises(DatasetManifestError, match="invalid"):
        parse_dataset_manifest_bytes(_canonical_json(payload))


def test_duplicate_unknown_and_noncanonical_json_are_rejected() -> None:
    with pytest.raises(DatasetManifestError, match="duplicate"):
        parse_dataset_manifest_bytes(b'{"schema_version":1,"schema_version":1}\n')

    payload = _manifest().model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(DatasetManifestError, match="invalid"):
        parse_dataset_manifest_bytes(_canonical_json(payload))

    canonical = canonical_manifest_bytes(_manifest())
    with pytest.raises(DatasetManifestError, match="canonically"):
        parse_dataset_manifest_bytes(canonical[:-1] + b" \n")


def test_load_rejects_hash_or_source_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    digest = write_dataset_manifest(_manifest(), path)

    with pytest.raises(DatasetManifestError, match="SHA-256"):
        load_dataset_manifest(path, "f" * 64, _config())
    with pytest.raises(DatasetManifestError, match="source"):
        load_dataset_manifest(path, digest, _config("f" * 40))


def test_manifest_lookup_rejects_unknown_shard() -> None:
    with pytest.raises(DatasetManifestError, match="unknown"):
        _manifest().shard("release-a/missing.tar")


def test_canonical_build_inventory_round_trip_and_source_binding(tmp_path: Path) -> None:
    inventory = _build_inventory()
    payload = canonical_build_inventory_bytes(inventory)
    path = tmp_path / "build-inventory.json"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert parse_manifest_build_inventory_bytes(payload) == inventory
    assert load_manifest_build_inventory(path, digest, _config()) == inventory

    with pytest.raises(ManifestBuildInventoryError, match="SHA-256"):
        load_manifest_build_inventory(path, "f" * 64, _config())
    with pytest.raises(ManifestBuildInventoryError, match="source"):
        load_manifest_build_inventory(path, digest, _config("f" * 40))


def test_build_inventory_is_strict_sorted_unique_and_canonical() -> None:
    first = ShardBuildRecord(
        path="release-a/000001.tar", release="r1", samples=1
    )
    second = ShardBuildRecord(
        path="release-b/000002.tar", release="r2", samples=2
    )
    with pytest.raises(ValidationError, match="sorted and unique"):
        ManifestBuildInventory(
            schema_version=1,
            source=_source(),
            shards=(second, first),
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        ManifestBuildInventory(
            schema_version=1,
            source=_source(),
            shards=(first, first),
        )

    payload = canonical_build_inventory_bytes(_build_inventory())
    with pytest.raises(ManifestBuildInventoryError, match="canonical"):
        parse_manifest_build_inventory_bytes(payload[:-1] + b" \n")


def test_builder_combines_only_explicit_and_remote_facts() -> None:
    inventory = _build_inventory(release="not-derived-from-path", samples=123)
    remote = RemoteShardRecord(
        path="release-a/000001.tar",
        bytes=456,
        sha256="a" * 64,
    )

    manifest = build_dataset_manifest(inventory, (remote,))

    assert manifest.source == inventory.source
    assert manifest.shards == (
        ShardRecord(
            path=remote.path,
            release="not-derived-from-path",
            bytes=456,
            sha256="a" * 64,
            samples=123,
        ),
    )


@pytest.mark.parametrize(
    "remote",
    [
        (),
        (
            RemoteShardRecord(
                path="release-b/extra.tar", bytes=1, sha256="b" * 64
            ),
        ),
        (
            RemoteShardRecord(
                path="release-a/000001.tar", bytes=1, sha256="c" * 64
            ),
            RemoteShardRecord(
                path="release-a/000001.tar", bytes=1, sha256="c" * 64
            ),
        ),
    ],
)
def test_builder_rejects_missing_extra_or_duplicate_remote_paths(
    remote: tuple[RemoteShardRecord, ...],
) -> None:
    with pytest.raises(RemoteManifestBuildError):
        build_dataset_manifest(_build_inventory(), remote)


def test_manifest_publication_is_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"existing")

    with pytest.raises(DatasetManifestExistsError, match="already exists"):
        write_dataset_manifest(_manifest(), destination)

    assert destination.read_bytes() == b"existing"
    assert not destination.with_name(".manifest.json.tmp").exists()


def test_manifest_publication_does_not_reuse_a_stale_temporary_name(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest.json"
    stale = destination.with_name(".manifest.json.tmp")
    stale.write_bytes(b"unrelated-stale-file")

    write_dataset_manifest(_manifest(), destination)

    assert parse_dataset_manifest_bytes(destination.read_bytes()) == _manifest()
    assert stale.read_bytes() == b"unrelated-stale-file"
    assert not tuple(tmp_path.glob(".manifest.json.*.tmp"))


def test_manifest_publication_fsyncs_file_and_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_fsync = os.fsync
    synced_modes: list[int] = []

    def record_fsync(file_descriptor: int) -> None:
        synced_modes.append(os.fstat(file_descriptor).st_mode)
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    write_dataset_manifest(_manifest(), tmp_path / "nested/manifest.json")

    assert len(synced_modes) == 2
    assert stat.S_ISREG(synced_modes[0])
    assert stat.S_ISDIR(synced_modes[1])
