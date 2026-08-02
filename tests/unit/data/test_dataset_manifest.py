from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import DataSourceConfig
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetManifestExistsError,
    DatasetManifestPublicationError,
    DatasetSourceIdentity,
    ManifestAggregates,
    RemoteManifestBuildError,
    RemoteShardRecord,
    ShardRecord,
    build_dataset_manifest,
    canonical_manifest_bytes,
    load_dataset_manifest,
    manifest_sha256,
    parse_dataset_manifest_bytes,
    write_dataset_manifest,
)


def _source() -> DatasetSourceIdentity:
    return DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )


def _record(
    path: str = "data/1_2024/shard-000000.tar",
    content: bytes = b"shard-one",
) -> ShardRecord:
    return ShardRecord(
        path=path,
        bytes=len(content),
        upstream_sha256=hashlib.sha256(content).hexdigest(),
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest.from_shards(_source(), (_record(),))


def _config_source() -> DataSourceConfig:
    return cast(
        DataSourceConfig,
        SimpleNamespace(
            repo_id="leafmoone/webdataset_danbooru",
            revision="master",
        ),
    )


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def test_manifest_round_trip_write_and_load(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"

    published_id = write_dataset_manifest(manifest, path)

    assert published_id == manifest.manifest_id == manifest_sha256(manifest)
    assert parse_dataset_manifest_bytes(path.read_bytes()) == manifest
    assert load_dataset_manifest(path, _config_source()) == manifest
    assert set(json.loads(path.read_bytes())) == {
        "aggregates",
        "manifest_id",
        "schema_version",
        "shards",
        "source",
    }
    assert set(json.loads(path.read_bytes())["shards"][0]) == {
        "bytes",
        "path",
        "upstream_sha256",
    }


def test_from_shards_sorts_and_builds_stable_identity() -> None:
    first = _record("data/a.tar", b"one")
    second = _record("data/b.tar", b"two")

    forward = DatasetManifest.from_shards(_source(), (first, second))
    reverse = DatasetManifest.from_shards(_source(), (second, first))

    assert forward == reverse
    assert tuple(item.path for item in forward.shards) == (first.path, second.path)
    assert forward.aggregates == ManifestAggregates(
        shards=2,
        bytes=first.bytes + second.bytes,
    )


@pytest.mark.parametrize("revision", ["main", "a" * 40, "Master", ""])
def test_source_is_the_operational_master_branch(revision: str) -> None:
    with pytest.raises(ValidationError, match="revision"):
        DatasetSourceIdentity(
            repo_id="leafmoone/webdataset_danbooru",
            revision=revision,  # type: ignore[arg-type]
        )


def test_source_repository_is_fixed() -> None:
    with pytest.raises(ValidationError, match="repo_id"):
        DatasetSourceIdentity(
            repo_id="another/dataset",  # type: ignore[arg-type]
            revision="master",
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
    [
        ("bytes", 0),
        ("bytes", True),
        ("upstream_sha256", "A" * 64),
        ("upstream_sha256", "0" * 63),
    ],
)
def test_shard_listing_fields_are_strict(field: str, value: object) -> None:
    payload = _record().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError, match=field):
        ShardRecord.model_validate(payload, strict=True)


def test_manifest_rejects_tampered_identity_and_noncanonical_encoding() -> None:
    manifest = _manifest()
    document = manifest.model_dump(mode="json")
    document["manifest_id"] = "0" * 64
    with pytest.raises(DatasetManifestError, match="invalid"):
        parse_dataset_manifest_bytes(_canonical_json(document))

    pretty = json.dumps(manifest.model_dump(mode="json"), indent=2).encode()
    with pytest.raises(DatasetManifestError, match="canonically encoded"):
        parse_dataset_manifest_bytes(pretty)


def test_manifest_rejects_duplicate_json_keys() -> None:
    payload = canonical_manifest_bytes(_manifest())
    duplicate = payload.replace(b'"schema_version":2', b'"schema_version":2,"schema_version":2')
    with pytest.raises(DatasetManifestError, match="duplicate JSON key"):
        parse_dataset_manifest_bytes(duplicate)


def test_remote_build_uses_only_listing_facts_and_rejects_duplicates() -> None:
    remote = RemoteShardRecord(
        path="data/a.tar",
        bytes=5,
        upstream_sha256="1" * 64,
    )
    manifest = build_dataset_manifest(_source(), (remote,))
    assert manifest.shards == (
        ShardRecord(
            path=remote.path,
            bytes=remote.bytes,
            upstream_sha256=remote.upstream_sha256,
        ),
    )

    with pytest.raises(RemoteManifestBuildError, match="duplicate"):
        build_dataset_manifest(_source(), (remote, remote))
    with pytest.raises(RemoteManifestBuildError, match="no WebDataset"):
        build_dataset_manifest(_source(), ())


def test_load_rejects_source_drift_symlink_and_invalid_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_dataset_manifest(_manifest(), path)
    wrong = cast(
        DataSourceConfig,
        SimpleNamespace(repo_id="leafmoone/webdataset_danbooru", revision="main"),
    )
    with pytest.raises(DatasetManifestError, match="source config"):
        load_dataset_manifest(path, wrong)

    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(DatasetManifestError, match="could not be read"):
        load_dataset_manifest(link, _config_source())

    path.write_bytes(b"{}\n")
    with pytest.raises(DatasetManifestError, match="invalid"):
        load_dataset_manifest(path, _config_source())


def test_write_is_no_clobber_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    original = _manifest()
    write_dataset_manifest(original, path)
    with pytest.raises(DatasetManifestExistsError, match="already exists"):
        write_dataset_manifest(original, path)
    assert parse_dataset_manifest_bytes(path.read_bytes()) == original
    assert tuple(tmp_path.glob(".manifest.json.*.tmp")) == ()


def test_write_rolls_back_visible_file_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(DatasetManifestPublicationError, match="could not be written"):
        write_dataset_manifest(_manifest(), path)
    assert not path.exists()
    assert tuple(tmp_path.glob(".manifest.json.*.tmp")) == ()
