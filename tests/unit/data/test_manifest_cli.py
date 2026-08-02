from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import sakuramoon.cli.manifest as cli_manifest
from sakuramoon.config.schema import DataSourceConfig, DataTransportConfig
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    RemoteShardRecord,
    ShardRecord,
    write_dataset_manifest,
)
from sakuramoon.data.modelscope import ModelScopeDatasetTransport

CONTENT = b"synthetic-cli-shard"
SHARD_PATH = "data/1_2024/shard-000000.tar"


def _source() -> DatasetSourceIdentity:
    return DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )


def _config_source() -> DataSourceConfig:
    return cast(
        DataSourceConfig,
        SimpleNamespace(
            repo_id="leafmoone/webdataset_danbooru",
            revision="master",
        ),
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest.from_shards(
        _source(),
        (
            ShardRecord(
                path=SHARD_PATH,
                bytes=len(CONTENT),
                upstream_sha256=hashlib.sha256(CONTENT).hexdigest(),
            ),
        ),
    )


def _transport_policy() -> DataTransportConfig:
    return DataTransportConfig(
        connect_timeout_seconds=1.0,
        read_timeout_seconds=2.0,
        max_retries=0,
        retry_backoff_seconds=0.0,
        stream_chunk_bytes=65536,
    )


def _config(manifest_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        security=SimpleNamespace(modelscope_token_env="MODELSCOPE_API_TOKEN"),
        data=SimpleNamespace(
            source=_config_source(),
            manifest=SimpleNamespace(
                path=manifest_path,
                initialize_if_missing=True,
                refresh_existing=False,
            ),
            transport=_transport_policy(),
        ),
    )


class _ListingTransport:
    def __init__(self, records: tuple[RemoteShardRecord, ...]) -> None:
        self.records = records
        self.list_calls = 0

    def list_files(self, source: DatasetSourceIdentity) -> tuple[RemoteShardRecord, ...]:
        assert source == _source()
        self.list_calls += 1
        return self.records


def _remote(content: bytes = CONTENT) -> tuple[RemoteShardRecord, ...]:
    return (
        RemoteShardRecord(
            path=SHARD_PATH,
            bytes=len(content),
            upstream_sha256=hashlib.sha256(content).hexdigest(),
        ),
    )


def _install_config(
    monkeypatch: pytest.MonkeyPatch, manifest_path: str
) -> None:
    loaded = SimpleNamespace(config=_config(manifest_path))

    def load(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return loaded

    monkeypatch.setattr(cli_manifest, "load_config", load)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, transport: _ListingTransport
) -> None:
    def from_environment(
        *_args: object, **_kwargs: object
    ) -> ModelScopeDatasetTransport:
        return cast(ModelScopeDatasetTransport, transport)

    monkeypatch.setattr(
        cli_manifest.ModelScopeDatasetTransport,
        "from_token_environment",
        from_environment,
    )


def _args(workspace: Path, mode: str) -> list[str]:
    return [
        "--config",
        "run.toml",
        "--config-root",
        str(workspace),
        "--root",
        str(workspace),
        "--mode",
        mode,
    ]


def test_local_reports_internal_manifest_id_without_user_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "locks/manifest.json"
    manifest = _manifest()
    write_dataset_manifest(manifest, path)
    _install_config(monkeypatch, "locks/manifest.json")

    assert cli_manifest.main(_args(tmp_path, "local")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "bytes": len(CONTENT),
        "manifest_id": manifest.manifest_id,
        "mode": "local",
        "ok": True,
        "repo_id": "leafmoone/webdataset_danbooru",
        "revision": "master",
        "shards": 1,
    }


def test_initialize_builds_absent_manifest_from_remote_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "locks/manifest.json"
    _install_config(monkeypatch, "locks/manifest.json")
    _install_transport(monkeypatch, _ListingTransport(_remote()))

    assert cli_manifest.main(_args(tmp_path, "initialize")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_id"] == _manifest().manifest_id
    assert path.is_file()


def test_initialize_does_not_replace_existing_drifted_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "locks/manifest.json"
    manifest = _manifest()
    write_dataset_manifest(manifest, path)
    before = path.read_bytes()
    _install_config(monkeypatch, "locks/manifest.json")
    transport = _ListingTransport(_remote(b"changed"))
    _install_transport(monkeypatch, transport)

    assert cli_manifest.main(_args(tmp_path, "initialize")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_id"] == manifest.manifest_id
    assert payload["ok"] is True
    assert transport.list_calls == 0
    assert path.read_bytes() == before


def test_remote_mode_requires_existing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_config(monkeypatch, "locks/manifest.json")
    _install_transport(monkeypatch, _ListingTransport(_remote()))
    assert cli_manifest.main(_args(tmp_path, "remote")) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "manifest_invalid"


@pytest.mark.parametrize(
    "extra",
    [
        ["--inventory", "inventory.json"],
        ["--inventory-sha256", "0" * 64],
        ["--output", "manifest.json"],
        ["--mode", "build"],
    ],
)
def test_removed_inventory_and_user_sha_arguments_are_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
) -> None:
    args = _args(tmp_path, "local") + extra
    assert cli_manifest.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "invalid_arguments",
        "ok": False,
    }


@pytest.mark.parametrize("configured", ["../manifest.json", "/tmp/manifest.json", "."])
def test_manifest_path_must_stay_inside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    configured: str,
) -> None:
    _install_config(monkeypatch, configured)
    assert cli_manifest.main(_args(tmp_path, "local")) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "manifest_path_invalid"


def test_unapproved_credential_variable_fails_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config("manifest.json")
    config.security.modelscope_token_env = "OTHER_TOKEN"

    def load(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(config=config)

    monkeypatch.setattr(
        cli_manifest,
        "load_config",
        load,
    )
    assert cli_manifest.main(_args(tmp_path, "initialize")) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "configuration_invalid"
