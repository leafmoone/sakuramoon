from __future__ import annotations

import hashlib
import http.client
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Protocol, cast

import pytest

from sakuramoon.config.schema import DataSourceConfig, DataTransportConfig
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetSourceIdentity,
    RemoteShardRecord,
    ShardRecord,
    load_dataset_manifest,
)
from sakuramoon.data.modelscope import (
    MODELSCOPE_TOKEN_ENVIRONMENT,
    DatasetAuthenticationError,
    DatasetTransportError,
    ModelScopeDatasetTransport,
    ShardIntegrityError,
    build_remote_dataset_manifest,
    ensure_dataset_manifest,
    fetch_dataset_shard,
    validate_remote_manifest,
)

CONTENT = b"synthetic-webdataset-shard"
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


def _manifest(content: bytes = CONTENT) -> DatasetManifest:
    return DatasetManifest.from_shards(
        _source(),
        (
            ShardRecord(
                path=SHARD_PATH,
                bytes=len(content),
                upstream_sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
    )


def _policy(**changes: object) -> DataTransportConfig:
    values: dict[str, object] = {
        "connect_timeout_seconds": 1.0,
        "read_timeout_seconds": 2.0,
        "max_retries": 2,
        "retry_backoff_seconds": 0.0,
        "stream_chunk_bytes": 65536,
    }
    values.update(changes)
    return DataTransportConfig.model_validate(values, strict=True)


class _Socket:
    def settimeout(self, value: float) -> None:
        self.timeout = value


class _Response:
    def __init__(
        self, status: int, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self._body = body
        self._headers = headers or {}
        self._position = 0

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)

    def read(self, amount: int | None = None) -> bytes:
        size = len(self._body) if amount is None else amount
        result = self._body[self._position : self._position + size]
        self._position += len(result)
        return result

    def close(self) -> None:
        pass


@dataclass
class _Plan:
    response: _Response | None = None
    error: Exception | None = None


class _Connection:
    plans: ClassVar[list[_Plan]] = []
    requests: ClassVar[list[dict[str, object]]] = []

    def __init__(
        self, host: str, port: int, timeout: float, context: object
    ) -> None:
        del context
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = _Socket()
        self.plan = self.plans.pop(0)

    def request(
        self,
        method: str,
        target: str,
        body: object | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: object,
    ) -> None:
        del body, kwargs
        self.requests.append(
            {"host": self.host, "method": method, "target": target, "headers": headers}
        )
        if self.plan.error is not None:
            raise self.plan.error

    def getresponse(self) -> _Response:
        if self.plan.response is None:
            raise AssertionError("response plan is missing")
        return self.plan.response

    def close(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, plans: list[_Plan]) -> None:
    _Connection.plans = list(plans)
    _Connection.requests = []
    monkeypatch.setattr(http.client, "HTTPSConnection", _Connection)


def _transport(monkeypatch: pytest.MonkeyPatch) -> ModelScopeDatasetTransport:
    monkeypatch.setenv(MODELSCOPE_TOKEN_ENVIRONMENT, "synthetic-token")
    return ModelScopeDatasetTransport.from_token_environment(
        MODELSCOPE_TOKEN_ENVIRONMENT, _policy()
    )


def _listing(entries: list[dict[str, object]]) -> _Response:
    return _Response(200, json.dumps({"Data": {"Files": entries}}).encode())


def _entry(
    *, path: str = SHARD_PATH, size: int = len(CONTENT), digest: str | None = None
) -> dict[str, object]:
    return {
        "Path": path,
        "Size": size,
        "Sha256": digest or hashlib.sha256(CONTENT).hexdigest(),
        "Type": "blob",
    }


def test_factory_requires_fixed_token_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODELSCOPE_TOKEN_ENVIRONMENT, "token")
    with pytest.raises(DatasetAuthenticationError, match="unexpected"):
        ModelScopeDatasetTransport.from_token_environment("OTHER_TOKEN", _policy())


def test_listing_uses_master_and_maps_only_operational_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        [
            _Plan(
                _listing(
                    [
                        _entry(),
                        {"Path": "README.md", "Size": 1, "Sha256": "0" * 64, "Type": "blob"},
                        {"Path": "data", "Size": 0, "Sha256": "", "Type": "tree"},
                    ]
                )
            ),
            _Plan(_listing([])),
        ],
    )
    records = _transport(monkeypatch).list_files(_source())
    assert records == (
        RemoteShardRecord(
            path=SHARD_PATH,
            bytes=len(CONTENT),
            upstream_sha256=hashlib.sha256(CONTENT).hexdigest(),
        ),
    )
    assert "Revision=master" in cast(str, _Connection.requests[0]["target"])


@pytest.mark.parametrize(
    "entry",
    [
        {"Path": SHARD_PATH, "Size": 0, "Sha256": "0" * 64, "Type": "blob"},
        {"Path": "../bad.tar", "Size": 1, "Sha256": "0" * 64, "Type": "blob"},
        {"Path": SHARD_PATH, "Size": 1, "Sha256": "bad", "Type": "blob"},
    ],
)
def test_listing_rejects_invalid_upstream_facts(
    monkeypatch: pytest.MonkeyPatch, entry: dict[str, object]
) -> None:
    _install(monkeypatch, [_Plan(_listing([entry]))])
    with pytest.raises(DatasetTransportError, match="listing is invalid"):
        _transport(monkeypatch).list_files(_source())


class _ListingTransport:
    stream_chunk_bytes = 65536

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


def test_build_and_remote_validation_require_exact_listing() -> None:
    transport = cast(ModelScopeDatasetTransport, _ListingTransport(_remote()))
    manifest = build_remote_dataset_manifest(transport, _source())
    assert manifest == _manifest()
    validate_remote_manifest(transport, manifest)

    duplicate = cast(
        ModelScopeDatasetTransport,
        _ListingTransport((_remote()[0], _remote()[0])),
    )
    with pytest.raises(ShardIntegrityError, match="differs"):
        validate_remote_manifest(duplicate, manifest)


def test_ensure_initializes_once_then_loads_snapshot_without_relisting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    initial = _ListingTransport(_remote())
    manifest = ensure_dataset_manifest(
        cast(ModelScopeDatasetTransport, initial),
        path,
        _config_source(),
        initialize_if_missing=True,
        refresh_existing=False,
    )
    assert initial.list_calls == 1
    assert load_dataset_manifest(path, _config_source()) == manifest
    published = path.read_bytes()

    existing = _ListingTransport(_remote(b"changed-upstream-after-snapshot"))
    loaded = ensure_dataset_manifest(
        cast(ModelScopeDatasetTransport, existing),
        path,
        _config_source(),
        initialize_if_missing=True,
        refresh_existing=False,
    )
    assert loaded == manifest
    assert existing.list_calls == 0
    assert path.read_bytes() == published


def test_ensure_fails_closed_for_invalid_manifest_policy(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(DatasetManifestError, match="initialization is disabled"):
        ensure_dataset_manifest(
            cast(ModelScopeDatasetTransport, _ListingTransport(_remote())),
            path,
            _config_source(),
            initialize_if_missing=False,
            refresh_existing=False,
        )
    with pytest.raises(DatasetManifestError, match="refresh is prohibited"):
        ensure_dataset_manifest(
            cast(ModelScopeDatasetTransport, _ListingTransport(_remote())),
            path,
            _config_source(),
            initialize_if_missing=True,
            refresh_existing=True,
        )

class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _MemoryTransport:
    stream_chunk_bytes = 4

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.downloads = 0

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        assert manifest.shard(shard.path) == shard
        self.downloads += 1
        output.write(self.content)


def test_fetch_verifies_internal_digest_and_reuses_verified_cache(tmp_path: Path) -> None:
    transport = _MemoryTransport(CONTENT)
    first = fetch_dataset_shard(
        cast(ModelScopeDatasetTransport, transport), _manifest(), SHARD_PATH, tmp_path
    )
    second = fetch_dataset_shard(
        cast(ModelScopeDatasetTransport, transport), _manifest(), SHARD_PATH, tmp_path
    )
    assert first.path.read_bytes() == CONTENT
    assert not first.cache_hit and second.cache_hit
    assert transport.downloads == 1


@pytest.mark.parametrize("content", [b"wrong-size", b"x" * len(CONTENT)])
def test_fetch_rejects_corrupt_download_and_cleans_partial(
    tmp_path: Path, content: bytes
) -> None:
    with pytest.raises(ShardIntegrityError, match="differs"):
        fetch_dataset_shard(
            cast(ModelScopeDatasetTransport, _MemoryTransport(content)),
            _manifest(),
            SHARD_PATH,
            tmp_path,
        )
    destination = tmp_path / SHARD_PATH
    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.partial").exists()
