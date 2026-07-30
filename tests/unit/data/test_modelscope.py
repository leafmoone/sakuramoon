from __future__ import annotations

import hashlib
import http.client
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import pytest

from sakuramoon.config.schema import DataTransportConfig
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.modelscope import (
    MODELSCOPE_DATASET_HOST,
    MODELSCOPE_TOKEN_ENVIRONMENT,
    DatasetAuthenticationError,
    DatasetTransportError,
    FetchedShard,
    ModelScopeDatasetTransport,
    ShardIntegrityError,
    fetch_dataset_shard,
    validate_remote_manifest,
)

CONTENT = b"synthetic-webdataset-shard"
REVISION = "0123456789abcdef0123456789abcdef01234567"
SHARD_PATH = "release-a/000001.tar"


def _manifest(content: bytes = CONTENT) -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision=REVISION,
        license_id="source-license",
        access_terms="source-access-terms",
    )
    shard = ShardRecord(
        path=SHARD_PATH,
        release="release-a",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        samples=13,
    )
    return DatasetManifest.from_shards(source, (shard,))


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
        self.closed = False

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)

    def read(self, amount: int | None = None) -> bytes:
        size = len(self._body) if amount is None else amount
        result = self._body[self._position : self._position + size]
        self._position += len(result)
        return result

    def close(self) -> None:
        self.closed = True


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


def _install(monkeypatch: pytest.MonkeyPatch, plans: list[_Plan]) -> type[_Connection]:
    _Connection.plans = list(plans)
    _Connection.requests = []
    monkeypatch.setattr(http.client, "HTTPSConnection", _Connection)
    return _Connection


def _transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: DataTransportConfig | None = None,
) -> ModelScopeDatasetTransport:
    monkeypatch.setenv(MODELSCOPE_TOKEN_ENVIRONMENT, "synthetic-token")
    return ModelScopeDatasetTransport.from_token_environment(
        MODELSCOPE_TOKEN_ENVIRONMENT, policy or _policy()
    )


def _listing(entries: list[dict[str, object]]) -> _Response:
    body = json.dumps({"Data": {"Files": entries}}).encode()
    return _Response(200, body)


def _entry(
    *, size: int = len(CONTENT), sha256: str | None = None
) -> dict[str, object]:
    return {
        "Path": SHARD_PATH,
        "Size": size,
        "Sha256": sha256 or hashlib.sha256(CONTENT).hexdigest(),
        "Type": "blob",
    }


def test_factory_requires_fixed_token_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODELSCOPE_TOKEN_ENVIRONMENT, "token")
    with pytest.raises(DatasetAuthenticationError, match="unexpected"):
        ModelScopeDatasetTransport.from_token_environment("OTHER_TOKEN", _policy())


def test_listing_uses_fixed_repo_revision_and_paginates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http = _install(monkeypatch, [_Plan(_listing([_entry()])), _Plan(_listing([]))])

    files = _transport(monkeypatch).list_files(_manifest())

    assert len(files) == 1
    assert len(http.requests) == 2
    first = http.requests[0]
    assert first["host"] == MODELSCOPE_DATASET_HOST
    assert first["method"] == "GET"
    assert REVISION in cast(str, first["target"])
    assert "PageNumber=1" in cast(str, first["target"])
    headers = cast(dict[str, str], first["headers"])
    assert headers["Authorization"] == "Bearer synthetic-token"


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [_entry(size=len(CONTENT) + 1)],
        [_entry(sha256="f" * 64)],
        [_entry(), _entry()],
    ],
)
def test_remote_listing_must_match_manifest(
    entries: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, [_Plan(_listing(entries)), _Plan(_listing([]))])
    with pytest.raises(ShardIntegrityError, match="differs"):
        validate_remote_manifest(_transport(monkeypatch), _manifest())


def test_remote_listing_match(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata: dict[str, object] = {
        "Path": "README.md",
        "Size": 1,
        "Sha256": "0" * 64,
        "Type": "blob",
    }
    _install(
        monkeypatch,
        [_Plan(_listing([_entry(), metadata])), _Plan(_listing([]))],
    )
    validate_remote_manifest(_transport(monkeypatch), _manifest())


def test_fetch_streams_to_partial_and_atomically_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http = _install(
        monkeypatch,
        [_Plan(_Response(200, CONTENT, {"Content-Length": str(len(CONTENT))}))],
    )

    result = fetch_dataset_shard(
        _transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path
    )

    assert result == FetchedShard(
        tmp_path / SHARD_PATH,
        SHARD_PATH,
        len(CONTENT),
        hashlib.sha256(CONTENT).hexdigest(),
        False,
    )
    assert result.path.read_bytes() == CONTENT
    assert not result.path.with_name("000001.tar.partial").exists()
    assert "FilePath=release-a%2F000001.tar" in cast(str, http.requests[0]["target"])


def test_valid_cache_hit_does_not_request_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / SHARD_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(CONTENT)
    http = _install(monkeypatch, [])

    result = fetch_dataset_shard(
        _transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path
    )

    assert result.cache_hit is True
    assert http.requests == []


def test_corrupt_download_is_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, [_Plan(_Response(200, b"wrong"))])

    with pytest.raises(ShardIntegrityError, match="differs"):
        fetch_dataset_shard(_transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path)

    assert not (tmp_path / SHARD_PATH).exists()
    assert not (tmp_path / SHARD_PATH).with_name("000001.tar.partial").exists()


def test_existing_corrupt_cache_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / SHARD_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wrong")
    _install(monkeypatch, [])

    with pytest.raises(ShardIntegrityError, match="cached"):
        fetch_dataset_shard(_transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path)

    assert target.read_bytes() == b"wrong"


def test_retryable_status_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http = _install(
        monkeypatch,
        [_Plan(_Response(503)), _Plan(_Response(200, CONTENT))],
    )
    fetch_dataset_shard(
        _transport(monkeypatch, policy=_policy(max_retries=1)),
        _manifest(),
        SHARD_PATH,
        tmp_path,
    )
    assert len(http.requests) == 2


def test_authentication_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http = _install(monkeypatch, [_Plan(_Response(401))])
    with pytest.raises(DatasetAuthenticationError):
        fetch_dataset_shard(_transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path)
    assert len(http.requests) == 1


def test_cross_host_redirect_drops_authentication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http = _install(
        monkeypatch,
        [
            _Plan(_Response(302, headers={"Location": "https://cdn.modelscope.cn/blob"})),
            _Plan(_Response(200, CONTENT)),
        ],
    )
    fetch_dataset_shard(_transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path)
    redirected = cast(dict[str, str], http.requests[1]["headers"])
    assert "Authorization" not in redirected
    assert "Cookie" not in redirected


def test_non_https_redirect_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(
        monkeypatch,
        [_Plan(_Response(302, headers={"Location": "http://example.com/blob"}))],
    )
    with pytest.raises(DatasetTransportError, match="unsafe redirect"):
        fetch_dataset_shard(_transport(monkeypatch), _manifest(), SHARD_PATH, tmp_path)
