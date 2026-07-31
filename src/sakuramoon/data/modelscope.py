"""Small HTTPS client for immutable ModelScope WebDataset shards."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import quote, urlencode, urljoin, urlsplit

from sakuramoon.config import ConfigurationError, resolve_secret
from sakuramoon.config.schema import DataTransportConfig
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ManifestBuildInventory,
    RemoteShardRecord,
    ShardRecord,
    build_dataset_manifest,
    is_safe_shard_path,
)

MODELSCOPE_DATASET_HOST = "modelscope.cn"
MODELSCOPE_TOKEN_ENVIRONMENT = "MODELSCOPE_API_TOKEN"
_HTTPS_PORT = 443
_MAX_REDIRECTS = 5
_LISTING_PAGE_SIZE = 1000
_LISTING_MAX_PAGES = 100_000
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class DatasetTransportError(RuntimeError):
    """ModelScope listing or download failed."""


class DatasetAuthenticationError(DatasetTransportError):
    """ModelScope authentication failed."""


class ShardIntegrityError(DatasetTransportError):
    """A shard does not match the dataset manifest."""


class _RetryableRequestError(Exception):
    pass


@dataclass(frozen=True)
class FetchedShard:
    path: Path
    relative_path: str
    bytes: int
    sha256: str
    cache_hit: bool


@dataclass(frozen=True)
class _Target:
    host: str
    request_target: str
    authenticated: bool


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


def _repo_path(source: DatasetSourceIdentity) -> str:
    owner, name = source.repo_id.split("/", 1)
    return f"{quote(owner, safe='')}/{quote(name, safe='')}"


class ModelScopeDatasetTransport:
    """Download one shard at a time without pre-downloading the dataset."""

    def __init__(self, token: str, policy: DataTransportConfig) -> None:
        if not token or "\r" in token or "\n" in token:
            raise DatasetAuthenticationError("ModelScope token is missing or invalid")
        self._token = token
        self._policy = policy

    @classmethod
    def from_token_environment(
        cls, token_environment_name: str, policy: DataTransportConfig
    ) -> ModelScopeDatasetTransport:
        if token_environment_name != MODELSCOPE_TOKEN_ENVIRONMENT:
            raise DatasetAuthenticationError("unexpected ModelScope token variable")
        try:
            token = resolve_secret(token_environment_name).get_secret_value()
        except ConfigurationError:
            raise DatasetAuthenticationError("ModelScope token is missing") from None
        return cls(token, policy)

    def _headers(self, target: _Target) -> dict[str, str]:
        headers = {
            "Accept": "application/json, application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "SakuraMoon/1",
        }
        if target.authenticated:
            headers["Authorization"] = f"Bearer {self._token}"
            headers["Cookie"] = f"m_session_id={self._token}"
        return headers

    @property
    def stream_chunk_bytes(self) -> int:
        return self._policy.stream_chunk_bytes

    def _redirect(self, current: _Target, location: str) -> _Target:
        parsed = urlsplit(
            urljoin(f"https://{current.host}{current.request_target}", location)
        )
        host = parsed.hostname.casefold() if parsed.hostname else ""
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in (None, _HTTPS_PORT)
        ):
            raise DatasetTransportError("ModelScope returned an unsafe redirect")
        request_target = parsed.path or "/"
        if parsed.query:
            request_target += f"?{parsed.query}"
        return _Target(
            host=host,
            request_target=request_target,
            authenticated=current.authenticated and host == MODELSCOPE_DATASET_HOST,
        )

    def _open(self, initial: _Target) -> tuple[http.client.HTTPResponse, Any]:
        target = initial
        for redirects in range(_MAX_REDIRECTS + 1):
            connection: http.client.HTTPSConnection | None = None
            try:
                connection = http.client.HTTPSConnection(
                    target.host,
                    _HTTPS_PORT,
                    timeout=self._policy.connect_timeout_seconds,
                    context=ssl.create_default_context(),
                )
                connection.request("GET", target.request_target, headers=self._headers(target))
                if connection.sock is not None:
                    connection.sock.settimeout(self._policy.read_timeout_seconds)
                response = connection.getresponse()
            except (OSError, TimeoutError, http.client.HTTPException):
                if connection is not None:
                    connection.close()
                raise _RetryableRequestError from None
            if response.status not in _REDIRECT_STATUSES:
                return response, connection
            location = response.getheader("Location")
            response.close()
            connection.close()
            if location is None or redirects == _MAX_REDIRECTS:
                raise DatasetTransportError("ModelScope redirect limit exceeded")
            target = self._redirect(target, location)
        raise DatasetTransportError("ModelScope redirect limit exceeded")

    def _with_retries(self, target: _Target) -> tuple[http.client.HTTPResponse, Any]:
        for attempt in range(self._policy.max_retries + 1):
            try:
                response, connection = self._open(target)
            except _RetryableRequestError:
                if attempt == self._policy.max_retries:
                    raise DatasetTransportError("ModelScope request failed") from None
            else:
                if response.status in (401, 403):
                    response.close()
                    connection.close()
                    raise DatasetAuthenticationError("ModelScope authentication failed")
                if response.status not in _RETRYABLE_STATUSES:
                    return response, connection
                response.close()
                connection.close()
                if attempt == self._policy.max_retries:
                    raise DatasetTransportError("ModelScope retry limit exceeded")
            time.sleep(self._policy.retry_backoff_seconds)
        raise DatasetTransportError("ModelScope retry limit exceeded")

    def list_files(self, source: DatasetSourceIdentity) -> tuple[RemoteShardRecord, ...]:
        """List all WebDataset shards for a fixed dataset source revision."""

        files: list[RemoteShardRecord] = []
        for page in range(1, _LISTING_MAX_PAGES + 1):
            query = urlencode(
                {
                    "Revision": source.revision,
                    "Recursive": "True",
                    "PageNumber": page,
                    "PageSize": _LISTING_PAGE_SIZE,
                }
            )
            target = _Target(
                MODELSCOPE_DATASET_HOST,
                f"/api/v1/datasets/{_repo_path(source)}/repo/tree?{query}",
                True,
            )
            response, connection = self._with_retries(target)
            try:
                if response.status != 200:
                    raise DatasetTransportError("ModelScope listing failed")
                payload = response.read()
                document = cast(dict[str, Any], json.loads(payload))
                data = cast(dict[str, Any], document["Data"])
                entries = cast(object, data["Files"])
                if not isinstance(entries, list):
                    raise TypeError
                if not entries:
                    return tuple(files)
                for raw_entry in cast(list[object], entries):
                    if not isinstance(raw_entry, dict):
                        raise TypeError
                    entry = cast(dict[str, object], raw_entry)
                    if entry.get("Type") in {"tree", "directory"}:
                        continue
                    path = entry["Path"]
                    size = entry["Size"]
                    sha256 = entry["Sha256"]
                    if (
                        not isinstance(path, str)
                        or type(size) is not int
                        or not isinstance(sha256, str)
                    ):
                        raise TypeError
                    if not path.casefold().endswith((".tar", ".tar.gz", ".tgz")):
                        continue
                    if not is_safe_shard_path(path):
                        raise TypeError
                    files.append(
                        RemoteShardRecord(
                            path=path,
                            bytes=size,
                            sha256=sha256.lower(),
                        )
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise DatasetTransportError("ModelScope listing is invalid") from None
            finally:
                response.close()
                connection.close()
        raise DatasetTransportError("ModelScope listing exceeded configured page limit")

    def download(self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer) -> None:
        """Stream one shard body into an already-open local file."""

        query = urlencode({"Revision": manifest.source.revision, "FilePath": shard.path})
        target = _Target(
            MODELSCOPE_DATASET_HOST,
            f"/api/v1/datasets/{_repo_path(manifest.source)}/repo?{query}",
            True,
        )
        response, connection = self._with_retries(target)
        try:
            if response.status == 404:
                raise DatasetTransportError("dataset shard does not exist")
            if response.status != 200:
                raise DatasetTransportError("dataset shard download failed")
            if response.getheader("Content-Encoding") not in (None, "identity"):
                raise DatasetTransportError("compressed HTTP transfer is not supported")
            while chunk := response.read(self._policy.stream_chunk_bytes):
                output.write(chunk)
        except (OSError, TimeoutError, http.client.HTTPException):
            raise DatasetTransportError("dataset shard transfer failed") from None
        finally:
            response.close()
            connection.close()


def validate_remote_manifest(
    transport: ModelScopeDatasetTransport, manifest: DatasetManifest
) -> None:
    """Require exact path, byte count and SHA equality with remote listing."""

    expected = {(item.path, item.bytes, item.sha256) for item in manifest.shards}
    remote_files = transport.list_files(manifest.source)
    observed = {(item.path, item.bytes, item.sha256) for item in remote_files}
    if len(remote_files) != len(manifest.shards) or observed != expected:
        raise ShardIntegrityError("remote dataset listing differs from manifest")


def build_remote_dataset_manifest(
    transport: ModelScopeDatasetTransport,
    inventory: ManifestBuildInventory,
) -> DatasetManifest:
    """Combine remote immutable file facts with explicit release/sample facts."""

    return build_dataset_manifest(inventory, transport.list_files(inventory.source))


def _verify_existing(path: Path, shard: ShardRecord, chunk_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size != shard.bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest() == shard.sha256


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fetch_dataset_shard(
    transport: ModelScopeDatasetTransport,
    manifest: DatasetManifest,
    shard_path: str,
    cache_root: Path,
) -> FetchedShard:
    """Stream, verify and atomically publish one requested shard."""

    shard = manifest.shard(shard_path)
    destination = cache_root / shard.path
    partial = destination.with_name(f"{destination.name}.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verify_existing(destination, shard, transport.stream_chunk_bytes):
        return FetchedShard(destination, shard.path, shard.bytes, shard.sha256, True)
    if destination.exists():
        raise ShardIntegrityError("cached shard differs from manifest")

    digest = hashlib.sha256()
    written = 0

    class _DigestWriter:
        def write(self, payload: bytes) -> int:
            nonlocal written
            written += len(payload)
            if written > shard.bytes:
                raise ShardIntegrityError("downloaded shard is larger than manifest")
            digest.update(payload)
            return handle.write(payload)

    published = False
    try:
        with partial.open("wb") as handle:
            transport.download(manifest, shard, _DigestWriter())
            handle.flush()
            os.fsync(handle.fileno())
        if written != shard.bytes or digest.hexdigest() != shard.sha256:
            raise ShardIntegrityError("downloaded shard differs from manifest")
        os.replace(partial, destination)
        published = True
        _fsync_directory(destination.parent)
    except Exception:
        cleanup_error: OSError | None = None
        if published:
            try:
                destination.unlink(missing_ok=True)
                _fsync_directory(destination.parent)
            except OSError as exc:
                cleanup_error = exc
        try:
            partial.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise DatasetTransportError(
                "dataset shard publication rollback failed"
            ) from None
        raise
    return FetchedShard(destination, shard.path, shard.bytes, shard.sha256, False)
