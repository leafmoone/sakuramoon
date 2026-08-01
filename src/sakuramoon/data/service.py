"""Process-owned dataset cache, persistent mainset, and bounded local IPC."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import socket
import stat
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

from sakuramoon.data.cache import CachedShard, ShardCache, ShardCacheError
from sakuramoon.data.manifest import DatasetManifest, manifest_sha256
from sakuramoon.data.modelscope import DatasetTransportError
from sakuramoon.data.service_protocol import (
    MAX_SERVICE_FRAME_BYTES,
    SERVICE_PROTOCOL_VERSION,
    DataServiceProtocolError,
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
    canonical_json_bytes,
    parse_frame,
)

_MAINSET_SCHEMA_VERSION = 1
_MAINSET_STATUSES = frozenset({"pending", "active", "completed"})


class DataServiceError(RuntimeError):
    """The data service cannot preserve its ownership or ordering contract."""


class DataServiceStateCommittedError(DataServiceError):
    """The mainset committed, but publication cleanup could not be confirmed."""


@dataclass(frozen=True, slots=True)
class DataServiceLimits:
    download_concurrency: int
    verified_shard_lookahead: int
    lease_channel_capacity: int
    ack_channel_capacity: int

    def __post_init__(self) -> None:
        values = (
            self.download_concurrency,
            self.verified_shard_lookahead,
            self.lease_channel_capacity,
            self.ack_channel_capacity,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("data service limits must be positive integers")


@dataclass(frozen=True, slots=True)
class DataServiceStats:
    in_flight_downloads: int
    verified_ready_shards: int
    outstanding_leases: int


@dataclass(frozen=True, slots=True)
class MainsetRow:
    ordinal: int
    path: str
    status: Literal["pending", "active", "completed"]

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.path) is not str
            or not self.path
            or self.status not in _MAINSET_STATUSES
        ):
            raise ValueError("mainset row is invalid")


@dataclass(frozen=True, slots=True)
class PersistentMainset:
    mainset_id: str
    manifest_sha256: str
    shuffle_identity: str
    worker_count: int
    rows: tuple[MainsetRow, ...]
    replayed_shards: int = 0
    replayed_samples: int = 0

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.mainset_id)
            or not _is_sha256(self.manifest_sha256)
            or not _is_sha256(self.shuffle_identity)
            or type(self.worker_count) is not int
            or self.worker_count <= 0
            or type(self.rows) is not tuple
            or not self.rows
            or type(self.replayed_shards) is not int
            or self.replayed_shards < 0
            or type(self.replayed_samples) is not int
            or self.replayed_samples < 0
        ):
            raise ValueError("persistent mainset is invalid")

    @property
    def active_paths(self) -> tuple[str, ...]:
        return tuple(row.path for row in self.rows if row.status == "active")

    @property
    def completed_paths(self) -> tuple[str, ...]:
        return tuple(row.path for row in self.rows if row.status == "completed")


@dataclass(frozen=True, slots=True)
class _OutstandingLease:
    descriptor: ShardLeaseDescriptor


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mainset_payload(state: PersistentMainset) -> dict[str, object]:
    return {
        "mainset_id": state.mainset_id,
        "manifest_sha256": state.manifest_sha256,
        "replayed_samples": state.replayed_samples,
        "replayed_shards": state.replayed_shards,
        "rows": [
            {"ordinal": row.ordinal, "path": row.path, "status": row.status}
            for row in state.rows
        ],
        "schema_version": _MAINSET_SCHEMA_VERSION,
        "shuffle_identity": state.shuffle_identity,
        "worker_count": state.worker_count,
    }


def _mainset_bytes(state: PersistentMainset) -> bytes:
    return canonical_json_bytes(_mainset_payload(state))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _MainsetStore:
    """Atomically persist the one service-owned full-manifest table."""

    def __init__(
        self, path: Path, manifest: DatasetManifest, *, worker_count: int
    ) -> None:
        if type(worker_count) is not int or worker_count <= 0:
            raise ValueError("mainset worker_count must be a positive integer")
        self.path = path
        self.manifest = manifest
        self.worker_count = worker_count
        self.manifest_sha256 = manifest_sha256(manifest)
        self._paths = frozenset(record.path for record in manifest.shards)

    def validate(self, state: PersistentMainset) -> None:
        paths = tuple(row.path for row in state.rows)
        active = tuple(row.path for row in state.rows if row.status == "active")
        first_pending = next(
            (
                row.ordinal
                for row in state.rows
                if row.status == "pending"
            ),
            len(state.rows),
        )
        if (
            state.manifest_sha256 != self.manifest_sha256
            or state.worker_count != self.worker_count
            or len(state.rows) != len(self.manifest.shards)
            or tuple(row.ordinal for row in state.rows) != tuple(range(len(state.rows)))
            or len(set(paths)) != len(paths)
            or frozenset(paths) != self._paths
            or len(active) > self.worker_count
            or any(
                row.status != "pending"
                for row in state.rows[first_pending:]
            )
        ):
            raise DataServiceError("persistent mainset identity or rows are invalid")

    def load(self) -> PersistentMainset | None:
        if self.path.is_symlink():
            raise DataServiceError("persistent mainset must not be a symlink")
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_bytes())
            if not isinstance(raw, dict):
                raise TypeError
            document = cast(dict[str, object], raw)
            if set(document) != {
                "mainset_id",
                "manifest_sha256",
                "replayed_samples",
                "replayed_shards",
                "rows",
                "schema_version",
                "shuffle_identity",
                "worker_count",
            } or document["schema_version"] != _MAINSET_SCHEMA_VERSION:
                raise ValueError
            raw_rows = document["rows"]
            if not isinstance(raw_rows, list):
                raise TypeError
            rows: list[MainsetRow] = []
            for raw_row in cast(list[object], raw_rows):
                if not isinstance(raw_row, dict):
                    raise TypeError
                row = cast(dict[str, object], raw_row)
                if set(row) != {"ordinal", "path", "status"}:
                    raise ValueError
                rows.append(
                    MainsetRow(
                        ordinal=cast(int, row["ordinal"]),
                        path=cast(str, row["path"]),
                        status=cast(
                            Literal["pending", "active", "completed"],
                            row["status"],
                        ),
                    )
                )
            state = PersistentMainset(
                mainset_id=cast(str, document["mainset_id"]),
                manifest_sha256=cast(str, document["manifest_sha256"]),
                shuffle_identity=cast(str, document["shuffle_identity"]),
                worker_count=cast(int, document["worker_count"]),
                rows=tuple(rows),
                replayed_shards=cast(int, document["replayed_shards"]),
                replayed_samples=cast(int, document["replayed_samples"]),
            )
            self.validate(state)
            return state
        except DataServiceError:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            raise DataServiceError("persistent mainset is invalid") from None

    def cleanup_residue(self) -> None:
        patterns = (
            f".{self.path.name}.*.tmp",
            f".{self.path.name}.*.rollback",
        )
        removed = False
        try:
            for pattern in patterns:
                for residue in self.path.parent.glob(pattern):
                    if residue.is_symlink() or not residue.is_file():
                        raise DataServiceError(
                            "persistent mainset residue is invalid"
                        )
                    residue.unlink()
                    removed = True
            if removed:
                _fsync_directory(self.path.parent)
        except OSError:
            raise DataServiceError(
                "persistent mainset residue cleanup failed"
            ) from None

    def save(self, state: PersistentMainset) -> None:
        self.validate(state)
        if self.path.is_symlink():
            raise DataServiceError("persistent mainset must not be a symlink")
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        rollback = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.rollback"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rollback_linked = False
        published = False
        try:
            with temporary.open("xb") as handle:
                handle.write(_mainset_bytes(state))
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                os.link(self.path, rollback)
                rollback_linked = True
                _fsync_directory(self.path.parent)
            os.replace(temporary, self.path)
            published = True
            _fsync_directory(self.path.parent)
        except OSError:
            rollback_error: OSError | None = None
            if published:
                try:
                    if rollback_linked:
                        os.replace(rollback, self.path)
                        rollback_linked = False
                    else:
                        self.path.unlink(missing_ok=True)
                    _fsync_directory(self.path.parent)
                except OSError as exc:
                    rollback_error = exc
            for residue in (temporary, rollback if rollback_linked else None):
                if residue is None:
                    continue
                try:
                    residue.unlink(missing_ok=True)
                except OSError as exc:
                    rollback_error = rollback_error or exc
            if rollback_error is not None:
                raise DataServiceError("persistent mainset rollback failed") from None
            raise DataServiceError("persistent mainset publication failed") from None
        if rollback_linked:
            try:
                rollback.unlink()
                _fsync_directory(self.path.parent)
            except OSError:
                raise DataServiceStateCommittedError(
                    "persistent mainset committed but cleanup failed"
                ) from None

    def new(
        self,
        *,
        previous_id: str | None = None,
        replayed_shards: int = 0,
        replayed_samples: int = 0,
    ) -> PersistentMainset:
        paths = [record.path for record in self.manifest.shards]
        secrets.SystemRandom().shuffle(paths)
        mainset_id = secrets.token_hex(32)
        while mainset_id == previous_id:
            mainset_id = secrets.token_hex(32)
        state = PersistentMainset(
            mainset_id=mainset_id,
            manifest_sha256=self.manifest_sha256,
            shuffle_identity=secrets.token_hex(32),
            worker_count=self.worker_count,
            rows=tuple(
                MainsetRow(ordinal, path, "pending")
                for ordinal, path in enumerate(paths)
            ),
            replayed_shards=replayed_shards,
            replayed_samples=replayed_samples,
        )
        self.validate(state)
        return state


class DataSupplyService:
    """Own cache and a persistent full-manifest order while issuing local leases."""

    def __init__(
        self,
        manifest: DatasetManifest,
        cache: ShardCache,
        mainset_path: Path,
        ownership_lock_path: Path,
        identity: DataServiceSessionIdentity,
        limits: DataServiceLimits,
    ) -> None:
        if manifest_sha256(manifest) != identity.manifest_sha256:
            raise DataServiceError("service manifest identity differs from the client")
        if limits.lease_channel_capacity < identity.worker_count:
            raise DataServiceError("lease capacity is smaller than worker topology")
        if limits.ack_channel_capacity < identity.worker_count:
            raise DataServiceError("ACK capacity is smaller than worker topology")
        if limits.verified_shard_lookahead < identity.worker_count:
            raise DataServiceError("verified lookahead is smaller than worker topology")
        if not ownership_lock_path.is_absolute():
            raise DataServiceError("service ownership lock path must be absolute")
        try:
            ownership_lock_path.resolve(strict=False).relative_to(
                cache.root.resolve(strict=False)
            )
        except ValueError:
            pass
        else:
            raise DataServiceError("service ownership lock must be outside shared cache")
        self.manifest = manifest
        self.cache = cache
        self.identity = identity
        self.limits = limits
        self.store = _MainsetStore(
            mainset_path, manifest, worker_count=identity.worker_count
        )
        self.ownership_lock_path = ownership_lock_path
        self._mainset: PersistentMainset | None = None
        self._recovery_pending = set[str]()
        self._executor = ThreadPoolExecutor(
            max_workers=limits.download_concurrency,
            thread_name_prefix="sakuramoon-data-fetch",
        )
        self._futures: dict[str, Future[CachedShard]] = {}
        self._ready: dict[str, CachedShard] = {}
        self._outstanding: dict[str, _OutstandingLease] = {}
        self._worker_leases: dict[int, str] = {}
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self._ownership_handle: BinaryIO | None = None

    @property
    def mainset(self) -> PersistentMainset:
        if self._mainset is None:
            raise DataServiceError("persistent mainset is not loaded")
        return self._mainset

    @property
    def recovery_pending(self) -> frozenset[str]:
        return frozenset(self._recovery_pending)

    def start(self) -> None:
        with self._lock:
            if self._started or self._closed:
                raise DataServiceError("data service lifecycle is invalid")
            self.store.path.parent.mkdir(parents=True, exist_ok=True)
            self.cache.root.mkdir(parents=True, exist_ok=True)
            ownership: BinaryIO | None = None
            try:
                self.ownership_lock_path.parent.mkdir(parents=True, exist_ok=True)
                if self.ownership_lock_path.is_symlink():
                    raise OSError
                descriptor = os.open(
                    self.ownership_lock_path,
                    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                    0o600,
                )
                ownership = os.fdopen(descriptor, "a+b")
                fcntl.flock(ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if ownership is not None:
                    ownership.close()
                raise DataServiceError(
                    "another data service owns the shared cache"
                ) from None
            self._ownership_handle = ownership
            self.store.cleanup_residue()
            self.cache.cleanup_manifest_partials()
            state = self.store.load()
            if state is None:
                state = self.store.new()
                self.store.save(state)
            recovered = state.active_paths
            if recovered:
                replayed_samples = sum(
                    self.manifest.shard(path).samples for path in recovered
                )
                state = replace(
                    state,
                    replayed_shards=state.replayed_shards + len(recovered),
                    replayed_samples=state.replayed_samples + replayed_samples,
                )
                self.store.save(state)
            self._mainset = state
            self._recovery_pending = set(recovered)
            protected = frozenset(recovered)
            for path in recovered:
                cached = self.cache.fetch(path, protected_paths=protected)
                self._ready[path] = cached
                self._recovery_pending.remove(path)
            self._started = True
            self._schedule_lookahead()

    def _ensure_running(self) -> None:
        if not self._started or self._closed:
            raise DataServiceError("data service is not running")

    def _protected_paths(self) -> frozenset[str]:
        return frozenset(
            {
                *self.mainset.active_paths,
                *self._ready,
                *self._futures,
            }
        )

    def _schedule_lookahead(self) -> None:
        if self._recovery_pending:
            raise DataServiceError("recovered active reprepare barrier is incomplete")
        occupied = len(self._ready) + len(self._futures)
        if occupied >= self.limits.verified_shard_lookahead:
            return
        excluded = {*self._ready, *self._futures}
        for row in self.mainset.rows:
            if row.status != "pending" or row.path in excluded:
                continue
            protected = self._protected_paths()
            self._futures[row.path] = self._executor.submit(
                self.cache.fetch,
                row.path,
                protected_paths=protected,
            )
            excluded.add(row.path)
            occupied += 1
            if occupied >= self.limits.verified_shard_lookahead:
                break

    def _next_path(self) -> str | None:
        outstanding = {
            lease.descriptor.record.path for lease in self._outstanding.values()
        }
        for row in self.mainset.rows:
            if row.status == "completed" or row.path in outstanding:
                continue
            if row.path in self._ready or row.path in self._futures:
                return row.path
            if row.status == "active":
                raise DataServiceError("active mainset row was not re-prepared")
        return None

    def _materialize_next(self) -> tuple[str, CachedShard] | None:
        path = self._next_path()
        if path is None:
            return None
        cached = self._ready.get(path)
        if cached is None:
            future = self._futures.pop(path)
            try:
                cached = future.result()
            except (DatasetTransportError, OSError, ShardCacheError) as exc:
                raise DataServiceError(
                    f"verified lookahead failed: {type(exc).__name__}"
                ) from None
            self._ready[path] = cached
        return path, cached

    def _replace_row_status(
        self,
        state: PersistentMainset,
        path: str,
        status: Literal["pending", "active", "completed"],
    ) -> PersistentMainset:
        matches = [row for row in state.rows if row.path == path]
        if len(matches) != 1:
            raise DataServiceError("mainset row identity is invalid")
        row = matches[0]
        rows = tuple(
            replace(candidate, status=status) if candidate.path == path else candidate
            for candidate in state.rows
        )
        updated = replace(state, rows=rows)
        self.store.validate(updated)
        if row.status == status:
            return state
        return updated

    def _state_identity(self) -> str:
        return hashlib.sha256(_mainset_bytes(self.mainset)).hexdigest()

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        with self._lock:
            self._ensure_running()
            if (
                type(worker_id) is not int
                or not 0 <= worker_id < self.identity.worker_count
            ):
                raise DataServiceError("lease worker identity is invalid")
            if worker_id in self._worker_leases:
                raise DataServiceError("worker already owns an active lease")
            if len(self._outstanding) >= self.limits.lease_channel_capacity:
                raise DataServiceError("lease channel capacity is exhausted")
            self._schedule_lookahead()
            materialized = self._materialize_next()
            if materialized is None:
                return None
            path, cached = materialized
            row = next(row for row in self.mainset.rows if row.path == path)
            if row.status == "pending":
                activated = self._replace_row_status(self.mainset, path, "active")
                self.store.save(activated)
                self._mainset = activated
            elif row.status != "active":
                raise DataServiceError("completed mainset row reached lease issuance")
            if (
                cached.fetched.relative_path != path
                or not cached.fetched.path.is_absolute()
                or cached.fetched.path != self._ready[path].fetched.path
            ):
                raise DataServiceError("cache returned an invalid shard descriptor")
            self._ready.pop(path)
            state_identity = self._state_identity()
            lease_id = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "mainset_id": self.mainset.mainset_id,
                        "path": path,
                        "state": state_identity,
                        "worker_id": worker_id,
                    }
                )
            ).hexdigest()
            descriptor = ShardLeaseDescriptor(
                lease_id=lease_id,
                worker_id=worker_id,
                state_identity=state_identity,
                record=self.manifest.shard(path),
                local_path=cached.fetched.path,
            )
            self._outstanding[lease_id] = _OutstandingLease(descriptor)
            self._worker_leases[worker_id] = lease_id
            self._schedule_lookahead()
            return descriptor

    def acknowledge(self, lease_id: str, worker_id: int, state_identity: str) -> None:
        with self._lock:
            self._ensure_running()
            lease = self._outstanding.get(lease_id)
            if (
                lease is None
                or lease.descriptor.worker_id != worker_id
                or lease.descriptor.state_identity != state_identity
                or self._worker_leases.get(worker_id) != lease_id
            ):
                raise DataServiceError(
                    "completion ACK identity does not match the lease"
                )
            path = lease.descriptor.record.path
            completed = self._replace_row_status(self.mainset, path, "completed")
            remaining_leases = len(self._outstanding) - 1
            if all(row.status == "completed" for row in completed.rows):
                if remaining_leases or self._ready or self._futures:
                    raise DataServiceError(
                        "mainset cannot rotate with outstanding supply state"
                    )
                published = self.store.new(
                    previous_id=completed.mainset_id,
                    replayed_shards=completed.replayed_shards,
                    replayed_samples=completed.replayed_samples,
                )
            else:
                published = completed
            self.store.save(published)
            self._mainset = published
            del self._outstanding[lease_id]
            del self._worker_leases[worker_id]
            self._schedule_lookahead()

    @property
    def done(self) -> bool:
        return False

    @property
    def stats(self) -> DataServiceStats:
        with self._lock:
            return DataServiceStats(
                in_flight_downloads=sum(
                    not future.done() for future in self._futures.values()
                ),
                verified_ready_shards=len(self._ready)
                + sum(future.done() for future in self._futures.values()),
                outstanding_leases=len(self._outstanding),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._ownership_handle is not None:
                try:
                    fcntl.flock(self._ownership_handle.fileno(), fcntl.LOCK_UN)
                    self._ownership_handle.close()
                except OSError:
                    raise DataServiceError(
                        "data service ownership release failed"
                    ) from None
                finally:
                    self._ownership_handle = None


def _recv_frame(connection: socket.socket) -> dict[str, Any]:
    chunks = bytearray()
    while len(chunks) <= MAX_SERVICE_FRAME_BYTES:
        block = connection.recv(min(4096, MAX_SERVICE_FRAME_BYTES + 1 - len(chunks)))
        if not block:
            break
        chunks.extend(block)
        if chunks.endswith(b"\n"):
            break
    return parse_frame(bytes(chunks))


def _response(payload: dict[str, object]) -> bytes:
    framed = canonical_json_bytes(payload)
    if len(framed) > MAX_SERVICE_FRAME_BYTES:
        raise DataServiceError("service response exceeds the frame bound")
    return framed


class DataServiceServer:
    """Single-owner AF_UNIX request server; trainers cannot control lifecycle."""

    def __init__(
        self,
        service: DataSupplyService,
        socket_path: Path,
        *,
        request_timeout_seconds: float,
    ) -> None:
        if (
            not socket_path.is_absolute()
            or type(request_timeout_seconds) is not float
            or request_timeout_seconds <= 0.0
        ):
            raise ValueError("data service socket settings are invalid")
        self.service = service
        self.socket_path = socket_path
        self.request_timeout_seconds = request_timeout_seconds

    def _dispatch(self, request: dict[str, Any]) -> dict[str, object]:
        operation = request.get("op")
        if operation == "health":
            if set(request) != {"op", "protocol_version", "session_identity"}:
                raise DataServiceError("health request fields are invalid")
            identity = DataServiceSessionIdentity.from_dict(request["session_identity"])
            if (
                request["protocol_version"] != SERVICE_PROTOCOL_VERSION
                or identity != self.service.identity
            ):
                raise DataServiceError("service identity differs from the client")
            return {
                "done": self.service.done,
                "ok": True,
                "protocol_version": SERVICE_PROTOCOL_VERSION,
                "session_sha256": self.service.identity.sha256,
            }
        if operation == "lease":
            if set(request) != {"op", "worker_id"}:
                raise DataServiceError("lease request fields are invalid")
            descriptor = self.service.lease(request["worker_id"])
            return {
                "done": descriptor is None,
                "lease": None if descriptor is None else descriptor.as_dict(),
                "ok": True,
            }
        if operation == "ack":
            if set(request) != {"lease_id", "op", "state_identity", "worker_id"}:
                raise DataServiceError("ACK request fields are invalid")
            self.service.acknowledge(
                request["lease_id"], request["worker_id"], request["state_identity"]
            )
            return {"ok": True}
        raise DataServiceError("service operation is invalid")

    def serve(
        self,
        stop_event: threading.Event,
        *,
        ready_callback: Callable[[], None] | None = None,
    ) -> None:
        listener: socket.socket | None = None
        bound_inode: int | None = None
        try:
            # Acquire cache/mainset ownership before touching the shared socket.
            # A losing process must never unlink or replace the winning endpoint.
            self.service.start()
            if self.socket_path.exists() or self.socket_path.is_symlink():
                if self.socket_path.is_symlink() or not stat.S_ISSOCK(
                    self.socket_path.stat().st_mode
                ):
                    raise DataServiceError("data service socket path already exists")
                stale_inode = self.socket_path.stat().st_ino
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.settimeout(0.1)
                    probe.connect(str(self.socket_path))
                except OSError:
                    try:
                        current_inode = self.socket_path.stat().st_ino
                    except OSError:
                        current_inode = None
                    if current_inode == stale_inode:
                        self.socket_path.unlink(missing_ok=True)
                    elif current_inode is not None:
                        raise DataServiceError(
                            "another data service owns the socket"
                        ) from None
                else:
                    raise DataServiceError("another data service owns the socket")
                finally:
                    probe.close()
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.socket_path))
            bound_inode = self.socket_path.stat().st_ino
            os.chmod(self.socket_path, 0o600)
            listener.listen(self.service.limits.ack_channel_capacity)
            listener.settimeout(0.2)
            if ready_callback is not None:
                ready_callback()
            while not stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(self.request_timeout_seconds)
                    try:
                        request = _recv_frame(connection)
                        response = self._dispatch(request)
                    except DataServiceStateCommittedError:
                        raise
                    except (
                        DataServiceError,
                        DataServiceProtocolError,
                        OSError,
                        TypeError,
                    ):
                        response = cast(
                            dict[str, object],
                            {"error": "request_rejected", "ok": False},
                        )
                    connection.sendall(_response(response))
        finally:
            if listener is not None:
                listener.close()
            self.service.close()
            if bound_inode is not None:
                try:
                    current = self.socket_path.stat()
                    if (
                        current.st_ino == bound_inode
                        and stat.S_ISSOCK(current.st_mode)
                    ):
                        self.socket_path.unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = [
    "DataServiceError",
    "DataServiceLimits",
    "DataServiceServer",
    "DataServiceStateCommittedError",
    "DataServiceStats",
    "DataSupplyService",
    "MainsetRow",
    "PersistentMainset",
]
