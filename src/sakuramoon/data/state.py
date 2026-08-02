"""Shard-level at-least-once state for one local training process."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sakuramoon.data.manifest import DatasetManifest

if TYPE_CHECKING:
    from sakuramoon.data.cache import CachedShard, ShardCache


class ShardStateError(RuntimeError):
    """Shard state is invalid or an operation violates replay order."""


class ShardStateCommittedError(ShardStateError):
    """The state committed, but cleanup durability could not be confirmed."""


@dataclass(frozen=True, init=False)
class ShardRunState:
    completed: tuple[str, ...]
    active_shards: tuple[str, ...]
    worker_count: int
    replayed_shards: int

    def __init__(
        self,
        completed: tuple[str, ...],
        active_shards: tuple[str, ...] | str = (),
        worker_count: int = 1,
        replayed_shards: int = 0,
        *,
        active: str | None = None,
    ) -> None:
        """Build v3 state while accepting the legacy singleton constructor shape."""

        if active is not None:
            if active_shards:
                raise ValueError("active and active_shards cannot both be set")
            active_shards = (active,)
        elif isinstance(active_shards, str):
            active_shards = (active_shards,)
        object.__setattr__(self, "completed", completed)
        object.__setattr__(self, "active_shards", active_shards)
        object.__setattr__(self, "worker_count", worker_count)
        object.__setattr__(self, "replayed_shards", replayed_shards)

    @property
    def active(self) -> str | None:
        """Expose the historical singleton view used by the D015 lease path."""

        if len(self.active_shards) > 1:
            raise ShardStateError("singleton active view has multiple active shards")
        return self.active_shards[0] if self.active_shards else None

    @classmethod
    def empty(cls, worker_count: int = 1) -> ShardRunState:
        return cls(
            completed=(),
            active_shards=(),
            worker_count=worker_count,
            replayed_shards=0,
        )


def _payload(state: ShardRunState, manifest_id: str) -> dict[str, object]:
    return {
        "active_shards": list(state.active_shards),
        "completed": list(state.completed),
        "manifest_id": manifest_id,
        "replayed_shards": state.replayed_shards,
        "schema_version": 4,
        "worker_count": state.worker_count,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ShardStateStore:
    """Read and atomically replace one small JSON state file."""

    def __init__(
        self, path: Path, manifest: DatasetManifest, *, worker_count: int = 1
    ) -> None:
        if type(worker_count) is not int or worker_count <= 0:
            raise ValueError("worker_count must be a positive integer")
        self.path = path
        self.manifest = manifest
        self.worker_count = worker_count
        self._manifest_id = manifest.manifest_id
        self._known_paths = frozenset(shard.path for shard in manifest.shards)

    def _validate_state(self, state: ShardRunState) -> None:
        completed = state.completed
        active_shards = state.active_shards
        if (
            type(state.worker_count) is not int
            or state.worker_count != self.worker_count
            or type(completed) is not tuple
            or any(type(item) is not str for item in completed)
            or completed != tuple(sorted(set(completed)))
            or not set(completed).issubset(self._known_paths)
            or type(active_shards) is not tuple
            or any(type(item) is not str for item in active_shards)
            or active_shards != tuple(sorted(set(active_shards)))
            or len(active_shards) > self.worker_count
            or not set(active_shards).issubset(self._known_paths)
            or not set(active_shards).isdisjoint(completed)
            or type(state.replayed_shards) is not int
            or state.replayed_shards < 0
        ):
            raise ShardStateError("shard state is invalid")

    def load(self) -> ShardRunState:
        if not self.path.exists():
            return ShardRunState.empty(self.worker_count)
        try:
            raw_document = json.loads(self.path.read_bytes())
            if not isinstance(raw_document, dict):
                raise TypeError
            document = cast(dict[str, Any], raw_document)
            if document.get("schema_version") != 4:
                raise ShardStateError("unsupported shard state schema version")
            if set(document) != {
                "active_shards",
                "completed",
                "manifest_id",
                "replayed_shards",
                "schema_version",
                "worker_count",
            }:
                raise ValueError
            if document["manifest_id"] != self._manifest_id:
                raise ValueError
            active = cast(object, document["active_shards"])
            completed = cast(object, document["completed"])
            replayed_shards = document["replayed_shards"]
            worker_count = document["worker_count"]
            if not isinstance(active, list) or not isinstance(completed, list):
                raise TypeError
            active_items = cast(list[object], active)
            completed_items = cast(list[object], completed)
            if not all(isinstance(item, str) for item in active_items) or not all(
                isinstance(item, str) for item in completed_items
            ):
                raise TypeError
            state = ShardRunState(
                completed=tuple(cast(list[str], completed_items)),
                active_shards=tuple(cast(list[str], active_items)),
                worker_count=cast(int, worker_count),
                replayed_shards=cast(int, replayed_shards),
            )
            self._validate_state(state)
            return state
        except ShardStateError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise ShardStateError("shard state is invalid") from None

    def save(self, state: ShardRunState) -> None:
        self._validate_state(state)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        rollback = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.rollback"
        )
        body = (
            json.dumps(
                _payload(state, self._manifest_id),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rollback_linked = False
        published = False
        try:
            with temporary.open("wb") as handle:
                handle.write(body)
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
            rollback_removed = False
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
                    rollback_removed = rollback_removed or residue == rollback
                except OSError as exc:
                    rollback_error = rollback_error or exc
            if rollback_removed:
                try:
                    _fsync_directory(self.path.parent)
                except OSError as exc:
                    rollback_error = rollback_error or exc
            if rollback_error is not None:
                raise ShardStateError("shard state rollback failed") from None
            raise ShardStateError("shard state could not be saved") from None
        if rollback_linked:
            try:
                rollback.unlink()
                _fsync_directory(self.path.parent)
            except OSError:
                raise ShardStateCommittedError(
                    "shard state saved but rollback cleanup failed"
                ) from None

    def recover(self) -> ShardRunState:
        state = self.load()
        if not state.active_shards:
            return state
        recovered = replace(
            state,
            replayed_shards=state.replayed_shards + len(state.active_shards),
        )
        self.save(recovered)
        return recovered

    def begin(self, state: ShardRunState, shard_path: str) -> ShardRunState:
        self._validate_state(state)
        if shard_path not in self._known_paths:
            raise ShardStateError("unknown shard cannot become active")
        if shard_path in state.completed:
            raise ShardStateError("completed shard cannot become active")
        if shard_path in state.active_shards:
            return state
        if len(state.active_shards) >= state.worker_count:
            raise ShardStateError("active shard capacity is exhausted")
        started = replace(
            state,
            active_shards=tuple(sorted((*state.active_shards, shard_path))),
        )
        self.save(started)
        return started

    def complete(self, state: ShardRunState, shard_path: str) -> ShardRunState:
        self._validate_state(state)
        if shard_path not in state.active_shards:
            raise ShardStateError("only an active shard can be completed")
        completed = tuple(sorted((*state.completed, shard_path)))
        active_shards = tuple(
            active for active in state.active_shards if active != shard_path
        )
        finished = replace(
            state, completed=completed, active_shards=active_shards
        )
        self.save(finished)
        return finished


class SingleProcessShardCoordinator:
    """Coordinate bounded active shards and preserve shard-level replay semantics."""

    def __init__(self, cache: ShardCache, store: ShardStateStore) -> None:
        self.cache = cache
        self.store = store
        self.state = store.recover()
        self._recovery_pending = set(self.state.active_shards)
        self._prepared = set[str]()

    def prepare(self, shard_path: str) -> CachedShard | None:
        if shard_path in self.state.completed:
            return None
        if shard_path in self._prepared:
            raise ShardStateError("active shard is already prepared")
        if shard_path not in self.state.active_shards and self._recovery_pending:
            raise ShardStateError(
                "all recovered active shards must be replayed and prepared before a new shard"
            )
        self.state = self.store.begin(self.state, shard_path)
        cached = self.cache.fetch(
            shard_path, protected_paths=frozenset(self.state.active_shards)
        )
        self._recovery_pending.discard(shard_path)
        self._prepared.add(shard_path)
        return cached

    def mark_completed(self, shard_path: str) -> None:
        if shard_path not in self._prepared:
            raise ShardStateError("active shard must be prepared before completion")
        self.state = self.store.complete(self.state, shard_path)
        self._prepared.remove(shard_path)
        self._recovery_pending.discard(shard_path)

    @contextmanager
    def lease(self, shard_path: str) -> Generator[CachedShard | None]:
        """Mark a prepared shard complete only after its consumer exits normally."""

        cached = self.prepare(shard_path)
        completed_normally = False
        try:
            yield cached
            completed_normally = True
        finally:
            if completed_normally and cached is not None:
                self.mark_completed(shard_path)
