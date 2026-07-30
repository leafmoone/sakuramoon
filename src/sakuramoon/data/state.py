"""Shard-level at-least-once state for one local training process."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from sakuramoon.data.cache import CachedShard, ShardCache
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetManifestError,
    manifest_sha256,
)


class ShardStateError(RuntimeError):
    """Shard state is invalid or an operation violates replay order."""


@dataclass(frozen=True)
class ShardRunState:
    completed: tuple[str, ...]
    active: str | None
    replayed_shards: int
    replayed_samples: int

    @classmethod
    def empty(cls) -> ShardRunState:
        return cls(completed=(), active=None, replayed_shards=0, replayed_samples=0)


def _payload(state: ShardRunState, manifest_digest: str) -> dict[str, object]:
    return {
        "active": state.active,
        "completed": list(state.completed),
        "manifest_sha256": manifest_digest,
        "replayed_samples": state.replayed_samples,
        "replayed_shards": state.replayed_shards,
        "schema_version": 2,
    }


class ShardStateStore:
    """Read and atomically replace one small JSON state file."""

    def __init__(self, path: Path, manifest: DatasetManifest) -> None:
        self.path = path
        self.manifest = manifest
        self._manifest_sha256 = manifest_sha256(manifest)
        self._known_paths = frozenset(shard.path for shard in manifest.shards)

    def load(self) -> ShardRunState:
        if not self.path.exists():
            return ShardRunState.empty()
        try:
            document = cast(dict[str, Any], json.loads(self.path.read_bytes()))
            if set(document) != {
                "active",
                "completed",
                "manifest_sha256",
                "replayed_samples",
                "replayed_shards",
                "schema_version",
            }:
                raise ValueError
            if (
                document["schema_version"] != 2
                or document["manifest_sha256"] != self._manifest_sha256
            ):
                raise ValueError
            active = document["active"]
            completed = cast(object, document["completed"])
            replayed_shards = document["replayed_shards"]
            replayed_samples = document["replayed_samples"]
            if active is not None and not isinstance(active, str):
                raise TypeError
            if not isinstance(completed, list):
                raise TypeError
            completed_items = cast(list[object], completed)
            if not all(isinstance(item, str) for item in completed_items):
                raise TypeError
            if type(replayed_shards) is not int or replayed_shards < 0:
                raise ValueError
            if type(replayed_samples) is not int or replayed_samples < 0:
                raise ValueError
            completed_paths = tuple(cast(list[str], completed_items))
            if (
                completed_paths != tuple(sorted(set(completed_paths)))
                or not set(completed_paths).issubset(self._known_paths)
                or active not in self._known_paths | {None}
                or active in completed_paths
            ):
                raise ValueError
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise ShardStateError("shard state is invalid") from None
        return ShardRunState(
            completed=completed_paths,
            active=active,
            replayed_shards=replayed_shards,
            replayed_samples=replayed_samples,
        )

    def save(self, state: ShardRunState) -> None:
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        body = (
            json.dumps(
                _payload(state, self._manifest_sha256),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise ShardStateError("shard state could not be saved") from None

    def recover(self) -> ShardRunState:
        state = self.load()
        if state.active is None:
            return state
        try:
            samples = self.manifest.shard(state.active).samples
        except DatasetManifestError:
            raise ShardStateError("active shard is absent from the manifest") from None
        recovered = replace(
            state,
            replayed_shards=state.replayed_shards + 1,
            replayed_samples=state.replayed_samples + samples,
        )
        self.save(recovered)
        return recovered

    def begin(self, state: ShardRunState, shard_path: str) -> ShardRunState:
        if shard_path not in self._known_paths:
            raise ShardStateError("unknown shard cannot become active")
        if shard_path in state.completed:
            raise ShardStateError("completed shard cannot become active")
        if state.active not in (None, shard_path):
            raise ShardStateError("active shard must be replayed before another shard")
        started = replace(state, active=shard_path)
        self.save(started)
        return started

    def complete(self, state: ShardRunState, shard_path: str) -> ShardRunState:
        if state.active != shard_path:
            raise ShardStateError("only the active shard can be completed")
        completed = tuple(sorted((*state.completed, shard_path)))
        finished = replace(state, completed=completed, active=None)
        self.save(finished)
        return finished


class SingleProcessShardCoordinator:
    """Prepare one shard at a time and preserve shard-level replay semantics."""

    def __init__(self, cache: ShardCache, store: ShardStateStore) -> None:
        self.cache = cache
        self.store = store
        self.state = store.recover()

    def prepare(self, shard_path: str) -> CachedShard | None:
        if shard_path in self.state.completed:
            return None
        self.state = self.store.begin(self.state, shard_path)
        return self.cache.fetch(shard_path, protected_paths=frozenset({shard_path}))

    def mark_completed(self, shard_path: str) -> None:
        self.state = self.store.complete(self.state, shard_path)
