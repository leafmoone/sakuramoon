"""Single-process shard cache with explicit byte quota and LRU eviction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sakuramoon.data.manifest import DatasetManifest
from sakuramoon.data.modelscope import (
    FetchedShard,
    ModelScopeDatasetTransport,
    fetch_dataset_shard,
)


class ShardCacheError(RuntimeError):
    """The shard cache cannot satisfy its explicit quota."""


@dataclass(frozen=True)
class CacheQuota:
    low_bytes: int
    high_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.low_bytes) is not int
            or type(self.high_bytes) is not int
            or self.low_bytes < 0
            or self.low_bytes >= self.high_bytes
        ):
            raise ValueError("cache quota requires 0 <= low_bytes < high_bytes")


@dataclass(frozen=True)
class CachedShard:
    fetched: FetchedShard
    evicted_paths: tuple[str, ...]
    usage_bytes: int


class ShardCache:
    """Cache verified manifest shards without a service or background worker."""

    def __init__(
        self,
        root: Path,
        manifest: DatasetManifest,
        transport: ModelScopeDatasetTransport,
        quota: CacheQuota,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.transport = transport
        self.quota = quota

    def _path(self, shard_path: str) -> Path:
        return self.root / shard_path

    def usage_bytes(self) -> int:
        total = 0
        for shard in self.manifest.shards:
            path = self._path(shard.path)
            if path.is_file():
                total += path.stat().st_size
        return total

    def _evict_for(
        self, shard_path: str, *, protected_paths: frozenset[str]
    ) -> tuple[str, ...]:
        shard = self.manifest.shard(shard_path)
        destination = self._path(shard_path)
        additional = 0 if destination.is_file() else shard.bytes
        usage = self.usage_bytes()
        if additional > self.quota.high_bytes:
            raise ShardCacheError("shard is larger than the cache high watermark")
        if usage + additional <= self.quota.high_bytes:
            return ()

        target = max(self.quota.low_bytes, additional)
        candidates: list[tuple[int, str, Path, int]] = []
        for candidate in self.manifest.shards:
            if candidate.path == shard_path or candidate.path in protected_paths:
                continue
            path = self._path(candidate.path)
            if path.is_file():
                stat = path.stat()
                candidates.append((stat.st_mtime_ns, candidate.path, path, stat.st_size))
        evicted: list[str] = []
        projected = usage + additional
        for _, relative_path, path, size in sorted(candidates):
            if projected <= target:
                break
            path.unlink()
            projected -= size
            evicted.append(relative_path)
        if projected > self.quota.high_bytes:
            raise ShardCacheError("cache quota cannot be met without protected shards")
        return tuple(evicted)

    def fetch(
        self, shard_path: str, *, protected_paths: frozenset[str] = frozenset()
    ) -> CachedShard:
        evicted = self._evict_for(shard_path, protected_paths=protected_paths)
        fetched = fetch_dataset_shard(
            self.transport, self.manifest, shard_path, self.root
        )
        try:
            os.utime(fetched.path, None)
            usage = self.usage_bytes()
        except OSError:
            raise ShardCacheError("cache file could not be updated") from None
        if usage > self.quota.high_bytes:
            raise ShardCacheError("cache usage exceeds the high watermark")
        return CachedShard(fetched=fetched, evicted_paths=evicted, usage_bytes=usage)
