from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, cast

import pytest

import sakuramoon.data.cache as cache_module
from sakuramoon.data.cache import CachedShard, CacheQuota, ShardCache, ShardCacheError
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.modelscope import FetchedShard, ModelScopeDatasetTransport


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _ConcurrentTransport:
    stream_chunk_bytes = 4

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        del manifest
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.03)
            index = int(Path(shard.path).stem)
            output.write(bytes([index]) * shard.bytes)
        finally:
            with self._lock:
                self.active -= 1


def _manifest(sizes: tuple[int, ...] = (10, 10, 10)) -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    shards = tuple(
        ShardRecord(
            path=f"release/{index:06d}.tar",
            bytes=size,
            upstream_sha256=hashlib.sha256(bytes([index]) * size).hexdigest(),
        )
        for index, size in enumerate(sizes)
    )
    return DatasetManifest.from_shards(source, shards)


def _transport() -> ModelScopeDatasetTransport:
    return cast(ModelScopeDatasetTransport, object())


def _write(root: Path, relative_path: str, payload: bytes, mtime_ns: int) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _install_fetch(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def fetch(
        transport: ModelScopeDatasetTransport,
        manifest: DatasetManifest,
        shard_path: str,
        root: Path,
    ) -> FetchedShard:
        del transport
        calls.append(shard_path)
        shard = manifest.shard(shard_path)
        index = int(Path(shard_path).stem)
        path = root / shard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * shard.bytes)
        return FetchedShard(
            path, shard.path, shard.bytes, shard.upstream_sha256, False
        )

    monkeypatch.setattr(cache_module, "fetch_dataset_shard", fetch)


@pytest.mark.parametrize("low,high", [(-1, 10), (10, 10), (11, 10), (True, 10)])
def test_quota_requires_ordered_nonnegative_integer_bytes(low: int, high: int) -> None:
    with pytest.raises(ValueError, match="quota"):
        CacheQuota(low, high)


def test_fetch_evicts_oldest_file_to_low_watermark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    first, second, third = manifest.shards
    _write(tmp_path, first.path, bytes([0]) * 10, 1)
    _write(tmp_path, second.path, bytes([1]) * 10, 2)
    calls: list[str] = []
    _install_fetch(monkeypatch, calls)
    cache = ShardCache(tmp_path, manifest, _transport(), CacheQuota(20, 25))

    result = cache.fetch(third.path)

    assert result.evicted_paths == (first.path,)
    assert result.usage_bytes == 20
    assert not (tmp_path / first.path).exists()
    assert (tmp_path / second.path).exists()
    assert result.fetched.path == tmp_path / third.path
    assert calls == [third.path]


def test_fetch_below_high_watermark_does_not_evict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    first, second, _ = manifest.shards
    _write(tmp_path, first.path, bytes([0]) * 10, 1)
    calls: list[str] = []
    _install_fetch(monkeypatch, calls)
    cache = ShardCache(tmp_path, manifest, _transport(), CacheQuota(20, 25))

    result = cache.fetch(second.path)

    assert result.evicted_paths == ()
    assert result.usage_bytes == 20
    assert (tmp_path / first.path).exists()


def test_fetch_rejects_shard_larger_than_high_watermark(tmp_path: Path) -> None:
    manifest = _manifest((30,))
    cache = ShardCache(tmp_path, manifest, _transport(), CacheQuota(10, 20))

    with pytest.raises(ShardCacheError, match="larger"):
        cache.fetch(manifest.shards[0].path)


def test_protected_files_are_not_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    first, second, third = manifest.shards
    _write(tmp_path, first.path, bytes([0]) * 10, 1)
    _write(tmp_path, second.path, bytes([1]) * 10, 2)
    _install_fetch(monkeypatch, [])
    cache = ShardCache(tmp_path, manifest, _transport(), CacheQuota(20, 25))

    result = cache.fetch(third.path, protected_paths=frozenset({first.path}))

    assert result.evicted_paths == (second.path,)
    assert (tmp_path / first.path).exists()


def test_concurrent_fetches_use_real_bounded_transport_parallelism(
    tmp_path: Path,
) -> None:
    manifest = _manifest((10, 10))
    transport = _ConcurrentTransport()
    cache = ShardCache(
        tmp_path,
        manifest,
        cast(ModelScopeDatasetTransport, transport),
        CacheQuota(20, 30),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                cache.fetch, (manifest.shards[0].path, manifest.shards[1].path)
            )
        )

    assert transport.max_active == 2
    assert tuple(result.fetched.relative_path for result in results) == tuple(
        shard.path for shard in manifest.shards
    )
    assert cache.usage_bytes() == 20


def test_in_flight_reservations_prevent_concurrent_quota_oversubscription(
    tmp_path: Path,
) -> None:
    manifest = _manifest((20, 20))
    cache = ShardCache(
        tmp_path,
        manifest,
        cast(ModelScopeDatasetTransport, _ConcurrentTransport()),
        CacheQuota(10, 30),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(cache.fetch, shard.path) for shard in manifest.shards
        )
        outcomes: list[object] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ShardCacheError as error:
                outcomes.append(error)

    assert sum(isinstance(item, CachedShard) for item in outcomes) == 1
    assert sum(isinstance(item, ShardCacheError) for item in outcomes) == 1
    assert cache.usage_bytes() == 20
