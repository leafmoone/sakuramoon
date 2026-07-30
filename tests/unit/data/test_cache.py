from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import cast

import pytest

import sakuramoon.data.cache as cache_module
from sakuramoon.data.cache import CacheQuota, ShardCache, ShardCacheError
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.modelscope import FetchedShard, ModelScopeDatasetTransport


def _manifest(sizes: tuple[int, ...] = (10, 10, 10)) -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="synthetic-license",
        access_terms="synthetic-terms",
    )
    shards = tuple(
        ShardRecord(
            path=f"release/{index:06d}.tar",
            release="release",
            bytes=size,
            sha256=hashlib.sha256(bytes([index]) * size).hexdigest(),
            samples=index + 1,
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
        return FetchedShard(path, shard.path, shard.bytes, shard.sha256, False)

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
