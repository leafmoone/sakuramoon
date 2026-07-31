from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import sakuramoon.data.cache as cache_module
from sakuramoon.data.cache import CachedShard, CacheQuota, ShardCache
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.modelscope import FetchedShard, ModelScopeDatasetTransport
from sakuramoon.data.state import (
    ShardRunState,
    ShardStateCommittedError,
    ShardStateError,
    ShardStateStore,
    SingleProcessShardCoordinator,
)


def _manifest(
    revision: str = "b" * 40, *, shard_count: int = 2
) -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision=revision,
        license_id="synthetic-license",
        access_terms="synthetic-terms",
    )
    shards = tuple(
        ShardRecord(
            path=f"release/{index:06d}.tar",
            release="release",
            bytes=4,
            sha256=hashlib.sha256(bytes([index]) * 4).hexdigest(),
            samples=(index + 1) * 10,
        )
        for index in range(shard_count)
    )
    return DatasetManifest.from_shards(source, shards)


def _coordinator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[str],
) -> SingleProcessShardCoordinator:
    manifest = _manifest()

    def fetch(
        transport: ModelScopeDatasetTransport,
        selected_manifest: DatasetManifest,
        shard_path: str,
        root: Path,
    ) -> FetchedShard:
        del transport
        calls.append(shard_path)
        shard = selected_manifest.shard(shard_path)
        path = root / shard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return FetchedShard(path, shard.path, shard.bytes, shard.sha256, False)

    monkeypatch.setattr(cache_module, "fetch_dataset_shard", fetch)
    cache = ShardCache(
        tmp_path / "cache",
        manifest,
        cast(ModelScopeDatasetTransport, object()),
        CacheQuota(4, 12),
    )
    store = ShardStateStore(tmp_path / "run/shards.json", manifest)
    return SingleProcessShardCoordinator(cache, store)


def test_state_begin_complete_round_trip(tmp_path: Path) -> None:
    manifest = _manifest()
    store = ShardStateStore(tmp_path / "state.json", manifest)
    first = manifest.shards[0]

    active = store.begin(ShardRunState.empty(), first.path)
    assert store.load() == active
    completed = store.complete(active, first.path)

    assert completed.completed == (first.path,)
    assert completed.active is None
    assert store.load() == completed


def test_schema_v3_persists_bounded_active_shards_and_worker_count(
    tmp_path: Path,
) -> None:
    manifest = _manifest(shard_count=3)
    store = ShardStateStore(tmp_path / "state.json", manifest, worker_count=2)
    state = ShardRunState.empty(worker_count=2)
    state = store.begin(state, manifest.shards[1].path)
    state = store.begin(state, manifest.shards[0].path)

    document = cast(dict[str, object], json.loads(store.path.read_bytes()))

    assert document["schema_version"] == 3
    assert document["worker_count"] == 2
    assert document["active_shards"] == [
        manifest.shards[0].path,
        manifest.shards[1].path,
    ]
    assert store.load() == state


@pytest.mark.parametrize("worker_count", [0, -1, True])
def test_worker_count_requires_positive_exact_integer(
    tmp_path: Path, worker_count: int
) -> None:
    with pytest.raises(ValueError, match="worker_count"):
        ShardStateStore(
            tmp_path / "state.json", _manifest(), worker_count=worker_count
        )


def test_state_rejects_worker_count_drift_and_active_overflow(tmp_path: Path) -> None:
    manifest = _manifest(shard_count=3)
    path = tmp_path / "state.json"
    writer = ShardStateStore(path, manifest, worker_count=2)
    active = writer.begin(
        ShardRunState.empty(worker_count=2), manifest.shards[0].path
    )

    with pytest.raises(ShardStateError, match="invalid"):
        ShardStateStore(path, manifest, worker_count=1).load()
    with pytest.raises(ShardStateError, match="invalid"):
        writer.save(
            ShardRunState(
                completed=(),
                active_shards=tuple(shard.path for shard in manifest.shards),
                worker_count=2,
            )
        )

    second = writer.begin(active, manifest.shards[1].path)
    with pytest.raises(ShardStateError, match="capacity"):
        writer.begin(second, manifest.shards[2].path)


def test_initial_state_publish_failure_rolls_back_visible_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakuramoon.data.state as state_module

    store = ShardStateStore(tmp_path / "state.json", _manifest())
    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 1:
                raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(state_module.os, "fsync", fail_first_directory_fsync)

    with pytest.raises(ShardStateError, match="could not be saved"):
        store.begin(ShardRunState.empty(), _manifest().shards[0].path)

    assert directory_fsyncs == 2
    assert store.load() == ShardRunState.empty()
    assert not tuple(tmp_path.glob(".state.json.*"))


def test_state_update_publish_failure_restores_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakuramoon.data.state as state_module

    manifest = _manifest()
    store = ShardStateStore(tmp_path / "state.json", manifest)
    original = store.begin(ShardRunState.empty(), manifest.shards[0].path)
    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_post_replace_directory_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(state_module.os, "fsync", fail_post_replace_directory_fsync)

    with pytest.raises(ShardStateError, match="could not be saved"):
        store.complete(original, manifest.shards[0].path)

    assert directory_fsyncs == 3
    assert store.load() == original
    assert not tuple(tmp_path.glob(".state.json.*"))


def test_state_distinguishes_committed_state_from_rollback_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakuramoon.data.state as state_module

    manifest = _manifest()
    store = ShardStateStore(tmp_path / "state.json", manifest)
    original = store.begin(ShardRunState.empty(), manifest.shards[0].path)
    expected = ShardRunState(
        completed=(manifest.shards[0].path,),
        active=None,
        replayed_shards=0,
        replayed_samples=0,
    )
    real_fsync = os.fsync
    directory_fsyncs = 0

    def fail_cleanup_directory_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 3:
                raise OSError("injected rollback cleanup fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(state_module.os, "fsync", fail_cleanup_directory_fsync)

    with pytest.raises(ShardStateCommittedError, match="state saved"):
        store.complete(original, manifest.shards[0].path)

    assert directory_fsyncs == 3
    assert store.load() == expected
    assert not tuple(tmp_path.glob(".state.json.*"))


def test_recovery_keeps_active_shard_and_counts_replay(tmp_path: Path) -> None:
    manifest = _manifest()
    store = ShardStateStore(tmp_path / "state.json", manifest)
    active = store.begin(ShardRunState.empty(), manifest.shards[1].path)

    recovered = store.recover()

    assert recovered.active == active.active
    assert recovered.replayed_shards == 1
    assert recovered.replayed_samples == manifest.shards[1].samples
    assert store.load() == recovered


class _RecordingCache:
    def __init__(self, root: Path, manifest: DatasetManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.protected_calls: list[tuple[str, frozenset[str]]] = []
        self.on_fetch: Callable[[str], None] | None = None

    def fetch(
        self, shard_path: str, *, protected_paths: frozenset[str]
    ) -> CachedShard:
        self.protected_calls.append((shard_path, protected_paths))
        if self.on_fetch is not None:
            self.on_fetch(shard_path)
        shard = self.manifest.shard(shard_path)
        path = self.root / shard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(shard.bytes))
        return CachedShard(
            fetched=FetchedShard(
                path=path,
                relative_path=shard.path,
                bytes=shard.bytes,
                sha256=shard.sha256,
                cache_hit=False,
            ),
            evicted_paths=(),
            usage_bytes=shard.bytes,
        )


def test_activation_is_persisted_before_fetch_and_all_active_are_protected(
    tmp_path: Path,
) -> None:
    manifest = _manifest(shard_count=3)
    store = ShardStateStore(tmp_path / "state.json", manifest, worker_count=2)
    cache = _RecordingCache(tmp_path / "cache", manifest)
    coordinator = SingleProcessShardCoordinator(cast(ShardCache, cache), store)

    def assert_persisted(shard_path: str) -> None:
        assert shard_path in store.load().active_shards

    cache.on_fetch = assert_persisted
    first, second, third = (shard.path for shard in manifest.shards)
    coordinator.prepare(first)
    coordinator.prepare(second)

    assert cache.protected_calls == [
        (first, frozenset({first})),
        (second, frozenset({first, second})),
    ]

    coordinator.mark_completed(first)
    assert coordinator.state.completed == (first,)
    assert coordinator.state.active_shards == (second,)

    coordinator.prepare(third)
    assert cache.protected_calls[-1] == (
        third,
        frozenset({second, third}),
    )


def test_active_shard_cannot_be_prepared_twice(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = _RecordingCache(tmp_path / "cache", manifest)
    coordinator = SingleProcessShardCoordinator(
        cast(ShardCache, cache),
        ShardStateStore(tmp_path / "state.json", manifest),
    )
    first = manifest.shards[0].path

    coordinator.prepare(first)

    with pytest.raises(ShardStateError, match="already prepared"):
        coordinator.prepare(first)
    assert [path for path, _protected in cache.protected_calls] == [first]


def test_restart_counts_every_active_shard_and_requires_all_reprepared(
    tmp_path: Path,
) -> None:
    manifest = _manifest(shard_count=3)
    state_path = tmp_path / "state.json"
    writer = ShardStateStore(state_path, manifest, worker_count=2)
    state = ShardRunState.empty(worker_count=2)
    state = writer.begin(state, manifest.shards[0].path)
    writer.begin(state, manifest.shards[1].path)

    cache = _RecordingCache(tmp_path / "cache", manifest)
    restarted = SingleProcessShardCoordinator(
        cast(ShardCache, cache),
        ShardStateStore(state_path, manifest, worker_count=2),
    )
    first, second, third = (shard.path for shard in manifest.shards)

    assert restarted.state.active_shards == (first, second)
    assert restarted.state.replayed_shards == 2
    assert restarted.state.replayed_samples == (
        manifest.shards[0].samples + manifest.shards[1].samples
    )

    with pytest.raises(ShardStateError, match="replayed and prepared"):
        restarted.prepare(third)
    restarted.prepare(first)
    restarted.mark_completed(first)
    with pytest.raises(ShardStateError, match="replayed and prepared"):
        restarted.prepare(third)
    restarted.prepare(second)
    restarted.prepare(third)

    assert restarted.state.active_shards == (second, third)
    assert restarted.state.replayed_shards == 2
    assert restarted.state.replayed_samples == 30


def test_singleton_lease_remains_compatible_with_active_view(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = _RecordingCache(tmp_path / "cache", manifest)
    coordinator = SingleProcessShardCoordinator(
        cast(ShardCache, cache),
        ShardStateStore(tmp_path / "state.json", manifest),
    )
    first = manifest.shards[0].path

    with coordinator.lease(first) as cached:
        assert cached is not None
        assert coordinator.state.active == first
        assert coordinator.state.active_shards == (first,)

    assert coordinator.state.active is None
    assert coordinator.state.completed == (first,)


def test_coordinator_replays_active_first_and_skips_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    store = ShardStateStore(tmp_path / "run/shards.json", manifest)
    store.begin(ShardRunState.empty(), manifest.shards[0].path)
    calls: list[str] = []
    coordinator = _coordinator(monkeypatch, tmp_path, calls)

    with pytest.raises(ShardStateError, match="must be replayed"):
        coordinator.prepare(manifest.shards[1].path)
    prepared = coordinator.prepare(manifest.shards[0].path)
    assert prepared is not None
    coordinator.mark_completed(manifest.shards[0].path)
    assert coordinator.prepare(manifest.shards[0].path) is None

    assert calls == [manifest.shards[0].path]
    assert coordinator.state.replayed_shards == 1
    assert coordinator.state.replayed_samples == manifest.shards[0].samples


def test_failed_prepare_leaves_shard_active_for_next_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    coordinator = _coordinator(monkeypatch, tmp_path, [])
    shard_path = _manifest().shards[0].path

    def fail(*args: object, **kwargs: object) -> FetchedShard:
        del args, kwargs
        raise OSError("synthetic interruption")

    monkeypatch.setattr(cache_module, "fetch_dataset_shard", fail)
    with pytest.raises(OSError, match="interruption"):
        coordinator.prepare(shard_path)

    restarted = _coordinator(monkeypatch, tmp_path, [])
    assert restarted.state.active == shard_path
    assert restarted.state.replayed_shards == 1


def test_state_rejects_unknown_keys_and_paths(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "manifest_sha256": manifest_sha256(manifest),
                "completed": ["missing.tar"],
                "active_shards": [],
                "replayed_shards": 0,
                "replayed_samples": 0,
                "worker_count": 1,
                "unexpected": True,
            }
        )
    )

    with pytest.raises(ShardStateError, match="invalid"):
        ShardStateStore(path, manifest).load()


def test_state_rejects_different_manifest_with_identical_paths(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    original = _manifest()
    original_store = ShardStateStore(path, original)
    original_store.begin(ShardRunState.empty(), original.shards[0].path)

    changed = _manifest("c" * 40)
    assert tuple(shard.path for shard in changed.shards) == tuple(
        shard.path for shard in original.shards
    )
    with pytest.raises(ShardStateError, match="invalid"):
        ShardStateStore(path, changed).load()


@pytest.mark.parametrize("schema_version", [1, 2])
def test_state_rejects_legacy_schema(
    tmp_path: Path, schema_version: int
) -> None:
    manifest = _manifest()
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "completed": [manifest.shards[0].path],
                "active": None,
                "replayed_shards": 0,
                "replayed_samples": 0,
            }
        )
    )

    with pytest.raises(ShardStateError, match="unsupported"):
        ShardStateStore(path, manifest).load()


def test_only_active_shard_can_complete(tmp_path: Path) -> None:
    manifest = _manifest()
    store = ShardStateStore(tmp_path / "state.json", manifest)
    with pytest.raises(ShardStateError, match="active"):
        store.complete(ShardRunState.empty(), manifest.shards[0].path)
