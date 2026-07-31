from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import cast

import pytest

import sakuramoon.data.cache as cache_module
from sakuramoon.data.cache import CacheQuota, ShardCache
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


def _manifest(revision: str = "b" * 40) -> DatasetManifest:
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
        for index in range(2)
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
                "schema_version": 2,
                "manifest_sha256": manifest_sha256(manifest),
                "completed": ["missing.tar"],
                "active": None,
                "replayed_shards": 0,
                "replayed_samples": 0,
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


def test_state_rejects_legacy_unbound_schema(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed": [manifest.shards[0].path],
                "active": None,
                "replayed_shards": 0,
                "replayed_samples": 0,
            }
        )
    )

    with pytest.raises(ShardStateError, match="invalid"):
        ShardStateStore(path, manifest).load()


def test_only_active_shard_can_complete(tmp_path: Path) -> None:
    manifest = _manifest()
    store = ShardStateStore(tmp_path / "state.json", manifest)
    with pytest.raises(ShardStateError, match="active"):
        store.complete(ShardRunState.empty(), manifest.shards[0].path)
