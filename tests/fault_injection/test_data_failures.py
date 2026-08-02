from __future__ import annotations

import errno
import hashlib
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from sakuramoon.data.cache import CachedShard, ShardCache
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.modelscope import (
    FetchedShard,
    ModelScopeDatasetTransport,
    ShardIntegrityError,
    fetch_dataset_shard,
)
from sakuramoon.data.pipeline import PipelineSample, WebDatasetPipeline
from sakuramoon.data.state import ShardStateStore, SingleProcessShardCoordinator
from sakuramoon.fault_injection import (
    FaultScenario,
    run_until_ready_and_sigkill,
)

_CONTENT = b"complete-synthetic-shard"
_SHARD_PATH = "release/000001.tar"
_SOURCE_ROOT = Path(__file__).parents[2] / "src"


def _manifest() -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    return DatasetManifest.from_shards(
        source,
        (
            ShardRecord(
                path=_SHARD_PATH,
                bytes=len(_CONTENT),
                upstream_sha256=hashlib.sha256(_CONTENT).hexdigest(),
            ),
        ),
    )


class _Transport:
    stream_chunk_bytes = 4

    def __init__(self, body: bytes | None = None, error: OSError | None = None) -> None:
        self.body = body
        self.error = error

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        del manifest, shard
        if self.error is not None:
            raise self.error
        assert self.body is not None
        output.write(self.body)


class _Writer(Protocol):
    def write(self, payload: bytes) -> int: ...


class _ExitDataset(IterableDataset[torch.Tensor]):
    def __iter__(self) -> Iterator[torch.Tensor]:
        os._exit(29)


class _PreparedCache:
    def __init__(self, root: Path, manifest: DatasetManifest) -> None:
        self.root = root
        self.manifest = manifest

    def fetch(
        self, shard_path: str, *, protected_paths: frozenset[str] = frozenset()
    ) -> CachedShard:
        del protected_paths
        shard = self.manifest.shard(shard_path)
        path = self.root / shard_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_CONTENT)
        return CachedShard(
            FetchedShard(
                path, shard.path, shard.bytes, shard.upstream_sha256, False
            ),
            (),
            shard.bytes,
        )


def test_real_sigkill_leaves_partial_and_next_process_restarts_from_byte_zero(
    tmp_path: Path,
) -> None:
    script = f"""
import hashlib
import sys
import time
from pathlib import Path
sys.path.insert(0, {_SOURCE_ROOT.as_posix()!r})
from sakuramoon.data.manifest import DatasetManifest, DatasetSourceIdentity, ShardRecord
from sakuramoon.data.modelscope import fetch_dataset_shard
from sakuramoon.fault_injection import signal_ready_from_environment
content = b"complete-synthetic-shard"
source = DatasetSourceIdentity(repo_id="leafmoone/webdataset_danbooru", revision="master")
manifest = DatasetManifest.from_shards(source, (ShardRecord(path="release/000001.tar", bytes=len(content), upstream_sha256=hashlib.sha256(content).hexdigest()),))
class Interrupted:
    stream_chunk_bytes = 4
    def download(self, manifest, shard, output):
        output.write(b"partial")
        signal_ready_from_environment()
        time.sleep(30)
fetch_dataset_shard(Interrupted(), manifest, "release/000001.tar", Path(sys.argv[1]))
"""

    evidence = run_until_ready_and_sigkill(
        (sys.executable, "-c", script, str(tmp_path)),
        scenario=FaultScenario.DOWNLOAD_INTERRUPTION,
        timeout_seconds=30.0,
    )

    destination = tmp_path / _SHARD_PATH
    partial = destination.with_name(f"{destination.name}.partial")
    assert evidence.returncode == -signal.SIGKILL
    assert not destination.exists()
    assert partial.is_file()
    assert partial.stat().st_size < len(_CONTENT)

    fetched = fetch_dataset_shard(
        cast(ModelScopeDatasetTransport, _Transport(_CONTENT)),
        _manifest(),
        _SHARD_PATH,
        tmp_path,
    )
    assert fetched.path.read_bytes() == _CONTENT
    assert not partial.exists()


def test_truncated_checksum_and_enospc_never_publish_final_shard(
    tmp_path: Path,
) -> None:
    for name, transport in (
        ("truncated", _Transport(_CONTENT[:-1])),
        ("checksum", _Transport(b"x" * len(_CONTENT))),
        ("enospc", _Transport(error=OSError(errno.ENOSPC, "injected disk full"))),
    ):
        root = tmp_path / name
        expected = OSError if name == "enospc" else ShardIntegrityError
        with pytest.raises(expected):
            fetch_dataset_shard(
                cast(ModelScopeDatasetTransport, transport),
                _manifest(),
                _SHARD_PATH,
                root,
            )
        destination = root / _SHARD_PATH
        assert not destination.exists()
        assert not destination.with_name(f"{destination.name}.partial").exists()


def test_worker_os_exit_preserves_active_lease_and_restart_counts_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    state_path = tmp_path / "run/shards.json"
    cache = cast(ShardCache, _PreparedCache(tmp_path / "cache", manifest))
    first = SingleProcessShardCoordinator(cache, ShardStateStore(state_path, manifest))

    pipeline = object.__new__(WebDatasetPipeline)

    def worker_exit(
        _pipeline: WebDatasetPipeline,
        paths: tuple[Path, ...],
        records: tuple[ShardRecord, ...],
    ) -> Iterator[PipelineSample]:
        assert len(paths) == 1
        assert records == manifest.shards
        loader = DataLoader(_ExitDataset(), batch_size=None, num_workers=1, timeout=3)
        next(iter(loader))
        raise AssertionError("worker exit unexpectedly returned")
        yield  # pragma: no cover

    monkeypatch.setattr(WebDatasetPipeline, "_iter_paths", worker_exit)
    with pytest.raises(RuntimeError, match="worker.*exited"):
        tuple(pipeline.iter_leased_shards(first, (_SHARD_PATH,)))

    assert first.state.active == _SHARD_PATH
    restarted = SingleProcessShardCoordinator(
        cache, ShardStateStore(state_path, manifest)
    )
    assert restarted.state.active == _SHARD_PATH
    assert restarted.state.replayed_shards == 1

    def complete(
        _pipeline: WebDatasetPipeline,
        paths: tuple[Path, ...],
        records: tuple[ShardRecord, ...],
    ) -> Iterator[PipelineSample]:
        assert len(paths) == 1
        assert records == manifest.shards
        return
        yield  # pragma: no cover

    monkeypatch.setattr(WebDatasetPipeline, "_iter_paths", complete)
    assert tuple(pipeline.iter_leased_shards(restarted, (_SHARD_PATH,))) == ()
    assert restarted.state.completed == (_SHARD_PATH,)
    assert restarted.state.active is None
    assert tuple(pipeline.iter_leased_shards(restarted, (_SHARD_PATH,))) == ()


def test_state_enospc_is_hard_failure_without_temporary_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakuramoon.data.state as state_module

    store = ShardStateStore(tmp_path / "run/shards.json", _manifest())

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(state_module.os, "fsync", fail_fsync)
    with pytest.raises(Exception, match="could not be saved"):
        store.begin(store.load(), _SHARD_PATH)
    assert not store.path.exists()
    assert not tuple(store.path.parent.glob(".shards.json.*.tmp"))
