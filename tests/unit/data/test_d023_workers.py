from __future__ import annotations

# pyright: reportPrivateUsage=false
import io
import json
import tarfile
import threading
from collections.abc import Mapping
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import cast

import pytest
import torch
from PIL import Image

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.cache import CachedShard, ShardCache
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import (
    CollateError,
    _build_batch_loader,
    _completion_for,
    _PersistentShardDataset,
    _ShardWork,
    _shutdown_loader,
    _WorkerCompletion,
    _WorkerDone,
    iter_leased_batches,
)
from sakuramoon.data.manifest import DatasetManifest, DatasetSourceIdentity, ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.modelscope import FetchedShard
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.state import (
    ShardRunState,
    ShardStateStore,
    SingleProcessShardCoordinator,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [ord(char) + 300 for char in text]


def _fields() -> MetadataFieldMapping:
    return MetadataFieldMapping(
        id_field="id",
        width_field="width",
        height_field="height",
        caption_available_field="caption_available",
    )


def _metadata(raw: Mapping[str, object]) -> Mapping[str, object]:
    return raw


def _caption(_raw: Mapping[str, object]) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _observe_rejection(_reason: str) -> None:
    return None


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nl)


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (640, 512), color=(10, 20, 30)).save(output, format="JPEG")
    return output.getvalue()


def _write_tar(path: Path, sample_id: int) -> None:
    with tarfile.open(path, "w") as archive:
        for extension, payload in (
            (
                "json",
                json.dumps(
                    {
                        "id": sample_id,
                        "width": 640,
                        "height": 512,
                        "caption_available": False,
                    }
                ).encode(),
            ),
            ("jpg", _jpeg()),
        ):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class _PreparedCache:
    def __init__(self, root: Path, manifest: DatasetManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.protected: list[tuple[str, frozenset[str]]] = []

    def fetch(
        self, shard_path: str, *, protected_paths: frozenset[str] = frozenset()
    ) -> CachedShard:
        self.protected.append((shard_path, protected_paths))
        record = self.manifest.shard(shard_path)
        path = self.root / shard_path
        return CachedShard(
            FetchedShard(path, record.path, record.bytes, record.sha256, False),
            (),
            sum(item.bytes for item in self.manifest.shards),
        )


def _pipeline(path: Path, record: ShardRecord) -> WebDatasetPipeline:
    return WebDatasetPipeline(
        shard_paths=(path,),
        shard_records=(record,),
        metadata_adapter=_metadata,
        metadata_fields=_fields(),
        validation_ids=frozenset(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_caption,
        rejection_observer=_observe_rejection,
        base_seed=9,
        stage="S0",
        pass_index=0,
    )


def test_two_persistent_workers_coordinate_parent_state(tmp_path: Path) -> None:
    paths = tuple(tmp_path / f"{index}.tar" for index in range(4))
    for index, path in enumerate(paths, start=1):
        _write_tar(path, index)
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test",
        access_terms="test",
    )
    records = tuple(
        ShardRecord(
            path=path.name,
            release="trusted",
            bytes=path.stat().st_size,
            sha256="1" * 64,
            samples=1,
        )
        for path in paths
    )
    manifest = DatasetManifest.from_shards(source, records)
    cache = _PreparedCache(tmp_path, manifest)
    coordinator = SingleProcessShardCoordinator(
        cast(ShardCache, cache),
        ShardStateStore(tmp_path / "run/state.json", manifest, worker_count=2),
    )
    parent_thread_ready = threading.Event()
    release_parent_thread = threading.Event()

    def keep_torch_parent_thread_alive() -> None:
        value = torch.ones((64, 64), dtype=torch.float32)
        torch.mm(value, value)  # pyright: ignore[reportUnknownMemberType]
        parent_thread_ready.set()
        release_parent_thread.wait()

    parent_thread = threading.Thread(target=keep_torch_parent_thread_alive)
    parent_thread.start()
    assert parent_thread_ready.wait(timeout=5.0)
    try:
        batches = tuple(
            iter_leased_batches(
                _pipeline(paths[0], records[0]),
                coordinator,
                tuple(record.path for record in records),
                batch_size=1,
                worker_count=2,
                ready_batches=2,
                pin_memory=False,
                drop_last=True,
            )
        )
    finally:
        release_parent_thread.set()
        parent_thread.join(timeout=5.0)
    assert not parent_thread.is_alive()
    assert sorted(int(batch.sample_ids[0]) for batch in batches) == [1, 2, 3, 4]
    assert coordinator.state.completed == tuple(record.path for record in records)
    assert coordinator.state.active_shards == ()
    assert all(len(protected) >= 1 for _, protected in cache.protected)
    assert cache.protected[1][1] == frozenset({records[0].path, records[1].path})


def test_worker_and_ready_channels_are_bounded_and_two_processes_persist(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"{index}.tar" for index in range(4))
    for index, path in enumerate(paths, start=1):
        _write_tar(path, index)
    records = tuple(
        ShardRecord(
            path=path.name,
            release="trusted",
            bytes=path.stat().st_size,
            sha256="1" * 64,
            samples=1,
        )
        for path in paths
    )
    dataset = _PersistentShardDataset(
        _pipeline(paths[0], records[0]),
        batch_size=1,
        drop_last=True,
        worker_count=2,
    )
    loader = _build_batch_loader(
        dataset,
        worker_count=2,
        ready_batches=2,
        pin_memory=False,
        in_order=False,
    )
    assert dataset.input_queue_capacity == 2
    assert dataset.input_queue_capacity_per_worker == 1
    assert dataset.completion_queue_capacity == 2
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 1
    assert loader.persistent_workers is True
    assert loader.in_order is False
    loader_context = cast(
        BaseContext,
        loader.multiprocessing_context,  # pyright: ignore[reportUnknownMemberType]
    )
    assert loader_context is dataset.multiprocessing_context
    assert loader_context.get_start_method() == "spawn"

    iterator = iter(loader)
    next_command = 0
    for worker_id in range(2):
        dataset.submit(
            worker_id,
            _ShardWork(
                records[next_command].path, paths[next_command], records[next_command]
            )
        )
        next_command += 1
    worker_processes: set[tuple[int, int]] = set()
    completion_messages: dict[str, _WorkerCompletion] = {}
    completed = 0
    try:
        while completed < len(records):
            output = next(iterator)
            if not isinstance(output, _WorkerDone):
                continue
            completion = _completion_for(
                dataset.completion_queue,
                completion_messages,
                output.shard_path,
            )
            assert completion.normal
            assert completion.shard_path == output.shard_path
            assert (completion.worker_id, completion.worker_pid) == (
                output.worker_id,
                output.worker_pid,
            )
            worker_processes.add((output.worker_id, output.worker_pid))
            completed += 1
            if next_command < len(records):
                dataset.submit(
                    output.worker_id,
                    _ShardWork(
                        records[next_command].path,
                        paths[next_command],
                        records[next_command],
                    )
                )
                next_command += 1
    finally:
        for worker_id in range(2):
            dataset.stop(worker_id, records[0])
        _shutdown_loader(loader)

    assert {worker_id for worker_id, _pid in worker_processes} == {0, 1}
    assert len({pid for _worker_id, pid in worker_processes}) == 2


def test_recovered_active_barrier_and_worker_count_have_no_fallback(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"{index}.tar" for index in range(3))
    for index, path in enumerate(paths, start=1):
        _write_tar(path, index)
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test",
        access_terms="test",
    )
    records = tuple(
        ShardRecord(
            path=path.name,
            release="trusted",
            bytes=path.stat().st_size,
            sha256="1" * 64,
            samples=1,
        )
        for path in paths
    )
    manifest = DatasetManifest.from_shards(source, records)
    state_path = tmp_path / "run/state.json"
    store = ShardStateStore(state_path, manifest, worker_count=2)
    active = store.begin(ShardRunState.empty(worker_count=2), records[0].path)
    store.begin(active, records[1].path)
    cache = _PreparedCache(tmp_path, manifest)
    coordinator = SingleProcessShardCoordinator(
        cast(ShardCache, cache),
        ShardStateStore(state_path, manifest, worker_count=2),
    )
    requested = (records[2].path, records[1].path, records[0].path)

    with pytest.raises(CollateError, match="exactly match"):
        tuple(
            iter_leased_batches(
                _pipeline(paths[0], records[0]),
                coordinator,
                requested,
                batch_size=1,
                worker_count=1,
                ready_batches=1,
                pin_memory=False,
                drop_last=True,
            )
        )
    assert cache.protected == []
    assert coordinator.state.active_shards == (records[0].path, records[1].path)

    batches = tuple(
        iter_leased_batches(
            _pipeline(paths[0], records[0]),
            coordinator,
            requested,
            batch_size=1,
            worker_count=2,
            ready_batches=2,
            pin_memory=False,
            drop_last=True,
        )
    )
    assert sorted(int(batch.sample_ids[0]) for batch in batches) == [1, 2, 3]
    assert [path for path, _protected in cache.protected[:2]] == [
        records[0].path,
        records[1].path,
    ]
    assert cache.protected[1][1] == frozenset({records[0].path, records[1].path})
    assert cache.protected[2][0] == records[2].path
    assert len(cache.protected[2][1]) == 2
    assert records[2].path in cache.protected[2][1]
    assert coordinator.state.replayed_shards == 2
    assert coordinator.state.replayed_samples == 2
