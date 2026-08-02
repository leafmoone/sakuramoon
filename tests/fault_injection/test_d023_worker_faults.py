from __future__ import annotations

import io
import json
import os
import tarfile
import time
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.cache import CachedShard, ShardCache
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import TrainingBatch, iter_leased_batches
from sakuramoon.data.manifest import DatasetManifest, DatasetSourceIdentity, ShardRecord
from sakuramoon.data.modelscope import FetchedShard
from sakuramoon.data.pipeline import MetadataAdapter, WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.state import ShardStateStore, SingleProcessShardCoordinator


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [ord(char) + 300 for char in text]


def _fields():
    from sakuramoon.data.metadata import MetadataFieldMapping

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


def _write_tar(path: Path, sample_id: int) -> None:
    image = io.BytesIO()
    Image.new("RGB", (640, 512), color=(10, 20, 30)).save(image, format="JPEG")
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
            ("jpg", image.getvalue()),
        ):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class _PreparedCache:
    def __init__(self, root: Path, manifest: DatasetManifest) -> None:
        self.root = root
        self.manifest = manifest

    def fetch(
        self, shard_path: str, *, protected_paths: frozenset[str] = frozenset()
    ) -> CachedShard:
        del protected_paths
        record = self.manifest.shard(shard_path)
        return CachedShard(
            FetchedShard(
                self.root / shard_path,
                record.path,
                record.bytes,
                record.upstream_sha256,
                False,
            ),
            (),
            sum(item.bytes for item in self.manifest.shards),
        )


def _crash_worker(_raw: Mapping[str, object]) -> Mapping[str, object]:
    # Let the parent persist and enqueue both bounded active shards first.
    time.sleep(0.5)
    os._exit(23)


def _manifest(
    tmp_path: Path, count: int = 2
) -> tuple[DatasetManifest, tuple[Path, ...]]:
    paths = tuple(tmp_path / f"{index}.tar" for index in range(count))
    for sample_id, path in enumerate(paths, start=1):
        _write_tar(path, sample_id)
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    records = tuple(
        ShardRecord(
            path=path.name,
            bytes=path.stat().st_size,
            upstream_sha256="1" * 64,
        )
        for path in paths
    )
    return DatasetManifest.from_shards(source, records), paths


def _pipeline(
    path: Path,
    record: ShardRecord,
    *,
    metadata_adapter: MetadataAdapter = _metadata,
) -> WebDatasetPipeline:
    return WebDatasetPipeline(
        shard_paths=(path,),
        shard_records=(record,),
        metadata_adapter=metadata_adapter,
        metadata_fields=_fields(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_caption,
        rejection_observer=_observe_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )


def _batches(
    pipeline: WebDatasetPipeline,
    coordinator: SingleProcessShardCoordinator,
    manifest: DatasetManifest,
) -> Generator[TrainingBatch, None, None]:
    return cast(
        Generator[TrainingBatch, None, None],
        iter_leased_batches(
            pipeline,
            coordinator,
            tuple(record.path for record in manifest.shards),
            batch_size=1,
            worker_count=2,
            ready_batches=2,
            pin_memory=False,
            drop_last=True,
        ),
    )


def test_real_worker_exit_keeps_both_active_and_restart_replays_from_start(
    tmp_path: Path,
) -> None:
    manifest, paths = _manifest(tmp_path)
    cache = cast(ShardCache, _PreparedCache(tmp_path, manifest))
    state_path = tmp_path / "run/state.json"
    first = SingleProcessShardCoordinator(
        cache, ShardStateStore(state_path, manifest, worker_count=2)
    )

    with pytest.raises(RuntimeError, match="worker.*exited"):
        tuple(
            _batches(
                _pipeline(paths[0], manifest.shards[0], metadata_adapter=_crash_worker),
                first,
                manifest,
            )
        )

    assert first.state.active_shards == tuple(record.path for record in manifest.shards)
    assert first.state.completed == ()

    restarted = SingleProcessShardCoordinator(
        cache, ShardStateStore(state_path, manifest, worker_count=2)
    )
    assert restarted.state.active_shards == tuple(
        record.path for record in manifest.shards
    )
    assert restarted.state.replayed_shards == 2

    batches = tuple(
        _batches(_pipeline(paths[0], manifest.shards[0]), restarted, manifest)
    )
    assert sorted(int(batch.sample_ids[0]) for batch in batches) == [1, 2]
    assert restarted.state.completed == tuple(record.path for record in manifest.shards)
    assert restarted.state.active_shards == ()


def test_parent_early_close_keeps_prefetched_shards_active_for_restart(
    tmp_path: Path,
) -> None:
    manifest, paths = _manifest(tmp_path)
    cache = cast(ShardCache, _PreparedCache(tmp_path, manifest))
    state_path = tmp_path / "run/state.json"
    first = SingleProcessShardCoordinator(
        cache, ShardStateStore(state_path, manifest, worker_count=2)
    )
    batches = _batches(_pipeline(paths[0], manifest.shards[0]), first, manifest)

    next(batches)
    batches.close()

    assert first.state.completed == ()
    assert first.state.active_shards == tuple(record.path for record in manifest.shards)
    restarted = SingleProcessShardCoordinator(
        cache, ShardStateStore(state_path, manifest, worker_count=2)
    )
    replayed = tuple(
        _batches(_pipeline(paths[0], manifest.shards[0]), restarted, manifest)
    )
    assert sorted(int(batch.sample_ids[0]) for batch in replayed) == [1, 2]
    assert restarted.state.replayed_shards == 2
