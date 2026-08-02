from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.cache import CachedShard, ShardCache
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import CollateError, iter_leased_batches
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.modelscope import FetchedShard
from sakuramoon.data.pipeline import (
    PipelineSampleError,
    WebDatasetPipeline,
    local_shard_order,
    rng_identity,
)
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.state import (
    ShardRunState,
    ShardStateStore,
    SingleProcessShardCoordinator,
)

_MISSING_RELEASE = object()


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [ord(character) + 300 for character in text]


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nl)


def _metadata_fields() -> MetadataFieldMapping:
    return MetadataFieldMapping(
        id_field="id",
        width_field="width",
        height_field="height",
        caption_available_field="caption_available",
    )


def _identity_metadata(
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    return raw


def _shard_record(path: Path) -> ShardRecord:
    return ShardRecord(
        path=path.name,
        bytes=path.stat().st_size,
        upstream_sha256="1" * 64,
    )


def _empty_fields(raw: Mapping[str, object]) -> CaptionFields:
    assert raw
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _ignore_rejection(_reason: str) -> None:
    return


def _jpeg() -> bytes:
    return _jpeg_size(640, 512)


def _jpeg_size(width: int, height: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(output, format="JPEG")
    return output.getvalue()


def _write_tar(
    path: Path,
    records: tuple[tuple[int, bytes], ...] | None = None,
    *,
    metadata_fields: MetadataFieldMapping | None = None,
    raw_release: object = "r1",
) -> None:
    if records is None:
        records = ((1, _jpeg()), (2, b"not-an-image"))
    fields = metadata_fields or _metadata_fields()
    with tarfile.open(path, "w") as archive:
        for sample_id, image in records:
            document: dict[str, object] = {
                fields.id_field: sample_id,
                fields.width_field: 640,
                fields.height_field: 512,
                fields.caption_available_field: False,
            }
            if raw_release is not _MISSING_RELEASE:
                document["release"] = raw_release
            metadata = json.dumps(document).encode()
            for extension, payload in (("json", metadata), ("jpg", image)):
                info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_real_webdataset_iteration_processes_every_sample_in_training_shard(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "samples.tar"
    _write_tar(shard, records=((1, _jpeg()), (2, _jpeg())))
    parser_calls = 0

    def parser(raw: Mapping[str, object]) -> CaptionFields:
        nonlocal parser_calls
        parser_calls += 1
        return _empty_fields(raw)

    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(_shard_record(shard),),
        metadata_adapter=_identity_metadata,
        metadata_fields=_metadata_fields(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=parser,
        rejection_observer=_ignore_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )

    samples = tuple(pipeline)

    assert tuple(sample.sample_id for sample in samples) == (1, 2)
    assert samples[0].source_shard == shard.name
    assert samples[0].image.shape == (3, 512, 512)
    assert samples[0].image.dtype == torch.uint8
    assert samples[0].audit.source_width == 640
    assert parser_calls == 2


def test_source_shard_is_preserved_when_raw_release_is_missing_and_fields_are_aliased(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "samples.tar"
    fields = MetadataFieldMapping(
        id_field="sample_id",
        width_field="image_width",
        height_field="image_height",
        caption_available_field="has_caption",
    )
    _write_tar(
        shard,
        ((1, _jpeg()),),
        raw_release=_MISSING_RELEASE,
    )

    def adapter(raw: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "sample_id": raw["id"],
            "image_width": raw["width"],
            "image_height": raw["height"],
            "has_caption": raw["caption_available"],
        }

    def parser(raw: Mapping[str, object]) -> CaptionFields:
        assert "id" in raw
        assert "sample_id" not in raw
        return _empty_fields(raw)

    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(_shard_record(shard),),
        metadata_adapter=adapter,
        metadata_fields=fields,
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=parser,
        rejection_observer=_ignore_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )

    sample = next(iter(pipeline))

    assert sample.source_shard == shard.name


def test_pipeline_rejects_unknown_sample_url_and_mismatched_shard_record(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "samples.tar"
    _write_tar(shard, ((1, _jpeg()),))
    record = _shard_record(shard)
    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(record,),
        metadata_adapter=_identity_metadata,
        metadata_fields=_metadata_fields(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_empty_fields,
        rejection_observer=_ignore_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )
    raw_sample = {
        "__url__": str(tmp_path / "unknown.tar"),
        "json": json.dumps(
            {
                "id": 1,
                "width": 640,
                "height": 512,
                "caption_available": False,
            }
        ).encode(),
    }

    with pytest.raises(PipelineSampleError, match="trusted shard records"):
        pipeline._process(  # pyright: ignore[reportPrivateUsage]
            raw_sample, {str(shard): record}
        )

    wrong_record = ShardRecord(
        path="other.tar",
        bytes=shard.stat().st_size,
        upstream_sha256="2" * 64,
    )
    with pytest.raises(PipelineSampleError, match="manifest shard path"):
        WebDatasetPipeline(
            shard_paths=(shard,),
            shard_records=(wrong_record,),
            metadata_adapter=_identity_metadata,
            metadata_fields=_metadata_fields(),
            buckets=(BucketShape(512, 512),),
            min_crop_retention=0.8,
            probabilities=_probabilities(),
            tokenizer=_Tokenizer(),
            framing=FramingContract(34, 5, 0),
            caption_fields_parser=_empty_fields,
            rejection_observer=_ignore_rejection,
            base_seed=9,
            stage="S0",
            cycle_index=0,
        )


@pytest.mark.parametrize(
    ("bucket", "rejected_image", "valid_image"),
    [
        (BucketShape(512, 512), _jpeg_size(500, 500), _jpeg_size(640, 512)),
        (BucketShape(256, 1024), _jpeg_size(2000, 300), _jpeg_size(1024, 256)),
    ],
    ids=["no_upscale", "retention"],
)
def test_image_rejection_skips_sample_and_continues_real_tar_iteration(
    tmp_path: Path,
    bucket: BucketShape,
    rejected_image: bytes,
    valid_image: bytes,
) -> None:
    shard = tmp_path / "samples.tar"
    _write_tar(shard, ((1, rejected_image), (2, valid_image)))
    rejections: list[str] = []
    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(_shard_record(shard),),
        metadata_adapter=_identity_metadata,
        metadata_fields=_metadata_fields(),
        buckets=(bucket,),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_empty_fields,
        rejection_observer=rejections.append,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )

    samples = tuple(pipeline)

    assert tuple(sample.sample_id for sample in samples) == (2,)
    assert rejections == ["no_upscale" if bucket.width == 512 else "retention"]


def test_rng_identity_changes_by_stage_cycle_and_domain() -> None:
    first = rng_identity(base_seed=4, stage="S0", cycle_index=0, sample_id=10)
    repeated = rng_identity(base_seed=4, stage="S0", cycle_index=0, sample_id=10)
    next_cycle = rng_identity(base_seed=4, stage="S0", cycle_index=1, sample_id=10)

    assert first == repeated
    assert first != next_cycle
    assert first.caption_seed != first.crop_seed


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
        return CachedShard(
            FetchedShard(
                path, shard.path, shard.bytes, shard.upstream_sha256, False
            ),
            (),
            sum(item.bytes for item in self.manifest.shards),
        )


def test_durable_loader_consumes_and_completes_each_shard_once(tmp_path: Path) -> None:
    paths: list[Path] = []
    for sample_id in range(1, 4):
        path = tmp_path / f"{sample_id}.tar"
        _write_tar(path, ((sample_id, _jpeg()),))
        paths.append(path)
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    shards = tuple(
        ShardRecord(
            path=path.name,
            bytes=path.stat().st_size,
            upstream_sha256=f"{index + 1:064x}",
        )
        for index, path in enumerate(paths)
    )
    manifest = DatasetManifest.from_shards(source, shards)
    coordinator = SingleProcessShardCoordinator(
        cast(ShardCache, _PreparedCache(tmp_path, manifest)),
        ShardStateStore(tmp_path / "run/state.json", manifest),
    )
    pipeline = WebDatasetPipeline(
        shard_paths=(paths[0],),
        shard_records=(shards[0],),
        metadata_adapter=_identity_metadata,
        metadata_fields=_metadata_fields(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_empty_fields,
        rejection_observer=_ignore_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )
    batches = tuple(
        iter_leased_batches(
            pipeline,
            coordinator,
            tuple(shard.path for shard in shards),
            batch_size=1,
            worker_count=1,
            ready_batches=1,
            pin_memory=False,
            drop_last=True,
        )
    )

    consumed = sorted(int(batch.sample_ids[0].item()) for batch in batches)

    assert consumed == [1, 2, 3]
    assert tuple(batch.source_shards for batch in batches) == tuple(
        (path.name,) for path in paths
    )
    assert coordinator.state.completed == tuple(sorted(shard.path for shard in shards))
    assert coordinator.state.active is None


def test_durable_loader_rejects_multi_worker_state_mismatch(tmp_path: Path) -> None:
    shard = tmp_path / "samples.tar"
    _write_tar(shard, ((1, _jpeg()),))
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    manifest = DatasetManifest.from_shards(
        source,
        (
            ShardRecord(
                path=shard.name,
                bytes=shard.stat().st_size,
                upstream_sha256="1" * 64,
            ),
        ),
    )
    coordinator = SingleProcessShardCoordinator(
        cast(ShardCache, _PreparedCache(tmp_path, manifest)),
        ShardStateStore(tmp_path / "run/state.json", manifest),
    )
    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(manifest.shards[0],),
        metadata_adapter=_identity_metadata,
        metadata_fields=_metadata_fields(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_empty_fields,
        rejection_observer=_ignore_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )

    with pytest.raises(CollateError, match="exactly one worker"):
        tuple(
            iter_leased_batches(
                pipeline,
                coordinator,
                (shard.name,),
                batch_size=1,
                worker_count=2,
                ready_batches=2,
                pin_memory=False,
                drop_last=True,
            )
        )


def test_worker_loader_cannot_bypass_durable_lease(tmp_path: Path) -> None:
    shard = tmp_path / "samples.tar"
    _write_tar(shard, ((1, _jpeg()),))
    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(_shard_record(shard),),
        metadata_adapter=_identity_metadata,
        metadata_fields=_metadata_fields(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_empty_fields,
        rejection_observer=_ignore_rejection,
        base_seed=9,
        stage="S0",
        cycle_index=0,
    )
    loader = DataLoader(pipeline, batch_size=None, num_workers=1)

    with pytest.raises(PipelineSampleError, match="durable shard lease"):
        next(iter(loader))


@pytest.mark.parametrize(
    "shard_paths",
    [("https://example.invalid/data.tar",), (Path("relative.tar"),)],
)
def test_pipeline_rejects_nonlocal_or_relative_shard_paths(
    shard_paths: object,
) -> None:
    with pytest.raises(PipelineSampleError, match="local"):
        WebDatasetPipeline(
            shard_paths=shard_paths,  # pyright: ignore[reportArgumentType]
            shard_records=(
                ShardRecord(
                    path="data.tar",
                    bytes=1,
                    upstream_sha256="1" * 64,
                ),
            ),
            metadata_adapter=_identity_metadata,
            metadata_fields=_metadata_fields(),
            buckets=(BucketShape(512, 512),),
            min_crop_retention=0.8,
            probabilities=_probabilities(),
            tokenizer=_Tokenizer(),
            framing=FramingContract(34, 5, 0),
            caption_fields_parser=_empty_fields,
            rejection_observer=_ignore_rejection,
            base_seed=9,
            stage="S0",
            cycle_index=0,
        )


def test_local_shard_order_replays_active_and_skips_completed(tmp_path: Path) -> None:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    shards = tuple(
        ShardRecord(
            path=f"r/{index}.tar",
            bytes=1,
            upstream_sha256=f"{index + 1:064x}",
        )
        for index in range(3)
    )
    manifest = DatasetManifest.from_shards(source, shards)
    state = ShardRunState(
        completed=(shards[0].path,),
        active=shards[2].path,
        replayed_shards=1,
    )

    assert local_shard_order(tmp_path, manifest, state) == (
        tmp_path / shards[2].path,
        tmp_path / shards[1].path,
    )
