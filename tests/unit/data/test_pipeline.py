from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from PIL import Image

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import BucketedBatchDataset, build_batch_loader
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.pipeline import (
    WebDatasetPipeline,
    local_shard_order,
    rng_identity,
)
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.state import ShardRunState


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


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (640, 512), color=(10, 20, 30)).save(output, format="JPEG")
    return output.getvalue()


def _write_tar(path: Path, records: tuple[tuple[int, bytes], ...] | None = None) -> None:
    if records is None:
        records = ((1, _jpeg()), (2, b"not-an-image"))
    with tarfile.open(path, "w") as archive:
        for sample_id, image in records:
            metadata = json.dumps(
                {
                    "id": sample_id,
                    "release": "r1",
                    "width": 640,
                    "height": 512,
                    "caption_available": False,
                }
            ).encode()
            for extension, payload in (("json", metadata), ("jpg", image)):
                info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_real_webdataset_iteration_excludes_validation_before_decode(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "samples.tar"
    _write_tar(shard)
    parser_calls = 0

    def parser(raw: Mapping[str, object]) -> CaptionFields:
        nonlocal parser_calls
        parser_calls += 1
        return _empty_fields(raw)

    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        validation_ids=frozenset({2}),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=parser,
        base_seed=9,
        stage="S0",
        pass_index=0,
    )

    samples = tuple(pipeline)

    assert len(samples) == 1
    assert samples[0].sample_id == 1
    assert samples[0].image.shape == (3, 512, 512)
    assert samples[0].image.dtype == torch.uint8
    assert samples[0].audit.source_width == 640
    assert parser_calls == 1


def test_rng_identity_changes_by_stage_pass_and_domain() -> None:
    first = rng_identity(base_seed=4, stage="S0", pass_index=0, sample_id=10)
    repeated = rng_identity(base_seed=4, stage="S0", pass_index=0, sample_id=10)
    next_pass = rng_identity(base_seed=4, stage="S0", pass_index=1, sample_id=10)

    assert first == repeated
    assert first != next_pass
    assert first.caption_seed != first.crop_seed


@pytest.mark.parametrize("worker_count", [1, 2, 3])
def test_persistent_worker_sweep_consumes_each_shard_once(
    tmp_path: Path, worker_count: int
) -> None:
    paths: list[Path] = []
    for sample_id in range(1, 4):
        path = tmp_path / f"{sample_id}.tar"
        _write_tar(path, ((sample_id, _jpeg()),))
        paths.append(path)
    pipeline = WebDatasetPipeline(
        shard_paths=tuple(paths),
        validation_ids=frozenset(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_empty_fields,
        base_seed=9,
        stage="S0",
        pass_index=0,
    )
    batches = BucketedBatchDataset(
        pipeline,
        batch_size=1,
        padding_token_id=248044,
        drop_last=True,
    )
    loader = build_batch_loader(
        batches,
        worker_count=worker_count,
        ready_batches=worker_count,
        pin_memory=False,
    )

    consumed = sorted(
        int(batch.sample_ids[0].item())
        for batch in loader
    )

    assert consumed == [1, 2, 3]


def test_local_shard_order_replays_active_and_skips_completed(tmp_path: Path) -> None:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test",
        access_terms="test",
    )
    shards = tuple(
        ShardRecord(
            path=f"r/{index}.tar",
            release="r",
            bytes=1,
            sha256=f"{index + 1:064x}",
            samples=1,
        )
        for index in range(3)
    )
    manifest = DatasetManifest.from_shards(source, shards)
    state = ShardRunState(
        completed=(shards[0].path,),
        active=shards[2].path,
        replayed_shards=1,
        replayed_samples=1,
    )

    assert local_shard_order(tmp_path, manifest, state) == (
        tmp_path / shards[2].path,
        tmp_path / shards[1].path,
    )
