from collections.abc import Mapping

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import PipelineSampleRejected, WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text == SYSTEM_PREFIX:
            return list(range(34))
        if text == MAIN_SUFFIX:
            return list(range(100, 105))
        return [123]


def _fields(_raw: Mapping[str, object]) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(
        condition_route=0.0,
        condition_only=0.0,
        tag=0.0,
        candidate_source=0.0,
        nl=nl,
    )


def test_pipeline_skips_corrupt_image_and_reports_rejection() -> None:
    pipeline = object.__new__(WebDatasetPipeline)
    pipeline.metadata_adapter = lambda raw: raw
    pipeline.metadata_fields = MetadataFieldMapping(id_field="id")
    pipeline.base_seed = 7
    pipeline.stage = "S0"
    pipeline.cycle_index = 0
    pipeline.caption_fields_parser = _fields
    pipeline.probabilities = _probabilities()
    pipeline.condition_mode = "artist_or_character"
    pipeline.tokenizer = _Tokenizer()
    pipeline.framing = FramingContract(34, 5, 248044)
    pipeline.buckets = (BucketShape(512, 512),)
    pipeline.min_crop_retention = 0.8
    rejections: list[str] = []
    pipeline.rejection_observer = rejections.append

    shard = "data/synthetic/shard-000000.tar"
    sample = {
        "__url__": shard,
        "__key__": "synthetic/000000",
        "json": b'{"id": 1, "release": "synthetic", "width": 512, "height": 512, "caption_available": false}',
        "jpg": b"not an image",
    }
    result = pipeline._process(
        sample,
        {shard: ShardRecord(path=shard, bytes=1)},
    )

    assert result is None
    assert rejections == ["decode_error"]


def test_pipeline_skips_policy_rejected_sample_before_image_decode() -> None:
    pipeline = object.__new__(WebDatasetPipeline)
    pipeline.metadata_adapter = lambda raw: raw
    pipeline.metadata_fields = MetadataFieldMapping(id_field="id")
    pipeline.base_seed = 7
    pipeline.stage = "S0"
    pipeline.cycle_index = 0
    pipeline.caption_fields_parser = lambda _raw: (_ for _ in ()).throw(
        PipelineSampleRejected("ai_image_corrupted")
    )
    pipeline.probabilities = _probabilities()
    pipeline.condition_mode = "artist_or_character"
    pipeline.tokenizer = _Tokenizer()
    pipeline.framing = FramingContract(34, 5, 248044)
    pipeline.buckets = (BucketShape(512, 512),)
    pipeline.min_crop_retention = 0.8
    rejections: list[str] = []
    pipeline.rejection_observer = rejections.append

    shard = "data/synthetic/shard-000000.tar"
    sample = {
        "__url__": shard,
        "__key__": "synthetic/000001",
        "json": b'{"id": 1}',
    }
    result = pipeline._process(
        sample,
        {shard: ShardRecord(path=shard, bytes=1)},
    )

    assert result is None
    assert rejections == ["ai_image_corrupted"]
