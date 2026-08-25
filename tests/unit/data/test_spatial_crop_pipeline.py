"""P6/P10 pipeline-level guarantees for the shifted-bucket spatial branch.

A decodable image drives ``WebDatasetPipeline._process`` end to end so the
test pins the two core-path guarantees that pure plan-level tests cannot:

* when the policy is absent, disabled, or geometrically infeasible the sample
  falls back to the ordinary aspect-bucket crop and the emitted image is
  bit-identical to the pre-spatial behaviour (no data change);
* when the policy applies, the emitted image is exactly the target bucket and
  the audit carries a coherent ``shifted_bucket`` record.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any, cast

import torch
from PIL import Image

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import (
    PipelineSample,
    WebDatasetPipeline,
)
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.spatial_crop import SpatialCropPolicy
from sakuramoon.data.transparent_white import TransparentWhiteTelemetry

_SHARD = "data/synthetic/shard-000000.tar"


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


def _flat_png(size: int, color: tuple[int, int, int] = (12, 20, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _pipeline(
    *,
    spatial_policy: SpatialCropPolicy | None,
    buckets: tuple[BucketShape, ...] = (BucketShape(512, 512),),
) -> WebDatasetPipeline:
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
    pipeline.buckets = buckets
    pipeline.min_crop_retention = 0.8
    pipeline.rejection_observer = lambda _reason: None
    pipeline.spatial_policy = spatial_policy
    pipeline.transparent_policy = None
    pipeline.transparent_telemetry = TransparentWhiteTelemetry()  # pyright: ignore[reportAttributeAccessIssue]
    return cast(Any, pipeline)


def _process_sample(
    pipeline: WebDatasetPipeline,
    image_bytes: bytes,
    *,
    sample_id: int = 1,
) -> PipelineSample:
    sample = {
        "__url__": _SHARD,
        "json": b'{"id": ' + str(sample_id).encode("ascii") + b"}",
        "png": image_bytes,
    }
    result = pipeline._process(
        sample,
        {_SHARD: ShardRecord(path=_SHARD, bytes=1)},
    )
    assert result is not None, "decodable sample must not be rejected"
    return cast(PipelineSample, result)


def _policy(
    *,
    enabled: bool,
    probability: float,
) -> SpatialCropPolicy:
    return SpatialCropPolicy(
        enabled=enabled,
        probability=probability,
        min_equivalent_zoom=1.02,
        max_equivalent_zoom=1.10,
        min_crop_retention=0.8,
    )


def test_absent_disabled_and_infeasible_paths_are_bit_identical() -> None:
    # 512x512 source into a 512 bucket: maximum feasible zoom is 1.0, below the
    # 1.02 floor, so an enabled policy cannot apply.  Every case must emit the
    # ordinary aspect-bucket crop.
    image_bytes = _flat_png(512)

    none_result = _process_sample(
        _pipeline(spatial_policy=None), image_bytes
    )
    disabled_result = _process_sample(
        _pipeline(spatial_policy=_policy(enabled=False, probability=0.0)),
        image_bytes,
    )
    infeasible_result = _process_sample(
        _pipeline(spatial_policy=_policy(enabled=True, probability=1.0)),
        image_bytes,
    )

    assert torch.equal(none_result.image, disabled_result.image)
    assert torch.equal(none_result.image, infeasible_result.image)

    for result in (none_result, disabled_result, infeasible_result):
        assert result.audit.crop_policy == "aspect_bucket"
        assert result.audit.spatial_applied is False
        assert result.audit.resized_width == 512
        assert result.audit.resized_height == 512

    # Absent and disabled skip the spatial branch entirely (reason "none");
    # an enabled policy that cannot apply reports the concrete geometry miss.
    assert none_result.audit.spatial_fallback_reason == "none"
    assert disabled_result.audit.spatial_fallback_reason == "none"
    assert infeasible_result.audit.spatial_fallback_reason == (
        "insufficient_source_resolution"
    )


def test_not_selected_policy_falls_back_to_aspect_bucket() -> None:
    # A positive-probability policy that is not selected for this sample must
    # also emit the ordinary crop, byte for byte.
    image_bytes = _flat_png(512)

    baseline = _process_sample(
        _pipeline(spatial_policy=None), image_bytes
    )
    unselected = _process_sample(
        _pipeline(spatial_policy=_policy(enabled=True, probability=0.0)),
        image_bytes,
    )

    assert torch.equal(baseline.image, unselected.image)
    assert unselected.audit.crop_policy == "aspect_bucket"
    assert unselected.audit.spatial_applied is False
    assert unselected.audit.spatial_fallback_reason == "not_selected"


def test_applied_shifted_bucket_emits_bucket_sized_image() -> None:
    # A 540x540 source leaves headroom (maximum feasible zoom 540/512 ~ 1.055),
    # so an always-selected policy applies and the image is cropped to the
    # 512 bucket from an enlarged canvas.
    image_bytes = _flat_png(540)
    result = _process_sample(
        _pipeline(spatial_policy=_policy(enabled=True, probability=1.0)),
        image_bytes,
    )

    assert result.audit.crop_policy == "shifted_bucket"
    assert result.audit.spatial_selected is True
    assert result.audit.spatial_applied is True
    assert result.audit.spatial_fallback_reason == "none"
    assert result.audit.requested_equivalent_zoom > 1.0
    assert 1.0 < result.audit.actual_equivalent_zoom <= 1.10
    assert result.audit.final_crop_retention <= result.audit.base_crop_retention
    assert result.audit.resized_width > 512
    assert (
        result.target_height == 512
        and result.target_width == 512
        and tuple(result.image.shape) == (3, 512, 512)
    )
