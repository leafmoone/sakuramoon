"""Pipeline-level guarantees for the camera-viewport branch.

A decodable image drives ``WebDatasetPipeline._process`` end to end:

* absent / disabled / not-applied camera emits the ordinary aspect-bucket
  crop bit-identically (no data change);
* an applied camera emits exactly the R x R viewport and the audit carries
  a coherent camera record.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any, cast

import torch
from PIL import Image

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import BucketShape, generate_base_buckets, scale_buckets
from sakuramoon.data.camera_viewport import CameraViewportPolicy, camera_stage_edge
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


def _flat_png(width: int, height: int, color: tuple[int, int, int] = (12, 20, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _buckets() -> tuple[BucketShape, ...]:
    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        transpose_closed=True,
    )
    return scale_buckets(generate_base_buckets(config), 512)


BUCKETS = _buckets()


def _identity_adapter(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return metadata


def _noop_observer(reason: str) -> None:
    return None


def _pipeline(
    *,
    camera_policy: CameraViewportPolicy | None,
) -> WebDatasetPipeline:
    pipeline = object.__new__(WebDatasetPipeline)
    pipeline.metadata_adapter = _identity_adapter
    pipeline.metadata_fields = MetadataFieldMapping(id_field="id")
    pipeline.base_seed = 7
    pipeline.stage = "S0"
    pipeline.cycle_index = 0
    pipeline.caption_fields_parser = _fields
    pipeline.probabilities = _probabilities()
    pipeline.condition_mode = "artist_or_character"
    pipeline.tokenizer = _Tokenizer()
    pipeline.framing = FramingContract(34, 5, 248044)
    pipeline.buckets = BUCKETS
    pipeline.min_crop_retention = 0.8
    pipeline.rejection_observer = _noop_observer
    pipeline.spatial_policy = None
    pipeline.transparent_policy = None
    pipeline.transparent_telemetry = TransparentWhiteTelemetry()  # pyright: ignore[reportAttributeAccessIssue]
    pipeline.camera_policy = camera_policy
    pipeline._camera_stage_edge = camera_stage_edge(BUCKETS)  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    return cast(Any, pipeline)


def _process_sample(
    pipeline: WebDatasetPipeline,
    image_bytes: bytes,
    *,
    sample_id: int = 1,
) -> PipelineSample:
    sample = {
        "__url__": _SHARD,
        "__key__": f"synthetic/{sample_id:06d}",
        "json": b'{"id": ' + str(sample_id).encode("ascii") + b"}",
        "png": image_bytes,
    }
    result = pipeline._process(  # pyright: ignore[reportPrivateUsage]
        sample,
        {_SHARD: ShardRecord(path=_SHARD, bytes=1)},
    )
    assert result is not None, "decodable sample must not be rejected"
    return result


def _policy(
    *,
    enabled: bool,
    probability: float,
) -> CameraViewportPolicy:
    return CameraViewportPolicy(
        enabled=enabled,
        probability=probability,
        min_equivalent_zoom=1.10,
        max_equivalent_zoom=1.50,
    )


def test_absent_disabled_and_not_applied_paths_are_bit_identical() -> None:
    # 600x600: ordinary 512x512 bucket, ideal zoom 1.0 below the 1.10 floor,
    # so the enabled policy must fall back (near_square_below_min).
    image_bytes = _flat_png(600, 600)

    none_result = _process_sample(_pipeline(camera_policy=None), image_bytes)
    disabled_result = _process_sample(
        _pipeline(camera_policy=_policy(enabled=False, probability=0.0)),
        image_bytes,
    )
    infeasible_result = _process_sample(
        _pipeline(camera_policy=_policy(enabled=True, probability=1.0)),
        image_bytes,
    )

    assert torch.equal(none_result.image, disabled_result.image)
    assert torch.equal(none_result.image, infeasible_result.image)

    for result in (none_result, disabled_result, infeasible_result):
        audit = result.audit
        assert audit.crop_policy == "aspect_bucket"
        assert audit.camera_applied is False
        assert audit.camera_equivalent_zoom == 0.0
        assert audit.camera_full_width == 0
        assert audit.camera_full_height == 0
        assert audit.resized_width == 512
        assert audit.resized_height == 512

    # Absent/disabled skip the branch entirely; an enabled always-selected
    # policy that misses the geometry is selected-but-fallen-back.
    assert none_result.audit.camera_selected is False
    assert disabled_result.audit.camera_selected is False
    assert none_result.audit.camera_fallback_reason == "none"
    assert disabled_result.audit.camera_fallback_reason == "none"
    assert infeasible_result.audit.camera_selected is True
    assert infeasible_result.audit.camera_fallback_reason == "near_square_below_min"


def test_camera_applied_emits_square_viewport() -> None:
    # 1024x512 source: ordinary 736x352 bucket, ideal zoom sqrt(2) in band,
    # so an always-selected policy applies the horizontal camera viewport.
    image_bytes = _flat_png(1024, 512)

    baseline = _process_sample(_pipeline(camera_policy=None), image_bytes)
    applied = _process_sample(
        _pipeline(camera_policy=_policy(enabled=True, probability=1.0)),
        image_bytes,
    )

    assert not torch.equal(baseline.image, applied.image)
    audit = applied.audit
    assert audit.crop_policy == "camera_viewport"
    assert audit.camera_policy == "hdm_shifted_square_v2"
    assert audit.camera_selected is True
    assert audit.camera_applied is True
    assert audit.camera_fallback_reason == "none"
    assert audit.camera_orientation == "horizontal"
    assert 1.10 <= audit.camera_equivalent_zoom <= 1.50
    assert audit.camera_full_width == 1024
    assert audit.camera_full_height == 512
    assert audit.camera_final_retention > 0.0
    assert audit.camera_normalized_offset == 0.0 or (
        0.0 <= audit.camera_normalized_offset <= 1.0
    )
    assert applied.target_height == 512
    assert applied.target_width == 512
    assert tuple(applied.image.shape) == (3, 512, 512)
    # Ordinary-path fields stay coherent on the camera branch.
    assert audit.spatial_applied is False


def test_camera_not_selected_is_bit_identical() -> None:
    # A positive-probability policy that is not selected for this sample
    # must also emit the ordinary crop, byte for byte.
    image_bytes = _flat_png(1024, 512)
    baseline = _process_sample(_pipeline(camera_policy=None), image_bytes)
    for sample_id in range(1, 64):
        unselected = _process_sample(
            _pipeline(camera_policy=_policy(enabled=True, probability=0.25)),
            image_bytes,
            sample_id=sample_id,
        )
        if unselected.audit.camera_selected:
            continue
        assert torch.equal(baseline.image, unselected.image)
        assert unselected.audit.crop_policy == "aspect_bucket"
        assert unselected.audit.camera_fallback_reason == "not_selected"
        assert unselected.audit.camera_applied is False
        break
    else:
        raise AssertionError("expected at least one unselected draw in 64 samples")


def test_short_edge_fallback_is_bit_identical() -> None:
    # 400x800: short edge 400 < 512, the camera cannot apply; the ordinary
    # crop is emitted unchanged.
    image_bytes = _flat_png(400, 800)
    baseline = _process_sample(_pipeline(camera_policy=None), image_bytes)
    result = _process_sample(
        _pipeline(camera_policy=_policy(enabled=True, probability=1.0)),
        image_bytes,
    )
    assert torch.equal(baseline.image, result.image)
    assert result.audit.camera_fallback_reason == "short_edge_too_small"
    assert result.audit.camera_applied is False
    assert result.audit.crop_policy == "aspect_bucket"


def test_vertical_orientation_applies() -> None:
    # 512x1024 source: vertical orientation, ideal zoom sqrt(2).
    image_bytes = _flat_png(512, 1024)
    result = _process_sample(
        _pipeline(camera_policy=_policy(enabled=True, probability=1.0)),
        image_bytes,
    )
    assert result.audit.camera_applied is True
    assert result.audit.camera_orientation == "vertical"
    assert result.audit.camera_full_width == 512
    assert result.audit.camera_full_height == 1024
    assert tuple(result.image.shape) == (3, 512, 512)
