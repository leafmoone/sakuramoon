"""Stage-scaled camera viewport discovery and current-G1 integration.

The camera planner must NOT infer R from ``data.buckets.base_area_px``
(262144 = 512^2, the 512-equivalent BASE bucket family) and must NOT use
the unscaled base family as the current training vocabulary. R is the
unique square of the CURRENT-STAGE vocabulary:

    scale_buckets(
        generate_base_buckets(config.data.buckets),
        config.stage.resolution,
    )

Current G1 (config/train_g1.toml [stage].resolution = 256) => R = 256.
Future stages (512/768/1024) => R = 512/768/1024 with the same planner.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from PIL import Image

from sakuramoon.config import load_config
from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import (
    BucketRejection,
    BucketShape,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.camera_viewport import (
    CameraViewportPolicy,
    camera_stage_edge,
    discover_square_bucket,
    plan_camera_viewport,
)
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import PipelineSample, WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.transparent_white import TransparentWhiteTelemetry

CONFIG_ROOT = Path("config")

_MIN_CROP_RETENTION = 0.8


def _base_buckets_config() -> DataBucketsConfig:
    return DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        shape_count=17,
        transpose_closed=True,
    )


def _stage_buckets(stage_edge: int) -> tuple[BucketShape, ...]:
    return scale_buckets(
        generate_base_buckets(_base_buckets_config()), stage_edge
    )


def _policy() -> CameraViewportPolicy:
    return CameraViewportPolicy(
        enabled=True,
        probability=1.0,
        min_equivalent_zoom=1.10,
        max_equivalent_zoom=1.50,
    )


def test_current_g1_stage_resolution_is_256() -> None:
    loaded = load_config(
        Path("train_g1.toml"), config_root=CONFIG_ROOT, validate_secrets=False
    )
    config = loaded.config
    assert config.stage.resolution == 256
    # The 512-equivalent base area is the BASE family definition only; it
    # is not the current G1 target edge.
    assert config.data.buckets.base_area_px == 262144


def test_stage_vocabulary_square_discovery_is_stage_scaled() -> None:
    for edge in (256, 512, 768, 1024):
        buckets = _stage_buckets(edge)
        assert discover_square_bucket(buckets, stage_edge=edge) == edge
        assert camera_stage_edge(buckets) == edge
        squares = tuple(b for b in buckets if b.width == b.height)
        assert squares == (BucketShape(width=edge, height=edge),)


def test_current_g1_square_bucket_is_256_not_512() -> None:
    buckets = _stage_buckets(256)
    assert BucketShape(width=256, height=256) in buckets
    assert BucketShape(width=512, height=512) not in buckets


def test_planner_is_fail_closed_on_vocabulary_stage_mismatch() -> None:
    # The 256 vocabulary with a claimed stage edge of 512 must fall back
    # (no_square_bucket), never silently adopt a wrong R.
    buckets = _stage_buckets(256)
    assignment = assign_bucket(
        512, 256, buckets, min_crop_retention=_MIN_CROP_RETENTION
    )
    assert not isinstance(assignment, BucketRejection)
    plan = plan_camera_viewport(
        assignment,
        _policy(),
        buckets=buckets,
        stage_edge=512,
        source_size=(512, 256),
        policy_seed=1,
        offset_seed=1,
    )
    assert plan.applied is False
    assert plan.fallback_reason == "no_square_bucket"


def _g1_plan(source_width: int, source_height: int):
    buckets = _stage_buckets(256)
    assignment = assign_bucket(
        source_width, source_height, buckets, min_crop_retention=_MIN_CROP_RETENTION
    )
    assert not isinstance(assignment, BucketRejection), (
        f"{source_width}x{source_height} must be ordinarily admitted at G1"
    )
    plan = plan_camera_viewport(
        assignment,
        _policy(),
        buckets=buckets,
        stage_edge=256,
        source_size=(source_width, source_height),
        policy_seed=1,
        offset_seed=1,
    )
    return plan


def test_g1_source_512x256_is_camera_feasible() -> None:
    plan = _g1_plan(512, 256)
    assert plan.applied is True
    assert plan.orientation == "horizontal"
    assert plan.full_width == 512
    assert plan.full_height == 256
    assert math.isclose(plan.equivalent_zoom, math.sqrt(2.0), rel_tol=1e-12)
    assert plan.crop_box[2] - plan.crop_box[0] == 256
    assert plan.crop_box[3] - plan.crop_box[1] == 256


def test_g1_source_256x512_is_camera_feasible() -> None:
    plan = _g1_plan(256, 512)
    assert plan.applied is True
    assert plan.orientation == "vertical"
    assert plan.full_width == 256
    assert plan.full_height == 512
    assert math.isclose(plan.equivalent_zoom, math.sqrt(2.0), rel_tol=1e-12)


def test_g1_source_384x256_and_256x384_are_camera_feasible() -> None:
    for width, height in ((384, 256), (256, 384)):
        plan = _g1_plan(width, height)
        assert plan.applied is True, f"{width}x{height} must be camera-feasible"
        assert (plan.full_width, plan.full_height) == (width, height)
        expected_zoom = math.sqrt(width * height / (256.0 * 256.0))
        assert math.isclose(plan.equivalent_zoom, expected_zoom, rel_tol=1e-12)
        assert 1.10 <= plan.equivalent_zoom <= 1.50


def test_g1_source_256x256_is_not_upscaled() -> None:
    # short edge == R: no upscale is required and the sample is admitted;
    # the camera falls back (already square, ideal zoom 1.0 < 1.10).
    plan = _g1_plan(256, 256)
    assert plan.applied is False
    assert plan.fallback_reason == "near_square_below_min"


def test_g1_sub_stage_128_source_is_ordinarily_rejected() -> None:
    buckets = _stage_buckets(256)
    assignment = assign_bucket(
        128, 128, buckets, min_crop_retention=_MIN_CROP_RETENTION
    )
    assert isinstance(assignment, BucketRejection)
    assert assignment.reason == "no_upscale"


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


def _flat_png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 20, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


_G1_BUCKETS = _stage_buckets(256)
_SHARD = "data/synthetic/shard-000000.tar"


def _g1_pipeline() -> WebDatasetPipeline:
    pipeline = object.__new__(WebDatasetPipeline)
    pipeline.metadata_adapter = lambda raw: raw
    pipeline.metadata_fields = MetadataFieldMapping(id_field="id")
    pipeline.base_seed = 7
    pipeline.stage = "G1"
    pipeline.cycle_index = 0
    pipeline.caption_fields_parser = _fields
    pipeline.probabilities = _probabilities()
    pipeline.condition_mode = "artist_or_character"
    pipeline.tokenizer = _Tokenizer()
    pipeline.framing = FramingContract(34, 5, 248044)
    pipeline.buckets = _G1_BUCKETS
    pipeline.min_crop_retention = _MIN_CROP_RETENTION
    pipeline.rejection_observer = lambda _reason: None
    pipeline.spatial_policy = None
    pipeline.transparent_policy = None
    pipeline.transparent_telemetry = TransparentWhiteTelemetry()  # pyright: ignore[reportAttributeAccessIssue]
    pipeline.camera_policy = _policy()
    pipeline._camera_stage_edge = camera_stage_edge(_G1_BUCKETS)  # pyright: ignore[reportAttributeAccessIssue]
    return cast(Any, pipeline)


def test_g1_pipeline_emits_256x256_viewport_for_512x256_source() -> None:
    sample = {
        "__url__": _SHARD,
        "__key__": "synthetic/000001",
        "json": b'{"id": 1}',
        "png": _flat_png(512, 256),
    }
    result = _g1_pipeline()._process(
        sample,
        {_SHARD: ShardRecord(path=_SHARD, bytes=1)},
    )
    assert result is not None
    processed = cast(PipelineSample, result)
    assert processed.image.shape[-2:] == (256, 256)
    assert processed.target_height == 256
    assert processed.target_width == 256
    audit = processed.audit
    assert audit.camera_applied is True
    assert audit.camera_full_width == 512
    assert audit.camera_full_height == 256
    assert math.isclose(audit.camera_equivalent_zoom, math.sqrt(2.0), rel_tol=1e-6)
    assert audit.resized_width == 512
    assert audit.resized_height == 256
    assert audit.crop_box[2] - audit.crop_box[0] == 256
    assert audit.crop_box[3] - audit.crop_box[1] == 256
