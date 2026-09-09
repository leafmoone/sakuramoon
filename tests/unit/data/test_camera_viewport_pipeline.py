"""Pipeline-level guarantees for the shifted-square camera path.

A decodable image drives ``WebDatasetPipeline._process`` end to end:

* every effectively-off config shape (absent table, enabled=false with any
  legal p, enabled=true p=0) emits the ordinary dev path bit-identically,
  including configs where ordinary spatial crop is enabled;
* P100 emits exactly one R x R shifted-square view per accepted sample and
  rejects (never upscales, never falls back) sources below the target;
* the camera-selected path never consults the ordinary prepare_image /
  assign_bucket admission;
* the emitted pixels are the crop-after-scale result (never a direct
  R x R patch), including EXIF-rotated sources and JPEG draft decoding with
  the draft-coverage re-decode fallback.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data import pipeline as pipeline_module
from sakuramoon.data.buckets import BucketShape, generate_base_buckets, scale_buckets
from sakuramoon.data.camera_viewport import CameraViewportPolicy
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
R = 512  # stage edge of the test bucket vocabulary


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if not text:
            return []
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


def _gradient(width: int, height: int) -> Image.Image:
    """A per-pixel-unique gradient: a direct R x R patch differs from any
    scaled view, so equality against the scale-then-crop reference proves
    the crop-after-scale contract."""

    xs = (
        np.tile((np.arange(width, dtype=np.uint16) * 7 + 3), height)
        .reshape(height, width)
        .astype(np.uint8)
    )
    ys = (
        np.repeat((np.arange(height, dtype=np.uint16) * 13 + 5), width)
        .reshape(height, width)
        .astype(np.uint8)
    )
    zs = (
        (np.tile(np.arange(width, dtype=np.uint16), height)
        + np.repeat(np.arange(height, dtype=np.uint16) * 3, width)
        + 11)
        .reshape(height, width)
        .astype(np.uint8)
    )
    return Image.fromarray(np.stack([xs, ys, zs], axis=-1))


def _gradient_png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    _gradient(width, height).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_source(width: int, height: int) -> Image.Image:
    xs = (
        np.tile((np.arange(width, dtype=np.uint16) * 5 + 1), height)
        .reshape(height, width)
        .astype(np.uint8)
    )
    ys = (
        np.repeat((np.arange(height, dtype=np.uint16) * 11 + 7), width)
        .reshape(height, width)
        .astype(np.uint8)
    )
    zs = (
        (np.tile(np.arange(width, dtype=np.uint16) + 256, height)
        - np.repeat(np.arange(height, dtype=np.uint16), width))
        .reshape(height, width)
        .astype(np.uint8)
    )
    return Image.fromarray(np.stack([xs, ys, zs], axis=-1))


def _buckets() -> tuple[BucketShape, ...]:
    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        transpose_closed=True,
    )
    return scale_buckets(generate_base_buckets(config), R)


BUCKETS = _buckets()


def _identity_adapter(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return metadata


def _pipeline(
    *,
    camera_policy: CameraViewportPolicy | None,
    min_crop_retention: float = 0.8,
    spatial_policy: SpatialCropPolicy | None = None,
    rejection_observer: Callable[[str], None] | None = None,
) -> WebDatasetPipeline:
    reasons: list[str] = []
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
    pipeline.min_crop_retention = min_crop_retention
    pipeline.rejection_observer = (
        (lambda reason: reasons.append(reason))  # type: ignore[misc]
        if rejection_observer is None
        else rejection_observer
    )
    pipeline._rejected_reasons = reasons  # type: ignore[attr-defined]
    pipeline.spatial_policy = spatial_policy
    pipeline.camera_policy = camera_policy
    pipeline._camera_stage_edge = R  # type: ignore[reportPrivateUsage, attr-defined]
    pipeline.transparent_policy = None
    pipeline.transparent_telemetry = TransparentWhiteTelemetry()  # type: ignore[reportAttributeAccessIssue]
    return cast(Any, pipeline)


def _process(
    pipeline: WebDatasetPipeline,
    image_bytes: bytes,
    *,
    sample_id: int = 1,
) -> PipelineSample | None:
    sample = {
        "__url__": _SHARD,
        "__key__": f"synthetic/{sample_id:06d}",
        "json": b'{"id": ' + str(sample_id).encode("ascii") + b"}",
        "png": image_bytes,
    }
    return pipeline._process(  # type: ignore[reportPrivateUsage]
        sample,
        {_SHARD: ShardRecord(path=_SHARD, bytes=1)},
    )


def _policy(enabled: bool, probability: float) -> CameraViewportPolicy:
    return CameraViewportPolicy(enabled=enabled, probability=probability)


class TestOffShapesBitIdentical:
    """A, incl. spatial-enabled configs: off camera == dev behavior."""

    OFF_SHAPES: tuple[CameraViewportPolicy | None, ...] = (
        None,
        _policy(False, 0.0),
        _policy(False, 1.0),
        _policy(True, 0.0),
    )

    @pytest.mark.parametrize("camera_policy", OFF_SHAPES)
    def test_off_camera_matches_absent_camera(
        self, camera_policy: CameraViewportPolicy | None
    ) -> None:
        image_bytes = _gradient_png(1024, 512)
        baseline = _process(_pipeline(camera_policy=None), image_bytes)
        assert baseline is not None
        result = _process(_pipeline(camera_policy=camera_policy), image_bytes)
        assert result is not None
        assert torch.equal(baseline.image, result.image)
        assert (baseline.target_height, baseline.target_width) == (
            result.target_height,
            result.target_width,
        )
        assert baseline.audit == result.audit
        assert result.audit.crop_policy == "aspect_bucket"
        assert result.audit.camera_applied is False
        assert result.audit.camera_selected is False

    @pytest.mark.parametrize("camera_policy", OFF_SHAPES)
    def test_off_camera_with_spatial_enabled(
        self, camera_policy: CameraViewportPolicy | None
    ) -> None:
        spatial = SpatialCropPolicy(
            enabled=True,
            probability=1.0,
            min_equivalent_zoom=1.02,
            max_equivalent_zoom=1.10,
            min_crop_retention=0.8,
        )
        image_bytes = _gradient_png(1024, 512)
        baseline = _process(
            _pipeline(camera_policy=None, spatial_policy=spatial), image_bytes
        )
        assert baseline is not None
        result = _process(
            _pipeline(camera_policy=camera_policy, spatial_policy=spatial),
            image_bytes,
        )
        assert result is not None
        assert torch.equal(baseline.image, result.image)
        assert baseline.audit == result.audit

    @pytest.mark.parametrize("camera_policy", OFF_SHAPES)
    def test_off_camera_keeps_ordinary_rejections(
        self, camera_policy: CameraViewportPolicy | None
    ) -> None:
        # A source the ordinary path rejects (retention at 0.9) stays
        # rejected for every effectively-off camera shape.
        image_bytes = _gradient_png(2560, 512)  # 5:1 -> retention 0.8
        for policy in (None, camera_policy):
            pipeline = _pipeline(camera_policy=policy, min_crop_retention=0.9)
            assert _process(pipeline, image_bytes) is None
            assert pipeline._rejected_reasons  # type: ignore[attr-defined]


class TestP100Geometry:
    @pytest.mark.parametrize(
        ("width", "height", "orientation"),
        [
            (512, 512, "square"),
            (1024, 512, "horizontal"),
            (1536, 512, "horizontal"),
            (2560, 512, "horizontal"),
            (1600, 1000, "horizontal"),
            (512, 1024, "vertical"),
            (512, 2560, "vertical"),
        ],
    )
    def test_applies_shifted_square_view(
        self, width: int, height: int, orientation: str
    ) -> None:
        result = _process(
            _pipeline(camera_policy=_policy(True, 1.0)),
            _gradient_png(width, height),
        )
        assert result is not None
        audit = result.audit
        assert audit.crop_policy == "camera_viewport"
        assert audit.camera_policy == "hdm_shifted_square_v2"
        assert audit.camera_selected is True
        assert audit.camera_applied is True
        assert audit.camera_fallback_reason == "none"
        assert audit.camera_orientation == orientation
        assert audit.spatial_applied is False
        assert (result.target_height, result.target_width) == (R, R)
        assert tuple(result.image.shape) == (3, R, R)
        assert audit.resized_width == audit.camera_full_width
        assert audit.resized_height == audit.camera_full_height
        left, top, right, bottom = audit.crop_box
        assert right - left == R
        assert bottom - top == R
        assert left + R <= audit.camera_full_width
        assert top + R <= audit.camera_full_height
        assert audit.camera_equivalent_zoom >= 1.0
        assert 0.0 < audit.camera_final_retention <= 1.0

    def test_square_source_is_identity_view(self) -> None:
        image_bytes = _gradient_png(512, 512)
        baseline = _process(_pipeline(camera_policy=None), image_bytes)
        result = _process(
            _pipeline(camera_policy=_policy(True, 1.0)), image_bytes
        )
        assert result is not None
        assert result.audit.camera_orientation == "square"
        assert result.audit.camera_equivalent_zoom == 1.0
        # Identity zoom + zero offset: the ordinary 512x512 bucket crop of
        # a 512x512 source is the whole image, exactly like the camera view.
        assert baseline is not None
        assert torch.equal(baseline.image, result.image)

    def test_boundary_short_edge_accepted(self) -> None:
        for width, height in ((1024, 512), (512, 1024)):
            result = _process(
                _pipeline(camera_policy=_policy(True, 1.0)),
                _gradient_png(width, height),
            )
            assert result is not None
            assert result.audit.camera_applied is True

    @pytest.mark.parametrize(("width", "height"), ((800, 511), (511, 800), (511, 511)))
    def test_no_upscale_rejects_below_target(
        self, width: int, height: int
    ) -> None:
        pipeline = _pipeline(camera_policy=_policy(True, 1.0))
        assert _process(pipeline, _gradient_png(width, height)) is None
        assert pipeline._rejected_reasons == ["camera_no_upscale"]  # type: ignore[attr-defined]

    def test_p100_has_no_legacy_fallback(self) -> None:
        # Every P100 source either applies or is rejected: the legacy
        # near_square_below_min / aspect_above_max / quantized_no_effect
        # states cannot occur.
        for width, height in ((600, 600), (1024, 512), (2560, 512), (800, 511)):
            pipeline = _pipeline(camera_policy=_policy(True, 1.0))
            result = _process(pipeline, _gradient_png(width, height))
            if result is None:
                assert pipeline._rejected_reasons == ["camera_no_upscale"]  # type: ignore[attr-defined]
                continue
            assert result.audit.camera_fallback_reason == "none"

    def test_not_selected_matches_baseline_bitwise(self) -> None:
        image_bytes = _gradient_png(1024, 512)
        baseline = _process(_pipeline(camera_policy=None), image_bytes)
        assert baseline is not None
        seen = False
        for sample_id in range(1, 64):
            result = _process(
                _pipeline(camera_policy=_policy(True, 0.25)),
                image_bytes,
                sample_id=sample_id,
            )
            assert result is not None
            if result.audit.camera_selected:
                continue
            assert torch.equal(baseline.image, result.image)
            assert result.audit.crop_policy == "aspect_bucket"
            assert result.audit.camera_fallback_reason == "not_selected"
            assert result.audit.camera_applied is False
            seen = True
            break
        assert seen, "expected at least one unselected draw in 64 samples"


class TestOrdinaryAdmissionDecoupling:
    def test_camera_path_never_calls_prepare_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        original = pipeline_module.prepare_image

        def counting_prepare_image(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return original(*args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(pipeline_module, "prepare_image", counting_prepare_image)
        image_bytes = _gradient_png(2560, 512)  # ordinary 5:1 retention 0.8
        # min_crop_retention=0.9 makes the ordinary path reject this source,
        # while the camera path must still emit the R x R view.
        pipeline = _pipeline(
            camera_policy=_policy(True, 1.0), min_crop_retention=0.9
        )
        result = _process(pipeline, image_bytes)
        assert result is not None
        assert result.audit.camera_applied is True
        assert calls == []

    def test_off_path_still_calls_prepare_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        original = pipeline_module.prepare_image

        def counting_prepare_image(*args: object, **kwargs: object) -> object:
            calls.append(1)
            return original(*args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(pipeline_module, "prepare_image", counting_prepare_image)
        image_bytes = _gradient_png(1024, 512)
        for policy in (None, _policy(False, 1.0), _policy(True, 0.0)):
            pipeline = _pipeline(camera_policy=policy, min_crop_retention=0.9)
            _process(pipeline, image_bytes)
            assert calls[-1:] == [1]
            calls.clear()

    def test_retention_rejection_persists_when_off(self) -> None:
        image_bytes = _gradient_png(2560, 512)
        for policy in (None, _policy(True, 0.0)):
            pipeline = _pipeline(camera_policy=policy, min_crop_retention=0.9)
            assert _process(pipeline, image_bytes) is None


class TestImageCorrectness:
    def test_output_is_scale_then_crop_not_direct_patch(self) -> None:
        width, height = 1600, 1000
        image_bytes = _gradient_png(width, height)
        pipeline = _pipeline(camera_policy=_policy(True, 1.0))
        result = _process(pipeline, image_bytes)
        assert result is not None
        left, top, right, bottom = result.audit.crop_box
        # Independent reference: EXIF-normalize (no-op for RGB PNG), scale
        # the full image to the planned canvas, crop the planned box.
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        expected = source.resize(
            (result.audit.resized_width, result.audit.resized_height),
            resample=Image.Resampling.LANCZOS,
        ).crop((left, top, right, bottom))
        expected_tensor = torch.from_numpy(np.asarray(expected)).permute(2, 0, 1)
        assert torch.equal(result.image, expected_tensor)
        # A direct R x R patch from the unscaled source is a different image.
        direct = source.crop((0, 0, R, R))
        direct_tensor = torch.from_numpy(np.asarray(direct)).permute(2, 0, 1)
        assert not torch.equal(result.image, direct_tensor)

    def test_exif_rotation_plans_on_post_exif_size(self) -> None:
        # Stored 1000x2000 landscape pixels with EXIF orientation 6 ->
        # post-EXIF 2000x1000: the plan must use the post-EXIF size.
        source = _gradient(1000, 2000)
        exif = Image.Exif()
        exif[0x0112] = 6  # orientation: rotate 270 CW
        buffer = io.BytesIO()
        source.save(buffer, format="PNG", exif=exif.tobytes())
        image_bytes = buffer.getvalue()

        result = _process(
            _pipeline(camera_policy=_policy(True, 1.0)), image_bytes
        )
        assert result is not None
        assert (result.audit.source_width, result.audit.source_height) == (
            2000,
            1000,
        )
        assert result.audit.camera_orientation == "horizontal"
        assert (result.audit.camera_full_width, result.audit.camera_full_height) == (
            1024,
            R,
        )
        assert tuple(result.image.shape) == (3, R, R)

    def test_jpeg_draft_large_source_applies(self) -> None:
        width, height = 5000, 4000  # 20 MP >= the 16 MP draft threshold
        image = _jpeg_source(width, height)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()
        result = _process(
            _pipeline(camera_policy=_policy(True, 1.0)), image_bytes
        )
        assert result is not None
        assert result.audit.camera_applied is True
        # post-EXIF 5000x4000 -> canvas round(5000*512/4000)=640 x 512
        assert (result.audit.camera_full_width, result.audit.camera_full_height) == (
            640,
            R,
        )
        assert tuple(result.image.shape) == (3, R, R)
        assert torch.isfinite(result.image.float()).all()

    def test_draft_coverage_redecodes_when_undersized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force every JPEG draft through the smallest reduction PIL allows
        # (decode comes back below the planned canvas on the long axis), so
        # the pipeline must re-decode the original compressed bytes in full;
        # the output must then equal the full-decode scale-then-crop
        # reference.
        original_draft = Image.Image.draft  # type: ignore[attr-defined]

        def forcing_draft(self, mode, size):  # type: ignore[no-untyped-def]
            return original_draft(self, mode, (100, 100))

        monkeypatch.setattr(Image.Image, "draft", forcing_draft)
        width, height = 5000, 4000
        image = _jpeg_source(width, height)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()
        result = _process(
            _pipeline(camera_policy=_policy(True, 1.0)), image_bytes
        )
        assert result is not None
        left, top, right, bottom = result.audit.crop_box
        reference = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        expected = reference.resize(
            (result.audit.resized_width, result.audit.resized_height),
            resample=Image.Resampling.LANCZOS,
        ).crop((left, top, right, bottom))
        expected_tensor = torch.from_numpy(np.asarray(expected)).permute(2, 0, 1)
        assert torch.equal(result.image, expected_tensor)
