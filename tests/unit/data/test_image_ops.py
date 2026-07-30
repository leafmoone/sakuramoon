from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from sakuramoon.data.buckets import (
    BucketAssignment,
    BucketShape,
    SourceDimensions,
    scan_bucket_assignments,
)
from sakuramoon.data.image_ops import (
    DECODE_DIMENSION_SCAN_SAMPLES,
    DecodedDimensionObservation,
    DimensionMismatchError,
    ImageRejected,
    ImageScanError,
    ImageScanReport,
    ImageScanReportExistsError,
    canonical_image_scan_report_bytes,
    normalize_image,
    observe_decoded_dimensions,
    prepare_image,
    resize_and_crop,
    scan_decoded_dimensions,
    write_image_scan_report,
)


def test_normalize_applies_exif_orientation_then_rgb() -> None:
    image = Image.new("L", (40, 20), color=128)
    image.getexif()[274] = 6

    normalized = normalize_image(image)

    assert normalized.size == (20, 40)
    assert normalized.mode == "RGB"


def test_resize_crop_is_deterministic_for_seed() -> None:
    image = Image.new("RGB", (120, 80))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x, y, (x + y) % 256))
    assignment = BucketAssignment(
        source_width=120,
        source_height=80,
        bucket=BucketShape(height=64, width=64),
        resized_width=96,
        resized_height=64,
        crop_retention=2 / 3,
    )

    first = resize_and_crop(image, assignment, crop_seed=123)
    repeated = resize_and_crop(image, assignment, crop_seed=123)
    different = resize_and_crop(image, assignment, crop_seed=456)

    assert first.crop_box == repeated.crop_box
    assert first.image.tobytes() == repeated.image.tobytes()
    assert first.crop_box != different.crop_box
    assert first.image.size == (64, 64)


def test_resize_crop_rejects_dimension_mismatch() -> None:
    image = Image.new("RGB", (100, 80))
    assignment = BucketAssignment(
        source_width=120,
        source_height=80,
        bucket=BucketShape(height=64, width=64),
        resized_width=96,
        resized_height=64,
        crop_retention=2 / 3,
    )
    with pytest.raises(ValueError, match="differ"):
        resize_and_crop(image, assignment, crop_seed=1)


def test_prepare_image_runs_normalize_route_resize_crop() -> None:
    image = Image.new("L", (640, 512), color=127)

    result = prepare_image(
        image,
        (BucketShape(512, 512),),
        min_crop_retention=0.8,
        crop_seed=9,
    )

    assert result.image.mode == "RGB"
    assert result.image.size == (512, 512)
    assert result.assignment.source_width == 640
    assert result.assignment.crop_retention == 0.8


def test_prepare_image_uses_post_exif_dimensions_for_routing() -> None:
    image = Image.new("RGB", (640, 512))
    image.getexif()[274] = 6

    result = prepare_image(
        image,
        (BucketShape(height=640, width=512),),
        min_crop_retention=0.8,
        crop_seed=4,
    )

    assert result.assignment.source_width == 512
    assert result.assignment.source_height == 640
    assert result.image.size == (512, 640)


def test_prepare_image_exposes_no_upscale_rejection() -> None:
    image = Image.new("RGB", (500, 500))

    with pytest.raises(ImageRejected) as captured:
        prepare_image(
            image,
            (BucketShape(512, 512),),
            min_crop_retention=0.8,
            crop_seed=1,
        )

    assert captured.value.reason == "no_upscale"


def test_prepare_image_exposes_retention_rejection() -> None:
    image = Image.new("RGB", (2000, 300))

    with pytest.raises(ImageRejected) as captured:
        prepare_image(
            image,
            (BucketShape(height=256, width=1024),),
            min_crop_retention=0.8,
            crop_seed=1,
        )

    assert captured.value.reason == "retention"


def test_decode_dimension_observation_uses_post_exif_size() -> None:
    image = Image.new("RGB", (640, 512))
    image.getexif()[274] = 6

    observation = observe_decoded_dimensions(
        image,
        declared_width=512,
        declared_height=640,
    )

    assert observation.matches is True
    assert (observation.decoded_width, observation.decoded_height) == (512, 640)


def _dimension_observations(
    mismatch_count: int,
    *,
    total: int = DECODE_DIMENSION_SCAN_SAMPLES,
) -> Iterator[DecodedDimensionObservation]:
    for index in range(total):
        yield DecodedDimensionObservation(
            declared_width=512,
            declared_height=512,
            decoded_width=513 if index < mismatch_count else 512,
            decoded_height=512,
        )


def test_dimension_scan_accepts_exact_point_one_percent_boundary() -> None:
    report = scan_decoded_dimensions(_dimension_observations(100))

    assert report.total_samples == DECODE_DIMENSION_SCAN_SAMPLES
    assert report.mismatch_samples == 100
    assert report.mismatch_rate == 0.001
    assert report.accepted is True


def test_dimension_scan_hard_fails_above_threshold_with_report() -> None:
    with pytest.raises(DimensionMismatchError, match="0.1 percent") as captured:
        scan_decoded_dimensions(_dimension_observations(101))

    assert captured.value.report.mismatch_samples == 101
    assert captured.value.report.accepted is False


def test_dimension_scan_requires_exact_100k_observations() -> None:
    with pytest.raises(ImageScanError, match="exactly 100,000"):
        scan_decoded_dimensions(
            _dimension_observations(0, total=DECODE_DIMENSION_SCAN_SAMPLES - 1)
        )
    with pytest.raises(ImageScanError, match="exceeds 100,000"):
        scan_decoded_dimensions(
            _dimension_observations(0, total=DECODE_DIMENSION_SCAN_SAMPLES + 1)
        )


def test_image_scan_report_is_canonical_fsynced_and_no_clobber(
    tmp_path: Path,
) -> None:
    bucket_report = scan_bucket_assignments(
        (SourceDimensions(512, 512), SourceDimensions(500, 500)),
        (BucketShape(512, 512),),
        min_crop_retention=0.8,
        expected_samples=2,
    )
    dimension_report = scan_decoded_dimensions(_dimension_observations(0))
    report = ImageScanReport(bucket_report, dimension_report)
    payload = canonical_image_scan_report_bytes(report)
    destination = tmp_path / "image-scan.json"

    digest = write_image_scan_report(report, destination)

    assert destination.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    document = json.loads(payload)
    assert document["schema_version"] == 1
    assert document["bucket_scan"]["assigned_samples"] == 1
    assert document["bucket_scan"]["no_upscale_rejections"] == 1
    assert document["dimension_scan"]["accepted"] is True
    with pytest.raises(ImageScanReportExistsError, match="already exists"):
        write_image_scan_report(report, destination)
    assert destination.read_bytes() == payload
