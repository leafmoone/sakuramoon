from __future__ import annotations

import pytest

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import (
    BucketAssignment,
    BucketError,
    BucketRejection,
    BucketSampleCount,
    BucketScanReport,
    BucketShape,
    SourceDimensions,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
    scan_bucket_assignments,
)


def _config() -> DataBucketsConfig:
    return DataBucketsConfig.model_validate(
        {
            "base_area_px": 262144,
            "quantum_px": 32,
            "min_short_edge_px": 256,
            "max_aspect_ratio": 4.0,
            "shape_count": 17,
            "transpose_closed": True,
        },
        strict=True,
    )


def test_generate_exact_17_base_buckets() -> None:
    buckets = generate_base_buckets(_config())

    assert tuple((shape.height, shape.width) for shape in buckets) == (
        (1024, 256),
        (896, 288),
        (832, 320),
        (736, 352),
        (672, 384),
        (640, 416),
        (576, 448),
        (544, 480),
        (512, 512),
        (480, 544),
        (448, 576),
        (416, 640),
        (384, 672),
        (352, 736),
        (320, 832),
        (288, 896),
        (256, 1024),
    )
    assert all(
        BucketShape(shape.width, shape.height) in buckets for shape in buckets
    )
    assert max(abs(shape.area - 512**2) / 512**2 for shape in buckets) < 0.025


@pytest.mark.parametrize("edge", [256, 512, 768, 1024])
def test_scale_preserves_same_shape_family(edge: int) -> None:
    base = generate_base_buckets(_config())
    scaled = scale_buckets(base, edge)  # type: ignore[arg-type]

    assert len(scaled) == 17
    assert tuple((shape.height, shape.width) for shape in scaled) == tuple(
        (shape.height * edge // 512, shape.width * edge // 512) for shape in base
    )
    assert all(BucketShape(shape.width, shape.height) in scaled for shape in scaled)


def test_scale_rejects_unapproved_resolution() -> None:
    with pytest.raises(BucketError, match="approved"):
        scale_buckets(generate_base_buckets(_config()), 640)  # type: ignore[arg-type]


def test_assignment_filters_no_upscale_then_uses_nearest_aspect() -> None:
    buckets = generate_base_buckets(_config())

    result = assign_bucket(1024, 500, buckets, min_crop_retention=0.5)

    assert isinstance(result, BucketAssignment)
    assert result.bucket == BucketShape(height=352, width=736)
    assert result.resized_width >= result.bucket.width
    assert result.resized_height >= result.bucket.height


def test_assignment_rejects_when_no_bucket_fits_without_upscale() -> None:
    result = assign_bucket(
        500,
        500,
        generate_base_buckets(_config()),
        min_crop_retention=0.8,
    )
    assert result == BucketRejection("no_upscale")


def test_assignment_rejects_excessive_cover_crop() -> None:
    result = assign_bucket(
        2000,
        300,
        generate_base_buckets(_config()),
        min_crop_retention=0.8,
    )
    assert result == BucketRejection("retention")


def test_retention_threshold_is_inclusive() -> None:
    result = assign_bucket(
        640,
        512,
        (BucketShape(512, 512),),
        min_crop_retention=0.8,
    )

    assert isinstance(result, BucketAssignment)
    assert result.resized_width == 640
    assert result.resized_height == 512
    assert result.crop_retention == 0.8


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1, 1, True],
)
def test_assignment_rejects_invalid_retention_threshold(threshold: object) -> None:
    with pytest.raises(BucketError, match="finite float"):
        assign_bucket(
            640,
            512,
            (BucketShape(512, 512),),
            min_crop_retention=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("width,height", [(0, 10), (10, -1), (True, 10)])
def test_assignment_rejects_invalid_dimensions(width: int, height: int) -> None:
    with pytest.raises(BucketError, match="positive integers"):
        assign_bucket(
            width,
            height,
            (BucketShape(512, 512),),
            min_crop_retention=0.8,
        )


def test_streaming_bucket_scan_counts_assignments_and_rejections() -> None:
    square = BucketShape(height=512, width=512)
    wide = BucketShape(height=256, width=1024)
    dimensions = (
        SourceDimensions(640, 512),
        SourceDimensions(500, 500),
        SourceDimensions(2000, 300),
        SourceDimensions(512, 512),
    )

    report = scan_bucket_assignments(
        (item for item in dimensions),
        (square, wide),
        min_crop_retention=0.8,
        expected_samples=4,
    )

    assert report == BucketScanReport(
        expected_samples=4,
        total_samples=4,
        assigned_samples=2,
        no_upscale_rejections=1,
        retention_rejections=1,
        bucket_counts=(
            BucketSampleCount(height=512, width=512, samples=2),
            BucketSampleCount(height=256, width=1024, samples=0),
        ),
    )
    assert tuple(
        (item.height, item.width, item.samples) for item in report.bucket_counts
    ) == ((512, 512, 2), (256, 1024, 0))


def test_bucket_scan_requires_exact_manifest_count_and_unique_shapes() -> None:
    shape = BucketShape(512, 512)
    with pytest.raises(BucketError, match="does not match"):
        scan_bucket_assignments(
            (SourceDimensions(512, 512),),
            (shape,),
            min_crop_retention=0.8,
            expected_samples=2,
        )
    with pytest.raises(BucketError, match="exceeds"):
        scan_bucket_assignments(
            (SourceDimensions(512, 512), SourceDimensions(512, 512)),
            (shape,),
            min_crop_retention=0.8,
            expected_samples=1,
        )
    with pytest.raises(BucketError, match="unique"):
        scan_bucket_assignments(
            (SourceDimensions(512, 512),),
            (shape, shape),
            min_crop_retention=0.8,
            expected_samples=1,
        )


@pytest.mark.parametrize("height,width", [(0, 1), (1, -1), (True, 1)])
def test_bucket_shape_rejects_invalid_dimensions(height: int, width: int) -> None:
    with pytest.raises(BucketError, match="bucket dimensions"):
        BucketShape(height, width)


def test_bucket_scan_report_rejects_noncanonical_counts() -> None:
    with pytest.raises(BucketError, match="sample counts"):
        BucketSampleCount(height=512, width=512, samples=-1)

    duplicate = BucketSampleCount(height=512, width=512, samples=1)
    with pytest.raises(BucketError, match="inconsistent"):
        BucketScanReport(
            expected_samples=2,
            total_samples=2,
            assigned_samples=2,
            no_upscale_rejections=0,
            retention_rejections=0,
            bucket_counts=(duplicate, duplicate),
        )
