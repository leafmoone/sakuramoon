"""Generate and select the locked 17 near-equal-area image buckets."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sakuramoon.config.schema import DataBucketsConfig

StageEdge = Literal[256, 512, 768, 1024]
RejectionReason = Literal["no_upscale", "retention"]
RETENTION_HISTOGRAM_SCALE = 10_000
RETENTION_HISTOGRAM_BINS = RETENTION_HISTOGRAM_SCALE + 1


class BucketError(ValueError):
    """Bucket configuration or image dimensions are invalid."""


@dataclass(frozen=True, order=True)
class BucketShape:
    height: int
    width: int

    def __post_init__(self) -> None:
        if (
            type(self.height) is not int
            or type(self.width) is not int
            or self.height <= 0
            or self.width <= 0
        ):
            raise BucketError("bucket dimensions must be positive integers")

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def aspect_log2(self) -> float:
        return math.log2(self.width / self.height)


@dataclass(frozen=True)
class BucketAssignment:
    source_width: int
    source_height: int
    bucket: BucketShape
    resized_width: int
    resized_height: int
    crop_retention: float


@dataclass(frozen=True)
class BucketRejection:
    reason: RejectionReason


@dataclass(frozen=True)
class SourceDimensions:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width <= 0
            or self.height <= 0
        ):
            raise BucketError("source dimensions must be positive integers")


@dataclass(frozen=True)
class BucketSampleCount:
    height: int
    width: int
    samples: int

    def __post_init__(self) -> None:
        if (
            type(self.height) is not int
            or type(self.width) is not int
            or type(self.samples) is not int
            or self.height <= 0
            or self.width <= 0
            or self.samples < 0
        ):
            raise BucketError("bucket sample counts are invalid")


@dataclass(frozen=True)
class CropRetentionQuantiles:
    p01: float
    p50: float
    p99: float
    histogram_resolution: float

    def __post_init__(self) -> None:
        values = (self.p01, self.p50, self.p99, self.histogram_resolution)
        if (
            any(type(value) is not float or not math.isfinite(value) for value in values)
            or not 0.0 <= self.p01 <= self.p50 <= self.p99 <= 1.0
            or self.histogram_resolution != 1.0 / RETENTION_HISTOGRAM_SCALE
        ):
            raise BucketError("crop retention quantiles are invalid")


@dataclass(frozen=True)
class BucketScanReport:
    expected_samples: int
    total_samples: int
    assigned_samples: int
    no_upscale_rejections: int
    retention_rejections: int
    bucket_counts: tuple[BucketSampleCount, ...]
    crop_retention_quantiles: CropRetentionQuantiles | None

    def __post_init__(self) -> None:
        rejection_total = self.no_upscale_rejections + self.retention_rejections
        shapes = tuple(BucketShape(item.height, item.width) for item in self.bucket_counts)
        expected_order = tuple(
            sorted(shapes, key=lambda shape: (shape.aspect_log2, shape.height, shape.width))
        )
        if (
            type(self.expected_samples) is not int
            or self.expected_samples <= 0
            or type(self.total_samples) is not int
            or type(self.assigned_samples) is not int
            or type(self.no_upscale_rejections) is not int
            or type(self.retention_rejections) is not int
            or self.total_samples != self.expected_samples
            or self.assigned_samples < 0
            or self.no_upscale_rejections < 0
            or self.retention_rejections < 0
            or not shapes
            or shapes != expected_order
            or len(set(shapes)) != len(shapes)
            or self.assigned_samples + rejection_total != self.total_samples
            or sum(item.samples for item in self.bucket_counts)
            != self.assigned_samples
            or (self.crop_retention_quantiles is None) != (self.assigned_samples == 0)
        ):
            raise BucketError("bucket scan report counts are inconsistent")


def _nearest_quantum_ratio(area: int, short: int, quantum: int) -> int:
    denominator = short * quantum
    units = (2 * area + denominator) // (2 * denominator)
    return units * quantum


def generate_base_buckets(config: DataBucketsConfig) -> tuple[BucketShape, ...]:
    """Generate the configured 512-equivalent transpose-closed shape set."""

    edge = math.isqrt(config.base_area_px)
    if edge * edge != config.base_area_px:
        raise BucketError("bucket base area must be a square")
    shapes: set[BucketShape] = set()
    for short in range(config.min_short_edge_px, edge + 1, config.quantum_px):
        long = _nearest_quantum_ratio(
            config.base_area_px, short, config.quantum_px
        )
        if long / short > config.max_aspect_ratio:
            continue
        shapes.add(BucketShape(height=long, width=short))
        shapes.add(BucketShape(height=short, width=long))
    ordered = tuple(sorted(shapes, key=lambda shape: (shape.aspect_log2, shape.height)))
    if len(ordered) != config.shape_count:
        raise BucketError("bucket parameters do not generate the configured shape count")
    if any(BucketShape(shape.width, shape.height) not in shapes for shape in ordered):
        raise BucketError("bucket shapes are not transpose closed")
    return ordered


def scale_buckets(
    base_buckets: tuple[BucketShape, ...], target_edge: StageEdge
) -> tuple[BucketShape, ...]:
    """Scale the same shape family from the 512-equivalent base."""

    if target_edge not in (256, 512, 768, 1024):
        raise BucketError("target edge is not an approved stage resolution")
    scaled: list[BucketShape] = []
    for shape in base_buckets:
        height_numerator = shape.height * target_edge
        width_numerator = shape.width * target_edge
        if height_numerator % 512 or width_numerator % 512:
            raise BucketError("bucket cannot be scaled exactly to the target resolution")
        scaled.append(
            BucketShape(
                height=height_numerator // 512,
                width=width_numerator // 512,
            )
        )
    return tuple(scaled)


def _cover_size(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    if target_width * source_height >= target_height * source_width:
        numerator, denominator = target_width, source_width
    else:
        numerator, denominator = target_height, source_height
    resized_width = (source_width * numerator + denominator - 1) // denominator
    resized_height = (source_height * numerator + denominator - 1) // denominator
    return resized_width, resized_height


def assign_bucket(
    source_width: int,
    source_height: int,
    buckets: tuple[BucketShape, ...],
    *,
    min_crop_retention: float,
) -> BucketAssignment | BucketRejection:
    """Apply no-upscale, nearest-aspect, cover, then retention in locked order."""

    if (
        type(source_width) is not int
        or type(source_height) is not int
        or source_width <= 0
        or source_height <= 0
    ):
        raise BucketError("source dimensions must be positive integers")
    if (
        type(min_crop_retention) is not float
        or not math.isfinite(min_crop_retention)
        or not 0.0 <= min_crop_retention <= 1.0
    ):
        raise BucketError("minimum crop retention must be a finite float in [0, 1]")
    if not buckets:
        raise BucketError("at least one bucket is required")
    eligible = tuple(
        shape
        for shape in buckets
        if shape.width <= source_width and shape.height <= source_height
    )
    if not eligible:
        return BucketRejection("no_upscale")
    source_aspect = math.log2(source_width / source_height)
    selected = min(
        eligible,
        key=lambda shape: (
            abs(shape.aspect_log2 - source_aspect),
            shape.height,
            shape.width,
        ),
    )
    resized_width, resized_height = _cover_size(
        source_width,
        source_height,
        selected.width,
        selected.height,
    )
    retention = selected.area / (resized_width * resized_height)
    if retention < min_crop_retention:
        return BucketRejection("retention")
    return BucketAssignment(
        source_width=source_width,
        source_height=source_height,
        bucket=selected,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_retention=retention,
    )


def scan_bucket_assignments(
    dimensions: Iterable[SourceDimensions],
    buckets: tuple[BucketShape, ...],
    *,
    min_crop_retention: float,
    expected_samples: int,
) -> BucketScanReport:
    """Stream a complete manifest dimension scan without retaining sample rows."""

    if type(expected_samples) is not int or expected_samples <= 0:
        raise BucketError("expected sample count must be a positive integer")
    if len(set(buckets)) != len(buckets):
        raise BucketError("bucket scan shapes must be unique")
    ordered_buckets = tuple(
        sorted(buckets, key=lambda shape: (shape.aspect_log2, shape.height, shape.width))
    )
    counts = {shape: 0 for shape in ordered_buckets}
    retention_histogram = [0] * RETENTION_HISTOGRAM_BINS
    total = 0
    no_upscale = 0
    retention = 0
    for item in dimensions:
        total += 1
        if total > expected_samples:
            raise BucketError("bucket scan exceeds the expected sample count")
        result = assign_bucket(
            item.width,
            item.height,
            buckets,
            min_crop_retention=min_crop_retention,
        )
        if isinstance(result, BucketAssignment):
            counts[result.bucket] += 1
            histogram_index = min(
                int(result.crop_retention * RETENTION_HISTOGRAM_SCALE),
                RETENTION_HISTOGRAM_SCALE,
            )
            retention_histogram[histogram_index] += 1
        elif result.reason == "no_upscale":
            no_upscale += 1
        else:
            retention += 1
    if total != expected_samples:
        raise BucketError("bucket scan does not match the expected sample count")
    bucket_counts = tuple(
        BucketSampleCount(shape.height, shape.width, counts[shape])
        for shape in ordered_buckets
    )
    assigned_samples = sum(counts.values())

    def nearest_rank_quantile(quantile: float) -> float:
        rank = math.ceil(quantile * assigned_samples)
        cumulative = 0
        for index, samples in enumerate(retention_histogram):
            cumulative += samples
            if cumulative >= rank:
                return index / RETENTION_HISTOGRAM_SCALE
        raise BucketError("crop retention histogram is inconsistent")

    crop_retention_quantiles = None
    if assigned_samples:
        crop_retention_quantiles = CropRetentionQuantiles(
            p01=nearest_rank_quantile(0.01),
            p50=nearest_rank_quantile(0.50),
            p99=nearest_rank_quantile(0.99),
            histogram_resolution=1.0 / RETENTION_HISTOGRAM_SCALE,
        )
    return BucketScanReport(
        expected_samples=expected_samples,
        total_samples=total,
        assigned_samples=assigned_samples,
        no_upscale_rejections=no_upscale,
        retention_rejections=retention,
        bucket_counts=bucket_counts,
        crop_retention_quantiles=crop_retention_quantiles,
    )
