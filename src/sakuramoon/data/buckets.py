"""Generate and select the locked 17 near-equal-area image buckets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from sakuramoon.config.schema import DataBucketsConfig

StageEdge = Literal[256, 512, 768, 1024]
RejectionReason = Literal["no_upscale", "retention"]


class BucketError(ValueError):
    """Bucket configuration or image dimensions are invalid."""


@dataclass(frozen=True, order=True)
class BucketShape:
    height: int
    width: int

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
