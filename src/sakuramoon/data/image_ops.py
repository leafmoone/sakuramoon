"""EXIF-aware RGB normalization and deterministic cover resize/crop."""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageOps

from sakuramoon.data.buckets import (
    BucketAssignment,
    BucketRejection,
    BucketShape,
    RejectionReason,
    assign_bucket,
)


class ImageRejected(ValueError):
    """Image cannot enter the selected stage without upscale or excessive crop."""

    def __init__(self, reason: RejectionReason) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ProcessedImage:
    image: Image.Image
    assignment: BucketAssignment
    crop_box: tuple[int, int, int, int]


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation once, then convert to RGB."""

    return ImageOps.exif_transpose(image).convert("RGB")


def resize_and_crop(
    image: Image.Image,
    assignment: BucketAssignment,
    *,
    crop_seed: int,
) -> ProcessedImage:
    """Resize by cover and choose one deterministic uniform crop offset."""

    if image.size != (assignment.source_width, assignment.source_height):
        raise ValueError("image dimensions differ from the bucket assignment")
    resized = image.resize(
        (assignment.resized_width, assignment.resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    max_left = assignment.resized_width - assignment.bucket.width
    max_top = assignment.resized_height - assignment.bucket.height
    generator = random.Random(crop_seed)
    left = generator.randrange(max_left + 1)
    top = generator.randrange(max_top + 1)
    box = (
        left,
        top,
        left + assignment.bucket.width,
        top + assignment.bucket.height,
    )
    return ProcessedImage(
        image=resized.crop(box), assignment=assignment, crop_box=box
    )


def prepare_image(
    image: Image.Image,
    buckets: tuple[BucketShape, ...],
    *,
    min_crop_retention: float,
    crop_seed: int,
) -> ProcessedImage:
    """Normalize, route, resize, and crop one decoded image."""

    normalized = normalize_image(image)
    assignment = assign_bucket(
        normalized.width,
        normalized.height,
        buckets,
        min_crop_retention=min_crop_retention,
    )
    if isinstance(assignment, BucketRejection):
        raise ImageRejected(assignment.reason)
    return resize_and_crop(normalized, assignment, crop_seed=crop_seed)
