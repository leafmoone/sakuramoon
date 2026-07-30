from __future__ import annotations

import pytest
from PIL import Image

from sakuramoon.data.buckets import BucketAssignment, BucketShape
from sakuramoon.data.image_ops import (
    ImageRejected,
    normalize_image,
    prepare_image,
    resize_and_crop,
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
