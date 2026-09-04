"""Performance audit for the camera-viewport data path.

Measures decode + resize + crop wall time for the ORDINARY aspect-bucket
path and the CAMERA path over representative geometries (256/512 square,
2:1 horizontal, 2:1 vertical) in JPEG and PNG. The camera path must
contain exactly ONE resize (source -> full canvas); a second resize of the
ordinary-resized image is the double-resize regression this audit guards.
Resize/crop call counts are instrumented by wrapping Image.resize/crop.

Output: reports/camera-viewport-v2-performance.json

Usage:
    python scripts/audit_camera_viewport_performance.py \
        --output reports/camera-viewport-v2-performance.json
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from PIL import Image

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import (
    BucketRejection,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.camera_viewport import (
    CameraViewportPolicy,
    plan_camera_viewport,
)
from sakuramoon.data.image_ops import normalize_image, prepare_image

STAGE_EDGE = 512
MIN_CROP_RETENTION = 0.8
WARMUP = 3
REPEATS = 10

_GEOMETRIES: tuple[tuple[str, int, int], ...] = (
    ("square_256", 256, 256),
    ("square_512", 512, 512),
    ("wide_2to1", 1024, 512),
    ("tall_2to1", 512, 1024),
)


def _encode(width: int, height: int, fmt: str) -> bytes:
    """Synthetic photographic-ish content (gradient + noise-free blocks)."""

    import random

    rng = random.Random(f"{fmt}\0{width}x{height}")
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            block_x = (x // 32) * 32
            block_y = (y // 32) * 32
            seed_value = rng.random()
            r = int(255 * ((block_x * 7 + block_y * 13 + int(seed_value * 512)) % 251) / 251)
            g = int(255 * ((block_x * 11 + block_y * 5 + int(seed_value * 251)) % 251) / 251)
            b = int(255 * ((block_x * 3 + block_y * 17 + int(seed_value * 127)) % 251) / 251)
            pixels[x, y] = (r, g, b)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=90)
    return buffer.getvalue()


class _ResizeCropCounter:
    def __init__(self) -> None:
        self.resizes = 0
        self.crops = 0

    def __enter__(self) -> Self:
        # resize/crop are Image.Image class methods (no module-level attrs).
        self._orig_resize = Image.Image.resize
        self._orig_crop = Image.Image.crop

        def _counted_resize(self_image, *args, **kwargs):
            self.resizes += 1
            return self._orig_resize(self_image, *args, **kwargs)

        def _counted_crop(self_image, *args, **kwargs):
            self.crops += 1
            return self._orig_crop(self_image, *args, **kwargs)

        Image.Image.resize = _counted_resize
        Image.Image.crop = _counted_crop
        return self

    def __exit__(self, *exc) -> None:
        Image.Image.resize = self._orig_resize
        Image.Image.crop = self._orig_crop


def _time(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats * 1000.0


def _ordinary_path(
    decoded: Image.Image,
    buckets: tuple,
    source_size: tuple[int, int],
):
    return prepare_image(
        decoded,
        buckets,
        min_crop_retention=MIN_CROP_RETENTION,
        crop_seed=1,
        source_size=source_size,
    )


def _camera_path(decoded: Image.Image, plan):
    if not plan.applied:
        return None
    normalized = normalize_image(decoded)
    return normalized.resize(
        (plan.full_width, plan.full_height),
        resample=Image.Resampling.LANCZOS,
    ).crop(plan.crop_box)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/camera-viewport-v2-performance.json"),
    )
    args = parser.parse_args(argv)

    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        shape_count=17,
        transpose_closed=True,
    )
    buckets = scale_buckets(generate_base_buckets(config), STAGE_EDGE)
    policy = CameraViewportPolicy(
        enabled=True,
        probability=1.0,
        min_equivalent_zoom=1.10,
        max_equivalent_zoom=1.50,
    )

    results: list[dict[str, object]] = []
    double_resize_seen = False
    for geometry, width, height in _GEOMETRIES:
        for fmt in ("JPEG", "PNG"):
            image_bytes = _encode(width, height, fmt)
            decoded = Image.open(io.BytesIO(image_bytes))
            source_width, source_height = decoded.size
            assignment = assign_bucket(
                source_width,
                source_height,
                buckets,
                min_crop_retention=MIN_CROP_RETENTION,
            )
            if isinstance(assignment, BucketRejection):
                                # Sub-stage sources (e.g. 256px at stage edge 512) fail the
                                # ordinary retention gate and never reach the camera path;
                                # record and skip instead of timing a dead path.
                                results.append(
                                    {
                                        "geometry": geometry,
                                        "format": fmt,
                                        "source": f"{width}x{height}",
                                        "admissible": False,
                                        "rejection_reason": str(assignment.reason),
                                    }
                                )
                                print(
                                    f"[camera-perf] {geometry}/{fmt} rejected: {assignment.reason}",
                                    flush=True,
                                )
                                continue

            ordinary_path = functools.partial(
                _ordinary_path,
                decoded,
                buckets,
                (source_width, source_height),
            )
            plan = plan_camera_viewport(
                assignment,
                policy,
                buckets=buckets,
                stage_edge=STAGE_EDGE,
                source_size=(source_width, source_height),
                policy_seed=1,
                offset_seed=1,
            )
            applied = plan.applied
            camera_path = functools.partial(
                _camera_path,
                decoded,
                plan,
            )

            with _ResizeCropCounter() as counter:
                ordinary_ms = _time(ordinary_path, WARMUP, REPEATS)
                ordinary_resizes, ordinary_crops = counter.resizes, counter.crops
            with _ResizeCropCounter() as counter:
                if applied:
                    camera_ms = _time(camera_path, WARMUP, REPEATS)
                    camera_resizes, camera_crops = counter.resizes, counter.crops
                else:
                    camera_ms = 0.0
                    camera_resizes = 0
                    camera_crops = 0
            calls = WARMUP + REPEATS
            # Exactly one resize per sampled image on the camera path
            # (source -> full canvas); any other per-image count is the
            # double-resize regression this audit guards.
            if applied and camera_resizes != calls:
                double_resize_seen = True
            results.append(
                {
                    "geometry": geometry,
                    "format": fmt,
                    "source": f"{width}x{height}",
                    "admissible": True,
                    "ordinary_bucket": f"{assignment.bucket.width}x{assignment.bucket.height}",
                    "camera_applied": applied,
                    "camera_full_canvas": (
                        f"{plan.full_width}x{plan.full_height}" if applied else None
                    ),
                    "ordinary_ms": round(ordinary_ms, 4),
                    "ordinary_resize_calls": ordinary_resizes,
                    "ordinary_crop_calls": ordinary_crops,
                    "camera_ms": round(camera_ms, 4),
                    "camera_resize_calls": camera_resizes,
                    "camera_crop_calls": camera_crops,
                    "camera_resize_calls_per_image": (
                        round(camera_resizes / calls, 6) if applied else None
                    ),
                    "timed_calls": calls,
                }
            )
            print(
                f"[camera-perf] {geometry}/{fmt} ordinary={ordinary_ms:.3f}ms "
                f"(resizes={ordinary_resizes}) camera={camera_ms:.3f}ms "
                f"(resizes={camera_resizes})",
                flush=True,
            )

    document = {
        "audit": "camera_viewport_v2_performance",
        "stage_edge": STAGE_EDGE,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "no_double_resize": not double_resize_seen,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[camera-perf] wrote {args.output}", flush=True)
    if double_resize_seen:
        raise SystemExit("double resize detected on the camera path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
