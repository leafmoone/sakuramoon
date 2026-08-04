"""EXIF-aware RGB normalization and deterministic cover resize/crop."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from sakuramoon.data.buckets import (
    BucketAssignment,
    BucketRejection,
    BucketScanReport,
    BucketShape,
    RejectionReason,
    assign_bucket,
)

DECODE_DIMENSION_SCAN_SAMPLES = 100_000
MAX_DIMENSION_MISMATCH_RATE = 0.001


class ImageRejected(ValueError):
    """Image cannot enter the selected stage without upscale or excessive crop."""

    def __init__(self, reason: RejectionReason) -> None:
        super().__init__(reason)
        self.reason = reason


class ImageScanError(RuntimeError):
    """Image scan input or artifact publication failed."""


class DimensionMismatchError(ImageScanError):
    """The fixed decode scan exceeded the allowed mismatch rate."""

    def __init__(self, report: DimensionScanReport) -> None:
        super().__init__("decoded dimension mismatch rate exceeds 0.1 percent")
        self.report = report


@dataclass(frozen=True)
class ProcessedImage:
    image: Image.Image
    assignment: BucketAssignment
    crop_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DecodedDimensionObservation:
    declared_width: int
    declared_height: int
    decoded_width: int
    decoded_height: int

    def __post_init__(self) -> None:
        values = (
            self.declared_width,
            self.declared_height,
            self.decoded_width,
            self.decoded_height,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ImageScanError("dimension observations must be positive integers")

    @property
    def matches(self) -> bool:
        return (self.declared_width, self.declared_height) == (
            self.decoded_width,
            self.decoded_height,
        )


@dataclass(frozen=True)
class DimensionScanReport:
    expected_samples: int
    total_samples: int
    matching_samples: int
    mismatch_samples: int
    mismatch_rate: float
    maximum_mismatch_rate: float
    accepted: bool

    def __post_init__(self) -> None:
        if (
            type(self.expected_samples) is not int
            or type(self.total_samples) is not int
            or type(self.matching_samples) is not int
            or type(self.mismatch_samples) is not int
            or type(self.mismatch_rate) is not float
            or type(self.maximum_mismatch_rate) is not float
            or type(self.accepted) is not bool
            or self.total_samples <= 0
        ):
            raise ImageScanError("dimension scan report is inconsistent")
        expected_rate = self.mismatch_samples / self.total_samples
        if (
            self.expected_samples != DECODE_DIMENSION_SCAN_SAMPLES
            or self.total_samples != self.expected_samples
            or self.matching_samples < 0
            or self.mismatch_samples < 0
            or self.matching_samples + self.mismatch_samples != self.total_samples
            or self.maximum_mismatch_rate != MAX_DIMENSION_MISMATCH_RATE
            or self.mismatch_rate != expected_rate
            or self.accepted != (expected_rate <= MAX_DIMENSION_MISMATCH_RATE)
        ):
            raise ImageScanError("dimension scan report is inconsistent")


@dataclass(frozen=True)
class ImageScanReport:
    bucket_scan: BucketScanReport
    dimension_scan: DimensionScanReport


def normalize_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation once, then convert to RGB."""

    return ImageOps.exif_transpose(image).convert("RGB")


def observe_decoded_dimensions(
    image: Image.Image,
    *,
    declared_width: int,
    declared_height: int,
) -> DecodedDimensionObservation:
    """Record dimensions after the same EXIF normalization used by preparation."""

    normalized = normalize_image(image)
    return DecodedDimensionObservation(
        declared_width=declared_width,
        declared_height=declared_height,
        decoded_width=normalized.width,
        decoded_height=normalized.height,
    )


def scan_decoded_dimensions(
    observations: Iterable[DecodedDimensionObservation],
) -> DimensionScanReport:
    """Require exactly 100k decoded observations and hard-fail above 0.1 percent."""

    total = 0
    mismatches = 0
    for observation in observations:
        total += 1
        if total > DECODE_DIMENSION_SCAN_SAMPLES:
            raise ImageScanError("dimension scan exceeds 100,000 observations")
        if not observation.matches:
            mismatches += 1
    if total != DECODE_DIMENSION_SCAN_SAMPLES:
        raise ImageScanError("dimension scan must contain exactly 100,000 observations")
    mismatch_rate = mismatches / total
    report = DimensionScanReport(
        expected_samples=DECODE_DIMENSION_SCAN_SAMPLES,
        total_samples=total,
        matching_samples=total - mismatches,
        mismatch_samples=mismatches,
        mismatch_rate=mismatch_rate,
        maximum_mismatch_rate=MAX_DIMENSION_MISMATCH_RATE,
        accepted=mismatch_rate <= MAX_DIMENSION_MISMATCH_RATE,
    )
    if not report.accepted:
        raise DimensionMismatchError(report)
    return report


def canonical_image_scan_report_bytes(report: ImageScanReport) -> bytes:
    retention_quantiles = report.bucket_scan.crop_retention_quantiles
    payload = {
        "bucket_scan": {
            "assigned_samples": report.bucket_scan.assigned_samples,
            "bucket_counts": [
                {
                    "height": item.height,
                    "samples": item.samples,
                    "width": item.width,
                }
                for item in report.bucket_scan.bucket_counts
            ],
            "expected_samples": report.bucket_scan.expected_samples,
            "no_upscale_rejections": report.bucket_scan.no_upscale_rejections,
            "crop_retention_quantiles": (
                None
                if retention_quantiles is None
                else {
                    "histogram_resolution": retention_quantiles.histogram_resolution,
                    "method": "nearest_rank_fixed_histogram",
                    "p01": retention_quantiles.p01,
                    "p50": retention_quantiles.p50,
                    "p99": retention_quantiles.p99,
                }
            ),
            "retention_rejections": report.bucket_scan.retention_rejections,
            "total_samples": report.bucket_scan.total_samples,
        },
        "dimension_scan": {
            "accepted": report.dimension_scan.accepted,
            "expected_samples": report.dimension_scan.expected_samples,
            "matching_samples": report.dimension_scan.matching_samples,
            "maximum_mismatch_rate": report.dimension_scan.maximum_mismatch_rate,
            "mismatch_rate": report.dimension_scan.mismatch_rate,
            "mismatch_samples": report.dimension_scan.mismatch_samples,
            "total_samples": report.dimension_scan.total_samples,
        },
        "schema_version": 1,
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def write_image_scan_report(report: ImageScanReport, destination: Path) -> None:
    """Atomically replace one canonical, regenerable scan report."""

    payload = canonical_image_scan_report_bytes(report)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except ImageScanError:
        raise
    except OSError:
        raise ImageScanError("image scan report could not be written") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def resize_and_crop(
    image: Image.Image,
    assignment: BucketAssignment,
    *,
    crop_seed: int,
) -> ProcessedImage:
    """Resize by cover and choose one deterministic uniform crop offset."""

    if type(crop_seed) is not int:
        raise ValueError("crop seed must be an integer")
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
