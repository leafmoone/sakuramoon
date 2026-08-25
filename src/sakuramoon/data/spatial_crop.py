"""Shifted-bucket spatial crop policy: full-canvas zoom/shift training data.

Pure geometry plus isolated deterministic RNG domains; no tensors and no I/O,
so the policy and every plan are spawn-picklable and DataLoader-worker safe.
The plan never rejects: every geometric miss falls back to the ordinary
aspect-bucket crop of the same sample.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from sakuramoon.data.buckets import BucketAssignment

if TYPE_CHECKING:
    from sakuramoon.config.schema import DataSpatialCropConfig

SPATIAL_POLICY_DOMAIN = "spatial-policy"
SPATIAL_ZOOM_DOMAIN = "spatial-zoom"
SPATIAL_OFFSET_X_DOMAIN = "spatial-offset-x"
SPATIAL_OFFSET_Y_DOMAIN = "spatial-offset-y"

SPATIAL_FALLBACK_REASONS: tuple[str, ...] = (
    "none",
    "not_selected",
    "insufficient_source_resolution",
    "base_zoom_at_or_above_max",
    "quantized_no_effect",
    "retention_guard",
)

ZOOM_HISTOGRAM_LABELS: tuple[str, ...] = (
    "[1.00,1.02)",
    "[1.02,1.04)",
    "[1.04,1.06)",
    "[1.06,1.08)",
    "[1.08,1.101]",
)


def zoom_histogram_bin(zoom: float) -> int:
    """Fixed bin index (0-4) of one actual equivalent zoom value."""

    if type(zoom) is not float or not math.isfinite(zoom) or zoom < 1.0:
        raise ValueError("zoom histogram bin requires a finite zoom at or above 1.0")
    if zoom < 1.02:
        return 0
    if zoom < 1.04:
        return 1
    if zoom < 1.06:
        return 2
    if zoom < 1.08:
        return 3
    return 4


@dataclass(frozen=True)
class SpatialCropPolicy:
    """Validated shifted-bucket crop policy (exact floats, spawn-picklable)."""

    enabled: bool
    probability: float
    min_equivalent_zoom: float
    max_equivalent_zoom: float
    min_crop_retention: float

    @classmethod
    def from_config(
        cls,
        config: DataSpatialCropConfig,
        *,
        min_crop_retention: float,
    ) -> SpatialCropPolicy:
        """Re-validate the exact constraints at the data boundary."""

        if type(min_crop_retention) is not float or not 0.0 <= min_crop_retention <= 1.0:
            raise ValueError("spatial crop min_crop_retention must be a float in [0, 1]")
        if not (0.0 <= config.probability <= 1.0):
            raise ValueError("spatial crop probability must be a float in [0, 1]")
        if config.min_equivalent_zoom >= config.max_equivalent_zoom:
            raise ValueError("spatial crop min_equivalent_zoom must be below max_equivalent_zoom")
        if config.enabled and config.probability <= 0.0:
            raise ValueError("spatial crop probability must be positive when enabled")
        if not config.enabled and config.probability != 0.0:
            raise ValueError("spatial crop probability must be zero when disabled")
        if 1.0 / config.max_equivalent_zoom**2 < min_crop_retention:
            raise ValueError(
                "spatial crop max_equivalent_zoom violates the min_crop_retention guard"
            )
        return cls(
            enabled=config.enabled,
            probability=config.probability,
            min_equivalent_zoom=config.min_equivalent_zoom,
            max_equivalent_zoom=config.max_equivalent_zoom,
            min_crop_retention=min_crop_retention,
        )


@dataclass(frozen=True)
class ShiftedBucketPlan:
    """One deterministic shifted-bucket decision for one sample.

    For a non-applied plan the canvas/crop fields carry ordinary-path
    placeholders and only ``fallback_reason`` and (when selected) the
    requested zoom are meaningful.
    """

    applied: bool
    fallback_reason: str
    canvas_width: int
    canvas_height: int
    crop_box: tuple[int, int, int, int]
    requested_equivalent_zoom: float
    actual_equivalent_zoom: float
    base_crop_retention: float
    final_crop_retention: float
    normalized_offset_x: float
    normalized_offset_y: float


def _quantize_full_canvas(
    canvas_width_float: float,
    canvas_height_float: float,
    bucket_width: int,
    bucket_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int] | None:
    """The single integer-rounding rule for spatial canvases.

    Half-up rounding, clamped into the source rectangle, and required to
    dominate the bucket in both dimensions. Returns ``None`` when no
    integer canvas can satisfy the invariants; callers fall back, never
    reject.
    """

    width = min(
        max(math.floor(canvas_width_float + 0.5), bucket_width), source_width
    )
    height = min(
        max(math.floor(canvas_height_float + 0.5), bucket_height), source_height
    )
    if width < bucket_width or height < bucket_height:
        return None
    return width, height


def plan_shifted_bucket(
    assignment: BucketAssignment,
    policy: SpatialCropPolicy,
    *,
    source_size: tuple[int, int] | None = None,
    policy_seed: int,
    zoom_seed: int,
    offset_x_seed: int,
    offset_y_seed: int,
) -> ShiftedBucketPlan:
    """Deterministic shifted-bucket plan; never rejects (fallback only).

    Selection, zoom, and the two independent offsets each draw from one
    isolated seeded RNG domain, so the ordinary crop RNG is never consumed.
    """

    if any(type(seed) is not int or seed < 0 for seed in (
        policy_seed,
        zoom_seed,
        offset_x_seed,
        offset_y_seed,
    )):
        raise ValueError("spatial crop seeds must be nonnegative integers")
    if source_size is None:
        source_width, source_height = assignment.source_width, assignment.source_height
    else:
        source_width, source_height = source_size
        if (
            type(source_width) is not int
            or type(source_height) is not int
            or source_width <= 0
            or source_height <= 0
        ):
            raise ValueError("spatial crop source size must be positive integers")

    bucket_width = assignment.bucket.width
    bucket_height = assignment.bucket.height
    bucket_area = bucket_width * bucket_height
    resized_area = assignment.resized_width * assignment.resized_height
    base_zoom = math.sqrt(resized_area / bucket_area)
    base_retention = assignment.crop_retention

    def fallback(
        reason: str, requested: float = 0.0
    ) -> ShiftedBucketPlan:
        return ShiftedBucketPlan(
            applied=False,
            fallback_reason=reason,
            canvas_width=assignment.resized_width,
            canvas_height=assignment.resized_height,
            crop_box=(0, 0, 0, 0),
            requested_equivalent_zoom=requested,
            actual_equivalent_zoom=1.0 / math.sqrt(base_retention),
            base_crop_retention=base_retention,
            final_crop_retention=base_retention,
            normalized_offset_x=0.0,
            normalized_offset_y=0.0,
        )

    selected = random.Random(policy_seed).random() < policy.probability
    if not selected:
        return fallback("not_selected")

    maximum_feasible_zoom = math.sqrt(
        source_width * source_height / bucket_area
    )
    if base_zoom >= policy.max_equivalent_zoom:
        return fallback("base_zoom_at_or_above_max")
    low = max(policy.min_equivalent_zoom, base_zoom)
    high = min(policy.max_equivalent_zoom, maximum_feasible_zoom)
    if high <= low:
        reason = (
            "insufficient_source_resolution"
            if maximum_feasible_zoom < policy.min_equivalent_zoom
            else "quantized_no_effect"
        )
        return fallback(reason)

    u = random.Random(zoom_seed).random()
    requested = low + (high - low) * math.sqrt(u)
    target_area = requested * requested * bucket_area
    aspect = source_width / source_height
    quantized = _quantize_full_canvas(
        math.sqrt(target_area * aspect),
        math.sqrt(target_area / aspect),
        bucket_width,
        bucket_height,
        source_width,
        source_height,
    )
    if quantized is None:
        return fallback("quantized_no_effect", requested=requested)
    canvas_width, canvas_height = quantized
    canvas_area = canvas_width * canvas_height
    actual_zoom = math.sqrt(canvas_area / bucket_area)
    final_retention = bucket_area / canvas_area
    if actual_zoom <= base_zoom + 1e-9:
        return fallback("quantized_no_effect", requested=requested)
    if final_retention < policy.min_crop_retention:
        return fallback("retention_guard", requested=requested)
    if actual_zoom > policy.max_equivalent_zoom:
        return fallback("quantized_no_effect", requested=requested)

    available_width = canvas_width - bucket_width
    available_height = canvas_height - bucket_height
    left = random.Random(offset_x_seed).randrange(available_width + 1)
    top = random.Random(offset_y_seed).randrange(available_height + 1)
    normalized_x = (
        2.0 * left / available_width - 1.0 if available_width > 0 else 0.0
    )
    normalized_y = (
        2.0 * top / available_height - 1.0 if available_height > 0 else 0.0
    )
    return ShiftedBucketPlan(
        applied=True,
        fallback_reason="none",
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        crop_box=(left, top, left + bucket_width, top + bucket_height),
        requested_equivalent_zoom=requested,
        actual_equivalent_zoom=actual_zoom,
        base_crop_retention=base_retention,
        final_crop_retention=final_retention,
        normalized_offset_x=normalized_x,
        normalized_offset_y=normalized_y,
    )


@dataclass(frozen=True, slots=True)
class SpatialCropCounts:
    """Fixed spatial-crop counters aggregated over one batch of audits.

    Strict zero semantics: when no spatial crop was applied every scalar is
    zero and the fixed reason/histogram keys carry the zero counts. The
    fallback-reason counts always sum to the batch size.
    """

    selected: int
    applied: int
    fallback_reasons: MappingProxyType[str, int]
    zoom_histogram: MappingProxyType[str, int]
    actual_zoom_sum: float
    actual_zoom_max: float
    abs_offset_x_sum: float
    abs_offset_y_sum: float
    both_axes_count: int

    def __post_init__(self) -> None:
        for name in (
            "selected",
            "applied",
            "both_axes_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"spatial crop {name} must be a nonnegative integer")
        if self.applied > self.selected:
            raise ValueError("spatial crop applied count exceeds selected count")
        if self.both_axes_count > self.applied:
            raise ValueError("spatial crop both-axes count exceeds applied count")
        if sorted(self.fallback_reasons) != sorted(SPATIAL_FALLBACK_REASONS):
            raise ValueError("spatial crop fallback reasons must keep the fixed keys")
        if sorted(self.zoom_histogram) != sorted(ZOOM_HISTOGRAM_LABELS):
            raise ValueError("spatial crop zoom histogram must keep the fixed labels")
        for count in self.fallback_reasons.values():
            if type(count) is not int or count < 0:
                raise TypeError("spatial crop fallback counts must be nonnegative")
        for count in self.zoom_histogram.values():
            if type(count) is not int or count < 0:
                raise TypeError("spatial crop zoom bin counts must be nonnegative")
        if sum(self.zoom_histogram.values()) != self.applied:
            raise ValueError("spatial crop zoom histogram must cover applied samples")
        for name in (
            "actual_zoom_sum",
            "actual_zoom_max",
            "abs_offset_x_sum",
            "abs_offset_y_sum",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"spatial crop {name} must be a nonnegative finite float")
        if self.applied == 0:
            if (
                self.actual_zoom_sum != 0.0
                or self.actual_zoom_max != 0.0
                or self.abs_offset_x_sum != 0.0
                or self.abs_offset_y_sum != 0.0
            ):
                raise ValueError(
                    "spatial crop sums must be zero when nothing was applied"
                )
        else:
            if self.actual_zoom_max < 1.0:
                raise ValueError("applied spatial crops must zoom at or above 1.0")


def _audit_box_size(audit: object) -> tuple[int, int]:
    box = audit.crop_box  # type: ignore[attr-defined]
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width <= 0 or height <= 0:
        raise ValueError("spatial crop audit crop box is invalid")
    return width, height


def aggregate_spatial_crop(audits: Iterable[object]) -> SpatialCropCounts:
    """Aggregate the fixed spatial-crop counters from ImageAudit records."""

    fallback_reasons = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
    zoom_histogram = {label: 0 for label in ZOOM_HISTOGRAM_LABELS}
    selected = 0
    applied = 0
    both_axes = 0
    zoom_sum = 0.0
    zoom_max = 0.0
    offset_x_sum = 0.0
    offset_y_sum = 0.0
    for audit in audits:
        reason = audit.spatial_fallback_reason  # type: ignore[attr-defined]
        if reason not in fallback_reasons:
            raise ValueError(f"unknown spatial crop fallback reason: {reason}")
        fallback_reasons[reason] += 1
        if audit.spatial_selected:  # type: ignore[attr-defined]
            selected += 1
        if not audit.spatial_applied:  # type: ignore[attr-defined]
            continue
        applied += 1
        zoom = audit.actual_equivalent_zoom  # type: ignore[attr-defined]
        if type(zoom) is not float or not math.isfinite(zoom) or zoom < 1.0:
            raise ValueError("applied spatial crop zoom must be a finite float at 1.0")
        zoom_sum += zoom
        zoom_max = max(zoom_max, zoom)
        zoom_histogram[ZOOM_HISTOGRAM_LABELS[zoom_histogram_bin(zoom)]] += 1
        offset_x = audit.normalized_offset_x  # type: ignore[attr-defined]
        offset_y = audit.normalized_offset_y  # type: ignore[attr-defined]
        for value in (offset_x, offset_y):
            if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("normalized spatial offsets must be finite floats in [-1, 1]")
        offset_x_sum += abs(offset_x)
        offset_y_sum += abs(offset_y)
        box_width, box_height = _audit_box_size(audit)
        if (
            audit.resized_width - box_width > 0  # type: ignore[attr-defined]
            and audit.resized_height - box_height > 0  # type: ignore[attr-defined]
        ):
            both_axes += 1
    return SpatialCropCounts(
        selected=selected,
        applied=applied,
        fallback_reasons=MappingProxyType(fallback_reasons),
        zoom_histogram=MappingProxyType(zoom_histogram),
        actual_zoom_sum=zoom_sum,
        actual_zoom_max=zoom_max,
        abs_offset_x_sum=offset_x_sum,
        abs_offset_y_sum=offset_y_sum,
        both_axes_count=both_axes,
    )


__all__ = [
    "SPATIAL_FALLBACK_REASONS",
    "SPATIAL_OFFSET_X_DOMAIN",
    "SPATIAL_OFFSET_Y_DOMAIN",
    "SPATIAL_POLICY_DOMAIN",
    "SPATIAL_ZOOM_DOMAIN",
    "ShiftedBucketPlan",
    "SpatialCropCounts",
    "SpatialCropPolicy",
    "ZOOM_HISTOGRAM_LABELS",
    "aggregate_spatial_crop",
    "plan_shifted_bucket",
    "zoom_histogram_bin",
]
