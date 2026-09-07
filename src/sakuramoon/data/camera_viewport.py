"""HDM-style shifted-square camera viewport policy.

Pure geometry plus isolated deterministic RNG domains; no tensors and no I/O,
so the policy and every plan are spawn-picklable and DataLoader-worker safe.
The plan never rejects: every geometric miss falls back to the ordinary
aspect-bucket path of the same sample (no sample admission change).

Camera v2 is a pure data-geometry strategy, not a conditioning branch:
discover the unique square stage bucket R from the bucket vocabulary (no
hardcoded R), resize the short edge to R (LANCZOS, never enlarge), keep the
full long axis, then draw one uniform inclusive R x R crop. The exact
full-canvas crop box yields exact full-canvas image coordinates through
``sakuramoon.conditioning.camera.transform_camera_coordinates``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sakuramoon.data.buckets import BucketAssignment, BucketShape

if TYPE_CHECKING:
    from sakuramoon.config.schema import (
        DataCameraMirrorBalanceConfig,
        DataCameraViewportConfig,
    )

CAMERA_POLICY_DOMAIN = "camera-policy"
CAMERA_OFFSET_DOMAIN = "camera-offset"
# Isolated deterministic RNG domain for the mirror-balance selection draw.
# It is a separate domain so an enabled/disabled mirror policy can never
# perturb the caption, crop, camera-selection, or camera-offset streams.
MIRROR_POLICY_DOMAIN = "camera-mirror-policy"

MIRROR_MODE_V1 = "vertical_mirror_pair_v1"

MIRROR_SEVERITY_BAND_LABELS: tuple[str, ...] = ("lt2", "2to4", "ge4")

MIRROR_SIDE_LABELS: tuple[str, ...] = ("start", "center", "end")

CAMERA_VAE_SCALE = 16

CAMERA_FALLBACK_REASONS: tuple[str, ...] = (
    "none",
    "not_selected",
    "short_edge_too_small",
    "near_square_below_min",
    "aspect_above_max",
    "quantized_no_effect",
    "no_square_bucket",
)

CAMERA_ZOOM_BAND_LABELS: tuple[str, ...] = (
    "[1.10,1.20)",
    "[1.20,1.35)",
    "[1.35,1.501]",
)

CAMERA_SHIFT_TOKEN_BIN_LABELS: tuple[str, ...] = (
    "[0,1)",
    "[1,2)",
    "[2,4)",
    "[4,8)",
    "[8,16)",
    "[16,inf)",
)

CAMERA_ORIENTATION_KEYS: tuple[str, ...] = ("horizontal", "vertical")

CAMERA_MAX_EQUIVALENT_ZOOM_CEILING = 1.5


class CameraViewportError(ValueError):
    """Camera viewport policy, plan, or aggregation state is invalid."""


def camera_zoom_band(zoom: float) -> int:
    """Fixed bin index (0-2) of one actual equivalent zoom value."""

    if type(zoom) is not float or not math.isfinite(zoom):
        raise CameraViewportError(
            "camera zoom band requires a finite zoom value"
        )
    if not 1.10 <= zoom <= 1.501:
        raise CameraViewportError(
            "camera zoom band value is outside [1.10, 1.501]"
        )
    if zoom < 1.20:
        return 0
    if zoom < 1.35:
        return 1
    return 2


def camera_shift_token_bin(latent_shift: float) -> int:
    """Fixed bin index (0-5) of one absolute latent-center shift magnitude."""

    if (
        type(latent_shift) is not float
        or not math.isfinite(latent_shift)
        or latent_shift < 0.0
    ):
        raise CameraViewportError(
            "camera shift token bin requires a nonnegative finite latent shift"
        )
    if latent_shift < 1.0:
        return 0
    if latent_shift < 2.0:
        return 1
    if latent_shift < 4.0:
        return 2
    if latent_shift < 8.0:
        return 3
    if latent_shift < 16.0:
        return 4
    return 5


def _round_half_up(value: float) -> int:
    """The single integer-rounding rule for camera canvases (half-up)."""

    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise CameraViewportError("camera canvas quantization requires a positive finite float")
    return math.floor(value + 0.5)


def camera_stage_edge(buckets: tuple[BucketShape, ...]) -> int:
    """Width of the unique square bucket in the stage-scaled vocabulary.

    Returns 0 when discovery fails; the planner maps that to the
    ``no_square_bucket`` fallback instead of raising.
    """

    squares = tuple(shape for shape in buckets if shape.width == shape.height)
    if len(squares) != 1:
        return 0
    return squares[0].width


def discover_square_bucket(
    buckets: tuple[BucketShape, ...], *, stage_edge: int
) -> int | None:
    """The unique square stage bucket R, or ``None`` when discovery fails.

    Discovery is vocabulary-driven (no hardcoded R): exactly one bucket must
    be square, and it must equal the stage resolution on both edges.
    """

    if type(stage_edge) is not int or stage_edge <= 0:
        raise CameraViewportError("stage edge must be a positive integer")
    squares = tuple(shape for shape in buckets if shape.width == shape.height)
    if len(squares) != 1:
        return None
    square = squares[0]
    if square.width != stage_edge or square.height != stage_edge:
        return None
    return square.width


@dataclass(frozen=True)
class CameraViewportPolicy:
    """Validated shifted-square camera viewport policy (exact floats)."""

    enabled: bool
    probability: float
    min_equivalent_zoom: float
    max_equivalent_zoom: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise CameraViewportError("camera viewport enabled must be a bool")
        if (
            type(self.probability) is not float
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise CameraViewportError("camera viewport probability must be a finite float in [0, 1]")
        for name in ("min_equivalent_zoom", "max_equivalent_zoom"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 1.0:
                raise CameraViewportError(
                    f"camera viewport {name} must be a finite float above 1.0"
                )
        if self.min_equivalent_zoom >= self.max_equivalent_zoom:
            raise CameraViewportError(
                "camera viewport min_equivalent_zoom must be below max_equivalent_zoom"
            )
        if self.max_equivalent_zoom > CAMERA_MAX_EQUIVALENT_ZOOM_CEILING:
            raise CameraViewportError(
                "camera viewport max_equivalent_zoom must not exceed 1.5"
            )
        if self.enabled and self.probability <= 0.0:
            raise CameraViewportError(
                "camera viewport probability must be positive when enabled"
            )
        if not self.enabled and self.probability != 0.0:
            raise CameraViewportError(
                "camera viewport probability must be zero when disabled"
            )

    @classmethod
    def from_config(cls, config: DataCameraViewportConfig) -> CameraViewportPolicy:
        """Re-validate the exact constraints at the data boundary."""

        return cls(
            enabled=config.enabled,
            probability=config.probability,
            min_equivalent_zoom=config.min_equivalent_zoom,
            max_equivalent_zoom=config.max_equivalent_zoom,
        )


@dataclass(frozen=True)
class CameraViewportPlan:
    """One deterministic shifted-square camera decision for one sample.

    For a non-applied plan the canvas/crop/shift fields are strict zeros or
    ordinary-path placeholders and only ``fallback_reason`` (and
    ``orientation == "none"``) are meaningful.
    """

    applied: bool
    fallback_reason: str
    orientation: str
    viewport: int
    full_width: int
    full_height: int
    crop_box: tuple[int, int, int, int]
    left: int
    top: int
    equivalent_zoom: float
    retention: float
    normalized_offset: float
    signed_pixel_center_shift: float
    absolute_pixel_center_shift: float
    latent_center_shift: float
    camera_shift_x: float
    camera_shift_y: float

    def __post_init__(self) -> None:
        if type(self.applied) is not bool:
            raise CameraViewportError("camera viewport plan applied must be a bool")
        if self.fallback_reason not in CAMERA_FALLBACK_REASONS:
            raise CameraViewportError(
                f"unknown camera viewport fallback reason: {self.fallback_reason}"
            )
        if self.orientation not in (*CAMERA_ORIENTATION_KEYS, "none"):
            raise CameraViewportError(
                f"unknown camera viewport orientation: {self.orientation}"
            )
        if self.applied:
            if self.fallback_reason != "none" or self.orientation == "none":
                raise CameraViewportError(
                    "applied camera viewport plan must carry reason none and a real orientation"
                )
            if (
                type(self.viewport) is not int
                or self.viewport <= 0
                or self.full_width <= 0
                or self.full_height <= 0
            ):
                raise CameraViewportError(
                    "applied camera viewport plan needs positive canvas dimensions"
                )
            left, top, right, bottom = self.crop_box
            if (
                left != self.left
                or top != self.top
                or right - left != self.viewport
                or bottom - top != self.viewport
                or left + self.viewport > self.full_width
                or top + self.viewport > self.full_height
            ):
                raise CameraViewportError(
                    "applied camera viewport crop box is inconsistent"
                )
            for name in (
                "equivalent_zoom",
                "retention",
                "normalized_offset",
                "signed_pixel_center_shift",
                "absolute_pixel_center_shift",
                "latent_center_shift",
                "camera_shift_x",
                "camera_shift_y",
            ):
                value = getattr(self, name)
                if type(value) is not float or not math.isfinite(value):
                    raise CameraViewportError(
                        f"camera viewport {name} must be a finite float"
                    )
            if not 1.10 <= self.equivalent_zoom <= 1.501:
                raise CameraViewportError(
                    "applied camera viewport zoom is outside [1.10, 1.501]"
                )
            if not 0.0 < self.retention <= 1.0:
                raise CameraViewportError(
                    "applied camera viewport retention must be in (0, 1]"
                )
            if self.absolute_pixel_center_shift < 0.0:
                raise CameraViewportError(
                    "absolute pixel center shift must be nonnegative"
                )
            if self.latent_center_shift != self.absolute_pixel_center_shift / CAMERA_VAE_SCALE:
                raise CameraViewportError(
                    "latent center shift must be the pixel shift divided by 16.0"
                )
        else:
            if (
                self.orientation != "none"
                or self.viewport != 0
                or self.full_width != 0
                or self.full_height != 0
                or self.crop_box != (0, 0, 0, 0)
                or self.left != 0
                or self.top != 0
                or self.equivalent_zoom != 0.0
                or self.retention != 0.0
                or self.normalized_offset != 0.0
                or self.signed_pixel_center_shift != 0.0
                or self.absolute_pixel_center_shift != 0.0
                or self.latent_center_shift != 0.0
                or self.camera_shift_x != 0.0
                or self.camera_shift_y != 0.0
            ):
                raise CameraViewportError(
                    "non-applied camera viewport plan fields must be strict zero"
                )



def _fallback_plan(
    assignment: BucketAssignment, reason: str
) -> CameraViewportPlan:
    return CameraViewportPlan(
        applied=False,
        fallback_reason=reason,
        orientation="none",
        viewport=0,
        full_width=0,
        full_height=0,
        crop_box=(0, 0, 0, 0),
        left=0,
        top=0,
        equivalent_zoom=0.0,
        retention=0.0,
        normalized_offset=0.0,
        signed_pixel_center_shift=0.0,
        absolute_pixel_center_shift=0.0,
        latent_center_shift=0.0,
        camera_shift_x=0.0,
        camera_shift_y=0.0,
    )


def plan_camera_viewport(
    assignment: BucketAssignment,
    policy: CameraViewportPolicy,
    *,
    buckets: tuple[BucketShape, ...],
    stage_edge: int,
    source_size: tuple[int, int] | None,
    policy_seed: int,
    offset_seed: int,
) -> CameraViewportPlan:
    """Deterministic shifted-square camera plan; never rejects (fallback only).

    Selection draws from the isolated ``camera-policy`` RNG domain and the
    crop offset from ``camera-offset``; the ordinary crop RNG is never
    consumed. Camera eligibility requires the sample to be ordinarily
    admitted (the caller passes the ordinary ``BucketAssignment``).
    """

    for name, seed in (("policy_seed", policy_seed), ("offset_seed", offset_seed)):
        if type(seed) is not int or seed < 0:
            raise CameraViewportError(f"camera viewport {name} must be a nonnegative integer")
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
            raise CameraViewportError(
                "camera viewport source size must be positive integers"
            )

    if type(stage_edge) is int and stage_edge > 0:
        viewport = discover_square_bucket(buckets, stage_edge=stage_edge)
    else:
        viewport = None
    if viewport is None:
        return _fallback_plan(assignment, "no_square_bucket")

    if not random.Random(policy_seed).random() < policy.probability:
        return _fallback_plan(assignment, "not_selected")

    r = viewport
    if source_width >= source_height:
        orientation = "horizontal"
        short_edge = source_height
        long_edge = source_width
    else:
        orientation = "vertical"
        short_edge = source_width
        long_edge = source_height

    if short_edge < r:
        return _fallback_plan(assignment, "short_edge_too_small")

    # Ideal equivalent zoom from the natural source aspect (pre-quantization):
    # z_ideal = sqrt(full_ideal * r / r^2) = sqrt(long_edge / short_edge).
    zoom_ideal = math.sqrt(long_edge / short_edge)
    if zoom_ideal < policy.min_equivalent_zoom:
        return _fallback_plan(assignment, "near_square_below_min")
    if zoom_ideal > policy.max_equivalent_zoom:
        return _fallback_plan(assignment, "aspect_above_max")

    # Integer-quantized full canvas (half-up, never enlarges: r <= short_edge).
    long_quantized = _round_half_up(long_edge * r / short_edge)
    if orientation == "horizontal":
        full_width, full_height = long_quantized, r
    else:
        full_width, full_height = r, long_quantized

    equivalent_zoom = math.sqrt(full_width * full_height / float(r * r))
    if not (policy.min_equivalent_zoom <= equivalent_zoom <= policy.max_equivalent_zoom):
        return _fallback_plan(assignment, "quantized_no_effect")

    # Uniform inclusive full-range offset on the long axis (both endpoints).
    if orientation == "horizontal":
        available = full_width - r
        left = random.Random(offset_seed).randrange(available + 1)
        top = 0
    else:
        available = full_height - r
        top = random.Random(offset_seed).randrange(available + 1)
        left = 0

    if orientation == "horizontal":
        camera_shift_x = 2.0 * left / r + 1.0 - full_width / r
        camera_shift_y = 0.0
        signed_shift = float(left + r // 2) - float(full_width) / 2.0
    else:
        camera_shift_x = 0.0
        camera_shift_y = 2.0 * top / r + 1.0 - full_height / r
        signed_shift = float(top + r // 2) - float(full_height) / 2.0

    absolute_shift = abs(signed_shift)
    retention = float(r * r) / float(full_width * full_height)
    normalized_offset = left / available if available > 0 else 0.0
    if orientation == "vertical":
        normalized_offset = top / available if available > 0 else 0.0

    return CameraViewportPlan(
        applied=True,
        fallback_reason="none",
        orientation=orientation,
        viewport=r,
        full_width=full_width,
        full_height=full_height,
        crop_box=(left, top, left + r, top + r),
        left=left,
        top=top,
        equivalent_zoom=equivalent_zoom,
        retention=retention,
        normalized_offset=normalized_offset,
        signed_pixel_center_shift=signed_shift,
        absolute_pixel_center_shift=absolute_shift,
        latent_center_shift=absolute_shift / CAMERA_VAE_SCALE,
        camera_shift_x=camera_shift_x,
        camera_shift_y=camera_shift_y,
    )


@dataclass(frozen=True, slots=True)
class CameraViewportCounts:
    """Fixed camera-viewport counters aggregated over one batch of audits.

    Strict zero semantics: when nothing was applied every scalar aggregate
    is zero and the fixed reason/band/bin keys carry the zero counts. The
    fallback-reason counts always sum to the batch size.
    """

    selected: int
    applied: int
    fallback_reasons: Mapping[str, int]
    orientation_counts: Mapping[str, int]
    zoom_bands: Mapping[str, int]
    shift_token_bins: Mapping[str, int]
    camera_zoom_sum: float
    camera_zoom_mean: float
    camera_zoom_max: float
    camera_retention_sum: float
    camera_retention_mean: float
    camera_retention_min: float
    camera_abs_pixel_shift_sum: float
    camera_abs_pixel_shift_mean: float
    camera_abs_pixel_shift_max: float
    camera_abs_latent_shift_sum: float
    camera_abs_latent_shift_mean: float
    camera_abs_latent_shift_max: float

    def __post_init__(self) -> None:
        for name in ("selected", "applied"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CameraViewportError(
                    f"camera viewport {name} must be a nonnegative integer"
                )
        if self.applied > self.selected:
            raise CameraViewportError(
                "camera viewport applied count exceeds selected count"
            )
        if sorted(self.fallback_reasons) != sorted(CAMERA_FALLBACK_REASONS):
            raise CameraViewportError(
                "camera viewport fallback reasons must keep the fixed keys"
            )
        if sorted(self.orientation_counts) != sorted(CAMERA_ORIENTATION_KEYS):
            raise CameraViewportError(
                "camera viewport orientation counts must keep the fixed keys"
            )
        if sorted(self.zoom_bands) != sorted(CAMERA_ZOOM_BAND_LABELS):
            raise CameraViewportError(
                "camera viewport zoom bands must keep the fixed labels"
            )
        if sorted(self.shift_token_bins) != sorted(CAMERA_SHIFT_TOKEN_BIN_LABELS):
            raise CameraViewportError(
                "camera viewport shift token bins must keep the fixed labels"
            )
        for mapping in (
            self.fallback_reasons,
            self.orientation_counts,
            self.zoom_bands,
            self.shift_token_bins,
        ):
            for count in mapping.values():
                if type(count) is not int or count < 0:
                    raise CameraViewportError(
                        "camera viewport histogram counts must be nonnegative integers"
                    )
        if sum(self.zoom_bands.values()) != self.applied:
            raise CameraViewportError(
                "camera viewport zoom bands must cover applied samples"
            )
        if sum(self.shift_token_bins.values()) != self.applied:
            raise CameraViewportError(
                "camera viewport shift token bins must cover applied samples"
            )
        if sum(self.orientation_counts.values()) != self.applied:
            raise CameraViewportError(
                "camera viewport orientation counts must cover applied samples"
            )
        for name in (
            "camera_zoom_sum",
            "camera_zoom_mean",
            "camera_zoom_max",
            "camera_retention_sum",
            "camera_retention_mean",
            "camera_retention_min",
            "camera_abs_pixel_shift_sum",
            "camera_abs_pixel_shift_mean",
            "camera_abs_pixel_shift_max",
            "camera_abs_latent_shift_sum",
            "camera_abs_latent_shift_mean",
            "camera_abs_latent_shift_max",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise CameraViewportError(
                    f"camera viewport {name} must be a nonnegative finite float"
                )
        if self.applied == 0:
            if any(
                getattr(self, name) != 0.0
                for name in (
                    "camera_zoom_sum",
                    "camera_zoom_mean",
                    "camera_zoom_max",
                    "camera_retention_sum",
                    "camera_retention_mean",
                    "camera_retention_min",
                    "camera_abs_pixel_shift_sum",
                    "camera_abs_pixel_shift_mean",
                    "camera_abs_pixel_shift_max",
                    "camera_abs_latent_shift_sum",
                    "camera_abs_latent_shift_mean",
                    "camera_abs_latent_shift_max",
                )
            ):
                raise CameraViewportError(
                    "camera viewport aggregates must be zero when nothing was applied"
                )
        else:
            if self.camera_zoom_max < 1.10 or self.camera_retention_min <= 0.0:
                raise CameraViewportError(
                    "applied camera viewport aggregates violate the zoom/retention bounds"
                )


def aggregate_camera_viewport(audits: Iterable[object]) -> CameraViewportCounts:
    """Aggregate the fixed camera-viewport counters from ImageAudit records."""

    fallback_reasons = {reason: 0 for reason in CAMERA_FALLBACK_REASONS}
    orientation_counts = {key: 0 for key in CAMERA_ORIENTATION_KEYS}
    zoom_bands = {label: 0 for label in CAMERA_ZOOM_BAND_LABELS}
    shift_token_bins = {label: 0 for label in CAMERA_SHIFT_TOKEN_BIN_LABELS}
    selected = 0
    applied = 0
    zoom_sum = 0.0
    zoom_max = 0.0
    retention_sum = 0.0
    retention_min = math.inf
    pixel_shift_sum = 0.0
    pixel_shift_max = 0.0
    latent_shift_sum = 0.0
    latent_shift_max = 0.0
    for audit in audits:
        reason = audit.camera_fallback_reason  # type: ignore[attr-defined]
        if reason not in fallback_reasons:
            raise CameraViewportError(
                f"unknown camera viewport fallback reason: {reason}"
            )
        fallback_reasons[reason] += 1
        if audit.camera_selected:  # type: ignore[attr-defined]
            selected += 1
        if not audit.camera_applied:  # type: ignore[attr-defined]
            continue
        applied += 1
        orientation = audit.camera_orientation  # type: ignore[attr-defined]
        if orientation not in orientation_counts:
            raise CameraViewportError(
                f"applied camera viewport has invalid orientation: {orientation}"
            )
        orientation_counts[orientation] += 1
        zoom = float(audit.camera_equivalent_zoom)  # type: ignore[attr-defined]
        zoom_sum += zoom
        zoom_max = max(zoom_max, zoom)
        zoom_bands[CAMERA_ZOOM_BAND_LABELS[camera_zoom_band(zoom)]] += 1
        retention = float(audit.camera_final_retention)  # type: ignore[attr-defined]
        retention_sum += retention
        retention_min = min(retention_min, retention)
        pixel_shift = float(audit.camera_pixel_center_shift)  # type: ignore[attr-defined]
        latent_shift = float(audit.camera_latent_center_shift)  # type: ignore[attr-defined]
        abs_pixel = abs(pixel_shift)
        abs_latent = abs(latent_shift)
        pixel_shift_sum += abs_pixel
        pixel_shift_max = max(pixel_shift_max, abs_pixel)
        latent_shift_sum += abs_latent
        latent_shift_max = max(latent_shift_max, abs_latent)
        shift_token_bins[
            CAMERA_SHIFT_TOKEN_BIN_LABELS[camera_shift_token_bin(abs_latent)]
        ] += 1
    if applied:
        return CameraViewportCounts(
            selected=selected,
            applied=applied,
            fallback_reasons=dict(fallback_reasons),
            orientation_counts=dict(orientation_counts),
            zoom_bands=dict(zoom_bands),
            shift_token_bins=dict(shift_token_bins),
            camera_zoom_sum=zoom_sum,
            camera_zoom_mean=zoom_sum / applied,
            camera_zoom_max=zoom_max,
            camera_retention_sum=retention_sum,
            camera_retention_mean=retention_sum / applied,
            camera_retention_min=retention_min,
            camera_abs_pixel_shift_sum=pixel_shift_sum,
            camera_abs_pixel_shift_mean=pixel_shift_sum / applied,
            camera_abs_pixel_shift_max=pixel_shift_max,
            camera_abs_latent_shift_sum=latent_shift_sum,
            camera_abs_latent_shift_mean=latent_shift_sum / applied,
            camera_abs_latent_shift_max=latent_shift_max,
        )
    return CameraViewportCounts(
        selected=selected,
        applied=0,
        fallback_reasons=dict(fallback_reasons),
        orientation_counts=dict(orientation_counts),
        zoom_bands=dict(zoom_bands),
        shift_token_bins=dict(shift_token_bins),
        camera_zoom_sum=0.0,
        camera_zoom_mean=0.0,
        camera_zoom_max=0.0,
        camera_retention_sum=0.0,
        camera_retention_mean=0.0,
        camera_retention_min=0.0,
        camera_abs_pixel_shift_sum=0.0,
        camera_abs_pixel_shift_mean=0.0,
        camera_abs_pixel_shift_max=0.0,
        camera_abs_latent_shift_sum=0.0,
        camera_abs_latent_shift_mean=0.0,
        camera_abs_latent_shift_max=0.0,
    )


@dataclass(frozen=True)
class CameraMirrorPolicy:
    """Validated vertical mirror-balanced supervision policy (exact floats).

    Data-strategy only: the policy decides whether an eligible vertical camera
    sample also trains a mirrored crop of the SAME full canvas, with exactly
    the same source, zoom, retention, |shift|, caption and conditioning. The
    training objective weights the pair as one logical sample
    (0.5 * L_original + 0.5 * L_mirror). The payload it authorizes carries one
    mirror image tensor; the geometry and RNG domain stay here.
    """

    enabled: bool
    mode: str
    min_latent_shift: float
    pair_probability: float
    pair_weight: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise CameraViewportError("mirror policy enabled must be a bool")
        if self.mode != MIRROR_MODE_V1:
            raise CameraViewportError(
                f"mirror policy mode must be {MIRROR_MODE_V1!r}"
            )
        if (
            type(self.min_latent_shift) is not float
            or not math.isfinite(self.min_latent_shift)
            or not 0.0 < self.min_latent_shift <= 4.0
        ):
            raise CameraViewportError(
                "mirror policy min_latent_shift must be a finite float in (0, 4]"
            )
        if (
            type(self.pair_probability) is not float
            or not math.isfinite(self.pair_probability)
            or not 0.0 <= self.pair_probability <= 1.0
        ):
            raise CameraViewportError(
                "mirror policy pair_probability must be a finite float in [0, 1]"
            )
        if (
            type(self.pair_weight) is not float
            or not math.isfinite(self.pair_weight)
            or self.pair_weight != 1.0
        ):
            raise CameraViewportError(
                "mirror policy pair_weight must be exactly 1.0 "
                "(one logical sample, two 0.5-weighted physical views)"
            )

    @classmethod
    def from_config(
        cls, config: DataCameraMirrorBalanceConfig | None
    ) -> CameraMirrorPolicy | None:
        if config is None:
            return None
        return cls(
            enabled=config.enabled,
            mode=config.mode,
            min_latent_shift=config.min_latent_shift,
            pair_probability=config.pair_probability,
            pair_weight=config.pair_weight,
        )


@dataclass(frozen=True)
class CameraMirrorPayload:
    """The mirror branch of one logical mirror pair (worker-level data).

    Carries only what is needed to train the mirror view: the mirror crop
    image (same R x R target as the original view) plus the mirror geometry
    needed to rebuild its full-canvas coordinates. The source, full canvas,
    zoom, retention, |shift|, caption tokens, Qwen states and condition routing
    are shared with the original row by construction and are NOT duplicated
    here.
    """

    mirror_image: torch.Tensor
    crop_box: tuple[int, int, int, int]
    signed_pixel_center_shift: float
    camera_shift_y: float
    normalized_offset: float

    def __post_init__(self) -> None:
        if not isinstance(self.mirror_image, torch.Tensor) or self.mirror_image.ndim != 3:
            raise CameraViewportError("mirror payload image must be a [3, R, R] tensor")
        if self.mirror_image.shape[0] != 3:
            raise CameraViewportError("mirror payload image must be RGB")
        if self.mirror_image.shape[-2] != self.mirror_image.shape[-1]:
            raise CameraViewportError("mirror payload image must be square")
        if self.mirror_image.dtype != torch.uint8:
            raise CameraViewportError("mirror payload image must be uint8")
        if len(self.crop_box) != 4 or any(type(edge) is not int or edge < 0 for edge in self.crop_box):
            raise CameraViewportError("mirror payload crop box must be four nonnegative ints")
        left, top, right, bottom = self.crop_box
        if (
            right - left != self.mirror_image.shape[-1]
            or bottom - top != self.mirror_image.shape[-2]
        ):
            raise CameraViewportError(
                "mirror payload crop box must match the mirror image size"
            )
        for name in (
            "signed_pixel_center_shift",
            "camera_shift_y",
            "normalized_offset",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise CameraViewportError(
                    f"mirror payload {name} must be a finite float"
                )
        if not 0.0 <= self.normalized_offset <= 1.0:
            raise CameraViewportError(
                "mirror payload normalized_offset must be in [0, 1]"
            )


def mirror_severity_band(latent_shift: float) -> str:
    """Fixed severity band label of one absolute latent-center shift."""

    if (
        type(latent_shift) is not float
        or not math.isfinite(latent_shift)
        or latent_shift < 0.0
    ):
        raise CameraViewportError(
            "mirror severity band requires a nonnegative finite latent shift"
        )
    if latent_shift < 2.0:
        return MIRROR_SEVERITY_BAND_LABELS[0]
    if latent_shift < 4.0:
        return MIRROR_SEVERITY_BAND_LABELS[1]
    return MIRROR_SEVERITY_BAND_LABELS[2]


def mirror_side(normalized_offset: float) -> str:
    """Fixed side band of one normalized long-axis crop offset in [0, 1].

    ``start`` = offset < 1/3, ``center`` = 1/3 <= offset < 2/3,
    ``end`` = offset >= 2/3. The mirror offset is exactly 1 - offset, so
    start and end exchange under mirroring while center maps to center.
    """

    if (
        type(normalized_offset) is not float
        or not math.isfinite(normalized_offset)
        or not 0.0 <= normalized_offset <= 1.0
    ):
        raise CameraViewportError(
        "mirror side requires a finite normalized offset in [0, 1]"
    )
    if normalized_offset < 1.0 / 3.0:
        return MIRROR_SIDE_LABELS[0]
    if normalized_offset < 2.0 / 3.0:
        return MIRROR_SIDE_LABELS[1]
    return MIRROR_SIDE_LABELS[2]


def camera_mirror_eligible(
    plan: CameraViewportPlan, *, min_latent_shift: float
) -> bool:
    """V1 eligibility: applied vertical camera with a severe long-axis shift.

    Horizontal views, latent shift below the threshold, and odd viewports
    (which cannot give the exact signed-shift antisymmetry) are ineligible.
    Fallback / not-selected plans are ineligible by construction.
    """

    if type(min_latent_shift) is not float or not math.isfinite(min_latent_shift):
        raise CameraViewportError("mirror eligibility requires a finite threshold")
    if not plan.applied:
        return False
    if plan.orientation != "vertical":
        return False
    if plan.viewport % 2 != 0:
        return False
    return plan.latent_center_shift >= min_latent_shift


def plan_mirror_geometry(plan: CameraViewportPlan) -> tuple[int, tuple[int, int, int, int], float, float, float] | None:
    """Mirror crop geometry for one vertical camera plan, or None.

    Returns ``(mirror_top, mirror_crop_box, mirror_signed_shift,
    mirror_camera_shift_y, mirror_normalized_offset)``. The exact contract
    ``signed_shift(mirror_top) == -signed_shift(top)`` is re-verified on the
    computed geometry; a violation (defensive; even viewports satisfy it
    exactly in float64) yields None instead of an invalid geometry.
    """

    if plan.orientation != "vertical":
        raise CameraViewportError("mirror geometry requires a vertical plan")
    full_height = plan.full_height
    viewport = plan.viewport
    if viewport % 2 != 0:
        return None
    available = full_height - viewport
    original_top = plan.top
    if not 0 <= original_top <= available:
        raise CameraViewportError("mirror geometry requires a plan within the canvas")
    mirror_top = available - original_top
    mirror_box = (0, mirror_top, viewport, mirror_top + viewport)
    original_signed = float(original_top + viewport // 2) - float(full_height) / 2.0
    mirror_signed = float(mirror_top + viewport // 2) - float(full_height) / 2.0
    if mirror_signed != -original_signed:
        return None
    mirror_shift_y = 2.0 * mirror_top / viewport + 1.0 - full_height / viewport
    mirror_offset = mirror_top / available if available > 0 else 0.5
    return mirror_top, mirror_box, mirror_signed, mirror_shift_y, mirror_offset


@dataclass(frozen=True, slots=True)
class CameraMirrorCounts:
    """Fixed camera-mirror counters aggregated over one batch.

    Conservation equations (enforced in ``__post_init__``):
      mirror_applied <= mirror_selected <= mirror_eligible <= vertical_applied
      severity_lt2 + severity_2to4 + severity_ge4 == vertical_applied
      original_start + original_center + original_end == mirror_applied
      mirror_start + mirror_center + mirror_end == mirror_applied
      mirror_extra_views == mirror_applied
      physical_views == logical_samples + mirror_extra_views

    ``severity_*`` count the vertical-applied camera population of the batch
    (a population statistic, so the later >=4 review can see the full
    distribution); every other key is intervention activity and is zero when
    the feature produced nothing (which also covers the disabled case).
    """

    logical_samples: int
    vertical_applied: int
    mirror_eligible: int
    mirror_selected: int
    mirror_applied: int
    severity_lt2: int
    severity_2to4: int
    severity_ge4: int
    original_start: int
    original_center: int
    original_end: int
    mirror_start: int
    mirror_center: int
    mirror_end: int
    mirror_extra_views: int
    physical_views: int

    def __post_init__(self) -> None:
        for name in (
            "logical_samples",
            "vertical_applied",
            "mirror_eligible",
            "mirror_selected",
            "mirror_applied",
            "severity_lt2",
            "severity_2to4",
            "severity_ge4",
            "original_start",
            "original_center",
            "original_end",
            "mirror_start",
            "mirror_center",
            "mirror_end",
            "mirror_extra_views",
            "physical_views",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CameraViewportError(
                    f"camera mirror count {name} must be a nonnegative integer"
                )
        if not (
            self.mirror_applied
            <= self.mirror_selected
            <= self.mirror_eligible
            <= self.vertical_applied
            <= self.logical_samples
        ):
            raise CameraViewportError(
                "camera mirror counts violate applied<=selected<=eligible<=vertical<=logical"
            )
        if (
            self.severity_lt2 + self.severity_2to4 + self.severity_ge4
            != self.vertical_applied
        ):
            raise CameraViewportError(
                "camera mirror severity bands must cover the vertical-applied population"
            )
        if (
            self.original_start + self.original_center + self.original_end
            != self.mirror_applied
        ):
            raise CameraViewportError(
                "camera mirror original side counts must cover applied pairs"
            )
        if (
            self.mirror_start + self.mirror_center + self.mirror_end
            != self.mirror_applied
        ):
            raise CameraViewportError(
                "camera mirror mirror side counts must cover applied pairs"
            )
        if self.mirror_extra_views != self.mirror_applied:
            raise CameraViewportError(
                "camera mirror extra views must equal applied pairs"
            )
        if self.physical_views != self.logical_samples + self.mirror_extra_views:
            raise CameraViewportError(
                "camera mirror physical views must equal logical + extra views"
            )


def zero_camera_mirror_counts(logical_samples: int) -> CameraMirrorCounts:
    """The strict-zero table for a batch with no camera-mirror activity."""

    return CameraMirrorCounts(
        logical_samples=logical_samples,
        vertical_applied=0,
        mirror_eligible=0,
        mirror_selected=0,
        mirror_applied=0,
        severity_lt2=0,
        severity_2to4=0,
        severity_ge4=0,
        original_start=0,
        original_center=0,
        original_end=0,
        mirror_start=0,
        mirror_center=0,
        mirror_end=0,
        mirror_extra_views=0,
        physical_views=logical_samples,
    )


def aggregate_camera_mirror(samples: Iterable[object]) -> CameraMirrorCounts:
    """Aggregate the fixed camera-mirror counters from PipelineSamples.

    Reads duck-typed fields: ``audit.camera_applied``,
    ``audit.camera_orientation``, ``audit.camera_latent_center_shift``,
    ``audit.camera_normalized_offset`` and the per-sample mirror facts
    (``mirror`` payload, ``mirror_eligible``, ``mirror_selected``).
    """

    vertical_applied = 0
    eligible = 0
    selected = 0
    applied = 0
    severity = {label: 0 for label in MIRROR_SEVERITY_BAND_LABELS}
    original_sides = {label: 0 for label in MIRROR_SIDE_LABELS}
    mirror_sides = {label: 0 for label in MIRROR_SIDE_LABELS}
    logical_samples = 0
    for sample in samples:
        logical_samples += 1
        audit = sample.audit  # type: ignore[attr-defined]
        if audit.camera_applied and audit.camera_orientation == "vertical":
            vertical_applied += 1
            severity[
                mirror_severity_band(
                    abs(float(audit.camera_latent_center_shift))
                )
            ] += 1
        if sample.mirror_eligible:  # type: ignore[attr-defined]
            eligible += 1
        if sample.mirror_selected:  # type: ignore[attr-defined]
            selected += 1
        payload = sample.mirror  # type: ignore[attr-defined]
        if payload is None:
            continue
        applied += 1
        original_sides[mirror_side(float(audit.camera_normalized_offset))] += 1
        mirror_sides[mirror_side(payload.normalized_offset)] += 1
    if vertical_applied == 0 and eligible == 0 and selected == 0 and applied == 0:
        return zero_camera_mirror_counts(logical_samples)
    return CameraMirrorCounts(
        logical_samples=logical_samples,
        vertical_applied=vertical_applied,
        mirror_eligible=eligible,
        mirror_selected=selected,
        mirror_applied=applied,
        severity_lt2=severity[MIRROR_SEVERITY_BAND_LABELS[0]],
        severity_2to4=severity[MIRROR_SEVERITY_BAND_LABELS[1]],
        severity_ge4=severity[MIRROR_SEVERITY_BAND_LABELS[2]],
        original_start=original_sides[MIRROR_SIDE_LABELS[0]],
        original_center=original_sides[MIRROR_SIDE_LABELS[1]],
        original_end=original_sides[MIRROR_SIDE_LABELS[2]],
        mirror_start=mirror_sides[MIRROR_SIDE_LABELS[0]],
        mirror_center=mirror_sides[MIRROR_SIDE_LABELS[1]],
        mirror_end=mirror_sides[MIRROR_SIDE_LABELS[2]],
        mirror_extra_views=applied,
        physical_views=logical_samples + applied,
    )


__all__ = [
    "CAMERA_FALLBACK_REASONS",
    "CAMERA_MAX_EQUIVALENT_ZOOM_CEILING",
    "CAMERA_OFFSET_DOMAIN",
    "CAMERA_ORIENTATION_KEYS",
    "CAMERA_POLICY_DOMAIN",
    "CAMERA_SHIFT_TOKEN_BIN_LABELS",
    "CAMERA_VAE_SCALE",
    "CAMERA_ZOOM_BAND_LABELS",
    "MIRROR_MODE_V1",
    "MIRROR_POLICY_DOMAIN",
    "MIRROR_SEVERITY_BAND_LABELS",
    "MIRROR_SIDE_LABELS",
    "CameraMirrorCounts",
    "CameraMirrorPayload",
    "CameraMirrorPolicy",
    "CameraViewportCounts",
    "CameraViewportError",
    "CameraViewportPlan",
    "CameraViewportPolicy",
    "aggregate_camera_mirror",
    "aggregate_camera_viewport",
    "camera_mirror_eligible",
    "camera_shift_token_bin",
    "camera_stage_edge",
    "camera_zoom_band",
    "discover_square_bucket",
    "mirror_severity_band",
    "mirror_side",
    "plan_camera_viewport",
    "plan_mirror_geometry",
    "zero_camera_mirror_counts",
]
