"""Shifted-square camera viewport: one source sample, one training view.

Pure geometry plus isolated deterministic RNG domains; no tensors and no I/O,
so the policy and every plan are spawn-picklable and DataLoader-worker safe.

Semantics (square-policy contract):

* Activation is config-only. The camera path runs iff the resolved policy
  is ``enabled`` and ``probability > 0.0``. An inactive camera never builds
  a plan, never draws the camera RNG domains, and keeps the ordinary
  dev path (aspect bucket / spatial crop / caption / RNG) bit-identical.
* A camera-selected sample plans its geometry directly from the post-EXIF
  source size; the ordinary ``prepare_image`` / ``assign_bucket`` admission
  is never consulted for it. The short edge is scaled to the stage square
  target ``R``, the long edge is quantized by rounding half up, and one
  uniform integer offset (both endpoints inclusive) is drawn on the scaled
  long axis. The emitted view is the ``R x R`` crop of that full canvas
  (crop-after-scale), located exactly by its crop box so the training
  full-canvas coordinate transform stays exact.
* No-upscale rule: a camera-selected source whose post-EXIF short edge is
  below ``R`` is rejected explicitly (``no_upscale``). It is never scaled
  up and never silently returned to an ordinary bucket.
* ``equivalent_zoom`` and ``retention`` are descriptive numerics only:
  zoom is ``sqrt(canvas area / R^2)`` (any finite value >= 1.0; a square
  source is the identity zoom 1.0) and retention is the geometric area
  ratio ``R^2 / canvas area``. There is no zoom window, no aspect ceiling,
  and no fixed observation bins in this module.
* Pure pixel geometry: no VAE scale or token stride constants live here.

The stage square target ``R`` is discovered from the bucket vocabulary
(the unique square stage bucket); a missing or ambiguous square target
fails fast at construction instead of degrading into per-sample handling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeGuard

from sakuramoon.data.buckets import BucketShape

if TYPE_CHECKING:
    from sakuramoon.config.schema import DataCameraViewportConfig

CAMERA_POLICY_DOMAIN = "camera-policy"
CAMERA_OFFSET_DOMAIN = "camera-offset"

# Fixed audit record states. ``no_upscale`` marks a camera-selected sample
# whose source short edge is below the stage target; it is a rejection,
# not a fallback, so it never appears in an ImageAudit.
CAMERA_FALLBACK_REASONS: tuple[str, ...] = ("none", "not_selected", "no_upscale")

# Applied camera orientations, derived from the source aspect: exactly one
# canvas edge (the short axis) equals the viewport, the other may extend.
CAMERA_ORIENTATION_KEYS: tuple[str, ...] = ("horizontal", "vertical", "square")


class CameraViewportError(ValueError):
    """Camera viewport configuration or geometry violates its contract."""


def _round_half_up(value: float) -> int:
    """Quantize a positive value with a consistent round-half-up rule."""

    if not math.isfinite(value) or value <= 0.0:
        raise CameraViewportError("quantized value must be finite and positive")
    return math.floor(value + 0.5)


def camera_stage_edge(buckets: tuple[BucketShape, ...]) -> int:
    """Width of the unique square bucket in the stage-scaled vocabulary.

    Fail-fast by contract: the camera viewport requires exactly one square
    bucket (the stage square target ``R``). A missing or ambiguous square
    target raises here (construction / factory boundary) instead of being
    swallowed as a per-sample decode error later.
    """

    squares = tuple(shape for shape in buckets if shape.width == shape.height)
    if len(squares) != 1:
        raise CameraViewportError(
            "camera viewport requires exactly one square bucket in the "
            f"stage vocabulary, found {len(squares)}"
        )
    return squares[0].width


@dataclass(frozen=True, slots=True)
class CameraViewportPolicy:
    """Resolved camera viewport policy handed to the pipeline worker.

    ``enabled`` and ``probability`` are independent config values: the
    effective activation is ``enabled AND probability > 0``. ``p = 0`` is
    a legal off switch and ``enabled = false`` is off regardless of p.
    """

    enabled: bool
    probability: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise CameraViewportError("camera viewport enabled must be a boolean")
        if (
            type(self.probability) is not float
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise CameraViewportError(
                "camera viewport probability must be a finite number in [0, 1]"
            )

    @classmethod
    def from_config(cls, config: DataCameraViewportConfig) -> CameraViewportPolicy:
        return cls(enabled=config.enabled, probability=config.probability)


def camera_is_active(policy: CameraViewportPolicy | None) -> TypeGuard[CameraViewportPolicy]:
    """Single activation gate for every camera construction and planning site.

    The camera path is active iff a resolved policy exists and is enabled
    with ``probability > 0.0``. Construction boundaries (pipeline
    ``__init__``, lease clone, production factory) call
    ``camera_stage_edge`` only under this gate, so an effectively-off
    camera never imposes the square-bucket requirement; the planning site
    uses the same gate, so direct construction, factory issuance, and
    clones can never disagree about activation.

    The ``TypeGuard`` lets the gate double as type narrowing: in every
    guarded branch the caller sees a non-Optional resolved policy.
    """

    return policy is not None and policy.enabled and policy.probability > 0.0


@dataclass(frozen=True, slots=True)
class CameraViewportPlan:
    """Planned shifted-square camera geometry for one source sample.

    When ``applied`` the canvas is ``full_width x full_height`` with the
    short axis exactly equal to ``viewport`` (``R``) and the ``R x R``
    crop located by ``left`` / ``top``. ``equivalent_zoom`` and
    ``retention`` are descriptive (zoom = 1 / sqrt(retention) >= 1.0).
    When not applied every geometry field is zero.
    """

    applied: bool
    fallback_reason: str
    orientation: str
    viewport: int
    full_width: int
    full_height: int
    left: int
    top: int
    crop_box: tuple[int, int, int, int]
    equivalent_zoom: float
    retention: float

    def __post_init__(self) -> None:
        if type(self.applied) is not bool:
            raise CameraViewportError("camera plan applied must be a boolean")
        if self.fallback_reason not in CAMERA_FALLBACK_REASONS:
            raise CameraViewportError(
                f"unknown camera plan fallback reason: {self.fallback_reason}"
            )
        if self.applied == (self.fallback_reason != "none"):
            raise CameraViewportError("camera plan applied/reason are inconsistent")
        geometry = (
            self.viewport,
            self.full_width,
            self.full_height,
            self.left,
            self.top,
        )
        if any(type(value) is not int or value < 0 for value in geometry):
            raise CameraViewportError(
                "camera plan geometry must be non-negative integers"
            )
        if type(self.crop_box) is not tuple or len(self.crop_box) != 4:
            raise CameraViewportError("camera plan crop box must be a 4-tuple")
        if any(type(value) is not int or value < 0 for value in self.crop_box):
            raise CameraViewportError("camera plan crop box must hold non-negative ints")
        if (
            type(self.equivalent_zoom) is not float
            or not math.isfinite(self.equivalent_zoom)
            or self.equivalent_zoom < 0.0
            or type(self.retention) is not float
            or not math.isfinite(self.retention)
            or self.retention < 0.0
        ):
            raise CameraViewportError(
                "camera plan zoom/retention must be finite non-negative floats"
            )
        if not self.applied:
            if self.orientation != "none":
                raise CameraViewportError("unapplied camera plan must have no orientation")
            if any(value != 0 for value in geometry) or any(
                value != 0 for value in self.crop_box
            ):
                raise CameraViewportError("unapplied camera plan must be all-zero geometry")
            if self.equivalent_zoom != 0.0 or self.retention != 0.0:
                raise CameraViewportError("unapplied camera plan must carry zero numerics")
            return
        if self.orientation not in CAMERA_ORIENTATION_KEYS:
            raise CameraViewportError(
                f"applied camera plan has invalid orientation: {self.orientation}"
            )
        if self.viewport <= 0:
            raise CameraViewportError("applied camera plan requires a positive viewport")
        if (
            self.full_width < self.viewport
            or self.full_height < self.viewport
            or min(self.full_width, self.full_height) != self.viewport
        ):
            raise CameraViewportError(
                "applied camera canvas must keep its short edge equal to the viewport"
            )
        if self.orientation == "horizontal" and self.full_height != self.viewport:
            raise CameraViewportError("horizontal camera canvas must span the viewport vertically")
        if self.orientation == "vertical" and self.full_width != self.viewport:
            raise CameraViewportError("vertical camera canvas must span the viewport horizontally")
        if self.orientation == "square" and (
            self.full_width != self.viewport or self.full_height != self.viewport
        ):
            raise CameraViewportError("square camera canvas must be the viewport square")
        if self.orientation == "horizontal":
            if self.top != 0 or not 0 <= self.left <= self.full_width - self.viewport:
                raise CameraViewportError("horizontal camera offset must lie on the long axis")
        elif self.orientation == "vertical":
            if self.left != 0 or not 0 <= self.top <= self.full_height - self.viewport:
                raise CameraViewportError("vertical camera offset must lie on the long axis")
        else:
            if self.left != 0 or self.top != 0:
                raise CameraViewportError("square camera canvas admits no offset")
        if self.crop_box != (
            self.left,
            self.top,
            self.left + self.viewport,
            self.top + self.viewport,
        ):
            raise CameraViewportError("camera plan crop box disagrees with left/top/viewport")
        canvas_area = self.full_width * self.full_height
        viewport_area = self.viewport * self.viewport
        if not (self.equivalent_zoom >= 1.0 and self.retention > 0.0):
            raise CameraViewportError(
                "applied camera plan requires zoom >= 1.0 and positive retention"
            )
        if abs(self.equivalent_zoom * math.sqrt(self.retention) - 1.0) > 1e-9:
            raise CameraViewportError(
                "camera plan zoom and retention are not consistent (zoom = 1/sqrt(retention))"
            )
        if (
            abs(self.equivalent_zoom - math.sqrt(canvas_area / viewport_area)) > 1e-9
            or abs(self.retention - viewport_area / canvas_area) > 1e-9
        ):
            raise CameraViewportError("camera plan numerics disagree with the canvas geometry")


def _unapplied_plan(
    fallback_reason: Literal["not_selected", "no_upscale"],
) -> CameraViewportPlan:
    """All-zero unapplied plan for one fallback outcome.

    Constructing the zero plan at each fallback site (after the selection
    RNG draw) keeps the planner's RNG call positions identical for the
    not-selected, no-upscale, and applied paths.
    """
    return CameraViewportPlan(
        applied=False,
        fallback_reason=fallback_reason,
        orientation="none",
        viewport=0,
        full_width=0,
        full_height=0,
        left=0,
        top=0,
        crop_box=(0, 0, 0, 0),
        equivalent_zoom=0.0,
        retention=0.0,
    )


def plan_camera_viewport(
    policy: CameraViewportPolicy,
    *,
    stage_edge: int,
    source_size: tuple[int, int],
    policy_seed: int,
    offset_seed: int,
) -> CameraViewportPlan:
    """Plan one shifted-square camera viewport from the post-EXIF source size.

    Contract: callers run this only while the camera is active (``enabled``
    and ``probability > 0``). Selection draws the isolated camera-policy RNG
    domain and the offset draws the isolated camera-offset RNG domain; no
    ordinary crop RNG domain is touched. The returned plan is either
    applied (shifted-square geometry), ``not_selected`` (ordinary path), or
    ``no_upscale`` (explicit rejection, no fallback).
    """

    if type(stage_edge) is not int or stage_edge <= 0:
        raise CameraViewportError("stage edge must be a positive integer")
    if type(source_size) is not tuple or len(source_size) != 2:
        raise CameraViewportError("source size must be a (width, height) pair")
    source_width, source_height = source_size
    if (
        type(source_width) is not int
        or type(source_height) is not int
        or source_width <= 0
        or source_height <= 0
    ):
        raise CameraViewportError("source size must be positive integers")
    if (
        type(policy_seed) is not int
        or policy_seed < 0
        or type(offset_seed) is not int
        or offset_seed < 0
    ):
        raise CameraViewportError("camera seeds must be non-negative integers")
    if not policy.enabled or policy.probability <= 0.0:
        raise CameraViewportError(
            "plan_camera_viewport requires an active camera policy"
        )

    selection_rng = random.Random(policy_seed)
    if selection_rng.random() >= policy.probability:
        return _unapplied_plan("not_selected")

    short_edge = min(source_width, source_height)
    if short_edge < stage_edge:
        return _unapplied_plan("no_upscale")

    long_edge = max(source_width, source_height)
    long_quantized = _round_half_up(long_edge * stage_edge / short_edge)
    if source_width == source_height:
        orientation = "square"
        full_width = stage_edge
        full_height = stage_edge
    elif source_width > source_height:
        orientation = "horizontal"
        full_width = long_quantized
        full_height = stage_edge
    else:
        orientation = "vertical"
        full_width = stage_edge
        full_height = long_quantized
    long_axis_range = max(full_width, full_height) - stage_edge
    offset = random.Random(offset_seed).randrange(long_axis_range + 1)
    if orientation == "horizontal":
        left, top = offset, 0
    elif orientation == "vertical":
        left, top = 0, offset
    else:
        left, top = 0, 0
    canvas_area = full_width * full_height
    viewport_area = stage_edge * stage_edge
    return CameraViewportPlan(
        applied=True,
        fallback_reason="none",
        orientation=orientation,
        viewport=stage_edge,
        full_width=full_width,
        full_height=full_height,
        left=left,
        top=top,
        crop_box=(left, top, left + stage_edge, top + stage_edge),
        equivalent_zoom=math.sqrt(canvas_area / viewport_area),
        retention=viewport_area / canvas_area,
    )


@dataclass(frozen=True, slots=True)
class CameraViewportCounts:
    """Fixed camera-viewport counters aggregated over one batch of audits.

    Only the counts with a real consumer are kept: selection / application
    totals plus the applied-orientation mix. Strict zero semantics: with no
    applied sample every counter is zero and the fixed orientation keys
    stay visible.
    """

    selected: int
    applied: int
    orientation_counts: Mapping[str, int]

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
        if set(self.orientation_counts) != set(CAMERA_ORIENTATION_KEYS):
            raise CameraViewportError(
                "camera viewport orientation counts must carry the fixed key set"
            )
        for value in self.orientation_counts.values():
            if type(value) is not int or value < 0:
                raise CameraViewportError(
                    "camera viewport orientation counts must be nonnegative integers"
                )
        if sum(self.orientation_counts.values()) != self.applied:
            raise CameraViewportError(
                "camera viewport orientation counts must sum to the applied count"
            )


def aggregate_camera_viewport(audits: Iterable[object]) -> CameraViewportCounts:
    """Aggregate the fixed camera-viewport counters from ImageAudit records."""

    orientation_counts = {key: 0 for key in CAMERA_ORIENTATION_KEYS}
    selected = 0
    applied = 0
    for audit in audits:
        if not audit.camera_selected:  # type: ignore[attr-defined]
            continue
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
    return CameraViewportCounts(
        selected=selected,
        applied=applied,
        orientation_counts=dict(orientation_counts),
    )


__all__ = [
    "CAMERA_FALLBACK_REASONS",
    "CAMERA_OFFSET_DOMAIN",
    "CAMERA_ORIENTATION_KEYS",
    "CAMERA_POLICY_DOMAIN",
    "CameraViewportCounts",
    "CameraViewportError",
    "CameraViewportPlan",
    "CameraViewportPolicy",
    "aggregate_camera_viewport",
    "camera_stage_edge",
    "plan_camera_viewport",
]
