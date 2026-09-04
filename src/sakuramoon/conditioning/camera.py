"""Exact full-canvas coordinate transform for camera-viewport crops.

A camera-applied sample crops an R x R square from a full canvas of
``full_width x full_height`` pixels. The training-time token coordinates
are the full-canvas crop mapping; the frozen image-coordinate map of the
square viewport (``image_coordinates(R/16, R/16)``) is an affine function
of it:

    transform(base, zoom=z, x_shift, y_shift) = (base + [y_shift, x_shift]) / z

with ``z = sqrt(full_width * full_height / R**2)`` and, for the horizontal
orientation (full_height == R, crop left edge ``left``):

    x_shift = 2 * left / R + 1 - full_width / R
    y_shift = 0

(the vertical orientation is symmetric with the roles exchanged). No new
token is added and no new position branch exists: the packed-layout
anchors (text = 0, condition = 0) are unchanged and only the image-token
coordinates differ from the ordinary path.
"""

from __future__ import annotations

import math

import torch


def transform_camera_coordinates(
    base: torch.Tensor,
    *,
    zoom: float,
    x_shift: float,
    y_shift: float,
) -> torch.Tensor:
    """Affine camera transform of a [T, 2] FP32 base coordinate map.

    The base columns are (y, x), matching
    ``sakuramoon.conditioning.rope.image_coordinates``. The identity case
    (zoom == 1.0 and both shifts == 0.0) returns the input tensor itself
    without allocation.
    """

    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError("camera coordinate base must be a [T, 2] tensor")
    if base.dtype != torch.float32:
        raise ValueError("camera coordinate base must be FP32")
    if base.device.type == "meta":
        raise ValueError("camera coordinate base must be materialized")
    for name, value in (("zoom", zoom), ("x_shift", x_shift), ("y_shift", y_shift)):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"camera coordinate {name} must be a finite float")
    if zoom <= 0.0:
        raise ValueError("camera coordinate zoom must be positive")
    if zoom == 1.0 and x_shift == 0.0 and y_shift == 0.0:
        return base
    delta = torch.tensor((y_shift, x_shift), dtype=torch.float32, device=base.device)
    return (base + delta) / zoom


def camera_transform_params(
    *,
    viewport: int,
    full_width: int,
    full_height: int,
    left: int,
    top: int,
) -> tuple[float, float, float]:
    """(zoom, x_shift, y_shift) for one applied camera-viewport plan.

    The orientation is implicit in which canvas edge equals the viewport:
    ``full_height == viewport`` selects the horizontal orientation and
    ``full_width == viewport`` the vertical one.
    """

    if (
        type(viewport) is not int
        or type(full_width) is not int
        or type(full_height) is not int
        or viewport <= 0
        or full_width <= 0
        or full_height <= 0
        or type(left) is not int
        or type(top) is not int
        or left < 0
        or top < 0
    ):
        raise ValueError("camera transform params require positive integer geometry")
    if not (full_width == viewport or full_height == viewport):
        raise ValueError(
            "camera transform params require one canvas edge equal to the viewport"
        )
    if left + viewport > full_width or top + viewport > full_height:
        raise ValueError("camera transform params crop box escapes the canvas")
    zoom = math.sqrt(float(full_width * full_height) / float(viewport * viewport))
    if full_height == viewport:
        x_shift = 2.0 * left / viewport + 1.0 - full_width / viewport
        y_shift = 0.0
    else:
        x_shift = 0.0
        y_shift = 2.0 * top / viewport + 1.0 - full_height / viewport
    return zoom, x_shift, y_shift


__all__ = [
    "camera_transform_params",
    "transform_camera_coordinates",
]
