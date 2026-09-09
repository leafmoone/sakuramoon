"""Coordinate algebra for the camera-viewport conditioning transform.

The single-zoom affine ``(base + [y_shift, x_shift]) / zoom`` must be
exactly equivalent (FP32, max_abs <= 2e-6) to the reference
``full_canvas_crop_coordinates`` crop-frame mapping on both axes.
"""

from __future__ import annotations

import math

import pytest
import torch

from sakuramoon.conditioning.camera import (
    camera_transform_params,
    transform_camera_coordinates,
)
from sakuramoon.conditioning.rope import (
    full_canvas_crop_coordinates,
    image_coordinates,
)

DEVICE = torch.device("cpu")
TOLERANCE = 2e-6


def _base(height: int, width: int) -> torch.Tensor:
    return image_coordinates(height, width, device=DEVICE)


def test_identity_fast_path_returns_same_tensor() -> None:
    base = _base(16, 32)
    assert transform_camera_coordinates(base, zoom=1.0, x_shift=0.0, y_shift=0.0) is base


def test_identity_recomputed() -> None:
    base = _base(16, 32)
    out = transform_camera_coordinates(base, zoom=1.0, x_shift=0.0, y_shift=0.0)
    assert torch.equal(out, base)


def test_horizontal_equivalence_vs_crop_coordinates() -> None:
    token_h, token_w = 32, 32
    base = _base(token_h, token_w)
    full_w, full_h = 1024, 512
    for left in (0, 64, 256, 512):
        zoom, x_shift, y_shift = camera_transform_params(
            viewport=512,
            full_width=full_w,
            full_height=full_h,
            left=left,
            top=0,
        )
        got = transform_camera_coordinates(
            base, zoom=zoom, x_shift=x_shift, y_shift=y_shift
        )
        expected = full_canvas_crop_coordinates(
            token_h,
            token_w,
            full_height=full_h,
            full_width=full_w,
            crop_box=(left, 0, left + 512, 512),
            device=DEVICE,
        )
        assert (got - expected).abs().max().item() <= TOLERANCE


def test_vertical_equivalence_vs_crop_coordinates() -> None:
    token_h, token_w = 32, 32
    base = _base(token_h, token_w)
    full_w, full_h = 512, 1024
    for top in (0, 128, 384, 512):
        zoom, x_shift, y_shift = camera_transform_params(
            viewport=512,
            full_width=full_w,
            full_height=full_h,
            left=0,
            top=top,
        )
        got = transform_camera_coordinates(
            base, zoom=zoom, x_shift=x_shift, y_shift=y_shift
        )
        expected = full_canvas_crop_coordinates(
            token_h,
            token_w,
            full_height=full_h,
            full_width=full_w,
            crop_box=(0, top, 512, top + 512),
            device=DEVICE,
        )
        assert (got - expected).abs().max().item() <= TOLERANCE


def test_canvas_extent_equivalence() -> None:
    token_h = token_w = 16
    base = _base(token_h, token_w)
    # Two arbitrary canvases: a narrow and a wide long axis.
    for full_w in (620, 1152):
        for left in (0, full_w - 512):
            zoom, x_shift, y_shift = camera_transform_params(
                viewport=512,
                full_width=full_w,
                full_height=512,
                left=left,
                top=0,
            )
            got = transform_camera_coordinates(
                base, zoom=zoom, x_shift=x_shift, y_shift=y_shift
            )
            expected = full_canvas_crop_coordinates(
                token_h,
                token_w,
                full_height=512,
                full_width=full_w,
                crop_box=(left, 0, left + 512, 512),
                device=DEVICE,
            )
            assert (got - expected).abs().max().item() <= TOLERANCE
    # The descriptive zoom is a property of the canvas aspect only:
    # any finite value >= 1.0 is legal (square source = identity 1.0).
    zoom_min, _, _ = camera_transform_params(
        viewport=512, full_width=620, full_height=512, left=0, top=0
    )
    zoom_max, _, _ = camera_transform_params(
        viewport=512, full_width=1152, full_height=512, left=0, top=0
    )
    assert zoom_min > 1.0
    assert zoom_max > zoom_min
    identity_zoom, ix, iy = camera_transform_params(
        viewport=512, full_width=512, full_height=512, left=0, top=0
    )
    assert (identity_zoom, ix, iy) == (1.0, 0.0, 0.0)


def test_transform_params_closed_form() -> None:
    zoom, x_shift, y_shift = camera_transform_params(
        viewport=512, full_width=1024, full_height=512, left=128, top=0
    )
    assert zoom == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert x_shift == pytest.approx(2.0 * 128 / 512 + 1.0 - 1024 / 512)
    assert y_shift == 0.0
    # vertical symmetry
    zoom_v, x_shift_v, y_shift_v = camera_transform_params(
        viewport=512, full_width=512, full_height=1024, left=0, top=256
    )
    assert zoom_v == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert x_shift_v == 0.0
    assert y_shift_v == pytest.approx(2.0 * 256 / 512 + 1.0 - 1024 / 512)


def test_sign_convention_positive_means_increasing_left_top() -> None:
    _, x_left, _ = camera_transform_params(
        viewport=512, full_width=1024, full_height=512, left=0, top=0
    )
    _, x_center, _ = camera_transform_params(
        viewport=512, full_width=1024, full_height=512, left=256, top=0
    )
    _, x_right, _ = camera_transform_params(
        viewport=512, full_width=1024, full_height=512, left=512, top=0
    )
    assert x_left < 0.0
    assert x_center == 0.0
    assert x_right > 0.0
    _, _, y_top = camera_transform_params(
        viewport=512, full_width=512, full_height=1024, left=0, top=0
    )
    _, _, y_bottom = camera_transform_params(
        viewport=512, full_width=512, full_height=1024, left=0, top=512
    )
    assert y_top < 0.0 < y_bottom


def test_transform_is_a_single_zoom_affine() -> None:
    base = _base(16, 32)
    zoom, x_shift, y_shift = camera_transform_params(
        viewport=512, full_width=768, full_height=512, left=96, top=0
    )
    got = transform_camera_coordinates(base, zoom=zoom, x_shift=x_shift, y_shift=y_shift)
    expected = (base + torch.tensor([y_shift, x_shift])) / zoom
    assert (got - expected).abs().max().item() == 0.0
    assert got.dtype == torch.float32
    assert got.shape == (16 * 32, 2)


def test_transform_rejects_invalid_inputs() -> None:
    base = _base(16, 32)
    with pytest.raises(ValueError):
        transform_camera_coordinates(
            base.to(torch.float64), zoom=1.0, x_shift=0.0, y_shift=0.0
        )
    with pytest.raises(ValueError):
        transform_camera_coordinates(base.flatten(), zoom=1.0, x_shift=0.0, y_shift=0.0)
    with pytest.raises(ValueError):
        transform_camera_coordinates(base, zoom=0.0, x_shift=0.0, y_shift=0.0)
    with pytest.raises(ValueError):
        transform_camera_coordinates(base, zoom=float("nan"), x_shift=0.0, y_shift=0.0)
    with pytest.raises(ValueError):
        transform_camera_coordinates(base, zoom=1.1, x_shift=float("inf"), y_shift=0.0)
    with pytest.raises(ValueError):
        transform_camera_coordinates(base, zoom=1.1, x_shift=1, y_shift=0.0)


def test_transform_rejects_meta_device() -> None:
    base = torch.empty(16 * 32, 2, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError):
        transform_camera_coordinates(base, zoom=1.0, x_shift=0.0, y_shift=0.0)


def test_transform_params_reject_invalid_geometry() -> None:
    with pytest.raises(ValueError):
        camera_transform_params(
            viewport=512, full_width=768, full_height=640, left=0, top=0
        )
    with pytest.raises(ValueError):
        camera_transform_params(
            viewport=512, full_width=1024, full_height=512, left=513, top=0
        )
    with pytest.raises(ValueError):
        camera_transform_params(
            viewport=512, full_width=1024, full_height=512, left=-1, top=0
        )
    with pytest.raises(ValueError):
        camera_transform_params(
            viewport=512, full_width=1024, full_height=512, left=0, top=1
        )
