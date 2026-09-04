"""Exact tests for the camera_eval fixed manifest and class assignment.

The manifest is the only variable of the camera-control evaluation matrix;
everything else (checkpoint, prompts, initial noise, sampler profile, CFG)
is fixed. Class assignments are statements about the hdm_shifted_square_v2
training distribution, never about the HDM public sources.
"""

from __future__ import annotations

import math

import pytest

from sakuramoon.cli.camera_eval import (
    CAMERA_EVAL_CLASSES,
    CAMERA_EVAL_MANIFEST_VERSION,
    COUPLED_ANCHOR_ZOOMS,
    CameraEvalPoint,
    build_camera_eval_manifest,
    classify_camera_eval_point,
    coupled_edge_shift,
)


def test_manifest_is_fixed_and_deterministic() -> None:
    first = build_camera_eval_manifest()
    second = build_camera_eval_manifest()
    assert first == second
    assert len(first) == 23
    names = tuple(point.name for point in first)
    assert len(set(names)) == 23
    assert CAMERA_EVAL_MANIFEST_VERSION == 1
    groups = {point.group for point in first}
    assert groups == {"legacy", "hdm_demo", "training_coupled", "diagonal"}
    for point in first:
        assert point.manifest_class == classify_camera_eval_point(
            point.zoom, point.x_shift, point.y_shift
        )


def test_expected_class_assignment_map() -> None:
    manifest = {
        point.name: point for point in build_camera_eval_manifest()
    }
    expected = {
        "legacy_center_z1": "extrapolation",
        "hdm_demo_x-0.25_z1": "decoupled_shift_extrapolation",
        "hdm_demo_x+0.25_z1": "decoupled_shift_extrapolation",
        "hdm_demo_y-0.25_z1": "decoupled_shift_extrapolation",
        "hdm_demo_y+0.25_z1": "decoupled_shift_extrapolation",
        "hdm_demo_z0.75": "zoom_out_extrapolation",
        "hdm_demo_z1.33": "interpolation",
    }
    for name, cls in expected.items():
        assert manifest[name].manifest_class == cls, name
    for point in build_camera_eval_manifest():
        if point.group == "training_coupled":
            assert point.manifest_class == "in_distribution_coupled", point.name
        if point.group == "diagonal":
            assert point.manifest_class == "decoupled_shift_extrapolation", point.name
    coupled = [
        point
        for point in build_camera_eval_manifest()
        if point.manifest_class == "in_distribution_coupled"
    ]
    assert len(coupled) == 12


def test_zoom_out_and_pure_z1_shift_never_in_distribution() -> None:
    for zoom in (0.5, 0.75, 0.99):
        assert (
            classify_camera_eval_point(zoom, 0.0, 0.0)
            == "zoom_out_extrapolation"
        )
    for shift in (-0.25, 0.25, -1.0, 1.0):
        assert (
            classify_camera_eval_point(1.0, shift, 0.0)
            == "decoupled_shift_extrapolation"
        )
        assert (
            classify_camera_eval_point(1.0, 0.0, shift)
            == "decoupled_shift_extrapolation"
        )
    # The legacy baseline (z=1, no shift) is out of the training band.
    assert classify_camera_eval_point(1.0, 0.0, 0.0) == "extrapolation"


def test_diagonal_shifts_never_in_distribution() -> None:
    for zoom in (1.1, 1.225, math.sqrt(2.0), 1.5):
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                assert (
                    classify_camera_eval_point(zoom, sx, sy)
                    == "decoupled_shift_extrapolation"
                )


def test_single_axis_shift_within_coupled_range_is_in_distribution() -> None:
    zoom = math.sqrt(2.0)
    edge = coupled_edge_shift(zoom)
    for shift in (-edge, -0.37, 0.0, 0.37, edge):
        assert (
            classify_camera_eval_point(zoom, shift, 0.0)
            == "in_distribution_coupled"
        )
        assert (
            classify_camera_eval_point(zoom, 0.0, shift)
            == "in_distribution_coupled"
        )
    # Beyond the coupled magnitude the configuration is unreachable in
    # training at this zoom.
    assert (
        classify_camera_eval_point(zoom, edge + 0.5, 0.0)
        == "decoupled_shift_extrapolation"
    )


def test_coupled_anchor_zooms_and_edge_magnitudes() -> None:
    assert COUPLED_ANCHOR_ZOOMS == (1.10, 1.225, math.sqrt(2.0), 1.50)
    assert coupled_edge_shift(1.10) == pytest.approx(0.21, abs=1e-12)
    # 1.225 is a decimal anchor: 1.225**2 - 1 = 0.500625 exactly (not 0.5).
    assert coupled_edge_shift(1.225) == pytest.approx(0.500625, abs=1e-12)
    assert 0.5 < coupled_edge_shift(1.225) < 0.501
    assert coupled_edge_shift(math.sqrt(2.0)) == pytest.approx(1.0, abs=1e-12)
    assert coupled_edge_shift(1.50) == pytest.approx(1.25, abs=1e-12)
    with pytest.raises(ValueError):
        coupled_edge_shift(1.0)
    with pytest.raises(ValueError):
        coupled_edge_shift(0.5)


def test_sign_convention_positive_means_increasing_left_top() -> None:
    manifest = {
        point.name: point for point in build_camera_eval_manifest()
    }
    for zoom in COUPLED_ANCHOR_ZOOMS:
        slug = f"z{zoom:.4f}".rstrip("0")
        left = manifest[f"coupled_{slug}_left_edge"]
        right = manifest[f"coupled_{slug}_right_edge"]
        center = manifest[f"coupled_{slug}_center"]
        assert left.x_shift < 0.0
        assert right.x_shift > 0.0
        assert center.x_shift == 0.0
        assert center.y_shift == 0.0
        assert left.x_shift == pytest.approx(-coupled_edge_shift(zoom))
        assert right.x_shift == pytest.approx(coupled_edge_shift(zoom))
        assert left.y_shift == 0.0 and right.y_shift == 0.0


def test_point_validation() -> None:
    with pytest.raises(ValueError):
        CameraEvalPoint(
            name="x",
            group="bogus",
            zoom=1.2,
            x_shift=0.0,
            y_shift=0.0,
            manifest_class="extrapolation",
        )
    with pytest.raises(ValueError):
        CameraEvalPoint(
            name="x",
            group="legacy",
            zoom=1.2,
            x_shift=0.0,
            y_shift=0.0,
            manifest_class="bogus-class",
        )
    with pytest.raises(ValueError):
        CameraEvalPoint(
            name="  padded",
            group="legacy",
            zoom=1.2,
            x_shift=0.0,
            y_shift=0.0,
            manifest_class="extrapolation",
        )
    with pytest.raises(ValueError):
        CameraEvalPoint(
            name="x",
            group="legacy",
            zoom=0.0,
            x_shift=0.0,
            y_shift=0.0,
            manifest_class="extrapolation",
        )
    with pytest.raises(ValueError):
        CameraEvalPoint(
            name="x",
            group="legacy",
            zoom=1.2,
            x_shift=1,  # int is not a float
            y_shift=0.0,
            manifest_class="extrapolation",
        )


def test_classifier_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError):
        classify_camera_eval_point(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError):
        classify_camera_eval_point(1.2, float("inf"), 0.0)
    with pytest.raises(ValueError):
        classify_camera_eval_point(-1.0, 0.0, 0.0)


def test_manifest_covers_all_five_classes() -> None:
    used = {point.manifest_class for point in build_camera_eval_manifest()}
    assert used == set(CAMERA_EVAL_CLASSES)
