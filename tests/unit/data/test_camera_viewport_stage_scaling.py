"""Multi-resolution stage scaling for the shifted-square camera viewport.

``R`` is always the unique square stage bucket of the stage-scaled bucket
vocabulary (never a hardcoded constant), and the same source aspect yields
consistently scaled canvases across every approved stage resolution.
"""

from __future__ import annotations

import math

import pytest

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
from sakuramoon.data.camera_viewport import (
    CameraViewportPolicy,
    camera_stage_edge,
    plan_camera_viewport,
)

POLICY = CameraViewportPolicy(enabled=True, probability=1.0)
STAGE_EDGES = (256, 512, 768, 1024)

_BUCKET_CONFIG = DataBucketsConfig(
    base_area_px=262144,
    quantum_px=32,
    min_short_edge_px=256,
    max_aspect_ratio=4.0,
    transpose_closed=True,
)


@pytest.mark.parametrize("stage_edge", STAGE_EDGES)
def test_stage_edge_matches_train_resolution(stage_edge: int) -> None:
    buckets = scale_buckets(generate_base_buckets(_BUCKET_CONFIG), stage_edge)
    assert camera_stage_edge(buckets) == stage_edge


@pytest.mark.parametrize("stage_edge", STAGE_EDGES)
def test_canvas_scales_with_stage_edge(stage_edge: int) -> None:
    buckets = scale_buckets(generate_base_buckets(_BUCKET_CONFIG), stage_edge)
    R = camera_stage_edge(buckets)
    plan = plan_camera_viewport(
        POLICY,
        stage_edge=R,
        source_size=(4000, 2000),  # 2:1, short edge >= every stage R
        policy_seed=1,
        offset_seed=2,
    )
    assert plan.applied
    assert (plan.full_width, plan.full_height) == (2 * R, R)
    assert plan.crop_box[2] - plan.crop_box[0] == R
    assert plan.crop_box[3] - plan.crop_box[1] == R
    assert plan.equivalent_zoom == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_aspect_is_preserved_across_resolutions() -> None:
    """The 3:1 canvas keeps a 3:1 canvas at every stage resolution."""

    canvases: list[tuple[int, int]] = []
    for stage_edge in STAGE_EDGES:
        buckets = scale_buckets(generate_base_buckets(_BUCKET_CONFIG), stage_edge)
        R = camera_stage_edge(buckets)
        plan = plan_camera_viewport(
            POLICY,
            stage_edge=R,
            source_size=(6000, 2000),  # 3:1, short edge >= every stage R
            policy_seed=1,
            offset_seed=2,
        )
        assert plan.applied
        canvases.append((plan.full_width, plan.full_height))
        assert plan.full_height == R
        assert plan.full_width == 3 * R

    assert [canvas[0] for canvas in canvases] == [768, 1536, 2304, 3072]


def test_vertical_orientation_scales_symmetrically() -> None:
    for stage_edge in (256, 1024):
        buckets = scale_buckets(generate_base_buckets(_BUCKET_CONFIG), stage_edge)
        R = camera_stage_edge(buckets)
        plan = plan_camera_viewport(
            POLICY,
            stage_edge=R,
            source_size=(2000, 10000),  # 1:5 portrait
            policy_seed=3,
            offset_seed=4,
        )
        assert plan.applied
        assert plan.orientation == "vertical"
        assert (plan.full_width, plan.full_height) == (R, 5 * R)
        assert plan.left == 0
        assert 0 <= plan.top <= 4 * R


def test_no_upscale_boundary_moves_with_stage_edge() -> None:
    for stage_edge in (256, 1024):
        buckets = scale_buckets(generate_base_buckets(_BUCKET_CONFIG), stage_edge)
        R = camera_stage_edge(buckets)
        accepted = plan_camera_viewport(
            POLICY,
            stage_edge=R,
            source_size=(2 * R, R),
            policy_seed=1,
            offset_seed=2,
        )
        rejected = plan_camera_viewport(
            POLICY,
            stage_edge=R,
            source_size=(2 * R, R - 1),
            policy_seed=1,
            offset_seed=2,
        )
        assert accepted.applied
        assert rejected.fallback_reason == "no_upscale"
