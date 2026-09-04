"""Planner guarantees for the hdm_shifted_square_v2 camera viewport.

Geometry authority is the SakuraMoon frozen invariant set (bucket
vocabulary, min_crop_retention admission, stage-scaled square discovery),
never the HDM public sources.
"""

from __future__ import annotations

import math
import pickle
import random
from types import SimpleNamespace
from typing import cast

import pytest

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import (
    BucketRejection,
    BucketShape,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.camera_viewport import (
    CAMERA_FALLBACK_REASONS,
    CAMERA_SHIFT_TOKEN_BIN_LABELS,
    CAMERA_ZOOM_BAND_LABELS,
    CameraViewportError,
    CameraViewportPolicy,
    aggregate_camera_viewport,
    camera_shift_token_bin,
    camera_stage_edge,
    camera_zoom_band,
    discover_square_bucket,
    plan_camera_viewport,
)

STAGE_EDGE = 512


def _buckets() -> tuple[BucketShape, ...]:
    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        shape_count=17,
        transpose_closed=True,
    )
    return scale_buckets(generate_base_buckets(config), STAGE_EDGE)


BUCKETS = _buckets()


def _assignment(width: int, height: int, buckets: tuple[BucketShape, ...] | None = None):
    result = assign_bucket(
        width,
        height,
        BUCKETS if buckets is None else buckets,
        min_crop_retention=0.8,
    )
    assert not isinstance(result, BucketRejection)
    return result


def _plan(
    width: int,
    height: int,
    *,
    buckets: tuple[BucketShape, ...] | None = None,
    probability: float = 1.0,
    policy_seed: int = 1,
    offset_seed: int = 1,
    min_zoom: float = 1.10,
    max_zoom: float = 1.50,
):
    policy = CameraViewportPolicy(
        enabled=True,
        probability=probability,
        min_equivalent_zoom=min_zoom,
        max_equivalent_zoom=max_zoom,
    )
    return plan_camera_viewport(
        _assignment(width, height, buckets),
        policy,
        buckets=BUCKETS if buckets is None else buckets,
        stage_edge=STAGE_EDGE,
        source_size=(width, height),
        policy_seed=policy_seed,
        offset_seed=offset_seed,
    )


def test_square_bucket_discovery() -> None:
    square = discover_square_bucket(BUCKETS, stage_edge=STAGE_EDGE)
    assert square == STAGE_EDGE
    assert discover_square_bucket(BUCKETS, stage_edge=256) is None
    assert (
        discover_square_bucket(
            # Duck-type stand-in: the discovery contract reads width/height only.
            cast(
                "tuple[BucketShape, ...]",
                (type("B", (), {"width": 512, "height": 512})(),),
            ),
            stage_edge=STAGE_EDGE,
        )
        == STAGE_EDGE
    )


def test_camera_stage_edge() -> None:
    assert camera_stage_edge(BUCKETS) == STAGE_EDGE
    assert camera_stage_edge((BUCKETS[0], BUCKETS[1])) == 0


def test_zoom_band_labels_and_bounds() -> None:
    assert CAMERA_ZOOM_BAND_LABELS == (
        "[1.10,1.20)",
        "[1.20,1.35)",
        "[1.35,1.501]",
    )
    assert camera_zoom_band(1.10) == 0
    assert camera_zoom_band(1.1999) == 0
    assert camera_zoom_band(1.20) == 1
    assert camera_zoom_band(1.35) == 2
    assert camera_zoom_band(1.50) == 2
    assert camera_zoom_band(1.501) == 2
    with pytest.raises(ValueError):
        camera_zoom_band(1.0999)
    with pytest.raises(ValueError):
        camera_zoom_band(1.502)


def test_shift_token_bin_labels_and_bounds() -> None:
    assert len(CAMERA_SHIFT_TOKEN_BIN_LABELS) == 6
    assert camera_shift_token_bin(0.0) == 0
    assert camera_shift_token_bin(0.99) == 0
    assert camera_shift_token_bin(1.0) == 1
    assert camera_shift_token_bin(1.99) == 1
    assert camera_shift_token_bin(2.0) == 2
    assert camera_shift_token_bin(3.99) == 2
    assert camera_shift_token_bin(4.0) == 3
    assert camera_shift_token_bin(7.99) == 3
    assert camera_shift_token_bin(8.0) == 4
    assert camera_shift_token_bin(15.99) == 4
    assert camera_shift_token_bin(16.0) == 5
    assert camera_shift_token_bin(1e9) == 5
    with pytest.raises(ValueError):
        camera_shift_token_bin(-1e-9)


def test_horizontal_geometry_exact() -> None:
    plan = _plan(1024, 512)
    assert plan.applied
    assert plan.fallback_reason == "none"
    assert plan.orientation == "horizontal"
    assert (plan.full_width, plan.full_height) == (1024, 512)
    assert plan.viewport == STAGE_EDGE
    assert plan.equivalent_zoom == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert plan.retention == pytest.approx(0.5, rel=1e-12)
    left, top, right, bottom = plan.crop_box
    assert top == 0 and bottom == STAGE_EDGE
    assert 0 <= left <= 512 and right == left + STAGE_EDGE
    # Coupled algebra: (base + [y_shift, x_shift]) / zoom must reproduce the
    # crop-frame coordinates, so the shifts are pinned to the crop geometry.
    assert plan.camera_shift_x == pytest.approx(
        2.0 * left / STAGE_EDGE + 1.0 - plan.full_width / STAGE_EDGE
    )
    assert plan.camera_shift_y == 0.0
    signed = left + STAGE_EDGE // 2 - plan.full_width / 2.0
    assert plan.signed_pixel_center_shift == pytest.approx(signed)
    assert plan.absolute_pixel_center_shift == pytest.approx(abs(signed))
    assert plan.latent_center_shift == pytest.approx(abs(signed) / 16.0)
    # Coupled consistency: camera_shift_x is exactly 2 * signed / viewport.
    assert plan.camera_shift_x == pytest.approx(2.0 * signed / STAGE_EDGE)


def test_vertical_geometry_exact() -> None:
    plan = _plan(512, 1024)
    assert plan.applied
    assert plan.orientation == "vertical"
    assert (plan.full_width, plan.full_height) == (512, 1024)
    left, top, right, bottom = plan.crop_box
    assert left == 0 and right == STAGE_EDGE
    assert 0 <= top <= 512 and bottom == top + STAGE_EDGE
    assert plan.camera_shift_y == pytest.approx(
        2.0 * top / STAGE_EDGE + 1.0 - plan.full_height / STAGE_EDGE
    )
    assert plan.camera_shift_x == 0.0
    signed = top + STAGE_EDGE // 2 - plan.full_height / 2.0
    assert plan.signed_pixel_center_shift == pytest.approx(signed)
    assert plan.absolute_pixel_center_shift == pytest.approx(abs(signed))
    assert plan.latent_center_shift == pytest.approx(abs(signed) / 16.0)
    assert plan.camera_shift_y == pytest.approx(2.0 * signed / STAGE_EDGE)


def test_exact_min_zoom_boundary_inclusive() -> None:
    # z_ideal = sqrt(1210/1000) = 1.10 exactly: the boundary must apply.
    plan = _plan(1210, 1000)
    assert plan.applied
    assert plan.orientation == "horizontal"
    assert plan.full_width == 620
    assert plan.equivalent_zoom >= 1.10 - 1e-12


def test_exact_max_zoom_boundary_inclusive() -> None:
    # z_ideal = sqrt(2250/1000) = 1.50 exactly: the boundary must apply.
    plan = _plan(2250, 1000)
    assert plan.applied
    assert plan.full_width == 1152
    assert plan.equivalent_zoom == pytest.approx(1.5, rel=1e-12)


def test_quantization_cannot_escape_the_zoom_band() -> None:
    # For every admissible source whose ideal zoom lies inside [1.10, 1.50],
    # the quantized canvas must also stay inside the band (half-up rounding
    # at 0.5 px cannot push z out of the band): no quantized_no_effect.
    rng = random.Random(20260905)
    checked = 0
    for short in range(STAGE_EDGE, 1600, 37):
        for aspect in (1.0 + 0.01 * index for index in range(10, 55)):
            if not (1.10**2 <= aspect <= 1.50**2):
                continue
            width = max(short, round(short * aspect))
            height = short
            result = assign_bucket(
                width,
                height,
                BUCKETS,
                min_crop_retention=0.8,
            )
            if isinstance(result, BucketRejection):
                continue
            plan = plan_camera_viewport(
                result,
                CameraViewportPolicy(True, 1.0, 1.10, 1.50),
                buckets=BUCKETS,
                stage_edge=STAGE_EDGE,
                source_size=(width, height),
                policy_seed=rng.randrange(2**63),
                offset_seed=rng.randrange(2**63),
            )
            assert plan.applied, (width, height, plan.fallback_reason)
            assert plan.fallback_reason != "quantized_no_effect"
            checked += 1
    assert checked > 200


def test_fallback_short_edge_too_small() -> None:
    plan = _plan(400, 800)
    assert not plan.applied
    assert plan.fallback_reason == "short_edge_too_small"


def test_fallback_near_square_below_min() -> None:
    plan = _plan(600, 600)
    assert not plan.applied
    assert plan.fallback_reason == "near_square_below_min"
    # 1210x1000 (z_ideal exactly 1.10) must NOT be near-square: boundary.
    assert _plan(1210, 1000).applied


def test_fallback_aspect_above_max() -> None:
    plan = _plan(1280, 512)
    assert not plan.applied
    assert plan.fallback_reason == "aspect_above_max"
    # 2250x1000 (z_ideal exactly 1.50) must NOT be above max: boundary.
    assert _plan(2250, 1000).applied


def test_fallback_not_selected() -> None:
    policy = CameraViewportPolicy(True, 0.25, 1.10, 1.50)
    not_selected = 0
    for seed in range(400):
        plan = plan_camera_viewport(
            _assignment(1024, 512),
            policy,
            buckets=BUCKETS,
            stage_edge=STAGE_EDGE,
            source_size=(1024, 512),
            policy_seed=seed,
            offset_seed=seed,
        )
        if plan.fallback_reason == "not_selected":
            not_selected += 1
            assert not plan.applied
            assert plan.equivalent_zoom == 0.0
            assert plan.crop_box == (0, 0, 0, 0)
            assert plan.orientation == "none"
    assert not_selected > 20  # p=0.25 over 400 draws


def test_fallback_no_square_bucket() -> None:
    buckets = tuple(b for b in BUCKETS if b.width != b.height)
    plan = _plan(1024, 512, buckets=buckets)
    assert not plan.applied
    assert plan.fallback_reason == "no_square_bucket"


def test_inclusive_endpoints_reachable() -> None:
    seen_zero = False
    seen_full = False
    available = 512
    for seed in range(200_000):
        plan = _plan(1024, 512, offset_seed=seed)
        if plan.fallback_reason == "not_selected":
            continue
        if plan.left == 0:
            seen_zero = True
        if plan.left == available:
            seen_full = True
        if seen_zero and seen_full:
            break
    assert seen_zero and seen_full


def test_determinism_and_seed_isolation() -> None:
    first = _plan(1024, 512, policy_seed=7, offset_seed=11)
    second = _plan(1024, 512, policy_seed=7, offset_seed=11)
    assert first == second
    varied_offsets = {
        _plan(1024, 512, offset_seed=seed).left for seed in range(64)
    }
    assert len(varied_offsets) > 1
    policy_varied = {
        _plan(
            1024,
            512,
            probability=0.5,
            policy_seed=seed,
            offset_seed=1,
        ).fallback_reason
        for seed in range(64)
    }
    assert policy_varied == {"none", "not_selected"}


def test_plan_and_policy_are_picklable() -> None:
    plan = _plan(1024, 512)
    assert pickle.loads(pickle.dumps(plan)) == plan
    policy = CameraViewportPolicy(True, 0.25, 1.10, 1.50)
    assert pickle.loads(pickle.dumps(policy)) == policy


def test_fallback_plan_strict_zeros() -> None:
    plan = _plan(600, 600)  # near_square_below_min
    assert not plan.applied
    assert plan.orientation == "none"
    assert plan.viewport == 0
    assert (plan.full_width, plan.full_height) == (0, 0)
    assert plan.crop_box == (0, 0, 0, 0)
    assert (plan.left, plan.top) == (0, 0)
    assert plan.equivalent_zoom == 0.0
    assert plan.retention == 0.0
    assert plan.normalized_offset == 0.0
    assert plan.signed_pixel_center_shift == 0.0
    assert plan.absolute_pixel_center_shift == 0.0
    assert plan.latent_center_shift == 0.0
    assert plan.camera_shift_x == 0.0
    assert plan.camera_shift_y == 0.0


def test_counts_strict_zero_contract() -> None:
    counts = aggregate_camera_viewport(
        (
            SimpleNamespace(
                camera_selected=False,
                camera_applied=False,
                camera_fallback_reason="not_selected",
                camera_orientation="none",
                camera_equivalent_zoom=0.0,
                camera_final_retention=0.0,
                camera_pixel_center_shift=0.0,
                camera_latent_center_shift=0.0,
            ),
        )
    )
    assert counts.applied == 0
    assert sum(counts.fallback_reasons.values()) == 1
    assert counts.camera_zoom_sum == 0.0
    assert counts.camera_retention_min == 0.0


def test_aggregate_conservation() -> None:
    audits = [
        SimpleNamespace(
            camera_selected=True,
            camera_applied=True,
            camera_fallback_reason="none",
            camera_orientation="horizontal",
            camera_equivalent_zoom=1.414,
            camera_final_retention=0.5,
            camera_pixel_center_shift=128.0,
            camera_latent_center_shift=8.0,
        ),
        SimpleNamespace(
            camera_selected=True,
            camera_applied=False,
            camera_fallback_reason="aspect_above_max",
            camera_orientation="none",
            camera_equivalent_zoom=0.0,
            camera_final_retention=0.0,
            camera_pixel_center_shift=0.0,
            camera_latent_center_shift=0.0,
        ),
        SimpleNamespace(
            camera_selected=False,
            camera_applied=False,
            camera_fallback_reason="not_selected",
            camera_orientation="none",
            camera_equivalent_zoom=0.0,
            camera_final_retention=0.0,
            camera_pixel_center_shift=0.0,
            camera_latent_center_shift=0.0,
        ),
    ]
    counts = aggregate_camera_viewport(audits)
    assert counts.selected == 2
    assert counts.applied == 1
    assert sum(counts.fallback_reasons.values()) == 3
    assert counts.fallback_reasons["none"] == 1
    assert counts.fallback_reasons["aspect_above_max"] == 1
    assert counts.fallback_reasons["not_selected"] == 1
    assert counts.orientation_counts == {"horizontal": 1, "vertical": 0}
    assert counts.camera_zoom_sum == pytest.approx(1.414)
    assert counts.camera_retention_mean == pytest.approx(0.5)
    assert counts.camera_abs_pixel_shift_mean == pytest.approx(128.0)
    assert counts.camera_abs_latent_shift_max == pytest.approx(8.0)
    assert set(counts.fallback_reasons) == set(CAMERA_FALLBACK_REASONS)
    assert set(counts.zoom_bands) == set(CAMERA_ZOOM_BAND_LABELS)
    assert set(counts.shift_token_bins) == set(CAMERA_SHIFT_TOKEN_BIN_LABELS)


def test_aggregate_rejects_unknown_reason() -> None:
    with pytest.raises(CameraViewportError):
        aggregate_camera_viewport(
            (
                SimpleNamespace(
                    camera_selected=False,
                    camera_applied=False,
                    camera_fallback_reason="bogus",
                    camera_orientation="none",
                    camera_equivalent_zoom=0.0,
                    camera_final_retention=0.0,
                    camera_pixel_center_shift=0.0,
                    camera_latent_center_shift=0.0,
                ),
            )
        )


def test_all_reasons_are_exposed() -> None:
    assert CAMERA_FALLBACK_REASONS == (
        "none",
        "not_selected",
        "short_edge_too_small",
        "near_square_below_min",
        "aspect_above_max",
        "quantized_no_effect",
        "no_square_bucket",
    )
