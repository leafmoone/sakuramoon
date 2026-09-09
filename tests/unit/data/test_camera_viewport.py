"""Planner-level tests for the shifted-square camera viewport.

Covers the P100 geometry matrix at R=256 (square, near-square, 1.2:1, 2:1,
3:1, 5:1 in both orientations, the short-edge boundary at R and the
no-upscale rejection just below it), the descriptive zoom/retention
numerics, selection determinism, inclusive offset endpoints, multi-stage
resolution scaling, plan/config validation, and the batch aggregate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import pytest

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.camera_viewport import (
    CAMERA_FALLBACK_REASONS,
    CAMERA_ORIENTATION_KEYS,
    CameraViewportError,
    CameraViewportPlan,
    CameraViewportPolicy,
    aggregate_camera_viewport,
    camera_stage_edge,
    plan_camera_viewport,
)
from sakuramoon.data.pipeline import ImageAudit

R = 256
POLICY_P1 = CameraViewportPolicy(enabled=True, probability=1.0)
SEEDS = {"policy_seed": 1, "offset_seed": 2}


def _plan(source_width: int, source_height: int, **kwargs: object) -> CameraViewportPlan:
    return plan_camera_viewport(
        POLICY_P1,
        stage_edge=R,
        source_size=(source_width, source_height),
        **SEEDS,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _applied_plan(width: int, height: int) -> CameraViewportPlan:
    plan = _plan(width, height)
    assert plan.applied, f"{width}x{height} must be accepted at R={R}"
    return plan


# Explicit Callable context so the invalid-plan mutators are checked against
# CameraViewportPlan instead of being inferred as unknown-parameter lambdas.
_INVALID_PLAN_MUTATORS: tuple[
    Callable[[CameraViewportPlan], CameraViewportPlan], ...
] = (
    lambda plan: replace(plan, applied=False),
    lambda plan: replace(plan, fallback_reason="near_square_below_min"),
    lambda plan: replace(plan, orientation="diagonal"),
    lambda plan: replace(plan, left=512),  # crop escapes the canvas
    lambda plan: replace(plan, crop_box=(64, 0, 300, R)),
    lambda plan: replace(plan, equivalent_zoom=0.5),
    lambda plan: replace(plan, retention=0.0),
    lambda plan: replace(plan, equivalent_zoom=math.sqrt(2.0), retention=0.25),
)


class TestPolicyAndStageEdge:
    def test_policy_valid_range(self) -> None:
        assert CameraViewportPolicy(True, 0.0).probability == 0.0
        assert CameraViewportPolicy(False, 1.0).probability == 1.0

    @pytest.mark.parametrize(
        ("enabled", "probability"),
        [
            (1, 1.0),  # type: ignore[arg-type]
            (True, 1.5),
            (True, -0.1),
            (True, float("nan")),
            (True, float("inf")),
        ],
    )
    def test_policy_rejects_invalid_values(
        self, enabled: bool, probability: float
    ) -> None:
        with pytest.raises(CameraViewportError):
            CameraViewportPolicy(enabled, probability)

    def test_stage_edge_discovers_unique_square(self) -> None:
        buckets = (BucketShape(512, 256), BucketShape(256, 256), BucketShape(256, 512))
        assert camera_stage_edge(buckets) == 256

    @pytest.mark.parametrize("squares", [0, 2])
    def test_stage_edge_fails_fast_without_unique_square(self, squares: int) -> None:
        buckets = tuple(BucketShape(320, 448) for _ in range(3))
        if squares == 2:
            buckets = buckets + (BucketShape(256, 256), BucketShape(512, 512))
        with pytest.raises(CameraViewportError):
            camera_stage_edge(buckets)


class TestP100GeometryMatrix:
    """All legal accepts at R=256 are 256x256 views; no legacy fallbacks."""

    @pytest.mark.parametrize(
        ("width", "height", "orientation", "canvas"),
        [
            (512, 512, "square", (256, 256)),
            (2600, 2500, "horizontal", (266, 256)),  # near-square
            (1200, 1000, "horizontal", (307, 256)),  # 1.2:1
            (2000, 1000, "horizontal", (512, 256)),  # 2:1
            (3000, 1000, "horizontal", (768, 256)),  # 3:1
            (5000, 1000, "horizontal", (1280, 256)),  # 5:1
            (2500, 2600, "vertical", (256, 266)),  # near-square
            (1000, 1200, "vertical", (256, 307)),  # 1.2:1
            (1000, 2000, "vertical", (256, 512)),  # 2:1
            (1000, 3000, "vertical", (256, 768)),  # 3:1
            (1000, 5000, "vertical", (256, 1280)),  # 5:1
            (640, 256, "horizontal", (640, 256)),  # short edge == R boundary
            (256, 640, "vertical", (256, 640)),  # short edge == R boundary
        ],
    )
    def test_matrix(
        self,
        width: int,
        height: int,
        orientation: str,
        canvas: tuple[int, int],
    ) -> None:
        plan = _applied_plan(width, height)
        assert plan.orientation == orientation
        assert (plan.full_width, plan.full_height) == canvas
        assert plan.viewport == R
        assert plan.crop_box == (plan.left, plan.top, plan.left + R, plan.top + R)
        assert 0 <= plan.left <= plan.full_width - R
        assert 0 <= plan.top <= plan.full_height - R
        if orientation == "horizontal":
            assert plan.top == 0
        elif orientation == "vertical":
            assert plan.left == 0
        else:
            assert plan.left == 0 and plan.top == 0
        # Descriptive numerics: zoom = sqrt(canvas/R^2), retention = R^2/canvas.
        canvas_area = plan.full_width * plan.full_height
        assert plan.equivalent_zoom == pytest.approx(
            math.sqrt(canvas_area / (R * R)), rel=1e-12
        )
        assert plan.retention == pytest.approx((R * R) / canvas_area, rel=1e-12)
        assert plan.equivalent_zoom >= 1.0
        assert 0.0 < plan.retention <= 1.0

    def test_square_is_identity(self) -> None:
        plan = _applied_plan(512, 512)
        assert plan.orientation == "square"
        assert (plan.full_width, plan.full_height) == (R, R)
        assert (plan.left, plan.top) == (0, 0)
        assert plan.equivalent_zoom == 1.0
        assert plan.retention == 1.0

    def test_near_square_zero_shift_range_is_legal(self) -> None:
        # 2504/2500 quantizes its long edge exactly to R: no shift range,
        # offset 0, and the view is still a legal applied camera view.
        plan = _applied_plan(2504, 2500)
        assert plan.orientation == "horizontal"
        assert (plan.full_width, plan.full_height) == (R, R)
        assert (plan.left, plan.top) == (0, 0)
        assert plan.equivalent_zoom == 1.0

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (630, 255),  # short edge 255 < 256
            (255, 630),
            (255, 255),
        ],
    )
    def test_no_upscale_rejection(self, width: int, height: int) -> None:
        plan = _plan(width, height)
        assert plan.applied is False
        assert plan.fallback_reason == "no_upscale"
        assert plan.orientation == "none"
        assert (plan.full_width, plan.full_height) == (0, 0)


class TestSelection:
    def test_p1_is_always_selected(self) -> None:
        for seed in range(64):
            plan = plan_camera_viewport(
                POLICY_P1,
                stage_edge=R,
                source_size=(4000, 2000),
                policy_seed=seed,
                offset_seed=seed,
            )
            assert plan.applied, f"seed {seed} must be selected at p=1"

    def test_selection_is_deterministic_per_seed(self) -> None:
        policy = CameraViewportPolicy(True, 0.5)
        first = [
            plan_camera_viewport(
                policy,
                stage_edge=R,
                source_size=(4000, 2000),
                policy_seed=seed,
                offset_seed=seed,
            )
            for seed in range(16)
        ]
        second = [
            plan_camera_viewport(
                policy,
                stage_edge=R,
                source_size=(4000, 2000),
                policy_seed=seed,
                offset_seed=seed,
            )
            for seed in range(16)
        ]
        assert first == second

    def test_p_half_is_mixed(self) -> None:
        policy = CameraViewportPolicy(True, 0.5)
        reasons = {
            plan_camera_viewport(
                policy,
                stage_edge=R,
                source_size=(4000, 2000),
                policy_seed=seed,
                offset_seed=0,
            ).fallback_reason
            for seed in range(64)
        }
        assert reasons == {"none", "not_selected"}

    def test_inactive_policy_refuses_to_plan(self) -> None:
        for policy in (
            CameraViewportPolicy(False, 1.0),
            CameraViewportPolicy(True, 0.0),
        ):
            with pytest.raises(CameraViewportError):
                plan_camera_viewport(
                    policy,
                    stage_edge=R,
                    source_size=(4000, 2000),
                    policy_seed=1,
                    offset_seed=2,
                )


class TestOffsetsAndResolution:
    def test_offset_endpoints_are_inclusive(self) -> None:
        # 5:1 at R=256 -> canvas 1280x256, range 1024 on the long axis.
        plan = _applied_plan(5000, 1000)
        assert plan.full_width == 1280
        range_max = plan.full_width - R
        offsets = {
            plan_camera_viewport(
                POLICY_P1,
                stage_edge=R,
                source_size=(5000, 1000),
                policy_seed=0,
                offset_seed=seed,
            ).left
            for seed in range(4096)
        }
        assert min(offsets) == 0
        assert max(offsets) == range_max
        assert offsets <= set(range(range_max + 1))

    @pytest.mark.parametrize(
        ("stage_edge", "canvas"),
        [
            (256, (512, 256)),
            (512, (1024, 512)),
            (768, (1536, 768)),
            (1024, (2048, 1024)),
        ],
    )
    def test_stage_scaling_keeps_the_same_aspect(
        self, stage_edge: int, canvas: tuple[int, int]
    ) -> None:
        plan = plan_camera_viewport(
            POLICY_P1,
            stage_edge=stage_edge,
            source_size=(4000, 2000),
            policy_seed=1,
            offset_seed=2,
        )
        assert plan.applied
        assert (plan.full_width, plan.full_height) == canvas
        assert plan.crop_box[2] - plan.crop_box[0] == stage_edge
        assert plan.crop_box[3] - plan.crop_box[1] == stage_edge

    def test_long_edge_quantization_rounds_half_up(self) -> None:
        # 1600x1000 at R=256: long_q = round_half_up(409.6) = 410.
        plan = _applied_plan(1600, 1000)
        assert (plan.full_width, plan.full_height) == (410, R)


class TestValidation:
    @pytest.mark.parametrize(
        ("stage_edge", "source_size", "seeds"),
        [
            (0, (2000, 1000), (1, 2)),
            (-1, (2000, 1000), (1, 2)),
            (256, (0, 1000), (1, 2)),
            (256, (2000, -1000), (1, 2)),
            (256, (2000.0, 1000), (1, 2)),  # type: ignore[arg-type]
            (256, (2000, 1000), (-1, 2)),
            (256, (2000, 1000), (1, -2)),
        ],
    )
    def test_planner_rejects_invalid_inputs(
        self,
        stage_edge: int,
        source_size: tuple[int, int],
        seeds: tuple[int, int],
    ) -> None:
        with pytest.raises(CameraViewportError):
            plan_camera_viewport(
                POLICY_P1,
                stage_edge=stage_edge,
                source_size=source_size,
                policy_seed=seeds[0],
                offset_seed=seeds[1],
            )

    def _valid_plan(self) -> CameraViewportPlan:
        return CameraViewportPlan(
            applied=True,
            fallback_reason="none",
            orientation="horizontal",
            viewport=R,
            full_width=512,
            full_height=R,
            left=64,
            top=0,
            crop_box=(64, 0, 320, R),
            equivalent_zoom=math.sqrt(2.0),
            retention=0.5,
        )

    @pytest.mark.parametrize("mutate", _INVALID_PLAN_MUTATORS)
    def test_plan_validation_rejects_inconsistent_geometry(
        self, mutate: Callable[[CameraViewportPlan], CameraViewportPlan]
    ) -> None:
        plan = self._valid_plan()
        with pytest.raises(CameraViewportError):
            mutate(plan)

    def test_unapplied_plan_must_be_zero(self) -> None:
        CameraViewportPlan(
            applied=False,
            fallback_reason="not_selected",
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
        with pytest.raises(CameraViewportError):
            CameraViewportPlan(
                applied=False,
                fallback_reason="no_upscale",
                orientation="none",
                viewport=R,  # nonzero geometry on an unapplied plan
                full_width=0,
                full_height=0,
                left=0,
                top=0,
                crop_box=(0, 0, 0, 0),
                equivalent_zoom=0.0,
                retention=0.0,
            )


def _audit(**overrides: object) -> ImageAudit:
    base: dict[str, object] = {
        "source_width": 2000,
        "source_height": 1000,
        "resized_width": 512,
        "resized_height": 256,
        "crop_box": (64, 0, 320, 256),
        "crop_retention": 0.5,
    }
    base.update(overrides)
    return ImageAudit(**base)  # type: ignore[arg-type]


class TestAggregate:
    def test_fixed_key_counts(self) -> None:
        audits = (
            _audit(
                camera_policy="hdm_shifted_square_v2",
                camera_selected=True,
                camera_applied=True,
                camera_orientation="horizontal",
            ),
            _audit(
                camera_policy="hdm_shifted_square_v2",
                camera_selected=True,
                camera_applied=True,
                camera_orientation="vertical",
            ),
            _audit(
                camera_policy="hdm_shifted_square_v2",
                camera_selected=True,
                camera_applied=True,
                camera_orientation="square",
            ),
            _audit(
                camera_policy="hdm_shifted_square_v2",
                camera_selected=True,
                camera_fallback_reason="not_selected",
            ),
            _audit(),  # camera absent from the policy entirely
        )
        counts = aggregate_camera_viewport(audits)
        assert counts.selected == 4
        assert counts.applied == 3
        assert counts.orientation_counts == {
            "horizontal": 1,
            "vertical": 1,
            "square": 1,
        }
        assert set(counts.orientation_counts) == set(CAMERA_ORIENTATION_KEYS)

    def test_empty_batch_is_strict_zero(self) -> None:
        counts = aggregate_camera_viewport(())
        assert counts.selected == 0
        assert counts.applied == 0
        assert counts.orientation_counts == {key: 0 for key in CAMERA_ORIENTATION_KEYS}

    def test_unknown_orientation_raises(self) -> None:
        with pytest.raises(CameraViewportError):
            aggregate_camera_viewport(
                (
                    _audit(
                        camera_selected=True,
                        camera_applied=True,
                        camera_orientation="diagonal",
                    ),
                )
            )

    def test_counts_reject_inconsistent_records(self) -> None:
        with pytest.raises(CameraViewportError):
            from sakuramoon.data.camera_viewport import CameraViewportCounts

            CameraViewportCounts(
                selected=1,
                applied=2,  # applied > selected
                orientation_counts={key: 0 for key in CAMERA_ORIENTATION_KEYS},
            )
        assert CAMERA_FALLBACK_REASONS == ("none", "not_selected", "no_upscale")
