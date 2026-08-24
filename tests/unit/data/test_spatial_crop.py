"""Unit tests for the shifted-bucket spatial crop plan and counters.

Covers the pure-geometry policy (P6 golden bit-identical regression, P10):
determinism, the four isolated RNG domains, every fallback reason, the strict
``SpatialCropCounts`` invariants, and the fixed zoom-histogram binning.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from sakuramoon.data.buckets import BucketAssignment, BucketShape
from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
    ShiftedBucketPlan,
    SpatialCropCounts,
    SpatialCropPolicy,
    aggregate_spatial_crop,
    plan_shifted_bucket,
    zoom_histogram_bin,
)

POLICY = SpatialCropPolicy(
    enabled=True,
    probability=1.0,
    min_equivalent_zoom=1.02,
    max_equivalent_zoom=1.10,
    min_crop_retention=0.8,
)
DISABLED_POLICY = SpatialCropPolicy(
    enabled=False,
    probability=0.0,
    min_equivalent_zoom=1.02,
    max_equivalent_zoom=1.10,
    min_crop_retention=0.8,
)
PROBABILITY_25 = SpatialCropPolicy(
    enabled=True,
    probability=0.25,
    min_equivalent_zoom=1.02,
    max_equivalent_zoom=1.10,
    min_crop_retention=0.8,
)
# Tight retention guard (0.9): reachable by the plan because zoom 1.10 gives
# final retention 1/1.21 = 0.826 < 0.9. Built directly; ``from_config`` would
# reject it, which is asserted separately.
TIGHT_POLICY = SpatialCropPolicy(
    enabled=True,
    probability=1.0,
    min_equivalent_zoom=1.02,
    max_equivalent_zoom=1.10,
    min_crop_retention=0.9,
)


def make_assignment(
    source_width: int = 512,
    source_height: int = 512,
    resized_width: int = 256,
    resized_height: int = 256,
    crop_retention: float = 1.0,
    bucket_width: int = 256,
    bucket_height: int = 256,
) -> BucketAssignment:
    return BucketAssignment(
        source_width=source_width,
        source_height=source_height,
        bucket=BucketShape(height=bucket_height, width=bucket_width),
        resized_width=resized_width,
        resized_height=resized_height,
        crop_retention=crop_retention,
    )


def make_audit(
    *,
    applied: bool,
    zoom: float = 1.0,
    fallback_reason: str = "none",
    selected: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    resized: tuple[int, int] = (256, 256),
    crop_box: tuple[int, int, int, int] = (0, 0, 256, 256),
) -> SimpleNamespace:
    return SimpleNamespace(
        spatial_applied=applied,
        spatial_selected=selected,
        spatial_fallback_reason=fallback_reason,
        actual_equivalent_zoom=zoom,
        normalized_offset_x=offset_x,
        normalized_offset_y=offset_y,
        resized_width=resized[0],
        resized_height=resized[1],
        crop_box=crop_box,
    )


class TestDisabledAndSelection:
    def test_disabled_policy_never_applied(self) -> None:
        assignment = make_assignment()
        for seed in (0, 7, 99):
            plan = plan_shifted_bucket(
                assignment,
                DISABLED_POLICY,
                policy_seed=seed,
                zoom_seed=seed,
                offset_x_seed=seed,
                offset_y_seed=seed,
            )
            assert plan.applied is False
            assert plan.fallback_reason == "not_selected"
            assert plan.requested_equivalent_zoom == 0.0

    def test_not_selected_keeps_ordinary_placeholders(self) -> None:
        # random.Random(0).random() == 0.8468... >= 0.25, so sample not selected.
        assignment = make_assignment(crop_retention=0.9)
        plan = plan_shifted_bucket(
            assignment,
            PROBABILITY_25,
            policy_seed=0,
            zoom_seed=1,
            offset_x_seed=2,
            offset_y_seed=3,
        )
        assert plan.applied is False
        assert plan.fallback_reason == "not_selected"
        assert (plan.canvas_width, plan.canvas_height) == (256, 256)
        assert plan.crop_box == (0, 0, 0, 0)
        assert plan.normalized_offset_x == 0.0
        assert plan.normalized_offset_y == 0.0
        assert plan.base_crop_retention == 0.9
        assert plan.final_crop_retention == 0.9
        # The ordinary-path equivalent zoom of the unmodified sample.
        assert plan.actual_equivalent_zoom == 1.0 / math.sqrt(0.9)

    def test_selection_draws_only_from_policy_domain(self) -> None:
        # The selection draw depends solely on the policy seed: sweeping the
        # other three domains never flips the selection decision.
        assignment = make_assignment()
        reference = plan_shifted_bucket(
            assignment,
            PROBABILITY_25,
            policy_seed=0,
            zoom_seed=0,
            offset_x_seed=0,
            offset_y_seed=0,
        )
        for other_seed in (1, 2, 3, 4):
            varied = plan_shifted_bucket(
                assignment,
                PROBABILITY_25,
                policy_seed=0,
                zoom_seed=other_seed,
                offset_x_seed=other_seed,
                offset_y_seed=other_seed,
            )
            assert varied.fallback_reason == reference.fallback_reason


class TestFallbackReasons:
    def test_insufficient_source_resolution(self) -> None:
        # Source equals the bucket: maximum feasible zoom is exactly 1.0.
        assignment = make_assignment(
            source_width=256,
            source_height=256,
            resized_width=256,
            resized_height=256,
            crop_retention=1.0,
        )
        plan = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=0,
            zoom_seed=0,
            offset_x_seed=0,
            offset_y_seed=0,
        )
        assert plan.applied is False
        assert plan.fallback_reason == "insufficient_source_resolution"

    def test_base_zoom_at_or_above_max(self) -> None:
        # Resized 304x304 into a 256x256 bucket implies base zoom 1.189 >= 1.10.
        assignment = make_assignment(resized_width=304, resized_height=304)
        plan = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=0,
            zoom_seed=0,
            offset_x_seed=0,
            offset_y_seed=0,
        )
        assert plan.applied is False
        assert plan.fallback_reason == "base_zoom_at_or_above_max"

    def test_quantized_no_effect_without_gain(self) -> None:
        # A tiny requested gain quantizes away: canvas lands back on the
        # resized size, so actual zoom equals the base zoom.
        assignment = make_assignment(
            source_width=274,
            source_height=274,
            resized_width=272,
            resized_height=272,
            crop_retention=256**2 / 272**2,
            bucket_width=256,
            bucket_height=256,
        )
        low_u_seeds = [
            seed
            for seed in range(10_000)
            if random.Random(seed).random() < 0.01
        ]
        assert low_u_seeds
        plan = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=0,
            zoom_seed=low_u_seeds[0],
            offset_x_seed=0,
            offset_y_seed=0,
        )
        assert plan.applied is False
        assert plan.fallback_reason == "quantized_no_effect"

    def test_quantized_no_effect_beyond_max_zoom(self) -> None:
        # Rounding the canvas up can overshoot the max zoom; the plan must
        # fall back instead of violating the zoom bound.
        assignment = make_assignment(source_width=400, source_height=400)
        high_u_seeds = [
            seed
            for seed in range(10_000)
            if random.Random(seed).random() > 0.9895
        ]
        assert high_u_seeds
        plan = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=0,
            zoom_seed=high_u_seeds[0],
            offset_x_seed=0,
            offset_y_seed=0,
        )
        assert plan.applied is False
        assert plan.fallback_reason == "quantized_no_effect"
        assert plan.requested_equivalent_zoom <= POLICY.max_equivalent_zoom + 1e-9

    def test_retention_guard(self) -> None:
        assignment = make_assignment(source_width=400, source_height=400)
        guarded_seeds = [
            seed
            for seed in range(10_000)
            if plan_shifted_bucket(
                assignment,
                TIGHT_POLICY,
                policy_seed=0,
                zoom_seed=seed,
                offset_x_seed=0,
                offset_y_seed=0,
            ).fallback_reason
            == "retention_guard"
        ]
        assert guarded_seeds
        plan = plan_shifted_bucket(
            assignment,
            TIGHT_POLICY,
            policy_seed=0,
            zoom_seed=guarded_seeds[0],
            offset_x_seed=0,
            offset_y_seed=0,
        )
        assert plan.applied is False
        assert plan.fallback_reason == "retention_guard"

    def test_from_config_rejects_retention_incompatible_max_zoom(self) -> None:
        # 1/1.10**2 == 0.826 < 0.9, so this policy is geometrically unsafe.
        class _FakeConfig:
            enabled = True
            probability = 0.25
            min_equivalent_zoom = 1.02
            max_equivalent_zoom = 1.10

        with pytest.raises(ValueError, match="min_crop_retention"):
            SpatialCropPolicy.from_config(
                _FakeConfig(),  # type: ignore[arg-type]
                min_crop_retention=0.9,
            )


class TestAppliedPlan:
    def test_applied_invariants(self) -> None:
        assignment = make_assignment(source_width=512, source_height=512)
        plan = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=1,
            zoom_seed=2,
            offset_x_seed=3,
            offset_y_seed=4,
        )
        assert plan.applied is True
        assert plan.fallback_reason == "none"
        # Canvas stays inside the source and dominates the bucket.
        assert 256 <= plan.canvas_width <= 512
        assert 256 <= plan.canvas_height <= 512
        assert plan.canvas_width * plan.canvas_height > 256 * 256
        # Zoom and retention invariants.
        assert 1.0 < plan.actual_equivalent_zoom <= POLICY.max_equivalent_zoom
        assert (
            POLICY.min_equivalent_zoom
            <= plan.requested_equivalent_zoom
            <= POLICY.max_equivalent_zoom
        )
        assert plan.final_crop_retention >= POLICY.min_crop_retention - 1e-12
        assert plan.base_crop_retention == 1.0
        # Crop box: exact bucket size, inside the canvas.
        left, top, right, bottom = plan.crop_box
        assert right - left == 256
        assert bottom - top == 256
        assert 0 <= left <= plan.canvas_width - 256
        assert 0 <= top <= plan.canvas_height - 256
        # Normalized offsets in [-1, 1]; zero exactly when no slack.
        for value in (plan.normalized_offset_x, plan.normalized_offset_y):
            assert -1.0 <= value <= 1.0

    def test_bit_identical_replay(self) -> None:
        for seed in range(50):
            assignment = make_assignment()
            first = plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=seed,
                zoom_seed=seed + 100,
                offset_x_seed=seed + 200,
                offset_y_seed=seed + 300,
            )
            second = plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=seed,
                zoom_seed=seed + 100,
                offset_x_seed=seed + 300,
                offset_y_seed=seed + 200,
            )
            # Offset domains are independent: swapped offset seeds change the
            # offsets but never the canvas or zoom.
            if first.applied:
                assert first.canvas_width == second.canvas_width
                assert first.canvas_height == second.canvas_height
                assert first.actual_equivalent_zoom == second.actual_equivalent_zoom
            same = plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=seed,
                zoom_seed=seed + 100,
                offset_x_seed=seed + 200,
                offset_y_seed=seed + 300,
            )
            assert first == same

    def test_policy_domain_only_drives_selection(self) -> None:
        # With probability 1.0 the policy draw cannot flip the outcome.
        assignment = make_assignment()
        reference = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=0,
            zoom_seed=5,
            offset_x_seed=6,
            offset_y_seed=7,
        )
        for seed in (1, 2, 3):
            assert (
                plan_shifted_bucket(
                    assignment,
                    POLICY,
                    policy_seed=seed,
                    zoom_seed=5,
                    offset_x_seed=6,
                    offset_y_seed=7,
                )
                == reference
            )

    def test_offset_domain_independence(self) -> None:
        assignment = make_assignment(source_width=512, source_height=512)
        base_plan = plan_shifted_bucket(
            assignment,
            POLICY,
            policy_seed=0,
            zoom_seed=5,
            offset_x_seed=6,
            offset_y_seed=7,
        )
        assert base_plan.applied
        # The x offset is a pure function of the x domain: sweeping the y
        # domain never changes it, and never changes the canvas or zoom.
        for y_seed in range(20):
            y_varied = plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=0,
                zoom_seed=5,
                offset_x_seed=6,
                offset_y_seed=y_seed,
            )
            assert y_varied.canvas_width == base_plan.canvas_width
            assert y_varied.canvas_height == base_plan.canvas_height
            assert y_varied.actual_equivalent_zoom == base_plan.actual_equivalent_zoom
            assert y_varied.normalized_offset_x == base_plan.normalized_offset_x
        # Sweeping the y domain must produce at least one different y offset.
        y_values = {
            plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=0,
                zoom_seed=5,
                offset_x_seed=6,
                offset_y_seed=y_seed,
            ).normalized_offset_y
            for y_seed in range(50)
        }
        assert len(y_values) > 1

    def test_seeds_must_be_nonnegative_integers(self) -> None:
        assignment = make_assignment()
        with pytest.raises(ValueError, match="nonnegative integers"):
            plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=-1,
                zoom_seed=0,
                offset_x_seed=0,
                offset_y_seed=0,
            )
        with pytest.raises(ValueError, match="nonnegative integers"):
            plan_shifted_bucket(
                assignment,
                POLICY,
                policy_seed=0,
                zoom_seed=0,
                offset_x_seed=0,
                offset_y_seed=1.0,  # type: ignore[arg-type]
            )

    def test_explicit_source_size_mismatch_is_rejected(self) -> None:
        assignment = make_assignment(source_width=512, source_height=512)
        with pytest.raises(ValueError, match="positive integers"):
            plan_shifted_bucket(
                assignment,
                POLICY,
                source_size=(0, 512),
                policy_seed=0,
                zoom_seed=0,
                offset_x_seed=0,
                offset_y_seed=0,
            )


class TestZoomHistogramBin:
    @pytest.mark.parametrize(
        ("zoom", "expected"),
        [
            (1.0, 0),
            (1.019, 0),
            (1.02, 1),
            (1.039, 1),
            (1.04, 2),
            (1.059, 2),
            (1.06, 3),
            (1.079, 3),
            (1.08, 4),
            (1.101, 4),
        ],
    )
    def test_bin_boundaries(self, zoom: float, expected: int) -> None:
        assert zoom_histogram_bin(zoom) == expected

    def test_rejects_out_of_domain(self) -> None:
        with pytest.raises(ValueError, match="at or above 1.0"):
            zoom_histogram_bin(0.999)
        with pytest.raises(ValueError, match="finite"):
            zoom_histogram_bin(float("inf"))
        with pytest.raises(ValueError):
            zoom_histogram_bin(2)  # type: ignore[arg-type]


class TestSpatialCropCounts:
    def _valid_counts(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "selected": 1,
            "applied": 1,
            "fallback_reasons": {
                "none": 1,
                "not_selected": 0,
                "insufficient_source_resolution": 0,
                "base_zoom_at_or_above_max": 0,
                "quantized_no_effect": 0,
                "retention_guard": 0,
            },
            "zoom_histogram": {label: 0 for label in ZOOM_HISTOGRAM_LABELS},
            "actual_zoom_sum": 1.05,
            "actual_zoom_max": 1.05,
            "abs_offset_x_sum": 0.5,
            "abs_offset_y_sum": 0.25,
            "both_axes_count": 1,
        }
        base.update(overrides)
        return base

    def _build(self, kwargs: dict[str, object]) -> SpatialCropCounts:
        return SpatialCropCounts(**kwargs)  # type: ignore[call-overload]

    def test_valid_counts(self) -> None:
        kwargs = self._valid_counts()
        kwargs["zoom_histogram"]["[1.04,1.06)"] = 1
        counts = self._build(kwargs)
        assert counts.applied == 1
        assert list(counts.fallback_reasons) == list(SPATIAL_FALLBACK_REASONS)

    def test_zero_semantics_when_nothing_applied(self) -> None:
        kwargs = self._valid_counts(
            selected=0,
            applied=0,
            fallback_reasons={
                "none": 0,
                "not_selected": 1,
                "insufficient_source_resolution": 0,
                "base_zoom_at_or_above_max": 0,
                "quantized_no_effect": 0,
                "retention_guard": 0,
            },
            actual_zoom_sum=0.0,
            actual_zoom_max=0.0,
            abs_offset_x_sum=0.0,
            abs_offset_y_sum=0.0,
            both_axes_count=0,
        )
        counts = self._build(kwargs)
        assert counts.applied == 0
        assert counts.actual_zoom_max == 0.0

    def test_applied_exceeds_selected_rejected(self) -> None:
        with pytest.raises(ValueError, match="applied count exceeds"):
            self._build(self._valid_counts(applied=2, selected=1))

    def test_both_axes_exceeds_applied_rejected(self) -> None:
        with pytest.raises(ValueError, match="both-axes"):
            self._build(self._valid_counts(both_axes_count=2))

    def test_missing_fallback_key_rejected(self) -> None:
        kwargs = self._valid_counts()
        kwargs["fallback_reasons"] = {"none": 1}
        with pytest.raises(ValueError, match="fixed keys"):
            self._build(kwargs)

    def test_unknown_fallback_key_rejected(self) -> None:
        kwargs = self._valid_counts()
        reasons = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
        reasons["mystery"] = 0
        with pytest.raises(ValueError, match="fixed keys"):
            self._build({**kwargs, "fallback_reasons": reasons})

    def test_histogram_sum_must_cover_applied(self) -> None:
        # selected must stay >= applied so the earlier check (applied > selected)
        # does not fire before the histogram-coverage invariant under test.
        kwargs = self._valid_counts(applied=2, selected=2)
        with pytest.raises(ValueError, match="cover applied"):
            self._build(kwargs)

    def test_applied_requires_zoom_at_least_one(self) -> None:
        # Cover the single applied sample in the histogram (sum == applied) so
        # the zoom-max >= 1.0 invariant is the check that fires, not the
        # earlier histogram-coverage check.
        kwargs = self._valid_counts(actual_zoom_max=0.9)
        kwargs["zoom_histogram"][ZOOM_HISTOGRAM_LABELS[0]] = 1
        with pytest.raises(ValueError, match="at or above 1.0"):
            self._build(kwargs)

    def test_negative_count_rejected(self) -> None:
        kwargs = self._valid_counts(selected=-1)
        with pytest.raises(TypeError, match="nonnegative integer"):
            self._build(kwargs)

    def test_non_finite_sum_rejected(self) -> None:
        # Cover the single applied sample in the histogram so the finite-sum
        # check is the one that fires, not the earlier histogram-coverage check.
        kwargs = self._valid_counts(actual_zoom_sum=float("inf"))
        kwargs["zoom_histogram"][ZOOM_HISTOGRAM_LABELS[0]] = 1
        with pytest.raises(ValueError, match="finite float"):
            self._build(kwargs)


class TestAggregateSpatialCrop:
    def test_empty_aggregate_is_all_zero(self) -> None:
        counts = aggregate_spatial_crop(())
        assert counts.selected == 0
        assert counts.applied == 0
        assert sum(counts.fallback_reasons.values()) == 0
        assert sum(counts.zoom_histogram.values()) == 0

    def test_mixed_batch(self) -> None:
        audits = [
            make_audit(
                applied=True,
                selected=True,
                zoom=1.05,
                offset_x=0.5,
                offset_y=-0.25,
                resized=(288, 288),
                crop_box=(8, 8, 264, 264),
            ),
            make_audit(
                applied=False,
                fallback_reason="not_selected",
                zoom=1.0,
                crop_box=(0, 0, 256, 256),
            ),
            make_audit(
                applied=False,
                selected=True,
                fallback_reason="quantized_no_effect",
                zoom=1.0,
                crop_box=(0, 0, 256, 256),
            ),
        ]
        counts = aggregate_spatial_crop(audits)
        assert counts.selected == 2
        assert counts.applied == 1
        assert counts.fallback_reasons["none"] == 1
        assert counts.fallback_reasons["not_selected"] == 1
        assert counts.fallback_reasons["quantized_no_effect"] == 1
        assert sum(counts.fallback_reasons.values()) == len(audits)
        assert counts.zoom_histogram["[1.04,1.06)"] == 1
        assert counts.actual_zoom_sum == 1.05
        assert counts.actual_zoom_max == 1.05
        assert counts.abs_offset_x_sum == 0.5
        assert counts.abs_offset_y_sum == 0.25
        assert counts.both_axes_count == 1

    def test_unknown_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown spatial crop fallback reason"):
            aggregate_spatial_crop(
                [make_audit(applied=False, fallback_reason="mystery")]
            )

    def test_offset_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="in \\[-1, 1\\]"):
            aggregate_spatial_crop(
                [make_audit(applied=True, zoom=1.05, offset_x=1.5)]
            )

    def test_applied_zoom_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="at 1.0"):
            aggregate_spatial_crop([make_audit(applied=True, zoom=0.9)])

    def test_invalid_crop_box_rejected(self) -> None:
        with pytest.raises(ValueError, match="crop box is invalid"):
            aggregate_spatial_crop(
                [make_audit(applied=True, zoom=1.05, crop_box=(0, 0, 0, 256))]
            )


def test_plan_golden_bit_identical() -> None:
    """P6 golden regression: one pinned (assignment, policy, seeds) decision.

    ``GOLDEN_PLAN`` is recorded once on the training hardware (sakrua6, the
    canonical CPython runtime) and must match bit for bit on every later
    run; any mismatch means the plan's geometry or RNG behavior changed.
    """

    assignment = make_assignment(source_width=512, source_height=512)
    plan = plan_shifted_bucket(
        assignment,
        POLICY,
        policy_seed=11,
        zoom_seed=22,
        offset_x_seed=33,
        offset_y_seed=44,
    )
    assert plan == GOLDEN_PLAN


# P6 golden regression: the canonical pinned decision, recorded on sakrua6 (the
# canonical CPython runtime) from a 512x512 source assigned to the 256x256
# bucket (resized 256x256, retention 1.0) under POLICY with seeds 11/22/33/44.
# Re-record via record_golden_plan() only together with a deliberate
# policy/geometry change.
GOLDEN_PLAN = ShiftedBucketPlan(
    applied=True,
    fallback_reason="none",
    canvas_width=281,
    canvas_height=281,
    crop_box=(18, 13, 274, 269),
    requested_equivalent_zoom=1.0983105358864984,
    actual_equivalent_zoom=1.09765625,
    base_crop_retention=1.0,
    final_crop_retention=0.829979356897709,
    normalized_offset_x=0.43999999999999995,
    normalized_offset_y=0.040000000000000036,
)


def record_golden_plan() -> ShiftedBucketPlan:
    """The canonical pinned decision; print its fields to update GOLDEN_PLAN."""

    return plan_shifted_bucket(
        make_assignment(source_width=512, source_height=512),
        POLICY,
        policy_seed=11,
        zoom_seed=22,
        offset_x_seed=33,
        offset_y_seed=44,
    )

