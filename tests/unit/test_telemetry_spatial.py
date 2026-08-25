"""P9 schema-9 telemetry tests for the fixed spatial-crop fields.

Covers the strict zero semantics of ``TrainingMetric``, the fallback-sum and
zoom-histogram invariants enforced by ``__post_init__``, and the JSON/W&B
payload shape the canary gates read (including the flattened spatial keys).
"""

from __future__ import annotations

import pytest

from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
)
from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    NOISE_T_BIN_COUNT,
    TIMING_PHASES,
    TRAINING_METRIC_SCHEMA_VERSION,
    TrainingMetric,
)


def _fallback(**overrides: int) -> dict[str, int]:
    base = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
    base.update(overrides)
    return base


def _histogram(**overrides: int) -> dict[str, int]:
    base = {label: 0 for label in ZOOM_HISTOGRAM_LABELS}
    base.update(overrides)
    return base


def _base_metric(**spatial: object) -> TrainingMetric:
    """A fully valid metric; pass spatial_* overrides to exercise the fields."""

    return TrainingMetric(
        successful_update=1,
        recorded_at_unix_ns=1,
        total_loss=1.0,
        high_noise_loss=1.0,
        low_noise_loss=0.0,
        high_noise_sample_count=20,
        low_noise_sample_count=0,
        t_bin_losses=(1.0,) * NOISE_T_BIN_COUNT,
        t_bin_sample_counts=(1,) * NOISE_T_BIN_COUNT,
        pre_clip_grad_norm=1.0,
        post_clip_grad_norm=1.0,
        condition_encoder_grad_norm=0.25,
        condition_global_projection_grad_norm=0.125,
        clip_fraction=0.0,
        learning_rate=0.0001,
        timestep_min=0.0,
        timestep_max=1.0,
        timestep_mean=0.5,
        timestep_std=0.1,
        effective_batch=20,
        image_tokens=1,
        text_tokens=1,
        dit_flops=1,
        samples_per_second=1.0,
        gpu_memory_allocated_bytes=1,
        gpu_memory_reserved_bytes=1,
        ready_queue_depth=0,
        ready_queue_wait_seconds=0.0,
        nonfinite_count=0,
        dropout_hits={key: 0 for key in DROPOUT_KEYS},
        condition_routes={"artist_text": 7, "character_text": 6, "null": 7},
        phase_seconds={phase: 0.0 for phase in TIMING_PHASES},
        growth_alpha=0.5,
        growth_new_slot_grad_norm=0.75,
        growth_new_block_grad_norm=0.6,
        growth_new_conditioner_grad_norm=0.45,
        **spatial,  # type: ignore[arg-type]
    )


def _applied_metric() -> TrainingMetric:
    # Fallback reasons partition every one of the 20 effective samples.
    fallback = _fallback(
        none=4,
        not_selected=15,
        retention_guard=1,
    )
    assert sum(fallback.values()) == 20
    # Zoom histogram covers the 4 applied samples in one bin.
    histogram = _histogram(**{"[1.02,1.04)": 4})
    assert sum(histogram.values()) == 4
    return _base_metric(
        spatial_crop_selected=5,
        spatial_crop_applied=4,
        spatial_fallback_reasons=fallback,
        spatial_zoom_histogram=histogram,
        spatial_actual_zoom_mean=1.03,
        spatial_actual_zoom_max=1.05,
        spatial_abs_offset_x_mean=0.4,
        spatial_abs_offset_y_mean=0.3,
        spatial_both_axes_count=2,
    )


class TestSchema:
    def test_schema_version_is_ten(self) -> None:
        assert TRAINING_METRIC_SCHEMA_VERSION == 10

    def test_json_payload_exposes_nested_spatial_tables(self) -> None:
        payload = _applied_metric().as_json_mapping()
        assert payload["schema_version"] == 10
        assert payload["spatial_crop_selected"] == 5
        assert payload["spatial_crop_applied"] == 4
        assert payload["spatial_both_axes_count"] == 2
        assert payload["spatial_actual_zoom_mean"] == 1.03
        assert payload["spatial_actual_zoom_max"] == 1.05
        assert payload["spatial_abs_offset_x_mean"] == 0.4
        assert payload["spatial_abs_offset_y_mean"] == 0.3
        assert payload["spatial_fallback_reasons"]["not_selected"] == 15
        assert payload["spatial_fallback_reasons"]["retention_guard"] == 1
        assert payload["spatial_zoom_histogram"]["[1.02,1.04)"] == 4


class TestWandbFlattening:
    def test_flattened_spatial_keys(self) -> None:
        payload = _applied_metric().as_wandb_mapping()
        for reason in SPATIAL_FALLBACK_REASONS:
            assert f"spatial_fallback_reasons/{reason}" in payload
        for label in ZOOM_HISTOGRAM_LABELS:
            assert f"spatial_zoom_histogram/{label}" in payload
        assert payload["spatial_fallback_reasons/not_selected"] == 15
        assert payload["spatial_fallback_reasons/retention_guard"] == 1
        assert payload["spatial_fallback_reasons/none"] == 4
        assert payload["spatial_zoom_histogram/[1.02,1.04)"] == 4
        # Scalar spatial fields publish alongside the flattened tables.
        assert payload["spatial_crop_selected"] == 5
        assert payload["spatial_crop_applied"] == 4


class TestStrictZeroSemantics:
    def test_disabled_run_metric_is_all_zero(self) -> None:
        metric = _base_metric()
        # Defaults: nothing selected or applied, every sample not_selected.
        assert metric.spatial_crop_selected == 0
        assert metric.spatial_crop_applied == 0
        assert metric.spatial_both_axes_count == 0
        assert metric.spatial_actual_zoom_mean == 0.0
        assert metric.spatial_actual_zoom_max == 0.0
        assert metric.spatial_abs_offset_x_mean == 0.0
        assert metric.spatial_abs_offset_y_mean == 0.0
        assert metric.spatial_fallback_reasons["not_selected"] == 20
        assert metric.spatial_zoom_histogram == {
            label: 0 for label in ZOOM_HISTOGRAM_LABELS
        }

    def test_applied_forbids_nonzero_aggregate_when_zero(self) -> None:
        # applied=0 but a zoom aggregate is nonzero: invariant violation.
        with pytest.raises(ValueError, match="zero when nothing was applied"):
            _base_metric(spatial_actual_zoom_mean=1.03)

    def test_applied_rejects_zoom_mean_below_one(self) -> None:
        with pytest.raises(ValueError, match="at least 1.0"):
            _applied_with(zoom_mean=0.9)

    def test_offset_mean_must_not_exceed_one(self) -> None:
        with pytest.raises(ValueError, match="offset means must not exceed one"):
            _applied_with(offset_x=1.5)


class TestInvariants:
    def _metric(self, **overrides: object) -> TrainingMetric:
        return _base_metric(**overrides)

    def test_applied_cannot_exceed_selected(self) -> None:
        with pytest.raises(ValueError, match="applied count exceeds"):
            self._metric(spatial_crop_selected=1, spatial_crop_applied=2)

    def test_both_axes_cannot_exceed_applied(self) -> None:
        with pytest.raises(ValueError, match="both-axes"):
            self._metric(spatial_both_axes_count=5)

    def test_fallback_sum_must_equal_effective_batch(self) -> None:
        with pytest.raises(ValueError, match="equal effective batch"):
            self._metric(spatial_fallback_reasons=_fallback(none=1))

    def test_zoom_histogram_must_cover_applied(self) -> None:
        # No samples applied, but the histogram claims one.
        with pytest.raises(ValueError, match="cover applied"):
            self._metric(spatial_zoom_histogram=_histogram(**{"[1.02,1.04)": 1}))

    def test_missing_fallback_key_rejected(self) -> None:
        fallback = _fallback(none=20)
        del fallback["not_selected"]
        with pytest.raises(ValueError, match="must contain every fixed key"):
            self._metric(spatial_fallback_reasons=fallback)


def _applied_with(
    *,
    zoom_mean: float = 1.03,
    offset_x: float = 0.4,
) -> TrainingMetric:
    fallback = _fallback(none=1, not_selected=19)
    histogram = _histogram(**{"[1.02,1.04)": 1})
    return _base_metric(
        spatial_crop_selected=1,
        spatial_crop_applied=1,
        spatial_fallback_reasons=fallback,
        spatial_zoom_histogram=histogram,
        spatial_actual_zoom_mean=zoom_mean,
        spatial_actual_zoom_max=zoom_mean,
        spatial_abs_offset_x_mean=offset_x,
        spatial_abs_offset_y_mean=0.0,
    )
