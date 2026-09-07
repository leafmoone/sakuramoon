"""Schema-12 telemetry tests for the fixed camera-mirror fields.

Pins the durable JSONL/W&B payload of the vertical mirror-balanced
supervision canary: the 16 fixed fields, the feature-absent default
(logical == physical == effective batch, zero activity), the update-level
conservation gates across microbatches (observer), and the degraded-pair
accounting (``selected - applied`` is the deterministic degraded count).

The population anchor (logical samples == effective batch) is asserted at
the observer level, the single production path that builds the metric;
the dataclass pins only the mirror block's internal consistency.

These are pure CPU tests; they never construct a runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sakuramoon.data.camera_viewport import (
    CAMERA_FALLBACK_REASONS,
    CAMERA_ORIENTATION_KEYS,
    CAMERA_SHIFT_TOKEN_BIN_LABELS,
    CAMERA_ZOOM_BAND_LABELS,
    CameraMirrorCounts,
)
from sakuramoon.data.caption import CONDITION_ROUTE_KEYS
from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    NOISE_T_BIN_COUNT,
    TIMING_PHASES,
    TRAINING_METRIC_SCHEMA_VERSION,
    TrainingMetric,
)
from sakuramoon.telemetry.observer import _camera_mirror_metrics

MIRROR_FIELDS = (
    "camera_mirror_logical_samples",
    "camera_mirror_vertical_applied",
    "camera_mirror_eligible",
    "camera_mirror_selected",
    "camera_mirror_applied",
    "camera_mirror_severity_lt2",
    "camera_mirror_severity_2to4",
    "camera_mirror_severity_ge4",
    "camera_mirror_original_start",
    "camera_mirror_original_center",
    "camera_mirror_original_end",
    "camera_mirror_mirror_start",
    "camera_mirror_mirror_center",
    "camera_mirror_mirror_end",
    "camera_mirror_extra_views",
    "camera_mirror_physical_views",
)

ACTIVITY_FIELDS = MIRROR_FIELDS[1:-1]


def _metric(effective_batch: int = 20, **mirror: Any) -> TrainingMetric:
    """A fully valid metric at the given effective batch.

    The camera band partition defaults to the whole batch on the ordinary
    band (feature/camera absent); callers pass camera_* overrides to
    exercise an applied camera. ``**mirror`` carries the camera_mirror_*
    fields under test.
    """

    condition_routes = {key: 0 for key in CONDITION_ROUTE_KEYS}
    condition_routes[next(iter(CONDITION_ROUTE_KEYS))] = effective_batch
    kwargs: dict[str, Any] = {
        "successful_update": 1,
        "recorded_at_unix_ns": 1,
        "total_loss": 1.0,
        "high_noise_loss": 1.0,
        "low_noise_loss": 0.0,
        "high_noise_sample_count": effective_batch,
        "low_noise_sample_count": 0,
        "t_bin_losses": (1.0,) + (0.0,) * (NOISE_T_BIN_COUNT - 1),
        "t_bin_sample_counts": (effective_batch,) + (0,) * (NOISE_T_BIN_COUNT - 1),
        "pre_clip_grad_norm": 1.0,
        "post_clip_grad_norm": 1.0,
        "condition_encoder_grad_norm": 0.25,
        "condition_global_projection_grad_norm": 0.125,
        "clip_fraction": 0.0,
        "learning_rate": 0.0001,
        "timestep_min": 0.0,
        "timestep_max": 1.0,
        "timestep_mean": 0.5,
        "timestep_std": 0.1,
        "effective_batch": effective_batch,
        "image_tokens": effective_batch * 64,
        "text_tokens": effective_batch * 4,
        "dit_flops": 1,
        "samples_per_second": 1.0,
        "gpu_memory_allocated_bytes": 1,
        "gpu_memory_reserved_bytes": 1,
        "ready_queue_depth": 0,
        "ready_queue_wait_seconds": 0.0,
        "nonfinite_count": 0,
        "dropout_hits": {key: 0 for key in DROPOUT_KEYS},
        "condition_routes": condition_routes,
        "camera_ordinary_loss_sum": 1.0,
        "camera_ordinary_loss_count": effective_batch,
        "phase_seconds": {key: 0.01 for key in TIMING_PHASES},
    }
    kwargs.update(mirror)
    return TrainingMetric(**kwargs)


def _counts(**overrides: int) -> CameraMirrorCounts:
    base: dict[str, int] = {
        "logical_samples": 4,
        "vertical_applied": 0,
        "mirror_eligible": 0,
        "mirror_selected": 0,
        "mirror_applied": 0,
        "severity_lt2": 0,
        "severity_2to4": 0,
        "severity_ge4": 0,
        "original_start": 0,
        "original_center": 0,
        "original_end": 0,
        "mirror_start": 0,
        "mirror_center": 0,
        "mirror_end": 0,
        "mirror_extra_views": 0,
        "physical_views": 4,
    }
    base.update(overrides)
    return CameraMirrorCounts(**base)


def _camera_applied_overrides(effective_batch: int = 4) -> dict[str, Any]:
    """A consistent 2-applied camera-viewport table for a 4-sample metric."""

    fallback = {reason: 0 for reason in CAMERA_FALLBACK_REASONS}
    fallback["none"] = effective_batch - 2
    fallback["not_selected"] = 2
    orientation = {key: 0 for key in CAMERA_ORIENTATION_KEYS}
    orientation["vertical"] = 2
    zoom = {label: 0 for label in CAMERA_ZOOM_BAND_LABELS}
    zoom["[1.10,1.20)"] = 2
    shift = {label: 0 for label in CAMERA_SHIFT_TOKEN_BIN_LABELS}
    shift["[2,4)"] = 2
    return {
        "camera_viewport_selected": 2,
        "camera_viewport_applied": 2,
        "camera_fallback_reasons": fallback,
        "camera_orientation_counts": orientation,
        "camera_zoom_histogram": zoom,
        "camera_shift_token_histogram": shift,
        "camera_equivalent_zoom_mean": 1.20,
        "camera_equivalent_zoom_max": 1.20,
        "camera_final_retention_mean": 0.70,
        "camera_final_retention_min": 0.60,
        "camera_abs_pixel_shift_mean": 32.0,
        "camera_abs_pixel_shift_max": 64.0,
        "camera_abs_latent_shift_mean": 3.0,
        "camera_abs_latent_shift_max": 4.0,
        "camera_mild_loss_sum": 0.5,
        "camera_mild_loss_count": 1,
        "camera_strong_loss_sum": 0.5,
        "camera_strong_loss_count": 1,
        "camera_ordinary_loss_sum": 0.5,
        "camera_ordinary_loss_count": 2,
    }


# ---------------------------------------------------------------------------
# TrainingMetric: default (feature absent)
# ---------------------------------------------------------------------------


def test_feature_absent_default_is_the_effective_batch_population() -> None:
    metric = _metric(effective_batch=4)
    document = metric.as_json_mapping()
    assert document["schema_version"] == TRAINING_METRIC_SCHEMA_VERSION == 12
    assert document["camera_mirror_logical_samples"] == 4
    assert document["camera_mirror_physical_views"] == 4
    for key in ACTIVITY_FIELDS:
        assert document[key] == 0, key


def test_json_mapping_pins_all_sixteen_fixed_fields() -> None:
    document = _metric().as_json_mapping()
    for key in MIRROR_FIELDS:
        assert key in document, key
        assert type(document[key]) is int, key


def test_wandb_mapping_publishes_all_sixteen_flat_fields() -> None:
    payload = _metric(effective_batch=4).as_wandb_mapping()
    for key in MIRROR_FIELDS:
        assert key in payload, key
        expected = 4 if key in (
            "camera_mirror_logical_samples",
            "camera_mirror_physical_views",
        ) else 0
        assert payload[key] == expected, key


# ---------------------------------------------------------------------------
# TrainingMetric: active update contract
# ---------------------------------------------------------------------------


def test_active_update_full_contract() -> None:
    metric = _metric(
        effective_batch=4,
        **_camera_applied_overrides(4),
        camera_mirror_logical_samples=4,
        camera_mirror_vertical_applied=2,
        camera_mirror_eligible=1,
        camera_mirror_selected=1,
        camera_mirror_applied=1,
        camera_mirror_severity_lt2=1,
        camera_mirror_severity_ge4=1,
        camera_mirror_original_center=1,
        camera_mirror_mirror_center=1,
        camera_mirror_extra_views=1,
        camera_mirror_physical_views=5,
    )
    document = metric.as_json_mapping()
    assert document["camera_mirror_applied"] == 1
    assert document["camera_mirror_extra_views"] == 1
    assert document["camera_mirror_physical_views"] == 5
    # No degraded pair in this update.
    assert document["camera_mirror_selected"] - document["camera_mirror_applied"] == 0


def test_degraded_update_contract() -> None:
    metric = _metric(
        effective_batch=4,
        **_camera_applied_overrides(4),
        camera_mirror_logical_samples=4,
        camera_mirror_vertical_applied=2,
        camera_mirror_eligible=1,
        camera_mirror_selected=1,
        camera_mirror_applied=0,
        camera_mirror_severity_lt2=1,
        camera_mirror_severity_ge4=1,
        camera_mirror_extra_views=0,
        camera_mirror_physical_views=4,
    )
    document = metric.as_json_mapping()
    # The eligible+selected pair failed to apply: selected-applied = 1.
    assert document["camera_mirror_selected"] - document["camera_mirror_applied"] == 1
    assert document["camera_mirror_physical_views"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("camera_mirror_logical_samples", 3),  # != effective batch (4)
        ("camera_mirror_applied", 1),  # > selected (0)
        ("camera_mirror_selected", 1),  # > eligible (0)
        ("camera_mirror_eligible", 1),  # > vertical (0)
        ("camera_mirror_vertical_applied", 1),  # severity sum (0) != vertical
        ("camera_mirror_severity_lt2", 1),  # severity sum != vertical
        ("camera_mirror_original_start", 1),  # side sum != applied
        ("camera_mirror_mirror_end", 1),  # side sum != applied
        ("camera_mirror_extra_views", 1),  # != applied
        ("camera_mirror_physical_views", 3),  # != logical (4) + extra (0)
    ),
)
def test_mirror_conservation_violations_fail_closed(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        _metric(effective_batch=4, **{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "camera_mirror_logical_samples",
        "camera_mirror_applied",
        "camera_mirror_severity_ge4",
        "camera_mirror_physical_views",
    ),
)
def test_mirror_negative_values_fail_closed(field: str) -> None:
    # Non-int / negative counts are type errors (fail closed, no coercion).
    with pytest.raises(TypeError):
        _metric(effective_batch=4, **{field: -1})


# ---------------------------------------------------------------------------
# Observer update-level aggregation (canary gates A-E)
# ---------------------------------------------------------------------------


def _observation(effective: int, *measurements: Any) -> SimpleNamespace:
    return SimpleNamespace(
        microbatches=measurements,
        loop=SimpleNamespace(update=SimpleNamespace(effective_samples=effective)),
    )


def _measurement(logical: int, counts: CameraMirrorCounts | None) -> SimpleNamespace:
    return SimpleNamespace(
        camera_mirror=counts,
        per_sample_loss=torch.zeros(logical, dtype=torch.float32),
    )


def test_gate_a_all_ordinary() -> None:
    # 4 ordinary logical samples, no mirror activity anywhere.
    aggregated = _camera_mirror_metrics(_observation(4, _measurement(4, None)))
    assert aggregated.logical_samples == 4
    assert aggregated.physical_views == 4
    assert aggregated.applied == 0
    for name in ("vertical_applied", "eligible", "selected", "extra_views"):
        assert getattr(aggregated, name) == 0, name


def test_gate_b_one_pair_plus_three_ordinary() -> None:
    # One mirror pair + 3 ordinary in a single 4-logical microbatch:
    # logical=4, applied=1, extra=1, physical=5.
    counts = _counts(
        logical_samples=4,
        vertical_applied=2,
        mirror_eligible=1,
        mirror_selected=1,
        mirror_applied=1,
        severity_lt2=1,
        severity_ge4=1,
        original_center=1,
        mirror_center=1,
        mirror_extra_views=1,
        physical_views=5,
    )
    aggregated = _camera_mirror_metrics(_observation(4, _measurement(4, counts)))
    assert aggregated.logical_samples == 4
    assert aggregated.applied == 1
    assert aggregated.extra_views == 1
    assert aggregated.physical_views == 5
    # The aggregate must load into a TrainingMetric and round-trip.
    metric = _metric(
        effective_batch=4,
        **_camera_applied_overrides(4),
        camera_mirror_logical_samples=aggregated.logical_samples,
        camera_mirror_vertical_applied=aggregated.vertical_applied,
        camera_mirror_eligible=aggregated.eligible,
        camera_mirror_selected=aggregated.selected,
        camera_mirror_applied=aggregated.applied,
        camera_mirror_severity_lt2=aggregated.severity_lt2,
        camera_mirror_severity_2to4=aggregated.severity_2to4,
        camera_mirror_severity_ge4=aggregated.severity_ge4,
        camera_mirror_original_start=aggregated.original_start,
        camera_mirror_original_center=aggregated.original_center,
        camera_mirror_original_end=aggregated.original_end,
        camera_mirror_mirror_start=aggregated.mirror_start,
        camera_mirror_mirror_center=aggregated.mirror_center,
        camera_mirror_mirror_end=aggregated.mirror_end,
        camera_mirror_extra_views=aggregated.extra_views,
        camera_mirror_physical_views=aggregated.physical_views,
    )
    document = metric.as_json_mapping()
    assert document["camera_mirror_physical_views"] == 5


def test_gate_c_selected_degraded() -> None:
    # eligible>=1, selected>=1, applied=0: physical stays 4 (no extra view)
    # and selected-applied is the degraded count.
    counts = _counts(
        logical_samples=4,
        vertical_applied=1,
        mirror_eligible=1,
        mirror_selected=1,
        mirror_applied=0,
        severity_ge4=1,
        physical_views=4,
    )
    aggregated = _camera_mirror_metrics(_observation(4, _measurement(4, counts)))
    assert aggregated.eligible == 1
    assert aggregated.selected == 1
    assert aggregated.applied == 0
    assert aggregated.physical_views == 4
    assert aggregated.selected - aggregated.applied == 1


def test_gate_d_multiple_microbatches_exact_conservation() -> None:
    # Microbatch 1: 2 ordinary (no mirror fields).
    # Microbatch 2: 2 logical, 1 pair applied (physical 3).
    counts = _counts(
        logical_samples=2,
        vertical_applied=1,
        mirror_eligible=1,
        mirror_selected=1,
        mirror_applied=1,
        severity_2to4=1,
        original_end=1,
        mirror_start=1,
        mirror_extra_views=1,
        physical_views=3,
    )
    aggregated = _camera_mirror_metrics(
        _observation(4, _measurement(2, None), _measurement(2, counts))
    )
    assert aggregated.logical_samples == 4
    assert aggregated.physical_views == 5
    assert aggregated.applied == 1
    assert aggregated.extra_views == 1
    assert aggregated.vertical_applied == 1
    assert aggregated.eligible == 1
    assert aggregated.selected == 1
    # Update-level conservation re-check.
    assert (
        aggregated.severity_lt2
        + aggregated.severity_2to4
        + aggregated.severity_ge4
        == aggregated.vertical_applied
    )
    assert (
        aggregated.original_start
        + aggregated.original_center
        + aggregated.original_end
        == aggregated.applied
    )
    assert (
        aggregated.mirror_start
        + aggregated.mirror_center
        + aggregated.mirror_end
        == aggregated.applied
    )


def test_gate_e_feature_disabled_strict_zero_activity() -> None:
    aggregated = _camera_mirror_metrics(
        _observation(8, _measurement(3, None), _measurement(5, None))
    )
    assert aggregated.logical_samples == 8
    assert aggregated.physical_views == 8
    for name in (
        "vertical_applied",
        "eligible",
        "selected",
        "applied",
        "severity_lt2",
        "severity_2to4",
        "severity_ge4",
        "original_start",
        "original_center",
        "original_end",
        "mirror_start",
        "mirror_center",
        "mirror_end",
        "extra_views",
    ):
        assert getattr(aggregated, name) == 0, name


def test_aggregate_logical_mismatch_fails_closed() -> None:
    # The population anchor lives in the observer: the summed logical
    # population must equal the effective batch, else fail closed.
    counts = _counts(
        logical_samples=4,
        vertical_applied=1,
        mirror_eligible=1,
        mirror_selected=1,
        mirror_applied=1,
        severity_ge4=1,
        original_center=1,
        mirror_center=1,
        mirror_extra_views=1,
        physical_views=5,
    )
    with pytest.raises(ValueError):
        _camera_mirror_metrics(_observation(5, _measurement(4, counts)))


def test_aggregate_rejects_untyped_counts() -> None:
    with pytest.raises(TypeError):
        _camera_mirror_metrics(
            _observation(2, _measurement(2, object()))  # type: ignore[arg-type]
        )
