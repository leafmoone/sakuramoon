"""Schema-11 telemetry tests for the fixed camera-viewport fields.

Pins the strict-zero legacy semantics, the four-way per-band loss
partition, the conservation invariants enforced by __post_init__, and the
JSON / W&B payload shape the canary gates read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from sakuramoon.data.camera_viewport import (
    CAMERA_FALLBACK_REASONS,
    CAMERA_SHIFT_TOKEN_BIN_LABELS,
    CAMERA_ZOOM_BAND_LABELS,
)
from sakuramoon.data.caption import CONDITION_ROUTE_KEYS
from sakuramoon.data.pipeline import ImageAudit, rng_identity
from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    NOISE_T_BIN_COUNT,
    TIMING_PHASES,
    TRAINING_METRIC_SCHEMA_VERSION,
    TrainingMetric,
)


def _fallback(**overrides: int) -> dict[str, int]:
    base = {reason: 0 for reason in CAMERA_FALLBACK_REASONS}
    base.update(overrides)
    return base


def _bands(**overrides: int) -> dict[str, int]:
    base = {label: 0 for label in CAMERA_ZOOM_BAND_LABELS}
    base.update(overrides)
    return base


def _bins(**overrides: int) -> dict[str, int]:
    base = {label: 0 for label in CAMERA_SHIFT_TOKEN_BIN_LABELS}
    base.update(overrides)
    return base


def _base_metric(**camera: object) -> TrainingMetric:
    """A fully valid metric; pass camera_* overrides to exercise the fields.

    The default (legacy, camera-absent) shape: every camera count is zero
    and the ordinary band carries the whole effective batch.
    """

    condition_routes = {key: 0 for key in CONDITION_ROUTE_KEYS}
    condition_routes[next(iter(CONDITION_ROUTE_KEYS))] = 20
    kwargs: dict[str, Any] = {
        "successful_update": 1,
        "recorded_at_unix_ns": 1,
        "total_loss": 1.0,
        "high_noise_loss": 1.0,
        "low_noise_loss": 0.0,
        "high_noise_sample_count": 20,
        "low_noise_sample_count": 0,
        "t_bin_losses": (1.0,) * NOISE_T_BIN_COUNT,
        "t_bin_sample_counts": (1,) * NOISE_T_BIN_COUNT,
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
        "effective_batch": 20,
        "image_tokens": 1024,
        "text_tokens": 64,
        "dit_flops": 1,
        "samples_per_second": 1.0,
        "gpu_memory_allocated_bytes": 1,
        "gpu_memory_reserved_bytes": 1,
        "ready_queue_depth": 0,
        "ready_queue_wait_seconds": 0.0,
        "nonfinite_count": 0,
        "dropout_hits": {key: 0 for key in DROPOUT_KEYS},
        "condition_routes": condition_routes,
        "camera_ordinary_loss_sum": 20.0,
        "camera_ordinary_loss_count": 20,
    }
    kwargs.update(camera)
    return TrainingMetric(
        phase_seconds={key: 0.01 for key in TIMING_PHASES}, **kwargs
    )


def test_schema_version_is_11() -> None:
    assert TRAINING_METRIC_SCHEMA_VERSION == 11


def test_legacy_metric_is_strict_zero_outside_ordinary_band() -> None:
    metric = _base_metric()
    document = metric.as_json_mapping()
    for key in (
        "camera_viewport_selected",
        "camera_viewport_applied",
        "camera_equivalent_zoom_mean",
        "camera_equivalent_zoom_max",
        "camera_final_retention_mean",
        "camera_final_retention_min",
        "camera_abs_pixel_shift_mean",
        "camera_abs_pixel_shift_max",
        "camera_abs_latent_shift_mean",
        "camera_abs_latent_shift_max",
        "camera_mild_loss_sum",
        "camera_mild_loss_count",
        "camera_medium_loss_sum",
        "camera_medium_loss_count",
        "camera_strong_loss_sum",
        "camera_strong_loss_count",
    ):
        assert document[key] == 0, key
    assert document["camera_ordinary_loss_count"] == 20
    assert document["camera_ordinary_loss_sum"] == 20.0
    fallback = cast("Mapping[str, int]", document["camera_fallback_reasons"])
    assert fallback["none"] == 20
    for reason in CAMERA_FALLBACK_REASONS:
        if reason != "none":
            assert fallback[reason] == 0
    zoom_histogram = cast(
        "Mapping[str, int]", document["camera_zoom_histogram"]
    )
    for label in CAMERA_ZOOM_BAND_LABELS:
        assert zoom_histogram[label] == 0
    shift_histogram = cast(
        "Mapping[str, int]", document["camera_shift_token_histogram"]
    )
    for label in CAMERA_SHIFT_TOKEN_BIN_LABELS:
        assert shift_histogram[label] == 0


def test_legacy_metric_without_ordinary_partition_fails() -> None:
    with pytest.raises(ValueError):
        _base_metric(camera_ordinary_loss_count=19)
    with pytest.raises(ValueError):
        _base_metric(camera_ordinary_loss_count=21)
    with pytest.raises(ValueError):
        _base_metric(camera_mild_loss_count=1, camera_ordinary_loss_count=19)


def test_applied_metric_full_contract() -> None:
    metric = _base_metric(
        camera_viewport_selected=20,
        camera_viewport_applied=3,
        camera_fallback_reasons=_fallback(none=3, not_selected=17),
        camera_orientation_counts={"horizontal": 2, "vertical": 1},
        camera_zoom_histogram=_bands(
            **{"[1.10,1.20)": 1, "[1.20,1.35)": 1, "[1.35,1.501]": 1}
        ),
        camera_shift_token_histogram=_bins(**{"[0,1)": 1, "[1,2)": 1, "[2,4)": 1}),
        camera_equivalent_zoom_mean=1.30,
        camera_equivalent_zoom_max=1.42,
        camera_final_retention_mean=0.60,
        camera_final_retention_min=0.50,
        camera_abs_pixel_shift_mean=64.0,
        camera_abs_pixel_shift_max=119.0,
        camera_abs_latent_shift_mean=4.0,
        camera_abs_latent_shift_max=7.4375,
        camera_mild_loss_sum=0.9,
        camera_mild_loss_count=1,
        camera_medium_loss_sum=0.9,
        camera_medium_loss_count=1,
        camera_strong_loss_sum=0.9,
        camera_strong_loss_count=1,
        camera_ordinary_loss_sum=16.1,
        camera_ordinary_loss_count=17,
    )
    document = metric.as_json_mapping()
    assert document["camera_viewport_applied"] == 3
    band_counts = {
        name: cast("int", document[name])
        for name in (
            "camera_mild_loss_count",
            "camera_medium_loss_count",
            "camera_strong_loss_count",
            "camera_ordinary_loss_count",
        )
    }
    assert sum(band_counts.values()) == 20
    zoom = cast("Mapping[str, int]", document["camera_zoom_histogram"])
    assert sum(zoom.values()) == 3
    shift = cast("Mapping[str, int]", document["camera_shift_token_histogram"])
    assert sum(shift.values()) == 3


def test_applied_without_selected_fails() -> None:
    with pytest.raises(ValueError):
        _base_metric(
            camera_viewport_applied=1,
            camera_fallback_reasons=_fallback(none=1, not_selected=19),
            camera_orientation_counts={"horizontal": 1, "vertical": 0},
            camera_zoom_histogram=_bands(),
            camera_shift_token_histogram=_bins(),
            camera_mild_loss_sum=1.0,
            camera_mild_loss_count=1,
            camera_ordinary_loss_count=19,
        )


def test_zoom_out_of_band_fails() -> None:
    with pytest.raises(ValueError):
        _base_metric(
            camera_viewport_selected=1,
            camera_viewport_applied=1,
            camera_fallback_reasons=_fallback(none=1, not_selected=19),
            camera_orientation_counts={"horizontal": 1, "vertical": 0},
            camera_zoom_histogram=_bands(**{"[1.35,1.501]": 1}),
            camera_shift_token_histogram=_bins(),
            camera_equivalent_zoom_mean=1.502,
            camera_equivalent_zoom_max=1.502,
            camera_final_retention_mean=0.44,
            camera_final_retention_min=0.44,
            camera_mild_loss_sum=1.0,
            camera_mild_loss_count=1,
            camera_ordinary_loss_count=19,
        )


def test_missing_fallback_key_fails() -> None:
    reasons = _fallback(none=20)
    del reasons["not_selected"]
    with pytest.raises(ValueError):
        _base_metric(camera_fallback_reasons=reasons)


def test_applied_zero_requires_full_ordinary_batch() -> None:
    with pytest.raises(ValueError):
        _base_metric(
            camera_viewport_selected=0,
            camera_fallback_reasons=_fallback(not_selected=20),
            camera_mild_loss_count=0,
            camera_ordinary_loss_count=19,
        )


def test_wandb_mapping_exposes_flat_and_namespaced_camera_keys() -> None:
    metric = _base_metric(
        camera_viewport_selected=20,
        camera_viewport_applied=2,
        camera_fallback_reasons=_fallback(none=2, not_selected=18),
        camera_orientation_counts={"horizontal": 2, "vertical": 0},
        camera_zoom_histogram=_bands(**{"[1.10,1.20)": 2}),
        camera_shift_token_histogram=_bins(**{"[0,1)": 2}),
        camera_equivalent_zoom_mean=1.15,
        camera_equivalent_zoom_max=1.19,
        camera_final_retention_mean=0.75,
        camera_final_retention_min=0.74,
        camera_mild_loss_sum=1.5,
        camera_mild_loss_count=2,
        camera_ordinary_loss_count=18,
    )
    payload = metric.as_wandb_mapping()
    for key in (
        "camera_viewport_applied",
        "camera_mild_loss_sum",
        "camera_ordinary_loss_count",
        f"camera_zoom_histogram/{CAMERA_ZOOM_BAND_LABELS[0]}",
        f"camera_fallback_reasons/{CAMERA_FALLBACK_REASONS[0]}",
        f"camera_shift_token_histogram/{CAMERA_SHIFT_TOKEN_BIN_LABELS[0]}",
    ):
        assert key in payload, key


def _default_audit() -> ImageAudit:
    return ImageAudit(
        source_width=1000,
        source_height=1000,
        resized_width=512,
        resized_height=512,
        crop_box=(0, 0, 512, 512),
        crop_retention=0.262144,
        crop_policy="aspect_bucket",
        spatial_selected=False,
        spatial_applied=False,
        spatial_fallback_reason="none",
        base_crop_retention=0.262144,
        final_crop_retention=0.262144,
        requested_equivalent_zoom=0.0,
        actual_equivalent_zoom=0.0,
        normalized_offset_x=0.0,
        normalized_offset_y=0.0,
    )


def test_image_audit_camera_defaults_are_strict_zero() -> None:
    audit = _default_audit()
    assert audit.camera_policy == "none"
    assert audit.camera_selected is False
    assert audit.camera_applied is False
    assert audit.camera_fallback_reason == "none"
    assert audit.camera_orientation == "none"
    assert audit.camera_equivalent_zoom == 0.0
    assert audit.camera_final_retention == 0.0
    assert audit.camera_normalized_offset == 0.0
    assert audit.camera_pixel_center_shift == 0.0
    assert audit.camera_latent_center_shift == 0.0
    assert audit.camera_shift_x == 0.0
    assert audit.camera_shift_y == 0.0
    assert audit.camera_full_width == 0
    assert audit.camera_full_height == 0


def test_rng_identity_has_isolated_camera_domains() -> None:
    identity = rng_identity(base_seed=7, stage="S0", cycle_index=0, sample_id=1)
    assert identity.camera_policy_seed != 0
    assert identity.camera_offset_seed != 0
    assert identity.camera_policy_seed != identity.camera_offset_seed
    for other in (
        identity.caption_seed,
        identity.crop_seed,
        identity.spatial_policy_seed,
        identity.spatial_zoom_seed,
        identity.spatial_offset_x_seed,
        identity.spatial_offset_y_seed,
    ):
        assert identity.camera_policy_seed != other
        assert identity.camera_offset_seed != other
    again = rng_identity(base_seed=7, stage="S0", cycle_index=0, sample_id=1)
    assert identity == again
    varied = rng_identity(base_seed=7, stage="S0", cycle_index=0, sample_id=2)
    assert varied.camera_policy_seed != identity.camera_policy_seed
