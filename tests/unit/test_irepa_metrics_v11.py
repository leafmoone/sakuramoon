"""Telemetry v11: the iREPA loss split and the MAIN-JLT-only t-bin contract.

Covers the durable TrainingMetric payload (schema 11, the two new timing
phases, the six iREPA fields, and their validation) and the observer split
aggregators that feed it: ``_irepa_metric_splits`` (the strict
``main + weighted == combined`` re-derivation, one-update lambda
uniformity) and ``_timestep_bin_stats`` (the t-bin histogram must stay
MAIN-JLT-only even when iREPA is enabled).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sakuramoon.data.caption import CaptionDropoutCounts, ConditionRouteCounts
from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
    SpatialCropCounts,
)
from sakuramoon.data.transparent_white import TransparentWhiteCounts
from sakuramoon.optim.clip import ClipResult
from sakuramoon.telemetry.metrics import (
    CORE_TIMING_PHASES,
    TIMING_PHASES,
    TRAINING_METRIC_SCHEMA_VERSION,
    TrainingMetric,
)
from sakuramoon.telemetry.observer import (
    UpdateMetricContext,
    _irepa_metric_splits,
    _timestep_bin_stats,
    build_training_metric,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.loop import SuccessfulLoopObservation
from sakuramoon.train.runtime import RuntimeMeasurement, SuccessfulTrainingObservation
from sakuramoon.train.step import SingleGpuUpdateResult, SingleGpuUpdateState

BATCH = 2
T_BIN_2 = 2  # t=0.10 -> floor(0.10 * 20)
T_BIN_10 = 10  # t=0.50 -> floor(0.50 * 20)


def _dropout_zero() -> CaptionDropoutCounts:
    return CaptionDropoutCounts(
        **{
            key: 0
            for key in (
                "all_condition",
                "condition_route",
                "condition_only",
                "rating",
                "year",
                "aesthetic",
                "quality",
                "anime_completeness",
                "anime_classification",
                "nsfw",
                "character",
                "copyright",
                "general",
                "artist",
                "candidate_source",
                "long_names",
                "long_no_names",
                "short_vibes",
                "nl2",
                "nl3",
            )
        }
    )


def _spatial_zero() -> SpatialCropCounts:
    fallback = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
    fallback["not_selected"] = BATCH
    return SpatialCropCounts(
        selected=0,
        applied=0,
        fallback_reasons=fallback,
        zoom_histogram={label: 0 for label in ZOOM_HISTOGRAM_LABELS},
        actual_zoom_sum=0.0,
        actual_zoom_max=0.0,
        abs_offset_x_sum=0.0,
        abs_offset_y_sum=0.0,
        both_axes_count=0,
    )


def _measurement(
    *,
    lambda_weight: float,
    irepa: torch.Tensor | None,
) -> RuntimeMeasurement:
    main = torch.tensor([1.0, 2.0])
    if irepa is None:
        irepa_per_sample = torch.zeros(BATCH)
        weighted = torch.zeros(BATCH)
        combined = main.clone()
        cosine = torch.zeros(BATCH)
    else:
        irepa_per_sample = irepa
        weighted = lambda_weight * irepa
        combined = main + weighted
        cosine = torch.tensor([0.5, 0.6])
    return RuntimeMeasurement(
        per_sample_loss=combined,
        image_tokens=100,
        text_tokens=10,
        dit_flops=1,
        sample_ids=(),
        shape_keys=(),
        high_noise_loss_sum=torch.tensor(3.0),
        high_noise_sample_count=torch.tensor(float(BATCH)),
        low_noise_loss_sum=torch.tensor(0.0),
        low_noise_sample_count=torch.tensor(0.0),
        timesteps=torch.tensor([0.1, 0.5]),
        dropout_hits=_dropout_zero(),
        condition_routes=ConditionRouteCounts(
            artist_text=1, character_text=1, null=0
        ),
        captions=(),
        caption_plans=(),
        spatial_crop=_spatial_zero(),
        transparent=TransparentWhiteCounts(
            tagged=0,
            composited=0,
            missing_alpha=0,
            special_alpha=0,
            conflict_bg=0,
            nl_suppressed=0,
        ),
        main_per_sample_loss=main,
        irepa_per_sample_loss=irepa_per_sample,
        irepa_weighted_per_sample_loss=weighted,
        irepa_cosine=cosine,
        irepa_lambda=lambda_weight,
    )


def _observation(
    *,
    lambda_weight: float,
    irepa: torch.Tensor | None,
    projector_grad_norm: float,
    total_loss: float,
) -> tuple[SuccessfulTrainingObservation, UpdateMetricContext]:
    update = SingleGpuUpdateResult(
        mean_loss=torch.tensor(total_loss),
        clip=ClipResult(
            pre_clip_norm=torch.tensor(1.0),
            post_clip_norm=torch.tensor(1.0),
            coefficient=torch.tensor(1.0),
        ),
        condition_encoder_grad_norm=torch.tensor(0.1),
        condition_global_projection_grad_norm=torch.tensor(0.05),
        microbatches=1,
        effective_samples=BATCH,
        state=SingleGpuUpdateState(
            attempted_updates=1, successful_updates=1, effective_samples=BATCH
        ),
        growth_alpha=1.0,
        irepa_projector_grad_norm=projector_grad_norm,
    )
    loop = SuccessfulLoopObservation(
        update=update,
        checkpoint_reason=None,
        data_wait_seconds=0.0,
        checkpoint_seconds=0.0,
        update_wall_seconds=1.0,
        phase_timer=None,
    )
    observation = SuccessfulTrainingObservation(
        loop=loop,
        microbatches=(_measurement(lambda_weight=lambda_weight, irepa=irepa),),
        phase_timer=PhaseTimer(device=torch.device("cpu")),
        learning_rate=0.0001,
        gpu_memory_allocated_bytes=1,
        gpu_memory_reserved_bytes=1,
    )
    context = UpdateMetricContext(
        dit_flops=1,
        samples_per_second=1.0,
        ready_queue_depth=0,
        supplemental_phase_seconds={},
    )
    return observation, context


def test_schema_version_is_11_and_irepa_phases_are_core() -> None:
    assert TRAINING_METRIC_SCHEMA_VERSION == 11
    assert "irepa_teacher" in CORE_TIMING_PHASES
    assert "irepa_projector" in CORE_TIMING_PHASES
    assert "irepa_teacher" in TIMING_PHASES
    assert "irepa_projector" in TIMING_PHASES


def test_timestep_bins_stay_main_only_when_irepa_is_enabled() -> None:
    # the combined per-sample loss (1.05 / 2.1) differs from the main loss
    # (1.0 / 2.0); the t-bin histogram must keep the main-only values
    observation, _context = _observation(
        lambda_weight=0.5,
        irepa=torch.tensor([0.1, 0.2]),
        projector_grad_norm=0.25,
        total_loss=1.575,
    )
    losses, counts = _timestep_bin_stats(observation)
    assert counts[T_BIN_2] == 1
    assert counts[T_BIN_10] == 1
    assert sum(counts) == BATCH
    assert losses[T_BIN_2] == pytest.approx(1.0)
    assert losses[T_BIN_10] == pytest.approx(2.0)


def test_irepa_splits_disabled_are_exact_zero() -> None:
    observation, _context = _observation(
        lambda_weight=0.0,
        irepa=None,
        projector_grad_norm=0.0,
        total_loss=1.5,
    )
    main, irepa, weighted, cosine, lamda = _irepa_metric_splits(observation)
    assert main == pytest.approx(1.5)
    assert irepa == 0.0
    assert weighted == 0.0
    assert cosine == 0.0
    assert lamda == 0.0


def test_irepa_splits_enabled_rederive_the_combined_objective() -> None:
    observation, _context = _observation(
        lambda_weight=0.5,
        irepa=torch.tensor([0.1, 0.2]),
        projector_grad_norm=0.25,
        total_loss=1.575,
    )
    main, irepa, weighted, cosine, lamda = _irepa_metric_splits(observation)
    assert main == pytest.approx(1.5)
    assert irepa == pytest.approx(0.15)
    assert weighted == pytest.approx(0.075)
    assert cosine == pytest.approx(0.55)
    assert lamda == 0.5


def _single_sample_measurement(
    *,
    lambda_weight: float,
    main: float,
    irepa: float,
    timestep: float,
) -> RuntimeMeasurement:
    main_vector = torch.tensor([main])
    irepa_vector = torch.tensor([irepa])
    weighted = lambda_weight * irepa_vector
    return RuntimeMeasurement(
        per_sample_loss=main_vector + weighted,
        image_tokens=50,
        text_tokens=5,
        dit_flops=1,
        sample_ids=(),
        shape_keys=(),
        high_noise_loss_sum=torch.tensor(main),
        high_noise_sample_count=torch.tensor(1.0),
        low_noise_loss_sum=torch.tensor(0.0),
        low_noise_sample_count=torch.tensor(0.0),
        timesteps=torch.tensor([timestep]),
        dropout_hits=_dropout_zero(),
        condition_routes=ConditionRouteCounts(
            artist_text=1, character_text=0, null=0
        ),
        captions=(),
        caption_plans=(),
        spatial_crop=SpatialCropCounts(
            selected=0,
            applied=0,
            fallback_reasons={
                **{reason: 0 for reason in SPATIAL_FALLBACK_REASONS},
                "not_selected": 1,
            },
            zoom_histogram={label: 0 for label in ZOOM_HISTOGRAM_LABELS},
            actual_zoom_sum=0.0,
            actual_zoom_max=0.0,
            abs_offset_x_sum=0.0,
            abs_offset_y_sum=0.0,
            both_axes_count=0,
        ),
        transparent=TransparentWhiteCounts(
            tagged=0,
            composited=0,
            missing_alpha=0,
            special_alpha=0,
            conflict_bg=0,
            nl_suppressed=0,
        ),
        main_per_sample_loss=main_vector,
        irepa_per_sample_loss=irepa_vector,
        irepa_weighted_per_sample_loss=weighted,
        irepa_cosine=torch.tensor([0.5]),
        irepa_lambda=lambda_weight,
    )


def _two_microbatch_observation(
    lambda_a: float, lambda_b: float
) -> SuccessfulTrainingObservation:
    """One update with two microbatches (one sample each)."""

    update = SingleGpuUpdateResult(
        mean_loss=torch.tensor(1.5),
        clip=ClipResult(
            pre_clip_norm=torch.tensor(1.0),
            post_clip_norm=torch.tensor(1.0),
            coefficient=torch.tensor(1.0),
        ),
        condition_encoder_grad_norm=torch.tensor(0.1),
        condition_global_projection_grad_norm=torch.tensor(0.05),
        microbatches=2,
        effective_samples=2,
        state=SingleGpuUpdateState(
            attempted_updates=1, successful_updates=1, effective_samples=2
        ),
        growth_alpha=1.0,
        irepa_projector_grad_norm=0.0,
    )
    loop = SuccessfulLoopObservation(
        update=update,
        checkpoint_reason=None,
        data_wait_seconds=0.0,
        checkpoint_seconds=0.0,
        update_wall_seconds=1.0,
        phase_timer=None,
    )
    return SuccessfulTrainingObservation(
        loop=loop,
        microbatches=(
            _single_sample_measurement(
                lambda_weight=lambda_a, main=1.0, irepa=0.1, timestep=0.1
            ),
            _single_sample_measurement(
                lambda_weight=lambda_b, main=2.0, irepa=0.2, timestep=0.5
            ),
        ),
        phase_timer=PhaseTimer(device=torch.device("cpu")),
        learning_rate=0.0001,
        gpu_memory_allocated_bytes=1,
        gpu_memory_reserved_bytes=1,
    )


def test_irepa_splits_reject_nonuniform_lambda_within_one_update() -> None:
    observation = _two_microbatch_observation(0.5, 0.25)
    with pytest.raises(ValueError, match="irepa_lambda differs"):
        _irepa_metric_splits(observation)


def test_irepa_splits_reject_a_shape_mismatch_in_the_split() -> None:
    observation, _context = _observation(
        lambda_weight=0.5,
        irepa=torch.tensor([0.1, 0.2]),
        projector_grad_norm=0.25,
        total_loss=1.575,
    )
    measurement = observation.microbatches[0]
    bad = replace(
        measurement,
        irepa_per_sample_loss=torch.tensor([0.1]),
    )
    corrupted = SuccessfulTrainingObservation(
        loop=observation.loop,
        microbatches=(bad,),
        phase_timer=observation.phase_timer,
        learning_rate=observation.learning_rate,
        gpu_memory_allocated_bytes=1,
        gpu_memory_reserved_bytes=1,
    )
    with pytest.raises(ValueError, match="shapes are inconsistent"):
        _irepa_metric_splits(corrupted)


def _build_metric(
    *,
    lambda_weight: float,
    irepa: torch.Tensor | None,
    projector_grad_norm: float,
    total_loss: float,
) -> TrainingMetric:
    observation, context = _observation(
        lambda_weight=lambda_weight,
        irepa=irepa,
        projector_grad_norm=projector_grad_norm,
        total_loss=total_loss,
    )
    return build_training_metric(
        observation, context=context, recorded_at_unix_ns=1
    )


def test_build_metric_disabled_keeps_total_loss_and_zero_irepa() -> None:
    metric = _build_metric(
        lambda_weight=0.0,
        irepa=None,
        projector_grad_norm=0.0,
        total_loss=1.5,
    )
    assert metric.main_loss == pytest.approx(metric.total_loss)
    assert metric.irepa_loss == 0.0
    assert metric.irepa_weighted_loss == 0.0
    assert metric.irepa_cosine_mean == 0.0
    assert metric.irepa_lambda == 0.0
    assert metric.irepa_projector_grad_norm == 0.0
    assert metric.phase_seconds["irepa_teacher"] == 0.0
    assert metric.phase_seconds["irepa_projector"] == 0.0


def test_build_metric_enabled_carries_the_irepa_split() -> None:
    metric = _build_metric(
        lambda_weight=0.5,
        irepa=torch.tensor([0.1, 0.2]),
        projector_grad_norm=0.25,
        total_loss=1.575,
    )
    assert metric.main_loss == pytest.approx(1.5)
    assert metric.irepa_loss == pytest.approx(0.15)
    assert metric.irepa_weighted_loss == pytest.approx(0.075)
    assert metric.irepa_cosine_mean == pytest.approx(0.55)
    assert metric.irepa_lambda == 0.5
    assert metric.irepa_projector_grad_norm == 0.25
    assert metric.total_loss == pytest.approx(1.575)


def test_metric_v11_payload_publishes_the_irepa_fields() -> None:
    metric = _build_metric(
        lambda_weight=0.5,
        irepa=torch.tensor([0.1, 0.2]),
        projector_grad_norm=0.25,
        total_loss=1.575,
    )
    json_payload = metric.as_json_mapping()
    wandb_payload = metric.as_wandb_mapping()
    assert json_payload["schema_version"] == TRAINING_METRIC_SCHEMA_VERSION
    for key, expected in (
        ("main_loss", 1.5),
        ("irepa_loss", 0.15),
        ("irepa_weighted_loss", 0.075),
        ("irepa_cosine_mean", 0.55),
        ("irepa_lambda", 0.5),
        ("irepa_projector_grad_norm", 0.25),
    ):
        assert json_payload[key] == pytest.approx(expected)
        assert wandb_payload[key] == pytest.approx(expected)
    assert json_payload["phase_seconds"]["irepa_teacher"] == 0.0
    assert "phase_seconds/irepa_projector" in wandb_payload


def test_metric_rejects_an_impossible_cosine_mean() -> None:
    metric = _build_metric(
        lambda_weight=0.5,
        irepa=torch.tensor([0.1, 0.2]),
        projector_grad_norm=0.25,
        total_loss=1.575,
    )
    with pytest.raises(ValueError, match="irepa_cosine_mean"):
        replace(metric, irepa_cosine_mean=-1.5)


def test_metric_rejects_a_negative_irepa_loss() -> None:
    metric = _build_metric(
        lambda_weight=0.0,
        irepa=None,
        projector_grad_norm=0.0,
        total_loss=1.5,
    )
    with pytest.raises(ValueError, match="irepa_loss"):
        replace(metric, irepa_loss=-0.1)
