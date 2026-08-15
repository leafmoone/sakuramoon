from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    NOISE_T_BIN_COUNT,
    NOISE_T_BIN_LABELS,
    TIMING_PHASES,
    TrainingMetric,
)
from sakuramoon.telemetry.observer import _timestep_bin_stats


def test_timestep_bin_stats_uses_half_open_bins_and_includes_t_one() -> None:
    measurement = SimpleNamespace(
        per_sample_loss=torch.tensor([2.0, 4.0, 8.0, 10.0]),
        timesteps=torch.tensor([0.0, 0.90, 0.95, 1.0], dtype=torch.float32),
    )
    observation = SimpleNamespace(
        microbatches=(measurement,),
        loop=SimpleNamespace(update=SimpleNamespace(effective_samples=4)),
    )

    losses, counts = _timestep_bin_stats(observation)

    assert len(losses) == len(counts) == NOISE_T_BIN_COUNT == 20
    assert counts[0] == 1
    assert losses[0] == 2.0
    assert counts[18] == 1
    assert losses[18] == 4.0
    assert counts[19] == 2
    assert losses[19] == 9.0
    assert sum(counts) == 4


def test_training_metric_flattens_t_bin_metrics_for_wandb() -> None:
    metric = TrainingMetric(
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
    )

    payload = metric.as_wandb_mapping()

    assert payload["train_loss_by_t/bin_18_t090_095"] == 1.0
    assert payload["train_count_by_t/bin_19_t095_100"] == 1
    assert payload["condition_routes/artist_text"] == 7
    assert payload["condition_routes/character_text"] == 6
    assert payload["condition_routes/null"] == 7
    assert len(NOISE_T_BIN_LABELS) == NOISE_T_BIN_COUNT


def test_training_metric_omits_empty_t_bin_loss_from_wandb() -> None:
    metric = TrainingMetric(
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
    )
    sparse = replace(
        metric,
        effective_batch=19,
        high_noise_sample_count=19,
        condition_routes={"artist_text": 7, "character_text": 6, "null": 6},
        t_bin_losses=tuple(
            0.0 if index == 18 else 1.0 for index in range(NOISE_T_BIN_COUNT)
        ),
        t_bin_sample_counts=tuple(
            0 if index == 18 else 1 for index in range(NOISE_T_BIN_COUNT)
        ),
    )

    payload = sparse.as_wandb_mapping()

    assert "train_loss_by_t/bin_18_t090_095" not in payload
    assert payload["train_count_by_t/bin_18_t090_095"] == 0
