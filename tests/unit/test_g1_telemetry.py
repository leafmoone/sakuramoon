from __future__ import annotations

from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    NOISE_T_BIN_COUNT,
    TIMING_PHASES,
    TRAINING_METRIC_SCHEMA_VERSION,
    TrainingMetric,
)


def test_growth_telemetry_is_in_jsonl_and_wandb_payloads() -> None:
    metric = TrainingMetric(
        successful_update=48_400,
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
    )
    json_payload = metric.as_json_mapping()
    wandb_payload = metric.as_wandb_mapping()

    assert json_payload["schema_version"] == TRAINING_METRIC_SCHEMA_VERSION == 8
    for key, value in (
        ("growth_alpha", 0.5),
        ("growth_new_slot_grad_norm", 0.75),
        ("growth_new_block_grad_norm", 0.6),
        ("growth_new_conditioner_grad_norm", 0.45),
    ):
        assert json_payload[key] == value
        assert wandb_payload[key] == value
