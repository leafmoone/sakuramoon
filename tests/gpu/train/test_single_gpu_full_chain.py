from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.data.caption import CaptionDropoutCounts
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.pipeline import ImageAudit, RngIdentity
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.model.dit import PackedDiT
from sakuramoon.optim.adamw8bit import build_adamw8bit
from sakuramoon.train.runtime import SingleGpuBatchRuntime
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    TrainableComposite,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _composite() -> TrainableComposite:
    return TrainableComposite(
        dit=PackedDiT(
            depth=16,
            input_channels=128,
            hidden_size=2560,
            intermediate_size=6912,
            q_heads=20,
            kv_heads=5,
            head_dim=128,
            rope_nope_dim=32,
            rope_y_dim=48,
            rope_x_dim=48,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=5120,
            out_channels=128,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        ).cuda(),
        text=TextConditioner(
            input_size=2048,
            adapter_size=1024,
            output_size=2560,
            groups=8,
            attention_heads=16,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ).cuda(),
        style=StyleResampler(
            input_size=2048,
            hidden_size=1024,
            intermediate_size=2048,
            output_size=2560,
            query_count=4,
            attention_heads=16,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ).cuda(),
    )


def _batch() -> TrainingBatch:
    return TrainingBatch(
        images=torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8),
        input_ids=torch.arange(98, dtype=torch.long).unsqueeze(0),
        attention_mask=torch.ones((1, 98), dtype=torch.bool),
        main_token_indices=torch.arange(98, dtype=torch.long).unsqueeze(0),
        main_mask=torch.ones((1, 98), dtype=torch.bool),
        main_token_lengths=(98,),
        artist_token_indices=torch.empty((1, 0), dtype=torch.long),
        artist_mask=torch.empty((1, 0), dtype=torch.bool),
        active_style_sample_indices=torch.empty((0,), dtype=torch.long),
        sample_ids=torch.tensor([1], dtype=torch.long),
        target_height=256,
        target_width=256,
        dense_length=98,
        use_null_style=torch.ones((1,), dtype=torch.bool),
        all_condition_dropped=torch.zeros((1,), dtype=torch.bool),
        dropout_hits=CaptionDropoutCounts(*(0 for _ in range(12))),
        releases=("synthetic-engineering-smoke",),
        audits=(ImageAudit(256, 256, 256, 256, (0, 0, 256, 256), 1.0),),
        rng_identities=(RngIdentity(7, "S0", 0, 1, 11, 13),),
    )


def test_real_qwen_vae_dit_loss_backward_and_update() -> None:
    repository_root = Path(__file__).parents[3]
    device = torch.device("cuda", 0)
    torch.cuda.default_generators[0].manual_seed(8024)
    qwen = load_local_qwen(repository_root, device)
    vae = load_local_mage_vae(repository_root, device)
    composite = _composite()
    optimizer = build_adamw8bit(
        composite,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=8025,
    )
    runtime = SingleGpuBatchRuntime(
        qwen=qwen.encoder,
        vae=vae,
        composite=composite,
        device=device,
        generator=torch.cuda.default_generators[0],
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        noise_observation_boundary=0.95,
        growth_alpha=0.0,
    )

    measurement = runtime.measure(_batch())
    assert measurement.per_sample_loss.shape == (1,)
    assert measurement.per_sample_loss.dtype is torch.float32
    assert torch.isfinite(measurement.per_sample_loss).all()
    step = SingleGpuStep(
        composite,
        optimizer,
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )
    step.backward(measurement.per_sample_loss)
    assert all(parameter.grad is not None for parameter in composite.text.parameters())
    assert composite.style.null_tokens.grad is not None
    update = step.finish_update()
    assert update.state == SingleGpuUpdateState(1, 1, 1)
    assert torch.isfinite(update.mean_loss)
    assert all(parameter.grad is None for parameter in composite.parameters())
