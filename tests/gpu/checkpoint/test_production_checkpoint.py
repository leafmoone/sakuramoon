from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
import torch

from sakuramoon.checkpoint import (
    CheckpointCadence,
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
    load_raw_checkpoint,
    save_raw_checkpoint,
)
from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.optim.adamw8bit import build_adamw8bit
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_RESOLVED_CONFIG = b'[run]\nname = "t044-production"\n'
_CONFIG_SHA256 = hashlib.sha256(_RESOLVED_CONFIG).hexdigest()


def _production_composite() -> TrainableComposite:
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
        ),
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
        ),
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
        ),
    )


def test_full_s0_composite_raw_save_and_restore(tmp_path: Path) -> None:
    torch.cuda.empty_cache()
    module = _production_composite().cuda()
    optimizer = build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=1220,
    )
    for spec in optimizer.audit.specs:
        spec.parameter.grad = torch.zeros_like(spec.parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    identity = CheckpointIdentity(
        checkpoint_id="full-s0",
        update=1,
        config_sha256=_CONFIG_SHA256,
        dependency_sha256="b" * 64,
        parameter_schema_sha256=optimizer.audit.schema_sha256,
    )
    state = RawCheckpointState(
        trainer=SingleGpuUpdateState(1, 1, 1),
        growth=GrowthCheckpointState(BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None),
        stage_budget=StageBudgetCheckpointState(0, 1000),
        checkpoint_cadence=CheckpointCadence(1, 1_800_000_001.0),
    )
    representative = next(
        spec for spec in optimizer.audit.specs if spec.parameter.numel() <= 1024
    )
    expected = representative.parameter.detach().clone()

    save_start = time.perf_counter()
    result = save_raw_checkpoint(
        tmp_path,
        identity,
        module,
        optimizer,
        state,
        resolved_config=_RESOLVED_CONFIG,
    )
    save_seconds = time.perf_counter() - save_start
    shard_sizes = [
        path.stat().st_size for path in (result.path / "model").glob("*.safetensors")
    ]
    with torch.no_grad():
        representative.parameter.add_(1.0)
    load_start = time.perf_counter()
    loaded = load_raw_checkpoint(result.path, module, optimizer, identity)
    load_seconds = time.perf_counter() - load_start

    assert loaded == state
    torch.testing.assert_close(representative.parameter, expected, atol=0, rtol=0)
    assert len(optimizer.audit_state()) == 239
    assert all(spec.step == 1 for spec in optimizer.audit_state())
    assert result.payload_bytes > 5_000_000_000
    assert shard_sizes and max(shard_sizes) <= 2 * 1024**3
    print(
        json.dumps(
            {
                "filesystem_class": "temporary_overlay_not_formal_nvme",
                "model_shards": len(shard_sizes),
                "payload_bytes": result.payload_bytes,
                "save_seconds": save_seconds,
                "load_seconds": load_seconds,
                "max_model_shard_bytes": max(shard_sizes),
            },
            sort_keys=True,
        )
    )
