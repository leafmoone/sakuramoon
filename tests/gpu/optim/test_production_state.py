from __future__ import annotations

import json
import time

import pytest
import torch
from torch import nn

from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.optim.adamw8bit import build_adamw8bit
from sakuramoon.optim.clip import clip_grad_norm_fp32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _production_optimizer(module: torch.nn.Module):
    return build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=911,
    )


def _production_composite() -> nn.Module:
    composite = nn.Module()
    composite.add_module(
        "dit",
        PackedDiT(
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
    )
    composite.add_module(
        "text",
        TextConditioner(
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
    )
    composite.add_module(
        "style",
        StyleResampler(
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
    return composite


def test_bf16_text_and_style_forward_backward() -> None:
    text = TextConditioner(
        input_size=16,
        adapter_size=16,
        output_size=24,
        groups=4,
        attention_heads=4,
        norm_eps=1e-6,
        mix_gate_init=0.0,
        layer_scale_init=1.0,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    ).cuda()
    style = StyleResampler(
        input_size=16,
        hidden_size=16,
        intermediate_size=32,
        output_size=24,
        query_count=4,
        attention_heads=4,
        norm_eps=1e-6,
        init_std=0.02,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    ).cuda()
    qwen = torch.randn(2, 4, 7, 16, device="cuda", dtype=torch.bfloat16)
    text_output = text(
        qwen,
        torch.tensor([[0, 1], [2, 3]], device="cuda"),
        torch.ones(2, 2, dtype=torch.bool, device="cuda"),
    )
    style_output = style(
        qwen,
        torch.tensor([[1], [2]], device="cuda"),
        torch.ones(2, 1, dtype=torch.bool, device="cuda"),
        torch.zeros(2, dtype=torch.bool, device="cuda"),
    )
    loss = text_output.tokens.float().square().mean() + (
        style_output.tokens.float().square().mean()
    )
    loss.backward()

    assert text_output.tokens.dtype == torch.bfloat16
    assert style_output.tokens.dtype == torch.bfloat16
    assert text.shared_projection.weight.dtype == torch.bfloat16
    assert text.gate_weight.dtype == torch.float32
    assert style.cross_attention.in_proj_weight.dtype == torch.bfloat16
    assert style.queries.dtype == torch.float32
    assert text.shared_projection.weight.grad is not None
    assert style.cross_attention.in_proj_weight.grad is not None


def test_full_s0_composite_optimizer_state_audit() -> None:
    torch.cuda.empty_cache()
    module = _production_composite().cuda()
    optimizer = _production_optimizer(module)
    assert (
        optimizer.audit.schema_sha256
        == "16a0887eb1b638bb42e5780d3a759e66a82f476221fd311f1f0ff9037a7682a6"
    )
    for spec in optimizer.audit.specs:
        spec.parameter.grad = torch.zeros_like(spec.parameter)

    torch.cuda.reset_peak_memory_stats()
    clip_start = torch.cuda.Event(enable_timing=True)
    clip_end = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    clip_start.record()
    clip = clip_grad_norm_fp32(module.parameters(), max_norm=1.0)
    clip_end.record()
    optimizer.step()
    step_end.record()
    torch.cuda.synchronize()

    state = optimizer.audit_state()
    initialized = [spec for spec in state if spec.initialized]
    quantized = [spec for spec in initialized if spec.state_class == "OptimState8bit"]
    regular = [spec for spec in initialized if spec.state_class == "Tensor"]
    decay_names = [spec.name for spec in optimizer.audit.decay]
    sensitive_names = [spec.name for spec in optimizer.audit.sensitive]
    assert len(initialized) == len(optimizer.audit.specs) == 239
    assert len(quantized) == 152
    assert len(regular) == 87
    assert sum(spec.state_bytes for spec in state) == 2_568_392_844
    assert all(spec.step == 1 for spec in initialized)
    assert all(
        spec.state_class == "OptimState8bit" and spec.block_size == 256
        for spec in state
        if spec.name in set(decay_names)
    )
    assert optimizer.optimizer.param_groups[0]["param_names"] == decay_names
    assert optimizer.optimizer.param_groups[1]["param_names"] == sensitive_names
    assert clip.pre_clip_norm.item() == 0.0
    steady_repeats = 3
    steady_clip_start = torch.cuda.Event(enable_timing=True)
    steady_clip_end = torch.cuda.Event(enable_timing=True)
    steady_step_end = torch.cuda.Event(enable_timing=True)
    steady_wall_start = time.perf_counter()
    steady_clip_start.record()
    for _ in range(steady_repeats):
        clip_grad_norm_fp32(module.parameters(), max_norm=1.0)
    steady_clip_end.record()
    torch.cuda.synchronize()
    steady_clip_wall_ms = (time.perf_counter() - steady_wall_start) * 1000.0
    steady_step_wall_start = time.perf_counter()
    for _ in range(steady_repeats):
        optimizer.step()
    steady_step_end.record()
    torch.cuda.synchronize()
    steady_step_wall_ms = (time.perf_counter() - steady_step_wall_start) * 1000.0
    final_state = optimizer.audit_state()
    assert all(spec.step == 1 + steady_repeats for spec in final_state)
    report = {
        "schema_sha256": optimizer.audit.schema_sha256,
        "trainable_fqns": len(state),
        "quantized_state_fqns": len(quantized),
        "regular_state_fqns": len(regular),
        "optimizer_state_bytes": sum(spec.state_bytes for spec in state),
        "cold_clip_ms": clip_start.elapsed_time(  # pyright: ignore[reportUnknownMemberType]
            clip_end
        ),
        "cold_optimizer_step_ms": clip_end.elapsed_time(  # pyright: ignore[reportUnknownMemberType]
            step_end
        ),
        "steady_clip_cuda_event_ms": steady_clip_start.elapsed_time(  # pyright: ignore[reportUnknownMemberType]
            steady_clip_end
        )
        / steady_repeats,
        "steady_clip_wall_ms": steady_clip_wall_ms / steady_repeats,
        "steady_optimizer_cuda_event_ms": steady_clip_end.elapsed_time(  # pyright: ignore[reportUnknownMemberType]
            steady_step_end
        )
        / steady_repeats,
        "steady_optimizer_wall_ms": steady_step_wall_ms / steady_repeats,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print(json.dumps(report, sort_keys=True))
