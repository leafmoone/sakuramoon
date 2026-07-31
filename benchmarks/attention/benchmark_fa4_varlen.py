"""Reproducible single-GPU FA4 varlen correctness and performance benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from collections.abc import Callable
from itertools import pairwise
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from sakuramoon.conditioning.packing import ValidatedCuSeqlens
from sakuramoon.model.attention import (
    AcceptedCuSeqlens,
    DenseGQAAttention,
    FA4VarlenGQAAttention,
    accept_fa4_boundaries,
    build_validated_cu_seqlens,
    dense_attention_mask,
    fa4_varlen_attention,
)


class _ProfilerTimeRange(Protocol):
    start: float
    end: float


class _ProfilerEvent(Protocol):
    name: str
    device_type: object
    time_range: _ProfilerTimeRange


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _p99_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return torch.quantile(
        (actual.float() - expected.float()).abs().flatten(), 0.99
    ).item()


def _accepted_boundaries(
    lengths: tuple[int, ...],
) -> tuple[ValidatedCuSeqlens, AcceptedCuSeqlens]:
    public = build_validated_cu_seqlens(lengths, device=torch.device("cuda"))
    accepted = accept_fa4_boundaries(
        public,
        total_tokens=sum(lengths),
        batch_size=len(lengths),
        device=public.tensor.device,
    )
    return public, accepted


def _production_fa4_module() -> FA4VarlenGQAAttention:
    return FA4VarlenGQAAttention(
        hidden_size=2560,
        q_heads=20,
        kv_heads=5,
        head_dim=128,
        rope_nope_dim=32,
        rope_y_dim=48,
        rope_x_dim=48,
        rope_position_scale=16.0,
        rope_theta=1000.0,
        norm_eps=1e-6,
        linear_dtype=torch.bfloat16,
        projection_bias=False,
        dropout=0.0,
    ).cuda()


def _production_dense_module() -> DenseGQAAttention:
    return DenseGQAAttention(
        hidden_size=2560,
        q_heads=20,
        kv_heads=5,
        head_dim=128,
        rope_nope_dim=32,
        rope_y_dim=48,
        rope_x_dim=48,
        rope_position_scale=16.0,
        rope_theta=1000.0,
        norm_eps=1e-6,
        linear_dtype=torch.bfloat16,
        projection_bias=False,
        dropout=0.0,
    ).cuda()


def _dense_reference_with_true_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start, end in pairwise(offsets):
        query_i = query[start:end].transpose(0, 1).unsqueeze(0)
        key_i = key[start:end].transpose(0, 1).unsqueeze(0)
        value_i = value[start:end].transpose(0, 1).unsqueeze(0)
        length = end - start
        mask = torch.ones(1, 1, length, length, device="cuda", dtype=torch.bool)
        outputs.append(
            F.scaled_dot_product_attention(
                query_i,
                key_i,
                value_i,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(outputs)


def _dense_per_sample_mask_free(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: tuple[int, ...],
) -> torch.Tensor:
    """Performance reference: separate samples and never pass an SDPA mask."""

    outputs: list[torch.Tensor] = []
    for start, end in pairwise(offsets):
        outputs.append(
            F.scaled_dot_product_attention(
                query[start:end].transpose(0, 1).unsqueeze(0),
                key[start:end].transpose(0, 1).unsqueeze(0),
                value[start:end].transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(outputs)


def _correctness_metrics() -> dict[str, Any]:
    lengths = (113, 197)
    offsets = (0, lengths[0], sum(lengths))
    _public_boundaries, boundaries = _accepted_boundaries(lengths)
    query_data = torch.randn(offsets[-1], 20, 128, device="cuda", dtype=torch.bfloat16)
    key_data = torch.randn(offsets[-1], 5, 128, device="cuda", dtype=torch.bfloat16)
    value_data = torch.randn(offsets[-1], 5, 128, device="cuda", dtype=torch.bfloat16)
    loss_weight = torch.randn_like(query_data, dtype=torch.float32)

    query = query_data.clone().requires_grad_()
    key = key_data.clone().requires_grad_()
    value = value_data.clone().requires_grad_()
    output = fa4_varlen_attention(query, key, value, boundaries)
    loss = (output.float() * loss_weight).mean()
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    gradients = (query.grad, key.grad, value.grad)
    assert all(gradient is not None for gradient in gradients)

    repeat_query = query_data.clone().requires_grad_()
    repeat_key = key_data.clone().requires_grad_()
    repeat_value = value_data.clone().requires_grad_()
    repeat_output = fa4_varlen_attention(
        repeat_query, repeat_key, repeat_value, boundaries
    )
    repeat_loss = (repeat_output.float() * loss_weight).mean()
    repeat_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    repeat_gradients = (repeat_query.grad, repeat_key.grad, repeat_value.grad)
    assert all(gradient is not None for gradient in repeat_gradients)

    dense_query = query_data.clone().requires_grad_()
    dense_key = key_data.clone().requires_grad_()
    dense_value = value_data.clone().requires_grad_()
    dense_output = _dense_reference_with_true_mask(
        dense_query, dense_key, dense_value, offsets
    )
    dense_loss = (dense_output.float() * loss_weight).mean()
    dense_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    dense_gradients = (dense_query.grad, dense_key.grad, dense_value.grad)
    assert all(gradient is not None for gradient in dense_gradients)

    gradient_names = ("query", "key", "value")
    gradient_bf16_floors = {"query": 1e-7, "key": 3e-6, "value": 3e-6}
    gradient_metrics: dict[str, dict[str, float]] = {}
    for name, gradient, repeat_gradient, dense_gradient in zip(
        gradient_names, gradients, repeat_gradients, dense_gradients, strict=True
    ):
        assert gradient is not None
        assert repeat_gradient is not None
        assert dense_gradient is not None
        repeat_p99 = _p99_error(gradient, repeat_gradient)
        gradient_metrics[name] = {
            "same_backend_repeat_p99": repeat_p99,
            "dense_reference_p99": _p99_error(gradient, dense_gradient),
            "derived_p99_tolerance": max(gradient_bf16_floors[name], 8.0 * repeat_p99),
        }

    repeat_output_p99 = _p99_error(output, repeat_output)
    return {
        "lengths": lengths,
        "output_same_backend_repeat_p99": repeat_output_p99,
        "output_dense_reference_p99": _p99_error(output, dense_output),
        "output_derived_p99_tolerance": max(0.002, 8.0 * repeat_output_p99),
        "loss_same_backend_repeat_abs": abs(loss.item() - repeat_loss.item()),
        "loss_dense_reference_abs": abs(loss.item() - dense_loss.item()),
        "gradients": gradient_metrics,
    }


def _full_module_correctness_metrics() -> dict[str, Any]:
    lengths = (11, 17)
    offsets = (0, lengths[0], sum(lengths))
    total_tokens = offsets[-1]
    _public_boundaries, boundaries = _accepted_boundaries(lengths)
    tokens = torch.randn(total_tokens, 2560, device="cuda", dtype=torch.bfloat16)
    coordinates = torch.randn(total_tokens, 2, device="cuda", dtype=torch.float32)
    loss_weight = torch.randn_like(tokens, dtype=torch.float32)

    padded_tokens = torch.zeros(
        len(lengths), max(lengths), 2560, device="cuda", dtype=torch.bfloat16
    )
    padded_coordinates = torch.zeros(
        len(lengths), max(lengths), 2, device="cuda", dtype=torch.float32
    )
    token_mask = torch.zeros(
        len(lengths), max(lengths), device="cuda", dtype=torch.bool
    )
    for sample_index, (start, end) in enumerate(pairwise(offsets)):
        length = end - start
        padded_tokens[sample_index, :length] = tokens[start:end]
        padded_coordinates[sample_index, :length] = coordinates[start:end]
        token_mask[sample_index, :length] = True

    fa4_module = _production_fa4_module()
    dense_module = _production_dense_module()
    dense_module.load_state_dict(fa4_module.state_dict())

    fa4_output = fa4_module(tokens, boundaries, coordinates)
    fa4_loss = (fa4_output.float() * loss_weight).mean()
    fa4_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    fa4_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in fa4_module.named_parameters()
        if parameter.grad is not None
    }

    fa4_module.zero_grad(set_to_none=True)
    repeat_output = fa4_module(tokens, boundaries, coordinates)
    repeat_loss = (repeat_output.float() * loss_weight).mean()
    repeat_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    repeat_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in fa4_module.named_parameters()
        if parameter.grad is not None
    }

    dense_output_padded = dense_module(
        padded_tokens,
        dense_attention_mask(token_mask),
        padded_coordinates,
    )
    dense_output = torch.cat(
        [row[:length] for row, length in zip(dense_output_padded, lengths, strict=True)]
    )
    dense_loss = (dense_output.float() * loss_weight).mean()
    dense_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    dense_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in dense_module.named_parameters()
        if parameter.grad is not None
    }

    gradient_metrics: dict[str, dict[str, float]] = {}
    for name, fa4_gradient in fa4_gradients.items():
        repeat_p99 = _p99_error(fa4_gradient, repeat_gradients[name])
        dense_scale_p99 = torch.quantile(
            dense_gradients[name].float().abs().flatten(), 0.99
        ).item()
        gradient_metrics[name] = {
            "same_backend_repeat_p99": repeat_p99,
            "dense_reference_p99": _p99_error(fa4_gradient, dense_gradients[name]),
            "dense_gradient_magnitude_p99": dense_scale_p99,
            "derived_p99_tolerance": max(
                5e-5,
                8.0 * repeat_p99,
                1.25 * dense_scale_p99,
            ),
        }

    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in fa4_module.named_parameters()
    }
    learning_rate = 64.0
    torch.optim.SGD(fa4_module.parameters(), lr=learning_rate).step()  # pyright: ignore[reportUnknownMemberType]
    torch.optim.SGD(dense_module.parameters(), lr=learning_rate).step()  # pyright: ignore[reportUnknownMemberType]
    update_metrics: dict[str, dict[str, float]] = {}
    for (name, fa4_parameter), (dense_name, dense_parameter) in zip(
        fa4_module.named_parameters(), dense_module.named_parameters(), strict=True
    ):
        if name != dense_name:
            raise RuntimeError("FA4 and dense modules expose different parameters")
        update_metrics[name] = {
            "fa4_update_max": (fa4_parameter.float() - parameter_before[name].float())
            .abs()
            .max()
            .item(),
            "dense_reference_p99": _p99_error(fa4_parameter, dense_parameter),
            "derived_p99_tolerance": max(
                0.002,
                80.0 * gradient_metrics[name]["dense_gradient_magnitude_p99"],
            ),
        }

    repeat_output_p99 = _p99_error(fa4_output, repeat_output)
    dense_output_p99 = _p99_error(fa4_output, dense_output)
    output_tolerance = max(0.002, 8.0 * repeat_output_p99)
    repeat_loss_abs = abs(fa4_loss.item() - repeat_loss.item())
    dense_loss_abs = abs(fa4_loss.item() - dense_loss.item())
    loss_tolerance = max(2e-6, 8.0 * repeat_loss_abs)
    return {
        "lengths": lengths,
        "parameter_names": sorted(fa4_gradients),
        "all_named_parameter_gradients_present": (
            fa4_gradients.keys()
            == repeat_gradients.keys()
            == dense_gradients.keys()
            == {name for name, _ in fa4_module.named_parameters()}
        ),
        "output_same_backend_repeat_p99": repeat_output_p99,
        "output_dense_reference_p99": dense_output_p99,
        "output_derived_p99_tolerance": output_tolerance,
        "output_comparison_passed": dense_output_p99 <= output_tolerance,
        "loss_same_backend_repeat_abs": repeat_loss_abs,
        "loss_dense_reference_abs": dense_loss_abs,
        "loss_derived_tolerance": loss_tolerance,
        "loss_comparison_passed": dense_loss_abs <= loss_tolerance,
        "gradients": gradient_metrics,
        "all_parameter_gradient_comparisons_passed": all(
            metrics["dense_reference_p99"] <= metrics["derived_p99_tolerance"]
            for metrics in gradient_metrics.values()
        ),
        "update_learning_rate": learning_rate,
        "updates": update_metrics,
        "all_parameters_updated": all(
            metrics["fa4_update_max"] > 0.0 for metrics in update_metrics.values()
        ),
        "all_parameter_update_comparisons_passed": all(
            metrics["dense_reference_p99"] <= metrics["derived_p99_tolerance"]
            for metrics in update_metrics.values()
        ),
    }


def _batched_cuda_event_ms(call: Callable[[], object], repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats  # pyright: ignore[reportUnknownMemberType]


def _synchronized_wall_ms(
    call: Callable[[], object], repeats: int
) -> list[float]:
    elapsed_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    return elapsed_ms


def _method_metrics(
    call: Callable[[], object],
    *,
    total_tokens: int,
    repeats: int,
) -> dict[str, float]:
    cuda_event_ms = _batched_cuda_event_ms(call, repeats)
    wall_ms = _synchronized_wall_ms(call, repeats)
    wall_mean_ms = sum(wall_ms) / len(wall_ms)
    return {
        "batched_cuda_event_ms_per_call": cuda_event_ms,
        "batched_cuda_event_tokens_per_second": total_tokens / (cuda_event_ms / 1000.0),
        "synchronized_wall_ms_mean": wall_mean_ms,
        "synchronized_wall_ms_p50": _percentile(wall_ms, 0.50),
        "synchronized_wall_ms_p95": _percentile(wall_ms, 0.95),
        "synchronized_wall_tokens_per_second_mean": total_tokens
        / (wall_mean_ms / 1000.0),
    }


def _wall_metrics(call: Callable[[], object], repeats: int) -> dict[str, float]:
    values = _synchronized_wall_ms(call, repeats)
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
    }


def _memory_metrics(call: Callable[[], object]) -> dict[str, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del result
    mib = float(1024**2)
    return {
        "baseline_allocated_mib": baseline_allocated / mib,
        "baseline_reserved_mib": baseline_reserved / mib,
        "peak_allocated_mib": peak_allocated / mib,
        "peak_reserved_mib": peak_reserved / mib,
        "peak_allocated_delta_mib": (peak_allocated - baseline_allocated) / mib,
        "peak_reserved_delta_mib": (peak_reserved - baseline_reserved) / mib,
    }


def _performance_metrics(warmup: int, repeats: int) -> dict[str, Any]:
    lengths = (1028, 1540)
    offsets = (0, lengths[0], sum(lengths))
    total_tokens = offsets[-1]
    public_boundaries, boundaries = _accepted_boundaries(lengths)
    query = torch.randn(total_tokens, 20, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(total_tokens, 5, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(total_tokens, 5, 128, device="cuda", dtype=torch.bfloat16)

    def fa4_call() -> torch.Tensor:
        return fa4_varlen_attention(query, key, value, boundaries)

    def dense_call() -> torch.Tensor:
        return _dense_per_sample_mask_free(query, key, value, offsets)

    block_count = 16

    def accepted_hot_multi_block_call() -> torch.Tensor:
        output = query
        for _ in range(block_count):
            output = fa4_varlen_attention(query, key, value, boundaries)
        return output

    def entry_inclusive_multi_block_call() -> torch.Tensor:
        accepted = accept_fa4_boundaries(
            public_boundaries,
            total_tokens=total_tokens,
            batch_size=len(lengths),
            device=query.device,
        )
        output = query
        for _ in range(block_count):
            output = fa4_varlen_attention(query, key, value, accepted)
        return output

    def entry_acceptance_call() -> AcceptedCuSeqlens:
        return accept_fa4_boundaries(
            public_boundaries,
            total_tokens=total_tokens,
            batch_size=len(lengths),
            device=query.device,
        )

    torch.cuda.synchronize()
    cold_started = time.perf_counter()
    fa4_call()
    torch.cuda.synchronize()
    fa4_cold_wall_ms = (time.perf_counter() - cold_started) * 1000.0
    cold_started = time.perf_counter()
    dense_call()
    torch.cuda.synchronize()
    dense_cold_wall_ms = (time.perf_counter() - cold_started) * 1000.0
    for _ in range(warmup):
        fa4_call()
        dense_call()
    torch.cuda.synchronize()

    fa4_metrics = _method_metrics(
        fa4_call,
        total_tokens=total_tokens,
        repeats=repeats,
    )
    dense_metrics = _method_metrics(
        dense_call,
        total_tokens=total_tokens,
        repeats=repeats,
    )

    fa4_memory = _memory_metrics(fa4_call)
    dense_memory = _memory_metrics(dense_call)
    multi_block_repeats = min(50, repeats)
    entry_acceptance_wall = _wall_metrics(entry_acceptance_call, multi_block_repeats)
    accepted_hot_wall = _wall_metrics(
        accepted_hot_multi_block_call,
        multi_block_repeats,
    )
    entry_inclusive_wall = _wall_metrics(
        entry_inclusive_multi_block_call,
        multi_block_repeats,
    )
    accepted_hot_cuda_event_ms = _batched_cuda_event_ms(
        accepted_hot_multi_block_call,
        multi_block_repeats,
    )

    profiled_iterations = min(5, multi_block_repeats)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        for _ in range(profiled_iterations):
            entry_inclusive_multi_block_call()
        torch.cuda.synchronize()

    profiler_events = profiler.events()
    if profiler_events is None:
        raise RuntimeError("profiler did not return CUDA events")
    cuda_events = [
        event
        for event in cast(list[_ProfilerEvent], profiler_events)
        if str(event.device_type).endswith("CUDA")
    ]
    cuda_events.sort(key=lambda event: event.time_range.start)
    copy_events = [event for event in cuda_events if "memcpy" in event.name.lower()]
    kernel_events = [
        event for event in cuda_events if "memcpy" not in event.name.lower()
    ]
    expected_kernel_events = profiled_iterations * block_count
    expected_copy_events = profiled_iterations * 2
    if len(kernel_events) != expected_kernel_events:
        raise RuntimeError("profiler did not observe one FA4 kernel per block")
    if len(copy_events) != expected_copy_events:
        raise RuntimeError("profiler did not observe one entry D2H and one entry H2D")
    gaps_us = [
        max(0.0, current.time_range.start - previous.time_range.end)
        for previous, current in pairwise(cuda_events)
    ]
    within_forward_kernel_gaps_us: list[float] = []
    for forward_index in range(profiled_iterations):
        start = forward_index * block_count
        block_events = kernel_events[start : start + block_count]
        within_forward_kernel_gaps_us.extend(
            max(0.0, current.time_range.start - previous.time_range.end)
            for previous, current in pairwise(block_events)
        )
    kernel_names = sorted({event.name for event in cuda_events})
    return {
        "lengths": lengths,
        "total_tokens": total_tokens,
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "fa4_first_call_cold_wall_ms": fa4_cold_wall_ms,
        "dense_first_call_cold_wall_ms": dense_cold_wall_ms,
        "dense_performance_reference": "per-sample SDPA with attn_mask=None",
        "full_true_mask_dense_is_permitted_for_correctness_only": True,
        "fa4": fa4_metrics,
        "dense_sdpa": dense_metrics,
        "memory": {
            "fa4": fa4_memory,
            "dense_sdpa": dense_memory,
        },
        "accepted_boundary": {
            "entry_d2h_checks_per_packed_forward": 1,
            "per_block_d2h_checks": 0,
            "block_count": block_count,
            "measured_packed_forwards": multi_block_repeats,
            "entry_acceptance_wall": entry_acceptance_wall,
            "accepted_hot_multi_block_wall": accepted_hot_wall,
            "entry_inclusive_multi_block_wall": entry_inclusive_wall,
            "entry_overhead_p50_ms": (
                entry_inclusive_wall["p50_ms"] - accepted_hot_wall["p50_ms"]
            ),
            "accepted_hot_batched_cuda_event_ms_per_forward": (
                accepted_hot_cuda_event_ms
            ),
            "accepted_hot_batched_cuda_event_ms_per_block": (
                accepted_hot_cuda_event_ms / block_count
            ),
        },
        "batched_cuda_event_speedup": (
            dense_metrics["batched_cuda_event_ms_per_call"]
            / fa4_metrics["batched_cuda_event_ms_per_call"]
        ),
        "synchronized_wall_speedup": (
            dense_metrics["synchronized_wall_ms_mean"]
            / fa4_metrics["synchronized_wall_ms_mean"]
        ),
        "profiler_iterations": profiled_iterations,
        "profiler_cuda_event_count": len(cuda_events),
        "profiler_cuda_events_per_packed_forward": (
            len(cuda_events) / profiled_iterations
        ),
        "profiler_non_copy_kernel_count": len(kernel_events),
        "profiler_non_copy_kernels_per_packed_forward": (
            len(kernel_events) / profiled_iterations
        ),
        "profiler_boundary_copy_event_count": len(copy_events),
        "profiler_boundary_copy_events_per_packed_forward": (
            len(copy_events) / profiled_iterations
        ),
        "profiler_one_d2h_one_h2d_per_packed_forward": True,
        "profiler_no_per_block_boundary_copy": True,
        "profiler_kernel_gap_us_p50": _percentile(gaps_us, 0.50),
        "profiler_kernel_gap_us_p95": _percentile(gaps_us, 0.95),
        "profiler_kernel_gap_us_max": max(gaps_us),
        "profiler_within_forward_kernel_gap_us_p50": _percentile(
            within_forward_kernel_gaps_us,
            0.50,
        ),
        "profiler_within_forward_kernel_gap_us_p95": _percentile(
            within_forward_kernel_gaps_us,
            0.95,
        ),
        "profiler_within_forward_kernel_gap_us_max": max(
            within_forward_kernel_gaps_us
        ),
        "profiler_kernel_names": kernel_names,
        "profiler_non_copy_kernel_names": sorted(
            {event.name for event in kernel_events}
        ),
        "profiler_boundary_copy_names": sorted({event.name for event in copy_events}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("warmup must be >=1 and repeats must be >=2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType]
    performance = _performance_metrics(args.warmup, args.repeats)
    report = {
        "schema_version": 2,
        "task_id": "K001",
        "benchmark_scope": "accepted_boundary_remediation",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flash_attn_4": importlib.metadata.version("flash-attn-4"),
        "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "upstream_repository_commit_provenance": "blocked_not_governed",
        "correctness": _correctness_metrics(),
        "full_module_correctness": _full_module_correctness_metrics(),
        "performance": performance,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
