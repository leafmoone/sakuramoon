"""Compare opt-in compiled primitives with references; never loads training assets."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from sakuramoon.config import load_config
from sakuramoon.model.fusion import (
    compiled,
    norm_modulation_reference,
    qk_rope_reference,
    residual_reference,
)


def _outputs(value: Any) -> tuple[torch.Tensor, ...]:
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def _run(function: Callable[..., Any], inputs: list[Any]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    output = _outputs(function(*inputs))
    # Nonuniform deterministic cotangents exercise rotations and shared reductions.
    loss = torch.stack([(item.float() * torch.linspace(-0.7, 0.9, item.numel(), device=item.device).reshape(item.shape)).sum() for item in output]).sum()
    parameters = [item for item in inputs if isinstance(item, torch.Tensor) and item.requires_grad]
    gradients = torch.autograd.grad(loss, parameters)
    return output, gradients


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compare(name: str, function: Callable[..., Any], inputs: list[Any], args: argparse.Namespace, device: torch.device) -> None:
    if args.explain:
        explanation = importlib.import_module("torch._dynamo").explain(function)(*inputs)
        print(f"{name}: graphs={explanation.graph_count}; graph_breaks={explanation.graph_break_count}; ops={explanation.op_count}")
        for graph in explanation.graphs:
            print(graph.code)
    reference_inputs = copy.deepcopy(inputs)
    reference, reference_grad = _run(function, reference_inputs)
    metrics = importlib.import_module("torch._inductor.metrics")
    metrics.reset()
    candidate = compiled(function)
    actual, actual_grad = _run(candidate, inputs)
    print(f"{name}: Inductor generated_kernel_count={metrics.generated_kernel_count} (compiler metric, not runtime launches)")
    for kind, expected, observed in (("forward", reference, actual), ("backward", reference_grad, actual_grad)):
        max_error = max(float((left.float() - right.float()).abs().max().detach()) for left, right in zip(expected, observed, strict=True))
        for left, right in zip(expected, observed, strict=True):
            torch.testing.assert_close(left, right, atol=args.atol, rtol=args.rtol)
        print(f"{name} {kind}: max_abs_error={max_error:.8g}")
    # Measure only after both forward and backward compilation/warmup.
    for label, operation in (("reference", function), ("inductor", candidate)):
        for _ in range(args.warmup):
            _run(operation, inputs)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            _run(operation, inputs)
        _synchronize(device)
        seconds = (time.perf_counter() - start) / args.iterations
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        forward_seconds = 0.0
        backward_seconds = 0.0
        parameters = [item for item in inputs if isinstance(item, torch.Tensor) and item.requires_grad]
        for _ in range(args.iterations):
            _synchronize(device)
            start = time.perf_counter()
            output = _outputs(operation(*inputs))
            _synchronize(device)
            forward_seconds += time.perf_counter() - start
            probes = tuple(torch.linspace(-0.7, 0.9, item.numel(), device=device, dtype=item.dtype).reshape(item.shape) for item in output)
            _synchronize(device)
            start = time.perf_counter()
            torch.autograd.grad(output, parameters, grad_outputs=probes)
            _synchronize(device)
            backward_seconds += time.perf_counter() - start
        print(f"{name} {label}: forward_backward_ms={seconds * 1000:.5f}, "
              f"forward_ms={forward_seconds * 1000 / args.iterations:.5f}, "
              f"backward_ms={backward_seconds * 1000 / args.iterations:.5f}, peak_allocated_bytes={peak}")
        if args.trace_dir:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profile:
                _run(operation, inputs)
                _synchronize(device)
            destination = args.trace_dir / f"{name}-{label}.json"
            profile.export_chrome_trace(str(destination))
            device_events = sum(str(event.device_type).endswith("CUDA") for event in cast(Any, profile).events())
            trace = json.loads(destination.read_text())
            kernel_events = sum(event.get("cat") == "kernel" for event in trace["traceEvents"])
            launches = kernel_events if device.type == "cuda" and kernel_events else None
            print(f"trace={destination}; device_events={device_events}; traced_kernel_launches={launches} (None: device kernel trace unavailable)")
            print(cast(Any, profile).key_averages().table(sort_by="self_cpu_time_total", row_limit=12))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--sample-lengths", type=int, nargs="+", default=[5, 3, 7])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--explain", action="store_true", help="print pre-Inductor FX graph; use TORCH_LOGS=output_code for generated kernels")
    parser.add_argument("--only", nargs="+", choices=("qk_rope", "packed_norm_modulation", "packed_residual", "dense_norm_modulation", "dense_residual"), help="run selected cases only; omitted cases remain unvalidated")
    args = parser.parse_args(argv)
    if min(args.sample_lengths) <= 0 or args.warmup < 1 or args.iterations < 1:
        parser.error("sample lengths, warmup and iterations must be positive")
    config = load_config(args.config, config_root=args.config_root, environment={}, validate_secrets=False).config
    dit, rope = config.model.dit, config.model.rope
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    cast(Any, torch).manual_seed(1729)
    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    def parameter(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    tokens, batch = sum(args.sample_lengths), len(args.sample_lengths)
    indices = torch.repeat_interleave(torch.arange(batch, device=device), torch.tensor(args.sample_lengths, device=device))
    coordinates = torch.randn(tokens, 2, device=device, dtype=torch.float32)
    frequencies = rope.theta ** (-torch.arange(0, rope.y_dim, 2, device=device, dtype=torch.float32) / rope.y_dim)
    print(f"torch={torch.__version__}; device={device}; CUDA={torch.version.cuda}; HIP={torch.version.hip}; dtype={dtype}")
    print(f"config dimensions: hidden={dit.hidden_size}, Q/K={dit.q_heads}/{dit.kv_heads}, head={dit.head_dim}; synthetic lengths={args.sample_lengths}")
    print("Primitive forward/backward benchmark only: NOT a full training-step or attention-backend acceptance result.")
    cases: list[tuple[str, Callable[..., Any], list[Any]]] = [
        ("qk_rope", qk_rope_reference, [random(tokens, dit.q_heads, dit.head_dim), random(tokens, dit.kv_heads, dit.head_dim), parameter(dit.head_dim), parameter(dit.head_dim), coordinates, frequencies, dit.norm_eps, rope.nope_dim, rope.y_dim, rope.x_dim, rope.position_scale]),
        ("packed_norm_modulation", norm_modulation_reference, [random(tokens, dit.hidden_size), parameter(dit.hidden_size), parameter(batch, dit.hidden_size), parameter(batch, dit.hidden_size), dit.norm_eps, indices]),
        ("packed_residual", residual_reference, [random(tokens, dit.hidden_size), random(tokens, dit.hidden_size), parameter(batch, dit.hidden_size), 0.7, indices]),
        ("dense_norm_modulation", norm_modulation_reference, [random(batch, max(args.sample_lengths), dit.hidden_size), parameter(dit.hidden_size), parameter(batch, dit.hidden_size), parameter(batch, dit.hidden_size), dit.norm_eps]),
        ("dense_residual", residual_reference, [random(batch, max(args.sample_lengths), dit.hidden_size), random(batch, max(args.sample_lengths), dit.hidden_size), parameter(batch, dit.hidden_size), 0.7]),
    ]
    for name, function, inputs in cases:
        if args.only and name not in args.only:
            print(f"{name}: not selected, unvalidated")
            continue
        _compare(name, function, inputs, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
