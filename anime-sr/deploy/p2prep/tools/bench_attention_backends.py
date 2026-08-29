"""P2-prep: throughput benchmark, WindowAttention (explicit core) vs
WindowAttentionSDPA (fused SDPA core).

Measures on the local accelerator (HCU on the remote DTK box, CUDA locally):
  * full-step (fwd + bwd) throughput, it/s and peak memory
  * per-phase breakdown: pre (proj + qk norms + RoPE) / core / output
    projection, each timed over its own loop
  * the GQA A/B: repeat_interleave (default) vs native q/kv head counts

Usage (remote DTK env, after the branch exists):

    source /opt/dtk-26.04/env.sh && export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
    /usr/local/bin/python3.11 tools/bench_attention_backends.py \
        --H 128 --W 128 --dim 384 --heads 6 --kv 3 --dtype bf16 \
        --iters 60 --out bench-sdpa.json
    # watch utilization in parallel:  hy-smi

Prints a JSON summary to stdout (and ``--out`` when given). HCU utilization
(``hy-smi``) is captured externally: keep a log of the run's wall-clock
window and attach it to the summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from anime_sr.model.window_attention import WindowAttention
from anime_sr.model.window_attention_sdpa import WindowAttentionSDPA, sdpa_variant


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_loop(fn, iters: int, warmup: int, device: torch.device) -> float:
    """Milliseconds per call of ``fn`` over a warmup+timed loop."""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1e3 / iters


def _front_half(mod: WindowAttention, x: torch.Tensor, H: int, W: int):
    """Shared front half (proj + qk RMSNorm + 2D RoPE) for both cores."""
    B, N, _ = x.shape
    q = mod.q_proj(x).view(B, N, mod.num_heads, mod.head_dim).transpose(1, 2)
    k = mod.k_proj(x).view(B, N, mod.num_kv_heads, mod.head_dim).transpose(1, 2)
    v = mod.v_proj(x).view(B, N, mod.num_kv_heads, mod.head_dim).transpose(1, 2)
    q, k = mod.q_norm(q), mod.k_norm(k)
    coords = mod._coords(H, W, (0, 0), x.device)
    q, k = mod.rope.apply(q, k, coords)
    return q, k, v


def _core(mod: WindowAttention, q, k, v, H: int, W: int, shift: bool):
    if isinstance(mod, WindowAttentionSDPA):
        if mod.global_attention:
            if not mod.gqa_native:
                k = k.repeat_interleave(mod.gqa_rep, dim=1)
                v = v.repeat_interleave(mod.gqa_rep, dim=1)
            return F.scaled_dot_product_attention(q, k, v)
        return mod._windowed_sdpa(q, k, v, H, W, shift)
    return mod._windowed_attention(q, k, v, H, W, shift)


def bench_step(mod: WindowAttention, x: torch.Tensor, H: int, W: int, shift: bool,
               iters: int, warmup: int, device: torch.device) -> dict:
    def step():
        y = mod(x, H, W, shift=shift)
        y.sum().backward()

    ms = _time_loop(step, iters, warmup, device)
    peak = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        step()
        peak = torch.cuda.max_memory_allocated(device) / 1e6
    return {"ms_per_step": ms, "it_per_s": 1000.0 / ms if ms > 0 else float("inf"), "peak_mb": peak}


def bench_phases(mod: WindowAttention, x: torch.Tensor, H: int, W: int, shift: bool,
                 iters: int, warmup: int, device: torch.device) -> dict:
    q, k, v = _front_half(mod, x, H, W)
    core_out = _core(mod, q, k, v, H, W, shift)

    def pre():
        _front_half(mod, x, H, W)

    def core():
        _core(mod, q, k, v, H, W, shift)

    def out():
        mod.o_proj(core_out.reshape(x.shape[0], H * W, -1))

    return {
        "pre_proj_norm_rope_ms": _time_loop(pre, iters, warmup, device),
        "core_ms": _time_loop(core, iters, warmup, device),
        "out_proj_ms": _time_loop(out, iters, warmup, device),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv", type=int, default=3)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    torch.manual_seed(0)
    parent = WindowAttention(dim=args.dim, num_heads=args.heads, num_kv_heads=args.kv,
                             window_size=args.window).to(device=device, dtype=dtype)
    sdpa_rep = WindowAttentionSDPA(dim=args.dim, num_heads=args.heads, num_kv_heads=args.kv,
                                    window_size=args.window, gqa_native=False).to(device=device, dtype=dtype)
    sdpa_rep.load_state_dict(parent.state_dict())
    sdpa_native = sdpa_variant(sdpa_rep)  # GQA-native twin

    torch.manual_seed(1234)
    x = torch.randn(1, args.H * args.W, args.dim, device=device, dtype=dtype)

    report: dict = {
        "device": str(device), "dtype": args.dtype, "grid": f"{args.H}x{args.W}",
        "dim": args.dim, "heads": args.heads, "kv": args.kv, "window": args.window,
        "iters": args.iters, "torch": torch.__version__,
        "hy_smi": "capture utilization in a parallel shell over the run window",
    }
    for shift in (False, True):
        block: dict = {}
        for label, mod in (("explicit", parent), ("sdpa_rep", sdpa_rep), ("sdpa_native", sdpa_native)):
            xs = x.detach().clone().requires_grad_(True)
            res = bench_step(mod, xs, args.H, args.W, shift, args.iters, args.warmup, device)
            res["phases_ms"] = bench_phases(mod, x, args.H, args.W, shift,
                                            max(args.warmup, 5), max(args.warmup, 5), device)
            block[label] = res
        report[f"shift{int(shift)}"] = block

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.out:
        print(f"[bench] -> {args.out}")


if __name__ == "__main__":
    main()
