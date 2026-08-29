"""P2-2 benchmark-ready: the THREE attention backends side by side.

Backends (U233 P2-2 item 5):
  * explicit   — WindowAttention, the frozen verified manual core
                 (hardware.attention_backend = correctness/manual/sdpa-correctness)
  * sdpa_rep   — WindowAttentionSDPA(gqa_native=False), repeat_interleave GQA
  * sdpa_nat   — WindowAttentionSDPA(gqa_native=True), native GQA (enable_gqa)

Measures on the local accelerator (HCU on the remote DTK box, CUDA/CPU locally):
  * full-step (fwd + bwd) throughput, it/s and peak memory
  * per-phase breakdown: pre (proj + qk norms + RoPE) / core / output
    projection, each timed over its own loop
  * the GQA A/B: repeat_interleave (default) vs native q/kv head counts
  * NUMERICAL DIFF: each variant's forward output vs the explicit core
    (rel-L2 + max abs) on shared weights and input — the parity number the
    benchmark report must carry alongside the timing
  * device utilization: best-effort sampling of ``hy-smi`` (DTK/HCU) or
    ``nvidia-smi`` (NVIDIA) in a background thread over the run window
    (null when neither tool exists)

Usage (remote DTK env):

    source /opt/dtk-26.04/env.sh && export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
    /usr/local/bin/python3.11 tools/bench_attention_backends.py \
        --H 128 --W 128 --dim 384 --heads 6 --kv 3 --dtype bf16 \
        --iters 60 --out bench-sdpa.json

NOTE (2026-08-30): on a bad-state HCU host (DTK/HSA driver leak triggered
by bf16/conv allocation patterns — see the anime-sr root-cause note),
bf16 runs are dangerous; the numeric-diff + fp32 paths are safe, bf16
timing/utilization runs belong on a healthy host.

Prints a JSON summary to stdout (and ``--out`` when given).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from anime_sr.model.window_attention import WindowAttention
from anime_sr.model.window_attention_sdpa import WindowAttentionSDPA, sdpa_variant


class _UtilSampler(threading.Thread):
    """Best-effort device utilization sampler (hy-smi / nvidia-smi)."""

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self.tool: str | None = None
        self._stop = threading.Event()

    def _probe(self) -> tuple[str, re.Pattern[str]] | None:
        for cmd, pat in (
            (["hy-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
             re.compile(r"^\s*(\d{1,3})\s*$")),
            (["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
             re.compile(r"^\s*(\d{1,3})\s*$")),
        ):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
                if r.returncode == 0 and pat.match(r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""):
                    return cmd, pat
            except (OSError, subprocess.SubprocessError, IndexError):
                continue
        return None

    def run(self) -> None:
        found = self._probe()
        if found is None:
            return
        self.tool = found[0][0]
        cmd, pat = found
        while not self._stop.wait(self.interval):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
                if r.returncode != 0:
                    continue
                for line in r.stdout.strip().splitlines():
                    m = pat.match(line)
                    if m:
                        self.samples.append(int(m.group(1)))
            except (OSError, subprocess.SubprocessError):
                continue

    def stop(self) -> dict:
        self._stop.set()
        if not self.tool:
            return {"tool": None, "samples": []}
        s = self.samples
        return {
            "tool": self.tool,
            "n_samples": len(s),
            "mean_pct": round(sum(s) / len(s), 1) if s else None,
            "max_pct": max(s) if s else None,
            "min_pct": min(s) if s else None,
        }


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


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).norm().item()
    den = a.norm().item() or 1.0
    return num / den


def numeric_diff(ref: WindowAttention, mod: WindowAttention, x: torch.Tensor,
                 H: int, W: int, shift: bool) -> dict:
    """Forward output of ``mod`` vs the explicit-core ``ref`` (shared
    weights/input): rel-L2 + max abs. Attempt-and-report: backends that
    reject the native-GQA head-count mismatch yield a ``rejected`` record
    (that rejection IS the finding, per the P2-2 parity gate)."""
    try:
        with torch.no_grad():
            a = ref(x, H, W, shift=shift).float()
            b = mod(x, H, W, shift=shift).float()
    except RuntimeError as exc:
        return {"rejected": True, "error": str(exc)[:300]}
    if not torch.isfinite(b).all():
        return {"rejected": True, "error": "non-finite output"}
    return {
        "rejected": False,
        "rel_l2": round(_rel_l2(a, b), 8),
        "max_abs_diff": round((a - b).abs().max().item(), 8),
    }


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
    ap.add_argument("--util-interval", type=float, default=2.0,
                    help="hy-smi/nvidia-smi sampling interval seconds (0 = off)")
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
    }
    sampler = (
        _UtilSampler(interval=args.util_interval)
        if (args.util_interval > 0 and device.type == "cuda")
        else None
    )
    if sampler is not None:
        sampler.start()

    for shift in (False, True):
        block: dict = {}
        for label, mod in (("explicit", parent), ("sdpa_rep", sdpa_rep), ("sdpa_native", sdpa_native)):
            xs = x.detach().clone().requires_grad_(True)
            res = bench_step(mod, xs, args.H, args.W, shift, args.iters, args.warmup, device)
            res["phases_ms"] = bench_phases(mod, x, args.H, args.W, shift,
                                            max(args.warmup, 5), max(args.warmup, 5), device)
            # numeric diff vs the explicit core (the P2-2 report must carry
            # the number alongside the timing; attempt-and-report for the
            # native-GQA variant when the backend rejects the head counts)
            res["numeric_vs_explicit"] = (
                None
                if label == "explicit"
                else numeric_diff(parent, mod, x, args.H, args.W, shift)
            )
            block[label] = res
        report[f"shift{int(shift)}"] = block

    if sampler is not None:
        report["utilization"] = sampler.stop()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.out:
        print(f"[bench] -> {args.out}")


if __name__ == "__main__":
    main()
