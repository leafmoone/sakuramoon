"""Same-pod optimizer-phase A/B: AdamW8bit vs D4-baseline fp32_rescue vs
cleanup fp32_rescue (production cleanup spec section 15 + section 14 sync
counts, production-shaped real G1 model).

Launched with torchrun (2 local ranks, production world size):

    torchrun --nproc_per_node=2 tests/gpu/optim/cmuon_perf_ab.py \
        --ckpt /sakuramoon-runtime/output_model/g1/ckpt_103500_raw-103500-update-cadence \
        --baseline-src /sakuramoon-runtime/sakuramoon/src \
        --out /sakuramoon-runtime/artifacts/g1/cmuon-cleanup/perf-ab.json

Design (spec 15): same model checkpoint, same world size, same shapes,
same synthetic-grad generation (per-iteration CPU-seeded randn -> device),
same warmup (>=5) and measured iterations (>=40). The three optimizer
configurations are measured SEQUENTIALLY on the same process/device so pod
drift cannot contaminate the comparison. Per configuration we record
optimizer-phase wall time (synced) and host sync counts (SyncCounter).

The D4 baseline fp32_rescue is imported from --baseline-src (the frozen
37602e4 tree) by sys.path swap in a child-less, single-process sequence:
each configuration is built, measured, then deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from _sync_counter import SyncCounter

from sakuramoon.checkpoint.artifact import build_trainable_composite


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _grad_gen(seed_base: int, it: int, params: list[torch.nn.Parameter]):
    """Deterministic synthetic gradients: per-iteration CPU seed, identical
    for every configuration."""
    g = torch.Generator().manual_seed(seed_base + it)
    for p in params:
        if p.requires_grad:
            p.grad = torch.randn(
                *p.shape, generator=g, dtype=torch.float32
            ).to(p.dtype, device=p.device)


def _measure(
    cfg_name: str,
    build: object,
    params: list[torch.nn.Parameter],
    device: torch.device,
    seed_base: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    opt = build()
    torch.cuda.synchronize(device)
    times: list[float] = []
    syncs: list[int] = []
    for it in range(warmup + iters):
        _grad_gen(seed_base, it, params)
        torch.cuda.synchronize(device)
        if it >= warmup:
            counter = SyncCounter()
            t0 = time.perf_counter()
            with counter:
                opt.step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
            syncs.append(counter.total())
    stats = {
        "n": len(times),
        "mean": sum(times) / len(times),
        "median": _percentile(times, 0.5),
        "p90": _percentile(times, 0.9),
        "std": (
            sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)
        ) ** 0.5,
        "min": min(times),
        "max": max(times),
        "host_syncs_per_step_mean": sum(syncs) / len(syncs),
        "host_syncs_per_step_samples": syncs[:5] + syncs[-5:],
        "sync_counts": None,
    }
    del opt
    torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--baseline-src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=777001)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "perf A/B requires the production world size (2)"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    report: dict[str, object] = {"rank": rank, "world_size": world_size}
    try:
        torch.manual_seed(20260902)
        config = json.loads(
            (Path(args.ckpt) / "model" / "config.json").read_text()
        )
        module = build_trainable_composite(config["architecture"], device=device)
        module.eval()
        params = [p for p in module.parameters() if p.requires_grad]

        # ---- common optimizer kwargs (identical across configurations) ----
        from sakuramoon.optim.cmuon import resolve_ns_map

        NS4 = resolve_ns_map(None, 4)
        LR = 0.00015625
        common = {
            "module": module,
            "lr": LR,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "block_size": 256,
            "bf16_stochastic_round": True,
            "matrix_weight_decay": 0.0,
            "sensitive_weight_decay": 0.0,
            "sr_seed": 44,
            "momentum_dtype": "bfloat16",
            "chunk_rescale_sqrt_n": False,
        }

        # 1) AdamW8bit default path
        from sakuramoon.optim.adamw8bit import build_adamw8bit

        report["adamw8bit"] = _measure(
            "adamw8bit",
            lambda: build_adamw8bit(**common),
            params,
            device,
            args.seed_base,
            args.warmup,
            args.iters,
        )

        # 2) D4 baseline fp32_rescue (frozen tree at --baseline-src)
        sys.path.insert(0, args.baseline_src)
        for mod in list(sys.modules):
            if mod.startswith("sakuramoon"):
                del sys.modules[mod]
        from sakuramoon.optim.cmuon import resolve_ns_map as _r4  # noqa: F401
        from sakuramoon.optim.cmuon import route_cmuon_parameters
        from sakuramoon.optim.fp32_rescue import (
            build_fp32_rescue,
        )
        from sakuramoon.optim.guarded_canonical import (
            GuardedCanonicalGuardConfig,
        )

        def _refs(model):
            routing = route_cmuon_parameters(
                model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
            )
            refs: dict[str, float] = {}
            for spec in routing.cmuon_specs:
                gg = torch.randn_like(spec.parameter)
                for ci in range(spec.chunk_count):
                    chunk_size = spec.chunk_size()
                    if spec.chunk_count == 1:
                        sig = gg.float().pow(2).mean().sqrt().item()
                    else:
                        start = ci * chunk_size
                        end = start + chunk_size
                        sl = [slice(None)] * gg.ndim
                        sl[spec.chunk_dim] = slice(start, end)
                        sig = gg[tuple(sl)].float().pow(2).mean().sqrt().item()
                    refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)
            return refs

        def _build_baseline_rescue():
            return build_fp32_rescue(
                **common,
                ns_steps_by_role=NS4,
                guard_cfg=GuardedCanonicalGuardConfig(
                    guard_ratio=0.05,
                    reference_decay=0.999,
                    min_reference=1e-12,
                    numerical_floor=1e-20,
                    warmup_observations=0,
                    invariant_check=True,
                ),
                guard_bootstrap_refs=_refs(module),
                rank=rank,
                world_size=world_size,
            )

        report["baseline_fp32_rescue"] = _measure(
            "baseline_fp32_rescue",
            _build_baseline_rescue,
            params,
            device,
            args.seed_base + 1000,
            args.warmup,
            args.iters,
        )

        # 3) cleanup fp32_rescue (this tree)
        sys.path.remove(args.baseline_src)
        for mod in list(sys.modules):
            if mod.startswith("sakuramoon"):
                del sys.modules[mod]
        from sakuramoon.optim.cmuon import route_cmuon_parameters as _r2
        from sakuramoon.optim.fp32_rescue import (
            build_fp32_rescue as _b2,
        )
        from sakuramoon.optim.guarded_canonical import (
            GuardedCanonicalGuardConfig as _G2,
        )

        def _refs2(model):
            routing = _r2(
                model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
            )
            refs: dict[str, float] = {}
            for spec in routing.cmuon_specs:
                gg = torch.randn_like(spec.parameter)
                for ci in range(spec.chunk_count):
                    chunk_size = spec.chunk_size()
                    if spec.chunk_count == 1:
                        sig = gg.float().pow(2).mean().sqrt().item()
                    else:
                        start = ci * chunk_size
                        end = start + chunk_size
                        sl = [slice(None)] * gg.ndim
                        sl[spec.chunk_dim] = slice(start, end)
                        sig = gg[tuple(sl)].float().pow(2).mean().sqrt().item()
                    refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)
            return refs

        def _build_cleanup_rescue():
            return _b2(
                **common,
                ns_steps_by_role=NS4,
                guard_cfg=_G2(
                    guard_ratio=0.05,
                    reference_decay=0.999,
                    min_reference=1e-12,
                    numerical_floor=1e-20,
                    warmup_observations=0,
                    invariant_check=True,
                ),
                guard_bootstrap_refs=_refs2(module),
                rank=rank,
                world_size=world_size,
            )

        report["cleanup_fp32_rescue"] = _measure(
            "cleanup_fp32_rescue",
            _build_cleanup_rescue,
            params,
            device,
            args.seed_base + 2000,
            args.warmup,
            args.iters,
        )
        report["verdict_pending"] = True
    finally:
        if rank == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
