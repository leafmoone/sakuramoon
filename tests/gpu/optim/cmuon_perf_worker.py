"""One-configuration optimizer-phase measurement worker (cleanup spec 15).

Runs under torchrun (2 local ranks, production world size). The ACTIVE
sakuramoon tree is selected by PYTHONPATH set by the launcher:

    PYTHONPATH=<tree>/src torchrun --nproc_per_node=2 \
        tests/gpu/optim/cmuon_perf_worker.py --config {adamw8bit,baseline,cleanup} \
        --ckpt ... --out ...json

Same model checkpoint, same world size, same shapes, same synthetic-grad
generation (per-iteration CPU-seeded randn -> device), same warmup and
iteration count for every configuration (the launcher guarantees identical
arguments). Records optimizer-phase wall time (synced) + host sync counts.
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
from sakuramoon.optim.cmuon import (
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
)

LR = 0.00015625
COMMON = {
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


def _grad_gen(seed_base: int, it: int, params: list[torch.nn.Parameter]) -> None:
    """Deterministic synthetic gradients: per-iteration CPU seed, identical
    for every configuration (spec 15 'same synthetic-grad generation')."""
    g = torch.Generator().manual_seed(seed_base + it)
    for p in params:
        if p.requires_grad:
            # DTK 2.9 randn has no scalar-size overload: draw (1,) then
            # reshape for the production model's 0-D scalar params.
            size = (1,) if p.ndim == 0 else tuple(p.shape)
            x = torch.randn(size, generator=g, dtype=torch.float32)
            if p.ndim == 0:
                x = x.reshape(())
            p.grad = x.to(device=p.device, dtype=p.dtype)


def _bootstrap_refs(model) -> dict[str, float]:
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


def _build(cfg_name: str, module, rank: int, world_size: int):
    if cfg_name == "adamw8bit":
        from sakuramoon.optim.adamw8bit import build_adamw8bit

        # build_adamw8bit has no momentum_dtype / chunk_rescale_sqrt_n
        # (CMuon-only knobs): pass only the shared policy args.
        adamw_common = {
            k: v
            for k, v in COMMON.items()
            if k not in ("momentum_dtype", "chunk_rescale_sqrt_n")
        }
        return build_adamw8bit(module=module, **adamw_common)
    from sakuramoon.optim.fp32_rescue import build_fp32_rescue

    return build_fp32_rescue(
        module=module,
        ns_steps_by_role=resolve_ns_map(None, 4),
        guard_cfg=GuardedCanonicalGuardConfig(
            guard_ratio=0.05,
            reference_decay=0.999,
            min_reference=1e-12,
            numerical_floor=1e-20,
            warmup_observations=0,
            invariant_check=True,
        ),
        guard_bootstrap_refs=_bootstrap_refs(module),
        rank=rank,
        world_size=world_size,
        **COMMON,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["adamw8bit", "baseline", "cleanup"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=777001)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "perf worker requires the production world size (2)"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    try:
        torch.manual_seed(20260902)
        config = json.loads(
            (Path(args.ckpt) / "model" / "config.json").read_text()
        )
        module = build_trainable_composite(config["architecture"], device=device)
        module.eval()
        params = [p for p in module.parameters() if p.requires_grad]
        opt = _build(args.config, module, rank, world_size)
        src = getattr(sys.modules.get("sakuramoon"), "__file__", "?")

        torch.cuda.synchronize(device)
        times: list[float] = []
        syncs: list[int] = []
        for it in range(args.warmup + args.iters):
            _grad_gen(args.seed_base, it, params)
            torch.cuda.synchronize(device)
            if it >= args.warmup:
                counter = SyncCounter()
                t0 = time.perf_counter()
                with counter:
                    opt.step()
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - t0)
                syncs.append(counter.total())
        mean = sum(times) / len(times)
        out = {
            "config": args.config,
            "rank": rank,
            "world_size": world_size,
            "tree": src,
            "n": len(times),
            "mean": mean,
            "median": _percentile(times, 0.5),
            "p90": _percentile(times, 0.9),
            "std": (
                sum((t - mean) ** 2 for t in times) / len(times)
            ) ** 0.5,
            "min": min(times),
            "max": max(times),
            "host_syncs_per_step_mean": sum(syncs) / len(syncs),
            "host_syncs_first5": syncs[:5],
            "host_syncs_last5": syncs[-5:],
        }
        if rank == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
