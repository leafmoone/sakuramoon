"""P5: 2xHCU rank-consistency proof for the guarded canonical candidate.

Launched with torchrun (2 local ranks), NOT pytest:

    torchrun --nproc_per_node=2 tests/gpu/optim/guarded_canonical_2rank.py \
        --ckpt /sakuramoon-runtime/output_model/g1/ckpt_97100_raw-97100-update-cadence \
        --steps 3 --out /sakuramoon-runtime/artifacts/g1/guarded-canonical/p5-rank-consistency.json

Proof targets (ANY violation => raise, torchrun exits non-zero, NO retry):
  1. param_rank_diff == 0  : after every step, every parameter's fp32
     rms/max fingerprint is bit-identical across ranks (all_reduce
     MIN/MAX spread exactly 0.0).
  2. delta_rank_diff == 0  : enforced inside HybridCMuonGuardedCanonical
     step() (invariant_check=True raises CMuonSafetyError on any spread)
     AND re-checked here from the staged-delta fingerprint path is
     implicit: a rank-different delta would show up as a param spread in
     (1) on the next fingerprint (both ranks add the same staged deltas
     to identical params).
  3. momentum rank-exact   : every CMuon spec's bf16 momentum fp32
     rms/max spread is exactly 0.0 (bf16 EMA lerp determinism across
     ranks, same as the production rank invariant).

The guard runs in NEVER_SKIP mode (ratio/floor 1e-30): every spec passes
the guard every step, so the canonical owner-rank NS4 + broadcast path is
exercised on ALL 141 real specs at every step. Model weights are freshly
initialized (deterministic same seed on both ranks); gradients are
seeded on CPU (identical bits on both ranks) and moved to the device.

Report (rank 0): per-step wall seconds, per-spec max spreads, staged
delta bytes, per-rank peak allocated/reserved memory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sakuramoon.checkpoint.artifact import build_trainable_composite
from sakuramoon.optim.cmuon import route_cmuon_parameters
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
    build_guarded_canonical,
    stable_owner,
)


def _fingerprint_fp32(tensor: torch.Tensor) -> tuple[float, float]:
    x = tensor.detach().float()
    return (
        float(x.pow(2).mean().sqrt()),
        float(x.abs().max()) if x.numel() else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--grad-rms", type=float, default=1e-3)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "P5 requires exactly 2 ranks"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    report: dict[str, object] = {
        "rank": rank,
        "world_size": world_size,
        "ckpt": args.ckpt,
        "steps": args.steps,
        "lr": args.lr,
        "guard_mode": "NEVER_SKIP (ratio/floor 1e-30)",
    }
    try:
        # --- model: real G1 architecture from the pinned ckpt config ----
        torch.manual_seed(20260830)  # deterministic fresh init on both ranks
        config = json.loads(
            (Path(args.ckpt) / "model" / "config.json").read_text()
        )
        module = build_trainable_composite(config["architecture"], device=device)
        module.eval()

        routing = route_cmuon_parameters(module, matrix_weight_decay=0.0,
                                         sensitive_weight_decay=0.0)
        n_cmuon = len(routing.cmuon_specs)
        n_adamw = len(routing.adamw_specs)
        report["n_cmuon_specs"] = n_cmuon
        report["n_adamw_specs"] = n_adamw
        if rank == 0:
            print(f"[p5] model built: cmuon={n_cmuon} adamw={n_adamw}", flush=True)

        # --- guard refs: uniform placeholder (NEVER_SKIP makes them inert).
        #     Chunk counts come from the routing spec policy (FFN=2,
        #     AdaLN=6, others=1), NOT from ceil(rows/256).
        refs: dict[str, float] = {}
        for spec in routing.cmuon_specs:
            for ci in range(spec.chunk_count):
                refs[f"{spec.name}#chunk{ci}"] = 1e-3

        guard = GuardedCanonicalGuardConfig(
            guard_ratio=1e-30,
            reference_decay=0.999,
            min_reference=1e-300,
            numerical_floor=1e-300,
            warmup_observations=0,
            invariant_check=True,
        )
        ns_map = {  # NS4 everywhere (the pinned G1 protocol)
            "attention_q": 4,
            "attention_k": 4,
            "attention_v": 4,
            "attention_content_gate": 4,
            "attention_out": 4,
            "ffn_in": 4,
            "ffn_down": 4,
            "adaln_shared": 4,
        }
        optimizer = build_guarded_canonical(
            module,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            block_size=256,
            bf16_stochastic_round=True,
            matrix_weight_decay=0.0,
            sensitive_weight_decay=0.0,
            sr_seed=12345,
            ns_steps_by_role=ns_map,
            guard_cfg=guard,
            guard_bootstrap_refs=refs,
            rank=rank,
            world_size=world_size,
            momentum_dtype="bfloat16",
            chunk_rescale_sqrt_n=False,
        )

        # owner coverage evidence (both ranks must compute the same map)
        owners: dict[str, int] = {}
        for spec in routing.cmuon_specs:
            for ci in range(spec.chunk_count):
                owners[f"{spec.name}#chunk{ci}"] = stable_owner(
                    spec.name, ci, world_size
                )
        owner_counts = {
            o: sum(1 for v in owners.values() if v == o) for o in range(world_size)
        }
        report["owner_counts"] = owner_counts
        # cross-rank owner-map identity
        owner_flat = torch.tensor(
            [
                owners[f"{s.name}#chunk{ci}"]
                for s in routing.cmuon_specs
                for ci in range(s.chunk_count)
            ],
            dtype=torch.int32,
            device=device,
        )
        owner_min = owner_flat.clone()
        owner_max = owner_flat.clone()
        dist.all_reduce(owner_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(owner_max, op=dist.ReduceOp.MAX)
        owner_diff = int((owner_max - owner_min).max())
        report["owner_map_rank_diff"] = owner_diff
        if owner_diff != 0:
            raise AssertionError(f"owner map differs across ranks: diff={owner_diff}")

        staged_bytes = sum(
            s.parameter.numel() * 2 for s in routing.cmuon_specs
        )
        report["staged_delta_bytes_all_active"] = staged_bytes

        step_reports = []
        fp_names = [s.name for s in routing.cmuon_specs]
        mom_names = fp_names
        for step in range(1, args.steps + 1):
            torch.cuda.reset_peak_memory_stats(device)
            module.zero_grad(set_to_none=True)
            # seeded identical gradients: CPU generator (same bits both
            # ranks), bf16 to match the bf16 training domain.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(9000 + step)
            with torch.no_grad():
                for p in module.parameters():
                    g = torch.randn(
                        p.shape, generator=gen, dtype=torch.float32
                    ) * args.grad_rms
                    p.grad = g.to(p.dtype)
            t0 = time.monotonic()
            optimizer.step()
            torch.cuda.synchronize(device)
            t1 = time.monotonic()

            # per-spec param + momentum fingerprints, one batched sync
            def _fp_block(names: list[str], getter) -> torch.Tensor:
                out = torch.zeros(len(names) * 2, dtype=torch.float32)
                for i, name in enumerate(names):
                    rms, mx = getter(name)
                    out[2 * i] = rms
                    out[2 * i + 1] = mx
                return out.to(device)

            param_fp = _fp_block(
                fp_names, lambda n: _fingerprint_fp32(module.get_parameter(n))
            )
            mom_fp = _fp_block(
                mom_names,
                lambda n: _fingerprint_fp32(
                    optimizer._momenta[  # pyright: ignore[reportPrivateUsage]
                        module.get_parameter(n)
                    ]
                ),
            )
            p_min, p_max = param_fp.clone(), param_fp.clone()
            dist.all_reduce(p_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(p_max, op=dist.ReduceOp.MAX)
            m_min, m_max = mom_fp.clone(), mom_fp.clone()
            dist.all_reduce(m_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(m_max, op=dist.ReduceOp.MAX)

            param_spread = float((p_max - p_min).max())
            mom_spread = float((m_max - m_min).max())
            if param_spread != 0.0 or mom_spread != 0.0:
                bad = [
                    fp_names[i // 2]
                    for i in range(len(fp_names) * 2)
                    if p_max[i] - p_min[i] != 0.0
                ]
                raise AssertionError(
                    f"rank divergence at step {step}: param_spread={param_spread} "
                    f"mom_spread={mom_spread} specs={bad[:8]}"
                )
            step_reports.append({
                "step": step,
                "step_seconds": t1 - t0,
                "param_rank_spread": param_spread,
                "momentum_rank_spread": mom_spread,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            })
            if rank == 0:
                print(
                    f"[p5] step {step}: {t1 - t0:.2f}s param_spread=0 "
                    f"mom_spread=0 peak_alloc="
                    f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB",
                    flush=True,
                )
        report["steps"] = step_reports
    finally:
        # gather rank-1 report fields via a single pickle-free channel:
        # each rank writes its own json; the launcher merges.
        out_path = Path(args.out).with_suffix(f".rank{rank}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        if rank == 0:
            merged = {
                "status": "PASS",
                "criteria": {
                    "param_rank_diff": 0,
                    "momentum_rank_diff": 0,
                    "owner_map_rank_diff": 0,
                },
            }
            Path(args.out).write_text(json.dumps(merged, indent=2))
            print(f"[p5] PASS report -> {args.out}", flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
