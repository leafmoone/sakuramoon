"""Fingerprint-phase memory/time probe (isolated, no production trainer).

Measures the post-commit parameter fingerprint cost of
``HybridCMuonCanonicalNS4FP32Rescue.step`` in an isolated process:

  * a temporary mock model with the production FQN layout
    (_MockComposite/_MockDiT scaled up) and a deterministic
    (seeded) gradient,
  * world_size=2 emulated in ONE process (identity collective mocks —
    the fingerprint block only runs for world_size > 1; the mock
    reproduces the unit-test methodology and the collective itself
    does not affect the fingerprint's autograd retention behavior),
  * REAL HCU Newton-Schulz (no NS mocks),
  * 2 warm-up steps, then 3 timed steps per version, each measured
    with cuda.synchronize + reset_peak_memory_stats,
  * no empty_cache and no heavy hooks inside the timed windows.

Run once per version (separate processes; the launcher picks
PYTHONPATH = BASE main-clone src vs FIXED worktree src):

    python dev-tools/cmuon_fingerprint_memprobe.py \
        --device cuda:0 --out /tmp/memprobe-base.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

_TESTS_OPT = Path(__file__).resolve().parents[1] / "tests" / "unit" / "optim"
if str(_TESTS_OPT) not in sys.path:
    sys.path.insert(0, str(_TESTS_OPT))

from test_cmuon import (
    _MockComposite,
    _MockDiT,
    _seed_grads,
)

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.optim.cmuon import (
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
)

NS4 = resolve_ns_map(None, 4)
LR = 0.00015625

# scaled-up production-layout mock: per block (hidden h, ffn i):
#   3.5*h^2 + 3*h*i elements (bf16); 5 blocks + input_projection
# h=4096, i=20000 -> ~1.52e9 bf16 elements (~ the GO 11.39 GiB
# retention-estimate scale at 8 bytes retained per element)
HIDDEN = 4096
INTER = 20000
N_BLOCKS = 5


def _identity_all_reduce(tensor, op=None, group=None, async_op=False):
    # single-process world_size=2: both "ranks" hold identical values
    return None


def _identity_broadcast(tensor, src=0, group=None, async_op=False):
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1234)

    model = _MockComposite(_MockDiT(HIDDEN, INTER, N_BLOCKS)).to(device)
    _seed_grads(model, 101)

    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    refs: dict[str, float] = {}
    for spec in routing.cmuon_specs:
        for ci in range(spec.chunk_count):
            refs[f"{spec.name}#chunk{ci}"] = 1.0
    guard = GuardedCanonicalGuardConfig(
        guard_ratio=0.05,
        reference_decay=0.999,
        min_reference=1e-12,
        numerical_floor=1e-20,
        warmup_observations=0,
        invariant_check=True,
    )
    capsule = f"/tmp/cmuon-fp-memprobe-r{device.index}"
    opt = build_fp32_rescue(
        model,
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
        sr_seed=44,
        ns_steps_by_role=NS4,
        guard_cfg=guard,
        guard_bootstrap_refs=refs,
        rank=0,
        world_size=2,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        hard_fail_artifact_root=capsule,
        legacy_forensic_dir=capsule,
        emergency_capsule_root=capsule,
    )
    opt.owner_of = lambda fqn, chunk: 0
    # collective mocks (single-process world_size=2)
    dist.all_reduce = _identity_all_reduce  # type: ignore[misc]
    dist.broadcast = _identity_broadcast  # type: ignore[misc]

    cmuon_elements = sum(
        spec.parameter.numel() for spec in routing.cmuon_specs
    )
    cmuon_bytes = sum(
        spec.parameter.numel() * spec.parameter.element_size()
        for spec in routing.cmuon_specs
    )

    # warm-up (allocator settle; NOT timed)
    for _ in range(args.warmup):
        opt.step()
    torch.cuda.synchronize(device)

    samples = []
    for i in range(args.measured):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        samples.append(
            {
                "run": i + 1,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "post_allocated_bytes": torch.cuda.memory_allocated(device),
                "time_s": t1 - t0,
            }
        )

    result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cmuon_specs": len(routing.cmuon_specs),
        "cmuon_elements": cmuon_elements,
        "cmuon_bytes": cmuon_bytes,
        "expected_baseline_retention_bytes": 8 * cmuon_elements,
        "world_size": 2,
        "invariant_check": True,
        "collectives": "identity-mock (single process)",
        "ns": "real HCU Newton-Schulz",
        "warmup_steps": args.warmup,
        "samples": samples,
        "max_param_rank_diff": float(opt.max_param_rank_diff),
        "module_file": Path(fr.__file__).resolve().as_posix(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
