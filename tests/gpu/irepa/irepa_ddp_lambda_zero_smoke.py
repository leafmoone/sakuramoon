"""P5: 2-rank DDP lambda=0 migration-resume smoke for the iREPA chain.

Launched with torchrun (2 local ranks), NOT pytest:

    torchrun --nproc_per_node=2 \
        tests/gpu/irepa/irepa_ddp_lambda_zero_smoke.py \
        --out /sakuramoon-runtime/artifacts/g1/irepa-ddp-lambda-zero-smoke.json

Runs the EXACT S18 chain (s18_chain.run_chain: source v3 checkpoint N
with two real updates, real ``migrate_irepa_checkpoint``, production
resume into a fresh v4 composite + the production
``hybrid_cmuon_canonical_ns4_fp32_rescue`` optimizer, one update N+1 at
``lambda(N+1) == exact zero``) at world size 2.  Every rank executes the
identical collective sequence: the guarded canonical optimizer performs
its owner election, owner-rank Newton-Schulz + broadcast, device-side
flag and rescue collectives internally (all guarded by
``world_size > 1``).  Rank 0 alone writes the source checkpoint and runs
the migration; both ranks load the shared migrated checkpoint.  The
guard state is world-size stamped (fail-closed on mismatch), so the
whole chain consistently runs at the target world size.

Proof targets (ANY violation => raise, torchrun exits non-zero, NO
retry):
  1. lambda=0 parity (per rank, cross-arm): the spec-18 comparisons —
     all pre-existing parameters / CMuon momenta / AdamW state / guard
     references / SR RNG at resume / losses bit exact (deterministic NS
     mode) or tolerance (production NS mode; see s18_chain.py).
  2. rank consistency (cross-rank): fp32 fingerprint (rms/max) of every
     parameter and CMuon momentum of BOTH arms is bit-identical across
     ranks (all_gather exact equality), i.e. the owner-rank NS broadcast
     keeps both ranks on identical bits after every update.  The
     optimizer's internal invariant_check=True additionally raises
     CMuonSafetyError inside step() on any staged-delta spread.

Model/optimizer/batch are the production S18 chain (production-shape
composite, production optimizer config, controlled seeds) — the smoke
is the spec-18 chain under DDP, not a toy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Self-contained launch (bare torchrun, no pytest/uv src-layout setup):
# make the src package root importable, then this test dir (for s18_chain).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s18_chain


def _fingerprint(name: str, tensor: torch.Tensor) -> tuple[float, float]:
    """fp32 fingerprint of one tensor (rms, max-abs) — bit-sensitive."""

    x = tensor.detach().float()
    return (
        float(x.pow(2).mean().sqrt()),
        float(x.abs().max()) if x.numel() else 0.0,
    )


def _cross_rank_fingerprints(result: s18_chain.ChainResult) -> dict[str, tuple[float, float]]:
    """Fingerprint every parameter and CMuon momentum of BOTH arms."""

    fingerprints: dict[str, tuple[float, float]] = {}
    for arm in ("a", "b"):
        params = result.params_a if arm == "a" else result.params_b
        momenta = (
            result.cmuon_momenta_a if arm == "a" else result.cmuon_momenta_b
        )
        for name in sorted(params):
            fingerprints[f"params.{arm}.{name}"] = _fingerprint(name, params[name])
        for name in sorted(momenta):
            fingerprints[f"momenta.{arm}.{name}"] = _fingerprint(
                name, momenta[name]
            )
    fingerprints["loss.main_a"] = _fingerprint("main_a", result.main_a)
    fingerprints["loss.main_b"] = _fingerprint("main_b", result.main_b)
    return fingerprints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tmp",
        default="/sakuramoon-runtime/s18-ddp-smoke",
        help="scratch root for the source + migrated checkpoints",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="report JSON path (rank 0 writes)",
    )
    parser.add_argument(
        "--ns-mode",
        choices=("deterministic", "production"),
        default="deterministic",
        help="deterministic: test-only deterministic NS (primary gate, bit "
        "exact); production: unpatched production NS (spec-17 leg, "
        "tolerance on NS-affected CMuon values)",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "P5 smoke requires exactly 2 ranks"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    report: dict[str, object] = {
        "rank": rank,
        "world_size": world_size,
        "ns_mode": args.ns_mode,
        "source_update": s18_chain.SOURCE_UPDATE,
        "next_update": s18_chain.NEXT_UPDATE,
    }

    try:
        tmp_path = Path(args.tmp)
        if rank == 0 and tmp_path.exists():
            shutil.rmtree(tmp_path)
        dist.barrier()  # pyright: ignore[reportUnknownMemberType]
        if rank == 0:
            tmp_path.mkdir(parents=True, exist_ok=True)

        deterministic_ns = args.ns_mode == "deterministic"
        t0 = time.time()
        result = s18_chain.run_chain(
            tmp_path,
            deterministic_ns=deterministic_ns,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        chain_seconds = time.time() - t0
        report["chain_seconds"] = chain_seconds

        # -- proof target 1: the spec-18 cross-arm comparisons (per rank) --
        s18_chain.assert_deterministic_parts(result)
        if deterministic_ns:
            s18_chain.assert_primary_gate(result)
            report["lambda_zero_parity"] = "bit_exact (primary gate)"
        else:
            gate = s18_chain.production_ns_gate_report(result)
            report["lambda_zero_parity"] = "tolerance (spec-17 HCU leg)"
            report["production_ns_gate"] = gate

        # -- proof target 2: cross-rank bit identity -------------------------
        local_fp = _cross_rank_fingerprints(result)
        gathered: list[dict[str, tuple[float, float]] | None] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_fp)  # pyright: ignore[reportUnknownMemberType]
        mismatches: list[str] = []
        for other_rank, other_fp in enumerate(gathered):
            assert other_fp is not None
            if other_rank == rank:
                continue
            for key in local_fp:
                if key not in other_fp or local_fp[key] != other_fp[key]:
                    mismatches.append(key)
        report["cross_rank_fingerprints"] = len(local_fp)
        report["cross_rank_mismatches"] = mismatches
        assert not mismatches, (
            f"{len(mismatches)} cross-rank fingerprint mismatches, e.g. "
            f"{mismatches[:8]}"
        )

        # Guard bookkeeping snapshot (for the report).
        report["guard_rescue_b"] = s18_chain.guard_rescue_snapshot(result.guard_b)

        if rank == 0:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[irepa-ddp-lambda-zero-smoke] PASS ({args.ns_mode}) "
                  f"world_size=2 chain={chain_seconds:.1f}s -> {out}",
                  flush=True)
        else:
            print(f"[irepa-ddp-lambda-zero-smoke] rank {rank} PASS "
                  f"({args.ns_mode}) chain={chain_seconds:.1f}s", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
