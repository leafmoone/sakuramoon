"""2-rank HCU regression: post-commit parameter fingerprint no-grad fix.

Launched with torchrun (2 local ranks), NOT pytest:

    torchrun --nproc_per_node=2 --master_port=29517 \
        tests/gpu/optim/cmuon_fingerprint_nograd_2rank.py \
        --out /tmp/cmuon-fp-nograd-2rank.json

Scenarios (real NCCL, real HCU NS, real F3 step; fixed step counts):
  S1 healthy_pass : 3 clean steps; the post-commit parameter fingerprint
     must run on every step with max_param_rank_diff == 0.0 and an
     INDEPENDENT post-step cross-rank fingerprint check (own all_reduce,
     outside the optimizer) must report spread == 0.0.
  S2 controlled_mismatch : 1 step; rank 0 only emulates a divergent
     parameter by subtracting 1e-3 from the parameter-fingerprint lo
     (MIN) reduce inside a dist.all_reduce wrapper. The real MIN reduce
     propagates the perturbation to BOTH ranks, so the invariant
     violation is a consensus failure: BOTH ranks must raise
     CMuonSafetyError with the EXACT same message (same diff value),
     and the step must not complete on either rank.

Pass criteria (ANY violation => raise => torchrun exits non-zero).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# reuse the production-layout mock from the unit tests (its FQN structure
# is what the CMuon router anchors on; 15 cmuon specs for (256,512,2))
_TESTS_OPT = str(Path(__file__).resolve().parents[2] / "unit" / "optim")
if _TESTS_OPT not in sys.path:
    sys.path.insert(0, _TESTS_OPT)

from test_cmuon import (
    _MockComposite,
    _MockDiT,
    _seed_grads,
)

from sakuramoon.optim.cmuon import (
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
)

NS4 = resolve_ns_map(None, 4)
LR = 0.00015625


def _build_opt(model, rank: int, world_size: int, capsule_root: str):
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    # string keys "fqn#chunkN" — the guard's lookup format (see
    # guarded_canonical.py __init__); constant 1.0 => identical on both
    # ranks (deterministic bootstrap)
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
    return build_fp32_rescue(
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
        rank=rank,
        world_size=world_size,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        hard_fail_artifact_root=capsule_root,
        legacy_forensic_dir=capsule_root,
        emergency_capsule_root=capsule_root,
    )


def _make_model_with_grads(device) -> _MockComposite:
    torch.manual_seed(1234)
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    _seed_grads(model, 101)
    return model


def _diff_from_message(message: str | None) -> float:
    if not message:
        return -1.0
    try:
        return float(message.split("diff ")[1].split(" ")[0])
    except (IndexError, ValueError):
        return -1.0


def _exchange_s2(
    s2: dict, device: torch.device, world_size: int
) -> list[dict]:
    """Exchange S2 fields across ranks via DEVICE tensors (the NCCL group is
    device-bound, so all_gather_object's CPU tensors are rejected)."""
    vals = torch.tensor(
        [
            1.0 if s2["raised"] else 0.0,
            1.0 if s2["step_completed"] else 0.0,
            _diff_from_message(s2["message"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    msg = (s2["message"] or "").encode("utf-8")[:256].ljust(256, b"\x00")
    msg_t = torch.frombuffer(bytearray(msg), dtype=torch.uint8).to(device)
    out_vals = [torch.empty_like(vals) for _ in range(world_size)]
    out_msgs = [torch.empty_like(msg_t) for _ in range(world_size)]
    dist.all_gather(out_vals, vals)
    dist.all_gather(out_msgs, msg_t)
    torch.cuda.synchronize(device)
    results: list[dict] = []
    for v, m in zip(out_vals, out_msgs):
        raw = m.cpu().numpy().tobytes()
        text = raw.split(b"\x00", 1)[0].decode("utf-8")
        results.append(
            {
                "raised": bool(v[0].item() > 0.5),
                "step_completed": bool(v[1].item() > 0.5),
                "diff": float(v[2].item()),
                "message": text if text else None,
            }
        )
    return results


def _independent_param_fp_spread(model) -> float:
    """Cross-rank fingerprint computed OUTSIDE the optimizer (own
    all_reduce, own no_grad) — verifies the committed parameters really
    are identical across ranks after each step."""
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    with torch.no_grad():
        fp = []
        for spec in routing.cmuon_specs:
            pf = spec.parameter.float()
            fp.append(pf.pow(2).mean().sqrt())
            fp.append(pf.abs().max())
        flat = torch.stack(fp)
        lo = flat.clone()
        hi = flat.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return float((hi - lo).max().item())


class _MismatchWrapper:
    """dist.all_reduce wrapper (rank-0 only). FP32 MIN reduce order per
    step is [delta_lo, param_lo]; on the SECOND fp32 MIN reduce (the
    parameter fingerprint lo) it emulates the other rank holding a
    strictly smaller value."""

    def __init__(self, inner, perturb: float):
        self.inner = inner
        self.perturb = perturb
        self.min_count = 0
        self.injected = False

    def __call__(self, tensor, op=None, group=None, async_op=False):
        if tensor.dtype == torch.float32 and op == dist.ReduceOp.MIN:
            if self.min_count == 1 and not self.injected:
                tensor.sub_(self.perturb)
                self.injected = True
            self.min_count += 1
        return self.inner(tensor, op=op, group=group, async_op=async_op)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert world_size == 2, "this regression requires exactly 2 ranks"
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)

    report: dict = {"rank": rank, "scenarios": {}}
    try:
        # ------------------------------------------------------------------
        # S1: healthy pass (3 steps)
        # ------------------------------------------------------------------
        model = _make_model_with_grads(device)
        capsule = f"/tmp/cmuon-fp-nograd-2rank-capsule-r{rank}"
        opt = _build_opt(model, rank, world_size, capsule)
        s1: dict = {"steps": [], "max_param_rank_diff_max": 0.0}
        for i in range(3):
            opt.step()
            indep = _independent_param_fp_spread(model)
            s1["steps"].append(
                {
                    "step": i + 1,
                    "max_param_rank_diff": float(opt.max_param_rank_diff),
                    "independent_spread": indep,
                    "observations": int(opt.observations),
                }
            )
            s1["max_param_rank_diff_max"] = max(
                s1["max_param_rank_diff_max"], float(opt.max_param_rank_diff)
            )
        report["scenarios"]["S1_healthy_pass"] = s1
        assert all(
            s["max_param_rank_diff"] == 0.0 for s in s1["steps"]
        ), f"S1: max_param_rank_diff != 0: {s1}"
        assert all(
            s["independent_spread"] == 0.0 for s in s1["steps"]
        ), f"S1: independent cross-rank fingerprint spread != 0: {s1}"
        assert s1["steps"][-1]["observations"] == 3

        # ------------------------------------------------------------------
        # S2: controlled mismatch (rank 0 emulates a divergent rank)
        # ------------------------------------------------------------------
        model = _make_model_with_grads(device)
        opt = _build_opt(model, rank, world_size, capsule)
        s2: dict = {"raised": False, "message": None, "step_completed": False}
        original_all_reduce = dist.all_reduce
        if rank == 0:
            dist.all_reduce = _MismatchWrapper(original_all_reduce, 1e-3)
        try:
            opt.step()
            s2["step_completed"] = True
        except CMuonSafetyError as ei:
            s2["raised"] = True
            s2["message"] = str(ei)
        finally:
            if rank == 0:
                dist.all_reduce = original_all_reduce
        report["scenarios"]["S2_controlled_mismatch"] = s2

        # cross-rank exchange via DEVICE tensors: the NCCL process group is
        # bound to a device (device_id=...), so all_gather_object (which
        # uses CPU tensors) is rejected. float64 + uint8 are NCCL-supported.
        gathered = _exchange_s2(s2, device, world_size)
        s2r0, s2r1 = gathered
        if rank == 0:
            # the MIN/MAX reduce synchronizes the mismatch to BOTH ranks:
            # a consensus failure must raise on every rank
            assert s2r0["raised"] and s2r1["raised"], (
                f"S2: both ranks must raise CMuonSafetyError: {s2r0} | {s2r1}"
            )
            msg = s2r0["message"]
            assert msg is not None and msg.startswith(
                "fp32-rescue rank invariant violated: "
                "cross-rank parameter fingerprint diff"
            ), f"S2: unexpected message: {msg}"
            assert s2r0["message"] == s2r1["message"], (
                "S2: identical message expected on both ranks "
                f"({s2r0['message']!r} vs {s2r1['message']!r})"
            )
            assert s2r0["diff"] == s2r1["diff"] and s2r0["diff"] > 0.0, (
                f"S2: positive identical diff expected: {s2r0} | {s2r1}"
            )
            assert not s2r0["step_completed"] and not s2r1["step_completed"]
            with open(args.out, "w") as fh:
                json.dump(
                    {
                        "s2_gathered": [s2r0, s2r1],
                        "s1_rank0": s1,
                        "s1_rank1_note": "rank1 s1 identical (same seeds)",
                    },
                    fh,
                    indent=2,
                )
        dist.destroy_process_group()
        if rank == 0:
            print("2RANK_PASS")
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as ei:  # noqa: BLE001
                print(
                    f"warning: final destroy_process_group failed: {ei}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
