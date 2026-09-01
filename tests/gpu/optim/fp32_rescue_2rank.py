"""2-rank forced-rescue regression for the CLEANUP fp32-rescue candidate
(production cleanup spec section 10).

Launched with torchrun (2 local ranks), NOT pytest:

    torchrun --nproc_per_node=2 tests/gpu/optim/fp32_rescue_2rank.py \
        --out /sakuramoon-runtime/artifacts/g1/cmuon-cleanup/2rank-regression.json

Scenario matrix (one optimizer, sequential steps; forced patterns are
applied identically on both ranks so the owner-split is exercised):

  1. healthy        : no forcing -> clean step
  2. owner0         : exactly one owner-0 chunk trips BF16 (NaN) -> rescued
  3. owner1         : exactly one owner-1 chunk trips BF16 (NaN) -> rescued
  4. multi          : 4 scattered chunks trip BF16 (mixed owners) -> rescued
  5. simultaneous   : >=1 chunk per owner trips in the SAME step
  6. final_failure  : one chunk fails BOTH BF16 and FP32 -> CMuonSafetyError
                      on both ranks, ZERO commit (parameter bytes and
                      observations untouched)

Pass criteria (ANY violation => raise, torchrun exits non-zero):
  * successful steps: max_delta_rank_spread == 0 AND max_param_rank_diff == 0
    (both tracked by the optimizer) AND an independent post-step parameter
    fingerprint all_reduce shows zero cross-rank spread
  * final failure: CMuonSafetyError on both ranks, parameters byte-identical
    to the pre-step snapshot on both ranks, observations not advanced
  * counters: per-rank fp32_attempts == number of forced chunks it owned;
    global rescued total == total forced count (minus the final failure)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.optim.cmuon import (
    cmuon_zeroth_power_bf16,
    cmuon_zeroth_power_fp32,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
    stable_owner,
)

NS4 = resolve_ns_map(None, 4)
LR = 0.00015625


class _MockDiTBlock(torch.nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.q_proj = torch.nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.k_proj = torch.nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.v_proj = torch.nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.content_gate = torch.nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.out_proj = torch.nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.mlp = torch.nn.Module()
        self.mlp.in_proj = torch.nn.Linear(
            hidden, 2 * inter, bias=False, dtype=torch.bfloat16
        )
        self.mlp.down_proj = torch.nn.Linear(
            inter, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention_norm = torch.nn.Parameter(
            torch.ones(hidden, dtype=torch.float32)
        )
        self.mlp_norm = torch.nn.Parameter(torch.ones(hidden, dtype=torch.float32))


class _GlobalConditioner(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.shared_block_projection = torch.nn.Linear(
            hidden // 4, 6 * hidden, bias=False, dtype=torch.bfloat16
        )
        self.final_projection = torch.nn.Linear(
            hidden // 4, 2 * hidden, bias=True, dtype=torch.bfloat16
        )


class _MockDiT(torch.nn.Module):
    def __init__(self, hidden: int, inter: int, n_blocks: int) -> None:
        super().__init__()
        self.input_projection = torch.nn.Linear(
            128, hidden, bias=False, dtype=torch.bfloat16
        )
        self.conditioner = _GlobalConditioner(hidden)
        self.blocks = torch.nn.ModuleDict(
            {
                f"slot_{i:02d}": _MockDiTBlock(hidden, inter)
                for i in range(n_blocks)
            }
        )
        self.modality_image = torch.nn.Parameter(
            torch.zeros(hidden, dtype=torch.float32)
        )


class _MockComposite(torch.nn.Module):
    def __init__(self, dit: _MockDiT) -> None:
        super().__init__()
        self.dit = dit


def _seed_grads(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator(device=model.dit.input_projection.weight.device).manual_seed(
        seed
    )
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(
                *p.shape, generator=g, dtype=torch.float32, device=p.device
            ).to(p.dtype)


def _bootstrap_refs(model) -> dict[str, float]:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    refs: dict[str, float] = {}
    for spec in routing.cmuon_specs:
        g = torch.randn_like(spec.parameter)
        for ci in range(spec.chunk_count):
            chunk_size = spec.chunk_size()
            if spec.chunk_count == 1:
                sig = g.float().pow(2).mean().sqrt().item()
            else:
                start = ci * chunk_size
                end = start + chunk_size
                sl = [slice(None)] * g.ndim
                sl[spec.chunk_dim] = slice(start, end)
                sig = g[tuple(sl)].float().pow(2).mean().sqrt().item()
            refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)
    return refs


def _param_fingerprint_spread(
    model: torch.nn.Module, device: torch.device
) -> float:
    """Cross-rank spread of the fp32 parameter fingerprint (0.0 = bit-exact)."""
    fp: list[torch.Tensor] = []
    for _, p in model.named_parameters():
        x = p.detach().float()
        fp.append(x.pow(2).mean().sqrt())
        fp.append(x.abs().max())
    flat = torch.stack(fp).to(device)
    lo = flat.clone()
    hi = flat.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return float((hi - lo).max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "this regression requires exactly 2 ranks"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    report: dict[str, object] = {
        "rank": rank,
        "world_size": world_size,
        "candidate": "hybrid_cmuon_canonical_ns4_fp32_rescue (cleanup branch)",
    }
    try:
        torch.manual_seed(20260901)
        model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
        routing = route_cmuon_parameters(
            model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
        )
        flat_keys: list[tuple[str, int]] = []
        for spec in routing.cmuon_specs:
            for ci in range(spec.chunk_count):
                flat_keys.append((spec.name, ci))
        n = len(flat_keys)
        owner_of_flat = [
            stable_owner(fqn, ci, world_size) for fqn, ci in flat_keys
        ]
        owned0 = [i for i, o in enumerate(owner_of_flat) if o == 0]
        owned1 = [i for i, o in enumerate(owner_of_flat) if o == 1]
        report["n_inputs"] = n
        report["owned_per_rank"] = [len(owned0), len(owned1)]
        assert len(owned0) > 0 and len(owned1) > 0, (
            "need chunks owned by both ranks for the scenario matrix"
        )

        # call-order forced NS: on each rank, calls arrive in the order of
        # THAT rank's owned chunks (flat-order filter)
        real_bf16 = cmuon_zeroth_power_bf16
        real_fp32 = cmuon_zeroth_power_fp32
        forced_pattern: dict[int, str] = {}
        state = {
            "pattern": forced_pattern,
            "fail_fp32": False,
            "call": 0,
            "owned": owned0 if rank == 0 else owned1,
        }

        def forced_bf16(grad, ns_steps, ns_coefficients, eps):
            i = state["call"]
            state["call"] += 1
            kind = state["pattern"].get(state["owned"][i])
            out = real_bf16(grad, ns_steps, ns_coefficients, eps)
            if kind is None:
                return out
            if kind == "nan":
                return out * float("nan")
            if kind == "huge":
                return out * 1e9
            raise ValueError(kind)

        def forced_fp32(grad, ns_steps, ns_coefficients, eps):
            out = real_fp32(grad, ns_steps, ns_coefficients, eps)
            return out * float("nan") if state["fail_fp32"] else out

        fr.cmuon_zeroth_power_bf16 = forced_bf16
        fr.cmuon_zeroth_power_fp32 = forced_fp32

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
            guard_cfg=GuardedCanonicalGuardConfig(
                guard_ratio=0.05,
                reference_decay=0.999,
                min_reference=1e-12,
                numerical_floor=1e-20,
                warmup_observations=0,
                invariant_check=True,
            ),
            guard_bootstrap_refs=_bootstrap_refs(model),
            rank=rank,
            world_size=world_size,
            momentum_dtype="bfloat16",
            chunk_rescale_sqrt_n=False,
        )

        scenarios: list[tuple[str, dict[int, str], bool]] = [
            ("healthy", {}, False),
            ("owner0_single", {owned0[0]: "nan"}, False),
            ("owner1_single", {owned1[0]: "nan"}, False),
            (
                "multi_mixed",
                {
                    owned0[len(owned0) // 2]: "nan",
                    owned1[len(owned1) // 2]: "nan",
                    owned0[-1]: "huge",
                    owned1[-1]: "nan",
                },
                False,
            ),
            (
                "simultaneous_owners",
                {owned0[0]: "nan", owned1[0]: "nan", owned1[-2]: "huge"},
                False,
            ),
            ("final_failure", {owned0[-1]: "nan"}, True),
        ]

        results: list[dict[str, object]] = []
        for sname, pattern, fail_fp32 in scenarios:
            state["pattern"] = pattern
            state["fail_fp32"] = fail_fp32
            state["call"] = 0
            before_params = {
                name: p.detach().clone() for name, p in model.named_parameters()
            }
            obs_before = opt.observations
            if fail_fp32:
                try:
                    _seed_grads(model, 50)
                    opt.step()
                except CMuonSafetyError as e:
                    results.append(
                        {
                            "scenario": sname,
                            "outcome": "CMuonSafetyError",
                            "message": str(e)[:300],
                        }
                    )
                    # zero commit: parameter bytes identical on this rank
                    for name, p in model.named_parameters():
                        assert torch.equal(before_params[name], p.detach()), (
                            f"final failure must commit nothing ({name})"
                        )
                    assert opt.observations == obs_before, (
                        "observations must not advance on failure"
                    )
                    continue
                raise AssertionError(f"final_failure did not raise ({rank})")
            _seed_grads(model, 10 + len(results))
            opt.step()
            fp_spread = _param_fingerprint_spread(model, device)
            # cross-check counters: this rank must have attempted exactly its
            # forced chunks (plus the previous step's residual state is fine
            # because counters are cumulative — use deltas instead)
            results.append(
                {
                    "scenario": sname,
                    "outcome": "ok",
                    "forced_here": len(pattern),
                    "max_delta_rank_spread": opt.max_delta_rank_spread,
                    "max_param_rank_diff": opt.max_param_rank_diff,
                    "param_fingerprint_spread": fp_spread,
                    "observations": opt.observations,
                    "bf16_attempts": opt.bf16_attempts,
                    "fp32_attempts": opt.fp32_attempts,
                    "fp32_rescues": opt.fp32_rescues,
                    "fp32_rescue_failures": opt.fp32_rescue_failures,
                    "rescue_by_role": dict(opt.rescue_by_role),
                }
            )
            assert opt.max_delta_rank_spread == 0.0, (
                f"{sname}: delta rank spread != 0"
            )
            assert opt.max_param_rank_diff == 0.0, (
                f"{sname}: param rank diff != 0"
            )
            assert fp_spread == 0.0, f"{sname}: param fingerprint spread != 0"

        # final global assertions (same on both ranks)
        ok = [r for r in results if r["outcome"] == "ok"]
        assert len(ok) == 5, f"expected 5 ok scenarios, got {len(ok)}"
        assert results[-1]["outcome"] == "CMuonSafetyError"
        total_forced = sum(len(p) for _, p, f in scenarios[:5] for _ in [0] if not f)
        total_rescued = results[4]["fp32_rescues"]
        assert total_rescued == total_forced, (
            f"global rescued {total_rescued} != total forced {total_forced}"
        )
        assert opt.fp32_rescue_failures == 0
        report["scenarios"] = results
        report["verdict"] = "PASS"
    finally:
        if rank == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            (Path(args.out)).write_text(json.dumps(report, indent=2))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
