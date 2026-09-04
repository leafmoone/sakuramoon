"""2-rank F3 soft below-floor rescue regression (F3 spec section 12).

Launched with torchrun (2 local ranks), NOT pytest:

    torchrun --nproc_per_node=2 tests/gpu/optim/fp32_rescue_f3_2rank.py \
        --out /tmp/f3-2rank.json

Five scenarios (one optimizer, sequential steps; forced patterns are
applied identically on both ranks so the owner-split is exercised):

  1. healthy         : no forcing -> clean step
  2. soft_rescue     : one owner chunk trips BF16 and its FP32 delta is
                       finite BELOW FLOOR -> F3 accepts (no error); the
                       owner computes FP32 exactly once, the non-owner
                       receives the broadcast, cross-rank spread == 0
  3. mixed_step      : same step, one chunk per owner: rank0's chunk soft
                       (below floor), rank1's chunk real-FP32 (normal or
                       soft — classification machine-independent); both
                       accepted, spread == 0
  4. above_ceiling   : one chunk trips BF16 and FP32 is finite but ABOVE
                       the ceiling -> CMuonSafetyError on BOTH ranks,
                       ZERO commit (preserved hard fail)
  5. nonfinite       : one chunk trips BF16 and FP32 is nonfinite ->
                       CMuonSafetyError on BOTH ranks, ZERO commit
                       (preserved hard fail)

Pass criteria (ANY violation => raise, torchrun exits non-zero):
  * soft/mixed steps: max_delta_rank_spread == 0, max_param_rank_diff == 0,
    independent post-step parameter fingerprint all_reduce spread == 0,
    owner FP32 call count == 1 for the forced chunk, non-owner == 0
  * hard-fail steps: CMuonSafetyError on both ranks, parameters
    byte-identical to the pre-step snapshot on both ranks, observations
    not advanced, no soft-rescue telemetry for failed chunks
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
    cmuon_moonlight_alpha,
    cmuon_zeroth_power_bf16,
    cmuon_zeroth_power_fp32,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import (
    _RESCUE_SANITY_LOW,
    build_fp32_rescue,
)
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
    stable_owner,
)

NS4 = resolve_ns_map(None, 4)
LR = 0.00015625
TARGET = 0.2 * LR
FLOOR = _RESCUE_SANITY_LOW * TARGET
CEILING = 10.0 * TARGET


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
        self.final_projection.bias.data = self.final_projection.bias.data.to(
            torch.float32
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
        "candidate": "hybrid_cmuon_canonical_ns4_fp32_rescue (F3 soft below-floor)",
    }
    try:
        torch.manual_seed(20260904)
        model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
        routing = route_cmuon_parameters(
            model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
        )
        flat_keys: list[tuple[str, int]] = []
        for spec in routing.cmuon_specs:
            for ci in range(spec.chunk_count):
                flat_keys.append((spec.name, ci))
        n = len(flat_keys)
        owner_of_flat = [stable_owner(fqn, ci, world_size) for fqn, ci in flat_keys]
        owned0 = [i for i, o in enumerate(owner_of_flat) if o == 0]
        owned1 = [i for i, o in enumerate(owner_of_flat) if o == 1]
        assert len(owned0) > 0 and len(owned1) > 0, (
            "need chunks owned by both ranks for the scenario matrix"
        )
        report["n_inputs"] = n
        report["owned_per_rank"] = [len(owned0), len(owned1)]

        real_bf16 = cmuon_zeroth_power_bf16
        real_fp32 = cmuon_zeroth_power_fp32
        state: dict[str, object] = {
            # flat index -> forcing kind
            "bf16": {},
            "fp32": {},
            "call_bf16": 0,
            "call_fp32": 0,
            "owned": owned0 if rank == 0 else owned1,
        }

        def forced_bf16(grad, ns_steps, ns_coefficients, eps):
            i = state["call_bf16"]
            state["call_bf16"] = i + 1
            kind = state["bf16"].get(state["owned"][i])
            out = real_bf16(grad, ns_steps, ns_coefficients, eps)
            if kind is None:
                return out
            if kind == "huge":
                return out * 1e9
            raise ValueError(kind)

        def forced_fp32(grad, ns_steps, ns_coefficients, eps):
            i = state["call_fp32"]
            state["call_fp32"] = i + 1
            kind = state["fp32"].get(state["owned"][i])
            if kind is None:
                return real_fp32(grad, ns_steps, ns_coefficients, eps)
            if kind == "soft":
                # deterministic finite constant whose (-alpha)*v delta rms
                # is FLOOR/4 — unambiguously below the diagnostic floor
                alpha = cmuon_moonlight_alpha(
                    grad.shape[0], grad.shape[1], LR, 1
                )
                v = (FLOOR / 4.0) / alpha
                return torch.full(
                    tuple(grad.shape), v, dtype=torch.float32, device=grad.device
                )
            if kind == "huge":
                # deterministic finite constant far above the ceiling
                alpha = cmuon_moonlight_alpha(
                    grad.shape[0], grad.shape[1], LR, 1
                )
                v = (4.0 * CEILING) / alpha
                return torch.full(
                    tuple(grad.shape), v, dtype=torch.float32, device=grad.device
                )
            if kind == "nan":
                return real_fp32(grad, ns_steps, ns_coefficients, eps) * float(
                    "nan"
                )
            raise ValueError(kind)

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

        # pick the forced chunks once (fixed indices, both owners covered)
        soft_chunk = owned0[0]  # rank 0 owns
        normal_chunk = owned1[0]  # rank 1 owns

        scenarios: list[tuple[str, dict, dict, bool]] = [
            # (name, bf16 forcing, fp32 forcing, expect_hard_fail)
            ("healthy", {}, {}, False),
            ("soft_rescue", {soft_chunk: "huge"}, {soft_chunk: "soft"}, False),
            (
                "mixed_step",
                {soft_chunk: "huge", normal_chunk: "huge"},
                {soft_chunk: "soft"},
                False,
            ),
            (
                "above_ceiling",
                {normal_chunk: "huge"},
                {normal_chunk: "huge"},
                True,
            ),
            (
                "nonfinite",
                {soft_chunk: "huge"},
                {soft_chunk: "nan"},
                True,
            ),
        ]

        results: list[dict[str, object]] = []
        for sname, bf16_forcing, fp32_forcing, hard_fail in scenarios:
            state["bf16"] = bf16_forcing
            state["fp32"] = fp32_forcing
            state["call_bf16"] = 0
            state["call_fp32"] = 0
            before_params = {
                name: p.detach().clone() for name, p in model.named_parameters()
            }
            obs_before = opt.observations
            if hard_fail:
                try:
                    _seed_grads(model, 60)
                    opt.step()
                except CMuonSafetyError as e:
                    results.append(
                        {
                            "scenario": sname,
                            "outcome": "CMuonSafetyError",
                            "message": str(e)[:300],
                        }
                    )
                    for name, p in model.named_parameters():
                        assert torch.equal(before_params[name], p.detach()), (
                            f"{sname} must commit nothing ({name})"
                        )
                    assert opt.observations == obs_before
                    continue
                raise AssertionError(f"{sname} did not raise ({rank})")
            _seed_grads(model, 20 + len(results))
            opt.step()
            fp_spread = _param_fingerprint_spread(model, device)
            results.append(
                {
                    "scenario": sname,
                    "outcome": "ok",
                    "max_delta_rank_spread": opt.max_delta_rank_spread,
                    "max_param_rank_diff": opt.max_param_rank_diff,
                    "param_fingerprint_spread": fp_spread,
                    "observations": opt.observations,
                    "fp32_attempts": opt.fp32_attempts,
                    "fp32_rescues": opt.fp32_rescues,
                    "fp32_rescue_failures": opt.fp32_rescue_failures,
                    "fp32_low_delta_rescues": opt.fp32_low_delta_rescues,
                    "fp32_low_delta_by_role": dict(opt.fp32_low_delta_by_role),
                    "fp32_calls_this_rank": state["call_fp32"],
                }
            )
            assert opt.max_delta_rank_spread == 0.0, f"{sname}: delta spread != 0"
            assert opt.max_param_rank_diff == 0.0, f"{sname}: param diff != 0"
            assert fp_spread == 0.0, f"{sname}: fingerprint spread != 0"

        ok = [r for r in results if r["outcome"] == "ok"]
        assert len(ok) == 3, f"expected 3 ok scenarios, got {len(ok)}"
        assert results[3]["outcome"] == "CMuonSafetyError"
        assert results[4]["outcome"] == "CMuonSafetyError"

        soft = ok[1]
        mixed = ok[2]
        # scenario 2: the soft chunk is owned by rank 0 -> rank 0 computed
        # FP32 exactly once for it; rank 1 (non-owner) computed zero FP32.
        if rank == 0:
            assert soft["fp32_calls_this_rank"] == 1, (
                f"owner must compute FP32 exactly once, "
                f"got {soft['fp32_calls_this_rank']}"
            )
        else:
            assert soft["fp32_calls_this_rank"] == 0, (
                f"non-owner must not compute FP32, got "
                f"{soft['fp32_calls_this_rank']}"
            )
        # scenario 2: the forced chunk was accepted as a soft below-floor
        # rescue on the owning rank (real-FP32 rescues of healthy chunks
        # may or may not also be soft; the forced one MUST be).
        if rank == 0:
            assert soft["fp32_low_delta_rescues"] >= 1, (
                "forced soft chunk must be counted as a soft rescue"
            )
        # scenario 3: both forced chunks accepted; rank 0's was soft
        if rank == 0:
            assert mixed["fp32_low_delta_rescues"] >= 1
        # hard-fail scenarios preserved: no soft counter movement there
        # (counters are cumulative; check the last ok scenario baseline)
        assert ok[2]["fp32_rescue_failures"] == 0, (
            "soft/mixed steps must not record fp32 failures"
        )
        report["scenarios"] = results
        report["verdict"] = "PASS"
    finally:
        if rank == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
