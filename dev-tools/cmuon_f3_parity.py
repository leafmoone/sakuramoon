"""F3 parity runner (F3 spec sections 8-11).

Runs one parity scenario from the tree it is invoked from and dumps a
state digest as JSON. Run once per tree (bb41292 / F3), then compare the
digests offline.

Modes:
  safe-real      10 steps, REAL BF16 NS kernel, no forcing. Fidelity leg
                 only (NOT cross-tree bit-compared: per S18 the HCU bf16
                 addmm kernel is nondeterministic across independent
                 calls, so two separate processes running identical code
                 are not guaranteed identical bits). Asserts: no FP32
                 rescue fired, all steps committed, parameters finite.
  safe-oracle    10 steps, TEST-ONLY deterministic BF16 NS oracle
                 (matmul + mul + add, NO addmm — bit-deterministic per
                 S16/S18), no other forcing. Cross-tree BIT EXACT
                 expected: proves the F3 patch leaves the BF16-safe path
                 untouched.
  normal         3 steps, deterministic BF16 failure stand-in (constant,
                 above ceiling) + REAL FP32 NS (bit-deterministic).
                 Cross-tree BIT EXACT expected: the normal rescue path is
                 unchanged by F3.
  normal-constant  deterministic BF16 failure + deterministic IN-BAND FP32
                 constant (rms ~ target). Same kernel costs as below-floor
                 with no F3 diagnostic branch taken: the perf comparison
                 pair for the soft-rescue overhead (spec section 17).
  below-floor    1 step, deterministic BF16 failure + deterministic
                 below-floor FP32 constant. F2 tree -> CMuonSafetyError
                 + zero commit; F3 tree -> SUCCESS + soft counter +
                 committed reference delta (the intended semantic
                 difference, section 10).
  hardfail-nonfinite  1 step, BF16 failure + nonfinite FP32. Both trees:
                 CMuonSafetyError + zero commit (preserved).
  hardfail-ceiling    1 step, BF16 failure + above-ceiling FP32. Both
                 trees: CMuonSafetyError + zero commit (preserved).

Usage:
  python dev-tools/cmuon_f3_parity.py --mode safe-oracle --steps 10 \
      --output /tmp/f3-parity-safe-oracle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn


def _find_repo_src() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / "src").is_dir():
            return base / "src"
    raise SystemExit("cannot find repo src/ relative to dev-tools/")


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tensor_sha(t: torch.Tensor) -> str:
    """Raw-bytes sha256 of a tensor (any dtype, incl. torchao 8-bit
    state subclasses and 0-d scalars). Reads the contiguous storage
    bytes directly — no dtype view, so tensor subclasses never
    dispatch."""
    import ctypes

    t = t.detach().contiguous().flatten().cpu()
    nbytes = t.untyped_storage().nbytes()
    return _sha_bytes(ctypes.string_at(t.data_ptr(), nbytes))


def _tsha_map(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    return {k: _tensor_sha(v) for k, v in tensors.items()}


# ---------------------------------------------------------------------------
# mock model (self-contained; identical to the unit/2-rank harnesses)
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.q_proj = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.k_proj = nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.v_proj = nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.content_gate = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.out_proj = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.mlp = nn.Module()
        self.mlp.in_proj = nn.Linear(
            hidden, 2 * inter, bias=False, dtype=torch.bfloat16
        )
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)
        self.attention_norm = nn.Parameter(torch.ones(hidden, dtype=torch.float32))
        self.mlp_norm = nn.Parameter(torch.ones(hidden, dtype=torch.float32))


class _DiT(nn.Module):
    def __init__(self, hidden: int, inter: int, n_blocks: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            128, hidden, bias=False, dtype=torch.bfloat16
        )
        self.blocks = nn.ModuleDict(
            {f"slot_{i:02d}": _Block(hidden, inter) for i in range(n_blocks)}
        )
        self.modality_image = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))


class _Composite(nn.Module):
    def __init__(self, dit: _DiT) -> None:
        super().__init__()
        self.dit = dit


def _seed_grads(model: nn.Module, seed: int) -> None:
    g = torch.Generator(
        device=model.dit.input_projection.weight.device
    ).manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(
                *p.shape, generator=g, dtype=torch.float32, device=p.device
            ).to(p.dtype)


def ns_deterministic_bf16_oracle(grad, ns_steps, coefficients, eps):
    """TEST ONLY: quintic NS in BF16 without the fused addmm kernel
    (matmul + elementwise mul/add) — bit-deterministic on HCU (S16/S18).
    Never used by production code."""
    a, b, c = coefficients
    ortho = grad.bfloat16()
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T
    ortho = ortho / ortho.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        gram_update = torch.mul(gram, b) + torch.mul(gram @ gram, c)
        ortho = torch.mul(ortho, a) + (gram_update @ ortho)
    if transposed:
        ortho = ortho.T
    return ortho


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "safe-real",
            "safe-oracle",
            "normal",
            "normal-constant",
            "below-floor",
            "hardfail-nonfinite",
            "hardfail-ceiling",
        ],
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(_find_repo_src()))
    import sakuramoon.optim.fp32_rescue as fr
    from sakuramoon.optim.cmuon import (
        cmuon_moonlight_alpha,
        resolve_ns_map,
        route_cmuon_parameters,
    )
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
    from sakuramoon.optim.fp32_rescue import (
        _RESCUE_SANITY_LOW,
        build_fp32_rescue,
    )
    from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig

    NS4 = resolve_ns_map(None, 4)
    LR = 0.00015625
    TARGET = 0.2 * LR
    FLOOR = _RESCUE_SANITY_LOW * TARGET
    CEILING = 10.0 * TARGET
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    torch.manual_seed(20260904)
    model = _Composite(_DiT(256, 512, 2)).to(device)
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
                sl = [slice(None)] * g.ndim
                sl[spec.chunk_dim] = slice(start, start + chunk_size)
                sig = g[tuple(sl)].float().pow(2).mean().sqrt().item()
            refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)

    # stable FQN labels for state keyed by parameter objects
    param_names = {id(p): n for n, p in model.named_parameters()}

    def _adamw_digest(opt) -> dict[str, str]:
        out: dict[str, str] = {}
        for p, state in sorted(
            opt.optimizer.state.items(), key=lambda kv: param_names.get(id(kv[0]), "?")
        ):
            name = param_names.get(id(p), f"param{id(p)}")
            for sname, sval in state.items():
                if isinstance(sval, torch.Tensor):
                    out[f"adamw[{name}][{sname}]"] = _tensor_sha(sval)
        return out

    def _digest(tag: str) -> dict[str, object]:
        return {
            "tag": tag,
            "params": _tsha_map({n: p.detach() for n, p in model.named_parameters()}),
            "momenta": _tsha_map(
                {f"buf[{i}]": b for i, b in enumerate(opt._momenta.values())}
            ),
            "refs": _sha_bytes(
                json.dumps({k: refs[k] for k in sorted(refs)}, sort_keys=True).encode()
            ),
            "adamw_state": _adamw_digest(opt),
            "sr_rng_state": _tensor_sha(opt.sr_rng.state),
            "counters": _counters(opt),
        }

    def _counters(opt) -> dict[str, object]:
        c = {
            "bf16_attempts": opt.bf16_attempts,
            "bf16_safety_failures": opt.bf16_safety_failures,
            "fp32_attempts": opt.fp32_attempts,
            "fp32_rescues": opt.fp32_rescues,
            "fp32_rescue_failures": opt.fp32_rescue_failures,
            "rescue_by_role": dict(opt.rescue_by_role),
            "observations": opt.observations,
        }
        # F3 process-local counters (absent on the bb41292 tree)
        if hasattr(opt, "fp32_low_delta_rescues"):
            c["fp32_low_delta_rescues"] = opt.fp32_low_delta_rescues
            c["fp32_low_delta_by_role"] = dict(opt.fp32_low_delta_by_role)
        return c

    # ---- forcing (applied identically on both trees) ---------------------
    if args.mode in ("safe-oracle",):
        fr.cmuon_zeroth_power_bf16 = ns_deterministic_bf16_oracle
    elif args.mode in (
        "normal",
        "normal-constant",
        "below-floor",
        "hardfail-nonfinite",
        "hardfail-ceiling",
    ):

        def evil_bf16(grad, ns_steps, ns_coefficients, eps):
            return torch.full(
                tuple(grad.shape), 1e9, dtype=torch.bfloat16, device=grad.device
            )

        fr.cmuon_zeroth_power_bf16 = evil_bf16

        if args.mode == "normal-constant":

            def band_fp32(grad, ns_steps, ns_coefficients, eps):
                alpha = cmuon_moonlight_alpha(
                    grad.shape[0], grad.shape[1], LR, 1
                )
                v = TARGET / alpha
                return torch.full(
                    tuple(grad.shape), v, dtype=torch.float32, device=grad.device
                )

            fr.cmuon_zeroth_power_fp32 = band_fp32
        elif args.mode == "below-floor":

            def soft_fp32(grad, ns_steps, ns_coefficients, eps):
                alpha = cmuon_moonlight_alpha(
                    grad.shape[0], grad.shape[1], LR, 1
                )
                v = (FLOOR / 4.0) / alpha
                return torch.full(
                    tuple(grad.shape), v, dtype=torch.float32, device=grad.device
                )

            fr.cmuon_zeroth_power_fp32 = soft_fp32
        elif args.mode == "hardfail-nonfinite":
            fr.cmuon_zeroth_power_fp32 = lambda grad, *a, **k: torch.full_like(
                grad, float("inf")
            )
        elif args.mode == "hardfail-ceiling":

            def ceil_fp32(grad, ns_steps, ns_coefficients, eps):
                alpha = cmuon_moonlight_alpha(
                    grad.shape[0], grad.shape[1], LR, 1
                )
                v = (4.0 * CEILING) / alpha
                return torch.full(
                    tuple(grad.shape), v, dtype=torch.float32, device=grad.device
                )

            fr.cmuon_zeroth_power_fp32 = ceil_fp32

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
        guard_bootstrap_refs=refs,
        rank=0,
        world_size=1,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
    )

    report: dict[str, object] = {
        "mode": args.mode,
        "steps": args.steps,
        "tree": str(_find_repo_src()),
        "per_step": [],
    }
    t0 = time.time()
    outcome = "SUCCESS"
    try:
        for i in range(args.steps):
            _seed_grads(model, 700 + i)
            try:
                opt.step()
            except CMuonSafetyError as e:
                outcome = "CMuonSafetyError"
                report["exception"] = {
                    "step": i,
                    "message": str(e),
                }
                break
            report["per_step"].append(_digest(f"step{i}"))
        if args.mode == "safe-real":
            assert opt.fp32_attempts == 0, (
                "safe-real must not trigger any FP32 rescue, "
                f"got {opt.fp32_attempts}"
            )
            for p in model.parameters():
                assert torch.isfinite(p.float()).all()
        report["final"] = _digest("final")
    finally:
        report["outcome"] = outcome
        report["seconds"] = round(time.time() - t0, 2)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({"mode": args.mode, "outcome": outcome,
                      "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
