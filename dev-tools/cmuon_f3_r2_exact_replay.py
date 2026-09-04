"""F3 R2 exact-input replay (F3 spec sections 6 + 14).

Re-runs the R2 production hard-fail input (the F2 minimal capsule from
obs-611-rank0, slot_02 attention content_gate chunk 0) through the exact
owner-rank rescue path, repeatedly, on the tree it is run from:

  * BF16 production kernel (addmm)   >= --bf16-repeats (default 20)
      -> records the rms distribution (HCU addmm nondeterminism is
         recorded, not fixed; every repeat must still be a hard-fail
         class: finite and above ceiling)
  * BF16 deterministic test-only oracle (matmul + mul + add, NO addmm)
      >= --oracle-repeats (default 3) -> must be BIT EXACT across repeats
      (S18: bf16 matmul is bit-deterministic; only the fused addmm kernel
         is not). TEST ONLY: never imported by production code.
  * FP32 production kernel (addmm, fp32) >= --fp32-repeats (default 3)
      -> must be BIT EXACT across repeats (S18: FP32 addmm deterministic)
      and the read-back rms must equal the capsule's fp32_delta_rms

Verdicts reported for the same read-back values:
  * F2 reference verdict (bb41292 step() predicate):
      nonfinite OR rms < floor OR rms > ceiling  -> HARD FAIL (below_floor)
  * F3 verdict (F3 step() predicate):
      nonfinite OR rms > ceiling -> HARD FAIL; otherwise ACCEPT, with
      rms < floor  -> below_floor_soft_rescue / zero_delta_soft_rescue

The staged delta (F3 accept path) is the ORIGINAL FP32 delta with the
existing single BF16 rounding; its sha256 is reported.

Usage (from a tree root that has src/ and dev-tools/):
  python dev-tools/cmuon_f3_r2_exact_replay.py \
      --capsule <capsule-dir> [--bf16-repeats 20] [--oracle-repeats 3] \
      [--fp32-repeats 3] [--device cuda] [--output out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def _find_repo_src() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / "src").is_dir():
            return base / "src"
    raise SystemExit("cannot find repo src/ relative to dev-tools/")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_sha256(t) -> str:
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def ns_deterministic_bf16_oracle(grad, ns_steps, coefficients, eps):
    """TEST ONLY: the quintic NS in BF16 without the fused addmm kernel
    (matmul + elementwise mul/add), per S18 bit-deterministic on HCU."""
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
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--bf16-repeats", type=int, default=20)
    parser.add_argument("--oracle-repeats", type=int, default=3)
    parser.add_argument("--fp32-repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(_find_repo_src()))
    from safetensors.torch import load_file

    from sakuramoon.optim.cmuon import (
        cmuon_moonlight_alpha,
        cmuon_zeroth_power_bf16,
        cmuon_zeroth_power_fp32,
    )
    from sakuramoon.optim.cmuon_hardfail import classify_fp32_verdict
    from sakuramoon.optim.fp32_rescue import _RESCUE_SANITY_LOW

    capsule = Path(args.capsule)
    meta = json.loads((capsule / "metadata.json").read_text())
    input_path = capsule / "input.safetensors"
    sha = _sha256_file(input_path)
    assert sha == meta["tensor_sha256"], (
        f"input sha mismatch: {sha} != {meta['tensor_sha256']}"
    )
    loaded = load_file(input_path, device=args.device)
    x = loaded["input"] if isinstance(loaded, dict) else loaded
    assert x.dtype == torch.bfloat16 and tuple(x.shape) == tuple(meta["shape"])
    # input fidelity re-checks (the sha above is the authoritative bits
    # check; the rms re-computation uses an fp32 chain while production
    # read it back through the bf16 chain, so allow a small relative band)
    assert bool(torch.isfinite(x.float()).all()), "input must be finite"
    in_rms = float(x.float().pow(2).mean().sqrt().item())
    assert abs(in_rms - meta["input_rms"]) / meta["input_rms"] < 1e-6

    lr = meta["lr"]
    ns_steps = meta["ns_steps"]
    coeffs = tuple(meta["ns_coefficients"])
    eps = meta["eps"]
    target = meta["target_delta_rms"]
    floor = meta["rescue_floor"]
    ceiling = meta["ceiling"]
    alpha_meta = meta["alpha"]
    alpha = cmuon_moonlight_alpha(x.shape[0], x.shape[1], lr, 1)
    assert abs(alpha - alpha_meta) < 1e-18, (
        f"alpha mismatch: recomputed {alpha} vs metadata {alpha_meta}"
    )
    # floor/ceiling/target fidelity vs the capsule constants
    assert abs(target - 0.2 * lr) < 1e-20
    assert abs(ceiling - 10.0 * target) < 1e-20
    assert abs(floor - _RESCUE_SANITY_LOW * target) < 1e-20, (
        "rescue_floor numeric value must be unchanged (0.05 x target)"
    )

    report: dict[str, object] = {
        "capsule": str(capsule),
        "fqn": meta["fqn"],
        "shape": list(x.shape),
        "input_sha256": sha,
        "input_rms": in_rms,
        "params": {
            "lr": lr,
            "alpha": alpha,
            "ns_steps": ns_steps,
            "ns_coefficients": list(coeffs),
            "eps": eps,
            "target_delta_rms": target,
            "rescue_floor": floor,
            "ceiling": ceiling,
        },
        "tree": str(_find_repo_src()),
    }

    # ---- BF16 production kernel (addmm): nondeterminism recorded ---------
    bf16_rms: list[float] = []
    bf16_finite: list[bool] = []
    t0 = time.time()
    for _ in range(args.bf16_repeats):
        ns = cmuon_zeroth_power_bf16(x, ns_steps, coeffs, eps)
        d = (-alpha) * ns
        bf16_rms.append(float(d.float().pow(2).mean().sqrt().item()))
        bf16_finite.append(bool(torch.isfinite(d.float()).all()))
    bf16_all_hard = all(
        f and r > ceiling for f, r in zip(bf16_finite, bf16_rms)
    )
    report["bf16_production_addmm"] = {
        "repeats": args.bf16_repeats,
        "rms": bf16_rms,
        "rms_min": min(bf16_rms),
        "rms_max": max(bf16_rms),
        "rms_median": statistics.median(bf16_rms),
        "all_finite": all(bf16_finite),
        "all_above_ceiling": bf16_all_hard,
        "seconds": round(time.time() - t0, 2),
    }

    # ---- BF16 deterministic oracle (matmul+mul+add): bit exact ----------
    oracle_shas: list[str] = []
    oracle_rms: list[float] = []
    t0 = time.time()
    for _ in range(args.oracle_repeats):
        ns = ns_deterministic_bf16_oracle(x, ns_steps, coeffs, eps)
        d = (-alpha) * ns
        oracle_shas.append(_tensor_sha256(d.cpu()))
        oracle_rms.append(float(d.float().pow(2).mean().sqrt().item()))
    report["bf16_deterministic_oracle"] = {
        "repeats": args.oracle_repeats,
        "rms": oracle_rms,
        "bit_exact": len(set(oracle_shas)) == 1,
        "delta_sha256": oracle_shas[0] if oracle_shas else None,
        "note": "TEST ONLY oracle (no addmm); never used by production",
        "seconds": round(time.time() - t0, 2),
    }

    # ---- FP32 production kernel (addmm, fp32): bit exact -----------------
    fp32_shas: list[str] = []
    fp32_rms: list[float] = []
    fp32_finite: list[bool] = []
    t0 = time.time()
    staged_sha: str | None = None
    staged_rms: float | None = None
    for _ in range(args.fp32_repeats):
        ns32 = cmuon_zeroth_power_fp32(x.float(), ns_steps, coeffs, eps)
        d32 = (-alpha) * ns32
        fp32_shas.append(_tensor_sha256(d32.cpu()))
        fp32_rms.append(float(d32.pow(2).mean().sqrt().item()))
        fp32_finite.append(bool(torch.isfinite(d32).all()))
        staged = d32.bfloat16().contiguous()
        staged_sha = _tensor_sha256(staged.cpu())
        staged_rms = float(staged.float().pow(2).mean().sqrt().item())
    fp32_exact = len(set(fp32_shas)) == 1
    report["fp32_production"] = {
        "repeats": args.fp32_repeats,
        "rms": fp32_rms,
        "rms_all_equal": len(set(fp32_rms)) == 1,
        "bit_exact": fp32_exact,
        "finite": all(fp32_finite),
        "seconds": round(time.time() - t0, 2),
    }
    report["staged_delta_bf16"] = {
        "sha256": staged_sha,
        "rms": staged_rms,
        "note": "original fp32 delta, single bf16 rounding (existing path)",
    }

    # ---- verdicts on the (identical) FP32 read-back ----------------------
    rms32 = fp32_rms[-1]
    finite32 = fp32_finite[-1]
    # F2 reference (bb41292): the tree classifier is the ground truth on
    # the F2 tree; on the F3 tree its below_floor branch is unreachable
    # from step() but the function itself is unchanged, so it still
    # reports what the F2 verdict would have been.
    f2_reason = classify_fp32_verdict(finite32, rms32, floor, ceiling)
    f2_outcome = "HARD_FAIL" if f2_reason is not None else "SUCCESSFUL_RESCUE"
    # F3 predicate (mirrors the F3 step() verdict block exactly):
    if not finite32 or rms32 > ceiling:
        f3_outcome = "HARD_FAIL"
        f3_reason = classify_fp32_verdict(finite32, rms32, floor, ceiling)
    elif rms32 < floor:
        f3_outcome = "SUCCESSFUL_RESCUE"
        f3_reason = (
            "zero_delta_soft_rescue" if rms32 == 0.0 else "below_floor_soft_rescue"
        )
    else:
        f3_outcome = "SUCCESSFUL_RESCUE"
        f3_reason = "normal_band"
    report["verdict_f2_reference"] = {
        "outcome": f2_outcome,
        "reason": f2_reason,
        "predicate": "not finite OR rms < floor OR rms > ceiling",
    }
    report["verdict_f3"] = {
        "outcome": f3_outcome,
        "reason": f3_reason,
        "predicate": "not finite OR rms > ceiling (floor is diagnostic)",
        "fp32_delta_rms_readback": rms32,
        "capsule_fp32_delta_rms": meta["fp32_delta_rms"],
        "readback_matches_capsule": abs(rms32 - meta["fp32_delta_rms"]) < 1e-12,
    }
    report["f3_semantic_change"] = (
        f2_outcome == "HARD_FAIL" and f3_outcome == "SUCCESSFUL_RESCUE"
    )
    report["capsule_class_recheck"] = {
        "bf16_above_ceiling": bf16_all_hard,
        "fp32_finite_below_floor": finite32 and rms32 < floor,
    }

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
