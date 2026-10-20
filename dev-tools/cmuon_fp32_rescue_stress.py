#!/usr/bin/env python3
"""Synthetic NS stress harness for the CMuon FP32-rescue safety band (F1).

Developer-only tool. The 112126/112106 hard failures predates the exact-input
artifact (its NS input was NOT saved), so exact replay of THAT event is
impossible. This harness runs the PRODUCTION NS functions
(``cmuon_zeroth_power_bf16`` / ``cmuon_zeroth_power_fp32``) on a small set of
representative inputs at the real production shape and parameters, to map
which input classes can trip each side of the safety band:

  * BF16 side:  nonfinite / delta_rms > ceiling  (what the BF16 attempt saw)
  * FP32 side:  nonfinite / < rescue_floor / > ceiling / ok
    (the category the production dump could NOT tell us)

Caveat (deliberate): the NS pass normalizes its input by its Frobenius norm,
so the RMS scale scan is DIAGNOSTIC ONLY — scale correlates with input class
behavior, it is not a causal dial.

Defaults are the production values of the failing event (forensic record):

  shape [2560,2560], lr 1.5624999650754035e-4, role attention_content_gate,
  ns_steps 4 (production [optimizer.cmuon_ns]), coefficients (3.4445,
  -4.7750, 2.0315), eps 1e-7, floor 0.05*(0.2*lr), ceiling 10*(0.2*lr).

Usage (HCU recommended; CPU works for small --shape smoke runs):

  python dev-tools/cmuon_fp32_rescue_stress.py \
      --device cuda --output-dir reports

``--with-trace`` adds the per-iteration NS replay trace (run on the same
device as the NS input — a CPU replay of a 2560x2560 BF16 matrix takes
several minutes per trace and uses a different kernel than production, so
it is opt-in and device-local).

Outputs: <output-dir>/cmuon-fp32-rescue-stress.json and .md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


def _find_repo_src() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "src"
    if (candidate / "sakuramoon").is_dir():
        return candidate
    raise SystemExit(f"cannot locate src/ next to {here.parent}")


def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt().item())


def _scale_to_rms(t: torch.Tensor, rms: float) -> torch.Tensor:
    cur = _rms(t)
    if cur == 0.0:
        return t
    return t * (rms / cur)


def build_family(family: str, shape: tuple[int, int], seed: int) -> torch.Tensor:
    """FP32 prototype matrix for one family (later scaled + cast to BF16)."""
    rows, cols = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    randn = lambda: torch.randn(rows, cols, generator=g)
    if family == "gaussian":
        return randn()
    if family == "constant":
        return torch.full((rows, cols), 0.1)
    if family == "sparse":
        mask = torch.rand(rows, cols, generator=g) < 0.001
        return randn() * mask
    if family in ("rank-1", "rank-8", "rank-64"):
        r = int(family.split("-")[1])
        out = torch.zeros(rows, cols)
        for _ in range(r):
            u = torch.randn(rows, generator=g)
            v = torch.randn(cols, generator=g)
            out += torch.outer(u, v)
        return out
    if family == "dup-rows":
        half = rows // 2
        base = torch.randn(half, cols, generator=g)
        return torch.cat([base, base], dim=0)
    if family == "dup-cols":
        half = cols // 2
        base = torch.randn(rows, half, generator=g)
        return torch.cat([base, base], dim=1)
    if family == "anisotropic":
        # Geometric singular-value decay 1.0 -> 1e-5 (top-1 dominant; the
        # D1 danger class: weak-signal, near-rank-1 energy concentration).
        # exp(linspace) instead of torch.geometric_series: the latter is not
        # available in the production torch build (2.9.0+das).
        u = torch.randn(rows, cols, generator=g)
        v = torch.randn(rows, cols, generator=g)
        q, _ = torch.linalg.qr(u)
        w, _ = torch.linalg.qr(v)
        sigmas = torch.exp(torch.linspace(0.0, math.log(1e-5), cols))
        return (q * sigmas[None, :]) @ w.T
    raise ValueError(f"unknown family {family!r}")


FAMILIES = (
    "gaussian",
    "constant",
    "sparse",
    "rank-1",
    "rank-8",
    "rank-64",
    "dup-rows",
    "dup-cols",
    "anisotropic",
)
CENTER_SCALE = 1.6460275276131142e-07  # u_t_rms of the 112126 forensic record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="2560,2560")
    parser.add_argument("--lr", type=float, default=1.5624999650754035e-04)
    parser.add_argument("--eps", type=float, default=1e-07)
    parser.add_argument("--ns-steps", type=int, default=4)
    parser.add_argument(
        "--coefficients", default="3.4445,-4.7750,2.0315"
    )
    parser.add_argument(
        "--scales",
        default=f"1e-8,{CENTER_SCALE},1e-7,1e-6",
        help="comma-separated RMS scales (diagnostic only; NS normalizes)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--with-trace",
        action="store_true",
        help="add the per-iteration NS replay trace (device-local; a CPU "
        "replay at production shape takes minutes per trace)",
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    src = _find_repo_src()
    sys.path.insert(0, str(src))
    from sakuramoon.optim.cmuon import (
        cmuon_moonlight_alpha,
        cmuon_zeroth_power_bf16,
        cmuon_zeroth_power_fp32,
    )
    from sakuramoon.optim.cmuon_hardfail import (
        classify_fp32_verdict,
    )
    from sakuramoon.optim.cmuon_ns_trace import (
        replay_result_to_json,
        trace_ns_replay,
    )
    from sakuramoon.optim.fp32_rescue import _RESCUE_SANITY_LOW

    rows, cols = (int(v) for v in args.shape.split(","))
    coefficients = tuple(float(v) for v in args.coefficients.split(","))
    scales = [float(v) for v in args.scales.split(",")]
    families = [v for v in args.families.split(",") if v]
    device = torch.device(args.device)

    alpha = cmuon_moonlight_alpha(rows, cols, args.lr, 1)
    target = 0.2 * args.lr
    ceiling = 10.0 * target
    floor = _RESCUE_SANITY_LOW * target

    started = time.time()
    results: list[dict[str, object]] = []
    for scale in scales:
        for fi, family in enumerate(families):
            proto = _scale_to_rms(build_family(family, (rows, cols), args.seed + fi), scale)
            x = proto.to(device).bfloat16()  # production chunk dtype is BF16
            bf16_runs: list[float] = []
            bf16_nonfinite = 0
            fp32_runs: list[float] = []
            fp32_nonfinite = 0
            trace_bf16 = trace_fp32 = None
            for rep in range(args.repeats):
                ns_b = cmuon_zeroth_power_bf16(x, args.ns_steps, coefficients, args.eps)
                db = (-alpha) * ns_b
                dbf = db.float()
                ok = bool(torch.isfinite(dbf).all().item())
                if not ok:
                    bf16_nonfinite += 1
                bf16_runs.append(float(dbf.pow(2).mean().sqrt().item()))
                ns_f = cmuon_zeroth_power_fp32(
                    x.float(), args.ns_steps, coefficients, args.eps
                )
                df = (-alpha) * ns_f
                okf = bool(torch.isfinite(df).all().item())
                if not okf:
                    fp32_nonfinite += 1
                fp32_runs.append(float(df.pow(2).mean().sqrt().item()))
                if rep == 0 and args.with_trace:
                    # Same device as the NS input (production kernel), NOT
                    # .cpu(): a CPU replay of a 2560x2560 BF16 matrix takes
                    # minutes per trace and uses a different kernel, so the
                    # trace is opt-in and device-local.
                    trace_bf16 = replay_result_to_json(
                        trace_ns_replay(
                            x,
                            ns_steps=args.ns_steps,
                            coefficients=coefficients,
                            eps=args.eps,
                            alpha=alpha,
                            working_dtype="bfloat16",
                        )
                    )
                    trace_fp32 = replay_result_to_json(
                        trace_ns_replay(
                            x,
                            ns_steps=args.ns_steps,
                            coefficients=coefficients,
                            eps=args.eps,
                            alpha=alpha,
                            working_dtype="float32",
                        )
                    )

            def _category(rms_list: list[float], nonfinite: int, lo: float | None, hi: float | None) -> str:
                finite = [v for v in rms_list]
                if nonfinite == len(rms_list):
                    return "nonfinite"
                if nonfinite:
                    return "nonfinite+finite"
                worst = max(finite)
                if hi is not None and worst > hi:
                    return "above_ceiling"
                if lo is not None and min(finite) < lo:
                    return "below_floor"
                return "ok"

            bf16_cat = _category(bf16_runs, bf16_nonfinite, None, ceiling)
            fp32_cat = _category(fp32_runs, fp32_nonfinite, floor, ceiling)
            results.append(
                {
                    "family": family,
                    "rms_scale": scale,
                    "repeats": args.repeats,
                    "bf16": {
                        "final_delta_rms": bf16_runs,
                        "nonfinite_repeats": bf16_nonfinite,
                        "category": bf16_cat,
                        "determinism_spread": max(bf16_runs) - min(bf16_runs),
                    },
                    "fp32": {
                        "final_delta_rms": fp32_runs,
                        "nonfinite_repeats": fp32_nonfinite,
                        "category": fp32_cat,
                        "verdict_reason": classify_fp32_verdict(
                            fp32_nonfinite < len(fp32_runs),
                            min(fp32_runs),
                            floor,
                            ceiling,
                        ),
                        "determinism_spread": max(fp32_runs) - min(fp32_runs),
                    },
                    "trace": None
                    if trace_bf16 is None or trace_fp32 is None
                    else {"bf16": trace_bf16, "fp32": trace_fp32},
                }
            )
            print(
                f"[stress] {family:12s} scale={scale:.3e} "
                f"bf16={bf16_cat:14s} fp32={fp32_cat}",
                flush=True,
            )
    elapsed = time.time() - started

    report = {
        "tool": "cmuon-fp32-rescue-stress",
        "shape": [rows, cols],
        "device": str(device),
        "lr": args.lr,
        "alpha": alpha,
        "target_delta_rms": target,
        "ceiling": ceiling,
        "rescue_floor": floor,
        "rescue_floor_constant": _RESCUE_SANITY_LOW,
        "ns_steps": args.ns_steps,
        "ns_coefficients": list(coefficients),
        "eps": args.eps,
        "role": "attention_content_gate",
        "repeats": args.repeats,
        "seed": args.seed,
        "scales": scales,
        "families": families,
        "elapsed_seconds": elapsed,
        "note": (
            "Scale scan is diagnostic only: NS normalizes by the Frobenius "
            "norm, so scale is not a causal dial. Categories describe what "
            "the PRODUCTION verdict logic would compute for each side."
        ),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cmuon-fp32-rescue-stress.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    # ---- markdown summary --------------------------------------------------
    lines = [
        "# CMuon FP32-rescue NS stress (F1 synthetic harness)",
        "",
        (
            f"- shape {rows}x{cols}, device {device}, lr {args.lr:.9e}, "
            f"alpha {alpha:.6e}, ns_steps {args.ns_steps}, eps {args.eps:g}"
        ),
        f"- target_delta_rms (0.2*lr) = {target:.6e}",
        f"- ceiling (10x) = {ceiling:.6e}",
        f"- rescue_floor (0.05x, constant {_RESCUE_SANITY_LOW}) = {floor:.6e}",
        f"- repeats {args.repeats}, elapsed {elapsed:.1f}s",
        "",
        "| family | rms scale | BF16 category | BF16 delta_rms (repeats) | FP32 category | FP32 delta_rms (repeats) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        b, f = r["bf16"], r["fp32"]  # type: ignore[index]
        lines.append(
            f"| {r['family']} | {r['rms_scale']:.3e} | {b['category']} | "
            f"{', '.join(f'{v:.3e}' for v in b['final_delta_rms'])} | "
            f"{f['category']} | "
            f"{', '.join(f'{v:.3e}' for v in f['final_delta_rms'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "- BF16 `above_ceiling`/`nonfinite` rows are inputs the BF16 attempt "
            "would flag (the production rescue trigger)."
        ),
        (
            "- FP32 `below_floor` rows are inputs whose FP32 NS output collapses "
            "below the Moonlight-sane band — a hard-fail category the old dump "
            "could not distinguish from above-ceiling/nonfinite."
        ),
        (
            "- `determinism_spread` > 0 on BF16 at one (family, scale) documents "
            "kernel nondeterminism across identical repeats on this device."
        ),
    ]
    (args.output_dir / "cmuon-fp32-rescue-stress.md").write_text("\n".join(lines) + "\n")
    print(f"[stress] wrote {json_path} and cmuon-fp32-rescue-stress.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
