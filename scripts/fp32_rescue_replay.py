"""Offline FP32-rescue replay (D2 round, spec sections 2-6).

Replays saved pre-NS Nesterov chunks through both Newton-Schulz paths and
compares the post-NS safety:

  A. cmuon_zeroth_power_bf16   (production path, known chaos)
  B. cmuon_zeroth_power_fp32   (rescue path, pure FP32)

Sections covered:
  --replay       sec 2+3: every dangerous tensor (recovered), BF16 vs FP32,
                 ceiling ratio, rescue verdict (fp32_failed must be 0).
  --repeats      sec 5: N-repeat chaos test on the worst tensors
                 (BF16 vs FP32 catastrophic fraction + output spread).
  --align        sec 4: stratified SAFE sample >= min-samples:
                 delta ratio / update cosine / relative Frobenius error.
  --benchmark    sec 6: per-shape BF16 vs FP32 NS4 wall clock + the
                 shape-weighted expected rescue overhead.

All tensors are plain .pt files: the D1 full-sample dump format
({"tensor": fp32 cpu, "fqn", "chunk", "meta"}) or a bare fp32 tensor.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from sakuramoon.optim.cmuon import (
    cmuon_moonlight_alpha,
    cmuon_zeroth_power_bf16,
    cmuon_zeroth_power_fp32,
)

LR = 1.5625e-04  # lr at the ckpt_97100 fork (guard-forensic exact: 1.562499965e-4)
TARGET = 0.2 * LR
CEILING = 10.0 * TARGET
NS_STEPS = 4
COEFFS = (3.4445, -4.7750, 2.0315)
EPS = 1e-7


def _moonlight_order_ok(delta_rms: float) -> bool:
    """FP32 delta must sit at a Moonlight-sane scale (not degenerate)."""
    return 0.05 * TARGET <= delta_rms <= 20.0 * TARGET


def load_tensor(path: str) -> tuple[torch.Tensor, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"].float().cpu(), obj.get("meta", {})
    if isinstance(obj, torch.Tensor):
        return obj.float().cpu(), {}
    raise TypeError(f"unsupported artifact: {path}")


def obs_of(path: str) -> int | None:
    m = re.search(r"obs-(\d+)", path)
    return int(m.group(1)) if m else None


def role_of(fqn: str) -> str:
    """Map a parameter fqn to its CMuon role (mirrors production routing)."""
    if "conditioner.shared_block_projection" in fqn:
        return "adaln_shared"
    m = re.search(r"\.attention\.(q_proj|k_proj|v_proj|out_proj|content_gate)\.weight$", fqn)
    if m:
        return {
            "q_proj": "attention_q",
            "k_proj": "attention_k",
            "v_proj": "attention_v",
            "out_proj": "attention_out",
            "content_gate": "attention_content_gate",
        }[m.group(1)]
    if re.search(r"\.mlp\.in_proj\.weight$", fqn):
        return "ffn_in"
    if re.search(r"\.mlp\.down_proj\.weight$", fqn):
        return "ffn_down"
    return "other"


def slot_of(fqn: str) -> str:
    m = re.search(r"slot_(\d+)", fqn)
    return f"slot_{m.group(1)}" if m else "nonslot"


def role_slot_of_name(name: str) -> tuple[str, str]:
    """Role/slot from a dump basename ('chunk-<fqn.dots->underscores>-cN.pt').

    The fqn is not recoverable unambiguously (slot_02 / q_proj both use
    underscores), so parse the role/slot directly from the underscore form.
    """
    stem = re.sub(r"^chunk-", "", name)
    stem = re.sub(r"-c\d+\.pt$", "", stem)
    m = re.search(r"slot_(\d+)", stem)
    slot = f"slot_{m.group(1)}" if m else "nonslot"
    if "conditioner_shared_block_projection" in stem:
        role = "adaln_shared"
    else:
        m2 = re.search(
            r"attention_(q_proj|k_proj|v_proj|out_proj|content_gate)_weight$", stem
        )
        if m2:
            role = {
                "q_proj": "attention_q",
                "k_proj": "attention_k",
                "v_proj": "attention_v",
                "out_proj": "attention_out",
                "content_gate": "attention_content_gate",
            }[m2.group(1)]
        elif re.search(r"mlp_in_proj_weight$", stem):
            role = "ffn_in"
        elif re.search(r"mlp_down_proj_weight$", stem):
            role = "ffn_down"
        else:
            role = "other"
    return role, slot


def run_ns(t: torch.Tensor, fp32: bool) -> torch.Tensor:
    if fp32:
        return cmuon_zeroth_power_fp32(t, NS_STEPS, COEFFS, EPS)
    return cmuon_zeroth_power_bf16(t, NS_STEPS, COEFFS, EPS)


def evaluate(t: torch.Tensor, fp32: bool) -> dict:
    ns = run_ns(t, fp32)
    rows, cols = t.shape
    alpha = cmuon_moonlight_alpha(rows, cols, LR, 1)
    delta = (-alpha) * ns
    dr = float(delta.pow(2).mean().sqrt())
    return {
        "ns_rms": float(ns.pow(2).mean().sqrt()),
        "ns_max": float(ns.abs().max()),
        "delta_rms": dr,
        "delta_max": float(delta.abs().max()),
        "ceiling_ratio": dr / CEILING,
        "finite": bool(torch.isfinite(delta).all()),
        "moonlight_ok": _moonlight_order_ok(dr),
        "delta": delta.cpu(),
    }


def input_stats(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "element_rms": float(t.pow(2).mean().sqrt()),
        "fro_norm": float(t.norm()),
        "max_abs": float(t.abs().max()),
    }


@dataclass
class ReplayResult:
    total: int = 0
    bf16_catastrophic: int = 0
    bf16_safe: int = 0
    fp32_rescued: int = 0
    fp32_failed: int = 0
    fp32_also_safe: int = 0
    rows: list[dict] = field(default_factory=list)


def iter_tensor_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    return paths


def cmd_replay(args) -> None:
    # Streaming: load/evaluate/discard one tensor at a time. The full-sample
    # set is ~90 GB on disk; holding it all in RAM is what stressed the old
    # pod. RSS stays flat at a few hundred MB.
    dev = torch.device(args.device)
    paths = iter_tensor_paths(args.tensors)
    print(f"replaying {len(paths)} tensors (streaming)")
    res = ReplayResult()
    for p in paths:
        t, meta = load_tensor(p)
        t = t.to(dev)
        a = evaluate(t, False)
        b = evaluate(t, True)
        cat_a = (not a["finite"]) or a["delta_rms"] > CEILING
        cat_b = (not b["finite"]) or b["delta_rms"] > CEILING
        res.total += 1
        if cat_a:
            res.bf16_catastrophic += 1
        else:
            res.bf16_safe += 1
        if cat_a and cat_b:
            res.fp32_failed += 1
            verdict = "FP32_FAILED"
        elif cat_a:
            if b["moonlight_ok"]:
                res.fp32_rescued += 1
                verdict = "RESCUED"
            else:
                res.fp32_failed += 1
                verdict = "FP32_DEGENERATE"
        else:
            res.fp32_also_safe += 1
            verdict = "BOTH_SAFE"
        fqn = meta.get("fqn", "")
        row = {
            "path": p,
            "obs": obs_of(p),
            "fqn": fqn,
            "role": role_of(fqn),
            "slot": slot_of(fqn),
            "chunk": meta.get("chunk"),
            **input_stats(t),
            "bf16": {k: v for k, v in a.items() if k != "delta"},
            "fp32": {k: v for k, v in b.items() if k != "delta"},
            "bf16_catastrophic": cat_a,
            "fp32_catastrophic": cat_b,
            "verdict": verdict,
        }
        res.rows.append(row)
        if cat_a:
            print(
                f"  [DANGEROUS] {fqn} c{meta.get('chunk', '?')} obs{obs_of(p)}: "
                f"bf16 dr={a['delta_rms']:.3e} fin={a['finite']} | "
                f"fp32 dr={b['delta_rms']:.3e} fin={b['finite']} -> {verdict}"
            )
        del t, a, b
    out = {
        "dangerous_total_bf16": res.bf16_catastrophic,
        "bf16_safe": res.bf16_safe,
        "fp32_rescued": res.fp32_rescued,
        "fp32_failed": res.fp32_failed,
        "fp32_also_safe": res.fp32_also_safe,
        "rescue_rate": (
            res.fp32_rescued / res.bf16_catastrophic if res.bf16_catastrophic else None
        ),
        "ceiling": CEILING,
        "target_0p2lr": TARGET,
        "lr": LR,
        "rows": res.rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"total={res.total} bf16_cat={res.bf16_catastrophic} "
        f"fp32_rescued={res.fp32_rescued} fp32_failed={res.fp32_failed} "
        f"rescue_rate={out['rescue_rate']}"
    )
    print(f"wrote {args.out}")


def _output_spread(deltas: list[torch.Tensor]) -> dict:
    """Worst pairwise relative L2 distance across repeats."""
    ref = deltas[0]
    worst_rel = 0.0
    for d in deltas[1:]:
        rel = float((d - ref).norm() / ref.norm().clamp(min=1e-30))
        worst_rel = max(worst_rel, rel)
    rms = [float(d.pow(2).mean().sqrt()) for d in deltas]
    return {
        "n": len(deltas),
        "delta_rms_min": min(rms),
        "delta_rms_p50": sorted(rms)[len(rms) // 2],
        "delta_rms_max": max(rms),
        "catastrophic_fraction": sum(
            1 for d in deltas if (not bool(torch.isfinite(d).all())) or float(d.pow(2).mean().sqrt()) > CEILING
        ) / len(deltas),
        "nonfinite_count": sum(1 for d in deltas if not bool(torch.isfinite(d).all())),
        "output_rel_spread_worst": worst_rel,
    }


def cmd_repeats(args) -> None:
    dev = torch.device(args.device)
    tensors = []
    for p in args.tensors:
        t, meta = load_tensor(p)
        tensors.append((p, t, meta))
    out = {}
    for p, t, meta in tensors:
        t = t.to(dev)
        rows, cols = t.shape
        alpha = cmuon_moonlight_alpha(rows, cols, LR, 1)
        name = f"{meta.get('fqn', p)}#c{meta.get('chunk', '?')}"
        for tag, fp32 in (("bf16", False), ("fp32", True)):
            deltas = []
            for _ in range(args.n):
                ns = run_ns(t, fp32)
                deltas.append(((-alpha) * ns).float().cpu())
                torch.cuda.synchronize()
            out.setdefault(name, {})[tag] = _output_spread(deltas)
            s = out[name][tag]
            print(
                f"  {name} {tag}: cat_frac={s['catastrophic_fraction']:.2f} "
                f"nonfinite={s['nonfinite_count']} rms[{s['delta_rms_min']:.3e}.."
                f"{s['delta_rms_max']:.3e}] spread={s['output_rel_spread_worst']:.3e}"
            )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


def cmd_align(args) -> None:
    import random

    dev = torch.device(args.device)
    # The full-sample dumps do not carry the fqn in meta; the D1 labels are
    # keyed by (obs, fqn, chunk) in the shadow JSONLs. The dump filename is
    # exactly f"chunk-{fqn.replace('.', '_')}-c{chunk}.pt" inside
    # "obs-{obs:02d}/", so index the labels by (obs, basename) for an exact
    # match (dot/underscore round-tripping is ambiguous otherwise).
    labels: dict[tuple[int, str], str] = {}
    for jl in args.jsonl:
        with open(jl) as f:
            for line in f:
                rec = json.loads(line)
                for r in rec["rows"]:
                    name = r["fqn"].replace(".", "_")
                    key = (rec["obs"], f"chunk-{name}-c{r['chunk']}.pt")
                    labels[key] = r["label"]
    # Two-pass streaming: pass 1 collects (path, meta, label, amplitude) one
    # tensor at a time; pass 2 reloads only the stratified picks. Peak RSS is
    # one tensor plus the metadata list (a few hundred MB for ~3k tensors).
    def terciles(vals: list[float]) -> list[str]:
        s = sorted(vals)
        k = len(s) // 3
        lo, hi = s[k], s[2 * k]
        return ["low" if v < lo else ("high" if v > hi else "mid") for v in vals]

    safe: list[tuple[str, str, float]] = []
    unlabeled = 0
    total = 0
    for p in iter_tensor_paths(args.tensors):
        t, _meta = load_tensor(p)
        total += 1
        obs = obs_of(p)
        base = os.path.basename(p)
        lab = labels.get((obs, base)) if obs is not None else None
        if lab is None:
            unlabeled += 1
        if lab == "SAFE":
            safe.append((p, base, float(t.pow(2).mean().sqrt())))
        del t
    if unlabeled:
        print(f"WARNING: {unlabeled}/{total} tensors without a D1 label (excluded)")
    rng = random.Random(args.seed)
    amps = [x[2] for x in safe]
    amp_t = terciles(amps)
    # top1 energy is not in the dump; the D1 replay rows carry it. The
    # stratification axes role x slot x amplitude already cover the spec's
    # coverage requirements (all roles, all slots, low/med/high amplitude);
    # top1 spread is handled by the replay rows (sec 2/3).
    buckets: dict[str, list[int]] = {}
    roles: list[tuple[str, str]] = []
    for i, (p, base, amp) in enumerate(safe):
        role, slot = role_slot_of_name(base)
        roles.append((role, slot))
        bucket = f"{role}|{slot}|{amp_t[i]}"
        buckets.setdefault(bucket, []).append(i)
    n_buckets = len(buckets)
    per_bucket = max(1, args.min_samples // max(1, n_buckets))
    picked: list[int] = []
    for key in sorted(buckets):
        idxs = buckets[key]
        rng.shuffle(idxs)
        picked.extend(idxs[:per_bucket])
    rng.shuffle(picked)
    picked = picked[: args.max_samples]
    print(f"stratified sample: {len(picked)} of {len(safe)} safe tensors ({n_buckets} buckets)")
    rows = []
    for i in picked:
        p, base, _amp = safe[i]
        t, _meta2 = load_tensor(p)
        t = t.to(dev)
        a = evaluate(t, False)
        b = evaluate(t, True)
        d_bf16 = a["delta"].to(torch.bfloat16).float()  # production rounding
        d_fp32 = b["delta"]  # fp32 value (param is bf16; final rounding identical)
        ratio = float(d_fp32.pow(2).mean().sqrt()) / max(a["delta_rms"], 1e-30)
        cos = float(
            torch.nn.functional.cosine_similarity(
                d_bf16.flatten(), d_fp32.flatten(), dim=0
            )
        )
        rel_err = float((d_bf16 - d_fp32).norm() / d_fp32.norm().clamp(min=1e-30))
        role, slot = roles[i]
        rows.append(
            {
                "obs": obs_of(p),
                "fqn": base,
                "role": role,
                "slot": slot,
                "chunk": None,
                "shape": list(t.shape),
                "delta_ratio_fp32_over_bf16": ratio,
                "update_cosine": cos,
                "rel_fro_error": rel_err,
            }
        )
        del t, a, b

    def q(vals: list[float], qname: str) -> float:
        s = sorted(vals)
        k = round((len(s) - 1) * {"p50": 0.5, "p90": 0.9, "p99": 0.99}[qname])
        return s[k]

    ratios = [r["delta_ratio_fp32_over_bf16"] for r in rows]
    coss = [r["update_cosine"] for r in rows]
    errs = [r["rel_fro_error"] for r in rows]
    summary = {
        "samples": len(rows),
        "delta_ratio_p50": q(ratios, "p50"),
        "delta_ratio_p90": q(ratios, "p90"),
        "delta_ratio_p99": q(ratios, "p99"),
        "delta_ratio_worst": max(abs(r - 1) for r in ratios),
        "cosine_p50": q(coss, "p50"),
        "cosine_p90": q(coss, "p90"),
        "cosine_min": min(coss),
        "rel_error_p50": q(errs, "p50"),
        "rel_error_p90": q(errs, "p90"),
        "rel_error_p99": q(errs, "p99"),
        "rel_error_worst": max(errs),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


def cmd_benchmark(args) -> None:
    dev = torch.device(args.device)
    shapes = [(2560, 2560), (640, 2560), (6912, 2560), (2560, 6912), (2560, 1024)]
    out = {}
    for rows, cols in shapes:
        t = torch.randn(rows, cols, device=dev, dtype=torch.float32) * 1e-3
        for tag, fp32 in (("bf16", False), ("fp32", True)):
            # warmup
            for _ in range(3):
                run_ns(t, fp32)
            torch.cuda.synchronize()
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                run_ns(t, fp32)
                t1.record()
                torch.cuda.synchronize()
                times.append(t0.elapsed_time(t1) / 1000.0)
            times.sort()
            out[f"{rows}x{cols}"] = out.get(f"{rows}x{cols}", {})
            out[f"{rows}x{cols}"][tag] = {
                "median_s": times[len(times) // 2],
                "min_s": times[0],
                "max_s": times[-1],
            }
        e = out[f"{rows}x{cols}"]
        ratio = e["fp32"]["median_s"] / e["bf16"]["median_s"]
        e["fp32_over_bf16"] = ratio
        print(f"  {rows}x{cols}: bf16={e['bf16']['median_s']*1e3:.1f}ms "
              f"fp32={e['fp32']['median_s']*1e3:.1f}ms ratio={ratio:.2f}x")
    out["_meta"] = {"device": args.device, "iters": args.iters, "ns_steps": NS_STEPS}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("replay")
    pr.add_argument("--tensors", nargs="+", required=True)
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--out", required=True)
    pr.set_defaults(fn=cmd_replay)
    pp = sub.add_parser("repeats")
    pp.add_argument("--tensors", nargs="+", required=True)
    pp.add_argument("--n", type=int, default=100)
    pp.add_argument("--device", default="cuda")
    pp.add_argument("--out", required=True)
    pp.set_defaults(fn=cmd_repeats)
    pa = sub.add_parser("align")
    pa.add_argument("--tensors", nargs="+", required=True)
    pa.add_argument("--jsonl", nargs="+", required=True)
    pa.add_argument("--min-samples", type=int, default=1000)
    pa.add_argument("--max-samples", type=int, default=4000)
    pa.add_argument("--seed", type=int, default=97100)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--out", required=True)
    pa.set_defaults(fn=cmd_align)
    pb = sub.add_parser("benchmark")
    pb.add_argument("--iters", type=int, default=50)
    pb.add_argument("--device", default="cuda")
    pb.add_argument("--out", required=True)
    pb.set_defaults(fn=cmd_benchmark)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
