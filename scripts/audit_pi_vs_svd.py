#!/usr/bin/env python3
"""Spec section 4: power-iteration accuracy audit on REAL shadow tensors.

Compares PI (with deflation) at several iteration counts against exact
torch.linalg.svdvals on the full nesterov-chunk dumps saved during the
shadow run (obs 1..N).  Reports per-iteration-count:

  * sigma1 relative error (median / max over tensors),
  * top1_energy absolute error (|pi_top1 - exact_top1|),
  * wall cost per tensor (median),

so the minimum sufficient iteration count can be chosen from data
instead of assumption.  Exact SVD runs on CPU (deterministic reference);
PI runs on the same device as its input (HCU in production).

Usage:
  python3 audit_pi_vs_svd.py <full-samples-dir> [--max-tensors 60]
    [--iters 5 10 20 30 50]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("--max-tensors", type=int, default=60)
    ap.add_argument("--iters", type=int, nargs="+", default=[5, 10, 20, 30, 50])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sakuramoon.optim.structural_calibration import _PowerIteration

    paths = sorted(Path(args.sample_dir).glob("obs-*/*.pt"))
    if not paths:
        paths = sorted(Path(args.sample_dir).glob("*.pt"))
    if not paths:
        raise SystemExit(f"no sample tensors under {args.sample_dir}")
    paths = paths[: args.max_tensors]
    print(f"auditing {len(paths)} tensors from {args.sample_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    results: dict[int, dict] = {
        k: {"rel_err_sigma1": [], "top1_energy_abs_err": [], "cost_ms": []}
        for k in args.iters
    }
    for i, p in enumerate(paths):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        a = blob["tensor"].float()
        exact = torch.linalg.svdvals(a.double())
        fro2 = float((exact**2).sum())
        exact_top1 = float(exact[0]) ** 2 / fro2
        exact_s1 = float(exact[0])
        ah = a.to(device)
        for k in args.iters:
            t0 = time.perf_counter()
            sig = _PowerIteration(ah, k, seed_base=1000).top4()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cost = (time.perf_counter() - t0) * 1000.0
            results[k]["rel_err_sigma1"].append(abs(sig[0] - exact_s1) / exact_s1)
            pi_top1 = sig[0] ** 2 / float(a.double().pow(2).sum())
            results[k]["top1_energy_abs_err"].append(abs(pi_top1 - exact_top1))
            results[k]["cost_ms"].append(cost)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    out = {}
    for k, r in results.items():
        out[k] = {
            "sigma1_rel_err_median": statistics.median(r["rel_err_sigma1"]),
            "sigma1_rel_err_max": max(r["rel_err_sigma1"]),
            "top1_energy_abs_err_median": statistics.median(r["top1_energy_abs_err"]),
            "top1_energy_abs_err_max": max(r["top1_energy_abs_err"]),
            "cost_ms_median": statistics.median(r["cost_ms"]),
        }
    print(json.dumps(out, indent=1))
    # pick the minimum sufficient iteration count: sigma1 rel err median
    # < 1e-3 and top1 energy abs err < 1e-4
    for k in args.iters:
        e = out[k]
        if e["sigma1_rel_err_median"] < 1e-3 and e["top1_energy_abs_err_median"] < 1e-4:
            print(f"MIN_SUFFICIENT_ITERS = {k}")
            break
    else:
        print("MIN_SUFFICIENT_ITERS = none (all below target accuracy)")


if __name__ == "__main__":
    main()
