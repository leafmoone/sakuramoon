#!/usr/bin/env python3
"""F1 OLD critical-path timings (for the F2 report's OLD timeline).

Run this script against the F1 tree (it imports the tree's own src via its
location). It replays exactly what the F1 production failure critical path
spent before ANY byte hit disk, on a production-scale 2560x2560 BF16 chunk
(the 112105 class):

  1. d2h          - frozen device clone -> .cpu()
  2. serialize    - _write_tensor_bytes + sha256 over the bytes
  3. trace_bf16   - cmuon_ns_trace.trace_ns_replay (working_dtype=bfloat16)
                    = a FULL production BF16 NS replay on CPU (4 iters)
  4. trace_fp32   - trace_ns_replay (working_dtype=float32)
                    = a FULL production FP32 NS replay on CPU (4 iters)
  5. metadata     - build_hard_fail_metadata (assembles with the traces)
  6. publish      - publish_hard_fail_artifact (temp + fsync + rename)
  total

This is the OLD timeline the F2 spec requires for the before/after
comparison (the 112105 event was SIGTERM'd 30s after the raise; this
measures why the publish could not finish inside that window).

One representative pass (the CPU NS replays are the slow, deterministic
dominant term; --n repeats each phase for stability, default 1 for speed,
<=3 recommended).

Usage (on salt10, from the F1 tree):
  /sakuramoon-runtime/sakuramoon-dtk-venv/bin/python \
      dev-tools/cmuon_oldpath_timings.py \
      --workdir /sakuramoon-runtime/cmuon-f1/out/oldpath-bench \
      --out oldpath-timings.json [--n 1] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    assert 1 <= args.n <= 3, "keep n small: each pass includes 2 CPU NS replays"

    src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src))
    import hashlib

    from sakuramoon.optim.cmuon import (
        cmuon_zeroth_power_bf16,
        cmuon_zeroth_power_fp32,
    )
    from sakuramoon.optim.cmuon_hardfail import (
        _write_tensor_bytes,
        build_hard_fail_metadata,
        publish_hard_fail_artifact,
    )
    from sakuramoon.optim.cmuon_ns_trace import (
        replay_result_to_json,
        trace_ns_replay,
    )

    device = torch.device(args.device)
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    ns_steps = 4
    coefficients = (3.4445, -4.7750, 2.0315)
    eps = 1e-8

    g = torch.Generator(device="cpu").manual_seed(20260903)
    chunk = (
        torch.randn(2560, 2560, generator=g, dtype=torch.float32) * 0.02
    ).to(torch.bfloat16).to(device)

    phases: dict[str, list[float]] = {
        "d2h": [],
        "serialize": [],
        "trace_bf16": [],
        "trace_fp32": [],
        "metadata": [],
        "publish": [],
        "total": [],
    }

    for i in range(args.n):
        t0 = time.perf_counter()
        frozen = chunk.detach().contiguous().clone().cpu()
        t1 = time.perf_counter()

        tensor_bytes, tensor_fmt, _name = _write_tensor_bytes(frozen)
        sha = hashlib.sha256(tensor_bytes).hexdigest()
        t2 = time.perf_counter()

        # F1 ran the real production NS on CPU with per-iteration capture.
        diag_bf16 = replay_result_to_json(
            trace_ns_replay(
                frozen,
                ns_steps=ns_steps,
                coefficients=coefficients,
                eps=eps,
                alpha=1.5625e-4,
                working_dtype="bfloat16",
            )
        )
        t3 = time.perf_counter()
        diag_fp32 = replay_result_to_json(
            trace_ns_replay(
                frozen,
                ns_steps=ns_steps,
                coefficients=coefficients,
                eps=eps,
                alpha=1.5625e-4,
                working_dtype="float32",
            )
        )
        t4 = time.perf_counter()

        metadata = build_hard_fail_metadata(
            observations=i,
            this_rank=0,
            world_size=2,
            fqn="oldpath.timings.content_gate.weight",
            chunk_idx=0,
            role="attention_content_gate",
            owner=0,
            input_tensor=frozen,
            alpha=1.5625e-4,
            ns_steps=ns_steps,
            ns_coefficients=coefficients,
            eps=eps,
            lr=1.5625e-4,
            target_delta_rms=3.125e-5,
            ceiling=0.5,
            rescue_floor=1.5625e-6,
            bf16_delta_rms=1.0,
            original_fp32_delta_rms=1.5e-6,
            original_fp32_finite=True,
            fp32_failure_reason="below_floor",
            bf16_failure_name="above_ceiling",
            failure_message="oldpath bench",
            tensor_sha256=sha,
            tensor_format=tensor_fmt,
            diagnostic_bf16=diag_bf16,
            diagnostic_fp32=diag_fp32,
            forensic_trace_error=None,
        )
        t5 = time.perf_counter()

        publish_hard_fail_artifact(
            root=workdir / "artifacts",
            observations=i,
            rank=0,
            world_size=2,
            fqn="oldpath.timings.content_gate.weight",
            chunk_idx=0,
            role="attention_content_gate",
            owner=0,
            input_tensor=frozen,
            metadata=metadata,
        )
        t6 = time.perf_counter()

        phases["d2h"].append(t1 - t0)
        phases["serialize"].append(t2 - t1)
        phases["trace_bf16"].append(t3 - t2)
        phases["trace_fp32"].append(t4 - t3)
        phases["metadata"].append(t5 - t4)
        phases["publish"].append(t6 - t5)
        phases["total"].append(t6 - t0)
        print(f"[oldpath] pass {i}: total={t6 - t0:.2f}s", flush=True)

    # Sanity: the raw CPU NS cost the F1 path incurred (both dtypes) —
    # reported for the report's narrative (not part of a phase above: the
    # traces ARE the replays).
    t_ns = time.perf_counter()
    cmuon_zeroth_power_bf16(frozen.clone(), ns_steps, coefficients, eps)
    cmuon_zeroth_power_fp32(frozen.float(), ns_steps, coefficients, eps)
    ns_sanity = time.perf_counter() - t_ns

    def _block(vals: list[float]) -> dict[str, float]:
        s = sorted(vals)
        return {
            "mean_s": round(sum(s) / len(s), 6),
            "min_s": round(s[0], 6),
            "max_s": round(s[-1], 6),
        }

    report = {
        "schema": "sakuramoon.cmuon_oldpath_timings.v1",
        "device": str(device),
        "input": {"shape": [2560, 2560], "dtype": "bfloat16"},
        "n_passes": args.n,
        "phases": {k: _block(v) for k, v in phases.items()},
        "cpu_ns_sanity_both_dtypes_s": round(ns_sanity, 6),
        "note": (
            "F1 critical path (measured on the F1 tree): the two CPU NS "
            "replays (trace_bf16 + trace_fp32) dominated; the 112105 "
            "owner was SIGTERM'd ~30s after the raise and the publish "
            "never finished"
        ),
        "wall_clock_unix_seconds": time.time(),
    }
    out = args.out or workdir / "oldpath-timings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
