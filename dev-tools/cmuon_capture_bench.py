#!/usr/bin/env python3
"""F2 capture-path benchmark: time the NEW hard-fail critical path on a
production-scale 2560x2560 BF16 chunk, per phase, over >=20 runs.

Phases timed (one iteration = the exact production failure critical path):
  1. device_clone     - frozen = chunk.detach().contiguous().clone()
  2. d2h              - .cpu()
  3. serialize        - _write_tensor_bytes (safetensors bytes)
  4. sha256           - hashlib over the exact file bytes
  5. metadata         - build_minimal_capsule_metadata (4 O(n) reductions)
  6. local_publish    - publish_minimal_capsule (temp dir + fsync files +
                        fsync dir + atomic rename) on the LOCAL overlay
  total_local = 1..6  (the SLA: p50 < 1s, p95 < 2s over >=20 runs)
  7. mirror           - mirror_capsule to the shared root (BEFST-EFFORT,
                        excluded from the local SLA, timed separately)

Runs with the forensic tree's src on PYTHONPATH (run it against the F2
tree). Output: a JSON report (per-phase p50/p95/max + totals + verdict
against the SLA).

Usage:
  python dev-tools/cmuon_capture_bench.py \
      --workdir /sakuramoon-runtime/cmuon-f2/bench \
      --out capture-bench.json [--n 24] [--device cuda:0]

Exit code: 0 = SLA met, 1 = SLA missed (verdict in the report).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _pctl(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending-sorted list."""
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, round((len(sorted_vals) - 1) * p)))
    return sorted_vals[k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    assert args.n >= 20, "spec: >=20 runs required"

    src = Path(__file__).resolve().parent.parent / "src"
    import sys

    sys.path.insert(0, str(src))
    import hashlib

    from sakuramoon.optim.cmuon_hardfail import (
        _write_tensor_bytes,
        build_minimal_capsule_metadata,
        mirror_capsule,
        publish_minimal_capsule,
    )

    device = torch.device(args.device)
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    shared_root = workdir / "shared-root"
    shared_root.mkdir(parents=True, exist_ok=True)

    # Production-scale 112105-class input: 2560x2560 BF16 on device.
    g = torch.Generator(device="cpu").manual_seed(20260903)
    chunk = torch.randn(2560, 2560, generator=g, dtype=torch.float32) * 0.02
    chunk = chunk.to(torch.bfloat16).to(device)

    phase_keys = (
        "device_clone",
        "d2h",
        "serialize",
        "sha256",
        "metadata",
        "local_publish",
    )
    timings: dict[str, list[float]] = {k: [] for k in phase_keys}
    timings["mirror"] = []
    timings["total_local"] = []
    timings["total_with_mirror"] = []

    last_event_dir: Path | None = None
    for i in range(args.n):
        t0 = time.perf_counter()
        frozen = chunk.detach().contiguous().clone()
        t1 = time.perf_counter()
        cpu = frozen.cpu()
        t2 = time.perf_counter()
        tensor_bytes, tensor_fmt, _tensor_name = _write_tensor_bytes(cpu)
        t3 = time.perf_counter()
        sha = hashlib.sha256(tensor_bytes).hexdigest()
        t4 = time.perf_counter()
        metadata = build_minimal_capsule_metadata(
            observations=i,
            this_rank=0,
            world_size=2,
            fqn="bench.scale.content_gate.weight",
            chunk_idx=0,
            role="attention_content_gate",
            owner=0,
            run_id="bench",
            hostname="bench",
            pid=0,
            process_steps=i + 1,
            last_successful_update=None,
            attempted_update=None,
            checkpoint_source=None,
            input_tensor=cpu,
            alpha=1.5625e-4,
            ns_steps=4,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            eps=1e-8,
            lr=1.5625e-4,
            target_delta_rms=3.125e-5,
            ceiling=0.5,
            rescue_floor=1.5625e-6,
            bf16_delta_rms=1.0,
            original_fp32_delta_rms=1.5e-6,
            original_fp32_finite=True,
            fp32_failure_reason="below_floor",
            bf16_failure_name="above_ceiling",
            failure_message="bench",
            tensor_sha256=sha,
            tensor_format=tensor_fmt,
            shared_mirror_root=str(shared_root),
        )
        t5 = time.perf_counter()
        event_dir = publish_minimal_capsule(
            root=workdir / "emergency",
            observations=i,
            rank=0,
            world_size=2,
            fqn="bench.scale.content_gate.weight",
            chunk_idx=0,
            role="attention_content_gate",
            owner=0,
            input_tensor=cpu,
            metadata=metadata,
        )
        t6 = time.perf_counter()
        last_event_dir = event_dir
        mirror = mirror_capsule(event_dir, shared_root)
        t7 = time.perf_counter()

        timings["device_clone"].append(t1 - t0)
        timings["d2h"].append(t2 - t1)
        timings["serialize"].append(t3 - t2)
        timings["sha256"].append(t4 - t3)
        timings["metadata"].append(t5 - t4)
        timings["local_publish"].append(t6 - t5)
        timings["mirror"].append(t7 - t6)
        timings["total_local"].append(t6 - t0)
        timings["total_with_mirror"].append(t7 - t0)
        if i % 5 == 0 or i == args.n - 1:
            print(
                f"[bench] run {i:2d}/{args.n}: local={t6 - t0:.3f}s "
                f"(publish={t6 - t5:.3f}s) mirror={mirror['status']} "
                f"({t7 - t6:.3f}s)",
                flush=True,
            )

    def _block(vals: list[float]) -> dict[str, float]:
        s = sorted(vals)
        return {
            "p50_s": round(_pctl(s, 0.5), 6),
            "p95_s": round(_pctl(s, 0.95), 6),
            "max_s": round(s[-1], 6),
            "mean_s": round(sum(s) / len(s), 6),
        }

    report: dict[str, object] = {
        "schema": "sakuramoon.cmuon_capture_bench.v1",
        "device": str(device),
        "input": {"shape": [2560, 2560], "dtype": "bfloat16", "bytes": 2560 * 2560 * 2},
        "n_runs": args.n,
        "phases": {k: _block(v) for k, v in timings.items()},
        "sla": {
            "total_local_p50_s": 1.0,
            "total_local_p95_s": 2.0,
            "runs_required": 20,
        },
    }
    total = report["phases"]["total_local"]  # type: ignore[index]
    report["verdict"] = (
        "PASS"
        if total["p50_s"] < 1.0 and total["p95_s"] < 2.0
        else "FAIL"
    )
    out = args.out or workdir / "capture-bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    # leave exactly one sample capsule behind for inspection
    if last_event_dir is not None:
        print(f"[bench] sample capsule: {last_event_dir}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
