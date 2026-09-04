#!/usr/bin/env python3
"""Offline EXACT replay of a CMuon FP32-rescue hard-fail artifact (F1).

Developer-only tool. Given a hard-fail event directory (the one the
optimizer publishes at
``<root>/obs-<obs>-rank<R>-<fqn>-chunk<C>/`` with ``input.safetensors``
| ``input.pt`` + ``metadata.json``), this script:

  1. loads the EXACT saved NS input tensor (original dtype) and metadata;
  2. reruns the PRODUCTION Newton-Schulz functions
     (``cmuon_zeroth_power_bf16`` / ``cmuon_zeroth_power_fp32`` — imported
     from the tree under test, never reimplemented) on that exact input
     with the recorded ns_steps / coefficients / eps / alpha;
  3. recomputes the BF16 and FP32 delta RMS + finiteness and derives the
     FP32 failure reason with the SAME priority the production verdict
     uses (``classify_fp32_verdict``);
  4. runs the per-iteration diagnostic NS traces (``cmuon_ns_trace``);
  5. compares recorded-original vs replay values and prints a JSON report
     + a human-readable summary.

It performs NO optimizer.step(), loads NO model weights, and touches NO
checkpoint. It only reads the artifact and runs pure NS math.

Usage (run with the forensic tree's src on PYTHONPATH):

  python dev-tools/cmuon_fp32_rescue_replay.py \
      --artifact /sakuramoon-runtime/artifacts/g1/cmuon-hard-fail/obs-605-rank0-dit_blocks_...-chunk0 \
      [--output report.json]

Exit code is always 0 (diagnostic tool); mismatches are flagged in the
report/summary, not as failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _find_repo_src() -> Path:
    """Locate the tree's src/ (this file lives in dev-tools/ at the root)."""
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "src"
    if (candidate / "sakuramoon").is_dir():
        return candidate
    raise SystemExit(f"cannot locate src/ next to {here.parent}")


def _load_input(artifact_dir: Path) -> tuple[torch.Tensor, dict[str, object]]:
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"missing metadata.json in {artifact_dir}")
    metadata = json.loads(metadata_path.read_text())
    tensor_path = None
    for name in ("input.safetensors", "input.pt"):
        if (artifact_dir / name).is_file():
            tensor_path = artifact_dir / name
            break
    if tensor_path is None:
        raise SystemExit(f"missing input.safetensors / input.pt in {artifact_dir}")
    fmt = str(metadata.get("tensor_format", ""))
    if fmt == "safetensors" and tensor_path.suffix == ".safetensors":
        from safetensors.torch import load_file  # type: ignore[import-untyped]

        tensor = load_file(str(tensor_path))["input"]
    else:
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)[
            "input"
        ]
    return tensor, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    src = _find_repo_src()
    sys.path.insert(0, str(src))
    # Import AFTER the path fix so we replay the tree under test.
    from sakuramoon.optim.cmuon import (
        cmuon_zeroth_power_bf16,
        cmuon_zeroth_power_fp32,
    )
    from sakuramoon.optim.cmuon_hardfail import (
        compare_recorded_vs_replayed,
    )
    from sakuramoon.optim.cmuon_ns_trace import (
        replay_result_to_json,
        trace_ns_replay,
    )

    tensor, metadata = _load_input(args.artifact)

    ns_steps = int(metadata["ns_steps"])
    coefficients = tuple(float(v) for v in metadata["ns_coefficients"])
    eps = float(metadata["eps"])
    alpha = float(metadata["alpha"])
    ceiling = float(metadata["ceiling"])
    rescue_floor = float(metadata["rescue_floor"])

    # ---- production NS recompute (exact input, recorded parameters) ------
    bf16_ns = cmuon_zeroth_power_bf16(tensor, ns_steps, coefficients, eps)
    bf16_delta = (-alpha) * bf16_ns
    bf16_delta_rms = float(bf16_delta.float().pow(2).mean().sqrt().item())
    bf16_finite = bool(torch.isfinite(bf16_delta.float()).all().item())

    fp32_ns = cmuon_zeroth_power_fp32(
        tensor.float(), ns_steps, coefficients, eps
    )
    fp32_delta = (-alpha) * fp32_ns
    fp32_delta_rms = float(fp32_delta.float().pow(2).mean().sqrt().item())
    fp32_finite = bool(torch.isfinite(fp32_delta.float()).all().item())
    fp32_reason = (
        None
        if (fp32_finite and rescue_floor <= fp32_delta_rms <= ceiling)
        else (
            "nonfinite"
            if not fp32_finite
            else ("below_floor" if fp32_delta_rms < rescue_floor else "above_ceiling")
        )
    )

    # ---- per-iteration diagnostic traces ----------------------------------
    trace_bf16 = replay_result_to_json(
        trace_ns_replay(
            tensor,
            ns_steps=ns_steps,
            coefficients=coefficients,
            eps=eps,
            alpha=alpha,
            working_dtype="bfloat16",
        )
    )
    trace_fp32 = replay_result_to_json(
        trace_ns_replay(
            tensor,
            ns_steps=ns_steps,
            coefficients=coefficients,
            eps=eps,
            alpha=alpha,
            working_dtype="float32",
        )
    )

    # ---- recorded vs replay comparison ------------------------------------
    # Re-derive the comparison against the FRESH replay values (the metadata
    # may carry its own CPU diagnostic replay recorded at fail time; both
    # are shown, and the fresh replay is the offline ground truth for this
    # run's kernel).
    report = {
        "artifact": str(args.artifact),
        "fqn": metadata.get("fqn"),
        "chunk": metadata.get("chunk"),
        "role": metadata.get("role"),
        "shape": metadata.get("shape"),
        "dtype": metadata.get("dtype"),
        "ns_steps": ns_steps,
        "ns_coefficients": list(coefficients),
        "eps": eps,
        "alpha": alpha,
        "lr": metadata.get("lr"),
        "target_delta_rms": metadata.get("target_delta_rms"),
        "ceiling": ceiling,
        "rescue_floor": rescue_floor,
        "recorded_original": {
            "bf16_delta_rms": metadata.get("bf16_delta_rms"),
            "original_fp32_delta_rms": metadata.get("original_fp32_delta_rms"),
            "original_fp32_finite": metadata.get("original_fp32_finite"),
            "fp32_failure_reason": metadata.get("fp32_failure_reason"),
        },
        "replay_production_ns": {
            "bf16_delta_rms": bf16_delta_rms,
            "bf16_finite": bf16_finite,
            "fp32_delta_rms": fp32_delta_rms,
            "fp32_finite": fp32_finite,
            "fp32_failure_reason": fp32_reason,
        },
        "comparison_recorded_vs_replay": [
            {
                "field": row.field,
                "recorded": row.recorded,
                "replayed": row.replayed,
                "abs_diff": row.abs_diff,
                "rel_diff": row.rel_diff,
            }
            for row in compare_recorded_vs_replayed(
                {
                    **metadata,
                    "diagnostic_replay_bf16_delta_rms": bf16_delta_rms,
                    "diagnostic_replay_fp32_delta_rms": fp32_delta_rms,
                    "diagnostic_replay_fp32_finite": fp32_finite,
                }
            )
        ],
        "diagnostic_trace_bf16": trace_bf16,
        "diagnostic_trace_fp32": trace_fp32,
        "diagnostic_trace_at_fail_time": {
            "bf16": metadata.get("diagnostic_replay_bf16"),
            "fp32": metadata.get("diagnostic_replay_fp32"),
        },
        "forensic_trace_error_at_fail_time": metadata.get("forensic_trace_error"),
    }

    out_text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_text + "\n")
        print(f"[replay] report written: {args.output}")
    print(out_text)

    # Human summary.
    print("\n== CMuon FP32-rescue hard-fail replay summary ==")
    print(
        f"input: {metadata.get('fqn')}#chunk{metadata.get('chunk')} "
        f"{metadata.get('shape')} {metadata.get('dtype')} "
        f"input_rms={metadata.get('input_rms')}"
    )
    print(
        f"replay BF16: delta_rms={bf16_delta_rms:.6e} finite={bf16_finite} "
        f"(recorded {metadata.get('bf16_delta_rms')})"
    )
    print(
        f"replay FP32: delta_rms={fp32_delta_rms:.6e} finite={fp32_finite} "
        f"reason={fp32_reason} "
        f"(recorded {metadata.get('original_fp32_delta_rms')} / "
        f"{metadata.get('fp32_failure_reason')})"
    )
    print(
        "note: replay runs the CURRENT tree's production NS functions on the "
        "saved exact input; any difference vs the recorded original values "
        "reflects kernel/platform behavior (HCU vs this run's device), not a "
        "verdict change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
