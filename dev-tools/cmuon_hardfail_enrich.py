#!/usr/bin/env python3
"""Offline ENRICHMENT for CMuon minimal hard-fail capsules (F2).

Developer-only tool. The F2 production failure critical path writes only
the MINIMAL capsule (exact input tensor + metadata.json, local-first,
durable). EVERYTHING diagnostic is done HERE, offline, against the
published capsule:

  A. load the exact input, verify the tensor sha256 against the metadata
  B. production BF16 NS replay  (>=3 repeats, kernel nondeterminism)
  C. production FP32 NS replay  (>=3 repeats, kernel nondeterminism)
  D. per-iteration BF16 NS trace (``cmuon_ns_trace``)
  E. per-iteration FP32 NS trace (``cmuon_ns_trace``)
  F. recorded-original vs replay comparison (per repeat + aggregate)
  G. original classification preservation check
  H. exact low-rank diagnostics on the FP32 input:
       - finite / mean / std / rms / absmax / zero_fraction
       - row & column RMS percentiles (p0/p1/p10/p50/p90/p99/p100)
       - full or randomized (top-k>=128) singular spectrum:
         top-32 values, spectral norm, Fro norm, stable rank,
         numerical rank at 1e-2..1e-6, entropy/effective-rank proxy
         (randomized => explicitly flagged as a proxy)
       - observed_delta_ratio = fp32_delta_rms / target_delta_rms
       - rank_implied_by_partial_polar = rows * observed_delta_ratio^2
         (hypothesis probe; NOT a verdict input on its own)
  I. low-rank hypothesis verdict:
       LOW_RANK_CONFIRMED / LOW_RANK_NOT_CONFIRMED / INCONCLUSIVE

It performs NO optimizer.step(), loads NO model weights, and touches NO
checkpoint. It only reads the capsule and runs pure NS/SVD math. It never
overwrites the minimal metadata: the output is a NEW immutable
``enrichment.json`` (timestamped if one already exists).

Usage (run with the forensic tree's src on PYTHONPATH):

  python dev-tools/cmuon_hardfail_enrich.py \
      --capsule /sakuramoon-runtime/cmuon-f1-emergency/obs-604-rank0-...-chunk0 \
      [--output enrichment.json] [--repeats 3] [--device cuda|cpu] \
      [--svd full|randomized|auto] [--topk 128]

Exit code: 0 = ran (verdict may still be INCONCLUSIVE — see the report);
2 = capsule integrity failure (sha mismatch / unreadable input).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPLAY_REL_TOL = 0.1  # documented HCU GEMM noise floor (5-8%) + margin
STABLE_RANK_DEGEN = 64.0
TOP32_ENERGY_DEGEN = 0.99
NUMERICAL_RANK_DEGEN = 64
RANK_IMPLY_DEGEN = 64.0


def _find_repo_src() -> Path:
    """Locate the tree's src/ (this file lives in dev-tools/ at the root)."""
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "src"
    if (candidate / "sakuramoon").is_dir():
        return candidate
    raise SystemExit(f"cannot locate src/ next to {here.parent}")


def _load_capsule(capsule_dir: Path) -> tuple[torch.Tensor, dict[str, object], str]:
    """Load (tensor, metadata) from a minimal capsule and verify the sha.
    Returns (cpu tensor, metadata, tensor_path_str)."""
    metadata_path = capsule_dir / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"missing metadata.json in {capsule_dir}")
    metadata = json.loads(metadata_path.read_text())
    tensor_path = None
    for name in ("input.safetensors", "input.pt"):
        if (capsule_dir / name).is_file():
            tensor_path = capsule_dir / name
            break
    if tensor_path is None:
        raise SystemExit(f"missing input.safetensors / input.pt in {capsule_dir}")
    fmt = str(metadata.get("tensor_format", ""))
    if fmt == "safetensors" and tensor_path.suffix == ".safetensors":
        from safetensors.torch import load_file  # type: ignore[import-untyped]

        tensor = load_file(str(tensor_path))["input"]
    else:
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)[
            "input"
        ]
    # Integrity: sha over the EXACT file bytes (what the capsule recorded).
    actual_sha = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    recorded_sha = str(metadata.get("tensor_sha256", ""))
    if actual_sha != recorded_sha:
        raise _CapsuleIntegrityError(
            f"tensor sha mismatch: recorded {recorded_sha} != actual {actual_sha}"
        )
    shape_meta = metadata.get("shape")
    if shape_meta is not None and [int(s) for s in shape_meta] != list(
        tensor.shape
    ):
        raise _CapsuleIntegrityError(
            f"shape mismatch: metadata {shape_meta} != tensor {list(tensor.shape)}"
        )
    return tensor, metadata, str(tensor_path)


class _CapsuleIntegrityError(RuntimeError):
    pass


def _pct(t: torch.Tensor, qs: tuple[float, ...]) -> list[float]:
    q = torch.tensor(qs, dtype=torch.float64)
    v = torch.quantile(t.to(torch.float64), q)
    return [float(x) for x in v.tolist()]


def _percentile_block(a: torch.Tensor) -> dict[str, object]:
    qs = (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.00)
    row = a.pow(2).mean(dim=1).sqrt()
    col = a.pow(2).mean(dim=0).sqrt()
    return {
        "row_rms_percentiles": dict(zip(("p0", "p1", "p10", "p50", "p90", "p99", "p100"), _pct(row, qs))),
        "col_rms_percentiles": dict(zip(("p0", "p1", "p10", "p50", "p90", "p99", "p100"), _pct(col, qs))),
    }


def _randomized_svd_topk(a: torch.Tensor, k: int, power_iter: int = 2) -> torch.Tensor:
    """Halko et al. randomized SVD: top-k singular values of a (m x n)."""
    m, n = a.shape
    k = max(1, min(k, min(m, n)))
    rng = torch.Generator(device="cpu").manual_seed(20260903)
    omega = torch.randn(max(m, n), k, generator=rng, dtype=torch.float64)
    if m >= n:
        y = a.to(torch.float64) @ omega[:n, :]
    else:
        y = omega[:m, :].T @ a.to(torch.float64)
    for _ in range(power_iter):
        if m >= n:
            y = a.to(torch.float64) @ (a.to(torch.float64).T @ y)
        else:
            y = a.to(torch.float64).T @ (a.to(torch.float64) @ y)
    q, _ = torch.linalg.qr(y)
    b = q.T @ a.to(torch.float64)
    return torch.linalg.svdvals(b)[:k]


def _spectrum_block(
    a: torch.Tensor, svd_mode: str, device: torch.device, topk: int
) -> dict[str, object]:
    """Singular-spectrum diagnostics of the FP32 input (a = input.float())."""
    full = True
    if svd_mode == "auto":
        # Full SVD is only cheap on the accelerator for production shapes;
        # on CPU fall back to randomized beyond ~512x512.
        full = (device.type != "cpu") or (a.numel() <= 512 * 512)
        if svd_mode == "randomized":
            full = False
    if svd_mode == "randomized":
        full = False
    if full:
        af = a.to(device, dtype=torch.float32)
        s = torch.linalg.svdvals(af.double()).cpu().to(torch.float64)
        source = "full_svd"
    else:
        k = max(128, topk)
        s = _randomized_svd_topk(a, k).cpu().to(torch.float64)
        source = "randomized_svd_topk"
    s = s[s > 0]
    s_max = float(s.max().item()) if s.numel() else 0.0
    fro = float(a.double().norm().item())
    spectral = s_max
    stable_rank = (fro**2 / (spectral**2)) if spectral > 0 else None
    sum_all_sq = float(s.pow(2).sum().item()) if s.numel() else 0.0
    top32 = s[:32]
    top32_energy = (
        float(top32.pow(2).sum().item()) / sum_all_sq if sum_all_sq > 0 else None
    )
    numerical_rank = {
        f"1e-{e}": int((s > (10.0 ** -e) * s_max).sum().item())
        for e in range(2, 7)
        if s_max > 0
    }
    p = s / s.sum() if s.numel() and float(s.sum().item()) > 0 else None
    entropy_eff = None
    if p is not None and float(p.max().item()) > 0:
        entropy_eff = float(
            torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum()).item()
        )
    return {
        "source": source,
        "proxy": not full,
        "num_singular_values": int(s.numel()),
        "top32_singular_values": [float(v) for v in top32.tolist()],
        "spectral_norm": spectral,
        "frobenius_norm": fro,
        "stable_rank": stable_rank,
        "top32_energy": top32_energy,
        "numerical_rank_thresholds": numerical_rank,
        "entropy_effective_rank": entropy_eff,
        "entropy_effective_rank_is_proxy": not full,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--svd", default="auto", choices=("auto", "full", "randomized"))
    parser.add_argument("--topk", type=int, default=128)
    args = parser.parse_args(argv)

    src = _find_repo_src()
    sys.path.insert(0, str(src))
    # Import AFTER the path fix so we replay the tree under test.
    from sakuramoon.optim.cmuon import (
        cmuon_zeroth_power_bf16,
        cmuon_zeroth_power_fp32,
    )
    from sakuramoon.optim.cmuon_hardfail import classify_fp32_verdict
    from sakuramoon.optim.cmuon_ns_trace import (
        replay_result_to_json,
        trace_ns_replay,
    )

    integrity_error: str | None = None
    tensor: torch.Tensor | None = None
    metadata: dict[str, object] = {}
    tensor_path = "<unavailable>"
    try:
        tensor, metadata, tensor_path = _load_capsule(args.capsule)
    except _CapsuleIntegrityError as exc:
        integrity_error = str(exc)
        report = {
            "capsule": str(args.capsule),
            "integrity": {"status": "FAILED", "error": integrity_error},
            "verdict": "INCONCLUSIVE",
        }
        out_text = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.write_text(out_text + "\n")
        print(out_text)
        return 2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cpu" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ns_steps = int(metadata["ns_steps"])
    coefficients = tuple(float(v) for v in metadata["ns_coefficients"])
    eps = float(metadata["eps"])
    alpha = float(metadata["alpha"])
    ceiling = float(metadata["ceiling"])
    rescue_floor = float(metadata["rescue_floor"])
    target_delta_rms = float(metadata["target_delta_rms"])
    recorded_fp32_rms = metadata.get("original_fp32_delta_rms")
    recorded_fp32_finite = metadata.get("original_fp32_finite")
    recorded_reason = metadata.get("fp32_failure_reason")
    recorded_bf16_rms = metadata.get("bf16_delta_rms")

    t_in_dev = tensor.to(device)
    repeats_bf16: list[dict[str, object]] = []
    repeats_fp32: list[dict[str, object]] = []
    for r in range(1, max(1, args.repeats) + 1):
        # B. production BF16 NS (exact production entry point, on device).
        bf16_ns = cmuon_zeroth_power_bf16(
            t_in_dev.bfloat16().clone(), ns_steps, coefficients, eps
        )
        bf16_delta = (-alpha) * bf16_ns
        repeats_bf16.append(
            {
                "repeat": r,
                "delta_rms": float(bf16_delta.float().pow(2).mean().sqrt().item()),
                "finite": bool(torch.isfinite(bf16_delta.float()).all().item()),
            }
        )
        # C. production FP32 NS (the exact production cast: input.float()).
        fp32_ns = cmuon_zeroth_power_fp32(
            t_in_dev.float().clone(), ns_steps, coefficients, eps
        )
        fp32_delta = (-alpha) * fp32_ns
        rms32 = float(fp32_delta.float().pow(2).mean().sqrt().item())
        fin32 = bool(torch.isfinite(fp32_delta.float()).all().item())
        repeats_fp32.append(
            {
                "repeat": r,
                "delta_rms": rms32,
                "finite": fin32,
                "failure_reason": classify_fp32_verdict(fin32, rms32, rescue_floor, ceiling),
            }
        )

    # D/E. per-iteration traces (first-repeat values; the repeat loop above
    # covers the kernel nondeterminism question).
    trace_bf16 = replay_result_to_json(
        trace_ns_replay(
            t_in_dev.bfloat16(),
            ns_steps=ns_steps,
            coefficients=coefficients,
            eps=eps,
            alpha=alpha,
            working_dtype="bfloat16",
        )
    )
    trace_fp32 = replay_result_to_json(
        trace_ns_replay(
            t_in_dev.float(),
            ns_steps=ns_steps,
            coefficients=coefficients,
            eps=eps,
            alpha=alpha,
            working_dtype="float32",
        )
    )

    # F. recorded-original vs replay comparison.
    rec_rms = None if recorded_fp32_rms is None else float(recorded_fp32_rms)
    fp32_replay_rms = [float(r["delta_rms"]) for r in repeats_fp32]
    finite_replays = all(bool(r["finite"]) for r in repeats_fp32)
    spread_rel = None
    if len(fp32_replay_rms) >= 2 and max(fp32_replay_rms) > 0:
        spread_rel = (max(fp32_replay_rms) - min(fp32_replay_rms)) / max(
            fp32_replay_rms
        )
    recorded_vs_replay_rel = None
    if rec_rms is not None and finite_replays and rec_rms > 0:
        recorded_vs_replay_rel = abs(fp32_replay_rms[0] - rec_rms) / rec_rms
    if device.type == "cuda":
        replay_reproducible = bool(
            finite_replays
            and (spread_rel is None or spread_rel <= REPLAY_REL_TOL)
            and (
                recorded_vs_replay_rel is None
                or recorded_vs_replay_rel <= REPLAY_REL_TOL
            )
        )
    else:
        # CPU replays compare against HCU-recorded originals across
        # platforms: directional only, not a reproducibility proof.
        replay_reproducible = finite_replays and (
            spread_rel is None or spread_rel <= REPLAY_REL_TOL
        )

    # G. original classification preservation.
    replay_reasons = [r["failure_reason"] for r in repeats_fp32]
    classification_preserved = bool(
        rec_rms is not None
        and recorded_reason is not None
        and all(reason == recorded_reason for reason in replay_reasons)
    )

    # H. exact low-rank diagnostics (FP32 input = the capsule tensor's
    # exact float() — the matrix the production FP32 NS consumed).
    a = tensor.float()
    finite = bool(torch.isfinite(a).all().item())
    zero_fraction = float((a == 0).float().mean().item())
    mean_v = float(a.mean().item())
    std_v = float(a.std().item()) if a.numel() > 1 else 0.0
    rms_v = float(a.pow(2).mean().sqrt().item())
    absmax_v = float(a.abs().max().item()) if a.numel() else 0.0
    stats = {
        "finite": finite,
        "mean": mean_v,
        "std": std_v,
        "rms": rms_v,
        "absmax": absmax_v,
        "zero_fraction": zero_fraction,
        **_percentile_block(a),
    }
    spectrum = _spectrum_block(a, args.svd, device, args.topk)
    observed_delta_ratio = (
        None if rec_rms is None or target_delta_rms <= 0 else rec_rms / target_delta_rms
    )
    rows = int(a.shape[0])
    rank_implied = (
        None
        if observed_delta_ratio is None
        else rows * observed_delta_ratio**2
    )

    # I. low-rank hypothesis verdict (spec §11 — explicit evidence gates).
    stable_rank = spectrum.get("stable_rank")
    top32_energy = spectrum.get("top32_energy")
    degen_rank = None
    for key in ("1e-3", "1e-4"):
        if key in spectrum.get("numerical_rank_thresholds", {}):
            degen_rank = spectrum["numerical_rank_thresholds"][key]  # type: ignore[index]
            break
    spectral_degenerate = bool(
        stable_rank is not None
        and stable_rank <= STABLE_RANK_DEGEN
        and top32_energy is not None
        and top32_energy >= TOP32_ENERGY_DEGEN
    )
    direction_consistent = bool(
        rank_implied is not None
        and rank_implied <= RANK_IMPLY_DEGEN
        and degen_rank is not None
        and degen_rank >= 1
        and max(rank_implied, 1.0) <= 8.0 * degen_rank
        and max(float(degen_rank), 1.0) <= 8.0 * max(rank_implied, 1.0)
    )
    if (
        integrity_error is None
        and finite
        and rec_rms is not None
        and recorded_fp32_finite is True
        and replay_reproducible
        and spectral_degenerate
        and direction_consistent
    ):
        verdict = "LOW_RANK_CONFIRMED"
    elif (
        integrity_error is None
        and finite
        and rec_rms is not None
        and replay_reproducible
        and spectrum.get("source") == "full_svd"
        and (not spectral_degenerate or not direction_consistent)
    ):
        verdict = "LOW_RANK_NOT_CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"

    report = {
        "schema": "sakuramoon.cmuon_capsule_enrichment.v1",
        "capsule": str(args.capsule),
        "tensor_path": tensor_path,
        "integrity": {"status": "OK", "tensor_sha256": str(metadata.get("tensor_sha256"))},
        "device": str(device),
        "capsule_schema": metadata.get("schema"),
        "fqn": metadata.get("fqn"),
        "chunk": metadata.get("chunk"),
        "shape": list(a.shape),
        "repeats": max(1, args.repeats),
        "recorded_original": {
            "bf16_delta_rms": recorded_bf16_rms,
            "fp32_delta_rms": rec_rms,
            "fp32_finite": recorded_fp32_finite,
            "fp32_failure_reason": recorded_reason,
            "target_delta_rms": target_delta_rms,
            "rescue_floor": rescue_floor,
            "ceiling": ceiling,
        },
        "replay_bf16": repeats_bf16,
        "replay_fp32": repeats_fp32,
        "replay_comparison": {
            "fp32_replay_spread_rel": spread_rel,
            "fp32_recorded_vs_replay_rel": recorded_vs_replay_rel,
            "bf16_replay_rms_first": repeats_bf16[0]["delta_rms"] if repeats_bf16 else None,
            "bf16_recorded_rms": recorded_bf16_rms,
            "replay_reproducible": replay_reproducible,
            "note": (
                "replays run the CURRENT tree's production NS on the exact "
                "saved input; recorded-vs-replay differences beyond the HCU "
                "noise floor indicate kernel/platform behavior, not a "
                "verdict change"
            ),
        },
        "trace_bf16": trace_bf16,
        "trace_fp32": trace_fp32,
        "classification_preserved": classification_preserved,
        "fp32_input_stats": stats,
        "spectrum": spectrum,
        "low_rank_probes": {
            "observed_delta_ratio": observed_delta_ratio,
            "rank_implied_by_partial_polar": rank_implied,
            "rank_implied_formula": "rows * (fp32_delta_rms/target_delta_rms)^2",
            "hypothesis": (
                "if the failed FP32 delta were a partial-polar update of a "
                "rank-r matrix, its energy ratio would be ~r/rows; this is a "
                "probe, never a verdict input on its own"
            ),
        },
        "low_rank_hypothesis": {
            "verdict": verdict,
            "gates": {
                "input_finite": finite,
                "recorded_fp32_finite": bool(recorded_fp32_finite),
                "replay_reproducible": replay_reproducible,
                "spectral_degenerate": spectral_degenerate,
                "direction_consistent": direction_consistent,
                "stable_rank": stable_rank,
                "top32_energy": top32_energy,
                "numerical_rank_1e-3_or_1e-4": degen_rank,
                "spectrum_is_proxy": bool(spectrum.get("proxy")),
            },
            "criteria": {
                "LOW_RANK_CONFIRMED": (
                    "exact tensor + finite + replay reproducible + stable "
                    f"rank <= {STABLE_RANK_DEGEN:g} + top-32 energy >= "
                    f"{TOP32_ENERGY_DEGEN} + rank-implied/numerical-rank "
                    "mutual consistency (8x band)"
                ),
                "LOW_RANK_NOT_CONFIRMED": (
                    "complete evidence (full SVD, finite, reproducible) but "
                    "the spectrum/direction gates fail"
                ),
                "INCONCLUSIVE": "missing evidence (nonfinite, sha failure, "
                "non-reproducible replay, or proxy spectrum failing a gate)",
            },
        },
        "enriched_at_unix_seconds": time.time(),
    }

    out = args.output
    if out is None:
        out = args.capsule / "enrichment.json"
    # Never overwrite an existing enrichment: timestamp a new immutable file.
    if out.exists():
        out = out.with_name(f"{out.stem}-{int(time.time())}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[enrich] report written: {out}")
    print(
        "\n== CMuon hard-fail capsule enrichment summary =="
    )
    print(
        f"input: {report['fqn']}#chunk{report['chunk']} {report['shape']} "
        f"input_rms={rms_v:.6e} zero_frac={zero_fraction:.4f}"
    )
    print(
        "replay FP32: "
        + " / ".join(f"{r['delta_rms']:.6e}({r['failure_reason']})" for r in repeats_fp32)
        + f"  recorded={rec_rms} ({recorded_reason})  preserved={classification_preserved}"
    )
    print(
        f"spectrum: source={spectrum['source']} stable_rank={stable_rank} "
        f"top32_energy={top32_energy} rank_implied={rank_implied}"
    )
    print(f"LOW_RANK_HYPOTHESIS = {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
