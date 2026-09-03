#!/usr/bin/env python3
"""iREPA Phase 3 frozen PE-Spatial teacher — standalone HCU performance audit.

Runs only on the DCU/HCU audit box (never in the pytest gate, never in
main training).  Measures the frozen teacher alone:

  - forward median / p95 latency per (shape, batch)
  - peak allocated / reserved memory
  - output token count and dtype
  - maximum viable batch at 512x512 (reported, never auto-microbatched)
  - bitwise forward determinism
  - isolated eager vs torch.compile benchmark (report data only)

Writes ``reports/irepa-phase3-teacher-audit.json`` and ``.md`` under the
repository root.  The report contains no model weights and no secrets.

Usage:
    python scripts/irepa_phase3_teacher_audit.py \
        [--repo-root PATH] [--device cuda:0] [--batches 4,16] \
        [--warmup 10] [--iters 30]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return float(ordered[rank - 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batches", type=str, default="4,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if args.warmup < 10 or args.iters < 30:
        raise SystemExit("audit contract requires warmup >= 10 and iters >= 30")
    repo_root: Path = args.repo_root
    device = torch.device(args.device)
    batches = [int(item) for item in args.batches.split(",") if item.strip()]

    from sakuramoon.config import load_config
    from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
    from sakuramoon.encoders.pe_spatial import (
        FrozenPESpatialEncoder,
        prepare_teacher_targets,
    )

    loaded = load_config(
        Path("train_g1.toml"),
        config_root=repo_root / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
            "WANDB_API_KEY": "synthetic-wandb-secret",
        },
    )
    base = generate_base_buckets(loaded.config.data.buckets)
    edge512 = scale_buckets(base, 512)
    landscape = max(edge512, key=lambda shape: shape.width / shape.height)
    portrait = (landscape.width, landscape.height)
    shapes: list[tuple[int, int]] = [
        (256, 256),
        (512, 512),
        (landscape.height, landscape.width),
        portrait,
    ]

    print(f"[audit] device={device} batches={batches} shapes={shapes}", flush=True)
    teacher = FrozenPESpatialEncoder.load_asset(
        repo_root, "model/pe_spatial_b16_512", device=device
    )
    print("[audit] teacher loaded (frozen, bf16)", flush=True)

    cases: list[dict[str, object]] = []
    for height, width in shapes:
        for batch in batches:
            generator = torch.Generator(device=device).manual_seed(0)
            images = (
                2.0
                * torch.rand(
                    batch, 3, height, width, generator=generator, device=device
                )
                - 1.0
            ).to(torch.bfloat16)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(args.warmup):
                    output = prepare_teacher_targets(teacher, images)
            torch.cuda.synchronize(device)
            elapsed_ms: list[float] = []
            with torch.no_grad():
                for _ in range(args.iters):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    output = prepare_teacher_targets(teacher, images)
                    torch.cuda.synchronize(device)
                    elapsed_ms.append((time.perf_counter() - start) * 1000.0)
            grid_h, grid_w = height // 16, width // 16
            cases.append(
                {
                    "shape": f"{height}x{width}",
                    "grid": [grid_h, grid_w],
                    "batch": batch,
                    "forward_median_ms": round(_median(elapsed_ms), 3),
                    "forward_p95_ms": round(_p95(elapsed_ms), 3),
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                    "output_tokens": batch * grid_h * grid_w,
                    "output_feature_width": int(output.patch_features.shape[-1]),
                    "output_dtype": str(output.patch_features.dtype),
                    "output_finite": bool(
                        torch.isfinite(output.patch_features.float()).all()
                    ),
                }
            )
            print(
                f"[audit] {height}x{width} b{batch}: median="
                f"{cases[-1]['forward_median_ms']}ms p95="
                f"{cases[-1]['forward_p95_ms']}ms",
                flush=True,
            )

    # Bitwise determinism at 512x512, batch 1.
    generator = torch.Generator(device=device).manual_seed(3)
    ref_images = (
        2.0
        * torch.rand(1, 3, 512, 512, generator=generator, device=device)
        - 1.0
    ).to(torch.bfloat16)
    with torch.no_grad():
        det_first = prepare_teacher_targets(teacher, ref_images)
        det_second = prepare_teacher_targets(teacher, ref_images)
    deterministic = bool(
        torch.equal(det_first.patch_features, det_second.patch_features)
    )
    print(f"[audit] deterministic_bitwise={deterministic}", flush=True)

    # Maximum viable batch at 512x512 (report only; no auto-microbatching).
    max_viable_batch: int | None = None
    last_error = ""
    for batch in sorted(set(batches + [20, 32, 48, 64])):
        probe: torch.Tensor | None = None
        try:
            generator = torch.Generator(device=device).manual_seed(0)
            probe = (
                2.0
                * torch.rand(
                    batch, 3, 512, 512, generator=generator, device=device
                )
                - 1.0
            ).to(torch.bfloat16)
            torch.cuda.synchronize(device)
            with torch.no_grad():
                prepare_teacher_targets(teacher, probe)
            torch.cuda.synchronize(device)
            max_viable_batch = batch
        except RuntimeError as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            break
        finally:
            if probe is not None:
                del probe
            torch.cuda.empty_cache()
    print(
        f"[audit] max viable batch @512x512={max_viable_batch}"
        + (f" (stopped: {last_error})" if last_error else ""),
        flush=True,
    )

    # Isolated eager vs torch.compile benchmark (report data only).
    compile_report: dict[str, object] = {"status": "not-run"}
    try:
        probe_batch = batches[0] if batches else 4
        generator = torch.Generator(device=device).manual_seed(0)
        compile_input = (
            2.0
            * torch.rand(
                probe_batch, 3, 512, 512, generator=generator, device=device
            )
            - 1.0
        ).to(torch.bfloat16)

        from collections.abc import Callable

        def _timed(fn: Callable[[], object], n: int) -> list[float]:
            samples: list[float] = []
            with torch.no_grad():
                for _ in range(n):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    fn()
                    torch.cuda.synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000.0)
            return samples

        eager_ms = _timed(
            lambda: prepare_teacher_targets(teacher, compile_input), 10
        )
        compiled_encoder = torch.compile(teacher)
        torch.cuda.synchronize(device)
        compile_start = time.perf_counter()
        with torch.no_grad():
            compiled_encoder(compile_input)
        torch.cuda.synchronize(device)
        first_compile_wall_s = time.perf_counter() - compile_start
        compiled_ms = _timed(lambda: compiled_encoder(compile_input), 10)
        speedup = (
            _median(eager_ms) / _median(compiled_ms)
            if _median(compiled_ms) > 0
            else float("inf")
        )

        # Recompile cost across the 17-shape 512 family (batch probe_batch).
        recompile_shapes: list[dict[str, object]] = []
        for shape in edge512:
            generator = torch.Generator(device=device).manual_seed(0)
            shape_input = (
                2.0
                * torch.rand(
                    probe_batch, 3, shape.height, shape.width,
                    generator=generator, device=device,
                )
                - 1.0
            ).to(torch.bfloat16)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                compiled_encoder(shape_input)
            torch.cuda.synchronize(device)
            recompile_shapes.append(
                {
                    "shape": f"{shape.height}x{shape.width}",
                    "first_call_ms": round((time.perf_counter() - start) * 1000.0, 1),
                }
            )
        compile_report = {
            "status": "ok",
            "eager_512_median_ms": round(_median(eager_ms), 3),
            "compiled_512_median_ms": round(_median(compiled_ms), 3),
            "first_compile_wall_s": round(first_compile_wall_s, 1),
            "speedup": round(speedup, 4),
            "recompiles_observed": sum(
                1 for item in recompile_shapes
                if float(item["first_call_ms"]) > 1000.0
            ),
            "recompile_first_calls": recompile_shapes,
            "recommended_v1_backend": (
                "eager" if speedup < 1.10 else "torch.compile"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - audit must report, not crash
        compile_report = {
            "status": f"failed: {type(exc).__name__}: {str(exc)[:300]}",
        }
    print(f"[audit] compile benchmark: {compile_report.get('status')}", flush=True)

    report = {
        "audit": "irepa-phase3-frozen-teacher",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "device": {
            "name": str(device),
            "cuda_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
        "teacher": {
            "model": "facebook/PE-Spatial-B16-512",
            "parameter_dtype": "bfloat16",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "frozen": True,
            "eval": True,
            "grad_graph": False,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "cases": cases,
        "deterministic_bitwise": deterministic,
        "max_viable_batch_512": max_viable_batch,
        "compile_benchmark": compile_report,
    }

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "irepa-phase3-teacher-audit.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# iREPA Phase 3 — frozen PE-Spatial teacher HCU audit",
        "",
        f"- device: `{torch.cuda.get_device_name(device)}` (torch {torch.__version__})",
        f"- generated: {report['generated_at_utc']}",
        f"- deterministic (bitwise, repeated forward): {deterministic}",
        f"- max viable batch @512x512: {max_viable_batch}",
        f"- compile recommendation: {compile_report.get('recommended_v1_backend', 'n/a')}",
        "",
        "| shape | grid | batch | median ms | p95 ms | peak alloc | peak reserved | tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        grid = case["grid"]
        lines.append(
            f"| {case['shape']} | {grid[0]}x{grid[1]} | {case['batch']} "
            f"| {case['forward_median_ms']} | {case['forward_p95_ms']} "
            f"| {case['peak_allocated_bytes'] / (1 << 20):.1f} MiB "
            f"| {case['peak_reserved_bytes'] / (1 << 20):.1f} MiB "
            f"| {case['output_tokens']} |"
        )
    md_path = reports_dir / "irepa-phase3-teacher-audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] wrote {json_path}", flush=True)
    print(f"[audit] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
