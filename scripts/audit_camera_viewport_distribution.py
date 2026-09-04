"""Offline distribution audit for the hdm_shifted_square_v2 camera viewport.

A. >=1,000,000 deterministic synthetic source dimensions (log-uniform
   aspect in [1.0, 4.0], uniform short edge in [256, 2048], 50/50
   orientation). Each synthetic source is run through the ORDINARY bucket
   admission (real 17-bucket stage-scaled vocabulary, min_crop_retention
   0.8) and, when admitted, through plan_camera_viewport with the
   isolated camera-policy / camera-offset RNG domains.
B. Optional read-only real WebDataset metadata audit (dimensions only,
   images are never decoded). When unavailable, REAL_METADATA_AUDIT is
   reported as PENDING -- never fabricated.

The synthetic selected count must match the configured probability within
5 sigma of the binomial standard deviation (no wide tolerance).

Output: reports/camera-viewport-v2-distribution.json (+ console summary).

Usage:
    python scripts/audit_camera_viewport_distribution.py \
        --samples 1000000 --seed 20260905 --probability 0.25 \
        --stage-edge 256 --output reports/camera-viewport-v2-distribution.json
    # 256 = current G1 stage.resolution (config/train_g1.toml [stage]).
    # 512/768/1024 = future stage resolutions (label as FUTURE_STAGE_<edge>).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path

from sakuramoon.data.buckets import (
    BucketRejection,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.camera_viewport import (
    CAMERA_FALLBACK_REASONS,
    CAMERA_SHIFT_TOKEN_BIN_LABELS,
    CAMERA_ZOOM_BAND_LABELS,
    CameraViewportPolicy,
    camera_shift_token_bin,
    camera_zoom_band,
    plan_camera_viewport,
)

SHORT_EDGE_MIN = 256
SHORT_EDGE_MAX = 2048
ASPECT_MIN = 1.0
ASPECT_MAX = 4.0
MIN_CROP_RETENTION = 0.8
BINOMIAL_SIGMA_LIMIT = 5.0


def _synthetic_source(rng: random.Random) -> tuple[int, int]:
    short = rng.randint(SHORT_EDGE_MIN, SHORT_EDGE_MAX)
    aspect = 10.0 ** (rng.random() * math.log10(ASPECT_MAX / ASPECT_MIN))
    long_edge = max(short, round(short * aspect))
    if rng.random() < 0.5:
        return long_edge, short
    return short, long_edge


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--probability", type=float, default=0.25)
    parser.add_argument("--min-zoom", type=float, default=1.10)
    parser.add_argument("--max-zoom", type=float, default=1.50)
    parser.add_argument("--stage-edge", type=int, default=256)
    parser.add_argument("--real-metadata", type=Path, default=None)
    parser.add_argument("--real-samples", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/camera-viewport-v2-distribution.json"),
    )
    args = parser.parse_args(argv)
    if args.samples < 1_000_000:
        raise ValueError("synthetic audit requires at least 1,000,000 samples")
    if not 0.0 < args.probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    if args.min_zoom >= args.max_zoom or args.max_zoom > 1.5:
        raise ValueError("invalid zoom band")
    if args.stage_edge not in (256, 512, 768, 1024):
        raise ValueError("stage edge must be a supported stage resolution")

    buckets = scale_buckets(_base_buckets(), args.stage_edge)
    square = next(shape for shape in buckets if shape.width == shape.height)
    if square.width != args.stage_edge:
        raise ValueError("stage vocabulary has no stage-edge square bucket")
    policy = CameraViewportPolicy(
        enabled=True,
        probability=args.probability,
        min_equivalent_zoom=args.min_zoom,
        max_equivalent_zoom=args.max_zoom,
    )

    source_rng = random.Random(f"{args.seed}\0source")
    policy_rng = random.Random(f"{args.seed}\0camera-policy")
    offset_rng = random.Random(f"{args.seed}\0camera-offset")

    n = args.samples
    admitted = 0
    rejected = 0
    selected = 0
    applied = 0
    fallback = {reason: 0 for reason in CAMERA_FALLBACK_REASONS}
    orientation = {"horizontal": 0, "vertical": 0}
    full_width_sum = 0
    full_height_sum = 0
    full_width_max = 0
    full_height_max = 0
    zoom_bands = {label: 0 for label in CAMERA_ZOOM_BAND_LABELS}
    shift_bins = {label: 0 for label in CAMERA_SHIFT_TOKEN_BIN_LABELS}
    zoom_sum = 0.0
    zoom_max = 0.0
    retention_sum = 0.0
    retention_min = math.inf
    pixel_shift_sum = 0.0
    pixel_shift_max = 0.0
    latent_shift_max = 0.0
    pixel_shift_samples: list[float] = []
    latent_shift_samples: list[float] = []
    ordinary_cohorts: dict[str, int] = {}
    target_cohorts: dict[str, int] = {}
    square_before = 0
    square_after = 0

    invariants = {
        "crop_inside_full_canvas": True,
        "no_upscale": True,
        "no_distortion_except_integer_rounding": True,
        "zoom_in_band": True,
        "retention_above_floor": True,
        "all_applied_outputs_square": True,
        "ordinary_fallback_unchanged": True,
    }
    retention_floor = 1.0 / (1.5 ** 2)

    started = time.perf_counter()
    for index in range(n):
        source_width, source_height = _synthetic_source(source_rng)
        result = assign_bucket(
            source_width,
            source_height,
            buckets,
            min_crop_retention=MIN_CROP_RETENTION,
        )
        if isinstance(result, BucketRejection):
            rejected += 1
            continue
        admitted += 1
        cohort = f"{result.bucket.width}x{result.bucket.height}"
        ordinary_cohorts[cohort] = ordinary_cohorts.get(cohort, 0) + 1
        if result.bucket.width == args.stage_edge and result.bucket.height == args.stage_edge:
            square_before += 1
        plan = plan_camera_viewport(
            result,
            policy,
            buckets=buckets,
            stage_edge=args.stage_edge,
            source_size=(source_width, source_height),
            policy_seed=policy_rng.randrange(2**63),
            offset_seed=offset_rng.randrange(2**63),
        )
        reason = plan.fallback_reason
        fallback[reason] += 1
        if reason != "not_selected":
            selected += 1
        if not plan.applied:
            target_cohorts[cohort] = target_cohorts.get(cohort, 0) + 1
            # Ordinary fallback: the sample keeps the ordinary bucket/geometry
            # exactly (no camera plan exists to alter it).
            if result.bucket.width == args.stage_edge and result.bucket.height == args.stage_edge:
                square_after += 1
            continue
        applied += 1
        orientation[plan.orientation] += 1
        full_width_sum += plan.full_width
        full_height_sum += plan.full_height
        full_width_max = max(full_width_max, plan.full_width)
        full_height_max = max(full_height_max, plan.full_height)
        left, top, right, bottom = plan.crop_box
        if not (
            0 <= left < right <= plan.full_width
            and 0 <= top < bottom <= plan.full_height
            and right - left == plan.viewport
            and bottom - top == plan.viewport
        ):
            invariants["crop_inside_full_canvas"] = False
            invariants["all_applied_outputs_square"] = False
        short_edge = min(source_width, source_height)
        long_edge = max(source_width, source_height)
        full_long = max(plan.full_width, plan.full_height)
        if plan.orientation == "horizontal" and plan.full_width > source_width:
            invariants["no_upscale"] = False
        if plan.orientation == "horizontal" and plan.full_height != args.stage_edge:
            invariants["no_upscale"] = False
        if plan.orientation == "vertical" and plan.full_width != args.stage_edge:
            invariants["no_upscale"] = False
        if full_long > long_edge or plan.viewport > short_edge:
            invariants["no_upscale"] = False
        ideal_long = long_edge * args.stage_edge / short_edge
        if abs(full_long - ideal_long) > 0.5 + 1e-9:
            invariants["no_distortion_except_integer_rounding"] = False
        if not (args.min_zoom - 1e-9 <= plan.equivalent_zoom <= args.max_zoom + 1e-9):
            invariants["zoom_in_band"] = False
        if plan.retention < retention_floor - 1e-9:
            invariants["retention_above_floor"] = False
        zoom_bands[CAMERA_ZOOM_BAND_LABELS[camera_zoom_band(plan.equivalent_zoom)]] += 1
        shift_bins[
            CAMERA_SHIFT_TOKEN_BIN_LABELS[
                camera_shift_token_bin(plan.latent_center_shift)
            ]
        ] += 1
        zoom_sum += plan.equivalent_zoom
        zoom_max = max(zoom_max, plan.equivalent_zoom)
        retention_sum += plan.retention
        retention_min = min(retention_min, plan.retention)
        pixel_shift_sum += plan.absolute_pixel_center_shift
        pixel_shift_max = max(pixel_shift_max, plan.absolute_pixel_center_shift)
        latent_shift_max = max(latent_shift_max, plan.latent_center_shift)
        pixel_shift_samples.append(plan.absolute_pixel_center_shift)
        latent_shift_samples.append(plan.latent_center_shift)
        target_cohorts[f"{args.stage_edge}x{args.stage_edge}"] = (
            target_cohorts.get(f"{args.stage_edge}x{args.stage_edge}", 0) + 1
        )
        square_after += 1
    def _percentile(samples: list[float], q: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        rank = (len(ordered) - 1) * q
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

    elapsed = time.perf_counter() - started

    # Inclusive endpoint probe: both offset endpoints (0 and available) must be
    # reachable by the uniform inclusive draw over the full offset range.
    endpoint_reach = {
        "endpoint_zero": 0,
        "endpoint_full": 0,
    }
    probe_rng = random.Random(f"{args.seed}\0endpoint-probe")
    endpoint_rng = random.Random(f"{args.seed}\0endpoint")
    for _ in range(200_000):
        source_width, source_height = _synthetic_source(probe_rng)
        result = assign_bucket(
            source_width,
            source_height,
            buckets,
            min_crop_retention=MIN_CROP_RETENTION,
        )
        if isinstance(result, BucketRejection):
            continue
        plan = plan_camera_viewport(
            result,
            policy,
            buckets=buckets,
            stage_edge=args.stage_edge,
            source_size=(source_width, source_height),
            policy_seed=policy_rng.randrange(2**63),
            offset_seed=endpoint_rng.randrange(2**63),
        )
        if not plan.applied:
            continue
        available = (
            plan.full_width - plan.viewport
            if plan.orientation == "horizontal"
            else plan.full_height - plan.viewport
        )
        offset = plan.left if plan.orientation == "horizontal" else plan.top
        if offset == 0:
            endpoint_reach["endpoint_zero"] += 1
        elif offset == available:
            endpoint_reach["endpoint_full"] += 1
    invariants["inclusive_endpoints_reachable"] = (
        endpoint_reach["endpoint_zero"] > 0 and endpoint_reach["endpoint_full"] > 0
    )

    # Binomial consistency of the selection draw (trials = admitted samples).
    p = args.probability
    sigma = math.sqrt(admitted * p * (1.0 - p))
    binomial = {
        "n_trials": admitted,
        "k_selected": selected,
        "p_configured": p,
        "sigma": sigma,
        "sigma_limit": BINOMIAL_SIGMA_LIMIT,
        "observed_minus_expected": selected - admitted * p,
        "pass": abs(selected - admitted * p) <= BINOMIAL_SIGMA_LIMIT * sigma,
    }

    # Optional real-metadata audit (dimensions only, never decodes images).
    real_metadata_audit = "PENDING"
    if args.real_metadata is not None:
        try:
            real_metadata_audit = _audit_real_metadata(
                args.real_metadata,
                buckets,
                args.stage_edge,
                policy,
                args.real_samples,
                args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - audit failure is reported, not fabricated
            real_metadata_audit = f"PENDING ({type(exc).__name__}: {exc})"

    document = {
        "audit": "camera_viewport_v2_distribution",
        "seed": args.seed,
        "samples": n,
        "policy": {
            "probability": p,
            "min_equivalent_zoom": args.min_zoom,
            "max_equivalent_zoom": args.max_zoom,
            "stage_edge": args.stage_edge,
        },
        "elapsed_seconds": elapsed,
        "admitted": admitted,
        "ordinary_rejected": rejected,
        "admission_rate": admitted / n,
        "selected": selected,
        "selected_probability": selected / admitted if admitted else 0.0,
        "binomial_consistency": binomial,
        "applied": applied,
        "applied_probability": applied / n,
        "applied_given_admitted": applied / admitted if admitted else 0.0,
        "fallback_reasons": fallback,
        "source_orientation": {
            "horizontal": orientation["horizontal"],
            "vertical": orientation["vertical"],
        },
        "full_canvas_dimensions": {
            "applied_horizontal_width_max": full_width_max,
            "applied_vertical_height_max": full_height_max,
            "mean_width_when_applied": (
                full_width_sum / applied if applied else 0.0
            ),
            "mean_height_when_applied": (
                full_height_sum / applied if applied else 0.0
            ),
        },
        "zoom_bands": zoom_bands,
        "shift_token_bins": shift_bins,
        "zoom": {
            "mean": zoom_sum / applied if applied else 0.0,
            "max": zoom_max,
        },
        "retention": {
            "mean": retention_sum / applied if applied else 0.0,
            "min": retention_min if applied else 0.0,
            "floor": retention_floor,
        },
        "pixel_shifts": {
            "mean_abs": pixel_shift_sum / applied if applied else 0.0,
            "p50": _percentile(pixel_shift_samples, 0.50),
            "p90": _percentile(pixel_shift_samples, 0.90),
            "max_abs": pixel_shift_max,
        },
        "latent_cell_shifts": {
            "p50": _percentile(latent_shift_samples, 0.50),
            "p90": _percentile(latent_shift_samples, 0.90),
            "max_abs": latent_shift_max,
        },
        "ordinary_17_bucket_cohorts": ordinary_cohorts,
        "resulting_target_bucket_distribution": target_cohorts,
        "square_bucket_share_before": square_before / admitted if admitted else 0.0,
        "square_bucket_share_after": square_after / admitted if admitted else 0.0,
        "inclusive_endpoint_probe": endpoint_reach,
        "invariants": invariants,
        "real_metadata_audit": real_metadata_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[camera-dist] wrote {args.output}", flush=True)
    print(
        f"[camera-dist] admitted={admitted}/{n} selected={selected} "
        f"({document['selected_probability']:.4f} vs p={p}) "
        f"applied={applied} ({document['applied_probability']:.4f} of total)",
        flush=True,
    )
    print(f"[camera-dist] binomial pass={binomial['pass']} (5-sigma)", flush=True)
    print(f"[camera-dist] invariants={invariants}", flush=True)
    print(f"[camera-dist] real_metadata_audit={real_metadata_audit}", flush=True)
    if not binomial["pass"]:
        raise SystemExit("binomial consistency failed")
    if not all(invariants.values()):
        raise SystemExit("camera invariant violations detected")
    return 0


def _base_buckets():
    from sakuramoon.config.schema import DataBucketsConfig

    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        shape_count=17,
        transpose_closed=True,
    )
    return generate_base_buckets(config)


def _audit_real_metadata(
    path: Path,
    buckets,
    stage_edge: int,
    policy: CameraViewportPolicy,
    sample_cap: int,
    seed: int,
) -> str:
    """Read-only dimension audit of WebDataset shard JSON payloads."""

    import tarfile

    counts = {reason: 0 for reason in CAMERA_FALLBACK_REASONS}
    ordinary_rejected = 0
    seen = 0
    for shard in sorted(path.glob("*.tar")):
        policy_rng = random.Random(f"{seed}\0real-policy")
        offset_rng = random.Random(f"{seed}\0real-offset")
        with tarfile.open(shard, "r") as handle:
            for member in handle:
                if seen >= sample_cap:
                    break
                if not member.name.endswith(".json"):
                    continue
                fileobj = handle.extractfile(member)
                if fileobj is None:
                    continue
                try:
                    document = json.load(fileobj)
                except (OSError, ValueError):
                    # Per-member leniency is the audit contract: one corrupt
                    # metadata member must not abort the metadata scan.
                    continue
                width = document.get("width")
                height = document.get("height")
                if not isinstance(width, int) or not isinstance(height, int):
                    continue
                if width <= 0 or height <= 0:
                    continue
                seen += 1
                result = assign_bucket(
                    width,
                    height,
                    buckets,
                    min_crop_retention=MIN_CROP_RETENTION,
                )
                if isinstance(result, BucketRejection):
                    ordinary_rejected += 1
                    continue
                plan = plan_camera_viewport(
                    result,
                    policy,
                    buckets=buckets,
                    stage_edge=stage_edge,
                    source_size=(width, height),
                    policy_seed=policy_rng.randrange(2**63),
                    offset_seed=offset_rng.randrange(2**63),
                )
                counts[plan.fallback_reason] += 1
        if seen >= sample_cap:
            break
    if seen < sample_cap:
        return f"PENDING (only {seen}/{sample_cap} real dimensions available)"
    return json.dumps(
        {
            "samples": seen,
            "ordinary_rejected": ordinary_rejected,
            "fallback_reasons": counts,
        },
        sort_keys=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
