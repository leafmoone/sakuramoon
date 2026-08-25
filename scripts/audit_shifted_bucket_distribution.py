#!/usr/bin/env python3
"""Read-only shifted-bucket distribution audit (spec P11).

Replays the pure-geometry ``plan_shifted_bucket`` decisions over a large
deterministic synthetic source-size population (>= 1,000,000 samples by
default), with the 17-bucket assignment of the loaded config, and reports:

- selection / application probabilities against the configured policy
  probability (no dynamic probability is ever introduced);
- the fixed 5-bin actual-zoom histogram, min/max/mean zoom, and the
  fraction of applied samples reaching z >= 1.08 (significance gate);
- the minimum final crop retention (guard gate, >= min_crop_retention);
- normalized offset spread (mean absolute x/y, both-axes fraction);
- per-bucket applied counts;
- cumulative spatial-equivalent passes = applied / 11_270_000 (the
  configured full-pass sample count of the G1 dataset).

The replay is offline and read-only: it reads a config TOML only, needs no
GPU, no data service, and no network, and writes nothing unless ``--out``
names a report file. The per-sample seeds are an isolated audit stream
(master seed x 7919 + index), independent of the training RngIdentity
domains; the audit verifies distributional invariants, not bit-exact
training RNG (that is the golden regression's job).

Exits 0 when every gate passes and 1 otherwise.

Usage (from the repository root, ``uv run --no-sync``)::

    uv run --no-sync python scripts/audit_shifted_bucket_distribution.py \
        --config config/train_g1_spatial_p25.toml --samples 1000000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from sakuramoon.config import load_config
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.buckets import (
    BucketRejection,
    BucketShape,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
    SpatialCropPolicy,
    plan_shifted_bucket,
    zoom_histogram_bin,
)

PASS_SAMPLES = 11_270_000
DEFAULT_SAMPLES = 1_000_000
MIN_SAMPLES = 1_000_000
DEFAULT_ZOOM_HIGH_GATE = 1.08
DEFAULT_MIN_ZOOM_HIGH_FRACTION = 0.10
AUDIT_MASTER_MULTIPLIER = 7919
_SEED_SPACE = 1 << 61


def _synthetic_source(index: int, stream: random.Random) -> tuple[int, int]:
    """One deterministic danbooru-like source size (width, height).

    Aspect is log-uniform in [0.25, 4] (the locked max_aspect_ratio) and the
    short edge is log-uniform in [128, 1024], biased toward the dense mid
    range of the real dataset.
    """

    aspect = 2.0 ** stream.uniform(-2.0, 2.0)
    short_edge = max(16, int(2.0 ** stream.uniform(7.0, 10.0)))
    long_edge = max(short_edge, round(short_edge * aspect))
    if aspect >= 1.0:
        return long_edge, short_edge
    return short_edge, long_edge


def _sample_seeds(
    master: int, index: int, stream: random.Random
) -> tuple[int, int, int, int]:
    """The four isolated per-sample seed domains of one replayed sample."""

    base = ((master + 1) * AUDIT_MASTER_MULTIPLIER + index) % _SEED_SPACE
    return base, base + 1, base + 2, base + 3


def replay(
    config: RuntimeConfig,
    buckets: tuple[BucketShape, ...],
    policy: SpatialCropPolicy,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Deterministic offline replay of the shifted-bucket plan."""

    min_retention = config.data.image.min_crop_retention
    stream = random.Random(seed)
    fallback_reasons = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
    zoom_histogram = {label: 0 for label in ZOOM_HISTOGRAM_LABELS}
    generated = 0
    rejected_no_upscale = 0
    rejected_retention = 0
    assigned = 0
    selected = 0
    applied = 0
    zoom_sum = 0.0
    zoom_min = float("inf")
    zoom_max = 0.0
    retention_min = 1.0
    offset_x_sum = 0.0
    offset_y_sum = 0.0
    both_axes = 0
    bucket_applied: dict[tuple[int, int], int] = {
        (shape.width, shape.height): 0 for shape in buckets
    }
    for index in range(samples):
        width, height = _synthetic_source(index, stream)
        generated += 1
        result = assign_bucket(
            width,
            height,
            buckets,
            min_crop_retention=min_retention,
        )
        if isinstance(result, BucketRejection):
            if result.reason == "no_upscale":
                rejected_no_upscale += 1
            else:
                rejected_retention += 1
            continue
        assigned += 1
        policy_seed, zoom_seed, offset_x_seed, offset_y_seed = _sample_seeds(
            seed, index, stream
        )
        plan = plan_shifted_bucket(
            result,
            policy,
            policy_seed=policy_seed,
            zoom_seed=zoom_seed,
            offset_x_seed=offset_x_seed,
            offset_y_seed=offset_y_seed,
        )
        if plan.fallback_reason != "not_selected":
            selected += 1
        if plan.applied:
            applied += 1
            zoom = plan.actual_equivalent_zoom
            zoom_sum += zoom
            zoom_min = min(zoom_min, zoom)
            zoom_max = max(zoom_max, zoom)
            zoom_histogram[ZOOM_HISTOGRAM_LABELS[zoom_histogram_bin(zoom)]] += 1
            retention_min = min(retention_min, plan.final_crop_retention)
            offset_x_sum += abs(plan.normalized_offset_x)
            offset_y_sum += abs(plan.normalized_offset_y)
            if (
                plan.canvas_width - result.bucket.width > 0
                and plan.canvas_height - result.bucket.height > 0
            ):
                both_axes += 1
            bucket_applied[(result.bucket.width, result.bucket.height)] += 1
        else:
            fallback_reasons[plan.fallback_reason] += 1
    report: dict[str, object] = {
        "schema": "sakuramoon.spatial_crop_audit.v1",
        "run_id": config.run.run_id,
        "samples_generated": generated,
        "samples_rejected": rejected_no_upscale + rejected_retention,
        "rejection_reasons": {
            "no_upscale": rejected_no_upscale,
            "retention": rejected_retention,
        },
        "samples_assigned": assigned,
        "seed": seed,
        "source_mode": "synthetic",
        "policy": {
            "enabled": policy.enabled,
            "probability": policy.probability,
            "min_equivalent_zoom": policy.min_equivalent_zoom,
            "max_equivalent_zoom": policy.max_equivalent_zoom,
            "min_crop_retention": min_retention,
        },
        "selected_probability": (selected / assigned) if assigned else 0.0,
        "applied_probability": (applied / assigned) if assigned else 0.0,
        "spatial_equivalent_passes": applied / PASS_SAMPLES,
        "applied_per_full_pass_estimate": (applied / assigned) * PASS_SAMPLES
        if assigned
        else 0.0,
        "fallback_reasons": dict(fallback_reasons),
        "zoom_histogram": dict(zoom_histogram),
        "zoom": {
            "min": zoom_min if applied else 0.0,
            "mean": (zoom_sum / applied) if applied else 0.0,
            "max": zoom_max,
        },
        "retention_min_of_applied": retention_min,
        "offsets": {
            "mean_abs_x": (offset_x_sum / applied) if applied else 0.0,
            "mean_abs_y": (offset_y_sum / applied) if applied else 0.0,
            "both_axes_fraction": (both_axes / applied) if applied else 0.0,
        },
        "bucket_applied": {
            f"{width}x{height}": count
            for (width, height), count in sorted(bucket_applied.items())
        },
    }
    return report


def zoom_gate(report: dict[str, object], zoom_high_gate: float) -> float:
    """Fraction of applied samples whose actual zoom reaches the gate.

    Bins are [1.00,1.02) [1.02,1.04) [1.04,1.06) [1.06,1.08) [1.08,1.101];
    a gate at or above 1.08 is served exactly by the top bin, while a gate
    inside the [1.06, 1.08) bin counts that whole bin (a documented
    over-count for gates below 1.08).
    """

    histogram = cast(dict[str, int], report["zoom_histogram"])
    applied = int(sum(histogram.values()))
    if applied == 0:
        return 0.0
    if zoom_high_gate < 1.08:
        gate_bins = (
            histogram[ZOOM_HISTOGRAM_LABELS[3]]
            + histogram[ZOOM_HISTOGRAM_LABELS[4]]
        )
    else:
        gate_bins = histogram[ZOOM_HISTOGRAM_LABELS[4]]
    return gate_bins / applied


def run_gates(
    report: dict[str, object],
    *,
    zoom_high_gate: float,
    min_zoom_high_fraction: float,
) -> dict[str, bool]:
    policy = cast(dict[str, object], report["policy"])
    zoom = cast(dict[str, object], report["zoom"])
    gates = {
        "zoom_max_within_bounds": cast(float, zoom["max"])
        <= cast(float, policy["max_equivalent_zoom"]) + 1e-9,
        "retention_guard_held": cast(float, report["retention_min_of_applied"])
        >= cast(float, policy["min_crop_retention"]) - 1e-9,
        "zoom_high_significant": zoom_gate(report, zoom_high_gate)
        >= min_zoom_high_fraction,
    }
    return gates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="config TOML to audit (e.g. config/train_g1_spatial_p25.toml)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: the parent of scripts/)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"samples to replay (minimum {MIN_SAMPLES})",
    )
    parser.add_argument("--seed", type=int, default=0, help="audit master seed")
    parser.add_argument(
        "--zoom-high-gate",
        type=float,
        default=DEFAULT_ZOOM_HIGH_GATE,
        help="zoom value the significance gate requires (default 1.08)",
    )
    parser.add_argument(
        "--min-zoom-high-fraction",
        type=float,
        default=DEFAULT_MIN_ZOOM_HIGH_FRACTION,
        help=(
            "minimum fraction of applied samples that must reach the zoom "
            f"gate for it to count as significant (default "
            f"{DEFAULT_MIN_ZOOM_HIGH_FRACTION})"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the JSON report here (default: stdout only)",
    )
    args = parser.parse_args(argv)
    if args.samples < MIN_SAMPLES:
        parser.error(f"--samples must be at least {MIN_SAMPLES}")
    if args.seed < 0:
        parser.error("--seed must be a nonnegative integer")

    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    loaded = load_config(
        config_path,
        config_root=repo_root / "config",
        validate_secrets=False,
    )
    config = loaded.config
    buckets = scale_buckets(
        generate_base_buckets(config.data.buckets), config.stage.resolution
    )
    policy = SpatialCropPolicy.from_config(
        config.data.spatial_crop,
        min_crop_retention=config.data.image.min_crop_retention,
    )
    report = replay(config, buckets, policy, samples=args.samples, seed=args.seed)
    gates = run_gates(
        report,
        zoom_high_gate=args.zoom_high_gate,
        min_zoom_high_fraction=args.min_zoom_high_fraction,
    )
    report["zoom_high_gate"] = args.zoom_high_gate
    report["min_zoom_high_fraction"] = args.min_zoom_high_fraction
    report["zoom_high_gate_fraction"] = zoom_gate(report, args.zoom_high_gate)
    report["gates"] = gates
    document = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(document + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    print(document)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
