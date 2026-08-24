#!/usr/bin/env python3
"""Read-only spatial-crop telemetry aggregation for the canary gates.

Reads the durable training-metric JSONL (schema version 9) and reports the
fixed spatial-crop aggregates used by the shifted-bucket canary gates:

* ``applied_probability``     cumulative applied samples / cumulative samples
* ``selected_probability``    cumulative selected samples / cumulative samples
* ``spatial_equivalent_passes`` cumulative applied samples / 11_270_000
* fallback-reason totals, zoom histogram totals, zoom/offset means

The metric files are opened strictly read-only; nothing in the run tree is
modified.

Usage:
    python scripts/aggregate_spatial_crop_metrics.py --metrics runs/g1/telemetry/metrics.jsonl
    python scripts/aggregate_spatial_crop_metrics.py --metrics runs/g1/telemetry --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
)
from sakuramoon.telemetry.metrics import TRAINING_METRIC_SCHEMA_VERSION

# One full pass over the training budget, in samples.
DEFAULT_PASS_SAMPLES = 11_270_000


def _iter_metric_files(metrics: Path) -> Iterator[Path]:
    if metrics.is_dir():
        files = sorted(path for path in metrics.iterdir() if path.suffix == ".jsonl")
        if not files:
            raise FileNotFoundError(f"no .jsonl metric files under {metrics}")
        yield from files
    else:
        if not metrics.is_file():
            raise FileNotFoundError(f"metric file not found: {metrics}")
        yield metrics


def _read_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield record


def aggregate(metrics: Path, pass_samples: int = DEFAULT_PASS_SAMPLES) -> dict[str, object]:
    """Aggregate the fixed spatial-crop fields from durable metric records."""

    if type(pass_samples) is not int or pass_samples <= 0:
        raise ValueError("pass_samples must be a positive integer")
    fallback_reasons = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
    zoom_histogram = {label: 0 for label in ZOOM_HISTOGRAM_LABELS}
    records = 0
    samples_total = 0
    selected_total = 0
    applied_total = 0
    both_axes_total = 0
    zoom_weighted_sum = 0.0
    offset_x_weighted_sum = 0.0
    offset_y_weighted_sum = 0.0
    zoom_max = 0.0
    for path in _iter_metric_files(metrics):
        for record in _read_records(path):
            schema = record.get("schema_version")
            if schema != TRAINING_METRIC_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: schema_version {schema!r} != "
                    f"{TRAINING_METRIC_SCHEMA_VERSION}"
                )
            record_id = record.get("successful_update")
            effective_batch = record.get("effective_batch")
            spatial_fallback = record.get("spatial_fallback_reasons")
            spatial_histogram = record.get("spatial_zoom_histogram")
            applied = record.get("spatial_crop_applied")
            selected = record.get("spatial_crop_selected")
            values = {
                "successful_update": record_id,
                "effective_batch": effective_batch,
                "spatial_crop_applied": applied,
                "spatial_crop_selected": selected,
                "spatial_both_axes_count": record.get("spatial_both_axes_count"),
                "spatial_actual_zoom_mean": record.get("spatial_actual_zoom_mean"),
                "spatial_actual_zoom_max": record.get("spatial_actual_zoom_max"),
                "spatial_abs_offset_x_mean": record.get("spatial_abs_offset_x_mean"),
                "spatial_abs_offset_y_mean": record.get(
                    "spatial_abs_offset_y_mean"
                ),
            }
            for name, value in values.items():
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        f"{path}: {name} must be an integer, got {value!r}"
                    )
            zoom_mean = values["spatial_actual_zoom_mean"]
            zoom_max_value = values["spatial_actual_zoom_max"]
            offset_x_mean = values["spatial_abs_offset_x_mean"]
            offset_y_mean = values["spatial_abs_offset_y_mean"]
            for name, value in (
                ("spatial_actual_zoom_mean", zoom_mean),
                ("spatial_actual_zoom_max", zoom_max_value),
                ("spatial_abs_offset_x_mean", offset_x_mean),
                ("spatial_abs_offset_y_mean", offset_y_mean),
            ):
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError(
                        f"{path}: {name} must be a finite float, got {value!r}"
                    )
            if not isinstance(spatial_fallback, dict):
                raise ValueError(f"{path}: spatial_fallback_reasons is not an object")
            if set(spatial_fallback) != set(SPATIAL_FALLBACK_REASONS):
                raise ValueError(
                    f"{path}: spatial_fallback_reasons keys differ from the fixed set"
                )
            if not isinstance(spatial_histogram, dict):
                raise ValueError(f"{path}: spatial_zoom_histogram is not an object")
            if set(spatial_histogram) != set(ZOOM_HISTOGRAM_LABELS):
                raise ValueError(
                    f"{path}: spatial_zoom_histogram keys differ from the fixed set"
                )
            for reason, count in spatial_fallback.items():
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"{path}: fallback count for {reason} is invalid")
                fallback_reasons[reason] += count
            for label, count in spatial_histogram.items():
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(
                        f"{path}: zoom histogram count for {label} is invalid"
                    )
                zoom_histogram[label] += count
            if applied > selected:
                raise ValueError(f"{path}: applied samples exceed selected samples")
            if sum(spatial_histogram.values()) != applied:
                raise ValueError(
                    f"{path}: zoom histogram does not cover the applied samples"
                )
            records += 1
            samples_total += effective_batch
            selected_total += selected
            applied_total += applied
            both_axes_total += values["spatial_both_axes_count"]
            zoom_weighted_sum += zoom_mean * applied
            offset_x_weighted_sum += offset_x_mean * applied
            offset_y_weighted_sum += offset_y_mean * applied
            zoom_max = max(zoom_max, zoom_max_value)
    if applied_total == 0:
        zoom_mean = 0.0
        offset_x_mean = 0.0
        offset_y_mean = 0.0
    else:
        zoom_mean = zoom_weighted_sum / applied_total
        offset_x_mean = offset_x_weighted_sum / applied_total
        offset_y_mean = offset_y_weighted_sum / applied_total
    if samples_total == 0:
        applied_probability = 0.0
        selected_probability = 0.0
    else:
        applied_probability = applied_total / samples_total
        selected_probability = selected_total / samples_total
    return {
        "records": records,
        "samples_total": samples_total,
        "selected_total": selected_total,
        "applied_total": applied_total,
        "both_axes_total": both_axes_total,
        "applied_probability": applied_probability,
        "selected_probability": selected_probability,
        "spatial_equivalent_passes": applied_total / pass_samples,
        "fallback_reasons": dict(fallback_reasons),
        "zoom_histogram": dict(zoom_histogram),
        "actual_zoom_mean": zoom_mean,
        "actual_zoom_max": zoom_max,
        "abs_offset_x_mean": offset_x_mean,
        "abs_offset_y_mean": offset_y_mean,
        "pass_samples": pass_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        required=True,
        type=Path,
        help="durable metric JSONL file or the directory containing it",
    )
    parser.add_argument(
        "--pass-samples",
        type=int,
        default=DEFAULT_PASS_SAMPLES,
        help="samples per full pass used for spatial-equivalent passes "
        f"(default {DEFAULT_PASS_SAMPLES})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output JSON path (read-only otherwise, default stdout)",
    )
    args = parser.parse_args(argv)
    report = aggregate(args.metrics, pass_samples=args.pass_samples)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is None:
        print(payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
