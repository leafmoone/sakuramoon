"""Canary-gate aggregation tool tests (scripts/aggregate_spatial_crop_metrics).

Regression coverage for the validator record-type contract: live schema-9
records mix integer counters (``successful_update``, ``effective_batch``,
``spatial_crop_applied``, ``spatial_crop_selected``,
``spatial_both_axes_count``) with finite-float geometry fields
(``spatial_actual_zoom_mean/max``, ``spatial_abs_offset_x/y_mean``). The
aggregator must accept that shape; the original validator lumped the float
fields into the integer-only check and crashed on the first live record
(``ValueError: spatial_actual_zoom_mean must be an integer``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
)
from sakuramoon.telemetry.metrics import TRAINING_METRIC_SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "aggregate_spatial_crop_metrics.py"


def _load_aggregate_module() -> Any:
    spec = importlib.util.spec_from_file_location("agg_spatial_crop", _SCRIPT_PATH)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _aggregate() -> Any:
    return _load_aggregate_module().aggregate


def _record(
    update: int,
    *,
    effective_batch: int = 400,
    selected: int = 0,
    applied: int = 0,
    both_axes: int = 0,
    zoom_mean: float = 1.0,
    zoom_max: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    fallback: dict[str, int] | None = None,
    histogram_bin: int | None = None,
    histogram_count: int | None = None,
    schema_version: int = TRAINING_METRIC_SCHEMA_VERSION,
) -> dict[str, Any]:
    """One metric record with the live (schema-9) field types."""
    fallback_counts = {reason: 0 for reason in SPATIAL_FALLBACK_REASONS}
    if fallback is not None:
        fallback_counts.update(fallback)
    histogram = {label: 0 for label in ZOOM_HISTOGRAM_LABELS}
    if applied > 0:
        bin_index = 3 if histogram_bin is None else histogram_bin
        histogram[ZOOM_HISTOGRAM_LABELS[bin_index]] = (
            applied if histogram_count is None else histogram_count
        )
    return {
        "schema_version": schema_version,
        "successful_update": update,
        "effective_batch": effective_batch,
        "spatial_crop_applied": applied,
        "spatial_crop_selected": selected,
        "spatial_both_axes_count": both_axes,
        "spatial_actual_zoom_mean": zoom_mean,
        "spatial_actual_zoom_max": zoom_max,
        "spatial_abs_offset_x_mean": offset_x,
        "spatial_abs_offset_y_mean": offset_y,
        "spatial_fallback_reasons": fallback_counts,
        "spatial_zoom_histogram": histogram,
    }


def _write_jsonl(tmp_path: Path, *records: dict[str, Any]) -> Path:
    path = tmp_path / "metrics.jsonl"
    lines = "".join(json.dumps(record) + "\n" for record in records)
    path.write_text(lines, encoding="utf-8")
    return path


def test_aggregate_accepts_live_record_types(tmp_path: Path) -> None:
    """Float zoom/offset fields alongside int counters must validate."""
    records = [
        _record(
            76467,
            selected=90,
            applied=85,
            both_axes=3,
            zoom_mean=1.0776580040757229,
            zoom_max=1.0999,
            offset_x=0.0123456,
            offset_y=0.0087654,
            fallback={"none": 85, "not_selected": 310, "quantized_no_effect": 5},
            histogram_bin=3,
        ),
        _record(
            76468,
            selected=10,
            applied=0,
            fallback={"not_selected": 390, "quantized_no_effect": 10},
        ),
    ]
    report = _aggregate()(_write_jsonl(tmp_path, *records), pass_samples=100)
    assert report["records"] == 2
    assert report["samples_total"] == 800
    assert report["selected_total"] == 100
    assert report["applied_total"] == 85
    assert report["both_axes_total"] == 3
    assert report["applied_probability"] == pytest.approx(85 / 800)
    assert report["selected_probability"] == pytest.approx(100 / 800)
    assert report["spatial_equivalent_passes"] == pytest.approx(0.85)
    assert report["actual_zoom_mean"] == pytest.approx(1.0776580040757229)
    assert report["actual_zoom_max"] == pytest.approx(1.0999)
    assert report["abs_offset_x_mean"] == pytest.approx(0.0123456)
    assert report["abs_offset_y_mean"] == pytest.approx(0.0087654)
    assert report["fallback_reasons"]["none"] == 85
    assert report["fallback_reasons"]["not_selected"] == 700
    assert report["fallback_reasons"]["quantized_no_effect"] == 15
    assert sum(report["zoom_histogram"].values()) == 85


def test_aggregate_rejects_non_object_record(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="record is not an object"):
        _aggregate()(path)


def test_aggregate_rejects_non_integer_counters(tmp_path: Path) -> None:
    record = _record(76467, applied=1, fallback={"none": 1})
    record["successful_update"] = 76467.0
    with pytest.raises(TypeError, match="successful_update must be an integer"):
        _aggregate()(_write_jsonl(tmp_path, record))


def test_aggregate_requires_float_zoom_fields(tmp_path: Path) -> None:
    record = _record(76467, applied=1, fallback={"none": 1})
    record["spatial_actual_zoom_mean"] = 1
    with pytest.raises(ValueError, match="must be a finite float"):
        _aggregate()(_write_jsonl(tmp_path, record))


def test_aggregate_rejects_applied_exceeding_selected(tmp_path: Path) -> None:
    record = _record(76467, selected=1, applied=2)
    with pytest.raises(
        ValueError, match="applied samples exceed selected samples"
    ):
        _aggregate()(_write_jsonl(tmp_path, record))


def test_aggregate_rejects_histogram_not_covering_applied(
    tmp_path: Path,
) -> None:
    record = _record(76467, selected=90, applied=85, histogram_bin=3)
    record["spatial_zoom_histogram"][ZOOM_HISTOGRAM_LABELS[3]] = 1
    with pytest.raises(
        ValueError, match="zoom histogram does not cover the applied samples"
    ):
        _aggregate()(_write_jsonl(tmp_path, record))


def test_aggregate_rejects_schema_mismatch(tmp_path: Path) -> None:
    record = _record(76467, schema_version=TRAINING_METRIC_SCHEMA_VERSION - 1)
    with pytest.raises(ValueError, match="schema_version"):
        _aggregate()(_write_jsonl(tmp_path, record))


def test_aggregate_missing_metric_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _aggregate()(tmp_path / "absent.jsonl")


def test_aggregate_empty_metrics_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _aggregate()(tmp_path)
