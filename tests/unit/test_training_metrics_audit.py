"""Tests for the read-only training metrics audit helper."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_training_metrics.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_training_metrics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_module = cast(Any, _load_module())


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    extra: str = "",
) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    content = "\n".join(lines) + ("\n" + extra if extra else "\n")
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_reports_missing_duplicate_and_conflicting_rows(tmp_path: Path) -> None:
    rows = [
        {"successful_update": update, "main_loss": 0.5}
        for update in range(10, 21)
        if update != 15
    ]
    rows.append({"successful_update": 12, "main_loss": 0.5})  # exact duplicate
    rows.append({"successful_update": 13, "main_loss": 0.9})  # conflicting row
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows, extra="{not json}")

    report = audit_module.audit(path, start=10, end=20, loss_field="main_loss")

    assert report["file"]["rows"] == len(rows)
    assert report["file"]["malformed_rows"] == 1
    assert report["window"]["expected_updates"] == 11
    assert report["window"]["distinct_updates"] == 10
    assert report["window"]["selected_rows"] == len(rows)
    assert report["window"]["missing_updates"] == [15]
    assert report["window"]["exact_duplicate_rows"] == 1
    assert report["window"]["conflicting_updates"] == [13]
    assert math.isclose(cast(float, report["window"]["coverage"]), 10 / 11, abs_tol=1e-6)
    assert report["notes"]


def test_statistics_are_computed_on_the_selected_window(tmp_path: Path) -> None:
    rows = [
        {
            "successful_update": 10 + index,
            "main_loss": loss,
            "pre_clip_grad_norm": gradient,
            "clip_fraction": clip,
            "nonfinite_count": nonfinite,
        }
        for index, (loss, gradient, clip, nonfinite) in enumerate(
            [
                (0.4, 0.1, 0.0, 0),
                (0.5, 0.2, 0.0, 0),
                (0.6, 0.3, 1.0, 2),
                (0.7, 0.4, 0.0, 0),
                (0.8, 0.5, 0.0, 0),
            ]
        )
    ]
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=10, end=14, loss_field="main_loss")
    stats = report["statistics"]

    assert stats["rows"] == 5
    assert math.isclose(cast(float, stats["loss_mean"]), 0.6, abs_tol=1e-6)
    assert math.isclose(cast(float, stats["loss_median"]), 0.6, abs_tol=1e-6)
    assert math.isclose(cast(float, stats["pre_clip_median"]), 0.3, abs_tol=1e-6)
    assert math.isclose(cast(float, stats["pre_clip_max"]), 0.5, abs_tol=1e-6)
    assert math.isclose(cast(float, stats["clip_fraction_mean"]), 0.2, abs_tol=1e-6)
    assert stats["clipped_updates"] == 1
    assert stats["nonfinite_total"] == 2


def test_gzip_input_and_full_range_default(tmp_path: Path) -> None:
    rows = [{"successful_update": update, "total_loss": 0.25} for update in (7, 8, 10)]
    path = tmp_path / "metrics.jsonl.gz"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=None, end=None, loss_field="main_loss")

    assert report["window"]["start"] == 7
    assert report["window"]["end"] == 10
    assert report["window"]["missing_updates"] == [9]
    assert math.isclose(cast(float, report["window"]["coverage"]), 3 / 4, abs_tol=1e-6)
    # main_loss is absent, so the helper falls back to total_loss.
    assert math.isclose(cast(float, report["statistics"]["loss_mean"]), 0.25, abs_tol=1e-6)


def test_rejects_reversed_window(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, [{"successful_update": 5, "main_loss": 0.5}])

    with pytest.raises(ValueError, match="start"):
        audit_module.audit(path, start=10, end=5, loss_field="main_loss")
