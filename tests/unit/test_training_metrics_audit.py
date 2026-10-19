"""Tests for the read-only training metrics audit helper."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
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


def _row(update: int, **fields: object) -> dict[str, object]:
    return {"successful_update": update, "recorded_at_unix_ns": 1, **fields}


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


def test_separates_byte_identical_same_payload_and_conflicting_rows(
    tmp_path: Path,
) -> None:
    rows = [
        _row(update, main_loss=0.5)
        for update in range(10, 21)
        if update != 15
    ]
    # Same payload, different producer timestamp: not a conflict.
    rows.append(_row(12, recorded_at_unix_ns=2, main_loss=0.5))
    # Byte-identical duplicate.
    rows.append(_row(11, main_loss=0.5))
    # Genuinely different content for the same update.
    rows.append(_row(13, main_loss=0.9))
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows, extra="{not json}")

    report = audit_module.audit(path, start=10, end=20, loss_field="main_loss")
    window = report["window"]

    assert report["file"]["rows"] == len(rows)
    assert report["file"]["malformed_rows"] == 1
    assert window["expected_updates"] == 11
    assert window["distinct_updates"] == 10
    assert window["selected_rows"] == len(rows)
    assert window["missing_count"] == 1
    assert window["missing_ranges"] == [[15, 15]]
    assert window["same_payload_extra_rows"] == 2
    assert window["byte_identical_extra_rows"] == 1
    assert window["conflicting_updates"] == [13]
    assert math.isclose(cast(float, window["coverage"]), 10 / 11, abs_tol=1e-6)


def test_statistics_deduplicate_updates(tmp_path: Path) -> None:
    rows = [
        _row(10 + index, main_loss=loss)
        for index, loss in enumerate([0.4, 0.5, 0.6, 0.7, 0.8])
    ]
    rows.append(_row(12, main_loss=0.6))  # duplicate update
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=10, end=14, loss_field="main_loss")
    stats = report["statistics"]

    assert stats["rows"] == 5
    assert stats["source_rows"] == 6
    assert math.isclose(cast(float, stats["loss_mean"]), 0.6, abs_tol=1e-9)
    assert math.isclose(cast(float, stats["loss_median"]), 0.6, abs_tol=1e-9)


def test_loss_fallback_is_reported(tmp_path: Path) -> None:
    rows = [_row(update, total_loss=0.25) for update in (7, 8, 10)]
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=7, end=10, loss_field="main_loss")
    stats = report["statistics"]

    assert stats["loss_field_used"] == "total_loss"
    assert stats["loss_fallback"] is True
    assert stats["loss_missing_rows"] == 3
    assert math.isclose(cast(float, stats["loss_mean"]), 0.25, abs_tol=1e-9)


def test_nonfinite_metric_values_do_not_crash_or_poison_statistics(
    tmp_path: Path,
) -> None:
    rows = [
        _row(10, main_loss=float("nan"), nonfinite_count=float("nan")),
        _row(11, main_loss=0.5, nonfinite_count=0),
    ]
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=10, end=11, loss_field="main_loss")
    stats = report["statistics"]

    assert stats["invalid_numeric_values"] >= 2
    assert stats["loss_mean"] == 0.5
    assert "NaN" not in json.dumps(report)


def test_partial_window_bounds_are_inferred(tmp_path: Path) -> None:
    rows = [_row(update, main_loss=0.5) for update in range(10, 15)]
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    start_only = audit_module.audit(path, start=12, end=None, loss_field="main_loss")
    end_only = audit_module.audit(path, start=None, end=12, loss_field="main_loss")

    assert start_only["window"]["end"] == 14
    assert start_only["window"]["expected_updates"] == 3
    assert end_only["window"]["start"] == 10
    assert end_only["window"]["expected_updates"] == 3


def test_empty_file_reports_no_update_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("", encoding="utf-8")

    report = audit_module.audit(path, start=5, end=None, loss_field="main_loss")

    assert report["file"]["rows"] == 0
    assert report["window"]["status"] == "no_update_rows"
    assert report["window"]["expected_updates"] is None
    assert report["window"]["coverage"] is None
    assert report["window"]["missing_count"] == 0


def test_missing_updates_are_compacted_into_ranges(tmp_path: Path) -> None:
    rows = [_row(update, main_loss=0.5) for update in (10, 11, 20)]
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=10, end=20, loss_field="main_loss")

    assert report["window"]["missing_count"] == 8
    assert report["window"]["missing_ranges"] == [[12, 19]]


def test_rows_without_update_are_counted(tmp_path: Path) -> None:
    rows = [
        {"step": 1},
        {"successful_update": "11"},
        _row(12, main_loss=0.5),
    ]
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, rows)

    report = audit_module.audit(path, start=12, end=12, loss_field="main_loss")

    assert report["file"]["rows"] == 3
    assert report["file"]["rows_without_update"] == 2


def test_out_must_not_overwrite_the_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, [_row(5, main_loss=0.5)])

    with pytest.raises(SystemExit) as caught:
        audit_module.main([str(path), "--out", str(path)])
    assert caught.value.code == 2
    assert "--out" in capsys.readouterr().err


def test_truncated_gzip_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl.gz"
    _write_rows(path, [_row(5, main_loss=0.5)])
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    report = audit_module.audit(path, start=5, end=5, loss_field="main_loss")

    assert report["file"]["container_error"]


@pytest.mark.parametrize("to_file", [False, True])
def test_cli_read_error_returns_nonzero_with_partial_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], to_file: bool,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.touch()

    def broken_lines(_path: Path) -> Iterator[tuple[int, str]]:
        yield 1, json.dumps(_row(5, main_loss=0.5)) + "\n"
        raise OSError(5, "injected mid-read failure")

    monkeypatch.setattr(audit_module, "_iter_lines", broken_lines)
    out = tmp_path / "report.json"
    args = [str(path)] + (["--out", str(out)] if to_file else [])
    assert audit_module.main(args) == 1
    captured = capsys.readouterr()
    report = json.loads(out.read_text() if to_file else captured.out)
    assert report["file"]["rows"] == 1
    assert "injected mid-read failure" in report["file"]["container_error"]
    assert report["statistics"]["loss_mean"] == 0.5
    assert report["statistics"]["rows"] == 1
    assert "partial report" in captured.err


def test_cli_read_and_metadata_failure_keeps_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.touch()

    def broken_lines(_path: Path) -> Iterator[tuple[int, str]]:
        yield 1, json.dumps(_row(5, main_loss=0.5)) + "\n"
        path.unlink()
        raise OSError(5, "backing file disappeared mid-read")

    monkeypatch.setattr(audit_module, "_iter_lines", broken_lines)
    assert audit_module.main([str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["statistics"]["loss_mean"] == 0.5
    assert "backing file disappeared" in report["file"]["container_error"]
    assert "metadata: FileNotFoundError" in report["file"]["container_error"]
    assert report["file"]["bytes"] is None
    assert report["file"]["modified"] is None


@pytest.mark.parametrize("to_file", [False, True])
def test_cli_truncated_gzip_process_exit_and_partial_report(
    tmp_path: Path, to_file: bool,
) -> None:
    path = tmp_path / "metrics.jsonl.gz"
    # A complete first gzip member is emitted before the truncated next
    # member fails.  This deterministically preserves the first statistics.
    first = gzip.compress((json.dumps(_row(5, main_loss=0.5)) + "\n").encode())
    second = gzip.compress(b'{"successful_update":6}\n')
    path.write_bytes(first + second[:10])
    out = tmp_path / "report.json"
    args = [sys.executable, str(SCRIPT), str(path)]
    if to_file:
        args += ["--out", str(out)]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    report = json.loads(out.read_text() if to_file else result.stdout)
    assert report["file"]["rows"] == 1
    assert report["statistics"]["loss_mean"] == 0.5
    assert "EOFError" in report["file"]["container_error"]
    assert "partial report" in result.stderr


@pytest.mark.parametrize("kind", ["missing", "window", "inferred_window"])
def test_cli_invalid_arguments_are_clear_nonzero(
    tmp_path: Path, kind: str,
) -> None:
    path = tmp_path / "metrics.jsonl"
    if kind != "missing":
        _write_rows(path, [_row(5, main_loss=0.5)])
    args = [sys.executable, str(SCRIPT), str(path)]
    if kind == "window":
        args += ["--start", "6", "--end", "5"]
    if kind == "inferred_window":
        args += ["--start", "6"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_output_write_error_is_clear_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, [_row(5, main_loss=0.5)])
    out = tmp_path / "absent" / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 3
    assert "cannot write --out" in result.stderr
    assert str(out) in result.stderr
    assert "FileNotFoundError" in result.stderr


def test_cli_success_still_returns_zero(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write_rows(path, [_row(5, main_loss=0.5)])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["file"]["container_error"] is None
