from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.preflight import (
    PREFLIGHT_CHECKS,
    PreflightError,
    run_single_gpu_preflight,
)


def _passing_checks() -> dict[str, Callable[[], None]]:
    return {name: (lambda: None) for name in PREFLIGHT_CHECKS}


def test_single_gpu_preflight_runs_every_fixed_check_in_order(tmp_path: Path) -> None:
    calls: list[str] = []
    checks: dict[str, Callable[[], None]] = {
        name: (lambda current=name: calls.append(current)) for name in PREFLIGHT_CHECKS
    }

    report = run_single_gpu_preflight(checks, tmp_path / "preflight.json")

    assert report.passed is True
    assert calls == list(PREFLIGHT_CHECKS)
    assert tuple(result.name for result in report.checks) == PREFLIGHT_CHECKS
    assert all(result.passed for result in report.checks)


def test_preflight_failure_stops_without_bypass_and_redacts_message(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    checks: dict[str, Callable[[], None]] = {
        name: (lambda current=name: calls.append(current)) for name in PREFLIGHT_CHECKS
    }

    def fail() -> None:
        calls.append("dataset_revision")
        raise RuntimeError("secret-shaped diagnostic text")

    checks["dataset_revision"] = fail
    destination = tmp_path / "preflight.json"

    with pytest.raises(PreflightError, match="dataset_revision"):
        run_single_gpu_preflight(checks, destination)

    assert calls == ["resolved_config", "local_assets", "dataset_revision"]
    text = destination.read_text()
    assert "secret-shaped" not in text
    payload = json.loads(text)
    assert payload["passed"] is False
    assert payload["checks"][-1] == {
        "error_type": "RuntimeError",
        "name": "dataset_revision",
        "passed": False,
    }


def test_preflight_rejects_missing_or_unknown_checks(tmp_path: Path) -> None:
    checks = _passing_checks()
    checks.pop("sample")
    checks["force"] = lambda: None

    with pytest.raises(ValueError, match="every fixed check"):
        run_single_gpu_preflight(checks, tmp_path / "preflight.json")


def test_failure_bundle_is_atomic_immutable_and_omits_exception_message(
    tmp_path: Path,
) -> None:
    snapshot = FailureSnapshot("failure-1", "update", "OutOfMemoryError", 3, 2, 8)

    bundle = write_failure_bundle(tmp_path, snapshot)

    assert (bundle / "COMPLETE").is_file()
    assert json.loads((bundle / "failure.json").read_text())["error_type"] == (
        "OutOfMemoryError"
    )
    with pytest.raises(FileExistsError):
        write_failure_bundle(tmp_path, snapshot)


def test_failure_bundle_does_not_replace_a_dangling_symlink(tmp_path: Path) -> None:
    snapshot = FailureSnapshot("failure-1", "update", "RuntimeError", 1, 0, 0)
    target = tmp_path / snapshot.failure_id
    target.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="already exists"):
        write_failure_bundle(tmp_path, snapshot)

    assert target.is_symlink()
