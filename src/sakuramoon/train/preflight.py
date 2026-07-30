"""Ordered, non-bypassable single-GPU preflight orchestration."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

PREFLIGHT_CHECKS = (
    "resolved_config",
    "local_assets",
    "dataset_revision",
    "single_gpu_runtime",
    "nvme_capacity",
    "frozen_encoders",
    "parameter_schema",
    "image_shapes",
    "text_shapes",
    "zero_update_loss",
    "optimizer_step",
    "sample",
    "checkpoint_round_trip",
)


class PreflightError(RuntimeError):
    """A mandatory preflight check failed."""


@dataclass(frozen=True, slots=True)
class PreflightCheckResult:
    name: str
    passed: bool
    error_type: str | None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: int
    hardware: str
    passed: bool
    checks: tuple[PreflightCheckResult, ...]


def _write_report(report: PreflightReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("preflight temporary report already exists")
    payload = (
        json.dumps(asdict(report), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def run_single_gpu_preflight(
    checks: Mapping[str, Callable[[], None]], destination: Path
) -> PreflightReport:
    """Run every mandatory check in fixed order and stop at the first failure."""

    if set(checks) != set(PREFLIGHT_CHECKS) or len(checks) != len(PREFLIGHT_CHECKS):
        raise ValueError("preflight requires every fixed check exactly once")
    results: list[PreflightCheckResult] = []
    for name in PREFLIGHT_CHECKS:
        try:
            checks[name]()
        except Exception as exc:
            results.append(PreflightCheckResult(name, False, type(exc).__name__))
            report = PreflightReport(1, "1GPU", False, tuple(results))
            _write_report(report, destination)
            raise PreflightError(f"mandatory preflight check failed: {name}") from exc
        results.append(PreflightCheckResult(name, True, None))
    report = PreflightReport(1, "1GPU", True, tuple(results))
    _write_report(report, destination)
    return report


__all__ = [
    "PREFLIGHT_CHECKS",
    "PreflightCheckResult",
    "PreflightError",
    "PreflightReport",
    "run_single_gpu_preflight",
]
