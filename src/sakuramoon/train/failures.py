"""Minimal atomic diagnostic bundles for fail-closed training."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_FAILURE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class FailureSnapshot:
    failure_id: str
    phase: str
    error_type: str
    attempted_updates: int
    successful_updates: int
    effective_samples: int

    def __post_init__(self) -> None:
        if _FAILURE_ID.fullmatch(self.failure_id) is None:
            raise ValueError("failure ID is invalid")
        if not self.phase or not self.error_type:
            raise ValueError("failure phase and type must be nonempty")
        if (
            type(self.attempted_updates) is not int
            or type(self.successful_updates) is not int
            or type(self.effective_samples) is not int
            or self.attempted_updates < 0
            or self.successful_updates < 0
            or self.successful_updates > self.attempted_updates
            or self.effective_samples < 0
        ):
            raise ValueError("failure counters are inconsistent")


def write_failure_bundle(root: Path, snapshot: FailureSnapshot) -> Path:
    """Publish one immutable diagnostic directory without exception messages."""

    root.mkdir(parents=True, exist_ok=True)
    target = root / snapshot.failure_id
    temporary = root / f".{snapshot.failure_id}.tmp"
    if (
        target.exists()
        or target.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError("failure diagnostic target already exists")
    temporary.mkdir()
    try:
        payload = (
            json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        report = temporary / "failure.json"
        with report.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        marker = temporary / "COMPLETE"
        with marker.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        for child in (temporary.iterdir() if temporary.exists() else ()):
            child.unlink()
        if temporary.exists():
            temporary.rmdir()
        raise
    return target


__all__ = ["FailureSnapshot", "write_failure_bundle"]
