"""Immutable evaluation metric artifacts and three-checkpoint comparisons."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from sakuramoon.eval.spec import (
    ArtifactKind,
    CheckpointKind,
    EvaluationCost,
    EvaluationJob,
)


@dataclass(frozen=True, slots=True)
class EvaluationArtifact:
    job: EvaluationJob
    value: float
    std: float | None
    cost: EvaluationCost
    automatic_release: bool = False

    def __post_init__(self) -> None:
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("evaluation metric value must be finite")
        if self.std is not None and (
            type(self.std) is not float or not math.isfinite(self.std) or self.std < 0.0
        ):
            raise ValueError("evaluation metric std must be finite and nonnegative")
        if self.job.metric == "is" and self.std is None:
            raise ValueError("IS artifact requires split standard deviation")
        if self.job.metric != "is" and self.std is not None:
            raise ValueError("only IS artifacts carry split standard deviation")
        if self.automatic_release is not False:
            raise ValueError("evaluation metrics cannot automatically release a checkpoint")

    def as_mapping(self) -> dict[str, object]:
        return {
            "automatic_release": False,
            "cost": {
                "gpu_seconds": self.cost.gpu_seconds,
                "training_pause_seconds": self.cost.training_pause_seconds,
                "wall_seconds": self.cost.wall_seconds,
            },
            "job": self.job.as_mapping(),
            "schema_version": 1,
            "std": self.std,
            "value": self.value,
        }


def write_evaluation_artifact(path: Path, artifact: EvaluationArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("evaluation artifact already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("evaluation artifact temporary path exists")
    body = (
        json.dumps(artifact.as_mapping(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            temporary.unlink()
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CheckpointMetricComparison:
    artifact_kind: ArtifactKind
    values: tuple[tuple[CheckpointKind, float], ...]

    def __post_init__(self) -> None:
        kinds = tuple(kind for kind, _ in self.values)
        if len(self.values) != 3 or set(kinds) != {"raw_latest", "pma10", "accepted"}:
            raise ValueError("comparison requires raw latest, PMA-10, and accepted")
        if any(type(value) is not float or not math.isfinite(value) for _, value in self.values):
            raise ValueError("comparison values must be finite floats")


__all__ = [
    "CheckpointMetricComparison",
    "EvaluationArtifact",
    "write_evaluation_artifact",
]
