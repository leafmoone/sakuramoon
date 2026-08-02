"""Manual quality indexing without converting FID/IS into a release gate."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from sakuramoon.eval.spec import (
    EvaluationCost,
    EvaluationJob,
    PromptManifest,
)

QualityField = Literal[
    "tag_control", "aesthetic", "nl_following", "composition", "detail"
]
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ManualQualityImage:
    prompt_id: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.prompt_id
            or not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("manual quality image identity is invalid")


@dataclass(frozen=True, slots=True)
class ManualQualityIndex:
    job: EvaluationJob
    prompt_manifest: PromptManifest
    images: tuple[ManualQualityImage, ...]
    cost: EvaluationCost
    automatic_release: bool = False

    def __post_init__(self) -> None:
        if self.job.job_id != self.job.content_addressed_id:
            raise ValueError("manual quality index job ID is not content-addressed")
        if (
            self.job.metric != "manual_quality"
            or self.job.artifact_kind != "manual_quality"
        ):
            raise ValueError("manual quality index requires a manual quality job")
        if self.prompt_manifest.sha256 != self.job.prompt_manifest_sha256:
            raise ValueError("manual quality prompt manifest differs from its job")
        if type(self.images) is not tuple or any(
            type(item) is not ManualQualityImage for item in self.images
        ):
            raise TypeError("manual quality images must be an immutable tuple")
        expected_ids = tuple(
            case.prompt_id
            for case in self.prompt_manifest.cases[: self.job.sample_count]
        )
        if tuple(item.prompt_id for item in self.images) != expected_ids:
            raise ValueError("manual quality images do not match the prompt plan")
        if len({item.relative_path for item in self.images}) != len(self.images):
            raise ValueError("manual quality image paths must be unique")
        if not self.job.training_paused and self.cost.training_pause_seconds != 0.0:
            raise ValueError(
                "training pause cost must be zero when the job does not pause training"
            )
        if self.automatic_release is not False:
            raise ValueError("manual quality cannot automatically release a checkpoint")

    def as_mapping(self) -> dict[str, object]:
        return {
            "artifact_kind": "manual_quality_index",
            "automatic_release": False,
            "cost": {
                "gpu_seconds": self.cost.gpu_seconds,
                "publication_seconds_included": False,
                "training_pause_seconds": self.cost.training_pause_seconds,
                "wall_seconds": self.cost.wall_seconds,
            },
            "images": [
                {
                    "prompt_id": item.prompt_id,
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                }
                for item in self.images
            ],
            "job": self.job.as_mapping(),
            "schema_version": 1,
        }


@dataclass(frozen=True, slots=True)
class ManualQualityObservation:
    prompt_id: str
    tag_control: float
    aesthetic: float
    nl_following: float
    composition: float
    detail: float
    severe_artifact: bool

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("manual quality prompt ID must not be empty")
        for name in (
            "tag_control",
            "aesthetic",
            "nl_following",
            "composition",
            "detail",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 5.0:
                raise ValueError("manual quality scores must be finite floats in [0,5]")
        if type(self.severe_artifact) is not bool:
            raise TypeError("severe artifact label must be boolean")


@dataclass(frozen=True, slots=True)
class ManualQualityReport:
    sample_count: int
    tag_control_mean: float
    aesthetic_mean: float
    nl_following_mean: float
    composition_mean: float
    detail_mean: float
    severe_artifact_rate: float
    automatic_release: bool = False

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("manual quality sample count must be positive")
        for name in (
            "tag_control_mean",
            "aesthetic_mean",
            "nl_following_mean",
            "composition_mean",
            "detail_mean",
        ):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 5.0
            ):
                raise ValueError("manual quality means must be finite floats in [0,5]")
        if (
            type(self.severe_artifact_rate) is not float
            or not math.isfinite(self.severe_artifact_rate)
            or not 0.0 <= self.severe_artifact_rate <= 1.0
        ):
            raise ValueError("severe artifact rate must be a finite float in [0,1]")
        if self.automatic_release is not False:
            raise ValueError("manual quality cannot automatically release a checkpoint")


@dataclass(frozen=True, slots=True)
class ManualQualityArtifact:
    job: EvaluationJob
    prompt_manifest: PromptManifest
    observations: tuple[ManualQualityObservation, ...]
    report: ManualQualityReport
    cost: EvaluationCost
    automatic_release: bool = False

    def __post_init__(self) -> None:
        if self.job.job_id != self.job.content_addressed_id:
            raise ValueError("manual quality job ID is not content-addressed")
        if (
            self.job.metric != "manual_quality"
            or self.job.artifact_kind != "manual_quality"
        ):
            raise ValueError("manual quality artifact requires a manual quality job")
        if self.prompt_manifest.sha256 != self.job.prompt_manifest_sha256:
            raise ValueError("manual quality prompt manifest differs from its job")
        if type(self.observations) is not tuple or any(
            type(item) is not ManualQualityObservation for item in self.observations
        ):
            raise TypeError("manual quality observations must be an immutable tuple")
        if len(self.observations) != self.job.sample_count:
            raise ValueError("manual quality observations do not match the job sample count")
        expected_ids = tuple(
            case.prompt_id
            for case in self.prompt_manifest.cases[: self.job.sample_count]
        )
        actual_ids = tuple(item.prompt_id for item in self.observations)
        if actual_ids != expected_ids:
            raise ValueError("manual quality observations do not match the prompt plan")
        if summarize_manual_quality(self.observations) != self.report:
            raise ValueError("manual quality report does not match its observations")
        if not self.job.training_paused and self.cost.training_pause_seconds != 0.0:
            raise ValueError(
                "training pause cost must be zero when the job does not pause training"
            )
        if self.automatic_release is not False:
            raise ValueError("manual quality cannot automatically release a checkpoint")

    def as_mapping(self) -> dict[str, object]:
        return {
            "artifact_kind": "manual_quality",
            "automatic_release": False,
            "cost": {
                "gpu_seconds": self.cost.gpu_seconds,
                "publication_seconds_included": False,
                "training_pause_seconds": self.cost.training_pause_seconds,
                "wall_seconds": self.cost.wall_seconds,
            },
            "job": self.job.as_mapping(),
            "observations": [
                {
                    "aesthetic": item.aesthetic,
                    "composition": item.composition,
                    "detail": item.detail,
                    "nl_following": item.nl_following,
                    "prompt_id": item.prompt_id,
                    "severe_artifact": item.severe_artifact,
                    "tag_control": item.tag_control,
                }
                for item in self.observations
            ],
            "report": {
                "aesthetic_mean": self.report.aesthetic_mean,
                "composition_mean": self.report.composition_mean,
                "detail_mean": self.report.detail_mean,
                "nl_following_mean": self.report.nl_following_mean,
                "sample_count": self.report.sample_count,
                "severe_artifact_rate": self.report.severe_artifact_rate,
                "tag_control_mean": self.report.tag_control_mean,
            },
            "schema_version": 1,
        }


def summarize_manual_quality(
    observations: tuple[ManualQualityObservation, ...],
) -> ManualQualityReport:
    if not observations:
        raise ValueError("manual quality observations must not be empty")
    identifiers = tuple(item.prompt_id for item in observations)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("manual quality prompt IDs must be unique")
    def mean(name: QualityField) -> float:
        return float(statistics.fmean(getattr(item, name) for item in observations))
    return ManualQualityReport(
        sample_count=len(observations),
        tag_control_mean=mean("tag_control"),
        aesthetic_mean=mean("aesthetic"),
        nl_following_mean=mean("nl_following"),
        composition_mean=mean("composition"),
        detail_mean=mean("detail"),
        severe_artifact_rate=sum(item.severe_artifact for item in observations)
        / len(observations),
    )


def write_manual_quality_artifact(
    path: Path,
    artifact: ManualQualityArtifact,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("manual quality artifact already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("manual quality artifact temporary path exists")
    body = (
        json.dumps(artifact.as_mapping(), sort_keys=True, separators=(",", ":"))
        + "\n"
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


def write_manual_quality_index(path: Path, index: ManualQualityIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("manual quality index already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("manual quality index temporary path exists")
    body = (
        json.dumps(index.as_mapping(), sort_keys=True, separators=(",", ":"))
        + "\n"
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


__all__ = [
    "ManualQualityArtifact",
    "ManualQualityImage",
    "ManualQualityIndex",
    "ManualQualityObservation",
    "ManualQualityReport",
    "summarize_manual_quality",
    "write_manual_quality_artifact",
    "write_manual_quality_index",
]
