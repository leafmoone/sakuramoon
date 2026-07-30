"""Deterministic checkpoint-driven evaluation job identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal

CheckpointKind = Literal["raw_latest", "pma10", "accepted"]
MetricName = Literal["fid", "is", "manual_quality", "vae_reconstruction"]
ArtifactKind = Literal[
    "fid_trend",
    "fid_formal",
    "is_trend",
    "is_formal",
    "manual_quality",
    "vae_reconstruction",
]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _safe_id(name: str, value: str) -> None:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _sha256(name: str, value: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class PromptCase:
    prompt_id: str
    prompt: str
    conditions: tuple[str, ...]
    seed: int
    height: int
    width: int

    def __post_init__(self) -> None:
        _safe_id("prompt_id", self.prompt_id)
        if (
            type(self.prompt) is not str
            or not self.prompt.strip()
            or self.prompt != self.prompt.strip()
            or "<think>" in self.prompt
            or "</think>" in self.prompt
        ):
            raise ValueError("prompt text is invalid")
        if (
            type(self.conditions) is not tuple
            or any(
                type(value) is not str or not value or value != value.strip()
                for value in self.conditions
            )
            or len(set(self.conditions)) != len(self.conditions)
        ):
            raise ValueError("prompt conditions are invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("prompt seed must be a nonnegative integer")
        if any(
            type(value) is not int or value <= 0 or value % 16 != 0
            for value in (self.height, self.width)
        ):
            raise ValueError("prompt dimensions must be positive multiples of 16")

    def as_mapping(self) -> dict[str, object]:
        return {
            "conditions": list(self.conditions),
            "height": self.height,
            "prompt": self.prompt,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class PromptManifest:
    cases: tuple[PromptCase, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("prompt manifest must not be empty")
        identifiers = tuple(case.prompt_id for case in self.cases)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prompt IDs must be unique")

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                {
                    "cases": [case.as_mapping() for case in self.cases],
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    checkpoint_id: str
    kind: CheckpointKind
    resolved_config_sha256: str

    def __post_init__(self) -> None:
        _safe_id("checkpoint_id", self.checkpoint_id)
        if self.kind not in ("raw_latest", "pma10", "accepted"):
            raise ValueError("checkpoint kind is invalid")
        _sha256("resolved_config_sha256", self.resolved_config_sha256)


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    job_id: str
    checkpoint: CheckpointRef
    metric: MetricName
    artifact_kind: ArtifactKind
    prompt_manifest_sha256: str
    sample_count: int
    cfg_scale: float
    solver_steps: int
    solver_nfe: int
    feature_extractor: str
    feature_extractor_version: str
    preprocess_sha256: str
    real_stats_sha256: str
    is_splits: int
    gpu_index: int
    training_paused: bool

    def __post_init__(self) -> None:
        _safe_id("job_id", self.job_id)
        if self.metric not in ("fid", "is", "manual_quality", "vae_reconstruction"):
            raise ValueError("evaluation metric is invalid")
        expected_prefix = {
            "fid": "fid_",
            "is": "is_",
            "manual_quality": "manual_quality",
            "vae_reconstruction": "vae_reconstruction",
        }[self.metric]
        if not self.artifact_kind.startswith(expected_prefix):
            raise ValueError("metric and artifact kind are inconsistent")
        for name, value in (
            ("prompt_manifest_sha256", self.prompt_manifest_sha256),
            ("preprocess_sha256", self.preprocess_sha256),
            ("real_stats_sha256", self.real_stats_sha256),
        ):
            _sha256(name, value)
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("evaluation sample count must be positive")
        if type(self.cfg_scale) is not float or self.cfg_scale != 2.9:
            raise ValueError("formal evaluation CFG must equal 2.9")
        if self.solver_steps != 50 or self.solver_nfe != 99:
            raise ValueError("formal evaluation must use Heun-50 with 99 NFE")
        if (
            not self.feature_extractor
            or not self.feature_extractor_version
            or type(self.feature_extractor) is not str
            or type(self.feature_extractor_version) is not str
        ):
            raise ValueError("feature extractor identity must be explicit")
        if type(self.is_splits) is not int or self.is_splits <= 0:
            raise ValueError("IS splits must be positive")
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError("evaluation GPU index must be nonnegative")
        if type(self.training_paused) is not bool:
            raise TypeError("training_paused must be explicit")

    def as_mapping(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "cfg_scale": self.cfg_scale,
            "checkpoint": {
                "checkpoint_id": self.checkpoint.checkpoint_id,
                "kind": self.checkpoint.kind,
                "resolved_config_sha256": self.checkpoint.resolved_config_sha256,
            },
            "feature_extractor": self.feature_extractor,
            "feature_extractor_version": self.feature_extractor_version,
            "gpu_index": self.gpu_index,
            "is_splits": self.is_splits,
            "job_id": self.job_id,
            "metric": self.metric,
            "preprocess_sha256": self.preprocess_sha256,
            "prompt_manifest_sha256": self.prompt_manifest_sha256,
            "real_stats_sha256": self.real_stats_sha256,
            "sample_count": self.sample_count,
            "schema_version": 1,
            "solver_nfe": self.solver_nfe,
            "solver_steps": self.solver_steps,
            "training_paused": self.training_paused,
        }


@dataclass(frozen=True, slots=True)
class EvaluationCost:
    wall_seconds: float
    gpu_seconds: float
    training_pause_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("wall_seconds", self.wall_seconds),
            ("gpu_seconds", self.gpu_seconds),
            ("training_pause_seconds", self.training_pause_seconds),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative float")
        if self.gpu_seconds > self.wall_seconds:
            raise ValueError("GPU seconds cannot exceed wall seconds for one GPU")
        if self.training_pause_seconds > self.wall_seconds:
            raise ValueError("training pause cannot exceed evaluation wall time")


__all__ = [
    "ArtifactKind",
    "CheckpointKind",
    "CheckpointRef",
    "EvaluationCost",
    "EvaluationJob",
    "MetricName",
    "PromptCase",
    "PromptManifest",
]
