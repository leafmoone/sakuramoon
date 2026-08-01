"""Deterministic checkpoint-driven evaluation job identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, cast

from sakuramoon.sampling.profiles import (
    SamplingProfileName,
    SamplingSolver,
    TimeSchedule,
    resolve_sampling_profile,
)

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
        if type(self.cases) is not tuple or any(
            type(case) is not PromptCase for case in self.cases
        ):
            raise TypeError("prompt manifest cases must be an immutable PromptCase tuple")
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

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> PromptManifest:
        if type(payload) is not bytes:
            raise TypeError("prompt manifest payload must be bytes")
        try:
            parsed: object = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("prompt manifest must be valid canonical JSON") from None
        if type(parsed) is not dict:
            raise ValueError("prompt manifest root must be an object")
        document = cast(dict[str, object], parsed)
        if set(document) != {"cases", "schema_version"}:
            raise ValueError("prompt manifest root fields are invalid")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("prompt manifest schema version is invalid")
        raw_cases = document["cases"]
        if type(raw_cases) is not list:
            raise ValueError("prompt manifest cases must be an array")
        cases: list[PromptCase] = []
        expected_case_fields = {
            "conditions",
            "height",
            "prompt",
            "prompt_id",
            "seed",
            "width",
        }
        for raw_case in cast(list[object], raw_cases):
            if type(raw_case) is not dict:
                raise ValueError("prompt manifest case must be an object")
            case = cast(dict[str, object], raw_case)
            if set(case) != expected_case_fields:
                raise ValueError("prompt manifest case fields are invalid")
            raw_conditions = case["conditions"]
            if type(raw_conditions) is not list:
                raise ValueError("prompt manifest conditions must be a string array")
            condition_values = cast(list[object], raw_conditions)
            if any(type(value) is not str for value in condition_values):
                raise ValueError("prompt manifest conditions must be a string array")
            cases.append(
                PromptCase(
                    prompt_id=cast(str, case["prompt_id"]),
                    prompt=cast(str, case["prompt"]),
                    conditions=tuple(cast(str, value) for value in condition_values),
                    seed=cast(int, case["seed"]),
                    height=cast(int, case["height"]),
                    width=cast(int, case["width"]),
                )
            )
        manifest = cls(tuple(cases))
        if manifest.canonical_bytes() != payload:
            raise ValueError("prompt manifest must use canonical serialization")
        return manifest

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    checkpoint_id: str
    kind: CheckpointKind
    resolved_config_sha256: str
    successful_update: int

    def __post_init__(self) -> None:
        _safe_id("checkpoint_id", self.checkpoint_id)
        if self.kind not in ("raw_latest", "pma10", "accepted"):
            raise ValueError("checkpoint kind is invalid")
        _sha256("resolved_config_sha256", self.resolved_config_sha256)
        if type(self.successful_update) is not int or self.successful_update < 0:
            raise ValueError("checkpoint successful update must be nonnegative")


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    job_id: str
    checkpoint: CheckpointRef
    metric: MetricName
    artifact_kind: ArtifactKind
    prompt_manifest_path: str
    prompt_selection: Literal["ordered_prefix"]
    prompt_manifest_sha256: str
    trigger_successful_update: int
    sample_count: int
    cfg_scale: float
    sampling_profile: SamplingProfileName
    solver: SamplingSolver
    time_schedule: TimeSchedule
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
        artifact_kinds_by_metric: dict[MetricName, tuple[ArtifactKind, ...]] = {
            "fid": ("fid_trend", "fid_formal"),
            "is": ("is_trend", "is_formal"),
            "manual_quality": ("manual_quality",),
            "vae_reconstruction": ("vae_reconstruction",),
        }
        expected_artifact_kinds = artifact_kinds_by_metric[self.metric]
        if self.artifact_kind not in expected_artifact_kinds:
            raise ValueError("metric and artifact kind are inconsistent")
        if (
            type(self.prompt_manifest_path) is not str
            or not self.prompt_manifest_path
            or self.prompt_manifest_path != self.prompt_manifest_path.strip()
        ):
            raise ValueError("prompt manifest path must be explicit")
        if self.prompt_selection != "ordered_prefix":
            raise ValueError("prompt selection must use the ordered manifest prefix")
        for name, value in (
            ("prompt_manifest_sha256", self.prompt_manifest_sha256),
            ("preprocess_sha256", self.preprocess_sha256),
            ("real_stats_sha256", self.real_stats_sha256),
        ):
            _sha256(name, value)
        if (
            type(self.trigger_successful_update) is not int
            or self.trigger_successful_update <= 0
        ):
            raise ValueError("trigger successful update must be positive")
        if self.checkpoint.successful_update > self.trigger_successful_update:
            raise ValueError("evaluation cannot use a future checkpoint")
        if (
            self.checkpoint.kind == "raw_latest"
            and self.checkpoint.successful_update != self.trigger_successful_update
        ):
            raise ValueError("raw latest checkpoint must match the trigger update")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("evaluation sample count must be positive")
        if self.metric in ("fid", "is") and self.sample_count < 2:
            raise ValueError("FID/IS evaluation requires at least two samples")
        if type(self.cfg_scale) is not float or self.cfg_scale != 2.9:
            raise ValueError("formal evaluation CFG must equal 2.9")
        selected = resolve_sampling_profile(self.sampling_profile)
        if self.sampling_profile != "reference":
            raise ValueError("formal evaluation must select the reference profile")
        if (
            self.solver != selected.solver
            or self.time_schedule != selected.time_schedule
            or self.solver_steps != selected.steps
            or self.solver_nfe != selected.nfe
        ):
            raise ValueError("formal evaluation sampling identity is inconsistent")
        for value in (self.feature_extractor, self.feature_extractor_version):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("feature extractor identity must be explicit")
        if type(self.is_splits) is not int or self.is_splits <= 0:
            raise ValueError("IS splits must be positive")
        if self.metric == "is" and self.sample_count % self.is_splits != 0:
            raise ValueError("IS sample count must be exactly divisible by splits")
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError("evaluation GPU index must be nonnegative")
        if type(self.training_paused) is not bool:
            raise TypeError("training_paused must be explicit")

    def identity_mapping(self) -> dict[str, object]:
        """Return every immutable input that defines this evaluator job."""

        return {
            "artifact_kind": self.artifact_kind,
            "cfg_scale": self.cfg_scale,
            "checkpoint": {
                "checkpoint_id": self.checkpoint.checkpoint_id,
                "kind": self.checkpoint.kind,
                "resolved_config_sha256": self.checkpoint.resolved_config_sha256,
                "successful_update": self.checkpoint.successful_update,
            },
            "feature_extractor": self.feature_extractor,
            "feature_extractor_version": self.feature_extractor_version,
            "gpu_index": self.gpu_index,
            "is_splits": self.is_splits,
            "metric": self.metric,
            "preprocess_sha256": self.preprocess_sha256,
            "prompt_manifest_path": self.prompt_manifest_path,
            "prompt_selection": self.prompt_selection,
            "prompt_manifest_sha256": self.prompt_manifest_sha256,
            "real_stats_sha256": self.real_stats_sha256,
            "sample_count": self.sample_count,
            "sampling_profile": self.sampling_profile,
            "schema_version": 1,
            "solver": self.solver,
            "solver_nfe": self.solver_nfe,
            "solver_steps": self.solver_steps,
            "trigger_successful_update": self.trigger_successful_update,
            "time_schedule": self.time_schedule,
            "training_paused": self.training_paused,
        }

    def comparison_mapping(self) -> dict[str, object]:
        """Return the shared evaluation protocol for checkpoint comparisons."""

        mapping = self.identity_mapping()
        for field in (
            "checkpoint",
            "gpu_index",
            "prompt_manifest_path",
            "trigger_successful_update",
            "training_paused",
        ):
            del mapping[field]
        return mapping

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.identity_mapping(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    @property
    def content_addressed_id(self) -> str:
        return f"eval-{self.identity_sha256[:24]}"

    def as_mapping(self) -> dict[str, object]:
        return {"job_id": self.job_id, **self.identity_mapping()}


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
