"""Deterministic checkpoint-driven evaluation job identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal, cast

from sakuramoon.data.caption import CaptionDropoutHits, CaptionPlan, Tag
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    render_caption_segments,
)
from sakuramoon.eval.timing import require_plausible_single_gpu_timing
from sakuramoon.sampling.profiles import (
    SamplingProfileName,
    SamplingSolver,
    TimeSchedule,
    resolve_sampling_profile,
)

CheckpointRole = Literal["raw", "model-only", "pma", "accepted"]
CheckpointArtifactKind = Literal["raw", "model-only", "pma", "release"]
ObjectiveProvenance = Literal["strict_jlt", "pre_fix"]
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
_NO_DROPOUT = CaptionDropoutHits(
    all_condition=False,
    nsfw=False,
    character=False,
    copyright=False,
    general=False,
    artist=False,
    candidate_source=False,
    long_names=False,
    long_no_names=False,
    short_vibes=False,
    nl2=False,
    nl3=False,
)
_CAPTION_TAG_FIELDS = ("nsfw", "character", "copyright", "general", "artists")


def _safe_id(name: str, value: str) -> None:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def _sha256(name: str, value: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def caption_plan_prompt_text(plan: CaptionPlan) -> str:
    """Return the exact untruncated Qwen text surface for a typed caption plan."""

    body, artist_text = render_caption_segments(plan)
    return f"{SYSTEM_PREFIX}{body}{MAIN_SUFFIX}{artist_text}"


def _caption_plan_mapping(plan: CaptionPlan) -> dict[str, object]:
    return {
        **{
            field: [
                {"canonical": tag.canonical, "text": tag.text}
                for tag in getattr(plan, field)
            ]
            for field in _CAPTION_TAG_FIELDS
        },
        "nl_text": plan.nl_text,
        "selected_nl": plan.selected_nl,
    }


def _parse_caption_tags(value: object, field: str) -> tuple[Tag, ...]:
    if type(value) is not list:
        raise ValueError(f"prompt caption {field} must be an array")
    result: list[Tag] = []
    for raw_tag in cast(list[object], value):
        if type(raw_tag) is not dict:
            raise ValueError(f"prompt caption {field} tag must be an object")
        tag = cast(dict[str, object], raw_tag)
        if set(tag) != {"canonical", "text"}:
            raise ValueError(f"prompt caption {field} tag fields are invalid")
        result.append(Tag(cast(str, tag["text"]), cast(str, tag["canonical"])))
    return tuple(result)


def _parse_caption_plan(value: object) -> CaptionPlan | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("prompt caption plan must be an object or null")
    document = cast(dict[str, object], value)
    if set(document) != {*_CAPTION_TAG_FIELDS, "nl_text", "selected_nl"}:
        raise ValueError("prompt caption plan fields are invalid")
    nl_text = document["nl_text"]
    selected_nl = document["selected_nl"]
    if nl_text is not None and type(nl_text) is not str:
        raise ValueError("prompt caption NL text is invalid")
    if selected_nl is not None and selected_nl not in (
        "long_names",
        "long_no_names",
        "short_vibes",
        "nl2",
        "nl3",
    ):
        raise ValueError("prompt caption NL branch is invalid")
    return CaptionPlan(
        nsfw=_parse_caption_tags(document["nsfw"], "nsfw"),
        character=_parse_caption_tags(document["character"], "character"),
        copyright=_parse_caption_tags(document["copyright"], "copyright"),
        general=_parse_caption_tags(document["general"], "general"),
        artists=_parse_caption_tags(document["artists"], "artists"),
        nl_text=nl_text,
        selected_nl=selected_nl,
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


@dataclass(frozen=True, slots=True)
class PromptCase:
    prompt_id: str
    prompt: str
    conditions: tuple[str, ...]
    seed: int
    height: int
    width: int
    caption_plan: CaptionPlan | None = None

    def __post_init__(self) -> None:
        _safe_id("prompt_id", self.prompt_id)
        if (
            type(self.prompt) is not str
            or not self.prompt.strip()
            or (
                self.caption_plan is None
                and self.prompt != self.prompt.strip()
            )
            or "<think>" in self.prompt
            or "</think>" in self.prompt
        ):
            raise ValueError("prompt text is invalid")
        if (
            type(self.conditions) is not tuple
            or any(
                type(value) is not str or not value or value != value.strip()
                or ", " in value
                or "\n" in value
                or "<think>" in value
                or "</think>" in value
                for value in self.conditions
            )
            or len(set(self.conditions)) != len(self.conditions)
        ):
            raise ValueError("prompt conditions must be complete tag boundaries")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("prompt seed must be a nonnegative integer")
        if any(
            type(value) is not int or value <= 0 or value % 16 != 0
            for value in (self.height, self.width)
        ):
            raise ValueError("prompt dimensions must be positive multiples of 16")
        if self.caption_plan is not None:
            if type(self.caption_plan) is not CaptionPlan:
                raise ValueError("structured evaluator caption plan is invalid")
            plan = self.caption_plan
            typed_tags = all(
                type(getattr(plan, field)) is tuple
                and all(type(tag) is Tag for tag in getattr(plan, field))
                for field in _CAPTION_TAG_FIELDS
            )
            has_content = any(
                getattr(plan, field) for field in _CAPTION_TAG_FIELDS
            ) or plan.nl_text is not None
            if (
                not typed_tags
                or not has_content
                or plan.all_condition_dropped
                or any(plan.dropout_hits.as_mapping().values())
                or self.prompt != caption_plan_prompt_text(plan)
            ):
                raise ValueError("structured evaluator caption plan is invalid")

    def as_mapping(self) -> dict[str, object]:
        return {
            "caption_plan": (
                _caption_plan_mapping(self.caption_plan)
                if self.caption_plan is not None
                else None
            ),
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
            "caption_plan",
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
                    caption_plan=_parse_caption_plan(case["caption_plan"]),
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
    role: CheckpointRole
    artifact_kind: CheckpointArtifactKind
    objective_provenance: ObjectiveProvenance
    resolved_config_sha256: str
    successful_update: int

    def __post_init__(self) -> None:
        _safe_id("checkpoint_id", self.checkpoint_id)
        if self.role not in ("raw", "model-only", "pma", "accepted"):
            raise ValueError("checkpoint role is invalid")
        if self.artifact_kind not in ("raw", "model-only", "pma", "release"):
            raise ValueError("checkpoint artifact kind is invalid")
        allowed_artifacts: dict[CheckpointRole, tuple[CheckpointArtifactKind, ...]] = {
            "raw": ("raw",),
            "model-only": ("model-only",),
            "pma": ("pma",),
            "accepted": ("raw", "release"),
        }
        if self.artifact_kind not in allowed_artifacts[self.role]:
            raise ValueError("checkpoint role and artifact kind are inconsistent")
        if self.objective_provenance not in ("strict_jlt", "pre_fix"):
            raise ValueError("checkpoint objective provenance is invalid")
        if self.objective_provenance == "pre_fix" and self.role != "model-only":
            raise ValueError("pre-fix objective is restricted to model-only inference")
        _sha256("resolved_config_sha256", self.resolved_config_sha256)
        if type(self.successful_update) is not int or self.successful_update < 0:
            raise ValueError("checkpoint successful update must be nonnegative")


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    job_id: str
    checkpoint: CheckpointRef
    metric: MetricName
    artifact_kind: ArtifactKind
    validation_selection_path: str
    validation_selection_id: str
    validation_manifest_id: str
    validation_shard_root: str
    validation_seed: int
    prompt_selection: Literal["validation_bucketed_prefix"]
    prompt_manifest_sha256: str
    trigger_successful_update: int
    sample_count: int
    batch_size: int
    cfg_scale: float
    sampling_profile: SamplingProfileName
    solver: SamplingSolver
    time_schedule: TimeSchedule
    solver_steps: int
    solver_nfe: int
    feature_extractor: str | None
    feature_extractor_version: str | None
    feature_extractor_path: str | None
    feature_extractor_sha256: str | None
    preprocess_path: str | None
    preprocess_sha256: str | None
    real_stats_path: str | None
    real_stats_sha256: str | None
    real_stats_metadata_path: str | None
    real_stats_metadata_sha256: str | None
    is_splits: int | None
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
            any(
                type(value) is not str
                or not value
                or value != value.strip()
                for value in (
                    self.validation_selection_path,
                    self.validation_shard_root,
                )
            )
            or type(self.validation_seed) is not int
            or self.validation_seed != 44
        ):
            raise ValueError("validation prompt source must be explicit")
        _sha256("validation_selection_id", self.validation_selection_id)
        _sha256("validation_manifest_id", self.validation_manifest_id)
        if self.prompt_selection != "validation_bucketed_prefix":
            raise ValueError("prompt selection must use the bucketed validation prefix")
        _sha256("prompt_manifest_sha256", self.prompt_manifest_sha256)
        if (
            type(self.trigger_successful_update) is not int
            or self.trigger_successful_update <= 0
        ):
            raise ValueError("trigger successful update must be positive")
        if self.checkpoint.successful_update > self.trigger_successful_update:
            raise ValueError("evaluation cannot use a future checkpoint")
        if (
            self.checkpoint.role == "raw"
            and self.checkpoint.successful_update != self.trigger_successful_update
        ):
            raise ValueError("raw checkpoint must match the trigger update")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("evaluation sample count must be positive")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("evaluation batch size must be positive")
        if self.sample_count % self.batch_size:
            raise ValueError(
                "evaluation sample count must be exactly divisible by batch size"
            )
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
        extractor_values = (
            self.feature_extractor,
            self.feature_extractor_version,
            self.feature_extractor_path,
            self.preprocess_path,
        )
        extractor_hashes = (
            ("feature_extractor_sha256", self.feature_extractor_sha256),
            ("preprocess_sha256", self.preprocess_sha256),
        )
        if self.metric in ("fid", "is"):
            if any(
                type(value) is not str or not value or value != value.strip()
                for value in extractor_values
            ):
                raise ValueError("feature extractor identity must be explicit")
            for name, value in extractor_hashes:
                if type(value) is not str:
                    raise ValueError("feature extractor identity must be explicit")
                _sha256(name, value)
        elif any(value is not None for value in (*extractor_values, *(value for _, value in extractor_hashes))):
            raise ValueError("manual quality jobs must not bind an extractor")
        if self.metric == "fid":
            if (
                type(self.real_stats_path) is not str
                or not self.real_stats_path
                or self.real_stats_path != self.real_stats_path.strip()
                or type(self.real_stats_sha256) is not str
                or type(self.real_stats_metadata_path) is not str
                or not self.real_stats_metadata_path
                or self.real_stats_metadata_path
                != self.real_stats_metadata_path.strip()
                or type(self.real_stats_metadata_sha256) is not str
            ):
                raise ValueError("FID real-stat identity must be explicit")
            _sha256("real_stats_sha256", self.real_stats_sha256)
            _sha256(
                "real_stats_metadata_sha256", self.real_stats_metadata_sha256
            )
            if self.is_splits is not None:
                raise ValueError("FID jobs must not bind IS splits")
        elif self.metric == "is":
            if any(
                value is not None
                for value in (
                    self.real_stats_path,
                    self.real_stats_sha256,
                    self.real_stats_metadata_path,
                    self.real_stats_metadata_sha256,
                )
            ):
                raise ValueError("IS jobs must not bind real statistics")
            if type(self.is_splits) is not int or self.is_splits <= 0:
                raise ValueError("IS splits must be positive")
            if self.sample_count % self.is_splits != 0:
                raise ValueError("IS sample count must be exactly divisible by splits")
        elif any(
            value is not None
            for value in (
                self.real_stats_path,
                self.real_stats_sha256,
                self.real_stats_metadata_path,
                self.real_stats_metadata_sha256,
                self.is_splits,
            )
        ):
            raise ValueError("manual quality jobs must not bind metric dependencies")
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError("evaluation GPU index must be nonnegative")
        if type(self.training_paused) is not bool:
            raise TypeError("training_paused must be explicit")

    def identity_mapping(self) -> dict[str, object]:
        """Return every immutable input that defines this evaluator job."""

        return {
            "artifact_kind": self.artifact_kind,
            "batch_size": self.batch_size,
            "cfg_scale": self.cfg_scale,
            "checkpoint": {
                "artifact_kind": self.checkpoint.artifact_kind,
                "checkpoint_id": self.checkpoint.checkpoint_id,
                "objective_provenance": self.checkpoint.objective_provenance,
                "resolved_config_sha256": self.checkpoint.resolved_config_sha256,
                "role": self.checkpoint.role,
                "successful_update": self.checkpoint.successful_update,
            },
            "feature_extractor": self.feature_extractor,
            "feature_extractor_path": self.feature_extractor_path,
            "feature_extractor_sha256": self.feature_extractor_sha256,
            "feature_extractor_version": self.feature_extractor_version,
            "gpu_index": self.gpu_index,
            "is_splits": self.is_splits,
            "metric": self.metric,
            "preprocess_path": self.preprocess_path,
            "preprocess_sha256": self.preprocess_sha256,
            "prompt_selection": self.prompt_selection,
            "prompt_manifest_sha256": self.prompt_manifest_sha256,
            "real_stats_path": self.real_stats_path,
            "real_stats_sha256": self.real_stats_sha256,
            "real_stats_metadata_path": self.real_stats_metadata_path,
            "real_stats_metadata_sha256": self.real_stats_metadata_sha256,
            "sample_count": self.sample_count,
            "sampling_profile": self.sampling_profile,
            "schema_version": 1,
            "solver": self.solver,
            "solver_nfe": self.solver_nfe,
            "solver_steps": self.solver_steps,
            "trigger_successful_update": self.trigger_successful_update,
            "time_schedule": self.time_schedule,
            "training_paused": self.training_paused,
            "validation_manifest_id": self.validation_manifest_id,
            "validation_seed": self.validation_seed,
            "validation_selection_id": self.validation_selection_id,
            "validation_selection_path": self.validation_selection_path,
            "validation_shard_root": self.validation_shard_root,
        }

    def comparison_mapping(self) -> dict[str, object]:
        """Return the shared evaluation protocol for checkpoint comparisons."""

        mapping = self.identity_mapping()
        for field in (
            "checkpoint",
            "gpu_index",
            "feature_extractor_path",
            "preprocess_path",
            "real_stats_path",
            "real_stats_metadata_path",
            "trigger_successful_update",
            "training_paused",
            "validation_selection_path",
            "validation_shard_root",
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
        require_plausible_single_gpu_timing(
            wall_seconds=self.wall_seconds,
            gpu_seconds=self.gpu_seconds,
        )
        if self.training_pause_seconds > self.wall_seconds:
            raise ValueError("training pause cannot exceed evaluation wall time")


__all__ = [
    "ArtifactKind",
    "CheckpointArtifactKind",
    "CheckpointRef",
    "CheckpointRole",
    "EvaluationCost",
    "EvaluationJob",
    "MetricName",
    "ObjectiveProvenance",
    "PromptCase",
    "PromptManifest",
    "caption_plan_prompt_text",
]
