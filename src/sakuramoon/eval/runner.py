"""Fail-closed preflight and execution for checkpoint-driven evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import Any, Literal, Protocol, cast

import torch

from sakuramoon.assets import require_local_models
from sakuramoon.checkpoint.load import (
    read_checkpoint_manifest,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.pma import PMA_WINDOW
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointManifest,
    identity_from_dict,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import (
    EvaluationEnabledConfig,
    EvaluationExtractorEnabledConfig,
    FidEnabledConfig,
    ObjectiveConfig,
    RuntimeConfig,
)
from sakuramoon.data.buckets import BucketError
from sakuramoon.data.manifest import DatasetManifestError, load_dataset_manifest
from sakuramoon.data.production import parse_modelscope_caption_fields
from sakuramoon.data.validation import (
    ValidationPromptError,
    ValidationSelection,
    ValidationSelectionError,
    canonical_validation_selection_bytes,
    load_validation_prompt_samples,
    load_validation_selection,
)
from sakuramoon.eval.artifacts import (
    CheckpointMetricComparison,
    EvaluationArtifact,
)
from sakuramoon.eval.extractor import (
    ExtractorContractError,
    RealStatsProvenance,
    TorchScriptFeatureExtractor,
    VerifiedLocalFile,
    load_real_feature_stats,
    load_real_stats_provenance,
    real_stats_provenance_path,
    verify_local_file,
)
from sakuramoon.eval.generate import CheckpointGenerator, GeneratedBatch
from sakuramoon.eval.jobs import (
    EvaluationFileDependencies,
    build_evaluation_jobs,
)
from sakuramoon.eval.manual_quality import (
    ManualQualityImage,
    ManualQualityIndex,
)
from sakuramoon.eval.metrics import (
    FeatureStats,
    FeatureStatsAccumulator,
    InceptionScoreAccumulator,
    frechet_inception_distance,
)
from sakuramoon.eval.publisher import AtomicEvaluationPublisher
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import (
    ArtifactKind,
    CheckpointArtifactKind,
    CheckpointRef,
    CheckpointRole,
    EvaluationCost,
    EvaluationJob,
    ObjectiveProvenance,
    PromptCase,
    PromptManifest,
)
from sakuramoon.eval.validation import (
    ValidationPromptPlan,
    ValidationPromptPlanError,
    build_validation_prompt_plan,
)
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.storage import StorageValidationError, require_evaluation_storage


class EvaluationGenerator(Protocol):
    def generate(self, cases: tuple[PromptCase, ...]) -> GeneratedBatch: ...


class EvaluationExtractor(Protocol):
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


def _runtime_evaluation_cost(
    *,
    wall_seconds: float,
    gpu_seconds: float,
    training_paused: bool,
    scope: Literal["checkpoint", "overall"],
) -> EvaluationCost:
    try:
        return EvaluationCost(
            wall_seconds=wall_seconds,
            gpu_seconds=gpu_seconds,
            training_pause_seconds=(wall_seconds if training_paused else 0.0),
        )
    except ValueError as error:
        raise RuntimeError(f"{scope} evaluator cost is invalid: {error}") from None


@dataclass(frozen=True, slots=True)
class EvaluationBlocker:
    code: str
    subject: str

    def as_mapping(self) -> dict[str, str]:
        return {"code": self.code, "subject": self.subject}


class EvaluationPreflightError(RuntimeError):
    def __init__(self, blockers: tuple[EvaluationBlocker, ...]) -> None:
        if not blockers:
            raise ValueError("evaluation preflight error requires blockers")
        self.blockers = blockers
        super().__init__(
            "; ".join(f"{item.code}:{item.subject}" for item in blockers)
        )


EvaluationClassification = Literal[
    "checkpoint_driven_evaluation",
    "synthetic_bounded_engineering_only",
]


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    role: CheckpointRole
    path: Path
    objective_provenance: ObjectiveProvenance
    accepted_source_pma: Path | None = None

    def __post_init__(self) -> None:
        if self.role not in ("raw", "model-only", "pma", "accepted"):
            raise ValueError("checkpoint selection role is invalid")
        if not self.path.is_absolute() or ".." in self.path.parts:
            raise ValueError(
                "checkpoint selection path must be a canonical absolute path"
            )
        if self.objective_provenance not in ("strict_jlt", "pre_fix"):
            raise ValueError("checkpoint selection objective provenance is invalid")
        if self.objective_provenance == "pre_fix" and self.role != "model-only":
            raise ValueError("pre-fix objective is restricted to model-only inference")
        if self.role != "model-only" and self.objective_provenance != "strict_jlt":
            raise ValueError("raw, PMA, and accepted checkpoints require strict JLT")
        if self.accepted_source_pma is not None:
            if self.role != "accepted":
                raise ValueError("source PMA is restricted to an accepted checkpoint")
            if (
                not self.accepted_source_pma.is_absolute()
                or ".." in self.accepted_source_pma.parts
            ):
                raise ValueError(
                    "accepted source PMA must be a canonical absolute path"
                )


@dataclass(frozen=True, slots=True)
class ValidatedCheckpoint:
    selection: CheckpointSelection
    reference: CheckpointRef
    jobs: tuple[EvaluationJob, ...]


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    loaded: LoadedConfig
    repository_root: Path
    manifest_path: Path
    selection_path: Path
    validation_shard_root: Path
    validation_selection: ValidationSelection
    prompts: PromptManifest
    batchable_cases: int
    checkpoints: tuple[ValidatedCheckpoint, ...]
    extractor_file: VerifiedLocalFile | None
    preprocess_file: VerifiedLocalFile | None
    real_stats_file: VerifiedLocalFile | None
    real_stats_metadata_file: VerifiedLocalFile | None
    real_stats_provenance: RealStatsProvenance | None
    real_stats: FeatureStats | None
    output_root: Path
    trigger_successful_update: int
    stage_end: bool
    engineering_only: bool
    plan_id: str

    def __post_init__(self) -> None:
        if type(self.stage_end) is not bool or type(self.engineering_only) is not bool:
            raise TypeError("evaluation plan mode fields must be explicit booleans")

    @property
    def jobs(self) -> tuple[EvaluationJob, ...]:
        return tuple(job for checkpoint in self.checkpoints for job in checkpoint.jobs)


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    plan_id: str
    output_path: Path
    artifact_count: int
    checkpoint_count: int
    classification: EvaluationClassification
    publication_seconds: float
    total_wall_seconds: float

    def __post_init__(self) -> None:
        if self.classification not in (
            "checkpoint_driven_evaluation",
            "synthetic_bounded_engineering_only",
        ):
            raise ValueError("evaluator result classification is invalid")
        if (
            type(self.publication_seconds) is not float
            or not math.isfinite(self.publication_seconds)
            or self.publication_seconds < 0.0
            or type(self.total_wall_seconds) is not float
            or not math.isfinite(self.total_wall_seconds)
            or self.total_wall_seconds < self.publication_seconds
        ):
            raise ValueError("evaluator result timing is invalid")


@dataclass(frozen=True, slots=True)
class _InspectedCheckpoint:
    selection: CheckpointSelection
    manifest: CheckpointManifest


@dataclass(frozen=True, slots=True)
class _RawProvenance:
    manifest: CheckpointManifest
    stage: str
    world_size: int
    resolution: int
    active_slot_ids: tuple[int, ...]
    alpha: float


def _enabled_evaluation(config: RuntimeConfig) -> EvaluationEnabledConfig:
    evaluation = config.evaluation
    if getattr(evaluation, "enabled", False) is not True:
        raise EvaluationPreflightError(
            (EvaluationBlocker("EVALUATION_DISABLED", "evaluation.enabled"),)
        )
    return cast(EvaluationEnabledConfig, evaluation)


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _repository_path(repository_root: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        candidate = path
    else:
        if ".." in PurePath(configured).parts:
            raise ValueError("repository path may not traverse")
        candidate = repository_root / path
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("repository path must be canonical and absolute")
    return candidate


def _checkpoint_artifact_allowed(
    role: CheckpointRole, artifact_kind: CheckpointArtifactKind
) -> bool:
    return artifact_kind in {
        "raw": ("raw",),
        "model-only": ("model-only",),
        "pma": ("pma",),
        "accepted": ("raw", "release"),
    }[role]


def _identity_lineage(identity: CheckpointIdentity) -> tuple[str, str, str]:
    return (
        identity.config_sha256,
        identity.dependency_sha256,
        identity.parameter_schema_sha256,
    )


def _checkpoint_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError(f"{name} is unreadable or invalid JSON") from None
    if not isinstance(payload, dict):
        raise CheckpointError(f"{name} must be a JSON object")
    return cast(dict[str, Any], payload)


def _validate_raw_strict_jlt(
    path: Path, expected_manifest: CheckpointManifest
) -> _RawProvenance:
    manifest, state = read_raw_checkpoint_state(path)
    if manifest != expected_manifest:
        raise CheckpointError("raw checkpoint identity changed during preflight")
    try:
        payload = (path / "resolved_config.toml").read_bytes()
        document = tomllib.loads(payload.decode("utf-8"))
        ObjectiveConfig.model_validate(document.get("objective"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError):
        raise CheckpointError(
            "raw checkpoint objective is not the strict JLT contract"
        ) from None
    growth = state.growth
    return _RawProvenance(
        manifest=manifest,
        stage=growth.stage,
        world_size=growth.world_size,
        resolution=growth.resolution,
        active_slot_ids=growth.active_slot_ids,
        alpha=growth.alpha,
    )


def _raw_matches_evaluation_target(
    provenance: _RawProvenance, config: RuntimeConfig
) -> bool:
    """Bind an inspected RAW source to the configured checkpoint target stage."""

    return (
        provenance.stage == config.stage.name
        and provenance.world_size == config.stage.world_size
        and provenance.resolution == config.stage.resolution
        and provenance.active_slot_ids == active_slot_ids(config.stage.depth)
        and (config.growth.enabled or provenance.alpha == 1.0)
    )


def _validate_pma_strict_jlt(
    path: Path,
    manifest: CheckpointManifest,
    raw_anchor: _RawProvenance,
    *,
    exact_raw_anchor: bool = True,
) -> None:
    if type(exact_raw_anchor) is not bool:
        raise TypeError("exact_raw_anchor must be a bool")
    document = _checkpoint_json(path / "pma_sources.json", "PMA sources")
    if set(document) != {
        "active_slot_ids",
        "resolution",
        "schema_version",
        "sources",
        "stage",
        "window",
        "world_size",
    }:
        raise CheckpointError("PMA sources have unknown or missing fields")
    raw_sources = document["sources"]
    if not isinstance(raw_sources, list):
        raise CheckpointError("PMA sources must be an array")
    source_values = cast(list[object], raw_sources)
    try:
        sources = tuple(identity_from_dict(item) for item in source_values)
    except CheckpointError:
        raise CheckpointError("PMA source identity is invalid") from None
    updates = tuple(item.update for item in sources)
    active_slot_ids = document["active_slot_ids"]
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or type(document["window"]) is not int
        or document["window"] != PMA_WINDOW
        or len(sources) != PMA_WINDOW
        or updates != tuple(sorted(set(updates)))
        or not isinstance(active_slot_ids, list)
        or any(type(item) is not int for item in cast(list[object], active_slot_ids))
        or type(document["stage"]) is not str
        or not document["stage"]
        or type(document["world_size"]) is not int
        or document["world_size"] <= 0
        or type(document["resolution"]) is not int
        or document["resolution"] <= 0
    ):
        raise CheckpointError("PMA source window is invalid")
    active_slots = tuple(
        cast(int, item) for item in cast(list[object], active_slot_ids)
    )
    anchor_identity = raw_anchor.manifest.identity
    if (
        manifest.identity.update != sources[-1].update
        or _identity_lineage(manifest.identity) != _identity_lineage(sources[-1])
        or any(
            _identity_lineage(source) != _identity_lineage(manifest.identity)
            for source in sources
        )
    ):
        raise CheckpointError("PMA source window does not match its output identity")
    if exact_raw_anchor:
        if sources[-1] != anchor_identity:
            raise CheckpointError("PMA source window does not end at the raw anchor")
    elif (
        manifest.identity.update > anchor_identity.update
        or _identity_lineage(manifest.identity) != _identity_lineage(anchor_identity)
    ):
        raise CheckpointError("accepted source PMA is newer than or differs from raw")
    if (
        raw_anchor.alpha != 1.0
        or document["stage"] != raw_anchor.stage
        or document["world_size"] != raw_anchor.world_size
        or document["resolution"] != raw_anchor.resolution
        or active_slots != raw_anchor.active_slot_ids
    ):
        raise CheckpointError("PMA source topology differs from the raw anchor")


def _validate_release_strict_jlt(
    path: Path,
    manifest: CheckpointManifest,
    pma_manifest: CheckpointManifest,
) -> None:
    document = _checkpoint_json(path / "release_source.json", "release source")
    if set(document) != {"automatic_release", "schema_version", "source"}:
        raise CheckpointError("release source has unknown or missing fields")
    try:
        source = identity_from_dict(cast(object, document["source"]))
    except CheckpointError:
        raise CheckpointError("release source identity is invalid") from None
    if (
        document["automatic_release"] is not False
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or source != pma_manifest.identity
        or manifest.identity.update != source.update
        or _identity_lineage(manifest.identity) != _identity_lineage(source)
    ):
        raise CheckpointError("release source is not the verified PMA checkpoint")


def _plan_id(
    loaded: LoadedConfig,
    checkpoints: tuple[ValidatedCheckpoint, ...],
    *,
    trigger_successful_update: int,
    stage_end: bool,
    engineering_only: bool,
) -> str:
    payload = {
        "checkpoints": [
            {
                "jobs": [job.job_id for job in checkpoint.jobs],
                "path": str(checkpoint.selection.path),
                "reference": asdict(checkpoint.reference),
            }
            for checkpoint in checkpoints
        ],
        "resolved_config_sha256": loaded.resolved_sha256,
        "schema_version": 1,
        "engineering_only": engineering_only,
        "stage_end": stage_end,
        "trigger_successful_update": trigger_successful_update,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return f"evaluation-{digest[:24]}"


def preflight_evaluator(
    loaded: LoadedConfig,
    *,
    repository_root: Path,
    selections: tuple[CheckpointSelection, ...],
    trigger_successful_update: int,
    stage_end: bool,
    engineering_only: bool = False,
) -> EvaluationPlan:
    """Validate every identity before loading a model or starting generation."""

    blockers: list[EvaluationBlocker] = []
    config = loaded.config
    evaluation = _enabled_evaluation(config)
    if (
        not repository_root.is_absolute()
        or ".." in repository_root.parts
        or not repository_root.is_dir()
        or _has_symlink_component(repository_root)
    ):
        raise EvaluationPreflightError(
            (EvaluationBlocker("REPOSITORY_ROOT_INVALID", str(repository_root)),)
        )
    if type(trigger_successful_update) is not int or trigger_successful_update <= 0:
        blockers.append(EvaluationBlocker("TRIGGER_UPDATE_INVALID", "successful_update"))
    if type(stage_end) is not bool:
        blockers.append(EvaluationBlocker("EVALUATION_MODE_INVALID", "stage_end"))
    if type(engineering_only) is not bool:
        blockers.append(
            EvaluationBlocker("EVALUATION_MODE_INVALID", "engineering_only")
        )
    if (
        config.run.intent != "eval"
        or not evaluation.explicit_job
        or evaluation.sampling.profile != "reference"
        or evaluation.gpu_index != 0
        or config.distributed.world_size != 1
        or config.distributed.backend != "native"
    ):
        blockers.append(EvaluationBlocker("EVALUATOR_CONFIG_INVALID", "runtime_topology"))
    if not selections:
        blockers.append(EvaluationBlocker("CHECKPOINT_REQUIRED", "--checkpoint"))
    roles = tuple(item.role for item in selections)
    if len(set(roles)) != len(roles):
        blockers.append(EvaluationBlocker("CHECKPOINT_ROLE_DUPLICATE", "--checkpoint"))
    if stage_end is True and type(engineering_only) is bool:
        expected_roles = {"raw"} if engineering_only else {"raw", "pma", "accepted"}
        if len(selections) != len(expected_roles) or set(roles) != expected_roles:
            blockers.append(
                EvaluationBlocker(
                    (
                        "ENGINEERING_STAGE_END_CHECKPOINT_SET_INVALID"
                        if engineering_only
                        else "STAGE_END_CHECKPOINT_SET_INVALID"
                    ),
                    ",".join(sorted(expected_roles)),
                )
            )
    try:
        require_evaluation_storage(config, repository_root)
    except StorageValidationError:
        blockers.append(
            EvaluationBlocker(
                "EVALUATION_STORAGE_INVALID", config.paths.artifact_dir
            )
        )

    manifest_path = Path(config.data.manifest.path)
    selection_path = Path(config.data.validation.selection_path)
    validation_shard_root = Path(config.data.validation.shard_root)
    validation_selection: ValidationSelection | None = None
    prompt_plan: ValidationPromptPlan | None = None
    prompts: PromptManifest | None = None
    try:
        manifest_path = _repository_path(repository_root, config.data.manifest.path)
        manifest = load_dataset_manifest(manifest_path, config.data.source)
    except (DatasetManifestError, OSError, ValueError):
        blockers.append(
            EvaluationBlocker("DATASET_MANIFEST_INVALID", str(manifest_path))
        )
        manifest = None
    if manifest is not None:
        try:
            selection_path = _repository_path(
                repository_root, config.data.validation.selection_path
            )
            validation_selection = load_validation_selection(
                selection_path, manifest
            )
        except (ValidationSelectionError, OSError, ValueError):
            blockers.append(
                EvaluationBlocker(
                    "VALIDATION_SELECTION_INVALID", str(selection_path)
                )
            )
    if validation_selection is not None:
        try:
            validation_shard_root = _repository_path(
                repository_root, config.data.validation.shard_root
            )
            samples = load_validation_prompt_samples(
                validation_selection,
                validation_shard_root,
                run_seed=config.run.seed,
                caption_fields_parser=parse_modelscope_caption_fields,
            )
            prompt_plan = build_validation_prompt_plan(
                config, validation_selection, samples
            )
            prompts = prompt_plan.prompts
        except ValidationPromptPlanError as error:
            blockers.append(EvaluationBlocker(error.code, error.subject))
        except ValidationPromptError:
            blockers.append(
                EvaluationBlocker(
                    "VALIDATION_PROMPT_SOURCE_INVALID", str(validation_shard_root)
                )
            )
        except (BucketError, OSError, ValueError):
            blockers.append(
                EvaluationBlocker(
                    "VALIDATION_BUCKET_CONTRACT_INVALID",
                    str(config.stage.resolution),
                )
            )

    due_requests = (
        scheduled_evaluations(
            evaluation,
            successful_update=trigger_successful_update,
            stage_end=stage_end,
        )
        if type(trigger_successful_update) is int
        and trigger_successful_update > 0
        and type(stage_end) is bool
        else ()
    )
    if engineering_only is True and stage_end is True and (
        due_requests or not evaluation.manual_quality.enabled
    ):
        blockers.append(
            EvaluationBlocker(
                "ENGINEERING_STAGE_END_METRIC_SET_INVALID", "manual_quality_only"
            )
        )
    needs_extractor = any(item.metric in ("fid", "is") for item in due_requests)
    needs_real_stats = any(item.metric == "fid" for item in due_requests)
    external_specs: list[tuple[str, str]] = []
    if needs_extractor:
        if not isinstance(evaluation.extractor, EvaluationExtractorEnabledConfig):
            raise RuntimeError("scheduled extractor configuration is inconsistent")
        external_specs.extend(
            (
                (
                    "FEATURE_EXTRACTOR_IDENTITY_INVALID",
                    evaluation.extractor.feature_extractor_path,
                ),
                (
                    "PREPROCESS_IDENTITY_INVALID",
                    evaluation.extractor.preprocess_path,
                ),
            )
        )
    if needs_real_stats:
        if not isinstance(evaluation.fid, FidEnabledConfig):
            raise RuntimeError("scheduled FID configuration is inconsistent")
        external_specs.append(
            ("REAL_STATS_IDENTITY_INVALID", evaluation.fid.real_stats_path)
        )
    verified: dict[str, VerifiedLocalFile] = {}
    for code, configured_path in external_specs:
        path = Path(configured_path)
        try:
            path = _repository_path(repository_root, configured_path)
            verified[code] = verify_local_file(path)
        except (ExtractorContractError, ValueError):
            blockers.append(EvaluationBlocker(code, str(path)))
    real_stats: FeatureStats | None = None
    real_stats_metadata_file: VerifiedLocalFile | None = None
    real_stats_provenance: RealStatsProvenance | None = None
    real_stats_file = verified.get("REAL_STATS_IDENTITY_INVALID")
    if real_stats_file is not None:
        try:
            real_stats = load_real_feature_stats(real_stats_file)
        except ExtractorContractError:
            blockers.append(
                EvaluationBlocker(
                    "REAL_STATS_CONTRACT_INVALID", str(real_stats_file.path)
                )
            )
        else:
            metadata_path = real_stats_provenance_path(real_stats_file.path)
            try:
                real_stats_metadata_file = verify_local_file(metadata_path)
            except ExtractorContractError:
                blockers.append(
                    EvaluationBlocker(
                        "REAL_STATS_PROVENANCE_IDENTITY_INVALID", str(metadata_path)
                    )
                )
    if (
        real_stats is not None
        and real_stats_file is not None
        and real_stats_metadata_file is not None
        and validation_selection is not None
        and prompts is not None
    ):
        extractor_file = verified.get("FEATURE_EXTRACTOR_IDENTITY_INVALID")
        preprocess_file = verified.get("PREPROCESS_IDENTITY_INVALID")
        if (
            not isinstance(evaluation.extractor, EvaluationExtractorEnabledConfig)
            or extractor_file is None
            or preprocess_file is None
        ):
            blockers.append(
                EvaluationBlocker(
                    "REAL_STATS_PROVENANCE_DEPENDENCY_MISSING", "extractor,preprocess"
                )
            )
        else:
            try:
                real_stats_provenance = load_real_stats_provenance(
                    real_stats_metadata_file,
                    real_stats_file=real_stats_file,
                    selection_id=validation_selection.selection_id,
                    manifest_id=validation_selection.manifest_id,
                    prompt_manifest_sha256=prompts.sha256,
                    preprocess_file=preprocess_file,
                    feature_extractor=evaluation.extractor.feature_extractor,
                    feature_extractor_version=(
                        evaluation.extractor.feature_extractor_version
                    ),
                    extractor_file=extractor_file,
                    stats_count=real_stats.count,
                )
            except ExtractorContractError:
                blockers.append(
                    EvaluationBlocker(
                        "REAL_STATS_PROVENANCE_CONTRACT_INVALID",
                        str(real_stats_metadata_file.path),
                    )
                )
    if prompts is not None:
        unresolved_condition = next(
            (case.prompt_id for case in prompts.cases if case.conditions), None
        )
        if unresolved_condition is not None:
            blockers.append(
                EvaluationBlocker(
                    "VALIDATION_PROMPT_CONDITION_INVALID", unresolved_condition
                )
            )
    try:
        require_local_models(repository_root)
    except FileNotFoundError:
        blockers.append(EvaluationBlocker("LOCAL_MODEL_ASSETS_MISSING", "model/"))

    inspected: list[_InspectedCheckpoint] = []
    if type(trigger_successful_update) is int and trigger_successful_update > 0:
        for selection in selections:
            if not selection.path.is_absolute():
                blockers.append(
                    EvaluationBlocker("CHECKPOINT_PATH_NOT_ABSOLUTE", selection.role)
                )
                continue
            if ".." in selection.path.parts:
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_PATH_NONCANONICAL", str(selection.path)
                    )
                )
                continue
            if _has_symlink_component(selection.path):
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_PATH_CONTAINS_SYMLINK", str(selection.path)
                    )
                )
                continue
            try:
                manifest = read_checkpoint_manifest(selection.path)
            except (CheckpointError, OSError):
                blockers.append(
                    EvaluationBlocker("CHECKPOINT_INVALID", str(selection.path))
                )
                continue
            artifact_kind = manifest.kind.value
            if not _checkpoint_artifact_allowed(selection.role, artifact_kind):
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_ROLE_KIND_MISMATCH", selection.role
                    )
                )
                continue
            inspected.append(_InspectedCheckpoint(selection, manifest))

    inspected_by_role = {item.selection.role: item for item in inspected}
    raw_provenance: dict[CheckpointRole, _RawProvenance] = {}
    derived_provenance: dict[CheckpointRole, ObjectiveProvenance] = {}
    for item in inspected:
        role = item.selection.role
        if item.manifest.kind.value == "raw":
            try:
                provenance = _validate_raw_strict_jlt(
                    item.selection.path, item.manifest
                )
            except (CheckpointError, OSError):
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_OBJECTIVE_PROVENANCE_UNVERIFIED", role
                    )
                )
                continue
            if not _raw_matches_evaluation_target(provenance, config):
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_TARGET_TOPOLOGY_MISMATCH", role
                    )
                )
                continue
            raw_provenance[role] = provenance
            derived_provenance[role] = "strict_jlt"
        elif item.manifest.kind.value == "model-only":
            if item.selection.objective_provenance != "pre_fix":
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_OBJECTIVE_PROVENANCE_UNVERIFIED", role
                    )
                )
                continue
            derived_provenance[role] = "pre_fix"

    pma_item = inspected_by_role.get("pma")
    raw_anchor = raw_provenance.get("raw")
    if pma_item is not None:
        if raw_anchor is None:
            blockers.append(
                EvaluationBlocker("PMA_SOURCE_CHAIN_INVALID", "raw_anchor_missing")
            )
        else:
            try:
                _validate_pma_strict_jlt(
                    pma_item.selection.path, pma_item.manifest, raw_anchor
                )
            except (CheckpointError, OSError):
                blockers.append(
                    EvaluationBlocker("PMA_SOURCE_CHAIN_INVALID", "pma")
                )
            else:
                derived_provenance["pma"] = "strict_jlt"

    accepted_item = inspected_by_role.get("accepted")
    if accepted_item is not None and accepted_item.manifest.kind.value == "release":
        source_path = accepted_item.selection.accepted_source_pma
        if source_path is None:
            blockers.append(
                EvaluationBlocker(
                    "ACCEPTED_RELEASE_SOURCE_PMA_REQUIRED", "accepted"
                )
            )
        elif raw_anchor is None:
            blockers.append(
                EvaluationBlocker(
                    "CHECKPOINT_OBJECTIVE_PROVENANCE_UNVERIFIED", "accepted"
                )
            )
        elif _has_symlink_component(source_path):
            blockers.append(
                EvaluationBlocker(
                    "ACCEPTED_RELEASE_SOURCE_PMA_INVALID", str(source_path)
                )
            )
        else:
            try:
                source_manifest = read_checkpoint_manifest(source_path)
                if source_manifest.kind.value != "pma":
                    raise CheckpointError("accepted release source is not PMA")
                _validate_pma_strict_jlt(
                    source_path,
                    source_manifest,
                    raw_anchor,
                    exact_raw_anchor=False,
                )
                _validate_release_strict_jlt(
                    accepted_item.selection.path,
                    accepted_item.manifest,
                    source_manifest,
                )
            except (CheckpointError, OSError):
                blockers.append(
                    EvaluationBlocker(
                        "CHECKPOINT_OBJECTIVE_PROVENANCE_UNVERIFIED", "accepted"
                    )
                )
            else:
                derived_provenance["accepted"] = "strict_jlt"
    elif (
        accepted_item is not None
        and accepted_item.selection.accepted_source_pma is not None
    ):
        blockers.append(
            EvaluationBlocker("ACCEPTED_RELEASE_SOURCE_PMA_UNEXPECTED", "accepted")
        )
        derived_provenance.pop("accepted", None)

    if stage_end is True and raw_anchor is not None:
        accepted_raw = raw_provenance.get("accepted")
        if accepted_raw is not None and (
            _identity_lineage(accepted_raw.manifest.identity)
            != _identity_lineage(raw_anchor.manifest.identity)
            or (
                accepted_raw.stage,
                accepted_raw.world_size,
                accepted_raw.resolution,
                accepted_raw.active_slot_ids,
            )
            != (
                raw_anchor.stage,
                raw_anchor.world_size,
                raw_anchor.resolution,
                raw_anchor.active_slot_ids,
            )
        ):
            blockers.append(
                EvaluationBlocker(
                    "STAGE_END_CHECKPOINT_TOPOLOGY_MISMATCH", "accepted"
                )
            )
            derived_provenance.pop("accepted", None)

    validated: list[ValidatedCheckpoint] = []
    if prompts is not None and validation_selection is not None:
        for item in inspected:
            selection = item.selection
            provenance = derived_provenance.get(selection.role)
            if provenance is None:
                continue
            if (
                selection.role == "raw"
                and item.manifest.identity.update != trigger_successful_update
            ):
                blockers.append(
                    EvaluationBlocker(
                        "RAW_TRIGGER_MISMATCH", str(item.manifest.identity.update)
                    )
                )
                continue
            artifact_kind: CheckpointArtifactKind = item.manifest.kind.value
            try:
                reference = CheckpointRef(
                    checkpoint_id=item.manifest.identity.checkpoint_id,
                    role=selection.role,
                    artifact_kind=artifact_kind,
                    objective_provenance=provenance,
                    resolved_config_sha256=item.manifest.identity.config_sha256,
                    successful_update=item.manifest.identity.update,
                )
                jobs = build_evaluation_jobs(
                    config,
                    checkpoint=reference,
                    successful_update=trigger_successful_update,
                    stage_end=stage_end,
                    prompts=prompts,
                    selection=validation_selection,
                    dependencies=EvaluationFileDependencies(
                        extractor=verified.get(
                            "FEATURE_EXTRACTOR_IDENTITY_INVALID"
                        ),
                        preprocess=verified.get("PREPROCESS_IDENTITY_INVALID"),
                        real_stats=real_stats_file,
                        real_stats_metadata=real_stats_metadata_file,
                    ),
                )
            except ValueError:
                blockers.append(
                    EvaluationBlocker("EVALUATION_JOB_INVALID", selection.role)
                )
                continue
            validated.append(ValidatedCheckpoint(selection, reference, jobs))

    due_jobs = tuple(job for checkpoint in validated for job in checkpoint.jobs)
    if validated and not due_jobs:
        blockers.append(
            EvaluationBlocker(
                "NO_EVALUATION_JOB_DUE", str(trigger_successful_update)
            )
        )
    if prompts is not None and due_jobs:
        maximum_samples = max(job.sample_count for job in due_jobs)
        if prompt_plan is None or maximum_samples > prompt_plan.batchable_cases:
            available = 0 if prompt_plan is None else prompt_plan.batchable_cases
            blockers.append(
                EvaluationBlocker(
                    "VALIDATION_SAMPLE_CAPACITY_INSUFFICIENT",
                    f"required={maximum_samples},batchable={available}",
                )
            )
        batch_size = evaluation.batch_size
        for start in range(0, min(maximum_samples, len(prompts.cases)), batch_size):
            cases = prompts.cases[start : start + batch_size]
            if len(cases) != batch_size or any(
                case.height != cases[0].height or case.width != cases[0].width
                for case in cases[1:]
            ):
                blockers.append(
                    EvaluationBlocker(
                        "PROMPT_BATCH_SHAPE_INVALID", f"validation_prefix:{start}"
                    )
                )
                break

    output_root = Path(config.paths.artifact_dir)
    try:
        output_root = _repository_path(repository_root, config.paths.artifact_dir)
    except ValueError:
        blockers.append(EvaluationBlocker("ARTIFACT_ROOT_INVALID", str(output_root)))
    else:
        if _has_symlink_component(output_root):
            blockers.append(
                EvaluationBlocker("ARTIFACT_ROOT_INVALID", str(output_root))
            )

    if blockers:
        raise EvaluationPreflightError(tuple(blockers))
    assert prompts is not None
    assert prompt_plan is not None
    assert validation_selection is not None
    extractor_file = verified.get("FEATURE_EXTRACTOR_IDENTITY_INVALID")
    preprocess_file = verified.get("PREPROCESS_IDENTITY_INVALID")
    checkpoint_tuple = tuple(validated)
    plan_id = _plan_id(
        loaded,
        checkpoint_tuple,
        trigger_successful_update=trigger_successful_update,
        stage_end=stage_end,
        engineering_only=engineering_only,
    )
    if (
        (output_root / plan_id).exists()
        or (output_root / plan_id).is_symlink()
        or (output_root / f".{plan_id}.incomplete").exists()
        or (output_root / f".{plan_id}.incomplete").is_symlink()
    ):
        raise EvaluationPreflightError(
            (EvaluationBlocker("EVALUATION_OUTPUT_EXISTS", plan_id),)
        )
    return EvaluationPlan(
        loaded=loaded,
        repository_root=repository_root,
        manifest_path=manifest_path,
        selection_path=selection_path,
        validation_shard_root=validation_shard_root,
        validation_selection=validation_selection,
        prompts=prompts,
        batchable_cases=prompt_plan.batchable_cases,
        checkpoints=checkpoint_tuple,
        extractor_file=extractor_file,
        preprocess_file=preprocess_file,
        real_stats_file=real_stats_file,
        real_stats_metadata_file=real_stats_metadata_file,
        real_stats_provenance=real_stats_provenance,
        real_stats=real_stats,
        output_root=output_root,
        trigger_successful_update=trigger_successful_update,
        stage_end=stage_end,
        engineering_only=engineering_only,
        plan_id=plan_id,
    )


GeneratorFactory = Callable[[EvaluationPlan, ValidatedCheckpoint], EvaluationGenerator]
ExtractorFactory = Callable[[EvaluationPlan], EvaluationExtractor]


def _production_generator(
    plan: EvaluationPlan, checkpoint: ValidatedCheckpoint
) -> EvaluationGenerator:
    evaluation = _enabled_evaluation(plan.loaded.config)
    device = torch.device("cuda", evaluation.gpu_index)
    return CheckpointGenerator(
        config=plan.loaded.config,
        checkpoint_path=checkpoint.selection.path,
        checkpoint=checkpoint.reference,
        repository_root=plan.repository_root,
        device=device,
    )


def _production_extractor(plan: EvaluationPlan) -> EvaluationExtractor:
    evaluation = _enabled_evaluation(plan.loaded.config)
    if plan.preprocess_file is None or plan.extractor_file is None:
        raise ExtractorContractError("verified evaluator extractor is missing")
    device = torch.device("cuda", evaluation.gpu_index)
    return TorchScriptFeatureExtractor(
        preprocess_file=plan.preprocess_file,
        extractor_file=plan.extractor_file,
        device=device,
    )


def _finalize_metric(
    job: EvaluationJob,
    *,
    fid_accumulator: FeatureStatsAccumulator | None,
    is_accumulator: InceptionScoreAccumulator | None,
    real_stats: FeatureStats | None,
) -> tuple[float, float | None]:
    if job.metric == "fid":
        if fid_accumulator is None:
            raise RuntimeError("FID accumulator is missing")
        if fid_accumulator.count != job.sample_count:
            raise RuntimeError("FID aggregation sample count differs from its job")
        if real_stats is None:
            raise RuntimeError("FID real statistics are missing")
        value = frechet_inception_distance(fid_accumulator.finalize(), real_stats)
        return value, None
    if job.metric == "is":
        if is_accumulator is None:
            raise RuntimeError("IS accumulator is missing")
        score = is_accumulator.finalize()
        return score.mean, score.std
    raise RuntimeError("manual quality jobs do not produce scalar metrics")


def run_evaluator(
    plan: EvaluationPlan,
    *,
    generator_factory: GeneratorFactory | None = None,
    extractor_factory: ExtractorFactory | None = None,
    measure_cuda: bool = True,
) -> EvaluationRunResult:
    """Execute every plan job and atomically publish only a complete run tree."""

    injected = (
        generator_factory is not None
        or extractor_factory is not None
        or not measure_cuda
    )
    if injected and not plan.engineering_only:
        raise RuntimeError(
            "injected evaluator execution must be classified as engineering-only"
        )
    if not plan.jobs:
        raise EvaluationPreflightError(
            (EvaluationBlocker("NO_EVALUATION_JOB_DUE", str(plan.trigger_successful_update)),)
        )
    stage_end_roles = {item.reference.role for item in plan.checkpoints}
    if plan.stage_end:
        stage_blockers: list[EvaluationBlocker] = []
        expected_roles = (
            {"raw"} if plan.engineering_only else {"raw", "pma", "accepted"}
        )
        if (
            len(plan.checkpoints) != len(expected_roles)
            or stage_end_roles != expected_roles
        ):
            stage_blockers.append(
                EvaluationBlocker(
                    (
                        "ENGINEERING_STAGE_END_CHECKPOINT_SET_INVALID"
                        if plan.engineering_only
                        else "STAGE_END_CHECKPOINT_SET_INVALID"
                    ),
                    ",".join(sorted(expected_roles)),
                )
            )
        if plan.engineering_only and any(
            job.metric != "manual_quality" for job in plan.jobs
        ):
            stage_blockers.append(
                EvaluationBlocker(
                    "ENGINEERING_STAGE_END_METRIC_SET_INVALID",
                    "manual_quality_only",
                )
            )
        if stage_blockers:
            raise EvaluationPreflightError(tuple(stage_blockers))
    evaluation = _enabled_evaluation(plan.loaded.config)
    if any(case.conditions for case in plan.prompts.cases):
        raise EvaluationPreflightError(
            (
                EvaluationBlocker(
                    "VALIDATION_PROMPT_CONDITION_INVALID", "generated_prompt_manifest"
                ),
            )
        )
    for job in plan.jobs:
        if (
            job.validation_selection_id
            != plan.validation_selection.selection_id
            or job.validation_manifest_id
            != plan.validation_selection.manifest_id
            or job.validation_seed != plan.validation_selection.seed
            or job.prompt_manifest_sha256 != plan.prompts.sha256
        ):
            raise EvaluationPreflightError(
                (
                    EvaluationBlocker(
                        "VALIDATION_PROVENANCE_CHANGED", job.job_id
                    ),
                )
            )
    maximum_samples = max(job.sample_count for job in plan.jobs)
    if maximum_samples > plan.batchable_cases:
        raise EvaluationPreflightError(
            (
                EvaluationBlocker(
                    "VALIDATION_SAMPLE_CAPACITY_INSUFFICIENT",
                    f"required={maximum_samples},batchable={plan.batchable_cases}",
                ),
            )
        )
    for start in range(0, maximum_samples, evaluation.batch_size):
        cases = plan.prompts.cases[start : start + evaluation.batch_size]
        if len(cases) != evaluation.batch_size or any(
            case.height != cases[0].height or case.width != cases[0].width
            for case in cases[1:]
        ):
            raise EvaluationPreflightError(
                (
                    EvaluationBlocker(
                        "PROMPT_BATCH_SHAPE_INVALID", f"validation_prefix:{start}"
                    ),
                )
            )
    if any(job.metric == "fid" for job in plan.jobs) and (
        plan.real_stats is None
        or plan.real_stats_file is None
        or plan.real_stats_metadata_file is None
        or plan.real_stats_provenance is None
    ):
        raise EvaluationPreflightError(
            (
                EvaluationBlocker(
                    "REAL_STATS_PROVENANCE_REQUIRED", "evaluation.fid.real_stats_path"
                ),
            )
        )
    make_generator = generator_factory or _production_generator
    make_extractor = extractor_factory or _production_extractor
    execution_classification: EvaluationClassification = (
        "synthetic_bounded_engineering_only"
        if plan.engineering_only
        else "checkpoint_driven_evaluation"
    )
    overall_wall_start = time.perf_counter()
    publisher = AtomicEvaluationPublisher(plan.output_root, plan.plan_id)
    publisher.write_bytes(
        "inputs/validation-selection.json",
        canonical_validation_selection_bytes(plan.validation_selection),
    )
    publisher.write_bytes(
        "inputs/validation-prompts.json", plan.prompts.canonical_bytes()
    )
    overall_start_event: torch.cuda.Event | None = None
    overall_end_event: torch.cuda.Event | None = None
    if measure_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA cost measurement is unavailable")
        overall_start_event = torch.cuda.Event(enable_timing=True)
        overall_end_event = torch.cuda.Event(enable_timing=True)
        overall_start_event.record()
    needs_metrics = any(job.metric in ("fid", "is") for job in plan.jobs)
    extractor = make_extractor(plan) if needs_metrics else None
    scalar_artifacts: list[EvaluationArtifact] = []
    artifact_count = 0
    for checkpoint in plan.checkpoints:
        if not checkpoint.jobs:
            continue
        wall_start = time.perf_counter()
        start_event: torch.cuda.Event | None = None
        end_event: torch.cuda.Event | None = None
        if measure_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        generator = make_generator(plan, checkpoint)
        metric_jobs = tuple(
            job for job in checkpoint.jobs if job.metric in ("fid", "is")
        )
        manual_jobs = tuple(
            job for job in checkpoint.jobs if job.metric == "manual_quality"
        )
        max_samples = max(job.sample_count for job in checkpoint.jobs)
        batch_size = evaluation.batch_size
        if max_samples % batch_size:
            raise RuntimeError("evaluation plan contains a partial batch")
        fid_accumulators = {
            job.job_id: FeatureStatsAccumulator()
            for job in metric_jobs
            if job.metric == "fid"
        }
        is_accumulators = {
            job.job_id: InceptionScoreAccumulator(
                sample_count=job.sample_count,
                splits=cast(int, job.is_splits),
            )
            for job in metric_jobs
            if job.metric == "is"
        }
        manual_images: dict[str, list[ManualQualityImage]] = {
            job.job_id: [] for job in manual_jobs
        }
        generation_metadata: dict[str, object] | None = None
        for start in range(0, max_samples, batch_size):
            raw_cases = plan.prompts.cases[start : start + batch_size]
            batch = generator.generate(raw_cases)
            if batch.cases != raw_cases:
                raise RuntimeError("generator changed the ordered prompt batch")
            metadata = batch.metadata.as_mapping()
            if generation_metadata is None:
                generation_metadata = metadata
            elif generation_metadata != metadata:
                raise RuntimeError("generation metadata changed during evaluator run")
            if metric_jobs and any(start < job.sample_count for job in metric_jobs):
                if extractor is None:
                    raise RuntimeError("feature extractor is missing")
                features, probabilities = extractor.extract(batch.images)
                if (
                    features.ndim != 2
                    or probabilities.ndim != 2
                    or features.shape[0] != len(batch.cases)
                    or probabilities.shape[0] != len(batch.cases)
                ):
                    raise RuntimeError(
                        "feature extractor batch size differs from generated images"
                    )
                for job in metric_jobs:
                    if start >= job.sample_count:
                        continue
                    if job.metric == "fid":
                        fid_accumulators[job.job_id].update(features)
                    else:
                        is_accumulators[job.job_id].update(probabilities)
            for job in manual_jobs:
                if start >= job.sample_count:
                    continue
                for offset, (case, image) in enumerate(
                    zip(batch.cases, batch.images, strict=True)
                ):
                    ordinal = start + offset
                    relative = (
                        f"manual/{job.job_id}/images/{ordinal:08d}-{case.prompt_id}.png"
                    )
                    _path, payload = publisher.write_png(relative, image)
                    manual_images[job.job_id].append(
                        ManualQualityImage(
                            prompt_id=case.prompt_id,
                            relative_path=relative,
                            sha256=hashlib.sha256(payload).hexdigest(),
                        )
                    )
        del generator
        metric_values = {
            job.job_id: _finalize_metric(
                job,
                fid_accumulator=fid_accumulators.get(job.job_id),
                is_accumulator=is_accumulators.get(job.job_id),
                real_stats=plan.real_stats,
            )
            for job in metric_jobs
        }
        if end_event is not None and start_event is not None:
            end_event.record()
            end_event.synchronize()
            gpu_seconds = float(
                start_event.elapsed_time(end_event) / 1000.0  # pyright: ignore[reportUnknownMemberType]
            )
        else:
            gpu_seconds = 0.0
        wall_seconds = float(time.perf_counter() - wall_start)
        cost = _runtime_evaluation_cost(
            wall_seconds=wall_seconds,
            gpu_seconds=gpu_seconds,
            training_paused=evaluation.training_paused,
            scope="checkpoint",
        )
        for job in checkpoint.jobs:
            publisher.write_json(f"jobs/{job.job_id}.json", job.as_mapping())
            if job.metric in ("fid", "is"):
                value, std = metric_values[job.job_id]
                artifact = EvaluationArtifact(
                    job=job, value=value, std=std, cost=cost
                )
                publisher.write_json(
                    f"artifacts/{job.job_id}.json",
                    {
                        **artifact.as_mapping(),
                        "execution_classification": execution_classification,
                    },
                )
                scalar_artifacts.append(artifact)
                artifact_count += 1
            else:
                index = ManualQualityIndex(
                    job=job,
                    prompt_manifest=plan.prompts,
                    images=tuple(manual_images[job.job_id]),
                    cost=cost,
                )
                publisher.write_json(
                    f"artifacts/{job.job_id}.json",
                    {
                        **index.as_mapping(),
                        "execution_classification": execution_classification,
                    },
                )
                artifact_count += 1
        publisher.write_json(
            f"generation/{checkpoint.reference.role}.json",
            {
                "checkpoint": asdict(checkpoint.reference),
                "execution_classification": execution_classification,
                "generation": generation_metadata,
                "job_ids": [job.job_id for job in checkpoint.jobs],
                "sample_count": max_samples,
                "schema_version": 1,
            },
        )

    by_kind: dict[str, list[EvaluationArtifact]] = {}
    for artifact in scalar_artifacts:
        by_kind.setdefault(artifact.job.artifact_kind, []).append(artifact)
    for artifact_kind, artifacts in by_kind.items():
        comparable = tuple(
            artifact
            for artifact in artifacts
            if artifact.job.checkpoint.role in {"raw", "pma", "accepted"}
        )
        if {artifact.job.checkpoint.role for artifact in comparable} != {
            "raw",
            "pma",
            "accepted",
        }:
            continue
        selected = tuple(
            next(
                artifact
                for artifact in comparable
                if artifact.job.checkpoint.role == role
            )
            for role in ("raw", "pma", "accepted")
        )
        comparison = CheckpointMetricComparison(
            cast(ArtifactKind, artifact_kind),
            selected,
        )
        publisher.write_json(
            f"comparisons/{artifact_kind}.json",
            {
                "artifact_kind": artifact_kind,
                "automatic_release": False,
                "execution_classification": execution_classification,
                "jobs": [artifact.job.job_id for artifact in selected],
                "schema_version": 1,
                "values": [
                    {"role": role, "value": value}
                    for role, value in comparison.values
                ],
            },
        )
        artifact_count += 1
    if overall_end_event is not None and overall_start_event is not None:
        overall_end_event.record()
        overall_end_event.synchronize()
        overall_gpu_seconds = float(
            overall_start_event.elapsed_time(overall_end_event) / 1000.0  # pyright: ignore[reportUnknownMemberType]
        )
    else:
        overall_gpu_seconds = 0.0
    overall_wall_seconds = float(time.perf_counter() - overall_wall_start)
    overall_cost = _runtime_evaluation_cost(
        wall_seconds=overall_wall_seconds,
        gpu_seconds=overall_gpu_seconds,
        training_paused=evaluation.training_paused,
        scope="overall",
    )
    publication_start = time.perf_counter()
    output = publisher.commit(
        {
            "artifact_count": artifact_count,
            "automatic_release": False,
            "checkpoint_count": len(plan.checkpoints),
            "classification": execution_classification,
            "cost": {
                "gpu_seconds": overall_cost.gpu_seconds,
                "publication_seconds_included": False,
                "training_pause_seconds": overall_cost.training_pause_seconds,
                "wall_seconds": overall_cost.wall_seconds,
            },
            "plan_id": plan.plan_id,
            "engineering_only": plan.engineering_only,
            "resolved_config_sha256": plan.loaded.resolved_sha256,
            "schema_version": 1,
            "stage_end": plan.stage_end,
            "publication_timing": {
                "atomic_commit_seconds": None,
                "recorded_in_run_result_only": True,
            },
            "trigger_successful_update": plan.trigger_successful_update,
            "validation": {
                "manifest_id": plan.validation_selection.manifest_id,
                "prompt_manifest_sha256": plan.prompts.sha256,
                "seed": plan.validation_selection.seed,
                "selection_id": plan.validation_selection.selection_id,
            },
        }
    )
    publication_seconds = float(time.perf_counter() - publication_start)
    total_wall_seconds = float(time.perf_counter() - overall_wall_start)
    return EvaluationRunResult(
        plan_id=plan.plan_id,
        output_path=output,
        artifact_count=artifact_count,
        checkpoint_count=len(plan.checkpoints),
        classification=execution_classification,
        publication_seconds=publication_seconds,
        total_wall_seconds=total_wall_seconds,
    )


__all__ = [
    "CheckpointSelection",
    "EvaluationBlocker",
    "EvaluationClassification",
    "EvaluationPlan",
    "EvaluationPreflightError",
    "EvaluationRunResult",
    "ValidatedCheckpoint",
    "preflight_evaluator",
    "run_evaluator",
]
