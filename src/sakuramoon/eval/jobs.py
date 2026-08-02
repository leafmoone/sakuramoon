"""Immutable evaluator jobs derived only from strict TOML and manifests."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from sakuramoon.config.schema import (
    EvaluationEnabledConfig,
    EvaluationExtractorEnabledConfig,
    FidEnabledConfig,
    IsEnabledConfig,
    ManualQualityEnabledConfig,
    RuntimeConfig,
)
from sakuramoon.data.validation import ValidationSelection
from sakuramoon.eval.extractor import VerifiedLocalFile
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import CheckpointRef, EvaluationJob, PromptManifest


@dataclass(frozen=True, slots=True)
class EvaluationFileDependencies:
    extractor: VerifiedLocalFile | None
    preprocess: VerifiedLocalFile | None
    real_stats: VerifiedLocalFile | None
    real_stats_metadata: VerifiedLocalFile | None


def _has_symlink_component(path: Path) -> bool:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def load_prompt_manifest(path: Path) -> PromptManifest:
    """Load one immutable canonical prompt plan without following symlinks."""

    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "configured prompt manifest path must be a canonical absolute path"
        )
    if _has_symlink_component(path):
        raise ValueError("configured prompt manifest path contains a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise ValueError("configured prompt manifest cannot be opened") from None
    try:
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("configured prompt manifest must be a regular file")
            payload = handle.read()
    except OSError:
        raise ValueError("configured prompt manifest cannot be read") from None
    return PromptManifest.from_canonical_bytes(payload)


def _job(
    config: RuntimeConfig,
    *,
    checkpoint: CheckpointRef,
    prompts: PromptManifest,
    selection: ValidationSelection,
    successful_update: int,
    metric: str,
    artifact_kind: str,
    sample_count: int,
    dependencies: EvaluationFileDependencies,
) -> EvaluationJob:
    evaluation = config.evaluation
    if not isinstance(evaluation, EvaluationEnabledConfig):
        raise TypeError("evaluation is disabled")
    uses_extractor = metric in ("fid", "is")
    if uses_extractor:
        if (
            not isinstance(evaluation.extractor, EvaluationExtractorEnabledConfig)
            or dependencies.extractor is None
            or dependencies.preprocess is None
        ):
            raise ValueError("verified evaluator extractor dependencies are required")
        extractor_name = evaluation.extractor.feature_extractor
        extractor_version = evaluation.extractor.feature_extractor_version
        extractor_path: str | None = evaluation.extractor.feature_extractor_path
        extractor_sha256: str | None = dependencies.extractor.sha256
        preprocess_path: str | None = evaluation.extractor.preprocess_path
        preprocess_sha256: str | None = dependencies.preprocess.sha256
    else:
        extractor_name = None
        extractor_version = None
        extractor_path = None
        extractor_sha256 = None
        preprocess_path = None
        preprocess_sha256 = None
    if metric == "fid":
        if (
            not isinstance(evaluation.fid, FidEnabledConfig)
            or dependencies.real_stats is None
            or dependencies.real_stats_metadata is None
        ):
            raise ValueError("verified FID real-stat dependencies are required")
        real_stats_path: str | None = evaluation.fid.real_stats_path
        real_stats_sha256: str | None = dependencies.real_stats.sha256
        real_stats_metadata_path: str | None = (
            f"{evaluation.fid.real_stats_path}.metadata.json"
        )
        real_stats_metadata_sha256: str | None = (
            dependencies.real_stats_metadata.sha256
        )
    else:
        real_stats_path = None
        real_stats_sha256 = None
        real_stats_metadata_path = None
        real_stats_metadata_sha256 = None
    is_splits = (
        evaluation.is_.splits
        if metric == "is" and isinstance(evaluation.is_, IsEnabledConfig)
        else None
    )
    candidate = EvaluationJob(
        job_id="eval-content-address-pending",
        checkpoint=checkpoint,
        metric=metric,  # pyright: ignore[reportArgumentType]
        artifact_kind=artifact_kind,  # pyright: ignore[reportArgumentType]
        validation_selection_path=config.data.validation.selection_path,
        validation_selection_id=selection.selection_id,
        validation_manifest_id=selection.manifest_id,
        validation_shard_root=config.data.validation.shard_root,
        validation_seed=selection.seed,
        prompt_selection="validation_bucketed_prefix",
        prompt_manifest_sha256=prompts.sha256,
        trigger_successful_update=successful_update,
        sample_count=sample_count,
        batch_size=evaluation.batch_size,
        cfg_scale=config.cfg.scale,
        sampling_profile=evaluation.sampling.profile,
        solver=evaluation.sampling.solver,
        time_schedule=evaluation.sampling.time_schedule,
        solver_steps=evaluation.sampling.steps,
        solver_nfe=evaluation.sampling.nfe,
        feature_extractor=extractor_name,
        feature_extractor_version=extractor_version,
        feature_extractor_path=extractor_path,
        feature_extractor_sha256=extractor_sha256,
        preprocess_path=preprocess_path,
        preprocess_sha256=preprocess_sha256,
        real_stats_path=real_stats_path,
        real_stats_sha256=real_stats_sha256,
        real_stats_metadata_path=real_stats_metadata_path,
        real_stats_metadata_sha256=real_stats_metadata_sha256,
        is_splits=is_splits,
        gpu_index=evaluation.gpu_index,
        training_paused=evaluation.training_paused,
    )
    return dataclasses.replace(candidate, job_id=candidate.content_addressed_id)


def build_evaluation_jobs(
    config: RuntimeConfig,
    *,
    checkpoint: CheckpointRef,
    successful_update: int,
    stage_end: bool,
    prompts: PromptManifest | None = None,
    selection: ValidationSelection | None = None,
    dependencies: EvaluationFileDependencies | None = None,
) -> tuple[EvaluationJob, ...]:
    evaluation = config.evaluation
    if not isinstance(evaluation, EvaluationEnabledConfig):
        return ()
    requests = scheduled_evaluations(
        evaluation,
        successful_update=successful_update,
        stage_end=stage_end,
    )
    manual_due = stage_end and evaluation.manual_quality.enabled
    if not requests and not manual_due:
        return ()
    if prompts is None or selection is None:
        raise ValueError("governed validation prompts and selection are required")
    selected_prompts = prompts
    if checkpoint.resolved_config_sha256 == "0" * 64:
        raise ValueError("checkpoint resolved-config identity is invalid")
    jobs: list[EvaluationJob] = []
    selected_dependencies = dependencies or EvaluationFileDependencies(
        None, None, None, None
    )
    for request in requests:
        if len(selected_prompts.cases) < request.sample_count:
            raise ValueError(
                "prompt manifest has fewer cases than the scheduled sample count"
            )
        jobs.append(
            _job(
                config,
                checkpoint=checkpoint,
                prompts=selected_prompts,
                selection=selection,
                successful_update=successful_update,
                metric=request.metric,
                artifact_kind=f"{request.metric}_{request.run_kind}",
                sample_count=request.sample_count,
                dependencies=selected_dependencies,
            )
        )
    if manual_due:
        if not isinstance(evaluation.manual_quality, ManualQualityEnabledConfig):
            raise RuntimeError("manual quality configuration is inconsistent")
        manual_samples = evaluation.manual_quality.samples
        if len(selected_prompts.cases) < manual_samples:
            raise ValueError(
                "prompt manifest has fewer cases than the manual quality sample count"
            )
        jobs.append(
            _job(
                config,
                checkpoint=checkpoint,
                prompts=selected_prompts,
                selection=selection,
                successful_update=successful_update,
                metric="manual_quality",
                artifact_kind="manual_quality",
                sample_count=manual_samples,
                dependencies=selected_dependencies,
            )
        )
    return tuple(jobs)


def write_evaluation_job(path: Path, job: EvaluationJob) -> None:
    """Publish one no-clobber job manifest and fsync its namespace."""

    if job.job_id != job.content_addressed_id:
        raise ValueError("evaluation job ID is not content-addressed")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("evaluation job already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("evaluation job temporary path exists")
    body = (
        json.dumps(job.as_mapping(), sort_keys=True, separators=(",", ":")) + "\n"
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
    "EvaluationFileDependencies",
    "build_evaluation_jobs",
    "load_prompt_manifest",
    "write_evaluation_job",
]
