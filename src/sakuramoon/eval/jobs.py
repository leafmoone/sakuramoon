"""Immutable evaluator jobs derived only from strict TOML and manifests."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import CheckpointRef, EvaluationJob, PromptManifest


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
    successful_update: int,
    metric: str,
    artifact_kind: str,
    sample_count: int,
) -> EvaluationJob:
    candidate = EvaluationJob(
        job_id="eval-content-address-pending",
        checkpoint=checkpoint,
        metric=metric,  # pyright: ignore[reportArgumentType]
        artifact_kind=artifact_kind,  # pyright: ignore[reportArgumentType]
        prompt_manifest_path=config.evaluation.prompt_manifest_path,
        prompt_selection="ordered_prefix",
        prompt_manifest_sha256=prompts.sha256,
        trigger_successful_update=successful_update,
        sample_count=sample_count,
        batch_size=config.evaluation.batch_size,
        cfg_scale=config.cfg.scale,
        sampling_profile=config.evaluation.sampling.profile,
        solver=config.evaluation.sampling.solver,
        time_schedule=config.evaluation.sampling.time_schedule,
        solver_steps=config.evaluation.sampling.steps,
        solver_nfe=config.evaluation.sampling.nfe,
        feature_extractor=config.evaluation.fid.feature_extractor,
        feature_extractor_version=(
            config.evaluation.fid.feature_extractor_version
        ),
        feature_extractor_path=config.evaluation.fid.feature_extractor_path,
        feature_extractor_sha256=config.evaluation.fid.feature_extractor_sha256,
        preprocess_path=config.evaluation.fid.preprocess_path,
        preprocess_sha256=config.evaluation.fid.preprocess_sha256,
        real_stats_path=config.evaluation.fid.real_stats_path,
        real_stats_sha256=config.evaluation.fid.real_stats_sha256,
        is_splits=config.evaluation.is_.splits,
        gpu_index=config.evaluation.gpu_index,
        training_paused=config.evaluation.training_paused,
    )
    return dataclasses.replace(candidate, job_id=candidate.content_addressed_id)


def build_evaluation_jobs(
    config: RuntimeConfig,
    *,
    checkpoint: CheckpointRef,
    successful_update: int,
    stage_end: bool,
    prompts: PromptManifest | None = None,
) -> tuple[EvaluationJob, ...]:
    requests = scheduled_evaluations(
        config.evaluation,
        successful_update=successful_update,
        stage_end=stage_end,
    )
    manual_due = stage_end and config.evaluation.manual_quality.enabled
    if not requests and not manual_due:
        return ()
    selected_prompts = prompts or load_prompt_manifest(
        Path(config.evaluation.prompt_manifest_path)
    )
    if selected_prompts.sha256 != config.evaluation.prompt_manifest_sha256:
        raise ValueError("prompt manifest hash differs from resolved configuration")
    if checkpoint.resolved_config_sha256 == "0" * 64:
        raise ValueError("checkpoint resolved-config identity is invalid")
    jobs: list[EvaluationJob] = []
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
                successful_update=successful_update,
                metric=request.metric,
                artifact_kind=f"{request.metric}_{request.run_kind}",
                sample_count=request.sample_count,
            )
        )
    if manual_due:
        manual_samples = config.evaluation.manual_quality.samples
        if len(selected_prompts.cases) < manual_samples:
            raise ValueError(
                "prompt manifest has fewer cases than the manual quality sample count"
            )
        jobs.append(
            _job(
                config,
                checkpoint=checkpoint,
                prompts=selected_prompts,
                successful_update=successful_update,
                metric="manual_quality",
                artifact_kind="manual_quality",
                sample_count=manual_samples,
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


__all__ = ["build_evaluation_jobs", "load_prompt_manifest", "write_evaluation_job"]
