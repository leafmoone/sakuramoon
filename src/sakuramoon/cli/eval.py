"""Build immutable evaluation jobs only from validated TOML and bound manifests."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import (
    CheckpointRef,
    EvaluationJob,
    PromptManifest,
)


def load_prompt_manifest(path: Path) -> PromptManifest:
    """Load one immutable canonical prompt plan without following symlinks."""

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


def build_evaluation_jobs(
    config: RuntimeConfig,
    *,
    checkpoint: CheckpointRef,
    successful_update: int,
    stage_end: bool,
) -> tuple[EvaluationJob, ...]:
    requests = scheduled_evaluations(
        config.evaluation,
        successful_update=successful_update,
        stage_end=stage_end,
    )
    if not requests:
        return ()
    prompts = load_prompt_manifest(Path(config.evaluation.prompt_manifest_path))
    if prompts.sha256 != config.evaluation.prompt_manifest_sha256:
        raise ValueError("prompt manifest hash differs from resolved configuration")
    if checkpoint.resolved_config_sha256 == "0" * 64:
        raise ValueError("checkpoint resolved-config identity is invalid")
    jobs: list[EvaluationJob] = []
    for request in requests:
        artifact_kind = f"{request.metric}_{request.run_kind}"
        if len(prompts.cases) < request.sample_count:
            raise ValueError(
                "prompt manifest has fewer cases than the scheduled sample count"
            )
        candidate = EvaluationJob(
            job_id="eval-content-address-pending",
            checkpoint=checkpoint,
            metric=request.metric,
            artifact_kind=artifact_kind,  # pyright: ignore[reportArgumentType]
            prompt_manifest_path=config.evaluation.prompt_manifest_path,
            prompt_selection="ordered_prefix",
            prompt_manifest_sha256=prompts.sha256,
            trigger_successful_update=successful_update,
            sample_count=request.sample_count,
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
            preprocess_sha256=config.evaluation.fid.preprocess_sha256,
            real_stats_sha256=config.evaluation.fid.real_stats_sha256,
            is_splits=config.evaluation.is_.splits,
            gpu_index=config.evaluation.gpu_index,
            training_paused=config.evaluation.training_paused,
        )
        jobs.append(
            dataclasses.replace(
                candidate,
                job_id=candidate.content_addressed_id,
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
