"""Build immutable evaluation jobs only from validated TOML and bound manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import (
    CheckpointRef,
    EvaluationJob,
    PromptManifest,
)


def build_evaluation_jobs(
    config: RuntimeConfig,
    *,
    checkpoint: CheckpointRef,
    prompts: PromptManifest,
    successful_update: int,
    stage_end: bool,
) -> tuple[EvaluationJob, ...]:
    if prompts.sha256 != config.evaluation.prompt_manifest_sha256:
        raise ValueError("prompt manifest hash differs from resolved configuration")
    if checkpoint.resolved_config_sha256 == "0" * 64:
        raise ValueError("checkpoint resolved-config identity is invalid")
    jobs: list[EvaluationJob] = []
    for request in scheduled_evaluations(
        config.evaluation,
        successful_update=successful_update,
        stage_end=stage_end,
    ):
        artifact_kind = f"{request.metric}_{request.run_kind}"
        identity = {
            "artifact_kind": artifact_kind,
            "checkpoint_id": checkpoint.checkpoint_id,
            "prompt_manifest_sha256": prompts.sha256,
            "sample_count": request.sample_count,
            "successful_update": successful_update,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        jobs.append(
            EvaluationJob(
                job_id=f"eval-{digest[:24]}",
                checkpoint=checkpoint,
                metric=request.metric,
                artifact_kind=artifact_kind,  # pyright: ignore[reportArgumentType]
                prompt_manifest_sha256=prompts.sha256,
                sample_count=request.sample_count,
                cfg_scale=config.cfg.scale,
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
        )
    return tuple(jobs)


def write_evaluation_job(path: Path, job: EvaluationJob) -> None:
    """Publish one no-clobber job manifest and fsync its namespace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("evaluation job already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("evaluation job temporary path exists")
    body = (json.dumps(job.as_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()
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


__all__ = ["build_evaluation_jobs", "write_evaluation_job"]
