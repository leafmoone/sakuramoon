from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sakuramoon.cli.eval import build_evaluation_jobs, write_evaluation_job
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.spec import CheckpointRef, PromptCase, PromptManifest


def test_jobs_bind_resolved_toml_checkpoint_and_prompt_manifest(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    prompts = PromptManifest(
        (PromptCase("p1", "1girl, red dress", ("artist:a",), 7, 512, 384),)
    )
    valid_payload["evaluation"]["prompt_manifest_sha256"] = prompts.sha256
    config = RuntimeConfig.model_validate(valid_payload)
    checkpoint = CheckpointRef("checkpoint-1", "raw_latest", "8" * 64)

    first = build_evaluation_jobs(
        config,
        checkpoint=checkpoint,
        prompts=prompts,
        successful_update=10,
        stage_end=False,
    )
    second = build_evaluation_jobs(
        config,
        checkpoint=checkpoint,
        prompts=prompts,
        successful_update=10,
        stage_end=False,
    )

    assert first == second
    assert [job.metric for job in first] == ["fid", "is"]
    assert all(job.prompt_manifest_sha256 == prompts.sha256 for job in first)
    path = tmp_path / "job.json"
    write_evaluation_job(path, first[0])
    assert json.loads(path.read_text())["job_id"] == first[0].job_id
    with pytest.raises(FileExistsError):
        write_evaluation_job(path, first[0])


def test_job_builder_rejects_prompt_hash_drift(valid_payload: dict[str, Any]) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    prompts = PromptManifest((PromptCase("p1", "landscape", (), 7, 512, 512),))

    with pytest.raises(ValueError, match="prompt manifest hash"):
        build_evaluation_jobs(
            config,
            checkpoint=CheckpointRef("checkpoint-1", "raw_latest", "8" * 64),
            prompts=prompts,
            successful_update=10,
            stage_end=False,
        )
