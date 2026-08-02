from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from sakuramoon.cli.eval import (
    build_evaluation_jobs,
    load_prompt_manifest,
    write_evaluation_job,
)
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.spec import CheckpointRef, PromptCase, PromptManifest


def _prompt_manifest(count: int) -> PromptManifest:
    return PromptManifest(
        tuple(
            PromptCase(f"p{index:05d}", f"prompt {index}", (), index, 512, 512)
            for index in range(count)
        )
    )


def test_jobs_bind_resolved_toml_checkpoint_and_prompt_manifest(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    prompts = _prompt_manifest(100)
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_bytes(prompts.canonical_bytes())
    valid_payload["evaluation"]["prompt_manifest_path"] = str(prompt_path)
    valid_payload["evaluation"]["prompt_manifest_sha256"] = prompts.sha256
    config = RuntimeConfig.model_validate(valid_payload)
    checkpoint = CheckpointRef(
        "checkpoint-1", "raw", "raw", "strict_jlt", "8" * 64, 10
    )

    first = build_evaluation_jobs(
        config,
        checkpoint=checkpoint,
        successful_update=10,
        stage_end=False,
    )
    second = build_evaluation_jobs(
        config,
        checkpoint=checkpoint,
        successful_update=10,
        stage_end=False,
    )

    assert first == second
    assert [job.metric for job in first] == ["fid", "is"]
    assert all(job.prompt_manifest_sha256 == prompts.sha256 for job in first)
    assert all(job.prompt_manifest_path == str(prompt_path) for job in first)
    assert all(job.prompt_selection == "ordered_prefix" for job in first)
    assert all(job.trigger_successful_update == 10 for job in first)
    assert all(job.batch_size == config.evaluation.batch_size for job in first)
    assert all(job.job_id == job.content_addressed_id for job in first)
    assert all(
        (
            job.sampling_profile,
            job.solver,
            job.solver_steps,
            job.solver_nfe,
            job.cfg_scale,
        )
        == ("reference", "heun_final_euler", 50, 99, 2.9)
        for job in first
    )
    path = tmp_path / "job.json"
    write_evaluation_job(path, first[0])
    payload = json.loads(path.read_text())
    assert payload["job_id"] == first[0].job_id
    assert payload["sampling_profile"] == "reference"
    assert payload["solver"] == "heun_final_euler"
    assert payload["solver_nfe"] == 99
    assert payload["trigger_successful_update"] == 10
    assert payload["batch_size"] == config.evaluation.batch_size
    assert payload["prompt_manifest_path"] == str(prompt_path)
    with pytest.raises(FileExistsError):
        write_evaluation_job(path, first[0])
    with pytest.raises(ValueError, match="not content-addressed"):
        write_evaluation_job(
            tmp_path / "invalid-job.json",
            dataclasses.replace(first[0], job_id="eval-arbitrary"),
        )


def test_job_builder_rejects_prompt_hash_drift(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    prompts = _prompt_manifest(100)
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_bytes(prompts.canonical_bytes())
    valid_payload["evaluation"]["prompt_manifest_path"] = str(prompt_path)
    config = RuntimeConfig.model_validate(valid_payload)

    with pytest.raises(ValueError, match="prompt manifest hash"):
        build_evaluation_jobs(
            config,
            checkpoint=CheckpointRef(
                "checkpoint-1", "raw", "raw", "strict_jlt", "8" * 64, 10
            ),
            successful_update=10,
            stage_end=False,
        )


def test_job_builder_rejects_undersized_prompt_plan(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    prompts = _prompt_manifest(99)
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_bytes(prompts.canonical_bytes())
    valid_payload["evaluation"]["prompt_manifest_path"] = str(prompt_path)
    valid_payload["evaluation"]["prompt_manifest_sha256"] = prompts.sha256
    config = RuntimeConfig.model_validate(valid_payload)

    with pytest.raises(ValueError, match="fewer cases"):
        build_evaluation_jobs(
            config,
            checkpoint=CheckpointRef(
                "checkpoint-1", "raw", "raw", "strict_jlt", "8" * 64, 10
            ),
            successful_update=10,
            stage_end=False,
        )


def test_non_due_update_does_not_read_missing_prompt_manifest(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    valid_payload["evaluation"]["prompt_manifest_path"] = str(
        tmp_path / "missing-prompts.json"
    )
    config = RuntimeConfig.model_validate(valid_payload)

    assert build_evaluation_jobs(
        config,
        checkpoint=CheckpointRef(
            "checkpoint-1", "raw", "raw", "strict_jlt", "8" * 64, 9
        ),
        successful_update=9,
        stage_end=False,
    ) == ()


def test_disabled_fid_is_cannot_skip_stage_end_manual_quality_prompt(
    valid_payload: dict[str, Any], tmp_path: Path
) -> None:
    valid_payload["evaluation"]["prompt_manifest_path"] = str(
        tmp_path / "missing-prompts.json"
    )
    valid_payload["evaluation"]["fid"]["enabled"] = False
    valid_payload["evaluation"]["is"]["enabled"] = False
    config = RuntimeConfig.model_validate(valid_payload)

    with pytest.raises(ValueError, match="cannot be opened"):
        build_evaluation_jobs(
            config,
            checkpoint=CheckpointRef(
                "checkpoint-1", "raw", "raw", "strict_jlt", "8" * 64, 10
            ),
            successful_update=10,
            stage_end=True,
        )


def test_formal_manual_quality_cannot_be_disabled(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["evaluation"]["manual_quality"]["enabled"] = False

    with pytest.raises(ValidationError, match="literal_error"):
        RuntimeConfig.model_validate(valid_payload)


def test_prompt_manifest_loader_rejects_noncanonical_or_symlinked_file(
    tmp_path: Path,
) -> None:
    prompts = _prompt_manifest(2)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(json.loads(prompts.canonical_bytes()), indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical serialization"):
        load_prompt_manifest(noncanonical)

    canonical = tmp_path / "canonical.json"
    canonical.write_bytes(prompts.canonical_bytes())
    assert load_prompt_manifest(canonical) == prompts
    with pytest.raises(ValueError, match="canonical absolute path"):
        load_prompt_manifest(Path("canonical.json"))
    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="canonical absolute path"):
        load_prompt_manifest(nested / ".." / "canonical.json")
    symlink = tmp_path / "prompts-link.json"
    symlink.symlink_to(canonical)
    with pytest.raises(ValueError, match="contains a symlink"):
        load_prompt_manifest(symlink)


@pytest.mark.parametrize("field", ["trend_samples", "acceptance_samples"])
def test_eval_config_rejects_is_sample_counts_not_divisible_by_splits(
    valid_payload: dict[str, Any], field: str
) -> None:
    valid_payload["evaluation"]["is"][field] = 101

    with pytest.raises(ValueError, match="exactly divisible"):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    "path",
    [
        ("evaluation", "fid", "feature_extractor_path"),
        ("evaluation", "fid", "feature_extractor_sha256"),
        ("evaluation", "fid", "preprocess_path"),
        ("evaluation", "fid", "preprocess_sha256"),
        ("evaluation", "fid", "real_stats_path"),
        ("evaluation", "fid", "real_stats_sha256"),
        ("evaluation", "batch_size"),
        ("evaluation", "output_reserve_gib"),
        ("evaluation", "manual_quality"),
    ],
)
def test_formal_evaluator_identity_and_batch_fields_are_required(
    valid_payload: dict[str, Any], path: tuple[str, ...]
) -> None:
    current = valid_payload
    for part in path[:-1]:
        current = cast(dict[str, Any], current[part])
    current.pop(path[-1])

    with pytest.raises(ValidationError, match="missing"):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evaluation", "fid", "feature_extractor_path"), "/tmp/extractor"),
        (("evaluation", "fid", "preprocess_path"), "../preprocess.json"),
        (("evaluation", "fid", "real_stats_path"), " padded "),
        (("evaluation", "prompt_manifest_path"), " padded "),
        (("evaluation", "manual_quality", "samples"), 101),
        (("evaluation", "fid", "trend_samples"), 101),
    ],
)
def test_formal_evaluator_rejects_unsafe_paths_or_partial_batches(
    valid_payload: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    current = valid_payload
    for part in path[:-1]:
        current = cast(dict[str, Any], current[part])
    current[path[-1]] = value

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evaluation", "fid", "trend_samples"), 1),
        (("evaluation", "is", "acceptance_samples"), 1),
        (("evaluation", "fid", "feature_extractor"), " "),
        (("evaluation", "fid", "feature_extractor_version"), " padded "),
        (("evaluation", "prompt_manifest_path"), " padded "),
    ],
)
def test_eval_config_rejects_impossible_counts_or_padded_identities(
    valid_payload: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    target: dict[str, Any] = valid_payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        RuntimeConfig.model_validate(valid_payload)
