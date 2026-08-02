from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from sakuramoon.eval.artifacts import (
    CheckpointMetricComparison,
    EvaluationArtifact,
    write_evaluation_artifact,
)
from sakuramoon.eval.manual_quality import (
    ManualQualityArtifact,
    ManualQualityObservation,
    summarize_manual_quality,
    write_manual_quality_artifact,
)
from sakuramoon.eval.spec import (
    CheckpointKind,
    CheckpointRef,
    EvaluationCost,
    EvaluationJob,
    PromptCase,
    PromptManifest,
)


def _job() -> EvaluationJob:
    candidate = EvaluationJob(
        job_id="eval-content-address-pending",
        checkpoint=CheckpointRef("checkpoint-1", "raw_latest", "1" * 64, 10),
        metric="fid",
        artifact_kind="fid_formal",
        prompt_manifest_path="prompts.json",
        prompt_selection="ordered_prefix",
        prompt_manifest_sha256="2" * 64,
        trigger_successful_update=10,
        sample_count=50000,
        cfg_scale=2.9,
        sampling_profile="reference",
        solver="heun_final_euler",
        time_schedule="linear",
        solver_steps=50,
        solver_nfe=99,
        feature_extractor="inception",
        feature_extractor_version="1.0",
        preprocess_sha256="3" * 64,
        real_stats_sha256="4" * 64,
        is_splits=10,
        gpu_index=0,
        training_paused=True,
    )
    return dataclasses.replace(candidate, job_id=candidate.content_addressed_id)


def test_artifact_is_immutable_records_cost_and_cannot_release(tmp_path: Path) -> None:
    artifact = EvaluationArtifact(
        _job(),
        value=12.5,
        std=None,
        cost=EvaluationCost(20.0, 18.0, 20.0),
    )
    path = tmp_path / "fid-formal.json"

    write_evaluation_artifact(path, artifact)

    payload = json.loads(path.read_text())
    assert payload["artifact_kind"] == "fid_formal"
    assert payload["automatic_release"] is False
    assert payload["job"]["artifact_kind"] == "fid_formal"
    assert payload["cost"] == {
        "gpu_seconds": 18.0,
        "training_pause_seconds": 20.0,
        "wall_seconds": 20.0,
    }
    with pytest.raises(FileExistsError):
        write_evaluation_artifact(path, artifact)
    with pytest.raises(ValueError, match="cannot automatically"):
        EvaluationArtifact(
            _job(),
            value=12.5,
            std=None,
            cost=EvaluationCost(20.0, 18.0, 20.0),
            automatic_release=True,
        )
    with pytest.raises(ValueError, match="pause cost must be zero"):
        candidate = dataclasses.replace(_job(), training_paused=False)
        EvaluationArtifact(
            dataclasses.replace(candidate, job_id=candidate.content_addressed_id),
            value=12.5,
            std=None,
            cost=EvaluationCost(20.0, 18.0, 1.0),
        )


@pytest.mark.parametrize(
    ("metric", "artifact_kind", "value", "std", "message"),
    [
        ("fid", "fid_formal", -1.0, None, "FID value must be nonnegative"),
        ("is", "is_formal", 0.999, 0.0, "IS value must be at least one"),
    ],
)
def test_artifact_rejects_impossible_metric_values_before_publication(
    tmp_path: Path,
    metric: str,
    artifact_kind: str,
    value: float,
    std: float | None,
    message: str,
) -> None:
    candidate = dataclasses.replace(
        _job(),
        metric=metric,
        artifact_kind=artifact_kind,
    )
    job = dataclasses.replace(candidate, job_id=candidate.content_addressed_id)
    path = tmp_path / f"{artifact_kind}.json"

    with pytest.raises(ValueError, match=message):
        artifact = EvaluationArtifact(
            job,
            value=value,
            std=std,
            cost=EvaluationCost(1.0, 0.0, 1.0),
        )
        write_evaluation_artifact(path, artifact)

    assert not path.exists()


def test_comparison_requires_all_three_checkpoint_kinds() -> None:
    def artifact(
        kind: CheckpointKind,
        *,
        value: float = 12.0,
        prompt_hash: str = "2" * 64,
        prompt_path: str = "prompts.json",
        gpu_index: int = 0,
        training_paused: bool = True,
        trigger_successful_update: int = 10,
        checkpoint_successful_update: int | None = None,
    ) -> EvaluationArtifact:
        if checkpoint_successful_update is None:
            checkpoint_successful_update = 9 if kind == "accepted" else 10
        candidate = dataclasses.replace(
            _job(),
            checkpoint=CheckpointRef(
                f"checkpoint-{kind}",
                kind,
                "1" * 64,
                checkpoint_successful_update,
            ),
            gpu_index=gpu_index,
            prompt_manifest_path=prompt_path,
            prompt_manifest_sha256=prompt_hash,
            trigger_successful_update=trigger_successful_update,
            training_paused=training_paused,
        )
        job = dataclasses.replace(candidate, job_id=candidate.content_addressed_id)
        return EvaluationArtifact(
            job,
            value=value,
            std=None,
            cost=EvaluationCost(20.0, 18.0, 20.0 if training_paused else 0.0),
        )

    comparison = CheckpointMetricComparison(
        "fid_formal",
        (
            artifact("raw_latest", value=12.0),
            artifact(
                "pma10",
                value=11.0,
                prompt_path="other/prompts.json",
                gpu_index=1,
                training_paused=False,
            ),
            artifact(
                "accepted",
                value=13.0,
                gpu_index=2,
                trigger_successful_update=11,
            ),
        ),
    )
    assert comparison.values[1] == ("pma10", 11.0)
    assert comparison.artifacts[2].job.checkpoint.successful_update == 9

    with pytest.raises(ValueError, match="raw latest"):
        CheckpointMetricComparison(
            "fid_formal",
            (artifact("raw_latest"), artifact("pma10"), artifact("pma10")),
        )
    with pytest.raises(ValueError, match="share metric"):
        CheckpointMetricComparison(
            "fid_formal",
            (
                artifact("raw_latest"),
                artifact("pma10"),
                artifact("accepted", prompt_hash="9" * 64),
            ),
        )


@pytest.mark.parametrize(
    ("metric", "artifact_kind"),
    [
        ("manual_quality", "manual_quality"),
        ("vae_reconstruction", "vae_reconstruction"),
    ],
)
def test_scalar_artifact_rejects_manual_and_vae_jobs(
    metric: str,
    artifact_kind: str,
) -> None:
    candidate = dataclasses.replace(
        _job(),
        metric=metric,
        artifact_kind=artifact_kind,
    )
    job = dataclasses.replace(candidate, job_id=candidate.content_addressed_id)

    with pytest.raises(ValueError, match="FID or IS"):
        EvaluationArtifact(
            job,
            value=1.0,
            std=None,
            cost=EvaluationCost(1.0, 0.0, 1.0),
        )


def test_manual_quality_keeps_priority_dimensions_and_no_auto_release() -> None:
    report = summarize_manual_quality(
        (
            ManualQualityObservation("p1", 5.0, 4.0, 3.0, 4.5, 4.0, False),
            ManualQualityObservation("p2", 4.0, 3.0, 5.0, 3.5, 3.0, True),
        )
    )

    assert report.tag_control_mean == 4.5
    assert report.nl_following_mean == 4.0
    assert report.severe_artifact_rate == 0.5
    assert report.automatic_release is False
    with pytest.raises(ValueError, match="cannot automatically"):
        dataclasses.replace(report, automatic_release=True)


def test_manual_quality_rejects_duplicate_prompts() -> None:
    observation = ManualQualityObservation("p1", 5.0, 4.0, 3.0, 4.5, 4.0, False)
    with pytest.raises(ValueError, match="unique"):
        summarize_manual_quality((observation, observation))


def test_manual_quality_artifact_binds_job_checkpoint_and_prompt_plan(
    tmp_path: Path,
) -> None:
    prompts = PromptManifest(
        (
            PromptCase("p1", "red dress", (), 1, 512, 512),
            PromptCase("p2", "blue sky", (), 2, 512, 512),
        )
    )
    observations = (
        ManualQualityObservation("p1", 5.0, 4.0, 3.0, 4.5, 4.0, False),
        ManualQualityObservation("p2", 4.0, 3.0, 5.0, 3.5, 3.0, True),
    )
    candidate = dataclasses.replace(
        _job(),
        metric="manual_quality",
        artifact_kind="manual_quality",
        prompt_manifest_sha256=prompts.sha256,
        sample_count=2,
    )
    job = dataclasses.replace(candidate, job_id=candidate.content_addressed_id)
    artifact = ManualQualityArtifact(
        job=job,
        prompt_manifest=prompts,
        observations=observations,
        report=summarize_manual_quality(observations),
        cost=EvaluationCost(30.0, 20.0, 30.0),
    )
    path = tmp_path / "manual-quality.json"

    write_manual_quality_artifact(path, artifact)

    payload = json.loads(path.read_text())
    assert payload["artifact_kind"] == "manual_quality"
    assert payload["automatic_release"] is False
    assert payload["job"]["checkpoint"] == {
        "checkpoint_id": "checkpoint-1",
        "kind": "raw_latest",
        "resolved_config_sha256": "1" * 64,
        "successful_update": 10,
    }
    assert payload["job"]["trigger_successful_update"] == 10
    assert [item["prompt_id"] for item in payload["observations"]] == ["p1", "p2"]
    with pytest.raises(FileExistsError):
        write_manual_quality_artifact(path, artifact)
    with pytest.raises(ValueError, match="prompt plan"):
        dataclasses.replace(artifact, observations=tuple(reversed(observations)))
