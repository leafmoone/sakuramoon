from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakuramoon.eval.artifacts import (
    CheckpointMetricComparison,
    EvaluationArtifact,
    write_evaluation_artifact,
)
from sakuramoon.eval.manual_quality import (
    ManualQualityObservation,
    summarize_manual_quality,
)
from sakuramoon.eval.spec import CheckpointRef, EvaluationCost, EvaluationJob


def _job() -> EvaluationJob:
    return EvaluationJob(
        job_id="eval-1",
        checkpoint=CheckpointRef("checkpoint-1", "raw_latest", "1" * 64),
        metric="fid",
        artifact_kind="fid_formal",
        prompt_manifest_sha256="2" * 64,
        sample_count=50000,
        cfg_scale=2.9,
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


def test_comparison_requires_all_three_checkpoint_kinds() -> None:
    comparison = CheckpointMetricComparison(
        "fid_formal",
        (("raw_latest", 12.0), ("pma10", 11.0), ("accepted", 13.0)),
    )
    assert comparison.values[1] == ("pma10", 11.0)

    with pytest.raises(ValueError, match="raw latest"):
        CheckpointMetricComparison(
            "fid_formal",
            (("raw_latest", 12.0), ("pma10", 11.0), ("pma10", 13.0)),
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


def test_manual_quality_rejects_duplicate_prompts() -> None:
    observation = ManualQualityObservation("p1", 5.0, 4.0, 3.0, 4.5, 4.0, False)
    with pytest.raises(ValueError, match="unique"):
        summarize_manual_quality((observation, observation))
