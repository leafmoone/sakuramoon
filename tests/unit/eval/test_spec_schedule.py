from __future__ import annotations

import dataclasses

import pytest

from sakuramoon.config.schema import (
    EvaluationConfig,
    EvaluationSamplingConfig,
    FidConfig,
    IsConfig,
)
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import (
    CheckpointRef,
    EvaluationJob,
    PromptCase,
    PromptManifest,
)


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        stage_end=True,
        explicit_job=True,
        prompt_manifest_path="prompts.json",
        prompt_manifest_sha256="1" * 64,
        gpu_index=0,
        training_paused=True,
        sampling=EvaluationSamplingConfig(profile="reference"),
        fid=FidConfig(
            enabled=True,
            every_successful_updates=10,
            trend_samples=100,
            acceptance_samples=50000,
            feature_extractor="locked-inception",
            feature_extractor_version="1.0",
            preprocess_sha256="2" * 64,
            real_stats_sha256="5" * 64,
        ),
        **{
            "is": IsConfig(
                enabled=True,
                every_successful_updates=20,
                trend_samples=200,
                acceptance_samples=50000,
                splits=10,
            )
        },
    )


def _job() -> EvaluationJob:
    return EvaluationJob(
        job_id="eval-1",
        checkpoint=CheckpointRef("checkpoint-1", "raw_latest", "1" * 64),
        metric="fid",
        artifact_kind="fid_trend",
        prompt_manifest_sha256="2" * 64,
        sample_count=100,
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


def test_schedule_is_config_driven_by_successful_updates_and_stage_end() -> None:
    config = _config()

    assert scheduled_evaluations(config, successful_update=9, stage_end=False) == ()
    at_ten = scheduled_evaluations(config, successful_update=10, stage_end=False)
    assert [(item.metric, item.run_kind, item.sample_count) for item in at_ten] == [
        ("fid", "trend", 100)
    ]
    at_twenty = scheduled_evaluations(config, successful_update=20, stage_end=False)
    assert [(item.metric, item.sample_count) for item in at_twenty] == [
        ("fid", 100),
        ("is", 200),
    ]
    formal = scheduled_evaluations(config, successful_update=21, stage_end=True)
    assert [(item.metric, item.run_kind, item.sample_count) for item in formal] == [
        ("fid", "formal", 50000),
        ("is", "formal", 50000),
    ]


def test_prompt_manifest_hash_is_deterministic_and_binds_every_field() -> None:
    manifest = PromptManifest(
        (
            PromptCase("p1", "1girl, red dress", ("artist:a",), 7, 512, 384),
            PromptCase("p2", "landscape", (), 8, 256, 512),
        )
    )
    repeated = PromptManifest(manifest.cases)
    changed = PromptManifest(
        (dataclasses.replace(manifest.cases[0], seed=9), manifest.cases[1])
    )

    assert manifest.canonical_bytes() == repeated.canonical_bytes()
    assert manifest.sha256 == repeated.sha256
    assert manifest.sha256 != changed.sha256


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"cfg_scale": 3.0}, "CFG"),
        ({"sampling_profile": "balanced"}, "reference profile"),
        ({"solver": "euler"}, "inconsistent"),
        ({"solver_nfe": 50}, "inconsistent"),
        ({"artifact_kind": "is_trend"}, "inconsistent"),
        ({"training_paused": 1}, "explicit"),
    ],
)
def test_job_rejects_nonformal_or_mixed_identity(
    changes: dict[str, object], expected: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        dataclasses.replace(_job(), **changes)
