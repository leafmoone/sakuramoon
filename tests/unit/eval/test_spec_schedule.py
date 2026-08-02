from __future__ import annotations

import dataclasses
from typing import cast

import pytest

from sakuramoon.config.schema import (
    EvaluationConfig,
    EvaluationEnabledConfig,
    EvaluationExtractorEnabledConfig,
    EvaluationSamplingConfig,
    FidEnabledConfig,
    IsEnabledConfig,
    ManualQualityEnabledConfig,
)
from sakuramoon.data.caption import CaptionDropoutHits, CaptionPlan, Tag
from sakuramoon.eval.schedule import scheduled_evaluations
from sakuramoon.eval.spec import (
    CheckpointRef,
    EvaluationJob,
    PromptCase,
    PromptManifest,
    caption_plan_prompt_text,
)


def _config() -> EvaluationConfig:
    return EvaluationEnabledConfig(
        enabled=True,
        stage_end=True,
        explicit_job=True,
        gpu_index=0,
        training_paused=True,
        sampling=EvaluationSamplingConfig(profile="reference"),
        extractor=EvaluationExtractorEnabledConfig(
            enabled=True,
            feature_extractor="locked-inception",
            feature_extractor_version="1.0",
            feature_extractor_path="extractor.pt",
            preprocess_path="preprocess.pt",
        ),
        fid=FidEnabledConfig(
            enabled=True,
            every_successful_updates=10,
            trend_samples=100,
            acceptance_samples=50000,
            real_stats_path="real-stats.safetensors",
        ),
        **{
            "is": IsEnabledConfig(
                enabled=True,
                every_successful_updates=20,
                trend_samples=200,
                acceptance_samples=50000,
                splits=10,
            )
        },
        manual_quality=ManualQualityEnabledConfig(enabled=True, samples=100),
        batch_size=10,
        output_reserve_gib=8,
    )


def _job() -> EvaluationJob:
    candidate = EvaluationJob(
        job_id="eval-content-address-pending",
        checkpoint=CheckpointRef(
            "checkpoint-1", "raw", "raw", "strict_jlt", "1" * 64, 10
        ),
        metric="fid",
        artifact_kind="fid_trend",
        validation_selection_path="validation-selection.json",
        validation_selection_id="7" * 64,
        validation_manifest_id="8" * 64,
        validation_shard_root="validation-shards",
        validation_seed=44,
        prompt_selection="validation_bucketed_prefix",
        prompt_manifest_sha256="2" * 64,
        trigger_successful_update=10,
        sample_count=100,
        batch_size=10,
        cfg_scale=2.9,
        sampling_profile="reference",
        solver="heun_final_euler",
        time_schedule="linear",
        solver_steps=50,
        solver_nfe=99,
        feature_extractor="inception",
        feature_extractor_version="1.0",
        feature_extractor_path="extractor.pt",
        feature_extractor_sha256="5" * 64,
        preprocess_path="preprocess.pt",
        preprocess_sha256="3" * 64,
        real_stats_path="real-stats.safetensors",
        real_stats_sha256="4" * 64,
        real_stats_metadata_path="real-stats.safetensors.metadata.json",
        real_stats_metadata_sha256="6" * 64,
        is_splits=None,
        gpu_index=0,
        training_paused=True,
    )
    return dataclasses.replace(candidate, job_id=candidate.content_addressed_id)


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
    with pytest.raises(TypeError, match="immutable PromptCase tuple"):
        PromptManifest(
            cast(tuple[PromptCase, ...], [manifest.cases[0]])
        )
    with pytest.raises(ValueError, match="complete tag boundaries"):
        PromptCase("p3", "portrait", ("artist one, artist two",), 1, 256, 256)


def test_structured_caption_plan_round_trips_without_dropout() -> None:
    no_dropout = CaptionDropoutHits(
        all_condition=False,
        nsfw=False,
        character=False,
        copyright=False,
        general=False,
        artist=False,
        candidate_source=False,
        long_names=False,
        long_no_names=False,
        short_vibes=False,
        nl2=False,
        nl3=False,
    )
    caption = CaptionPlan(
        nsfw=(Tag("safe", "safe"),),
        character=(Tag("alice", "alice"),),
        copyright=(),
        general=(Tag("blue_hair", "blue_hair"),),
        artists=(Tag("artist_name", "artist_name"),),
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=False,
        dropout_hits=no_dropout,
    )
    manifest = PromptManifest(
        (
            PromptCase(
                "structured-1",
                caption_plan_prompt_text(caption),
                (),
                44,
                256,
                256,
                caption,
            ),
        )
    )

    restored = PromptManifest.from_canonical_bytes(manifest.canonical_bytes())

    assert restored == manifest
    assert restored.cases[0].caption_plan == caption
    with pytest.raises(ValueError, match="structured evaluator"):
        dataclasses.replace(
            manifest.cases[0],
            caption_plan=dataclasses.replace(
                caption,
                dropout_hits=dataclasses.replace(no_dropout, general=True),
            ),
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"cfg_scale": 3.0}, "CFG"),
        ({"sampling_profile": "balanced"}, "reference profile"),
        ({"solver": "euler"}, "inconsistent"),
        ({"solver_nfe": 50}, "inconsistent"),
        ({"artifact_kind": "is_trend"}, "inconsistent"),
        ({"artifact_kind": "fid_unknown"}, "inconsistent"),
        ({"prompt_selection": "random"}, "bucketed validation prefix"),
        ({"training_paused": 1}, "explicit"),
    ],
)
def test_job_rejects_nonformal_or_mixed_identity(
    changes: dict[str, object], expected: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        dataclasses.replace(_job(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "checkpoint": CheckpointRef(
                "checkpoint-1", "accepted", "release", "strict_jlt", "1" * 64, 9
            )
        },
        {
            "checkpoint": CheckpointRef(
                "checkpoint-2", "raw", "raw", "strict_jlt", "1" * 64, 10
            )
        },
        {
            "checkpoint": CheckpointRef(
                "checkpoint-1", "raw", "raw", "strict_jlt", "6" * 64, 10
            )
        },
        {"validation_selection_path": "other-selection.json"},
        {"validation_selection_id": "9" * 64},
        {"prompt_manifest_sha256": "9" * 64},
        {"sample_count": 110},
        {"batch_size": 20},
        {"feature_extractor": "other-inception"},
        {"feature_extractor_version": "2.0"},
        {"preprocess_sha256": "8" * 64},
        {"real_stats_sha256": "7" * 64},
        {"gpu_index": 1},
        {"training_paused": False},
    ],
)
def test_job_content_identity_binds_every_checkpoint_metric_and_resource_field(
    changes: dict[str, object],
) -> None:
    baseline = _job()
    changed = dataclasses.replace(baseline, **changes)

    assert changed.identity_sha256 != baseline.identity_sha256


def test_job_content_identity_binds_trigger_independently_of_older_accepted() -> None:
    accepted = dataclasses.replace(
        _job(),
        checkpoint=CheckpointRef(
            "checkpoint-accepted",
            "accepted",
            "release",
            "strict_jlt",
            "1" * 64,
            9,
        ),
    )
    changed = dataclasses.replace(accepted, trigger_successful_update=11)

    assert changed.identity_sha256 != accepted.identity_sha256


def test_is_job_requires_exactly_divisible_splits() -> None:
    with pytest.raises(ValueError, match="exactly divisible"):
        dataclasses.replace(
            _job(),
            metric="is",
            artifact_kind="is_trend",
            sample_count=101,
            real_stats_path=None,
            real_stats_sha256=None,
            real_stats_metadata_path=None,
            real_stats_metadata_sha256=None,
            is_splits=10,
        )


def test_direct_job_construction_requires_complete_batches() -> None:
    with pytest.raises(ValueError, match="exactly divisible by batch size"):
        dataclasses.replace(_job(), batch_size=16)


def test_job_rejects_future_or_stale_raw_checkpoint_but_accepts_older_accepted() -> None:
    with pytest.raises(ValueError, match="future checkpoint"):
        dataclasses.replace(
            _job(),
            checkpoint=CheckpointRef(
                "checkpoint-future",
                "accepted",
                "release",
                "strict_jlt",
                "1" * 64,
                11,
            ),
        )
    with pytest.raises(ValueError, match="raw checkpoint"):
        dataclasses.replace(
            _job(),
            checkpoint=CheckpointRef(
                "checkpoint-stale", "raw", "raw", "strict_jlt", "1" * 64, 9
            ),
        )

    accepted = dataclasses.replace(
        _job(),
        checkpoint=CheckpointRef(
            "checkpoint-accepted",
            "accepted",
            "release",
            "strict_jlt",
            "1" * 64,
            9,
        ),
    )
    assert accepted.checkpoint.successful_update == 9
    older = dataclasses.replace(
        accepted,
        checkpoint=CheckpointRef(
            "checkpoint-accepted",
            "accepted",
            "release",
            "strict_jlt",
            "1" * 64,
            8,
        ),
    )
    assert older.identity_sha256 != accepted.identity_sha256


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_count": 1},
        {"feature_extractor": " "},
        {"feature_extractor_version": " version-with-padding "},
    ],
)
def test_job_rejects_impossible_sample_count_or_padded_extractor_identity(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_job(), **changes)
