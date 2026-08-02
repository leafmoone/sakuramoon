from __future__ import annotations

# pyright: reportPrivateUsage=false
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    RawCheckpointState,
    identity_to_dict,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import (
    EvaluationEnabledConfig,
    EvaluationExtractorDisabledConfig,
    EvaluationExtractorEnabledConfig,
    EvaluationSamplingConfig,
    FidDisabledConfig,
    FidEnabledConfig,
    IsDisabledConfig,
    ManualQualityDisabledConfig,
    ManualQualityEnabledConfig,
    RuntimeConfig,
)
from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.validation import (
    VALIDATION_SHARD_PATHS,
    ValidationPromptSample,
    ValidationSelection,
    select_validation_shards,
)
from sakuramoon.eval import generate as generate_module
from sakuramoon.eval import runner as runner_module
from sakuramoon.eval.extractor import RealStatsProvenance, VerifiedLocalFile
from sakuramoon.eval.generate import GeneratedBatch
from sakuramoon.eval.metrics import (
    FeatureStats,
    FeatureStatsAccumulator,
    InceptionScoreAccumulator,
)
from sakuramoon.eval.runner import (
    CheckpointSelection,
    EvaluationExtractor,
    EvaluationPlan,
    EvaluationPreflightError,
    ValidatedCheckpoint,
    preflight_evaluator,
    run_evaluator,
)
from sakuramoon.eval.spec import (
    ArtifactKind,
    CheckpointArtifactKind,
    CheckpointRef,
    CheckpointRole,
    EvaluationJob,
    MetricName,
    PromptCase,
    PromptManifest,
)
from sakuramoon.eval.timing import allowed_gpu_clock_overshoot_seconds
from sakuramoon.eval.validation import ValidationPromptPlan
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.sampling.sampler import GenerationMetadata
from sakuramoon.storage import StorageValidationError

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


def _dataset_manifest() -> DatasetManifest:
    paths = (*VALIDATION_SHARD_PATHS, "training-shard.tar")
    return DatasetManifest.from_shards(
        DatasetSourceIdentity(
            repo_id="leafmoone/webdataset_danbooru", revision="master"
        ),
        tuple(
            ShardRecord(
                path=path,
                bytes=index,
                upstream_sha256=f"{index:064x}",
            )
            for index, path in enumerate(paths, start=1)
        ),
    )


_SELECTION = select_validation_shards(_dataset_manifest())


def _checkpoint(role: CheckpointRole) -> CheckpointRef:
    artifact_kind = cast(
        CheckpointArtifactKind,
        {
            "raw": "raw",
            "model-only": "model-only",
            "pma": "pma",
            "accepted": "release",
        }[role],
    )
    return CheckpointRef(
        checkpoint_id=f"checkpoint-{role}",
        role=role,
        artifact_kind=artifact_kind,
        objective_provenance="strict_jlt",
        resolved_config_sha256=_HASH_A,
        successful_update=9 if role == "accepted" else 10,
    )


def _manifest(
    kind: CheckpointKind,
    checkpoint_id: str,
    update: int,
    *,
    config_sha256: str = _HASH_A,
) -> CheckpointManifest:
    return CheckpointManifest(
        kind=kind,
        identity=CheckpointIdentity(
            checkpoint_id,
            update,
            config_sha256,
            _HASH_B,
            _HASH_C,
        ),
        files=(FileRecord("model/model.safetensors", 1, _HASH_D),),
    )


def _job(
    checkpoint: CheckpointRef,
    metric: MetricName,
    artifact_kind: ArtifactKind,
    *,
    sample_count: int = 4,
) -> EvaluationJob:
    uses_extractor = metric in ("fid", "is")
    uses_real_stats = metric == "fid"
    candidate = EvaluationJob(
        job_id="eval-content-address-pending",
        checkpoint=checkpoint,
        metric=metric,
        artifact_kind=artifact_kind,
        validation_selection_path="evaluation/validation-selection.json",
        validation_selection_id=_SELECTION.selection_id,
        validation_manifest_id=_SELECTION.manifest_id,
        validation_shard_root="evaluation/validation-shards",
        validation_seed=44,
        prompt_selection="validation_bucketed_prefix",
        prompt_manifest_sha256=_PROMPTS.sha256,
        trigger_successful_update=10,
        sample_count=sample_count,
        batch_size=2,
        cfg_scale=2.9,
        sampling_profile="reference",
        solver="heun_final_euler",
        time_schedule="linear",
        solver_steps=50,
        solver_nfe=99,
        feature_extractor="test-extractor" if uses_extractor else None,
        feature_extractor_version="1" if uses_extractor else None,
        feature_extractor_path=(
            "evaluation/extractor.pt" if uses_extractor else None
        ),
        feature_extractor_sha256=_HASH_B if uses_extractor else None,
        preprocess_path="evaluation/preprocess.pt" if uses_extractor else None,
        preprocess_sha256=_HASH_C if uses_extractor else None,
        real_stats_path=(
            "evaluation/real-stats.safetensors" if uses_real_stats else None
        ),
        real_stats_sha256=_HASH_D if uses_real_stats else None,
        real_stats_metadata_path=(
            "evaluation/real-stats.safetensors.metadata.json"
            if uses_real_stats
            else None
        ),
        real_stats_metadata_sha256=_HASH_A if uses_real_stats else None,
        is_splits=2 if metric == "is" else None,
        gpu_index=0,
        training_paused=True,
    )
    return dataclasses.replace(candidate, job_id=candidate.content_addressed_id)


_PROMPTS = PromptManifest(
    tuple(
        PromptCase(f"prompt-{index}", f"description {index}", (), index, 16, 16)
        for index in range(4)
    )
)


def _plan(
    tmp_path: Path,
    *,
    roles: tuple[CheckpointRole, ...] = ("raw",),
    include_is: bool = True,
    include_manual: bool = True,
    plan_id: str = "evaluation-test",
    engineering_only: bool = True,
) -> EvaluationPlan:
    checkpoints: list[ValidatedCheckpoint] = []
    for role in roles:
        checkpoint = _checkpoint(role)
        jobs = [_job(checkpoint, "fid", "fid_formal")]
        if include_is:
            jobs.append(_job(checkpoint, "is", "is_formal"))
        if include_manual:
            jobs.append(_job(checkpoint, "manual_quality", "manual_quality"))
        checkpoints.append(
            ValidatedCheckpoint(
                selection=CheckpointSelection(
                    role=role,
                    path=(tmp_path / f"{role}-checkpoint").absolute(),
                    objective_provenance="strict_jlt",
                ),
                reference=checkpoint,
                jobs=tuple(jobs),
            )
        )
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            evaluation=SimpleNamespace(
                enabled=True,
                batch_size=2,
                gpu_index=0,
                training_paused=True,
            )
        ),
    )
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)
    real_stats = FeatureStats.from_features(
        torch.tensor(
            [[0.0, 0.0], [0.25, 0.0625], [0.5, 0.25], [0.75, 0.5625]],
            dtype=torch.float64,
        )
    )
    real_stats_provenance = RealStatsProvenance(
        selection_id=_SELECTION.selection_id,
        manifest_id=_SELECTION.manifest_id,
        prompt_manifest_sha256=_PROMPTS.sha256,
        preprocess_sha256=_HASH_C,
        feature_extractor="test-extractor",
        feature_extractor_version="1",
        feature_extractor_sha256=_HASH_B,
        real_stats_sha256=_HASH_D,
        sample_count=real_stats.count,
    )
    return EvaluationPlan(
        loaded=loaded,
        repository_root=tmp_path.absolute(),
        manifest_path=tmp_path / "dataset-manifest.json",
        selection_path=tmp_path / "validation-selection.json",
        validation_shard_root=tmp_path / "validation-shards",
        validation_selection=_SELECTION,
        prompts=_PROMPTS,
        batchable_cases=len(_PROMPTS.cases),
        checkpoints=tuple(checkpoints),
        extractor_file=VerifiedLocalFile(tmp_path / "extractor.pt", _HASH_B, 1),
        preprocess_file=VerifiedLocalFile(tmp_path / "preprocess.pt", _HASH_C, 1),
        real_stats_file=VerifiedLocalFile(
            tmp_path / "real-stats.safetensors", _HASH_D, 1
        ),
        real_stats_metadata_file=VerifiedLocalFile(
            tmp_path / "real-stats.safetensors.metadata.json", _HASH_A, 1
        ),
        real_stats_provenance=real_stats_provenance,
        real_stats=real_stats,
        output_root=(tmp_path / "artifacts").absolute(),
        trigger_successful_update=10,
        stage_end=False,
        engineering_only=engineering_only,
        plan_id=plan_id,
    )


class _FakeGenerator:
    def __init__(self, checkpoint: CheckpointRef) -> None:
        self.checkpoint = checkpoint

    def generate(self, cases: tuple[PromptCase, ...]) -> GeneratedBatch:
        images = torch.stack(
            tuple(
                torch.full(
                    (3, case.height, case.width),
                    (case.seed * 53 + 17) % 256,
                    dtype=torch.uint8,
                )
                for case in cases
            )
        )
        return GeneratedBatch(
            cases=cases,
            images=images,
            metadata=GenerationMetadata(
                checkpoint_id=self.checkpoint.checkpoint_id,
                checkpoint_kind=self.checkpoint.artifact_kind,
                objective_provenance=self.checkpoint.objective_provenance,
                profile="reference",
                cfg_scale=2.9,
            ),
        )


class _FakeExtractor:
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means = images.to(torch.float64).mean(dim=(1, 2, 3)).div(255.0)
        features = torch.stack((means, means.square()), dim=1)
        probabilities = torch.softmax(torch.stack((means, -means), dim=1), dim=1)
        return features, probabilities


def _generator_factory(
    _plan: EvaluationPlan, checkpoint: ValidatedCheckpoint
) -> _FakeGenerator:
    return _FakeGenerator(checkpoint.reference)


def _extractor_factory(_plan: EvaluationPlan) -> _FakeExtractor:
    return _FakeExtractor()


def test_runner_publishes_metrics_manual_index_and_complete_tree(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    result = run_evaluator(
        plan,
        generator_factory=_generator_factory,
        extractor_factory=_extractor_factory,
        measure_cuda=False,
    )

    assert result.artifact_count == 3
    assert result.checkpoint_count == 1
    assert (result.output_path / "COMPLETE").read_bytes() == b"complete\n"
    assert (
        result.output_path / "inputs" / "validation-prompts.json"
    ).read_bytes() == _PROMPTS.canonical_bytes()
    assert (
        result.output_path / "inputs" / "validation-selection.json"
    ).is_file()
    summary = json.loads((result.output_path / "summary.json").read_text())
    assert summary["automatic_release"] is False
    assert summary["artifact_count"] == 3
    assert summary["classification"] == "synthetic_bounded_engineering_only"
    assert summary["validation"]["selection_id"] == _SELECTION.selection_id
    assert summary["cost"]["gpu_seconds"] == 0.0
    assert summary["cost"]["publication_seconds_included"] is False
    assert summary["cost"]["training_pause_seconds"] == summary["cost"][
        "wall_seconds"
    ]
    assert summary["publication_timing"] == {
        "atomic_commit_seconds": None,
        "recorded_in_run_result_only": True,
    }
    assert result.publication_seconds >= 0.0
    assert result.total_wall_seconds >= result.publication_seconds
    artifacts = [
        json.loads(path.read_text())
        for path in sorted((result.output_path / "artifacts").glob("*.json"))
    ]
    assert {item["artifact_kind"] for item in artifacts} == {
        "fid_formal",
        "is_formal",
        "manual_quality_index",
    }
    assert {
        item["execution_classification"] for item in artifacts
    } == {"synthetic_bounded_engineering_only"}
    manual = next(
        item for item in artifacts if item["artifact_kind"] == "manual_quality_index"
    )
    assert [item["prompt_id"] for item in manual["images"]] == [
        case.prompt_id for case in _PROMPTS.cases
    ]
    for item in manual["images"]:
        payload = (result.output_path / item["relative_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    generation = json.loads(
        (result.output_path / "generation" / "raw.json").read_text()
    )
    assert generation["generation"]["nfe"] == 99
    assert generation["sample_count"] == 4
    assert not (plan.output_root / f".{plan.plan_id}.incomplete").exists()


def test_runner_preserves_raw_cuda_and_wall_measurements_in_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, plan_id="timing-preservation")
    checkpoint_wall_seconds = 1.0
    overall_wall_seconds = 3.0
    checkpoint_gpu_seconds = checkpoint_wall_seconds + (
        allowed_gpu_clock_overshoot_seconds(checkpoint_wall_seconds) / 2.0
    )
    overall_gpu_seconds = overall_wall_seconds + (
        allowed_gpu_clock_overshoot_seconds(overall_wall_seconds) / 2.0
    )
    counter_values = iter((100.0, 101.0, 102.0, 103.0, 104.0, 104.25, 104.5))

    def perf_counter() -> float:
        return next(counter_values)

    class FakeCudaEvent:
        def __init__(self, index: int) -> None:
            self.index = index

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, end: FakeCudaEvent) -> float:
            expected_end = {0: 1, 2: 3}[self.index]
            assert end.index == expected_end
            return (
                overall_gpu_seconds if self.index == 0 else checkpoint_gpu_seconds
            ) * 1000.0

    created_events: list[FakeCudaEvent] = []

    def event_factory(*, enable_timing: bool) -> FakeCudaEvent:
        assert enable_timing is True
        event = FakeCudaEvent(len(created_events))
        created_events.append(event)
        return event

    monkeypatch.setattr(runner_module.time, "perf_counter", perf_counter)
    monkeypatch.setattr(runner_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner_module.torch.cuda, "Event", event_factory)

    result = run_evaluator(
        plan,
        generator_factory=_generator_factory,
        extractor_factory=_extractor_factory,
        measure_cuda=True,
    )

    summary = json.loads((result.output_path / "summary.json").read_text())
    assert summary["cost"]["wall_seconds"] == overall_wall_seconds
    assert summary["cost"]["gpu_seconds"] == overall_gpu_seconds
    artifacts = [
        json.loads(path.read_text())
        for path in sorted((result.output_path / "artifacts").glob("*.json"))
    ]
    assert artifacts
    assert {
        (artifact["cost"]["wall_seconds"], artifact["cost"]["gpu_seconds"])
        for artifact in artifacts
    } == {(checkpoint_wall_seconds, checkpoint_gpu_seconds)}
    assert len(created_events) == 4


def test_runner_publishes_protocol_matched_three_checkpoint_comparison(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        roles=("raw", "pma", "accepted"),
        include_is=False,
        include_manual=False,
        plan_id="evaluation-comparison",
    )

    result = run_evaluator(
        plan,
        generator_factory=_generator_factory,
        extractor_factory=_extractor_factory,
        measure_cuda=False,
    )

    assert result.artifact_count == 4
    comparison = json.loads(
        (result.output_path / "comparisons" / "fid_formal.json").read_text()
    )
    assert [item["role"] for item in comparison["values"]] == [
        "raw",
        "pma",
        "accepted",
    ]
    assert comparison["automatic_release"] is False


def test_publisher_exists_before_extractor_or_generator_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    original_publisher = runner_module.AtomicEvaluationPublisher

    class RecordingPublisher(original_publisher):
        def __init__(self, output_root: Path, run_id: str) -> None:
            events.append("publisher")
            super().__init__(output_root, run_id)

    def extractor_factory(_plan: EvaluationPlan) -> _FakeExtractor:
        events.append("extractor")
        return _FakeExtractor()

    def generator_factory(
        _plan: EvaluationPlan, checkpoint: ValidatedCheckpoint
    ) -> _FakeGenerator:
        events.append("generator")
        return _FakeGenerator(checkpoint.reference)

    monkeypatch.setattr(
        runner_module, "AtomicEvaluationPublisher", RecordingPublisher
    )
    run_evaluator(
        _plan(tmp_path, include_is=False, include_manual=False),
        generator_factory=generator_factory,
        extractor_factory=extractor_factory,
        measure_cuda=False,
    )

    assert events[:3] == ["publisher", "extractor", "generator"]


def test_generator_is_released_before_metric_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    original_finalize = runner_module._finalize_metric

    class ReleasingGenerator(_FakeGenerator):
        def __del__(self) -> None:
            events.append("generator_released")

    def generator_factory(
        _plan: EvaluationPlan, checkpoint: ValidatedCheckpoint
    ) -> ReleasingGenerator:
        return ReleasingGenerator(checkpoint.reference)

    def finalize(
        job: EvaluationJob,
        *,
        fid_accumulator: FeatureStatsAccumulator | None,
        is_accumulator: InceptionScoreAccumulator | None,
        real_stats: FeatureStats,
    ) -> tuple[float, float | None]:
        events.append("metric_finalized")
        return original_finalize(
            job,
            fid_accumulator=fid_accumulator,
            is_accumulator=is_accumulator,
            real_stats=real_stats,
        )

    monkeypatch.setattr(runner_module, "_finalize_metric", finalize)
    run_evaluator(
        _plan(
            tmp_path,
            include_is=False,
            include_manual=False,
            plan_id="evaluation-release-order",
        ),
        generator_factory=generator_factory,
        extractor_factory=_extractor_factory,
        measure_cuda=False,
    )

    assert events == ["generator_released", "metric_finalized"]


def test_runner_rejects_extractor_batch_drift_before_publication(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        include_is=False,
        include_manual=False,
        plan_id="evaluation-bad-extractor",
    )

    class BadExtractor:
        def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            del images
            return torch.zeros((1, 2)), torch.full((1, 2), 0.5)

    def bad_extractor_factory(_plan: EvaluationPlan) -> EvaluationExtractor:
        return BadExtractor()

    with pytest.raises(RuntimeError, match="batch size"):
        run_evaluator(
            plan,
            generator_factory=_generator_factory,
            extractor_factory=bad_extractor_factory,
            measure_cuda=False,
        )

    assert not (plan.output_root / plan.plan_id).exists()
    assert (plan.output_root / f".{plan.plan_id}.incomplete").is_dir()


def test_preflight_reports_each_missing_external_identity_as_a_blocker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_storage(_config: RuntimeConfig, _root: Path) -> None:
        return None

    monkeypatch.setattr(runner_module, "require_evaluation_storage", fake_storage)
    config = _preflight_config(_PROMPTS)
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(),
            trigger_successful_update=10,
            stage_end=True,
        )

    blockers = {item.code: item.subject for item in captured.value.blockers}
    assert blockers["CHECKPOINT_REQUIRED"] == "--checkpoint"
    assert blockers["DATASET_MANIFEST_INVALID"].endswith(
        "evaluation/dataset-manifest.json"
    )
    assert blockers["FEATURE_EXTRACTOR_IDENTITY_INVALID"].endswith(
        "evaluation/extractor.pt"
    )
    assert blockers["PREPROCESS_IDENTITY_INVALID"].endswith(
        "evaluation/preprocess.pt"
    )
    assert blockers["REAL_STATS_IDENTITY_INVALID"].endswith(
        "evaluation/real-stats.safetensors"
    )


def test_preflight_rejects_noncanonical_or_symlinked_repository_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    nested = repository / "nested"
    nested.mkdir(parents=True)
    linked = tmp_path / "repository-link"
    linked.symlink_to(repository, target_is_directory=True)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)

    for invalid_root in (nested / "..", linked):
        with pytest.raises(EvaluationPreflightError) as captured:
            preflight_evaluator(
                loaded,
                repository_root=invalid_root,
                selections=(),
                trigger_successful_update=10,
                stage_end=False,
            )

        assert [item.code for item in captured.value.blockers] == [
            "REPOSITORY_ROOT_INVALID"
        ]


def test_checkpoint_selection_and_preflight_reject_noncanonical_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="canonical absolute path"):
        CheckpointSelection(
            "raw", nested / ".." / checkpoint.name, "strict_jlt"
        )

    linked = tmp_path / "checkpoint-link"
    linked.symlink_to(checkpoint, target_is_directory=True)
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)
    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(CheckpointSelection("raw", linked, "strict_jlt"),),
            trigger_successful_update=10,
            stage_end=False,
        )

    assert "CHECKPOINT_PATH_CONTAINS_SYMLINK" in {
        item.code for item in captured.value.blockers
    }


def test_preflight_rejects_noncanonical_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    config = _preflight_config(_PROMPTS)
    config.paths.artifact_dir = "../artifacts"
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "checkpoint").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )

    assert "ARTIFACT_ROOT_INVALID" in {
        item.code for item in captured.value.blockers
    }


def test_injected_execution_requires_permanent_engineering_classification(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        include_is=False,
        include_manual=False,
        engineering_only=False,
    )

    with pytest.raises(RuntimeError, match="engineering-only"):
        run_evaluator(
            plan,
            generator_factory=_generator_factory,
        )
    with pytest.raises(RuntimeError, match="engineering-only"):
        run_evaluator(plan, extractor_factory=_extractor_factory)
    with pytest.raises(RuntimeError, match="engineering-only"):
        run_evaluator(plan, measure_cuda=False)

    assert not plan.output_root.exists()


def test_engineering_execution_allows_default_production_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        include_is=False,
        include_manual=False,
        plan_id="evaluation-production-generator-engineering",
    )
    observed: list[str] = []

    def production_generator(
        _plan: EvaluationPlan, checkpoint: ValidatedCheckpoint
    ) -> _FakeGenerator:
        observed.append(checkpoint.reference.checkpoint_id)
        return _FakeGenerator(checkpoint.reference)

    monkeypatch.setattr(runner_module, "_production_generator", production_generator)

    result = run_evaluator(
        plan,
        extractor_factory=_extractor_factory,
        measure_cuda=False,
    )

    assert observed == ["checkpoint-raw"]
    assert result.classification == "synthetic_bounded_engineering_only"
    summary = json.loads((result.output_path / "summary.json").read_text())
    assert summary["classification"] == result.classification


def _preflight_config(prompts: PromptManifest) -> RuntimeConfig:
    del prompts
    evaluation = EvaluationEnabledConfig(
        enabled=True,
        explicit_job=True,
        stage_end=True,
        gpu_index=0,
        training_paused=True,
        batch_size=2,
        output_reserve_gib=1,
        sampling=EvaluationSamplingConfig(profile="reference"),
        extractor=EvaluationExtractorEnabledConfig(
            enabled=True,
            feature_extractor="test-extractor",
            feature_extractor_version="1",
            feature_extractor_path="evaluation/extractor.pt",
            preprocess_path="evaluation/preprocess.pt",
        ),
        fid=FidEnabledConfig(
            enabled=True,
            every_successful_updates=10,
            trend_samples=4,
            acceptance_samples=4,
            real_stats_path="evaluation/real-stats.safetensors",
        ),
        **{"is": IsDisabledConfig(enabled=False)},
        manual_quality=ManualQualityDisabledConfig(enabled=False),
    )
    return cast(
        RuntimeConfig,
        SimpleNamespace(
            run=SimpleNamespace(intent="eval", seed=44),
            distributed=SimpleNamespace(world_size=1, backend="native"),
            paths=SimpleNamespace(artifact_dir="artifacts/eval"),
            data=SimpleNamespace(
                source=SimpleNamespace(
                    repo_id="leafmoone/webdataset_danbooru", revision="master"
                ),
                manifest=SimpleNamespace(path="evaluation/dataset-manifest.json"),
                validation=SimpleNamespace(
                    selection_path="evaluation/validation-selection.json",
                    shard_root="evaluation/validation-shards",
                ),
            ),
            stage=SimpleNamespace(
                name="S0",
                world_size=1,
                depth=16,
                resolution=256,
            ),
            growth=SimpleNamespace(enabled=False),
            cfg=SimpleNamespace(scale=2.9),
            evaluation=evaluation,
        ),
    )


def _manual_only_preflight_config(prompts: PromptManifest) -> RuntimeConfig:
    config = _preflight_config(prompts)
    config.evaluation = EvaluationEnabledConfig(
        enabled=True,
        explicit_job=True,
        stage_end=True,
        gpu_index=0,
        training_paused=True,
        batch_size=2,
        output_reserve_gib=1,
        sampling=EvaluationSamplingConfig(profile="reference"),
        extractor=EvaluationExtractorDisabledConfig(enabled=False),
        fid=FidDisabledConfig(enabled=False),
        **{"is": IsDisabledConfig(enabled=False)},
        manual_quality=ManualQualityEnabledConfig(enabled=True, samples=2),
    )
    return config


def _patch_preflight_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    prompts: PromptManifest,
    *,
    update: int,
) -> None:
    manifest = CheckpointManifest(
        kind=CheckpointKind.RAW,
        identity=CheckpointIdentity(
            "checkpoint-raw", update, _HASH_A, _HASH_B, _HASH_C
        ),
        files=(FileRecord("model/model.safetensors", 1, _HASH_D),),
    )
    real_stats = FeatureStats.from_features(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    )

    def fake_verify(path: Path) -> VerifiedLocalFile:
        hashes = {
            "extractor.pt": _HASH_B,
            "preprocess.pt": _HASH_C,
            "real-stats.safetensors": _HASH_D,
            "real-stats.safetensors.metadata.json": _HASH_A,
        }
        return VerifiedLocalFile(path, hashes[path.name], 1)

    def fake_stats(_identity: VerifiedLocalFile) -> FeatureStats:
        return real_stats

    def fake_manifest(_path: Path) -> CheckpointManifest:
        return manifest

    def fake_dataset_manifest(*_args: object) -> DatasetManifest:
        return _dataset_manifest()

    def fake_selection(*_args: object) -> ValidationSelection:
        return _SELECTION

    def fake_samples(*_args: object, **_kwargs: object) -> tuple[ValidationPromptSample, ...]:
        return ()

    def fake_prompt_plan(*_args: object) -> ValidationPromptPlan:
        return ValidationPromptPlan(
            selection=_SELECTION,
            prompts=prompts,
            batchable_cases=len(prompts.cases),
            bucket_shapes=(BucketShape(prompts.cases[0].height, prompts.cases[0].width),),
        )

    def fake_real_stats_provenance(*_args: object, **_kwargs: object) -> RealStatsProvenance:
        return RealStatsProvenance(
            selection_id=_SELECTION.selection_id,
            manifest_id=_SELECTION.manifest_id,
            prompt_manifest_sha256=prompts.sha256,
            preprocess_sha256=_HASH_C,
            feature_extractor="test-extractor",
            feature_extractor_version="1",
            feature_extractor_sha256=_HASH_B,
            real_stats_sha256=_HASH_D,
            sample_count=real_stats.count,
        )

    def fake_models(_root: Path) -> None:
        return None

    def fake_storage(*_args: object) -> None:
        return None

    def fake_raw(
        _path: Path, expected: CheckpointManifest
    ) -> runner_module._RawProvenance:
        return runner_module._RawProvenance(
            expected,
            "S0",
            1,
            256,
            active_slot_ids(16),
            1.0,
        )

    monkeypatch.setattr(runner_module, "load_dataset_manifest", fake_dataset_manifest)
    monkeypatch.setattr(runner_module, "load_validation_selection", fake_selection)
    monkeypatch.setattr(runner_module, "load_validation_prompt_samples", fake_samples)
    monkeypatch.setattr(runner_module, "build_validation_prompt_plan", fake_prompt_plan)
    monkeypatch.setattr(runner_module, "verify_local_file", fake_verify)
    monkeypatch.setattr(runner_module, "load_real_feature_stats", fake_stats)
    monkeypatch.setattr(
        runner_module, "load_real_stats_provenance", fake_real_stats_provenance
    )
    monkeypatch.setattr(runner_module, "read_checkpoint_manifest", fake_manifest)
    monkeypatch.setattr(runner_module, "require_local_models", fake_models)
    monkeypatch.setattr(runner_module, "require_evaluation_storage", fake_storage)
    monkeypatch.setattr(runner_module, "_validate_raw_strict_jlt", fake_raw)


def test_raw_strict_jlt_provenance_comes_from_resolved_checkpoint_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(CheckpointKind.RAW, "raw-10", 10)
    growth = SimpleNamespace(
        stage="S0",
        world_size=1,
        resolution=256,
        active_slot_ids=active_slot_ids(16),
        alpha=1.0,
    )
    state = cast(RawCheckpointState, SimpleNamespace(growth=growth))

    def fake_raw_state(_path: Path) -> tuple[CheckpointManifest, RawCheckpointState]:
        return manifest, state

    monkeypatch.setattr(runner_module, "read_raw_checkpoint_state", fake_raw_state)
    checkpoint = tmp_path / "raw"
    checkpoint.mkdir()
    checkpoint.joinpath("resolved_config.toml").write_text(
        """[objective]
prediction_type = "x"
loss = "jlt_x_prediction_velocity_mse"
target_velocity = "x_to_v(clean,state,t,t_eps)"
endpoint_weighting = "inverse_square_clamped"
interpolation = "z_t=t*x+(1-t)*epsilon"
velocity_loss_dtype = "float32"
reduction = "per_sample_then_global_sample_mean"
""",
        encoding="utf-8",
    )

    evidence = runner_module._validate_raw_strict_jlt(checkpoint, manifest)
    assert evidence.manifest == manifest
    assert evidence.alpha == 1.0

    checkpoint.joinpath("resolved_config.toml").write_text(
        "[objective]\nprediction_type = \"x\"\nloss = \"other\"\n",
        encoding="utf-8",
    )
    with pytest.raises(CheckpointError, match="strict JLT"):
        runner_module._validate_raw_strict_jlt(checkpoint, manifest)


def test_pma_source_window_is_bound_to_raw_trigger_and_topology(
    tmp_path: Path,
) -> None:
    raw_manifest = _manifest(CheckpointKind.RAW, "raw-10", 10)
    pma_manifest = _manifest(CheckpointKind.PMA, "pma-10", 10)
    raw_anchor = runner_module._RawProvenance(
        raw_manifest,
        "S0",
        1,
        256,
        active_slot_ids(16),
        1.0,
    )
    sources = tuple(
        CheckpointIdentity(
            f"raw-{update}", update, _HASH_A, _HASH_B, _HASH_C
        )
        for update in range(1, 11)
    )
    checkpoint = tmp_path / "pma"
    checkpoint.mkdir()
    document: dict[str, object] = {
        "active_slot_ids": list(active_slot_ids(16)),
        "resolution": 256,
        "schema_version": 1,
        "sources": [identity_to_dict(item) for item in sources],
        "stage": "S0",
        "window": 10,
        "world_size": 1,
    }
    checkpoint.joinpath("pma_sources.json").write_text(
        json.dumps(document), encoding="utf-8"
    )

    runner_module._validate_pma_strict_jlt(
        checkpoint, pma_manifest, raw_anchor
    )

    older_manifest = _manifest(CheckpointKind.PMA, "pma-9", 9)
    older_sources = tuple(
        CheckpointIdentity(
            f"raw-{update}", update, _HASH_A, _HASH_B, _HASH_C
        )
        for update in range(10)
    )
    document["sources"] = [identity_to_dict(item) for item in older_sources]
    checkpoint.joinpath("pma_sources.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    runner_module._validate_pma_strict_jlt(
        checkpoint,
        older_manifest,
        raw_anchor,
        exact_raw_anchor=False,
    )
    with pytest.raises(CheckpointError, match="raw anchor"):
        runner_module._validate_pma_strict_jlt(
            checkpoint, older_manifest, raw_anchor
        )

    document["sources"] = [identity_to_dict(item) for item in sources]
    document["world_size"] = 2
    checkpoint.joinpath("pma_sources.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="topology"):
        runner_module._validate_pma_strict_jlt(
            checkpoint, pma_manifest, raw_anchor
        )

    document["world_size"] = 1
    source_documents = document["sources"]
    source_documents[-1] = identity_to_dict(
        CheckpointIdentity("other-10", 10, _HASH_A, _HASH_B, _HASH_C)
    )
    checkpoint.joinpath("pma_sources.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="raw anchor"):
        runner_module._validate_pma_strict_jlt(
            checkpoint, pma_manifest, raw_anchor
        )


def test_release_strict_jlt_requires_the_verified_pma_source(tmp_path: Path) -> None:
    pma_manifest = _manifest(CheckpointKind.PMA, "pma-10", 10)
    release_manifest = _manifest(CheckpointKind.RELEASE, "release-10", 10)
    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    document = {
        "automatic_release": False,
        "schema_version": 1,
        "source": identity_to_dict(pma_manifest.identity),
    }
    checkpoint.joinpath("release_source.json").write_text(
        json.dumps(document), encoding="utf-8"
    )

    runner_module._validate_release_strict_jlt(
        checkpoint, release_manifest, pma_manifest
    )

    document["source"] = identity_to_dict(
        CheckpointIdentity("other-pma", 10, _HASH_A, _HASH_B, _HASH_C)
    )
    checkpoint.joinpath("release_source.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="verified PMA"):
        runner_module._validate_release_strict_jlt(
            checkpoint, release_manifest, pma_manifest
        )


def test_stage_end_requires_exact_checkpoint_set_and_never_creates_publisher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)
    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "raw").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=True,
        )
    codes = {item.code for item in captured.value.blockers}
    assert "STAGE_END_CHECKPOINT_SET_INVALID" in codes

    forged = dataclasses.replace(
        _plan(tmp_path, engineering_only=False), stage_end=True
    )
    with pytest.raises(EvaluationPreflightError):
        run_evaluator(forged)
    assert not forged.output_root.exists()


def test_engineering_stage_end_binds_single_raw_manual_quality_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    loaded = LoadedConfig(
        _manual_only_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A
    )
    plan = preflight_evaluator(
        loaded,
        repository_root=tmp_path.absolute(),
        selections=(
            CheckpointSelection(
                "raw", (tmp_path / "raw").absolute(), "strict_jlt"
            ),
        ),
        trigger_successful_update=10,
        stage_end=True,
        engineering_only=True,
    )

    assert plan.engineering_only is True
    assert plan.stage_end is True
    assert [checkpoint.reference.role for checkpoint in plan.checkpoints] == ["raw"]
    assert [job.metric for job in plan.jobs] == ["manual_quality"]
    assert plan.plan_id != runner_module._plan_id(
        loaded,
        plan.checkpoints,
        trigger_successful_update=10,
        stage_end=True,
        engineering_only=False,
    )

    result = run_evaluator(
        plan,
        generator_factory=_generator_factory,
        measure_cuda=False,
    )
    summary = json.loads((result.output_path / "summary.json").read_text())
    assert result.classification == "synthetic_bounded_engineering_only"
    assert summary["classification"] == result.classification
    assert summary["engineering_only"] is True


def test_engineering_stage_end_rejects_formal_metric_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "raw").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=True,
            engineering_only=True,
        )

    assert "ENGINEERING_STAGE_END_METRIC_SET_INVALID" in {
        blocker.code for blocker in captured.value.blockers
    }


def test_complete_stage_end_chain_accepts_validation_nl_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    manifests = {
        "raw": _manifest(CheckpointKind.RAW, "raw-10", 10),
        "pma": _manifest(CheckpointKind.PMA, "pma-10", 10),
        "accepted": _manifest(CheckpointKind.RELEASE, "accepted-9", 9),
        "accepted-source-pma": _manifest(
            CheckpointKind.PMA, "accepted-source-pma-9", 9
        ),
    }

    def fake_manifest(path: Path) -> CheckpointManifest:
        return manifests[path.name]

    def fake_raw(
        _path: Path, expected: CheckpointManifest
    ) -> runner_module._RawProvenance:
        return runner_module._RawProvenance(
            expected, "S0", 1, 256, active_slot_ids(16), 1.0
        )

    monkeypatch.setattr(runner_module, "read_checkpoint_manifest", fake_manifest)
    monkeypatch.setattr(runner_module, "_validate_raw_strict_jlt", fake_raw)
    observed_pma: list[tuple[Path, bool]] = []

    def fake_pma(
        _path: Path,
        _manifest: CheckpointManifest,
        _raw_anchor: runner_module._RawProvenance,
        *,
        exact_raw_anchor: bool = True,
    ) -> None:
        observed_pma.append((_path, exact_raw_anchor))

    monkeypatch.setattr(runner_module, "_validate_pma_strict_jlt", fake_pma)

    def fake_release(
        _path: Path,
        _manifest: CheckpointManifest,
        source_manifest: CheckpointManifest,
    ) -> None:
        assert source_manifest == manifests["accepted-source-pma"]

    monkeypatch.setattr(runner_module, "_validate_release_strict_jlt", fake_release)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)
    selections = (
        CheckpointSelection("raw", (tmp_path / "raw").absolute(), "strict_jlt"),
        CheckpointSelection("pma", (tmp_path / "pma").absolute(), "strict_jlt"),
        CheckpointSelection(
            "accepted",
            (tmp_path / "accepted").absolute(),
            "strict_jlt",
            (tmp_path / "accepted-source-pma").absolute(),
        ),
    )

    plan = preflight_evaluator(
        loaded,
        repository_root=tmp_path.absolute(),
        selections=selections,
        trigger_successful_update=10,
        stage_end=True,
    )

    assert plan.stage_end is True
    assert all(not case.conditions for case in plan.prompts.cases)
    assert observed_pma == [
        ((tmp_path / "pma").absolute(), True),
        ((tmp_path / "accepted-source-pma").absolute(), False),
    ]


def test_unverifiable_model_only_and_release_provenance_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)

    model_manifest = _manifest(CheckpointKind.MODEL_ONLY, "model-10", 10)

    def read_model_manifest(_path: Path) -> CheckpointManifest:
        return model_manifest

    monkeypatch.setattr(
        runner_module, "read_checkpoint_manifest", read_model_manifest
    )
    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "model-only", (tmp_path / "model").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )
    assert "CHECKPOINT_OBJECTIVE_PROVENANCE_UNVERIFIED" in {
        item.code for item in captured.value.blockers
    }

    pre_fix = preflight_evaluator(
        loaded,
        repository_root=tmp_path.absolute(),
        selections=(
            CheckpointSelection(
                "model-only", (tmp_path / "model").absolute(), "pre_fix"
            ),
        ),
        trigger_successful_update=10,
        stage_end=False,
    )
    assert pre_fix.checkpoints[0].reference.objective_provenance == "pre_fix"

    release_manifest = _manifest(CheckpointKind.RELEASE, "release-10", 10)

    def read_release_manifest(_path: Path) -> CheckpointManifest:
        return release_manifest

    monkeypatch.setattr(
        runner_module, "read_checkpoint_manifest", read_release_manifest
    )
    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "accepted", (tmp_path / "release").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )
    assert "ACCEPTED_RELEASE_SOURCE_PMA_REQUIRED" in {
        item.code for item in captured.value.blockers
    }


def test_preflight_calls_evaluation_storage_and_surfaces_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)
    observed: list[Path] = []

    def storage(_config: RuntimeConfig, root: Path) -> None:
        observed.append(root)

    monkeypatch.setattr(runner_module, "require_evaluation_storage", storage)
    preflight_evaluator(
        loaded,
        repository_root=tmp_path.absolute(),
        selections=(
            CheckpointSelection(
                "raw", (tmp_path / "raw").absolute(), "strict_jlt"
            ),
        ),
        trigger_successful_update=10,
        stage_end=False,
    )
    assert observed == [tmp_path.absolute()]

    def failed_storage(_config: RuntimeConfig, _root: Path) -> None:
        raise StorageValidationError("blocked")

    monkeypatch.setattr(
        runner_module, "require_evaluation_storage", failed_storage
    )
    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "raw").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )
    assert "EVALUATION_STORAGE_INVALID" in {
        item.code for item in captured.value.blockers
    }


def test_initial_gaussian_noise_is_repeatable_fp32() -> None:
    cases = _PROMPTS.cases[:2]
    first = generate_module._initial_gaussian_noise(
        cases, device=torch.device("cpu")
    )
    repeated = generate_module._initial_gaussian_noise(
        cases, device=torch.device("cpu")
    )

    assert first.dtype == torch.float32
    assert first.shape == (2, 128, 1, 1)
    torch.testing.assert_close(first, repeated)


def test_preflight_fails_when_no_trend_job_is_due(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=9)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "checkpoint").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=9,
            stage_end=False,
        )

    assert [item.code for item in captured.value.blockers] == [
        "NO_EVALUATION_JOB_DUE"
    ]


def test_raw_checkpoint_must_equal_the_trigger_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=9)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "checkpoint").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )

    assert "RAW_TRIGGER_MISMATCH" in {
        item.code for item in captured.value.blockers
    }


def test_raw_checkpoint_source_stage_must_match_s0_evaluation_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_preflight_dependencies(monkeypatch, _PROMPTS, update=10)

    def s1_raw(
        _path: Path, expected: CheckpointManifest
    ) -> runner_module._RawProvenance:
        return runner_module._RawProvenance(
            expected,
            "S1",
            4,
            256,
            active_slot_ids(16),
            1.0,
        )

    monkeypatch.setattr(runner_module, "_validate_raw_strict_jlt", s1_raw)
    loaded = LoadedConfig(_preflight_config(_PROMPTS), (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "checkpoint").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )

    assert [item.as_mapping() for item in captured.value.blockers] == [
        {
            "code": "CHECKPOINT_TARGET_TOPOLOGY_MISMATCH",
            "subject": "raw",
        }
    ]


def test_preflight_rejects_mixed_shape_batch_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = PromptManifest(
        (
            PromptCase("p0", "zero", (), 0, 16, 16),
            PromptCase("p1", "one", (), 1, 16, 32),
            PromptCase("p2", "two", (), 2, 16, 16),
            PromptCase("p3", "three", (), 3, 16, 16),
        )
    )
    _patch_preflight_dependencies(monkeypatch, prompts, update=10)
    loaded = LoadedConfig(_preflight_config(prompts), (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "checkpoint").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )

    assert [item.as_mapping() for item in captured.value.blockers] == [
        {
            "code": "PROMPT_BATCH_SHAPE_INVALID",
            "subject": "validation_prefix:0",
        }
    ]


def test_preflight_blocks_ungoverned_nonempty_prompt_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = PromptManifest(
        (
            PromptCase("p0", "zero", ("artist:a",), 0, 16, 16),
            *_PROMPTS.cases[1:],
        )
    )
    _patch_preflight_dependencies(monkeypatch, prompts, update=10)
    loaded = LoadedConfig(_preflight_config(prompts), (), "resolved\n", _HASH_A)

    with pytest.raises(EvaluationPreflightError) as captured:
        preflight_evaluator(
            loaded,
            repository_root=tmp_path.absolute(),
            selections=(
                CheckpointSelection(
                    "raw", (tmp_path / "checkpoint").absolute(), "strict_jlt"
                ),
            ),
            trigger_successful_update=10,
            stage_end=False,
        )

    assert "VALIDATION_PROMPT_CONDITION_INVALID" in {
        item.code for item in captured.value.blockers
    }
