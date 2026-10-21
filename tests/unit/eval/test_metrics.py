# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Never, cast

import pytest
import torch

from sakuramoon.config.schema import EvaluationEnabledConfig, RuntimeConfig
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX
from sakuramoon.distributed import DistributedProgress
from sakuramoon.eval import features as feature_module
from sakuramoon.eval import runtime
from sakuramoon.eval.features import (
    CLIP_EMBEDDING_DIM,
    INCEPTION_FEATURE_DIM,
    INCEPTION_LOGIT_DIM,
    EvaluationFeatureError,
    ImageFeatureBatch,
    ValidationImageBatch,
    require_local_inception_weights,
)
from sakuramoon.eval.metrics import (
    FeatureStats,
    clip_maximum_mean_discrepancy,
    frechet_inception_distance,
    inception_score,
    kernel_inception_distance,
)
from sakuramoon.eval.runtime import (
    EvaluationError,
    EvaluationResult,
    TrainingEvaluator,
    _conditioning_inputs,
    _stage_cases,
)
from sakuramoon.eval.spec import PromptCase, PromptManifest


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text == SYSTEM_PREFIX:
            return [1] * 34
        if text == MAIN_SUFFIX:
            return [2] * 5
        return [3] * len(text.split())


def test_fid_is_zero_for_identical_feature_statistics() -> None:
    features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]],
        dtype=torch.float32,
    )
    stats = FeatureStats.from_features(features)

    assert frechet_inception_distance(stats, stats) == pytest.approx(0.0, abs=1e-10)


def test_sample_space_fid_matches_covariance_space_fid() -> None:
    generator = torch.Generator().manual_seed(7)
    generated = FeatureStats.from_features(torch.randn(6, 8, generator=generator))
    real = FeatureStats.from_features(torch.randn(6, 8, generator=generator))
    covariance_only = FeatureStats(
        generated.count,
        generated.mean,
        generated.covariance,
    )

    sample_space = frechet_inception_distance(generated, real)
    covariance_space = frechet_inception_distance(covariance_only, real)

    assert sample_space == pytest.approx(covariance_space, rel=1e-7, abs=1e-7)


def test_large_sample_fid_discards_sample_space_matrix() -> None:
    generator = torch.Generator().manual_seed(8)
    generated = FeatureStats.from_features(torch.randn(16, 4, generator=generator))
    real = FeatureStats.from_features(torch.randn(16, 4, generator=generator))

    assert generated.centered_features is None
    assert real.centered_features is None
    assert frechet_inception_distance(generated, real) >= 0.0


def test_inception_score_and_default_schedule() -> None:
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
        dtype=torch.float32,
    )
    score = inception_score(probabilities, splits=2)
    config = EvaluationEnabledConfig.model_validate(
        {
            "enabled": True,
            "sample_count": 1000,
            "batch_size": 4,
            "is_splits": 10,
            "prompt_path": "data/prompts.json",
            "validation_shard_root": "data/validation",
            "output_dir": "output_model/evaluation",
            "sampling_profile": "preview",
        },
        strict=True,
    )

    assert score.mean > 1.0
    assert score.sample_count == 4
    assert config.every_updates == 1000
    assert config.kid_subsets == 100
    assert config.kid_subset_size == 100
    assert config.resolved_real_sample_count == 1000


def test_evaluation_accepts_fewer_real_than_generated_samples() -> None:
    config = EvaluationEnabledConfig.model_validate(
        {
            "enabled": True,
            "sample_count": 50_000,
            "real_sample_count": 34_849,
            "batch_size": 64,
            "is_splits": 10,
            "kid_subsets": 100,
            "kid_subset_size": 1000,
            "prompt_path": "data/prompts-50k.json",
            "validation_shard_root": "data/validation",
            "output_dir": "output_model/evaluation-50k",
            "sampling_profile": "preview",
        },
        strict=True,
    )

    assert config.resolved_real_sample_count == 34_849


def test_kid_is_deterministic_and_rejects_oversized_subsets() -> None:
    generator = torch.Generator().manual_seed(41)
    generated = torch.randn(12, 8, generator=generator)
    real = torch.randn(12, 8, generator=generator)

    first = kernel_inception_distance(
        generated,
        real,
        subsets=5,
        subset_size=6,
        seed=99,
    )
    second = kernel_inception_distance(
        generated,
        real,
        subsets=5,
        subset_size=6,
        seed=99,
    )

    assert first == second
    assert first.subsets == 5
    assert first.subset_size == 6
    assert first.std >= 0.0
    with pytest.raises(ValueError, match="outside the available"):
        kernel_inception_distance(
            generated,
            real,
            subsets=1,
            subset_size=13,
            seed=99,
        )


def test_cmmd_matches_the_official_biased_rbf_estimator() -> None:
    generated = torch.nn.functional.normalize(
        torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    real = torch.nn.functional.normalize(
        torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 2.0, 3.0], [3.0, 1.0, 2.0]],
            dtype=torch.float32,
        ),
        dim=1,
    )
    gamma = 1.0 / 200.0

    def kernel(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(first.double(), second.double()).square()
        return torch.exp(-gamma * distances)

    expected = 1000.0 * (
        kernel(generated, generated).mean()
        + kernel(real, real).mean()
        - 2.0 * kernel(generated, real).mean()
    )

    assert clip_maximum_mean_discrepancy(
        generated, real, block_size=2
    ) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)
    assert clip_maximum_mean_discrepancy(generated, generated) == pytest.approx(
        0.0, abs=1e-12
    )
    with pytest.raises(ValueError, match="L2-normalized"):
        clip_maximum_mean_discrepancy(generated * 2.0, real)


def test_evaluation_rejects_a_prompt_whose_content_is_fully_dropped() -> None:
    case = PromptCase(
        prompt_id="long-prompt",
        prompt=" ".join(["word"] * 600),
        conditions=(),
        seed=1,
        height=256,
        width=256,
    )

    with pytest.raises(
        EvaluationError, match="long-prompt.*dropped all prompt content"
    ):
        _conditioning_inputs((case,), _Tokenizer(), torch.device("cpu"))


def test_evaluation_stages_every_prompt_as_one_to_one(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_bytes(
        PromptManifest(
            (
                PromptCase("wide", "wide prompt", (), 1, 256, 512),
                PromptCase("tall", "tall prompt", (), 2, 768, 384),
            )
        ).canonical_bytes()
    )

    cases = _stage_cases(path, 2, resolution=256)

    assert [(case.height, case.width) for case in cases] == [(256, 256)] * 2
    assert [(case.prompt_id, case.seed) for case in cases] == [
        ("wide", 1),
        ("tall", 2),
    ]


def _fake_features(images: torch.Tensor) -> ImageFeatureBatch:
    rows = images[:, 0, 0, 0].float()
    return ImageFeatureBatch(
        rows[:, None].expand(-1, INCEPTION_FEATURE_DIM).contiguous(),
        torch.nn.functional.one_hot(rows.long(), CLIP_EMBEDDING_DIM).float(),
        rows[:, None].expand(-1, INCEPTION_LOGIT_DIM).contiguous(),
    )


def _cpu_evaluator(
    extract: Callable[[torch.Tensor], ImageFeatureBatch] = _fake_features,
) -> TrainingEvaluator:
    evaluator = TrainingEvaluator.__new__(TrainingEvaluator)
    evaluator.config = cast(
        RuntimeConfig, SimpleNamespace(train=SimpleNamespace(resolution=16))
    )
    evaluator.evaluation = cast(
        EvaluationEnabledConfig,
        SimpleNamespace(
            sample_count=5,
            resolved_real_sample_count=3,
            batch_size=2,
            every_updates=10,
            prompt_path="prompts.json",
            validation_shard_root="validation",
        ),
    )
    evaluator.__dict__.update(
        root=Path("unused"),
        device=torch.device("cpu"),
        growth_alpha=1.0,
        composite=torch.nn.Identity(),
        qwen=None,
        vae=None,
        rank=0,
        world_size=1,
        is_main_process=True,
        _real_features=None,
        _feature_models=SimpleNamespace(extract=extract),
        progress=DistributedProgress(None, namespace="test", rank=0, world_size=1),
    )
    return evaluator


def _forbid_asset_io(*args: object, **kwargs: object) -> Never:
    raise AssertionError("unexpected asset/cache I/O")


def test_real_features_are_instance_local_and_reject_wrong_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = (
        ValidationImageBatch(("a", "b"), torch.ones(2, 3, 2, 2, dtype=torch.uint8)),
        ValidationImageBatch(("c",), torch.ones(1, 3, 2, 2, dtype=torch.uint8)),
    )
    extracted: list[int] = []
    batch_count = 2

    def validation_batches(
        *args: object, **kwargs: object
    ) -> tuple[ValidationImageBatch, ...]:
        return batches[:batch_count]

    def extract(images: torch.Tensor) -> ImageFeatureBatch:
        extracted.append(images.shape[0])
        return _fake_features(images)

    monkeypatch.setattr(runtime, "iter_validation_image_batches", validation_batches)
    first, second, incomplete = (_cpu_evaluator(extract) for _ in range(3))
    with monkeypatch.context() as guard:
        for name in ("open", "read_bytes", "read_text", "is_file", "exists"):
            guard.setattr(Path, name, _forbid_asset_io)
        guard.setattr(torch, "load", _forbid_asset_io)
        guard.setattr(torch, "save", _forbid_asset_io)
        real = first._real()
        assert first._real() is real
        assert second._real() is not real
        assert extracted == [2, 1, 2, 1]
        batch_count = 1
        with pytest.raises(
            EvaluationError, match="computed real feature count differs"
        ):
            incomplete._real()
        assert incomplete._real_features is None


@pytest.mark.parametrize(("rank", "training"), [(0, True), (1, False)])
def test_evaluate_regenerates_and_trims_three_uneven_rank_gathers(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    training: bool,
) -> None:
    evaluator = _cpu_evaluator()
    evaluator.rank, evaluator.world_size, evaluator.is_main_process = rank, 2, rank == 0
    evaluator.composite.train(training)
    cases = tuple(PromptCase(str(i), "prompt", (), i, 16, 16) for i in range(1, 6))
    stages: list[str] = []
    generated: list[tuple[str, ...]] = []
    finalized: list[ImageFeatureBatch] = []
    expected_result = EvaluationResult(
        10, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5, 3, Path("unused")
    )

    def run_all(stage: str, operation: Callable[[], object]) -> object:
        stages.append(stage.rsplit("/", 1)[-1])
        return operation()

    def run_on_rank(stage: str, owner: int, operation: Callable[[], object]) -> object:
        assert owner == 0
        stages.append(stage.rsplit("/", 1)[-1])
        return operation() if rank == 0 else None

    def stage_cases(*args: object, **kwargs: object) -> tuple[PromptCase, ...]:
        stages.append("prompts")
        return cases

    def generate(batch: tuple[PromptCase, ...], **kwargs: object) -> torch.Tensor:
        assert not evaluator.composite.training
        generated.append(tuple(case.prompt_id for case in batch))
        return torch.tensor([int(case.prompt_id) + offset for case in batch]).view(
            -1, 1, 1, 1
        )

    def gather(values: torch.Tensor) -> torch.Tensor:
        stages.append(f"gather-{values.shape[1]}")
        full = expected[values.shape[1]]
        local = full[:2] if rank == 0 else full[2:]
        assert values.shape == (3, values.shape[1])
        torch.testing.assert_close(values[: local.shape[0]], local)
        if rank == 0:
            assert torch.count_nonzero(values[2]).item() == 0
        return torch.cat((full[:2], torch.zeros_like(full[:1]), full[2:]))

    def finalize(update: int, features: ImageFeatureBatch) -> EvaluationResult:
        assert update == 10 and features.count == 5
        for values in (
            features.inception_features,
            features.clip_features,
            features.inception_logits,
        ):
            torch.testing.assert_close(values, expected[values.shape[1]])
        finalized.append(features)
        return expected_result

    evaluator.__dict__["progress"] = SimpleNamespace(
        run_all=run_all, run_on_rank=run_on_rank
    )
    monkeypatch.setattr(runtime, "_stage_cases", stage_cases)
    monkeypatch.setattr(runtime, "_generate", generate)
    monkeypatch.setattr(evaluator, "_gather", gather)
    monkeypatch.setattr(evaluator, "_finalize", finalize)
    with monkeypatch.context() as guard:
        for name in ("open", "read_bytes", "read_text", "is_file", "exists"):
            guard.setattr(Path, name, _forbid_asset_io)
        guard.setattr(torch, "load", _forbid_asset_io)
        guard.setattr(torch, "save", _forbid_asset_io)
        for offset in (10, 20):
            full = _fake_features(torch.arange(1, 6).add(offset).view(-1, 1, 1, 1))
            expected = {
                INCEPTION_FEATURE_DIM: full.inception_features,
                CLIP_EMBEDDING_DIM: full.clip_features,
                INCEPTION_LOGIT_DIM: full.inception_logits,
            }
            assert evaluator.evaluate(10) == (expected_result if rank == 0 else None)
            assert evaluator.composite.training is training
    assert generated == ([("1", "2")] if rank == 0 else [("3", "4"), ("5",)]) * 2
    assert len(finalized) == (2 if rank == 0 else 0)
    assert stages == [
        "initialize-feature-models",
        "prompts",
        "generate-local-features",
        f"gather-{INCEPTION_FEATURE_DIM}",
        f"gather-{CLIP_EMBEDDING_DIM}",
        f"gather-{INCEPTION_LOGIT_DIM}",
        "finalize-metrics",
    ] * 2


@pytest.mark.parametrize("training", [True, False])
def test_evaluate_restores_mode_on_generation_error(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
) -> None:
    evaluator = _cpu_evaluator()
    evaluator.composite.train(training)

    def stage_cases(*args: object, **kwargs: object) -> tuple[PromptCase, ...]:
        return (PromptCase("one", "prompt", (), 1, 16, 16),) * 5

    def generate(*args: object, **kwargs: object) -> Never:
        raise ValueError("generation failed")

    monkeypatch.setattr(runtime, "_stage_cases", stage_cases)
    monkeypatch.setattr(runtime, "_generate", generate)
    monkeypatch.setattr(evaluator, "_gather", _forbid_asset_io)
    monkeypatch.setattr(evaluator, "_finalize", _forbid_asset_io)
    with pytest.raises(ValueError, match="generation failed"):
        evaluator.evaluate(10)
    assert evaluator.composite.training is training


@pytest.mark.parametrize(
    ("problem", "index"),
    [("ok", 0)]
    + [
        (problem, index)
        for problem in ("missing", "size", "symlink")
        for index in (0, 1)
    ],
)
def test_inception_weights_require_both_local_files_without_content_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
    index: int,
) -> None:
    files = (("fid.pth", 3), ("is.pth", 5))
    root = tmp_path / "checkpoints"
    root.mkdir()
    paths = tuple(root / name for name, _ in files)
    for path, (_, size) in zip(paths, files, strict=True):
        path.write_bytes(b"x" * size)
    if problem != "ok":
        paths[index].unlink()
        if problem == "size":
            paths[index].write_bytes(b"x")
        elif problem == "symlink":
            target = tmp_path / "target.pth"
            target.write_bytes(b"x" * files[index][1])
            paths[index].symlink_to(target)
    monkeypatch.setattr(feature_module, "_INCEPTION_WEIGHT_FILES", files)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", _forbid_asset_io)
        guard.setattr(torch.hub, "download_url_to_file", _forbid_asset_io)
        guard.setattr(torch.hub, "load_state_dict_from_url", _forbid_asset_io)
        if problem == "ok":
            assert require_local_inception_weights() == paths
        else:
            with pytest.raises(
                EvaluationFeatureError,
                match="size differs" if problem == "size" else "missing",
            ):
                require_local_inception_weights()
