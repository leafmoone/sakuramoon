# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest
import torch

from sakuramoon.config.schema import EvaluationEnabledConfig
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX
from sakuramoon.eval.metrics import (
    FeatureStats,
    clip_maximum_mean_discrepancy,
    frechet_inception_distance,
    inception_score,
    kernel_inception_distance,
)
from sakuramoon.eval.runtime import _conditioning_inputs, _stage_cases
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


def test_evaluation_uses_normal_caption_boundary_truncation() -> None:
    case = PromptCase(
        prompt_id="long-prompt",
        prompt=" ".join(["word"] * 600),
        conditions=(),
        seed=1,
        height=256,
        width=256,
    )

    inputs = _conditioning_inputs((case,), _Tokenizer(), torch.device("cpu"))

    input_ids, attention_mask = inputs[:2]
    assert input_ids.shape == (2, 98)
    assert torch.equal(attention_mask.sum(dim=1), torch.tensor([39, 39]))


def test_evaluation_stages_every_prompt_as_one_to_one(tmp_path) -> None:
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
