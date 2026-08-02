from __future__ import annotations

# pyright: reportPrivateUsage=false
import pytest
import torch

from sakuramoon.eval.generate import _initial_gaussian_noise
from sakuramoon.eval.metrics import (
    FeatureStats,
    frechet_inception_distance,
    inception_score,
)
from sakuramoon.eval.spec import PromptCase
from sakuramoon.sampling.heun import heun_final_euler


def test_checkpoint_evaluator_seeded_noise_is_cuda_fp32() -> None:
    device = torch.device("cuda", 0)
    cases = (
        PromptCase("case-0", "synthetic prompt", (), 17, 256, 256),
        PromptCase("case-1", "synthetic prompt", (), 23, 256, 256),
    )

    first = _initial_gaussian_noise(cases, device=device)
    repeated = _initial_gaussian_noise(cases, device=device)

    assert first.device == device
    assert first.dtype == torch.float32
    assert first.shape == (2, 128, 16, 16)
    torch.testing.assert_close(first, repeated, atol=0.0, rtol=0.0)


def test_seeded_cuda_heun_feature_and_metric_pipeline_is_deterministic() -> None:
    device = torch.device("cuda", 0)

    def generate(seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((8, 4), generator=generator, device=device)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return (-0.25 * state + timestep[:, None]).float()

        result = heun_final_euler(velocity, noise, steps=50)
        assert result.nfe == 99
        return result.state

    first = generate(123)
    second = generate(123)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)

    first_stats = FeatureStats.from_features(first)
    second_stats = FeatureStats.from_features(second)
    score = frechet_inception_distance(first_stats, second_stats)
    probabilities = torch.softmax(first[:, :2], dim=1)
    is_score = inception_score(probabilities, splits=2)

    assert score == pytest.approx(0.0, abs=1e-12)
    assert is_score.mean >= 1.0
    assert is_score.sample_count == 8
