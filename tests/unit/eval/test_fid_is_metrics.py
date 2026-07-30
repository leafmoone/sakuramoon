"""FID and IS numerical contract tests."""

from __future__ import annotations

import pytest
import torch

from sakuramoon.eval.metrics import (
    FeatureStats,
    frechet_inception_distance,
    inception_score,
)


def test_fid_is_zero_for_identical_features_and_exact_for_mean_shift() -> None:
    real_features = torch.tensor(
        [[-1.0, -2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, -2.0]],
        dtype=torch.float32,
    )
    real = FeatureStats.from_features(real_features)
    shifted = FeatureStats.from_features(real_features + torch.tensor([2.0, -3.0]))

    assert frechet_inception_distance(real, real) == pytest.approx(0.0, abs=1e-12)
    assert frechet_inception_distance(shifted, real) == pytest.approx(13.0, abs=1e-10)


def test_fid_supports_singular_psd_covariance() -> None:
    real = FeatureStats.from_features(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    generated = FeatureStats.from_features(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))

    assert frechet_inception_distance(generated, real) == pytest.approx(1.0, abs=1e-10)


def test_feature_stats_reject_nonfinite_or_non_psd_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeatureStats.from_features(torch.tensor([[0.0, 1.0], [float("nan"), 2.0]]))

    with pytest.raises(ValueError, match="positive semidefinite"):
        FeatureStats(
            2,
            torch.zeros(2, dtype=torch.float64),
            torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64),
        )


def test_inception_score_uniform_is_one_and_balanced_one_hot_is_two() -> None:
    uniform = torch.full((8, 2), 0.5)
    balanced = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 4)

    uniform_score = inception_score(uniform, splits=2)
    balanced_score = inception_score(balanced, splits=2)

    assert uniform_score.mean == pytest.approx(1.0)
    assert uniform_score.std == pytest.approx(0.0)
    assert balanced_score.mean == pytest.approx(2.0)
    assert balanced_score.std == pytest.approx(0.0)
    assert balanced_score.sample_count == 8


def test_inception_score_accepts_float32_softmax_rounding() -> None:
    probabilities = torch.softmax(torch.randn((8, 4), dtype=torch.float32), dim=1)

    score = inception_score(probabilities, splits=2)

    assert score.mean >= 1.0


@pytest.mark.parametrize(
    ("probabilities", "splits", "expected"),
    [
        (torch.tensor([[0.2, 0.2], [0.5, 0.5]]), 1, "sum to one"),
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 3, "divide"),
        (torch.tensor([[1.0, -0.1], [0.0, 1.1]]), 1, "nonnegative"),
    ],
)
def test_inception_score_rejects_invalid_probabilities(
    probabilities: torch.Tensor, splits: int, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        inception_score(probabilities, splits=splits)
