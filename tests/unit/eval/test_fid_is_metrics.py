"""FID and IS numerical contract tests."""

from __future__ import annotations

import pytest
import torch

from sakuramoon.eval.metrics import (
    FeatureStats,
    FeatureStatsAccumulator,
    InceptionScoreAccumulator,
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


def test_streaming_fid_and_is_match_one_shot_aggregation() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn((12, 5), generator=generator)
    probabilities = torch.softmax(torch.randn((12, 4), generator=generator), dim=1)
    feature_accumulator = FeatureStatsAccumulator()
    score_accumulator = InceptionScoreAccumulator(sample_count=12, splits=3)

    for feature_chunk, probability_chunk in zip(
        (features[:2], features[2:7], features[7:]),
        (probabilities[:2], probabilities[2:7], probabilities[7:]),
        strict=True,
    ):
        feature_accumulator.update(feature_chunk)
        score_accumulator.update(probability_chunk)

    streamed_stats = feature_accumulator.finalize()
    one_shot_stats = FeatureStats.from_features(features)
    torch.testing.assert_close(streamed_stats.mean, one_shot_stats.mean)
    torch.testing.assert_close(streamed_stats.covariance, one_shot_stats.covariance)
    streamed_score = score_accumulator.finalize()
    one_shot_score = inception_score(probabilities, splits=3)
    assert streamed_score.mean == pytest.approx(one_shot_score.mean, abs=1e-12)
    assert streamed_score.std == pytest.approx(one_shot_score.std, abs=1e-12)


def test_streaming_aggregators_reject_incomplete_or_excess_input() -> None:
    feature_accumulator = FeatureStatsAccumulator()
    feature_accumulator.update(torch.ones((1, 2)))
    with pytest.raises(ValueError, match="at least two"):
        feature_accumulator.finalize()

    score_accumulator = InceptionScoreAccumulator(sample_count=4, splits=2)
    score_accumulator.update(torch.full((3, 2), 0.5))
    with pytest.raises(ValueError, match="incomplete"):
        score_accumulator.finalize()
    with pytest.raises(ValueError, match="more probabilities"):
        score_accumulator.update(torch.full((2, 2), 0.5))


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
