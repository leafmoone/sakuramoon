from __future__ import annotations

import itertools

import pytest
import torch

from sakuramoon.model.irepa import IREPA_TEACHER_FEATURE_WIDTH
from sakuramoon.objective.irepa import (
    IRepaAlignmentLossOutput,
    IRepaLambdaSchedule,
    irepa_alignment_loss,
    irepa_weight_for_update,
    spatial_zscore_target,
)

D = IREPA_TEACHER_FEATURE_WIDTH


# ---------------------------------------------------------------------------
# spatial z-score target
# ---------------------------------------------------------------------------


def test_zscore_matches_manual_reference_in_fp32() -> None:
    torch.manual_seed(2024)  # pyright: ignore[reportUnknownMemberType]
    features = torch.randn(2, 6, D) * 3.0 + 1.0
    gamma = 0.6
    eps = 1e-6

    target = spatial_zscore_target(features, gamma=gamma, eps=eps)

    assert target.dtype is torch.float32
    assert target.shape == features.shape
    assert not target.requires_grad
    x = features.float()
    mean = x.mean(dim=1, keepdim=True)
    var = ((x - mean).square().sum(dim=1, keepdim=True)) / (x.shape[1] - 1)
    expected = (x - gamma * mean) / (var.sqrt() + eps)
    torch.testing.assert_close(target, expected, atol=1e-6, rtol=1e-6)


def test_zscore_accepts_bfloat16_input_and_returns_fp32() -> None:
    torch.manual_seed(2025)  # pyright: ignore[reportUnknownMemberType]
    features = torch.randn(1, 4, D).bfloat16()

    target = spatial_zscore_target(features, gamma=0.0, eps=1e-6)

    assert target.dtype is torch.float32
    assert not target.requires_grad
    x = features.float()
    expected = x / (x.std(dim=1, unbiased=True, keepdim=True) + 1e-6)
    torch.testing.assert_close(target, expected, atol=1e-5, rtol=1e-5)


def test_zscore_rejects_single_token_and_degenerate_shapes() -> None:
    for batch, tokens in ((0, 4), (1, 0), (1, 1)):
        features = torch.randn(batch, tokens, D)
        with pytest.raises(ValueError, match="T > 1|B > 0"):
            spatial_zscore_target(features, gamma=0.6, eps=1e-6)
    with pytest.raises(ValueError, match="D ="):
        spatial_zscore_target(torch.randn(1, 4, D - 1), gamma=0.6, eps=1e-6)
    with pytest.raises(ValueError, match="dimensions"):
        spatial_zscore_target(torch.randn(4, D), gamma=0.6, eps=1e-6)


def test_zscore_rejects_nonfinite_input_and_output() -> None:
    features = torch.randn(1, 4, D)
    features[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        spatial_zscore_target(features, gamma=0.6, eps=1e-6)
    # zero-variance features: gamma=1.0 makes the numerator exactly zero, so
    # the eps guard keeps the output finite and exactly zero
    flat = torch.ones(1, 8, D) * 7.0
    torch.testing.assert_close(
        spatial_zscore_target(flat, gamma=1.0, eps=1e-6),
        torch.zeros(1, 8, D),
        atol=0.0,
        rtol=0.0,
    )


def test_zscore_validates_gamma_and_eps() -> None:
    features = torch.randn(1, 4, D)
    for gamma in (-0.1, 1.1):
        with pytest.raises(ValueError, match="gamma"):
            spatial_zscore_target(features, gamma=gamma, eps=1e-6)
    with pytest.raises(ValueError, match="gamma"):
        spatial_zscore_target(features, gamma=0, eps=1e-6)  # int, not float
    with pytest.raises(ValueError, match="gamma"):
        spatial_zscore_target(features, gamma=True, eps=1e-6)  # bool
    for eps in (0.0, -1e-6, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="eps"):
            spatial_zscore_target(features, gamma=0.6, eps=eps)
    with pytest.raises(ValueError, match="eps"):
        spatial_zscore_target(features, gamma=0.6, eps=1)


def test_zscore_rejects_non_floating_input() -> None:
    with pytest.raises(ValueError, match="floating"):
        spatial_zscore_target(torch.randint(0, 9, (1, 4, D)), gamma=0.6, eps=1e-6)


# ---------------------------------------------------------------------------
# cosine alignment loss
# ---------------------------------------------------------------------------


def _aligned_pair(student: torch.Tensor, target: torch.Tensor) -> None:
    output = irepa_alignment_loss(student, target)
    assert isinstance(output, IRepaAlignmentLossOutput)
    assert output.per_sample.dtype is torch.float32
    assert output.cosine_per_sample.dtype is torch.float32
    assert output.per_sample.shape == (student.shape[0],)
    assert output.cosine_per_sample.shape == (student.shape[0],)


def test_cosine_loss_identical_features_is_zero() -> None:
    torch.manual_seed(2026)  # pyright: ignore[reportUnknownMemberType]
    features = torch.randn(3, 5, D)
    _aligned_pair(features, features.clone())
    output = irepa_alignment_loss(features, features)
    torch.testing.assert_close(output.per_sample, torch.zeros(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        output.cosine_per_sample, torch.ones(3), atol=1e-6, rtol=0.0
    )


def test_cosine_loss_extremes_orthogonal_and_opposite() -> None:
    # per-token orthogonal: student lives in the first two feature channels
    # and target is the same vector rotated 90 degrees there, so every
    # per-token dot product is exactly 0
    torch.manual_seed(2029)  # pyright: ignore[reportUnknownMemberType]
    base = torch.randn(2, 5, D)
    base[:, :, 2:] = 0
    student = base.clone()
    target = base.clone()
    target[:, :, 0] = base[:, :, 1]
    target[:, :, 1] = -base[:, :, 0]
    output = irepa_alignment_loss(student, target)
    torch.testing.assert_close(output.per_sample, torch.ones(2), atol=1e-6, rtol=0.0)
    # opposite: cosine -1, loss 2
    features_neg = torch.randn(2, 4, D)
    output = irepa_alignment_loss(features_neg, -features_neg)
    torch.testing.assert_close(
        output.per_sample, torch.full((2,), 2.0), atol=2e-6, rtol=0.0
    )
    torch.testing.assert_close(
        output.cosine_per_sample, torch.full((2,), -1.0), atol=2e-6, rtol=0.0
    )


def test_cosine_loss_is_token_mean_per_sample_not_global_mean() -> None:
    # Dense [B, T, D] keeps one shared token count, so the one-sample-one-
    # weight contract is visible as the per-sample token MEAN: sample 0 mixes
    # three aligned and one opposite token (loss 0.5), sample 1 is fully
    # aligned (loss 0).
    torch.manual_seed(2030)  # pyright: ignore[reportUnknownMemberType]
    a = torch.randn(D)
    b = torch.randn(D)
    c = torch.randn(D)
    d = torch.randn(D)
    e = torch.randn(D)
    f = torch.randn(D)
    student = torch.stack((torch.stack((a, b, c, -a)), torch.stack((d, e, f, d))))
    target = torch.stack((torch.stack((a, b, c, a)), torch.stack((d, e, f, d))))

    output = irepa_alignment_loss(student, target)

    # the output is the [B] per-sample token-mean vector (never per token)
    assert output.per_sample.shape == (2,)
    assert output.per_sample[0].item() == pytest.approx(0.5, abs=1e-6)
    assert output.per_sample[1].item() == pytest.approx(0.0, abs=1e-6)
    assert output.cosine_per_sample[0].item() == pytest.approx(0.5, abs=1e-6)
    assert output.cosine_per_sample[1].item() == pytest.approx(1.0, abs=1e-6)


def test_cosine_loss_requires_exact_shapes_and_width() -> None:
    student = torch.randn(2, 4, D)
    target = torch.randn(2, 4, D)
    with pytest.raises(ValueError, match="exactly equal shapes"):
        irepa_alignment_loss(student, target[:, 1:, :])
    with pytest.raises(ValueError, match="D ="):
        irepa_alignment_loss(torch.randn(2, 4, D - 1), torch.randn(2, 4, D - 1))
    with pytest.raises(ValueError, match="must both be"):
        irepa_alignment_loss(student[:, 0, :], target)


def test_cosine_loss_rejects_nonfinite_and_integer_input() -> None:
    student = torch.randn(2, 4, D)
    target = torch.randn(2, 4, D)
    bad_target = target.clone()
    bad_target[1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="target contains nonfinite"):
        irepa_alignment_loss(student, bad_target)
    bad_student = student.clone()
    bad_student[0, 1, 2] = float("inf")
    with pytest.raises(ValueError, match="student_features contain nonfinite"):
        irepa_alignment_loss(bad_student, target)
    with pytest.raises(ValueError, match="floating point"):
        irepa_alignment_loss(student.long(), target.long())


def test_cosine_loss_bfloat16_input_is_fp32_computed() -> None:
    torch.manual_seed(2027)  # pyright: ignore[reportUnknownMemberType]
    student = torch.randn(2, 4, D).bfloat16()
    target = torch.randn(2, 4, D).bfloat16()
    reference = irepa_alignment_loss(student.float(), target.float())

    output = irepa_alignment_loss(student, target)

    torch.testing.assert_close(
        output.per_sample, reference.per_sample, atol=1e-6, rtol=1e-5
    )


def test_cosine_loss_keeps_autograd_graph_to_student() -> None:
    torch.manual_seed(2028)  # pyright: ignore[reportUnknownMemberType]
    student = torch.randn(2, 4, D, requires_grad=True)
    target = torch.randn(2, 4, D)

    output = irepa_alignment_loss(student, target)
    output.per_sample.sum().backward()  # pyright: ignore[reportUnknownMemberType]

    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())


# ---------------------------------------------------------------------------
# lambda schedule
# ---------------------------------------------------------------------------


def _schedule_kwargs(**overrides: object) -> dict[str, object]:
    """Schedule arguments WITHOUT successful_update (always passed explicitly)."""

    base: dict[str, object] = {
        "start_successful_update": 100,
        "target_weight": 0.5,
        "ramp_in_updates": 10,
        "ramp_out_after_updates": None,
        "ramp_out_updates": 10,
    }
    base.update(overrides)
    return base


def test_lambda_is_zero_before_start_and_at_start() -> None:
    for u in (0, 50, 99):
        assert (
            irepa_weight_for_update(successful_update=u, **_schedule_kwargs())
            == 0.0
        )
    # off-by-one contract: the first enabled update gets exactly 0.0
    assert (
        irepa_weight_for_update(
            **_schedule_kwargs(successful_update=100)
        )
        == 0.0
    )


def test_lambda_reaches_target_exactly_at_ramp_end() -> None:
    weight = irepa_weight_for_update(**_schedule_kwargs(successful_update=110))
    assert weight == 0.5
    # the ramp is strictly increasing in between
    ramp = [
        irepa_weight_for_update(**_schedule_kwargs(successful_update=100 + i))
        for i in range(1, 11)
    ]
    assert all(a < b for a, b in itertools.pairwise(ramp))
    # half-cosine midpoint: exactly half the target at half the ramp
    assert ramp[4] == pytest.approx(0.25)


def test_lambda_holds_without_ramp_out() -> None:
    for u in (110, 111, 5000, 10**6):
        assert (
            irepa_weight_for_update(**_schedule_kwargs(successful_update=u)) == 0.5
        )


def test_lambda_ramp_out_half_cosine_boundaries() -> None:
    kwargs = _schedule_kwargs(ramp_out_after_updates=120, ramp_out_updates=10)
    # continuity: the first ramp-out update still holds the target
    assert irepa_weight_for_update(**kwargs, successful_update=120) == 0.5
    ramp = [
        irepa_weight_for_update(**kwargs, successful_update=120 + i)
        for i in range(1, 11)
    ]
    assert all(a > b for a, b in itertools.pairwise(ramp))
    # half-cosine midpoint: exactly half the target
    assert ramp[4] == pytest.approx(0.25)
    # end and beyond: exactly 0.0
    assert irepa_weight_for_update(**kwargs, successful_update=130) == 0.0
    assert irepa_weight_for_update(**kwargs, successful_update=131) == 0.0


def test_lambda_zero_target_short_circuits_to_zero() -> None:
    for u in (0, 100, 105, 110, 200):
        assert (
            irepa_weight_for_update(
                **_schedule_kwargs(successful_update=u, target_weight=0.0)
            )
            == 0.0
        )


def test_lambda_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="successful_update"):
        irepa_weight_for_update(**_schedule_kwargs(successful_update=-1))
    with pytest.raises(ValueError, match="successful_update"):
        irepa_weight_for_update(
            **_schedule_kwargs(successful_update=True)  # bool is not int
        )
    with pytest.raises(ValueError, match="start_successful_update"):
        irepa_weight_for_update(
            **_schedule_kwargs(successful_update=0, start_successful_update=-1)
        )
    for weight in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="target_weight"):
            irepa_weight_for_update(
                **_schedule_kwargs(successful_update=0, target_weight=weight)
            )
    with pytest.raises(ValueError, match="target_weight"):
        irepa_weight_for_update(
            **_schedule_kwargs(successful_update=0, target_weight=1)  # int
        )
    with pytest.raises(ValueError, match="ramp_in_updates"):
        irepa_weight_for_update(
            **_schedule_kwargs(successful_update=150, ramp_in_updates=0)
        )
    with pytest.raises(ValueError, match="ramp_out_updates"):
        irepa_weight_for_update(
            **_schedule_kwargs(successful_update=150, ramp_out_updates=0)
        )
    with pytest.raises(ValueError, match="ramp_out_after_updates"):
        irepa_weight_for_update(
            **_schedule_kwargs(
                successful_update=150, ramp_out_after_updates=5, ramp_in_updates=10
            )
        )
    with pytest.raises(ValueError, match="ramp_out_after_updates"):
        irepa_weight_for_update(
            **_schedule_kwargs(
                successful_update=150, ramp_out_after_updates=10, ramp_in_updates=10
            )
        )


def test_lambda_dataclass_validation_and_delegation() -> None:
    with pytest.raises(ValueError, match="ramp_out_after_updates"):
        IRepaLambdaSchedule(
            start_successful_update=0,
            target_weight=0.5,
            ramp_in_updates=10,
            ramp_out_after_updates=5,
            ramp_out_updates=10,
        )
    schedule = IRepaLambdaSchedule(
        start_successful_update=100,
        target_weight=0.5,
        ramp_in_updates=10,
        ramp_out_after_updates=120,
        ramp_out_updates=10,
    )
    for u in range(95, 135):
        assert schedule.weight_for_update(u) == irepa_weight_for_update(
            **_schedule_kwargs(successful_update=u, ramp_out_after_updates=120)
        )


def test_lambda_resume_determinism_is_pure() -> None:
    first = IRepaLambdaSchedule(0, 0.5, 100, 500, 50)
    resumed = IRepaLambdaSchedule(0, 0.5, 100, 500, 50)
    values_a = [first.weight_for_update(u) for u in range(0, 600, 7)]
    values_b = [resumed.weight_for_update(u) for u in range(0, 600, 7)]
    assert values_a == values_b
    # a "failed update" does not advance the anchor: rebinding the same
    # successful-update number reproduces the same weight
    failed_anchor = first.weight_for_update(250)
    retried_anchor = first.weight_for_update(250)
    assert failed_anchor == retried_anchor
