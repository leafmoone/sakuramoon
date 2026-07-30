from __future__ import annotations

import pytest

from sakuramoon.eval.vae_reconstruction import (
    ReconstructionEvaluationError,
    ReconstructionObservation,
    summarize_reconstruction_quality,
)


def _observations(
    *,
    lpips: float = 0.02,
    ssim: float = 0.95,
    severe_errors: int = 0,
    detail_losses: int = 0,
) -> tuple[ReconstructionObservation, ...]:
    return tuple(
        ReconstructionObservation(
            sample_id=index + 1,
            cohort="stratified" if index < 1600 else "risk",
            lpips=lpips,
            ssim=ssim,
            severe_error=index < severe_errors,
            detail_loss=index < detail_losses,
        )
        for index in range(2000)
    )


def test_fixed_reconstruction_quality_gate_passes_all_thresholds() -> None:
    report = summarize_reconstruction_quality(
        _observations(severe_errors=19, detail_losses=99)
    )

    assert report.sample_count == 2000
    assert report.stratified_count == 1600
    assert report.risk_count == 400
    assert report.median_lpips == 0.02
    assert report.p95_lpips == 0.02
    assert report.median_ssim == 0.95
    assert report.severe_error_rate == 19 / 2000
    assert report.detail_loss_rate == 99 / 2000
    assert report.passed is True


@pytest.mark.parametrize(
    "observations",
    [
        _observations(lpips=0.031),
        _observations(lpips=0.081),
        _observations(ssim=0.939),
        _observations(severe_errors=20),
        _observations(detail_losses=100),
    ],
)
def test_reconstruction_quality_gate_rejects_each_failed_threshold(
    observations: tuple[ReconstructionObservation, ...],
) -> None:
    assert summarize_reconstruction_quality(observations).passed is False


def test_reconstruction_quality_uses_nearest_rank_p95() -> None:
    observations = list(_observations())
    for index in range(100):
        observation = observations[index]
        observations[index] = ReconstructionObservation(
            sample_id=observation.sample_id,
            cohort=observation.cohort,
            lpips=0.08,
            ssim=observation.ssim,
            severe_error=False,
            detail_loss=False,
        )

    report = summarize_reconstruction_quality(tuple(observations))

    assert report.p95_lpips == 0.02
    assert report.passed is True


@pytest.mark.parametrize(
    "observations,expected",
    [
        (_observations()[:-1], "exactly 2,000"),
        (_observations()[:-1] + (_observations()[0],), "unique"),
        (
            tuple(
                ReconstructionObservation(
                    sample_id=index + 1,
                    cohort="stratified",
                    lpips=0.02,
                    ssim=0.95,
                    severe_error=False,
                    detail_loss=False,
                )
                for index in range(2000)
            ),
            "1,600 stratified",
        ),
    ],
)
def test_reconstruction_quality_rejects_invalid_fixed_cohort(
    observations: tuple[ReconstructionObservation, ...], expected: str
) -> None:
    with pytest.raises(ReconstructionEvaluationError, match=expected):
        summarize_reconstruction_quality(observations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lpips", float("nan")),
        ("lpips", float("inf")),
        ("lpips", -0.1),
        ("lpips", 0),
        ("ssim", float("nan")),
        ("ssim", 1.1),
        ("severe_error", 0),
        ("detail_loss", 0),
    ],
)
def test_reconstruction_observation_rejects_invalid_metric_and_label_types(
    field: str, value: object
) -> None:
    payload: dict[str, object] = {
        "sample_id": 1,
        "cohort": "stratified",
        "lpips": 0.02,
        "ssim": 0.95,
        "severe_error": False,
        "detail_loss": False,
    }
    payload[field] = value

    with pytest.raises(ReconstructionEvaluationError):
        ReconstructionObservation(**payload)  # type: ignore[arg-type]
