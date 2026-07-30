from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sakuramoon.eval.vae_reconstruction import (
    ReconstructionArtifactExistsError,
    ReconstructionBatch,
    ReconstructionEvaluation,
    ReconstructionEvaluationError,
    ReconstructionMetricBatch,
    ReconstructionMetricIdentity,
    ReconstructionObservation,
    canonical_reconstruction_evaluation_bytes,
    evaluate_reconstruction_batches,
    summarize_reconstruction_quality,
    write_reconstruction_evaluation,
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


class _FakeVAE:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        self.encode_calls += 1
        return torch.zeros(
            image.shape[0],
            128,
            image.shape[2] // 16,
            image.shape[3] // 16,
            dtype=image.dtype,
            device=image.device,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        self.decode_calls += 1
        return torch.zeros(
            latent.shape[0],
            3,
            latent.shape[2] * 16,
            latent.shape[3] * 16,
            dtype=latent.dtype,
            device=latent.device,
        )


class _FakeMetrics:
    def __init__(self, *, invalid: str | None = None) -> None:
        self.calls = 0
        self.invalid = invalid

    def score(
        self, source: torch.Tensor, reconstruction: torch.Tensor
    ) -> ReconstructionMetricBatch:
        assert not torch.is_grad_enabled()
        assert source.shape == reconstruction.shape
        self.calls += 1
        shape = (source.shape[0] + 1,) if self.invalid == "shape" else (source.shape[0],)
        lpips = torch.full(shape, 0.02, dtype=torch.float32, device=source.device)
        ssim = torch.full(shape, 0.95, dtype=torch.float32, device=source.device)
        if self.invalid == "nan":
            lpips[0] = float("nan")
        return ReconstructionMetricBatch(lpips=lpips, ssim=ssim)


def _metric_identity() -> ReconstructionMetricIdentity:
    return ReconstructionMetricIdentity(
        lpips_implementation="synthetic-test-engine",
        lpips_version="1.0",
        lpips_weights="synthetic-test-weights",
        ssim_implementation="synthetic-test-engine",
        ssim_version="1.0",
        ssim_parameters="synthetic-test-window",
    )


def _batches() -> Iterator[ReconstructionBatch]:
    for start in range(0, 2000, 500):
        ids = tuple(range(start + 1, start + 501))
        yield ReconstructionBatch(
            sample_ids=ids,
            cohorts=tuple(
                "stratified" if sample_id <= 1600 else "risk"
                for sample_id in ids
            ),
            images=torch.zeros(500, 3, 16, 16),
            severe_errors=tuple(sample_id <= 19 for sample_id in ids),
            detail_losses=tuple(sample_id <= 99 for sample_id in ids),
        )


def test_executor_runs_real_round_trip_metric_batches_and_fixed_gate() -> None:
    vae = _FakeVAE()
    metrics = _FakeMetrics()

    evaluation = evaluate_reconstruction_batches(
        vae,
        metrics,
        _batches(),
        cohort_manifest_sha256="a" * 64,
        metric_identity=_metric_identity(),
    )

    assert vae.encode_calls == vae.decode_calls == metrics.calls == 4
    assert len(evaluation.observations) == 2000
    assert evaluation.report.passed is True
    assert evaluation.report.severe_error_count == 19
    assert evaluation.report.detail_loss_count == 99


@pytest.mark.parametrize("invalid", ["shape", "nan"])
def test_executor_rejects_invalid_metric_engine_output(invalid: str) -> None:
    with pytest.raises(ReconstructionEvaluationError, match="metric engine"):
        evaluate_reconstruction_batches(
            _FakeVAE(),
            _FakeMetrics(invalid=invalid),
            iter((_batches().__next__(),)),
            cohort_manifest_sha256="a" * 64,
            metric_identity=_metric_identity(),
        )


def test_executor_rejects_invalid_identity_before_model_work() -> None:
    vae = _FakeVAE()
    with pytest.raises(ReconstructionEvaluationError, match="SHA-256"):
        evaluate_reconstruction_batches(
            vae,
            _FakeMetrics(),
            _batches(),
            cohort_manifest_sha256="invalid",
            metric_identity=_metric_identity(),
        )
    assert vae.encode_calls == 0


def test_executor_rejects_cross_batch_duplicate_ids_before_second_forward() -> None:
    first = _batches().__next__()
    duplicate = ReconstructionBatch(
        sample_ids=first.sample_ids,
        cohorts=first.cohorts,
        images=first.images,
        severe_errors=first.severe_errors,
        detail_losses=first.detail_losses,
    )
    vae = _FakeVAE()
    with pytest.raises(ReconstructionEvaluationError, match="unique"):
        evaluate_reconstruction_batches(
            vae,
            _FakeMetrics(),
            iter((first, duplicate)),
            cohort_manifest_sha256="a" * 64,
            metric_identity=_metric_identity(),
        )
    assert vae.encode_calls == 1


def test_executor_rejects_intra_batch_duplicate_ids_before_forward() -> None:
    first = _batches().__next__()
    duplicate_ids = (first.sample_ids[0],) + first.sample_ids[:-1]
    duplicate = ReconstructionBatch(
        sample_ids=duplicate_ids,
        cohorts=first.cohorts,
        images=first.images,
        severe_errors=first.severe_errors,
        detail_losses=first.detail_losses,
    )
    vae = _FakeVAE()
    with pytest.raises(ReconstructionEvaluationError, match="unique"):
        evaluate_reconstruction_batches(
            vae,
            _FakeMetrics(),
            iter((duplicate,)),
            cohort_manifest_sha256="a" * 64,
            metric_identity=_metric_identity(),
        )
    assert vae.encode_calls == 0


def test_evaluation_artifact_is_canonical_bound_and_no_clobber(
    tmp_path: Path,
) -> None:
    evaluation = evaluate_reconstruction_batches(
        _FakeVAE(),
        _FakeMetrics(),
        _batches(),
        cohort_manifest_sha256="b" * 64,
        metric_identity=_metric_identity(),
    )
    payload = canonical_reconstruction_evaluation_bytes(evaluation)
    destination = tmp_path / "vae-reconstruction.json"

    digest = write_reconstruction_evaluation(evaluation, destination)

    assert destination.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    document = json.loads(payload)
    assert document["cohort_manifest_sha256"] == "b" * 64
    assert document["vae_path"] == "model/vae"
    assert len(document["observations"]) == 2000
    assert document["report"]["passed"] is True
    with pytest.raises(ReconstructionArtifactExistsError, match="already exists"):
        write_reconstruction_evaluation(evaluation, destination)


def test_evaluation_type_rejects_report_drift() -> None:
    observations = _observations()
    report = summarize_reconstruction_quality(observations)
    with pytest.raises(ReconstructionEvaluationError, match="does not match"):
        ReconstructionEvaluation(
            cohort_manifest_sha256="c" * 64,
            metric_identity=_metric_identity(),
            observations=observations,
            report=replace(report, passed=not report.passed),
        )
