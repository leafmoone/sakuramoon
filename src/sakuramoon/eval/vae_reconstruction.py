"""Strict aggregation for the fixed 2,000-image Mage-VAE quality gate."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

ReconstructionCohort = Literal["stratified", "risk"]

STRATIFIED_SAMPLES = 1600
RISK_SAMPLES = 400
TOTAL_SAMPLES = STRATIFIED_SAMPLES + RISK_SAMPLES

MAX_MEDIAN_LPIPS = 0.03
MAX_P95_LPIPS = 0.08
MIN_MEDIAN_SSIM = 0.94
MAX_SEVERE_ERROR_RATE = 0.01
MAX_DETAIL_LOSS_RATE = 0.05


class ReconstructionEvaluationError(ValueError):
    """Reconstruction observations do not satisfy the acceptance contract."""


@dataclass(frozen=True)
class ReconstructionObservation:
    sample_id: int
    cohort: ReconstructionCohort
    lpips: float
    ssim: float
    severe_error: bool
    detail_loss: bool

    def __post_init__(self) -> None:
        if type(self.sample_id) is not int or self.sample_id <= 0:
            raise ReconstructionEvaluationError("sample ID must be a positive integer")
        if self.cohort not in ("stratified", "risk"):
            raise ReconstructionEvaluationError("reconstruction cohort is invalid")
        if (
            type(self.lpips) is not float
            or not math.isfinite(self.lpips)
            or self.lpips < 0.0
        ):
            raise ReconstructionEvaluationError("LPIPS must be a finite non-negative float")
        if (
            type(self.ssim) is not float
            or not math.isfinite(self.ssim)
            or not -1.0 <= self.ssim <= 1.0
        ):
            raise ReconstructionEvaluationError("SSIM must be a finite float in [-1, 1]")
        if type(self.severe_error) is not bool or type(self.detail_loss) is not bool:
            raise ReconstructionEvaluationError("manual quality labels must be booleans")


@dataclass(frozen=True)
class ReconstructionQualityReport:
    sample_count: int
    stratified_count: int
    risk_count: int
    median_lpips: float
    p95_lpips: float
    median_ssim: float
    severe_error_count: int
    severe_error_rate: float
    detail_loss_count: int
    detail_loss_rate: float
    passed: bool


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def summarize_reconstruction_quality(
    observations: tuple[ReconstructionObservation, ...],
) -> ReconstructionQualityReport:
    """Aggregate the complete fixed cohort and apply every locked quality threshold."""

    if len(observations) != TOTAL_SAMPLES:
        raise ReconstructionEvaluationError(
            "reconstruction acceptance requires exactly 2,000 observations"
        )
    sample_ids = tuple(observation.sample_id for observation in observations)
    if len(set(sample_ids)) != TOTAL_SAMPLES:
        raise ReconstructionEvaluationError("reconstruction sample IDs must be unique")
    stratified_count = sum(
        observation.cohort == "stratified" for observation in observations
    )
    risk_count = sum(observation.cohort == "risk" for observation in observations)
    if stratified_count != STRATIFIED_SAMPLES or risk_count != RISK_SAMPLES:
        raise ReconstructionEvaluationError(
            "reconstruction cohort must contain 1,600 stratified and 400 risk samples"
        )

    lpips_values = [observation.lpips for observation in observations]
    ssim_values = [observation.ssim for observation in observations]
    median_lpips = float(statistics.median(lpips_values))
    p95_lpips = _nearest_rank(lpips_values, 0.95)
    median_ssim = float(statistics.median(ssim_values))
    severe_error_count = sum(observation.severe_error for observation in observations)
    detail_loss_count = sum(observation.detail_loss for observation in observations)
    severe_error_rate = severe_error_count / TOTAL_SAMPLES
    detail_loss_rate = detail_loss_count / TOTAL_SAMPLES
    passed = (
        median_lpips <= MAX_MEDIAN_LPIPS
        and p95_lpips <= MAX_P95_LPIPS
        and median_ssim >= MIN_MEDIAN_SSIM
        and severe_error_rate < MAX_SEVERE_ERROR_RATE
        and detail_loss_rate < MAX_DETAIL_LOSS_RATE
    )
    return ReconstructionQualityReport(
        sample_count=TOTAL_SAMPLES,
        stratified_count=stratified_count,
        risk_count=risk_count,
        median_lpips=median_lpips,
        p95_lpips=p95_lpips,
        median_ssim=median_ssim,
        severe_error_count=severe_error_count,
        severe_error_rate=severe_error_rate,
        detail_loss_count=detail_loss_count,
        detail_loss_rate=detail_loss_rate,
        passed=passed,
    )


__all__ = [
    "MAX_DETAIL_LOSS_RATE",
    "MAX_MEDIAN_LPIPS",
    "MAX_P95_LPIPS",
    "MAX_SEVERE_ERROR_RATE",
    "MIN_MEDIAN_SSIM",
    "RISK_SAMPLES",
    "STRATIFIED_SAMPLES",
    "TOTAL_SAMPLES",
    "ReconstructionEvaluationError",
    "ReconstructionObservation",
    "ReconstructionQualityReport",
    "summarize_reconstruction_quality",
]
