"""Strict aggregation for the fixed 2,000-image Mage-VAE quality gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import torch

ReconstructionCohort = Literal["stratified", "risk"]
ReconstructionResolutionClass = Literal["base_512", "extended"]

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


def _is_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor)


def _is_tuple_of(value: object, member_type: type[object]) -> bool:
    if type(value) is not tuple:
        return False
    members = cast(tuple[object, ...], value)
    return all(isinstance(member, member_type) for member in members)


def _require_metric_identity(value: object) -> ReconstructionMetricIdentity:
    if not isinstance(value, ReconstructionMetricIdentity):
        raise ReconstructionEvaluationError(
            "metric engine must expose one valid metric identity"
        )
    return value


def _validate_sha256(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReconstructionEvaluationError(f"{label} SHA-256 is invalid")


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


@dataclass(frozen=True)
class ReconstructionMetricBatch:
    lpips: torch.Tensor
    ssim: torch.Tensor


class ReconstructionMetricEngine(Protocol):
    @property
    def identity(self) -> ReconstructionMetricIdentity: ...

    def score(
        self,
        source: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> ReconstructionMetricBatch: ...


class ReconstructionVAE(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ReconstructionCohortMember:
    sample_id: int
    cohort: ReconstructionCohort
    image_height: int
    image_width: int
    resolution_class: ReconstructionResolutionClass

    def __post_init__(self) -> None:
        if type(self.sample_id) is not int or self.sample_id <= 0:
            raise ReconstructionEvaluationError("sample ID must be a positive integer")
        if self.cohort not in ("stratified", "risk"):
            raise ReconstructionEvaluationError("reconstruction cohort is invalid")
        if (
            type(self.image_height) is not int
            or type(self.image_width) is not int
            or self.image_height <= 0
            or self.image_width <= 0
            or self.image_height % 16
            or self.image_width % 16
        ):
            raise ReconstructionEvaluationError(
                "reconstruction cohort dimensions must be positive multiples of 16"
            )
        if self.resolution_class not in ("base_512", "extended"):
            raise ReconstructionEvaluationError(
                "reconstruction resolution class is invalid"
            )


@dataclass(frozen=True)
class ReconstructionCohortManifest:
    members: tuple[ReconstructionCohortMember, ...]

    def __post_init__(self) -> None:
        if not _is_tuple_of(self.members, ReconstructionCohortMember):
            raise ReconstructionEvaluationError(
                "reconstruction cohort manifest members must be an immutable tuple"
            )
        if len(self.members) != TOTAL_SAMPLES:
            raise ReconstructionEvaluationError(
                "reconstruction cohort manifest requires exactly 2,000 members"
            )
        sample_ids = tuple(member.sample_id for member in self.members)
        if sample_ids != tuple(sorted(sample_ids)) or len(set(sample_ids)) != TOTAL_SAMPLES:
            raise ReconstructionEvaluationError(
                "reconstruction cohort manifest IDs must be unique and sorted"
            )
        stratified = sum(member.cohort == "stratified" for member in self.members)
        risk = sum(member.cohort == "risk" for member in self.members)
        if stratified != STRATIFIED_SAMPLES or risk != RISK_SAMPLES:
            raise ReconstructionEvaluationError(
                "reconstruction cohort manifest requires 1,600 stratified and 400 risk members"
            )
        base_512 = sum(
            member.resolution_class == "base_512" for member in self.members
        )
        if base_512 != 1500:
            raise ReconstructionEvaluationError(
                "reconstruction cohort manifest requires 1,500 base-512 and 500 extended members"
            )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_reconstruction_cohort_manifest_bytes(self)
        ).hexdigest()


@dataclass(frozen=True)
class ReconstructionBatchSpec:
    sample_ids: tuple[int, ...]
    cohorts: tuple[ReconstructionCohort, ...]
    image_height: int
    image_width: int

    def __post_init__(self) -> None:
        if type(self.sample_ids) is not tuple or type(self.cohorts) is not tuple:
            raise ReconstructionEvaluationError(
                "reconstruction batch spec metadata must be immutable tuples"
            )
        if not self.sample_ids or len(self.sample_ids) != len(self.cohorts):
            raise ReconstructionEvaluationError(
                "reconstruction batch spec metadata lengths are invalid"
            )
        if any(type(sample_id) is not int or sample_id <= 0 for sample_id in self.sample_ids):
            raise ReconstructionEvaluationError("sample ID must be a positive integer")
        if any(cohort not in ("stratified", "risk") for cohort in self.cohorts):
            raise ReconstructionEvaluationError("reconstruction cohort is invalid")
        if (
            type(self.image_height) is not int
            or type(self.image_width) is not int
            or self.image_height <= 0
            or self.image_width <= 0
            or self.image_height % 16
            or self.image_width % 16
        ):
            raise ReconstructionEvaluationError(
                "reconstruction batch spec dimensions must be positive multiples of 16"
            )


@dataclass(frozen=True)
class ReconstructionBatch:
    sample_ids: tuple[int, ...]
    cohorts: tuple[ReconstructionCohort, ...]
    images: torch.Tensor
    severe_errors: tuple[bool, ...]
    detail_losses: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not _is_tensor(self.images):
            raise ReconstructionEvaluationError("reconstruction images must be a tensor")
        if (
            self.images.ndim != 4
            or self.images.shape[0] <= 0
            or self.images.shape[1] != 3
            or self.images.shape[2] % 16
            or self.images.shape[3] % 16
            or not torch.is_floating_point(self.images)
            or not bool(torch.isfinite(self.images).all().item())
        ):
            raise ReconstructionEvaluationError(
                "reconstruction images must be finite floating [B,3,H,W] multiples of 16"
            )
        batch = self.images.shape[0]
        if any(
            len(values) != batch
            for values in (
                self.sample_ids,
                self.cohorts,
                self.severe_errors,
                self.detail_losses,
            )
        ):
            raise ReconstructionEvaluationError(
                "reconstruction batch metadata length does not match images"
            )
        if any(type(sample_id) is not int or sample_id <= 0 for sample_id in self.sample_ids):
            raise ReconstructionEvaluationError("sample ID must be a positive integer")
        if any(cohort not in ("stratified", "risk") for cohort in self.cohorts):
            raise ReconstructionEvaluationError("reconstruction cohort is invalid")
        if any(type(value) is not bool for value in self.severe_errors + self.detail_losses):
            raise ReconstructionEvaluationError("manual quality labels must be booleans")

    @property
    def spec(self) -> ReconstructionBatchSpec:
        return ReconstructionBatchSpec(
            sample_ids=self.sample_ids,
            cohorts=self.cohorts,
            image_height=self.images.shape[2],
            image_width=self.images.shape[3],
        )


@dataclass(frozen=True)
class ReconstructionMetricIdentity:
    lpips_implementation: str
    lpips_version: str
    lpips_weights: str
    ssim_implementation: str
    ssim_version: str
    ssim_parameters: str

    def __post_init__(self) -> None:
        values = (
            self.lpips_implementation,
            self.lpips_version,
            self.lpips_weights,
            self.ssim_implementation,
            self.ssim_version,
            self.ssim_parameters,
        )
        if any(not value or value != value.strip() for value in values):
            raise ReconstructionEvaluationError(
                "reconstruction metric identity fields must be non-empty"
            )


@dataclass(frozen=True)
class ReconstructionEvaluation:
    cohort_manifest_sha256: str
    metric_identity: ReconstructionMetricIdentity
    observations: tuple[ReconstructionObservation, ...]
    report: ReconstructionQualityReport

    def __post_init__(self) -> None:
        _validate_sha256(self.cohort_manifest_sha256, "reconstruction cohort manifest")
        ids = tuple(item.sample_id for item in self.observations)
        if ids != tuple(sorted(ids)):
            raise ReconstructionEvaluationError(
                "reconstruction observations must be sorted by sample ID"
            )
        if summarize_reconstruction_quality(self.observations) != self.report:
            raise ReconstructionEvaluationError(
                "reconstruction report does not match observations"
            )


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


def _metric_values(
    metric_batch: ReconstructionMetricBatch,
    batch_size: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if (
        not _is_tensor(metric_batch.lpips)
        or not _is_tensor(metric_batch.ssim)
        or metric_batch.lpips.shape != (batch_size,)
        or metric_batch.ssim.shape != (batch_size,)
        or not torch.is_floating_point(metric_batch.lpips)
        or not torch.is_floating_point(metric_batch.ssim)
        or not bool(torch.isfinite(metric_batch.lpips).all().item())
        or not bool(torch.isfinite(metric_batch.ssim).all().item())
    ):
        raise ReconstructionEvaluationError(
            "metric engine must return finite floating LPIPS/SSIM vectors"
        )
    lpips = tuple(float(value.item()) for value in metric_batch.lpips.detach().cpu())
    ssim = tuple(float(value.item()) for value in metric_batch.ssim.detach().cpu())
    return lpips, ssim


def canonical_reconstruction_cohort_manifest_bytes(
    manifest: ReconstructionCohortManifest,
) -> bytes:
    payload = {
        "members": [
            {
                "cohort": member.cohort,
                "image_height": member.image_height,
                "image_width": member.image_width,
                "resolution_class": member.resolution_class,
                "sample_id": member.sample_id,
            }
            for member in manifest.members
        ],
        "schema_version": 1,
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _validate_batch_plan(
    cohort_manifest: ReconstructionCohortManifest,
    batch_plan: tuple[ReconstructionBatchSpec, ...],
) -> None:
    if not _is_tuple_of(batch_plan, ReconstructionBatchSpec):
        raise ReconstructionEvaluationError(
            "reconstruction batch plan must be an immutable tuple of batch specs"
        )
    planned_members = tuple(
        (
            sample_id,
            cohort,
            spec.image_height,
            spec.image_width,
        )
        for spec in batch_plan
        for sample_id, cohort in zip(spec.sample_ids, spec.cohorts, strict=True)
    )
    expected_members = tuple(
        (
            member.sample_id,
            member.cohort,
            member.image_height,
            member.image_width,
        )
        for member in cohort_manifest.members
    )
    if planned_members != expected_members:
        raise ReconstructionEvaluationError(
            "reconstruction batch plan does not exactly match the canonical cohort manifest"
        )


@torch.inference_mode()
def evaluate_reconstruction_batches(
    vae: ReconstructionVAE,
    metric_engine: ReconstructionMetricEngine,
    batches: Iterable[ReconstructionBatch],
    *,
    cohort_manifest: ReconstructionCohortManifest,
    batch_plan: tuple[ReconstructionBatchSpec, ...],
) -> ReconstructionEvaluation:
    """Execute VAE round trips and locked external metrics over the fixed cohort."""

    _validate_batch_plan(cohort_manifest, batch_plan)
    metric_identity = _require_metric_identity(metric_engine.identity)
    observations: list[ReconstructionObservation] = []
    batch_iterator = iter(batches)
    for expected_spec in batch_plan:
        try:
            batch = next(batch_iterator)
        except StopIteration:
            raise ReconstructionEvaluationError(
                "reconstruction batch source ended before its canonical plan"
            ) from None
        if batch.spec != expected_spec:
            raise ReconstructionEvaluationError(
                "reconstruction batch does not match its canonical batch plan"
            )
        latent = vae.encode(batch.images)
        reconstruction = vae.decode(latent)
        if reconstruction.shape != batch.images.shape:
            raise ReconstructionEvaluationError(
                "VAE reconstruction shape does not match source images"
            )
        if not bool(torch.isfinite(reconstruction).all().item()):
            raise ReconstructionEvaluationError("VAE reconstruction is non-finite")
        metric_batch = metric_engine.score(batch.images, reconstruction)
        lpips_values, ssim_values = _metric_values(
            metric_batch, batch.images.shape[0]
        )
        observations.extend(
            ReconstructionObservation(
                sample_id=sample_id,
                cohort=cohort,
                lpips=lpips,
                ssim=ssim,
                severe_error=severe_error,
                detail_loss=detail_loss,
            )
            for sample_id, cohort, lpips, ssim, severe_error, detail_loss in zip(
                batch.sample_ids,
                batch.cohorts,
                lpips_values,
                ssim_values,
                batch.severe_errors,
                batch.detail_losses,
                strict=True,
            )
        )
    try:
        next(batch_iterator)
    except StopIteration:
        pass
    else:
        raise ReconstructionEvaluationError(
            "reconstruction batch source exceeds its canonical plan"
        )
    ordered = tuple(sorted(observations, key=lambda item: item.sample_id))
    report = summarize_reconstruction_quality(ordered)
    return ReconstructionEvaluation(
        cohort_manifest_sha256=cohort_manifest.sha256,
        metric_identity=metric_identity,
        observations=ordered,
        report=report,
    )


def canonical_reconstruction_evaluation_bytes(
    evaluation: ReconstructionEvaluation,
) -> bytes:
    identity = evaluation.metric_identity
    report = evaluation.report
    payload = {
        "cohort_manifest_sha256": evaluation.cohort_manifest_sha256,
        "metric_identity": {
            "lpips_implementation": identity.lpips_implementation,
            "lpips_version": identity.lpips_version,
            "lpips_weights": identity.lpips_weights,
            "ssim_implementation": identity.ssim_implementation,
            "ssim_parameters": identity.ssim_parameters,
            "ssim_version": identity.ssim_version,
        },
        "observations": [
            {
                "cohort": item.cohort,
                "detail_loss": item.detail_loss,
                "lpips": item.lpips,
                "sample_id": item.sample_id,
                "severe_error": item.severe_error,
                "ssim": item.ssim,
            }
            for item in evaluation.observations
        ],
        "report": {
            "detail_loss_count": report.detail_loss_count,
            "detail_loss_rate": report.detail_loss_rate,
            "median_lpips": report.median_lpips,
            "median_ssim": report.median_ssim,
            "p95_lpips": report.p95_lpips,
            "passed": report.passed,
            "risk_count": report.risk_count,
            "sample_count": report.sample_count,
            "severe_error_count": report.severe_error_count,
            "severe_error_rate": report.severe_error_rate,
            "stratified_count": report.stratified_count,
        },
        "schema_version": 1,
        "vae_path": "model/vae",
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def write_reconstruction_evaluation(
    evaluation: ReconstructionEvaluation,
    destination: Path,
) -> str:
    payload = canonical_reconstruction_evaluation_bytes(evaluation)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except ReconstructionEvaluationError:
        raise
    except OSError:
        raise ReconstructionEvaluationError(
            "reconstruction evaluation artifact could not be written"
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "MAX_DETAIL_LOSS_RATE",
    "MAX_MEDIAN_LPIPS",
    "MAX_P95_LPIPS",
    "MAX_SEVERE_ERROR_RATE",
    "MIN_MEDIAN_SSIM",
    "RISK_SAMPLES",
    "STRATIFIED_SAMPLES",
    "TOTAL_SAMPLES",
    "ReconstructionBatch",
    "ReconstructionBatchSpec",
    "ReconstructionCohortManifest",
    "ReconstructionCohortMember",
    "ReconstructionEvaluation",
    "ReconstructionEvaluationError",
    "ReconstructionMetricBatch",
    "ReconstructionMetricEngine",
    "ReconstructionMetricIdentity",
    "ReconstructionObservation",
    "ReconstructionQualityReport",
    "ReconstructionVAE",
    "canonical_reconstruction_cohort_manifest_bytes",
    "canonical_reconstruction_evaluation_bytes",
    "evaluate_reconstruction_batches",
    "summarize_reconstruction_quality",
    "write_reconstruction_evaluation",
]
