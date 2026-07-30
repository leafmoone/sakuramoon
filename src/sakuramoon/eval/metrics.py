"""Deterministic FID and Inception Score aggregation from locked features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch


def _eigvalsh(matrix: torch.Tensor) -> torch.Tensor:
    result = torch.linalg.eigvalsh(matrix)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    return cast(torch.Tensor, result)


def _eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.linalg.eigh(matrix)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    return cast(tuple[torch.Tensor, torch.Tensor], result)


@dataclass(frozen=True, slots=True)
class FeatureStats:
    count: int
    mean: torch.Tensor
    covariance: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 2:
            raise ValueError("feature stats require at least two observations")
        if (
            self.mean.dtype != torch.float64
            or self.mean.device.type != "cpu"
            or self.mean.ndim != 1
            or self.covariance.dtype != torch.float64
            or self.covariance.device.type != "cpu"
            or self.covariance.shape != (self.mean.numel(), self.mean.numel())
        ):
            raise ValueError("feature stats must be CPU float64 vectors/matrices")
        if not bool(
            torch.isfinite(self.mean).all().item()
            and torch.isfinite(self.covariance).all().item()
        ):
            raise ValueError("feature stats must be finite")
        if not torch.allclose(
            self.covariance,
            self.covariance.T,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("feature covariance must be symmetric")
        minimum = float(_eigvalsh(self.covariance).min().item())
        if minimum < -1e-9:
            raise ValueError("feature covariance must be positive semidefinite")

    @classmethod
    def from_features(cls, features: torch.Tensor) -> FeatureStats:
        if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] == 0:
            raise ValueError("features must have shape [N,D] with N >= 2")
        values = features.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("features must be finite")
        mean = values.mean(dim=0)
        centered = values - mean
        covariance = centered.T @ centered / (values.shape[0] - 1)
        return cls(values.shape[0], mean, covariance)


def frechet_inception_distance(
    generated: FeatureStats,
    real: FeatureStats,
) -> float:
    """Compute stable Gaussian Frechet distance for two PSD covariances."""

    if generated.mean.shape != real.mean.shape:
        raise ValueError("generated and real feature dimensions differ")
    eigenvalues, eigenvectors = _eigh(generated.covariance)
    generated_root = (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(0.0).sqrt())
        @ eigenvectors.T
    )
    middle = generated_root @ real.covariance @ generated_root
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues = _eigvalsh(middle)
    if float(middle_eigenvalues.min().item()) < -1e-8:
        raise ValueError("covariance product is not positive semidefinite")
    covariance_trace = middle_eigenvalues.clamp_min(0.0).sqrt().sum()
    difference = generated.mean - real.mean
    value = (
        difference.square().sum()
        + torch.trace(generated.covariance)
        + torch.trace(real.covariance)
        - 2.0 * covariance_trace
    )
    result = float(value.clamp_min(0.0).item())
    if not math.isfinite(result):
        raise FloatingPointError("FID is nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class InceptionScore:
    mean: float
    std: float
    splits: int
    sample_count: int


def inception_score(probabilities: torch.Tensor, *, splits: int) -> InceptionScore:
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] < 2
        or probabilities.shape[1] < 2
    ):
        raise ValueError("IS probabilities must have shape [N,C]")
    if type(splits) is not int or splits <= 0 or probabilities.shape[0] % splits != 0:
        raise ValueError("IS splits must be positive and divide the sample count")
    values = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()) or bool((values < 0.0).any().item()):
        raise ValueError("IS probabilities must be finite and nonnegative")
    if not torch.allclose(
        values.sum(dim=1),
        torch.ones(values.shape[0], dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("IS probability rows must sum to one")
    values = values.clamp_min(torch.finfo(torch.float64).tiny)
    split_scores: list[float] = []
    for split in values.chunk(splits):
        marginal = split.mean(dim=0)
        kl = (split * (split.log() - marginal.log())).sum(dim=1)
        split_scores.append(float(kl.mean().exp().item()))
    score_tensor = torch.tensor(split_scores, dtype=torch.float64)
    return InceptionScore(
        mean=float(score_tensor.mean().item()),
        std=float(score_tensor.std(unbiased=False).item()),
        splits=splits,
        sample_count=values.shape[0],
    )


__all__ = [
    "FeatureStats",
    "InceptionScore",
    "frechet_inception_distance",
    "inception_score",
]
