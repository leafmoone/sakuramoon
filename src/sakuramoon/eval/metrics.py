"""FID and Inception Score math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch


def _eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.linalg.eigh(matrix),  # pyright: ignore[reportUnknownMemberType]
    )


def _eigvalsh(matrix: torch.Tensor) -> torch.Tensor:
    return cast(
        torch.Tensor,
        torch.linalg.eigvalsh(matrix),  # pyright: ignore[reportUnknownMemberType]
    )


@dataclass(frozen=True, slots=True)
class FeatureStats:
    count: int
    mean: torch.Tensor
    covariance: torch.Tensor

    @classmethod
    def from_features(cls, features: torch.Tensor) -> FeatureStats:
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("FID features must have shape [N,D] with N >= 2")
        values = features.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("FID features contain nonfinite values")
        mean = values.mean(dim=0)
        centered = values - mean
        covariance = centered.T @ centered / (values.shape[0] - 1)
        return cls(values.shape[0], mean, covariance)


def frechet_inception_distance(generated: FeatureStats, real: FeatureStats) -> float:
    if generated.mean.shape != real.mean.shape:
        raise ValueError("generated and real feature dimensions differ")
    eigenvalues, eigenvectors = _eigh(generated.covariance)
    generated_root = (
        eigenvectors
        @ torch.diag(eigenvalues.clamp_min(0.0).sqrt())
        @ eigenvectors.T
    )
    middle = generated_root @ real.covariance @ generated_root
    covariance_trace = _eigvalsh((middle + middle.T) * 0.5).clamp_min(0.0).sqrt().sum()
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
        or type(splits) is not int
        or splits <= 0
        or probabilities.shape[0] % splits
    ):
        raise ValueError("IS probabilities or split count are invalid")
    values = probabilities.detach().to(device="cpu", dtype=torch.float64)
    values = values.clamp_min(torch.finfo(torch.float64).tiny)
    values /= values.sum(dim=1, keepdim=True)
    scores: list[torch.Tensor] = []
    for split in values.chunk(splits):
        marginal = split.mean(dim=0)
        divergence = (split * (split.log() - marginal.log())).sum(dim=1)
        scores.append(divergence.mean().exp())
    result = torch.stack(scores)
    return InceptionScore(
        mean=float(result.mean().item()),
        std=float(result.std(unbiased=False).item()),
        splits=splits,
        sample_count=values.shape[0],
    )


__all__ = [
    "FeatureStats",
    "InceptionScore",
    "frechet_inception_distance",
    "inception_score",
]
