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


class FeatureStatsAccumulator:
    """Bounded CPU float64 covariance aggregation for long evaluator jobs."""

    def __init__(self) -> None:
        self._count = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return self._count

    def update(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must have shape [N,D] with N >= 1")
        values = features.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("features must be finite")
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if self._mean is None or self._m2 is None:
            self._count = batch_count
            self._mean = batch_mean
            self._m2 = batch_m2
            return
        if batch_mean.shape != self._mean.shape:
            raise ValueError("feature dimension changed during aggregation")
        combined_count = self._count + batch_count
        delta = batch_mean - self._mean
        self._m2 = (
            self._m2
            + batch_m2
            + torch.outer(delta, delta)
            * (self._count * batch_count / combined_count)
        )
        self._mean = self._mean + delta * (batch_count / combined_count)
        self._count = combined_count

    def finalize(self) -> FeatureStats:
        if self._count < 2 or self._mean is None or self._m2 is None:
            raise ValueError("feature stats require at least two observations")
        return FeatureStats(
            count=self._count,
            mean=self._mean,
            covariance=self._m2 / (self._count - 1),
        )


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


class InceptionScoreAccumulator:
    """Exact-split streaming IS aggregation without retaining all predictions."""

    def __init__(self, *, sample_count: int, splits: int) -> None:
        if (
            type(sample_count) is not int
            or sample_count < 2
            or type(splits) is not int
            or splits <= 0
            or sample_count % splits
        ):
            raise ValueError("IS sample count must be divisible into positive splits")
        self.sample_count = sample_count
        self.splits = splits
        self._split_size = sample_count // splits
        self._observed = 0
        self._sum_probabilities: torch.Tensor | None = None
        self._sum_p_log_p: torch.Tensor | None = None
        self._counts = torch.zeros(splits, dtype=torch.int64)

    def update(self, probabilities: torch.Tensor) -> None:
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] == 0
            or probabilities.shape[1] < 2
        ):
            raise ValueError("IS probabilities must have shape [N,C]")
        values = probabilities.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()) or bool(
            (values < 0.0).any().item()
        ):
            raise ValueError("IS probabilities must be finite and nonnegative")
        if not torch.allclose(
            values.sum(dim=1),
            torch.ones(values.shape[0], dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("IS probability rows must sum to one")
        if self._observed + values.shape[0] > self.sample_count:
            raise ValueError("IS received more probabilities than declared")
        if self._sum_probabilities is None or self._sum_p_log_p is None:
            self._sum_probabilities = torch.zeros(
                self.splits, values.shape[1], dtype=torch.float64
            )
            self._sum_p_log_p = torch.zeros(self.splits, dtype=torch.float64)
        elif self._sum_probabilities.shape[1] != values.shape[1]:
            raise ValueError("IS class dimension changed during aggregation")

        offset = 0
        while offset < values.shape[0]:
            split_index = self._observed // self._split_size
            split_offset = self._observed % self._split_size
            take = min(values.shape[0] - offset, self._split_size - split_offset)
            part = values[offset : offset + take]
            positive = part > 0.0
            self._sum_probabilities[split_index] += part.sum(dim=0)
            self._sum_p_log_p[split_index] += torch.where(
                positive, part * part.clamp_min(torch.finfo(torch.float64).tiny).log(), 0.0
            ).sum()
            self._counts[split_index] += take
            self._observed += take
            offset += take

    def finalize(self) -> InceptionScore:
        if (
            self._observed != self.sample_count
            or self._sum_probabilities is None
            or self._sum_p_log_p is None
            or not bool((self._counts == self._split_size).all().item())
        ):
            raise ValueError("IS aggregation is incomplete")
        scores: list[torch.Tensor] = []
        for split_index in range(self.splits):
            summed = self._sum_probabilities[split_index]
            marginal = summed / self._split_size
            positive = summed > 0.0
            cross_entropy_term = torch.where(
                positive,
                summed * marginal.clamp_min(torch.finfo(torch.float64).tiny).log(),
                0.0,
            ).sum()
            mean_kl = (
                self._sum_p_log_p[split_index] - cross_entropy_term
            ) / self._split_size
            scores.append(mean_kl.exp())
        score_tensor = torch.stack(scores)
        return InceptionScore(
            mean=float(score_tensor.mean().item()),
            std=float(score_tensor.std(unbiased=False).item()),
            splits=self.splits,
            sample_count=self.sample_count,
        )


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
    "FeatureStatsAccumulator",
    "InceptionScore",
    "InceptionScoreAccumulator",
    "frechet_inception_distance",
    "inception_score",
]
