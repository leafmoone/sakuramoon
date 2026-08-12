"""Strict distribution metrics for image-generation evaluation."""

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
    centered_features: torch.Tensor | None = None

    @classmethod
    def from_features(
        cls,
        features: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> FeatureStats:
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("FID features must have shape [N,D] with N >= 2")
        target = features.device if device is None else device
        values = features.detach().to(device=target, dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("FID features contain nonfinite values")
        mean = values.mean(dim=0)
        centered = values - mean
        covariance = centered.T @ centered / (values.shape[0] - 1)
        # The sample-space FID identity is cheaper only while N <= D. Retaining
        # centered [N,D] features for large evaluations would make the later
        # eigendecomposition N x N (50k x 50k for FID-50k).
        retained = centered if values.shape[0] <= values.shape[1] else None
        return cls(values.shape[0], mean, covariance, retained)


def frechet_inception_distance(
    generated: FeatureStats,
    real: FeatureStats,
    *,
    device: torch.device | None = None,
) -> float:
    if generated.mean.shape != real.mean.shape:
        raise ValueError("generated and real feature dimensions differ")
    target = generated.mean.device if device is None else device
    generated_mean = generated.mean.to(device=target, dtype=torch.float64)
    generated_covariance = generated.covariance.to(
        device=target,
        dtype=torch.float64,
    )
    real_mean = real.mean.to(device=target, dtype=torch.float64)
    real_covariance = real.covariance.to(device=target, dtype=torch.float64)
    centered = generated.centered_features
    if centered is None:
        eigenvalues, eigenvectors = _eigh(generated_covariance)
        generated_root = (
            eigenvectors
            @ torch.diag(eigenvalues.clamp_min(0.0).sqrt())
            @ eigenvectors.T
        )
        middle = generated_root @ real_covariance @ generated_root
    else:
        if centered.shape != (generated.count, generated_mean.shape[0]):
            raise ValueError("generated centered features have an invalid shape")
        values = centered.to(device=target, dtype=torch.float64)
        middle = values @ real_covariance @ values.T / (generated.count - 1)
    covariance_trace = _eigvalsh((middle + middle.T) * 0.5).clamp_min(0.0).sqrt().sum()
    difference = generated_mean - real_mean
    value = (
        difference.square().sum()
        + torch.trace(generated_covariance)
        + torch.trace(real_covariance)
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


@dataclass(frozen=True, slots=True)
class KernelDistance:
    mean: float
    std: float
    subsets: int
    subset_size: int


def _feature_matrix(
    name: str,
    features: torch.Tensor,
    *,
    device: torch.device | None,
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(f"{name} features must have shape [N,D] with N >= 2")
    target = features.device if device is None else device
    values = features.detach().to(device=target, dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} features contain nonfinite values")
    return values


def kernel_inception_distance(
    generated_features: torch.Tensor,
    real_features: torch.Tensor,
    *,
    subsets: int,
    subset_size: int,
    seed: int,
    device: torch.device | None = None,
) -> KernelDistance:
    """Return unbiased cubic-polynomial MMD over deterministic random subsets."""

    generated = _feature_matrix("generated KID", generated_features, device=device)
    real = _feature_matrix("real KID", real_features, device=device)
    if generated.shape[1] != real.shape[1]:
        raise ValueError("generated and real KID feature dimensions differ")
    if type(subsets) is not int or subsets <= 0:
        raise ValueError("KID subset count must be a positive integer")
    if (
        type(subset_size) is not int
        or subset_size < 2
        or subset_size > min(generated.shape[0], real.shape[0])
    ):
        raise ValueError("KID subset size is outside the available sample range")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("KID seed must be a 63-bit nonnegative integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    dimension = float(generated.shape[1])
    estimates: list[torch.Tensor] = []
    for _ in range(subsets):
        generated_indices = torch.randperm(
            generated.shape[0], generator=generator, device="cpu"
        )[:subset_size].to(generated.device)
        real_indices = torch.randperm(real.shape[0], generator=generator, device="cpu")[
            :subset_size
        ].to(real.device)
        generated_subset = generated.index_select(0, generated_indices)
        real_subset = real.index_select(0, real_indices)
        generated_kernel = (
            generated_subset @ generated_subset.T / dimension + 1.0
        ) ** 3
        real_kernel = (real_subset @ real_subset.T / dimension + 1.0) ** 3
        cross_kernel = (generated_subset @ real_subset.T / dimension + 1.0) ** 3
        denominator = subset_size * (subset_size - 1)
        estimate = (
            (generated_kernel.sum() - generated_kernel.diagonal().sum()) / denominator
            + (real_kernel.sum() - real_kernel.diagonal().sum()) / denominator
            - 2.0 * cross_kernel.mean()
        )
        estimates.append(estimate)
    values = torch.stack(estimates)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    if not math.isfinite(mean) or not math.isfinite(std):
        raise FloatingPointError("KID is nonfinite")
    return KernelDistance(mean, std, subsets, subset_size)


CMMD_SIGMA = 10.0
CMMD_SCALE = 1000.0


def _rbf_kernel_sum(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    sigma: float,
    block_size: int,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=first.device)
    gamma = 1.0 / (2.0 * sigma * sigma)
    for first_start in range(0, first.shape[0], block_size):
        first_block = first[first_start : first_start + block_size]
        first_norm = first_block.square().sum(dim=1, keepdim=True)
        for second_start in range(0, second.shape[0], block_size):
            second_block = second[second_start : second_start + block_size]
            second_norm = second_block.square().sum(dim=1).unsqueeze(0)
            distances = (
                first_norm + second_norm - 2.0 * (first_block @ second_block.T)
            ).clamp_min(0.0)
            total += torch.exp(-gamma * distances).sum()
    return total


def clip_maximum_mean_discrepancy(
    generated_features: torch.Tensor,
    real_features: torch.Tensor,
    *,
    device: torch.device | None = None,
    block_size: int = 1024,
) -> float:
    """Compute official CMMD: biased Gaussian-RBF MMD, sigma 10, scale 1000."""

    generated = _feature_matrix("generated CMMD", generated_features, device=device)
    real = _feature_matrix("real CMMD", real_features, device=device)
    if generated.shape[1] != real.shape[1]:
        raise ValueError("generated and real CMMD feature dimensions differ")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("CMMD block size must be a positive integer")
    for name, values in (("generated", generated), ("real", real)):
        norms = torch.linalg.vector_norm(values, dim=1)
        if not bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
        ):
            raise ValueError(f"{name} CMMD CLIP features must be L2-normalized")

    generated_sum = _rbf_kernel_sum(
        generated, generated, sigma=CMMD_SIGMA, block_size=block_size
    )
    real_sum = _rbf_kernel_sum(real, real, sigma=CMMD_SIGMA, block_size=block_size)
    cross_sum = _rbf_kernel_sum(
        generated, real, sigma=CMMD_SIGMA, block_size=block_size
    )
    value = CMMD_SCALE * (
        generated_sum / (generated.shape[0] ** 2)
        + real_sum / (real.shape[0] ** 2)
        - 2.0 * cross_sum / (generated.shape[0] * real.shape[0])
    )
    result = max(0.0, float(value.item()))
    if not math.isfinite(result):
        raise FloatingPointError("CMMD is nonfinite")
    return result


def inception_score(
    probabilities: torch.Tensor,
    *,
    splits: int,
    device: torch.device | None = None,
) -> InceptionScore:
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] < 2
        or probabilities.shape[1] < 2
        or type(splits) is not int
        or splits <= 0
        or probabilities.shape[0] % splits
    ):
        raise ValueError("IS probabilities or split count are invalid")
    target = probabilities.device if device is None else device
    values = probabilities.detach().to(device=target, dtype=torch.float64)
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
    "CMMD_SCALE",
    "CMMD_SIGMA",
    "FeatureStats",
    "InceptionScore",
    "KernelDistance",
    "clip_maximum_mean_discrepancy",
    "frechet_inception_distance",
    "inception_score",
    "kernel_inception_distance",
]
