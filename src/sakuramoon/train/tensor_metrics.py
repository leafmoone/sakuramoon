"""Shared tensor metric primitives for condition diagnostics.

Pure, torch-only helpers shared by the sampling-time diagnostics
(``condition_diagnostics``) and the per-update condition diagnostics
stream (``condition_diagnostics_stream``).  No heavy imports: this module
must stay importable on CPU-only test environments.
"""

from __future__ import annotations

import math

import torch


class TrainingSamplingError(RuntimeError):
    """A periodic training sample or condition diagnostic is invalid."""


def tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise TrainingSamplingError(
            "diagnostic tensor must be nonempty floating point"
        )
    return tensor.float().square().mean().sqrt()


def tensor_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.numel() == 0:
        raise TrainingSamplingError("diagnostic cosine tensors must match")
    left = first.float().flatten()
    right = second.float().flatten()
    denominator = torch.sqrt(left.square().sum()) * torch.sqrt(right.square().sum())
    if bool((denominator <= 1e-12).any()):
        raise TrainingSamplingError("diagnostic cosine is undefined at zero norm")
    return torch.dot(left, right) / denominator


def metric_float(name: str, value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise TrainingSamplingError(f"condition diagnostic is nonfinite: {name}")
    return result


__all__ = [
    "TrainingSamplingError",
    "metric_float",
    "tensor_cosine",
    "tensor_rms",
]
