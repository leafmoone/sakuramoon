"""FP32 gradient norms and finite-checked global or independent clipping."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ClipResult:
    pre_clip_norm: torch.Tensor
    post_clip_norm: torch.Tensor
    coefficient: torch.Tensor


def _validate_max_norm(max_norm: float) -> None:
    if (
        type(max_norm) not in (int, float)
        or not math.isfinite(max_norm)
        or max_norm < 0.0
    ):
        raise ValueError("gradient clip norm must be finite and nonnegative")


def _prepare_clip(
    gradients: list[torch.Tensor], *, max_norm: float, device: torch.device
) -> ClipResult:
    for gradient in gradients:
        if gradient.is_sparse:
            raise RuntimeError("sparse gradients are not supported")
    if any(gradient.device != device for gradient in gradients):
        raise ValueError("all gradients must share one device")
    # Preserve the legacy norm reduction at max_norm=1.
    norms = [
        torch.linalg.vector_norm(gradient, 2, dtype=torch.float32)
        for gradient in gradients
    ]
    pre_clip_norm = (
        torch.stack(norms).square().sum().sqrt()
        if norms
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    if not bool(torch.isfinite(pre_clip_norm).item()):
        raise FloatingPointError("global gradient norm is nonfinite")
    if max_norm > 0.0:
        # Use the actual positive norm, not an epsilon floor: thresholds may
        # legitimately be smaller than that floor. Zero/absent gradients need
        # no scaling and must not be reported as clipped, even at tiny limits.
        positive = pre_clip_norm > 0.0
        denominator = torch.where(positive, pre_clip_norm, torch.ones_like(pre_clip_norm))
        coefficient = torch.where(
            positive,
            (max_norm / denominator).clamp_max(1.0),
            torch.ones_like(pre_clip_norm),
        )
    else:
        coefficient = torch.ones_like(pre_clip_norm)
    return ClipResult(
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=pre_clip_norm * coefficient,
        coefficient=coefficient,
    )


def _apply_clip(gradients: list[torch.Tensor], coefficient: torch.Tensor) -> None:
    if not gradients:
        return
    dtypes = {gradient.dtype for gradient in gradients}
    if len(dtypes) == 1:
        torch._foreach_mul_(gradients, coefficient.to(next(iter(dtypes))))
    else:
        torch._foreach_mul_(
            gradients,
            [coefficient.to(gradient.dtype) for gradient in gradients],
        )


def clip_grad_norm_fp32(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float,
) -> ClipResult:
    """Clip one nonempty gradient set; zero disables scaling, not finite checks."""
    _validate_max_norm(max_norm)
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("cannot clip an empty gradient set")
    result = _prepare_clip(gradients, max_norm=max_norm, device=gradients[0].device)
    if max_norm > 0.0:
        _apply_clip(gradients, result.coefficient)
    return result


def clip_grad_norm_groups_fp32(
    condition_parameters: Iterable[nn.Parameter],
    rest_parameters: Iterable[nn.Parameter],
    *,
    condition_max_norm: float,
    rest_max_norm: float,
) -> tuple[ClipResult, ClipResult]:
    """Check BOTH partitions before scaling either; absent gradients are zero."""
    _validate_max_norm(condition_max_norm)
    _validate_max_norm(rest_max_norm)
    condition = tuple(condition_parameters)
    rest = tuple(rest_parameters)
    parameters = condition + rest
    if not parameters:
        raise RuntimeError("cannot clip empty parameter partitions")
    condition_gradients = [p.grad for p in condition if p.grad is not None]
    rest_gradients = [p.grad for p in rest if p.grad is not None]
    device = parameters[0].device
    condition_result = _prepare_clip(
        condition_gradients, max_norm=condition_max_norm, device=device
    )
    rest_result = _prepare_clip(rest_gradients, max_norm=rest_max_norm, device=device)
    if condition_max_norm > 0.0:
        _apply_clip(condition_gradients, condition_result.coefficient)
    if rest_max_norm > 0.0:
        _apply_clip(rest_gradients, rest_result.coefficient)
    return condition_result, rest_result


__all__ = ["ClipResult", "clip_grad_norm_fp32", "clip_grad_norm_groups_fp32"]
