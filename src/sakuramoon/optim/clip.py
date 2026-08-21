"""FP32 global gradient norm and clipping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ClipResult:
    pre_clip_norm: torch.Tensor
    post_clip_norm: torch.Tensor
    coefficient: torch.Tensor


def clip_grad_norm_fp32(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float,
) -> ClipResult:
    if max_norm != 1.0:
        raise ValueError("global gradient clip norm must equal 1.0")
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("cannot clip an empty gradient set")
    for gradient in gradients:
        if gradient.is_sparse:
            raise RuntimeError("sparse gradients are not supported")
    devices = {gradient.device for gradient in gradients}
    if len(devices) != 1:
        raise ValueError("all gradients must share one device")
    # One fused L2 norm per gradient (fp32 accumulation inside the kernel,
    # no separate fp32 cast pass) plus a single sum of squares, instead of
    # per-parameter cast+square+sum launches. The association order differs
    # from the historical per-parameter fp32 accumulator by rounding only
    # (relative ~1e-7), far below the clip decision boundary at norm 1.0.
    norms = [
        torch.linalg.vector_norm(gradient, 2, dtype=torch.float32)
        for gradient in gradients
    ]
    pre_clip_norm = torch.stack(norms).square().sum().sqrt()
    if not bool(torch.isfinite(pre_clip_norm).item()):
        raise FloatingPointError("global gradient norm is nonfinite")
    coefficient = (max_norm / pre_clip_norm.clamp_min(1e-12)).clamp_max(1.0)
    dtypes = {gradient.dtype for gradient in gradients}
    if len(dtypes) == 1:
        torch._foreach_mul_(gradients, coefficient.to(next(iter(dtypes))))
    else:
        torch._foreach_mul_(
            gradients,
            [coefficient.to(gradient.dtype) for gradient in gradients],
        )
    return ClipResult(
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=pre_clip_norm * coefficient,
        coefficient=coefficient,
    )


__all__ = ["ClipResult", "clip_grad_norm_fp32"]
