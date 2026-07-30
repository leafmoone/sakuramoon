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
    devices = {gradient.device for gradient in gradients}
    if len(devices) != 1:
        raise ValueError("all gradients must share one device")
    device = next(iter(devices))
    squared_norm = torch.zeros((), device=device, dtype=torch.float32)
    for gradient in gradients:
        if gradient.is_sparse:
            raise RuntimeError("sparse gradients are not supported")
        squared_norm.add_(gradient.float().square().sum())
    pre_clip_norm = squared_norm.sqrt()
    if not bool(torch.isfinite(pre_clip_norm).item()):
        raise FloatingPointError("global gradient norm is nonfinite")
    coefficient = (max_norm / pre_clip_norm.clamp_min(1e-12)).clamp_max(1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(gradient.dtype))
    return ClipResult(
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=pre_clip_norm * coefficient,
        coefficient=coefficient,
    )


__all__ = ["ClipResult", "clip_grad_norm_fp32"]
