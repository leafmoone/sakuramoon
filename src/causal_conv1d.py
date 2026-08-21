"""HCU compatibility shim for Transformers Qwen3.5 causal-conv1d API.

This module intentionally delegates to flash-linear-attention (FLA).  The
installed FLA build contains the Triton-Ascend implementation; no CUDA
causal-conv1d wheel is used.  The FLA import is deferred to first use so
that importing this shim (e.g. through a Transformers model import on a
machine without FLA) does not itself require the FLA package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_fla_causal_conv1d: Callable[..., Any] | None = None
_fla_causal_conv1d_update: Callable[..., Any] | None = None


def _require_fla() -> tuple[Callable[..., Any], Callable[..., Any]]:
    global _fla_causal_conv1d, _fla_causal_conv1d_update
    if _fla_causal_conv1d is None:
        from fla.modules.conv import causal_conv1d as fla_causal_conv1d
        from fla.modules.conv.triton import (
            causal_conv1d_update as fla_causal_conv1d_update,
        )

        _fla_causal_conv1d = fla_causal_conv1d
        _fla_causal_conv1d_update = fla_causal_conv1d_update
    return _fla_causal_conv1d, _fla_causal_conv1d_update


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Apply FLA causal convolution through the Transformers API layout.

    Transformers/causal-conv1d uses [B, D, T], while FLA uses [B, T, D].
    """
    if seq_idx is not None:
        raise NotImplementedError("FLA HCU causal convolution does not support seq_idx")
    if x.ndim != 3:
        raise ValueError(f"expected [B, D, T], got {tuple(x.shape)}")
    causal_conv1d, _ = _require_fla()
    x_btd = x.transpose(1, 2)
    y_btd, _ = causal_conv1d(
        x=x_btd,
        weight=weight,
        bias=bias,
        activation=activation,
        backend="triton",
        **kwargs,
    )
    return y_btd.transpose(1, 2)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """Apply FLA HCU recurrent causal-conv update in-place on conv_state."""
    if x.ndim != 3:
        raise ValueError(f"expected [B, D, T], got {tuple(x.shape)}")
    _, causal_conv1d_update = _require_fla()
    x_btd = x.transpose(1, 2)
    y_btd, _ = causal_conv1d_update(
        x=x_btd,
        cache=conv_state,
        weight=weight,
        bias=bias,
        activation=activation,
        **kwargs,
    )
    return y_btd.transpose(1, 2)
