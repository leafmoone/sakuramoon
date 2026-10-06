"""Qwen3.5 causal convolution: installed FLA kernels or a PyTorch reference.

The source-tree shim is also discovered on NVIDIA machines. It must provide
a working reference when FLA is absent, rather than advertising a callable
that fails only after the text model has been loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import torch
import torch.nn.functional as F

_fla_causal_conv1d: Callable[..., Any] | None = None
_fla_causal_conv1d_update: Callable[..., Any] | None = None


def _activate(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return x
    if activation in {"silu", "swish"}:
        return F.silu(x)
    raise ValueError("causal convolution supports only silu/swish activation")


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
    if find_spec("fla") is None:
        if kwargs:
            raise NotImplementedError(
                "PyTorch causal convolution does not support extra state arguments"
            )
        if weight.ndim != 2 or weight.shape[0] != x.shape[1]:
            raise ValueError("weight must have shape [D, kernel_width]")
        output = F.conv1d(
            x,
            weight.unsqueeze(1),
            bias,
            padding=weight.shape[1] - 1,
            groups=x.shape[1],
        )[..., : x.shape[-1]]
        return _activate(output, activation)
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
    if find_spec("fla") is None:
        if kwargs:
            raise NotImplementedError(
                "PyTorch convolution update does not support extra state arguments"
            )
        squeeze = x.ndim == 2
        sequence = x.unsqueeze(-1) if squeeze else x
        if (
            sequence.ndim != 3
            or weight.ndim != 2
            or sequence.shape[1] != weight.shape[0]
        ):
            raise ValueError("expected [B,D] or [B,D,T] input and [D,W] weights")
        expected = (sequence.shape[0], sequence.shape[1], weight.shape[1])
        if conv_state.shape != expected:
            raise ValueError("conv_state must have shape [B,D,kernel_width]")
        history = torch.cat((conv_state, sequence), dim=-1)
        output = F.conv1d(history, weight.unsqueeze(1), bias, groups=sequence.shape[1])[
            ..., 1:
        ]
        conv_state.copy_(history[..., -weight.shape[1] :])
        output = _activate(output, activation)
        return output.squeeze(-1) if squeeze else output
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
