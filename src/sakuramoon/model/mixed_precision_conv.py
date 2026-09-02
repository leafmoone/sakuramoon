"""Mixed-precision 2D convolution: BF16 weight with an FP32 bias."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _as_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if type(value) is int:
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = value
        if type(first) is not int or type(second) is not int:
            raise TypeError(f"{name} values must be ints, got {value!r}")
        return first, second
    raise TypeError(f"{name} must be an int or a pair of ints, got {value!r}")


class MixedPrecisionConv2d(nn.Conv2d):
    """Conv2d that keeps its weight trainable in BF16 and its bias in FP32.

    This is a genuine ``nn.Conv2d`` (standard convolution semantics, init
    and validation included); only the parameter precisions differ from the
    default module:

    - ``weight`` is created and initialized in BF16, so it participates in
      the standard matrix-decay (BF16) parameter policy;
    - ``bias`` is an FP32 parameter. The body is never replaced or
      permanently cast: in :meth:`forward` a temporary copy is cast to the
      input/weight dtype for the ``F.conv2d`` call, and autograd returns
      the bias gradient to the FP32 parameter.

    Constructor arguments are validated fail-closed before delegation to
    ``nn.Conv2d`` so the module has deterministic, documented errors
    independent of torch version behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("groups", groups),
        ):
            if type(value) is not int or value <= 0:
                raise TypeError(
                    f"{name} must be a positive int, got {value!r}"
                )
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"in/out channels must be divisible by groups ({groups})"
            )
        kh, kw = _as_pair(kernel_size, "kernel_size")
        sh, sw = _as_pair(stride, "stride")
        ph, pw = _as_pair(padding, "padding")
        dh, dw = _as_pair(dilation, "dilation")
        if min(kh, kw, sh, sw) <= 0:
            raise ValueError("kernel_size and stride must be positive")
        if min(ph, pw) < 0:
            raise ValueError("padding must be non-negative")
        if min(dh, dw) <= 0:
            raise ValueError("dilation must be positive")

        # dtype=torch.bfloat16: the weight is created and kaiming-initialized
        # in BF16 by nn.Conv2d. The bias is re-materialized in FP32 below.
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            dtype=torch.bfloat16,
        )
        if self.bias is not None:
            fan_in = (
                (in_channels // groups)
                * self.kernel_size[0]
                * self.kernel_size[1]
            )
            limit = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            self.bias = nn.Parameter(
                torch.empty(out_channels, dtype=torch.float32).uniform_(
                    -limit, limit
                )
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype is not torch.bfloat16:
            raise TypeError(
                "MixedPrecisionConv2d requires a bfloat16 input, "
                f"got {input.dtype}"
            )
        # Temporary cast only: the FP32 bias body is never modified.
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.conv2d(
            input,
            self.weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + ", weight_dtype=bfloat16, bias_dtype=float32"
        )


__all__ = ["MixedPrecisionConv2d"]
