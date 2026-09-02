from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sakuramoon.model.mixed_precision_conv import MixedPrecisionConv2d


def _backward(output: torch.Tensor) -> None:
    # Consume in FP32, as the production loss path does, so gradients
    # return to each parameter at its own precision.
    output.float().sum().backward()


def test_forward_keeps_weight_bf16_and_bias_fp32() -> None:
    torch.manual_seed(7)
    conv = MixedPrecisionConv2d(16, 32, 3, padding=1)

    assert conv.weight.dtype is torch.bfloat16
    assert conv.weight.requires_grad
    assert conv.bias is not None and conv.bias.dtype is torch.float32
    assert conv.bias.requires_grad

    inputs = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    output = conv(inputs)

    assert output.dtype is torch.bfloat16
    assert output.shape == (2, 32, 8, 8)
    # The bias body is never replaced or permanently cast.
    assert conv.bias.dtype is torch.float32
    assert conv.weight.dtype is torch.bfloat16


def test_forward_matches_f_conv2d_reference() -> None:
    torch.manual_seed(11)
    conv = MixedPrecisionConv2d(16, 32, 3, padding=1, stride=1, dilation=1)
    inputs = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    assert conv.bias is not None

    expected = F.conv2d(
        inputs,
        conv.weight,
        conv.bias.to(torch.bfloat16),
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )

    # The forward is exactly F.conv2d with the bias cast to the input
    # dtype, so the outputs must be bit-identical.
    assert torch.equal(conv(inputs), expected)


def test_non_bf16_input_is_rejected() -> None:
    conv = MixedPrecisionConv2d(16, 16, 3)

    with pytest.raises(TypeError, match="bfloat16"):
        conv(torch.randn(1, 16, 4, 4, dtype=torch.float32))


def test_backward_gradients_return_to_parameter_precisions() -> None:
    torch.manual_seed(13)
    conv = MixedPrecisionConv2d(16, 32, 3, padding=1)
    inputs = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)

    _backward(conv(inputs))

    assert conv.weight.grad is not None
    assert bool(torch.isfinite(conv.weight.grad.float()).all())
    assert conv.bias is not None and conv.bias.grad is not None
    assert conv.bias.grad.dtype is torch.float32
    assert bool(torch.isfinite(conv.bias.grad).all())
    # Parameter bodies keep their contract dtype after backward.
    assert conv.weight.dtype is torch.bfloat16
    assert conv.bias.dtype is torch.float32


def test_bias_false_forward_and_backward() -> None:
    torch.manual_seed(17)
    conv = MixedPrecisionConv2d(16, 32, 3, bias=False)

    assert conv.bias is None
    inputs = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)
    output = conv(inputs)

    assert output.dtype is torch.bfloat16
    _backward(output)
    assert conv.weight.grad is not None
    assert bool(torch.isfinite(conv.weight.grad.float()).all())


def test_groups_smoke() -> None:
    torch.manual_seed(19)
    conv = MixedPrecisionConv2d(16, 32, 3, groups=2, padding=1)
    inputs = torch.randn(2, 16, 8, 8, dtype=torch.bfloat16)

    output = conv(inputs)

    assert output.shape == (2, 32, 8, 8)
    _backward(output)
    assert conv.weight.grad is not None
    assert conv.bias is not None and conv.bias.grad is not None


def test_nonsquare_spatial_smoke() -> None:
    torch.manual_seed(23)
    conv = MixedPrecisionConv2d(8, 16, 3, padding=1)
    inputs = torch.randn(2, 8, 5, 7, dtype=torch.bfloat16)

    output = conv(inputs)

    assert output.shape == (2, 16, 5, 7)
    _backward(output)
    assert conv.weight.grad is not None


def test_stride_dilation_padding_smoke() -> None:
    torch.manual_seed(29)
    conv = MixedPrecisionConv2d(
        8,
        16,
        3,
        stride=2,
        padding=1,
        dilation=2,
    )
    inputs = torch.randn(1, 8, 9, 9, dtype=torch.bfloat16)

    output = conv(inputs)

    # out = (in + 2*padding - dilation*(kernel-1) - 1) // stride + 1
    #     = (9 + 2 - 2*(3-1) - 1) // 2 + 1 = (6 // 2) + 1 = 4
    assert output.shape == (1, 16, 4, 4)
    _backward(output)
    assert conv.weight.grad is not None


def test_invalid_constructor_arguments_are_rejected() -> None:
    with pytest.raises(TypeError, match="in_channels"):
        MixedPrecisionConv2d(0, 16, 3)
    with pytest.raises(ValueError, match="divisible by groups"):
        MixedPrecisionConv2d(16, 30, 3, groups=8)
    with pytest.raises(ValueError, match="positive"):
        MixedPrecisionConv2d(16, 16, 3, stride=0)
    with pytest.raises(ValueError, match="non-negative"):
        MixedPrecisionConv2d(16, 16, 3, padding=-1)
    with pytest.raises(ValueError, match="dilation"):
        MixedPrecisionConv2d(16, 16, 3, dilation=0)
    with pytest.raises(TypeError, match="kernel_size"):
        MixedPrecisionConv2d(16, 16, (3,))  # pyright: ignore[reportArgumentType]


def test_module_is_a_standard_conv2d_subclass() -> None:
    # It is a genuine nn.Conv2d (standard semantics + audit role), so the
    # canonical parameter audit recognizes the weight as a convolution
    # matrix without any special-casing.
    conv = MixedPrecisionConv2d(16, 16, 3)
    assert isinstance(conv, nn.Conv2d)
    assert isinstance(conv, nn.Module)
    named = dict(conv.named_parameters())
    assert set(named) == {"weight", "bias"}
