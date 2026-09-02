from __future__ import annotations

import torch
from torch import nn

from sakuramoon.model.irepa import (
    IREPA_TEACHER_FEATURE_WIDTH,
    IRepaAlignment,
    irepa_alignment_metadata,
)
from sakuramoon.model.mixed_precision_conv import MixedPrecisionConv2d
from sakuramoon.optim.groups import audit_trainable_parameters

INPUT_WIDTH = 2560


def _alignment() -> IRepaAlignment:
    return IRepaAlignment(INPUT_WIDTH)


def _backward(output: torch.Tensor) -> None:
    output.float().sum().backward()


def test_parameter_dtypes_are_locked() -> None:
    irepa = _alignment()
    assert isinstance(irepa.projector, MixedPrecisionConv2d)
    assert irepa.projector.weight.dtype is torch.bfloat16
    assert irepa.projector.bias is not None
    assert irepa.projector.bias.dtype is torch.float32
    assert irepa.projector.weight.requires_grad
    assert irepa.projector.bias.requires_grad
    assert irepa.projector.out_channels == IREPA_TEACHER_FEATURE_WIDTH


def test_forward_square_grid_matches_conv2d_reference() -> None:
    torch.manual_seed(7)
    irepa = _alignment()
    height, width = 16, 16
    image_hidden = torch.randn(2, height * width, INPUT_WIDTH, dtype=torch.bfloat16)

    output = irepa(image_hidden, (height, width))
    expected = irepa.projector(image_hidden.reshape(2, INPUT_WIDTH, height, width))

    assert output.shape == (2, height * width, IREPA_TEACHER_FEATURE_WIDTH)
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output.float()).all()
    # token order is row-major H*W and the forward is exactly the conv
    flat = expected.reshape(2, IREPA_TEACHER_FEATURE_WIDTH, -1).transpose(1, 2)
    assert torch.equal(output, flat)


def test_forward_non_square_grid() -> None:
    torch.manual_seed(11)
    irepa = _alignment()
    height, width = 9, 14
    image_hidden = torch.randn(1, height * width, INPUT_WIDTH, dtype=torch.bfloat16)

    output = irepa(image_hidden, (height, width))

    assert output.shape == (1, height * width, IREPA_TEACHER_FEATURE_WIDTH)
    assert output.dtype is torch.bfloat16
    _backward(output)
    assert irepa.projector.weight.grad is not None


def test_wrong_token_grid_mismatch_fails() -> None:
    irepa = _alignment()
    image_hidden = torch.randn(1, 100, INPUT_WIDTH, dtype=torch.bfloat16)

    try:
        irepa(image_hidden, (16, 16))
    except ValueError as error:
        assert "do not cover the image grid" in str(error)
    else:
        raise AssertionError("expected ValueError for H*W != T")


def test_wrong_hidden_width_fails() -> None:
    irepa = _alignment()
    image_hidden = torch.randn(1, 256, INPUT_WIDTH + 1, dtype=torch.bfloat16)

    try:
        irepa(image_hidden, (16, 16))
    except ValueError as error:
        assert "does not match the projector input width" in str(error)
    else:
        raise AssertionError("expected ValueError for D != in_channels")


def test_wrong_dtype_fails() -> None:
    irepa = _alignment()
    image_hidden = torch.randn(1, 256, INPUT_WIDTH, dtype=torch.float32)

    try:
        irepa(image_hidden, (16, 16))
    except TypeError as error:
        assert "bfloat16" in str(error)
    else:
        raise AssertionError("expected TypeError for non-bf16 input")


def test_backward_grads_finite_and_bias_stays_fp32() -> None:
    torch.manual_seed(13)
    irepa = _alignment()
    image_hidden = torch.randn(2, 64, INPUT_WIDTH, dtype=torch.bfloat16)

    output = irepa(image_hidden, (8, 8))
    _backward(output)

    weight_grad = irepa.projector.weight.grad
    bias_grad = irepa.projector.bias
    assert weight_grad is not None
    assert torch.isfinite(weight_grad.float()).all()
    assert bias_grad is not None and bias_grad.grad is not None
    assert torch.isfinite(bias_grad.grad).all()
    assert bias_grad.grad.dtype is torch.float32
    # the bias body itself is never recast by the forward
    assert irepa.projector.bias.dtype is torch.float32


def test_parameter_audit_groups_are_correct() -> None:
    module = nn.Module()
    module.irepa = _alignment()

    audit = audit_trainable_parameters(
        module,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
    )
    groups = {spec.name: spec.group for spec in audit.specs}
    dtypes = {spec.name: spec.parameter.dtype for spec in audit.specs}

    assert groups["irepa.projector.weight"] == "matrix_decay"
    assert groups["irepa.projector.bias"] == "sensitive_no_decay"
    assert dtypes["irepa.projector.weight"] is torch.bfloat16
    assert dtypes["irepa.projector.bias"] is torch.float32


def test_locked_v1_contract_rejects_other_shapes() -> None:
    for kwargs in (
        {"out_channels": 512},
        {"kernel_size": 5},
        {"stride": 2},
        {"padding": 0},
        {"dilation": 2},
        {"groups": 2},
        {"bias": False},
    ):
        try:
            IRepaAlignment(INPUT_WIDTH, **kwargs)
        except ValueError as error:
            assert "locked" in str(error)
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_artifact_config_matches_canonical_metadata() -> None:
    irepa = _alignment()

    assert irepa.artifact_config() == irepa_alignment_metadata(INPUT_WIDTH)
    metadata = irepa.artifact_config()
    assert metadata["class"] == "IRepaAlignment"
    assert metadata["schema_version"] == 1
    assert metadata["in_channels"] == INPUT_WIDTH
    assert metadata["out_channels"] == IREPA_TEACHER_FEATURE_WIDTH
    assert metadata["weight_dtype"] == "bfloat16"
    assert metadata["bias_dtype"] == "float32"
