"""Structural and contract tests for FrozenPESpatialEncoder (no weights).

The real-weight forward/geometry/reference tests live in
tests/gpu/irepa/ (they require the fingerprint-bound local asset and an
HCU).  Here the module is constructed with random initialization and cast
to the same BF16/frozen state the asset loader produces.
"""

from __future__ import annotations

import pytest
import torch

from sakuramoon.assets.pe_spatial import APPROVED_PE_SPATIAL_B16_512
from sakuramoon.encoders.pe_spatial import (
    TEACHER_FEATURE_WIDTH,
    TEACHER_NAME,
    TEACHER_PATCH_SIZE,
    FrozenPESpatialEncoder,
)
from sakuramoon.pe_spatial.config import PE_VISION_CONFIG


def _frozen_cpu_teacher() -> FrozenPESpatialEncoder:
    encoder = FrozenPESpatialEncoder()
    encoder.visual.to(  # pyright: ignore[reportUnknownMemberType]
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def test_module_is_frozen_eval_and_grad_free() -> None:
    encoder = _frozen_cpu_teacher()

    assert encoder.training is False
    assert encoder.visual.training is False  # pyright: ignore[reportUnknownMemberType]
    assert encoder.parameters() is not None
    assert all(
        parameter.requires_grad is False for parameter in encoder.parameters()
    )


def test_train_mode_is_forbidden() -> None:
    encoder = _frozen_cpu_teacher()

    with pytest.raises(RuntimeError, match="never enter train"):
        encoder.train()
    assert encoder.training is False
    assert encoder.train(False).training is False


def test_teacher_is_locked_to_approved_b16_512_architecture() -> None:
    encoder = _frozen_cpu_teacher()
    visual = encoder.visual
    assert visual.patch_size == TEACHER_PATCH_SIZE == 16  # pyright: ignore[reportUnknownMemberType]
    assert visual.width == TEACHER_FEATURE_WIDTH == 768  # pyright: ignore[reportUnknownMemberType]
    assert visual.layers == APPROVED_PE_SPATIAL_B16_512.depth == 12  # pyright: ignore[reportUnknownMemberType]
    assert visual.use_cls_token is True  # pyright: ignore[reportUnknownMemberType]
    assert visual.pool_type == "none"  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(visual.ln_post, torch.nn.Identity)  # pyright: ignore[reportUnknownMemberType]
    assert visual.use_rope2d is True  # pyright: ignore[reportUnknownMemberType]
    assert TEACHER_NAME in PE_VISION_CONFIG


def test_parameters_are_bfloat16() -> None:
    encoder = _frozen_cpu_teacher()

    assert encoder.dtype is torch.bfloat16
    assert all(
        parameter.dtype is torch.bfloat16 for parameter in encoder.parameters()
    )


def test_input_must_be_tensor() -> None:
    encoder = _frozen_cpu_teacher()

    with pytest.raises(TypeError, match="torch.Tensor"):
        encoder(object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="torch.Tensor"):
        encoder.forward("not a tensor")  # pyright: ignore[reportArgumentType]


def test_input_rank_and_channels_rejected() -> None:
    encoder = _frozen_cpu_teacher()

    with pytest.raises(ValueError, match="\\[B,3,H,W\\]"):
        encoder(torch.empty(2, 3, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="3 channels"):
        encoder(torch.empty(2, 4, 64, 64, dtype=torch.bfloat16))


def test_input_dtype_rejected() -> None:
    encoder = _frozen_cpu_teacher()

    with pytest.raises(TypeError, match="bfloat16"):
        encoder(torch.zeros(2, 3, 64, 64, dtype=torch.float32))


def test_non_divisible_dimensions_rejected() -> None:
    encoder = _frozen_cpu_teacher()

    with pytest.raises(ValueError, match="divisible by 16"):
        encoder(torch.zeros(2, 3, 84, 92, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="divisible by 16"):
        encoder(torch.zeros(2, 3, 92, 84, dtype=torch.bfloat16))


def test_output_dtype_contract_helper() -> None:
    # The output dtype contract (bf16 patch features) is exercised with the
    # real asset on HCU in tests/gpu/irepa; here we only pin the constant.
    assert TEACHER_FEATURE_WIDTH == 768
