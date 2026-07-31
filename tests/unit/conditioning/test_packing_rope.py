from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from sakuramoon.conditioning.modality import ModalityEmbedding
from sakuramoon.conditioning.packing import (
    build_validated_cu_seqlens,
    canvas_condition,
    dense_reference_mask,
    pack_sequences,
)
from sakuramoon.conditioning.rope import QKRoPE2D, image_coordinates, packed_coordinates


def _packed():
    text = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    text_mask = torch.tensor([[True, False, True], [True, False, False]])
    style = torch.full((2, 4, 8), -2.0)
    images = (torch.full((4, 8), 3.0), torch.full((2, 8), 4.0))
    return pack_sequences(text, text_mask, style, images, ((2, 2), (1, 2)))


def test_modality_embeddings_are_separate() -> None:
    module = ModalityEmbedding(hidden_size=8, init_std=0.02)
    tokens = torch.zeros(2, 8)

    text = module(tokens, "text")
    style = module(tokens, "style")
    image = module(tokens, "image")

    assert not torch.equal(text, style)
    assert not torch.equal(style, image)


def test_pack_order_cu_seqlens_and_sample_isolation() -> None:
    packed = _packed()

    assert torch.equal(packed.cu_seqlens, torch.tensor([0, 10, 17], dtype=torch.int32))
    assert packed.max_seqlen == 10
    assert packed.spans[0].text.start == 0
    assert packed.spans[0].style.start == 2
    assert packed.spans[0].image.start == 6
    mask = dense_reference_mask(packed)
    assert mask[:10, :10].all()
    assert mask[10:, 10:].all()
    assert not mask[:10, 10:].any()
    assert not mask[10:, :10].any()


@pytest.mark.parametrize("lengths", [(), (0,), (-1,), (True,), (2, 1.5)])
def test_boundary_factory_rejects_invalid_host_lengths(
    lengths: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        build_validated_cu_seqlens(
            cast(tuple[int, ...], lengths),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("field", ["style_dtype", "mask_device"])
def test_pack_rejects_cross_input_dtype_or_device(field: str) -> None:
    text = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    style = torch.zeros(1, 4, 8)
    if field == "style_dtype":
        style = style.to(torch.float64)
    else:
        mask = torch.ones(1, 2, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="share one"):
        pack_sequences(
            text,
            mask,
            style,
            (torch.zeros(1, 8),),
            ((1, 1),),
        )


def test_area_normalized_coordinates_and_text_style_zero_coordinates() -> None:
    square = image_coordinates(2, 2, device=torch.device("cpu"))
    expected = torch.tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
    )
    torch.testing.assert_close(square, expected)

    wide = image_coordinates(1, 2, device=torch.device("cpu"))
    torch.testing.assert_close(
        wide,
        torch.tensor([[0.0, -math.sqrt(2.0) / 2.0], [0.0, math.sqrt(2.0) / 2.0]]),
    )
    packed = _packed()
    coordinates = packed_coordinates(packed)
    assert torch.count_nonzero(coordinates[packed.spans[0].text.start : packed.spans[0].style.end]) == 0
    torch.testing.assert_close(coordinates[packed.spans[0].image.start : packed.spans[0].image.end], square)


def test_qk_norm_precedes_rope_and_kv_heads_are_not_repeated() -> None:
    module = QKRoPE2D(
        head_dim=128,
        nope_dim=32,
        y_dim=48,
        x_dim=48,
        position_scale=16.0,
        theta=1000.0,
        norm_eps=1e-6,
    )
    query = torch.randn(3, 4, 128)
    key = torch.randn(3, 1, 128)
    coordinates = torch.tensor([[0.0, 0.0], [0.5, -0.25], [-0.5, 0.25]])

    query_out, key_out = module(query, key, coordinates)

    assert query_out.shape == query.shape
    assert key_out.shape == key.shape
    torch.testing.assert_close(query_out[..., :32], module.q_norm(query)[..., :32])
    torch.testing.assert_close(key_out[..., :32], module.k_norm(key)[..., :32])
    assert not torch.equal(query_out[1, :, 32:], module.q_norm(query)[1, :, 32:])


@pytest.mark.parametrize("invalid", ["key_dtype", "coordinate_dtype", "device"])
def test_qk_rope_rejects_dtype_or_device_promotion(invalid: str) -> None:
    module = QKRoPE2D(
        head_dim=128,
        nope_dim=32,
        y_dim=48,
        x_dim=48,
        position_scale=16.0,
        theta=1000.0,
        norm_eps=1e-6,
    )
    query = torch.zeros(2, 4, 128)
    key = torch.zeros(2, 1, 128)
    coordinates = torch.zeros(2, 2)
    if invalid == "key_dtype":
        key = key.to(torch.bfloat16)
    elif invalid == "coordinate_dtype":
        coordinates = coordinates.to(torch.float64)
    else:
        coordinates = torch.zeros(2, 2, device="meta")

    with pytest.raises((TypeError, ValueError), match="dtype|float32|device"):
        module(query, key, coordinates)


def test_canvas_condition_uses_only_target_height_and_width() -> None:
    assert canvas_condition(512, 512) == (0.0, 0.0)
    size, aspect = canvas_condition(512, 1024)
    assert size == 0.5
    assert aspect == 1.0
