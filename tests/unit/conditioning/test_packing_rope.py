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
from sakuramoon.conditioning.rope import (
    QKRoPE2D,
    full_canvas_crop_coordinates,
    image_coordinates,
    packed_coordinates,
)


def _packed():
    text = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    text_mask = torch.tensor([[True, True, False], [True, False, False]])
    condition = torch.full((2, 8, 8), -2.0)
    images = (torch.full((4, 8), 3.0), torch.full((2, 8), 4.0))
    return pack_sequences(
        text,
        text_mask,
        (2, 1),
        condition,
        8,
        images,
        ((2, 2), (1, 2)),
    )


def test_modality_embeddings_are_separate() -> None:
    module = ModalityEmbedding(hidden_size=8, init_std=0.02)
    tokens = torch.zeros(2, 8)

    text = module(tokens, "text")
    condition = module(tokens, "condition")
    image = module(tokens, "image")

    assert not torch.equal(text, condition)
    assert not torch.equal(condition, image)


def test_pack_order_cu_seqlens_and_sample_isolation() -> None:
    packed = _packed()

    assert torch.equal(packed.cu_seqlens, torch.tensor([0, 14, 25], dtype=torch.int32))
    assert packed.max_seqlen == 14
    assert packed.spans[0].text.start == 0
    assert packed.spans[0].condition.start == 2
    assert packed.spans[0].image.start == 10
    mask = dense_reference_mask(packed)
    assert mask[:14, :14].all()
    assert mask[14:, 14:].all()
    assert not mask[:14, 14:].any()
    assert not mask[14:, :14].any()


@pytest.mark.parametrize("lengths", [(), (0,), (-1,), (True,), (2, 1.5)])
def test_boundary_factory_rejects_invalid_host_lengths(
    lengths: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        build_validated_cu_seqlens(
            cast(tuple[int, ...], lengths),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("field", ["condition_dtype", "mask_device"])
def test_pack_rejects_cross_input_dtype_or_device(field: str) -> None:
    text = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    condition = torch.zeros(1, 8, 8)
    if field == "condition_dtype":
        condition = condition.to(torch.float64)
    else:
        mask = torch.ones(1, 2, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="share one"):
        pack_sequences(
            text,
            mask,
            (2,),
            condition,
            8,
            (torch.zeros(1, 8),),
            ((1, 1),),
        )


def test_pack_rejects_mask_that_disagrees_with_host_lengths() -> None:
    with pytest.raises(ValueError, match="contiguous prefix"):
        pack_sequences(
            torch.zeros(1, 3, 8),
            torch.tensor([[True, False, True]]),
            (2,),
            torch.zeros(1, 8, 8),
            8,
            (torch.zeros(1, 8),),
            ((1, 1),),
        )


@pytest.mark.parametrize("lengths", [(), (0,), (4,), (True,)])
def test_pack_rejects_invalid_text_lengths(lengths: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="text_lengths"):
        pack_sequences(
            torch.zeros(1, 3, 8),
            torch.tensor([[True, True, False]]),
            lengths,
            torch.zeros(1, 8, 8),
            8,
            (torch.zeros(1, 8),),
            ((1, 1),),
        )


def test_area_normalized_coordinates_and_text_condition_zero_coordinates() -> None:
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
    coordinates = packed_coordinates(packed, (square, wide))
    assert (
        torch.count_nonzero(
            coordinates[
                packed.spans[0].text.start : packed.spans[0].condition.end
            ]
        )
        == 0
    )
    torch.testing.assert_close(coordinates[packed.spans[0].image.start : packed.spans[0].image.end], square)


def test_crop_coordinates_use_the_resized_full_canvas_frame() -> None:
    device = torch.device("cpu")
    full_crop = full_canvas_crop_coordinates(
        2,
        4,
        full_height=4,
        full_width=8,
        crop_box=(0, 0, 8, 4),
        device=device,
    )
    torch.testing.assert_close(full_crop, image_coordinates(2, 4, device=device))

    shifted_crop = full_canvas_crop_coordinates(
        2,
        4,
        full_height=8,
        full_width=12,
        crop_box=(4, 2, 12, 6),
        device=device,
    )
    expected_y = torch.tensor((-0.25, 0.25)) * math.sqrt(2.0 / 3.0)
    expected_x = torch.tensor((-1.0 / 6.0, 1.0 / 6.0, 0.5, 5.0 / 6.0)) * math.sqrt(
        3.0 / 2.0
    )
    grid_y, grid_x = torch.meshgrid(expected_y, expected_x, indexing="ij")
    torch.testing.assert_close(
        shifted_crop,
        torch.stack((grid_y.flatten(), grid_x.flatten()), dim=-1),
    )


@pytest.mark.parametrize(
    ("crop_box", "error"),
    [
        ((0, 0, 9, 4), "contained"),
        ((0, 0, 7, 4), "divide exactly"),
        ((0, 0, 8, 2), "equal integer pixel stride"),
    ],
)
def test_full_canvas_coordinates_fail_fast_on_invalid_geometry(
    crop_box: tuple[int, int, int, int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        full_canvas_crop_coordinates(
            2,
            2 if crop_box == (0, 0, 8, 2) else 4,
            full_height=4,
            full_width=8,
            crop_box=crop_box,
            device=torch.device("cpu"),
        )


def test_packed_coordinates_require_one_exact_map_per_sample() -> None:
    packed = _packed()
    square = image_coordinates(2, 2, device=torch.device("cpu"))
    wide = image_coordinates(1, 2, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="every packed sample"):
        packed_coordinates(packed, (square,))
    with pytest.raises(ValueError, match="token-grid shape"):
        packed_coordinates(packed, (square[:3], wide))


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
