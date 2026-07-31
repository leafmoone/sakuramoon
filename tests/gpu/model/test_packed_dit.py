from __future__ import annotations

from typing import cast

import pytest
import torch

from sakuramoon.conditioning.global_condition import BlockModulation
from sakuramoon.model.attention import (
    build_validated_cu_seqlens,
    dense_attention_mask,
)
from sakuramoon.model.block import DiTBlock, PackedDiTBlock
from sakuramoon.model.dit import PackedDiT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _block_kwargs() -> dict[str, object]:
    return {
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "q_heads": 20,
        "kv_heads": 5,
        "head_dim": 128,
        "rope_nope_dim": 32,
        "rope_y_dim": 48,
        "rope_x_dim": 48,
        "rope_position_scale": 16.0,
        "rope_theta": 1000.0,
        "norm_eps": 1e-6,
        "linear_dtype": torch.bfloat16,
        "projection_bias": False,
        "attention_dropout": 0.0,
        "mlp_dropout": 0.0,
    }


def _model() -> PackedDiT:
    return PackedDiT(
        depth=16,
        input_channels=128,
        hidden_size=2560,
        intermediate_size=6912,
        q_heads=20,
        kv_heads=5,
        head_dim=128,
        rope_nope_dim=32,
        rope_y_dim=48,
        rope_x_dim=48,
        rope_position_scale=16.0,
        rope_theta=1000.0,
        norm_eps=1e-6,
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        condition_hidden_size=1024,
        stable_slot_count=24,
        modulation_chunks=6,
        final_modulation_size=5120,
        out_channels=128,
        modality_init_std=0.02,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    ).cuda()


def _modulation(batch: int) -> BlockModulation:
    def value(scale: float) -> torch.Tensor:
        return scale * torch.randn(batch, 2560, device="cuda", dtype=torch.float32)

    return BlockModulation(
        attention_scale=value(0.01),
        attention_shift=value(0.01),
        attention_gate=value(0.1),
        mlp_scale=value(0.01),
        mlp_shift=value(0.01),
        mlp_gate=value(0.1),
    )


def test_packed_block_matches_dense_and_isolates_samples() -> None:
    torch.manual_seed(1201)  # pyright: ignore[reportUnknownMemberType]
    dense = DiTBlock(**_block_kwargs()).cuda()  # pyright: ignore[reportArgumentType]
    packed = PackedDiTBlock(**_block_kwargs()).cuda()  # pyright: ignore[reportArgumentType]
    packed.load_state_dict(dense.state_dict())
    lengths = (37, 53)
    maximum = max(lengths)
    dense_tokens = torch.randn(2, maximum, 2560, device="cuda", dtype=torch.bfloat16)
    token_mask = (
        torch.arange(maximum, device="cuda")[None]
        < torch.tensor(lengths, device="cuda")[:, None]
    )
    coordinates = torch.randn(2, maximum, 2, device="cuda", dtype=torch.float32)
    flat_tokens = torch.cat(
        tuple(dense_tokens[index, :length] for index, length in enumerate(lengths))
    )
    flat_coordinates = torch.cat(
        tuple(coordinates[index, :length] for index, length in enumerate(lengths))
    )
    boundaries = build_validated_cu_seqlens(
        lengths,
        device=torch.device("cuda"),
    )
    sample_indices = torch.repeat_interleave(
        torch.arange(2, device="cuda"),
        torch.tensor(lengths, device="cuda"),
        output_size=sum(lengths),
    )
    modulation = _modulation(2)

    dense_output = dense(
        dense_tokens,
        token_mask,
        dense_attention_mask(token_mask),
        coordinates,
        modulation,
        attention_growth=1.0,
        mlp_growth=1.0,
    )
    packed_output = packed(
        flat_tokens,
        boundaries,
        flat_coordinates,
        sample_indices,
        modulation,
        attention_growth=1.0,
        mlp_growth=1.0,
    )
    dense_valid = torch.cat(
        tuple(dense_output[index, :length] for index, length in enumerate(lengths))
    )

    torch.testing.assert_close(packed_output, dense_valid, atol=0.04, rtol=0.01)

    changed = flat_tokens.clone()
    changed[lengths[0] :] = torch.randn_like(changed[lengths[0] :]) * 10
    changed_output = packed(
        changed,
        boundaries,
        flat_coordinates,
        sample_indices,
        modulation,
        attention_growth=1.0,
        mlp_growth=1.0,
    )
    torch.testing.assert_close(
        changed_output[: lengths[0]],
        packed_output[: lengths[0]],
        atol=0,
        rtol=0,
    )


def test_full_packed_dit_three_stage_gradient_startup() -> None:
    torch.manual_seed(1202)  # pyright: ignore[reportUnknownMemberType]
    model = _model()
    latents = (
        torch.randn(128, 2, 2, device="cuda", dtype=torch.bfloat16),
        torch.randn(128, 2, 3, device="cuda", dtype=torch.bfloat16),
    )
    text_tokens = torch.randn(2, 3, 2560, device="cuda", dtype=torch.bfloat16)
    text_mask = torch.tensor(
        [[True, False, True], [True, False, False]],
        device="cuda",
    )
    style_tokens = torch.randn(2, 4, 2560, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([0.25, 0.75], device="cuda", dtype=torch.float32)
    size_scale = torch.zeros(2, device="cuda", dtype=torch.float32)
    aspect = torch.zeros(2, device="cuda", dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def loss() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        predictions = model(
            latents,
            text_tokens,
            text_mask,
            style_tokens,
            timestep,
            size_scale,
            aspect,
            growth_alpha=1.0,
        )
        total = torch.stack(
            tuple(
                (prediction - 1.0).float().square().sum() for prediction in predictions
            )
        ).sum()
        return total, predictions

    first_loss, first_predictions = loss()
    assert tuple(prediction.shape for prediction in first_predictions) == (
        (128, 2, 2),
        (128, 2, 3),
    )
    assert all(torch.count_nonzero(prediction) == 0 for prediction in first_predictions)
    first_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert model.output_head.projection.weight.grad is not None
    assert torch.count_nonzero(model.output_head.projection.weight.grad) > 0
    assert model.input_projection.weight.grad is not None
    assert torch.count_nonzero(model.input_projection.weight.grad) == 0
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.zero_grad(set_to_none=True)  # pyright: ignore[reportUnknownMemberType]

    second_loss, second_predictions = loss()
    assert any(torch.count_nonzero(prediction) > 0 for prediction in second_predictions)
    second_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    shared_grad = model.conditioner.shared_block_projection.weight.grad
    input_grad = model.input_projection.weight.grad
    first_block = cast(PackedDiTBlock, model.blocks["slot_00"])
    block_grad = first_block.attention.q_proj.weight.grad
    assert shared_grad is not None and torch.count_nonzero(shared_grad) > 0
    assert input_grad is not None and torch.count_nonzero(input_grad) > 0
    assert block_grad is not None and torch.count_nonzero(block_grad) == 0
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.zero_grad(set_to_none=True)  # pyright: ignore[reportUnknownMemberType]

    third_loss, _third_predictions = loss()
    third_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    block_grad = first_block.attention.q_proj.weight.grad
    assert block_grad is not None and torch.count_nonzero(block_grad) > 0
