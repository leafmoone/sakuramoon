from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.global_condition import (
    GlobalConditioner,
    fixed_sinusoidal_embedding,
)


def _module() -> GlobalConditioner:
    return GlobalConditioner(
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        hidden_dim=1024,
        model_dim=32,
        slot_count=24,
        active_slot_ids=tuple(range(24)),
        modulation_chunks=6,
        final_modulation_size=64,
    )


def test_fixed_embedding_zero_and_shape() -> None:
    embedded = fixed_sinusoidal_embedding(torch.zeros(2, dtype=torch.float32), 64)

    assert embedded.shape == (2, 64)
    assert torch.equal(embedded[:, :32], torch.ones(2, 32))
    assert torch.equal(embedded[:, 32:], torch.zeros(2, 32))


def test_modulation_paths_start_zero_and_are_independent() -> None:
    module = _module()
    output = module(
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        torch.tensor([0.0, 0.5], dtype=torch.float32),
        torch.tensor([0.0, -1.0], dtype=torch.float32),
        (0, 3, 7),
    )

    assert output.condition_hidden.shape == (2, 1024)
    for tensor in (
        output.block.attention_scale,
        output.block.attention_shift,
        output.block.attention_gate,
        output.block.mlp_scale,
        output.block.mlp_shift,
        output.block.mlp_gate,
    ):
        assert tensor.shape == (2, 3, 32)
        assert torch.count_nonzero(tensor) == 0
    assert torch.count_nonzero(output.final_scale) == 0
    assert torch.count_nonzero(output.final_shift) == 0
    assert (
        module.final_projection.weight.data_ptr()
        != module.shared_block_projection.weight.data_ptr()
    )


def test_final_path_applies_silu_and_does_not_reuse_block_projection() -> None:
    module = _module()
    with torch.no_grad():
        module.final_projection.weight.fill_(1.0)
        module.final_projection.bias.zero_()
        module.shared_block_projection.weight.fill_(7.0)
    output = module(
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        (0,),
    )
    expected = torch.nn.functional.silu(output.condition_hidden).sum(dim=-1)

    torch.testing.assert_close(output.final_scale[:, 0], expected)
    torch.testing.assert_close(output.final_shift[:, 0], expected)


def test_zero_initialized_outputs_receive_gradients() -> None:
    module = _module()
    output = module(
        torch.tensor([0.25], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        (2, 5),
    )
    loss = (
        output.block.attention_gate.sum()
        + output.block.mlp_gate.sum()
        + output.final_scale.sum()
        + output.final_shift.sum()
    )
    loss.backward()

    assert module.shared_block_projection.weight.grad is not None
    assert module.block_biases["slot_02"].grad is not None
    assert module.final_projection.weight.grad is not None
    assert torch.isfinite(module.shared_block_projection.weight.grad).all()


def test_active_slot_selection_removes_only_the_slot_axis() -> None:
    module = _module()
    output = module(
        torch.tensor([0.25], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        (2, 5),
    )

    selected = output.block.for_active_index(1)

    assert selected.attention_scale.shape == (1, 32)
    torch.testing.assert_close(
        selected.mlp_gate,
        output.block.mlp_gate[:, 1],
    )
    with pytest.raises(IndexError, match="selected slots"):
        output.block.for_active_index(2)


def test_rejects_invalid_timestep_and_slot() -> None:
    module = _module()
    with pytest.raises(ValueError, match="timestep"):
        module(
            torch.tensor([1.1], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            (0,),
        )
    with pytest.raises(ValueError, match="slot"):
        module(
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            (24,),
        )
