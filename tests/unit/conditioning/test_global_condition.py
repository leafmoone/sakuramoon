from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.global_condition import (
    GlobalConditioner,
    fixed_sinusoidal_embedding,
)
from sakuramoon.conditioning.packing import canvas_condition


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
        norm_eps=1e-6,
    )


def _condition_inputs(
    batch: int,
    *,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randn(batch, 8, 32, dtype=torch.float32)
    active_mask = (
        torch.ones(batch, dtype=torch.bool) if active is None else active
    )
    return tokens, active_mask


def test_fixed_embedding_zero_and_shape() -> None:
    embedded = fixed_sinusoidal_embedding(torch.zeros(2, dtype=torch.float32), 64)

    assert embedded.shape == (2, 64)
    assert torch.equal(embedded[:, :32], torch.ones(2, 32))
    assert torch.equal(embedded[:, 32:], torch.zeros(2, 32))


def test_fixed_embedding_nonzero_frequency_golden() -> None:
    embedded = fixed_sinusoidal_embedding(
        torch.tensor([1.0, -1.0], dtype=torch.float32),
        64,
    )
    selected = embedded[:, (0, 1, 31, 32, 33, 63)]
    expected = torch.tensor(
        [
            [
                0.54030231,
                0.73176098,
                0.99999999,
                0.84147098,
                0.68156135,
                0.00013335214,
            ],
            [
                0.54030231,
                0.73176098,
                0.99999999,
                -0.84147098,
                -0.68156135,
                -0.00013335214,
            ],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(selected, expected, atol=1e-7, rtol=1e-6)


def test_target_canvas_condition_integrates_with_cfg_shared_inputs() -> None:
    module = _module()
    size_scale, aspect = canvas_condition(height_px=512, width_px=1024)
    timestep = torch.tensor([0.375, 0.375], dtype=torch.float32)
    cfg_size_scale = torch.full((2,), size_scale, dtype=torch.float32)
    cfg_aspect = torch.full((2,), aspect, dtype=torch.float32)
    captured_inputs: list[torch.Tensor] = []

    def capture_condition_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured_inputs.append(inputs[0].detach().clone())

    hook = module.condition_mlp.register_forward_pre_hook(capture_condition_input)
    condition_tokens, condition_active = _condition_inputs(2)
    try:
        output = module(
            timestep,
            cfg_size_scale,
            cfg_aspect,
            condition_tokens,
            condition_active,
            (0, 3),
        )
    finally:
        hook.remove()

    assert size_scale == 0.5
    assert aspect == 1.0
    assert len(captured_inputs) == 1
    embedded = captured_inputs[0]
    assert embedded.shape == (2, 384)
    torch.testing.assert_close(
        embedded[:, (0, 128, 256, 288, 320, 352)],
        torch.tensor(
            [
                [
                    0.93050762,
                    0.36627253,
                    0.87758256,
                    0.47942554,
                    0.54030231,
                    0.84147098,
                ],
                [
                    0.93050762,
                    0.36627253,
                    0.87758256,
                    0.47942554,
                    0.54030231,
                    0.84147098,
                ],
            ],
            dtype=torch.float32,
        ),
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        output.base_hidden[0],
        output.base_hidden[1],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(output.condition_residual, torch.zeros_like(output.condition_residual))
    torch.testing.assert_close(output.total_hidden, output.base_hidden, atol=0, rtol=0)


def test_modulation_paths_start_zero_and_are_independent() -> None:
    module = _module()
    output = module(
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        torch.tensor([0.0, 0.5], dtype=torch.float32),
        torch.tensor([0.0, -1.0], dtype=torch.float32),
        *_condition_inputs(2),
        (0, 3, 7),
    )

    assert output.base_hidden.shape == (2, 1024)
    assert output.condition_residual.shape == (2, 1024)
    assert output.total_hidden.shape == (2, 1024)
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


def test_block_bias_chunks_follow_locked_modulation_order() -> None:
    module = _module()
    with torch.no_grad():
        values = torch.arange(1, 7, dtype=torch.float32).repeat_interleave(32)
        module.block_biases["slot_03"].copy_(values)
    output = module(
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        *_condition_inputs(1),
        (3,),
    )

    for expected, actual in enumerate(
        (
            output.block.attention_scale,
            output.block.attention_shift,
            output.block.attention_gate,
            output.block.mlp_scale,
            output.block.mlp_shift,
            output.block.mlp_gate,
        ),
        start=1,
    ):
        torch.testing.assert_close(actual, torch.full_like(actual, expected))


def test_condition_parameters_and_outputs_remain_fp32_under_autocast() -> None:
    module = _module()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.25], dtype=torch.float32),
            torch.tensor([-0.5], dtype=torch.float32),
            *_condition_inputs(1),
            (0, 3),
        )

    assert {parameter.dtype for parameter in module.parameters()} == {torch.float32}
    assert output.base_hidden.dtype == torch.float32
    assert output.condition_residual.dtype == torch.float32
    assert output.total_hidden.dtype == torch.float32
    assert output.block.attention_scale.dtype == torch.float32
    assert output.final_scale.dtype == torch.float32


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
        *_condition_inputs(1),
        (0,),
    )
    expected = torch.nn.functional.silu(output.total_hidden).sum(dim=-1)

    torch.testing.assert_close(output.final_scale[:, 0], expected)
    torch.testing.assert_close(output.final_shift[:, 0], expected)


def test_zero_initialized_outputs_receive_gradients() -> None:
    module = _module()
    output = module(
        torch.tensor([0.25], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        *_condition_inputs(1),
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
        *_condition_inputs(1),
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


def test_rejects_invalid_timestep_metadata_and_slot() -> None:
    module = _module()
    with pytest.raises(ValueError, match="timestep"):
        module(
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            *_condition_inputs(1),
            (0,),
        )
    with pytest.raises(ValueError, match="slot"):
        module(
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            *_condition_inputs(1),
            (24,),
        )


def test_global_condition_residual_masks_null_samples_exactly() -> None:
    module = _module()
    with torch.no_grad():
        module.condition_global_projection.weight.fill_(0.25)
    tokens, active = _condition_inputs(
        2,
        active=torch.tensor([True, False], dtype=torch.bool),
    )

    output = module(
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        tokens,
        active,
        (0,),
    )

    assert torch.count_nonzero(output.condition_residual[0]) > 0
    assert torch.count_nonzero(output.condition_residual[1]) == 0
    torch.testing.assert_close(
        output.total_hidden[1], output.base_hidden[1], atol=0, rtol=0
    )


def test_zero_initialized_global_projection_receives_gradient() -> None:
    module = _module()
    tokens, active = _condition_inputs(1)
    output = module(
        torch.tensor([0.25], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        tokens,
        active,
        (2,),
    )
    with torch.no_grad():
        module.shared_block_projection.weight.fill_(0.01)
    output = module(
        torch.tensor([0.25], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        tokens,
        active,
        (2,),
    )
    output.block.attention_scale.sum().backward()

    gradient = module.condition_global_projection.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0
