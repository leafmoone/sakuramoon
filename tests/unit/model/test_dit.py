from __future__ import annotations

import torch

from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.model.growth import active_slot_ids, new_slot_ids, slot_growth
from sakuramoon.model.output_head import FinalOutputHead


def _model(depth: int) -> DenseDiT:
    return DenseDiT(
        depth=depth,
        input_channels=8,
        hidden_size=8,
        intermediate_size=16,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        rope_nope_dim=0,
        rope_y_dim=2,
        rope_x_dim=2,
        rope_position_scale=1.0,
        rope_theta=10.0,
        norm_eps=1e-6,
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        condition_hidden_size=1024,
        stable_slot_count=24,
        modulation_chunks=6,
        final_modulation_size=16,
        out_channels=8,
        modality_init_std=0.02,
        linear_dtype=torch.float32,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    )


def _production_kwargs() -> dict[str, object]:
    return {
        "depth": 16,
        "input_channels": 128,
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
        "timestep_dim": 256,
        "size_dim": 64,
        "aspect_dim": 64,
        "condition_hidden_size": 1024,
        "stable_slot_count": 24,
        "modulation_chunks": 6,
        "final_modulation_size": 5120,
        "out_channels": 128,
        "modality_init_std": 0.02,
        "linear_dtype": torch.bfloat16,
        "sensitive_dtype": torch.float32,
        "projection_bias": False,
        "attention_dropout": 0.0,
        "mlp_dropout": 0.0,
        "output_weight_zero_init": True,
        "output_bias_zero_init": True,
    }


def _inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(1, 8, 2, 2),
        torch.randn(1, 3, 8),
        torch.tensor([[True, True, False]]),
        torch.randn(1, 4, 8),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )


def test_stable_slot_sets_insert_four_blocks_at_each_growth() -> None:
    assert active_slot_ids(16) == (
        0,
        1,
        3,
        4,
        6,
        7,
        9,
        10,
        12,
        13,
        15,
        16,
        18,
        19,
        21,
        22,
    )
    assert new_slot_ids(20) == (2, 8, 14, 20)
    assert new_slot_ids(24) == (5, 11, 17, 23)
    assert active_slot_ids(24) == tuple(range(24))
    assert slot_growth(20, 2, 0.25) == 0.25
    assert slot_growth(20, 3, 0.25) == 1.0


def test_old_block_fqns_are_stable_across_depths() -> None:
    model16 = _model(16)
    model20 = _model(20)
    names16 = {
        name for name, _ in model16.named_parameters() if name.startswith("blocks.")
    }
    names20 = {
        name for name, _ in model20.named_parameters() if name.startswith("blocks.")
    }

    assert names16 < names20
    assert all(
        any(name.startswith(f"blocks.slot_{slot:02d}.") for name in names20)
        for slot in new_slot_ids(20)
    )
    parameters16 = dict(model16.named_parameters())
    parameters20 = dict(model20.named_parameters())
    assert "conditioner.block_biases.slot_02" not in parameters16
    assert "conditioner.block_biases.slot_02" in parameters20


def test_alpha_zero_new_slots_preserve_old_hidden_function() -> None:
    model16 = _model(16)
    model20 = _model(20)
    source = model16.state_dict()
    target = model20.state_dict()
    with torch.no_grad():
        for name, tensor in source.items():
            if name in target and target[name].shape == tensor.shape:
                target[name].copy_(tensor)
    model20.load_state_dict(target)
    inputs = _inputs()

    features16 = model16.forward_features(*inputs, growth_alpha=1.0)
    features20 = model20.forward_features(*inputs, growth_alpha=0.0)

    torch.testing.assert_close(features16.joint_hidden, features20.joint_hidden)


def test_output_head_is_zero_initialized_and_image_shaped() -> None:
    head = FinalOutputHead(
        hidden_size=8,
        out_channels=4,
        norm_eps=1e-6,
        projection_dtype=torch.float32,
        weight_zero_init=True,
        bias_zero_init=True,
    )
    output = head(
        torch.randn(2, 6, 8, dtype=torch.bfloat16),
        torch.randn(2, 8),
        torch.randn(2, 8),
        (2, 3),
    )

    assert output.shape == (2, 4, 2, 3)
    assert output.dtype == torch.float32
    assert torch.count_nonzero(output) == 0


def test_output_head_restores_heterogeneous_packed_image_spans() -> None:
    head = FinalOutputHead(
        hidden_size=8,
        out_channels=4,
        norm_eps=1e-6,
        projection_dtype=torch.float32,
        weight_zero_init=True,
        bias_zero_init=True,
    )
    with torch.no_grad():
        head.projection.weight.fill_(0.25)
        head.projection.bias.fill_(0.5)
    image_hidden = torch.randn(10, 8, dtype=torch.bfloat16)
    sample_indices = torch.tensor([0] * 4 + [1] * 6, dtype=torch.int64)

    output = head.forward_packed(
        image_hidden,
        sample_indices,
        torch.zeros(2, 8),
        torch.zeros(2, 8),
        ((2, 2), (2, 3)),
    )

    assert tuple(sample.shape for sample in output) == ((4, 2, 2), (4, 2, 3))
    expected = head.projection(head.norm(image_hidden).float())
    torch.testing.assert_close(output[0].flatten(1).transpose(0, 1), expected[:4])
    torch.testing.assert_close(output[1].flatten(1).transpose(0, 1), expected[4:])


def test_dense_dit_predicts_only_latent_shape_and_records_metadata() -> None:
    model = _model(16)
    output = model(*_inputs(), growth_alpha=1.0)

    assert output.shape == (1, 8, 2, 2)
    assert torch.count_nonzero(output) == 0
    assert model.model_metadata() == {
        "prediction_type": "x",
        "out_channels": 8,
        "depth": 16,
        "stable_slot_count": 24,
    }


def test_packed_and_dense_production_state_dicts_are_isomorphic() -> None:
    with torch.device("meta"):
        dense = DenseDiT(**_production_kwargs())  # pyright: ignore[reportArgumentType]
        packed = PackedDiT(**_production_kwargs())  # pyright: ignore[reportArgumentType]

    dense_state = dense.state_dict()
    packed_state = packed.state_dict()
    assert dense_state.keys() == packed_state.keys()
    assert {
        name: (tensor.shape, tensor.dtype) for name, tensor in dense_state.items()
    } == {name: (tensor.shape, tensor.dtype) for name, tensor in packed_state.items()}
    assert packed.model_metadata()["attention_backend"] == "fa4_varlen"
