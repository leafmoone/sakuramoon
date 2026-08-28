from __future__ import annotations

import torch

from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.model.dit import DenseDiT


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
        condition_token_count=8,
        modality_init_std=0.02,
        linear_dtype=torch.float32,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    )


def test_alpha_zero_is_exactly_function_preserving_for_old_slots() -> None:
    torch.manual_seed(2031)
    source = _model(16)
    target = _model(20)
    target_state = target.state_dict()
    for name, tensor in source.state_dict().items():
        target_state[name].copy_(tensor)
    target.load_state_dict(target_state)
    latent = torch.randn(1, 8, 2, 2)
    text = torch.randn(1, 3, 8)
    text_mask = torch.tensor([[True, True, False]])
    condition = torch.randn(1, 8, 8)
    active = torch.tensor([True])
    timestep = torch.tensor([0.5])
    size_scale = torch.tensor([0.0])
    aspect = torch.tensor([0.0])
    coords = image_coordinates(2, 2, device=torch.device("cpu")).unsqueeze(0)

    source_features = source.forward_features(
        latent,
        text,
        text_mask,
        condition,
        active,
        timestep,
        size_scale,
        aspect,
        image_coordinates=coords,
        growth_alpha=1.0,
    )
    target_features = target.forward_features(
        latent,
        text,
        text_mask,
        condition,
        active,
        timestep,
        size_scale,
        aspect,
        image_coordinates=coords,
        growth_alpha=0.0,
    )

    assert torch.equal(source_features.joint_hidden, target_features.joint_hidden)
