"""Stable-slot dense DiT reference model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sakuramoon.conditioning.global_condition import (
    GlobalConditioner,
    GlobalConditionOutput,
)
from sakuramoon.conditioning.modality import ModalityEmbedding
from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.model.attention import dense_attention_mask
from sakuramoon.model.block import DiTBlock
from sakuramoon.model.growth import active_slot_ids, slot_growth, slot_name
from sakuramoon.model.output_head import FinalOutputHead


@dataclass(frozen=True)
class DenseDiTFeatures:
    joint_hidden: torch.Tensor
    token_mask: torch.Tensor
    image_start: int
    image_shape: tuple[int, int]
    condition: GlobalConditionOutput


class DenseDiT(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        input_channels: int,
        hidden_size: int,
        intermediate_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        rope_nope_dim: int,
        rope_y_dim: int,
        rope_x_dim: int,
        rope_position_scale: float,
        rope_theta: float,
        norm_eps: float,
        timestep_dim: int,
        size_dim: int,
        aspect_dim: int,
        condition_hidden_size: int,
        stable_slot_count: int,
        modulation_chunks: int,
        final_modulation_size: int,
        out_channels: int,
        modality_init_std: float,
        linear_dtype: torch.dtype,
        sensitive_dtype: torch.dtype,
        projection_bias: bool,
        attention_dropout: float,
        mlp_dropout: float,
        output_weight_zero_init: bool,
        output_bias_zero_init: bool,
    ) -> None:
        super().__init__()
        slots = active_slot_ids(depth)
        if input_channels != out_channels or stable_slot_count != 24:
            raise ValueError("DiT requires 128-channel latent I/O and 24 stable slots")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if sensitive_dtype != torch.float32:
            raise ValueError("sensitive DiT parameters must use float32")
        self.depth = depth
        self.hidden_size = hidden_size
        self.active_slot_ids = slots
        self.input_projection = nn.Linear(
            input_channels,
            hidden_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )
        self.modality = ModalityEmbedding(hidden_size, modality_init_std)
        self.conditioner = GlobalConditioner(
            timestep_dim=timestep_dim,
            size_dim=size_dim,
            aspect_dim=aspect_dim,
            hidden_dim=condition_hidden_size,
            model_dim=hidden_size,
            slot_count=stable_slot_count,
            active_slot_ids=slots,
            modulation_chunks=modulation_chunks,
            final_modulation_size=final_modulation_size,
        )
        self.blocks = nn.ModuleDict(
            {
                slot_name(slot_id): DiTBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    rope_nope_dim=rope_nope_dim,
                    rope_y_dim=rope_y_dim,
                    rope_x_dim=rope_x_dim,
                    rope_position_scale=rope_position_scale,
                    rope_theta=rope_theta,
                    norm_eps=norm_eps,
                    linear_dtype=linear_dtype,
                    projection_bias=projection_bias,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                )
                for slot_id in slots
            }
        )
        self.output_head = FinalOutputHead(
            hidden_size=hidden_size,
            out_channels=out_channels,
            norm_eps=norm_eps,
            projection_dtype=sensitive_dtype,
            weight_zero_init=output_weight_zero_init,
            bias_zero_init=output_bias_zero_init,
        )

    def _prepare_tokens(
        self,
        latent: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, int]]:
        if latent.ndim != 4 or latent.shape[1] != self.input_projection.in_features:
            raise ValueError("latent must have shape [B,input_channels,H,W]")
        batch, _, height, width = latent.shape
        if text_tokens.ndim != 3 or text_tokens.shape[0] != batch:
            raise ValueError("text_tokens must have shape [B,L,hidden_size]")
        if text_tokens.shape[-1] != self.hidden_size:
            raise ValueError("text token width differs from the DiT hidden size")
        if text_mask.shape != text_tokens.shape[:2] or text_mask.dtype != torch.bool:
            raise ValueError("text_mask must be boolean with shape [B,L]")
        if style_tokens.shape != (batch, 4, self.hidden_size):
            raise ValueError("style_tokens must have shape [B,4,hidden_size]")
        if (
            latent.dtype != self.input_projection.weight.dtype
            or text_tokens.dtype != latent.dtype
            or style_tokens.dtype != latent.dtype
        ):
            raise ValueError("latent, text, style, and DiT linear weights must share dtype")
        image_tokens = latent.permute(0, 2, 3, 1).reshape(
            batch,
            height * width,
            latent.shape[1],
        )
        image_tokens = self.input_projection(image_tokens)
        text_tokens = self.modality(text_tokens, "text")
        style_tokens = self.modality(style_tokens, "style")
        image_tokens = self.modality(image_tokens, "image")
        joint = torch.cat((text_tokens, style_tokens, image_tokens), dim=1)
        style_mask = torch.ones(batch, 4, device=latent.device, dtype=torch.bool)
        image_mask = torch.ones(
            batch,
            height * width,
            device=latent.device,
            dtype=torch.bool,
        )
        token_mask = torch.cat((text_mask, style_mask, image_mask), dim=1)
        coordinates = torch.zeros(
            batch,
            joint.shape[1],
            2,
            device=latent.device,
            dtype=torch.float32,
        )
        image_start = text_tokens.shape[1] + 4
        coordinates[:, image_start:] = image_coordinates(
            height,
            width,
            device=latent.device,
        )
        return joint, token_mask, coordinates, image_start, (height, width)

    def forward_features(
        self,
        latent: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> DenseDiTFeatures:
        joint, token_mask, coordinates, image_start, image_shape = self._prepare_tokens(
            latent,
            text_tokens,
            text_mask,
            style_tokens,
        )
        condition = self.conditioner(
            timestep,
            size_scale,
            aspect,
            self.active_slot_ids,
        )
        attention_mask = dense_attention_mask(token_mask)
        for active_index, slot_id in enumerate(self.active_slot_ids):
            growth = slot_growth(self.depth, slot_id, growth_alpha)
            joint = self.blocks[slot_name(slot_id)](
                joint,
                token_mask,
                attention_mask,
                coordinates,
                condition.block.for_active_index(active_index),
                attention_growth=growth,
                mlp_growth=growth,
            )
        return DenseDiTFeatures(
            joint_hidden=joint,
            token_mask=token_mask,
            image_start=image_start,
            image_shape=image_shape,
            condition=condition,
        )

    def forward(
        self,
        latent: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> torch.Tensor:
        features = self.forward_features(
            latent,
            text_tokens,
            text_mask,
            style_tokens,
            timestep,
            size_scale,
            aspect,
            growth_alpha=growth_alpha,
        )
        height, width = features.image_shape
        image_tokens = height * width
        image_hidden = features.joint_hidden[
            :,
            features.image_start : features.image_start + image_tokens,
        ]
        return self.output_head(
            image_hidden,
            features.condition.final_scale,
            features.condition.final_shift,
            features.image_shape,
        )

    def model_metadata(self) -> dict[str, int | str]:
        return {
            "prediction_type": "x",
            "out_channels": self.output_head.out_channels,
            "depth": self.depth,
            "stable_slot_count": 24,
        }


__all__ = ["DenseDiT", "DenseDiTFeatures"]
