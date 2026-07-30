"""Condition-modulated dense DiT reference block."""

from __future__ import annotations

import torch
from torch import nn

from sakuramoon.conditioning.global_condition import BlockModulation
from sakuramoon.model.attention import DenseGQAAttention
from sakuramoon.model.mlp import SwiGLU
from sakuramoon.model.norm import RMSNorm


class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
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
        linear_dtype: torch.dtype,
        projection_bias: bool,
        attention_dropout: float,
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_norm = RMSNorm(hidden_size, norm_eps)
        self.mlp_norm = RMSNorm(hidden_size, norm_eps)
        self.attention = DenseGQAAttention(
            hidden_size=hidden_size,
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
            dropout=attention_dropout,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            linear_dtype=linear_dtype,
            projection_bias=projection_bias,
            dropout=mlp_dropout,
        )

    def _validate_modulation(
        self,
        modulation: BlockModulation,
        batch: int,
    ) -> None:
        expected = (batch, self.hidden_size)
        tensors = (
            modulation.attention_scale,
            modulation.attention_shift,
            modulation.attention_gate,
            modulation.mlp_scale,
            modulation.mlp_shift,
            modulation.mlp_gate,
        )
        if any(tensor.shape != expected for tensor in tensors):
            raise ValueError("selected block modulation tensors must be [B,D]")

    @staticmethod
    def _modulate(
        normalized: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        scale = scale.to(normalized.dtype).unsqueeze(1)
        shift = shift.to(normalized.dtype).unsqueeze(1)
        return (1.0 + scale) * normalized + shift

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        coordinates: torch.Tensor,
        modulation: BlockModulation,
        *,
        attention_growth: float,
        mlp_growth: float,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [B,L,hidden_size]")
        if token_mask.shape != tokens.shape[:2] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean with shape [B,L]")
        if not 0.0 <= attention_growth <= 1.0 or not 0.0 <= mlp_growth <= 1.0:
            raise ValueError("growth switches must be in [0,1]")
        self._validate_modulation(modulation, tokens.shape[0])
        valid = token_mask.unsqueeze(-1)
        tokens = tokens * valid
        attention_input = self._modulate(
            self.attention_norm(tokens),
            modulation.attention_scale,
            modulation.attention_shift,
        )
        attention_output = self.attention(
            attention_input,
            attention_mask,
            coordinates,
        )
        attention_gate = modulation.attention_gate.to(tokens.dtype).unsqueeze(1)
        tokens = tokens + attention_growth * attention_gate * attention_output
        tokens = tokens * valid

        mlp_input = self._modulate(
            self.mlp_norm(tokens),
            modulation.mlp_scale,
            modulation.mlp_shift,
        )
        mlp_output = self.mlp(mlp_input)
        mlp_gate = modulation.mlp_gate.to(tokens.dtype).unsqueeze(1)
        tokens = tokens + mlp_growth * mlp_gate * mlp_output
        return tokens * valid


__all__ = ["DiTBlock"]
