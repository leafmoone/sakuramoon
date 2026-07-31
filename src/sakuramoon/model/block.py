"""Condition-modulated dense DiT reference block."""

from __future__ import annotations

import torch
from torch import nn

from sakuramoon.conditioning.global_condition import BlockModulation
from sakuramoon.model.attention import (
    AcceptedCuSeqlens,
    DenseGQAAttention,
    FA4VarlenGQAAttention,
)
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


class PackedDiTBlock(nn.Module):
    """Condition-modulated production block over flat varlen tokens."""

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
        self.attention = FA4VarlenGQAAttention(
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

    def _token_condition(
        self,
        value: torch.Tensor,
        sample_indices: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        if value.shape != (batch, self.hidden_size):
            raise ValueError("selected block modulation tensors must be [B,D]")
        return value.index_select(0, sample_indices)

    def forward(
        self,
        tokens: torch.Tensor,
        boundaries: AcceptedCuSeqlens,
        coordinates: torch.Tensor,
        sample_indices: torch.Tensor,
        modulation: BlockModulation,
        *,
        attention_growth: float,
        mlp_growth: float,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [T,hidden_size]")
        if (
            sample_indices.shape != tokens.shape[:1]
            or sample_indices.dtype != torch.int64
            or sample_indices.device != tokens.device
        ):
            raise ValueError("sample_indices must be int64 [T] on the token device")
        if not 0.0 <= attention_growth <= 1.0 or not 0.0 <= mlp_growth <= 1.0:
            raise ValueError("growth switches must be in [0,1]")
        batch = boundaries.batch_size
        if batch <= 0:
            raise ValueError("cu_seqlens must describe at least one sample")

        attention_scale = self._token_condition(
            modulation.attention_scale, sample_indices, batch
        ).to(tokens.dtype)
        attention_shift = self._token_condition(
            modulation.attention_shift, sample_indices, batch
        ).to(tokens.dtype)
        attention_input = (1.0 + attention_scale) * self.attention_norm(
            tokens
        ) + attention_shift
        attention_output = self.attention(
            attention_input,
            boundaries,
            coordinates,
        )
        attention_gate = self._token_condition(
            modulation.attention_gate, sample_indices, batch
        ).to(tokens.dtype)
        tokens = tokens + attention_growth * attention_gate * attention_output

        mlp_scale = self._token_condition(
            modulation.mlp_scale, sample_indices, batch
        ).to(tokens.dtype)
        mlp_shift = self._token_condition(
            modulation.mlp_shift, sample_indices, batch
        ).to(tokens.dtype)
        mlp_input = (1.0 + mlp_scale) * self.mlp_norm(tokens) + mlp_shift
        mlp_output = self.mlp(mlp_input)
        mlp_gate = self._token_condition(modulation.mlp_gate, sample_indices, batch).to(
            tokens.dtype
        )
        return tokens + mlp_growth * mlp_gate * mlp_output


__all__ = ["DiTBlock", "PackedDiTBlock"]
