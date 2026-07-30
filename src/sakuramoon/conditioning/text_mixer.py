"""Seven-layer grouped text mixing and bidirectional refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from sakuramoon.conditioning.bidirectional import BidirectionalAttentionOnly
from sakuramoon.conditioning.norm import FP32RMSNorm


@dataclass(frozen=True)
class TextConditioningOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    layer_weights: torch.Tensor


class TextConditioner(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        adapter_size: int,
        output_size: int,
        groups: int,
        attention_heads: int,
        norm_eps: float,
        mix_gate_init: float,
        layer_scale_init: float,
        projection_bias: bool,
        linear_dtype: torch.dtype,
        sensitive_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if adapter_size <= 0 or groups <= 0 or adapter_size % groups:
            raise ValueError("adapter_size must be divisible by groups")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if sensitive_dtype != torch.float32:
            raise ValueError("sensitive text parameters must use float32")
        self.input_size = input_size
        self.adapter_size = adapter_size
        self.groups = groups
        self.group_size = adapter_size // groups
        self.layer_norms = nn.ModuleList(FP32RMSNorm(input_size, norm_eps) for _ in range(7))
        self.shared_projection = nn.Linear(
            input_size,
            adapter_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )
        self.gate_weight = nn.Parameter(
            torch.zeros(7, groups, self.group_size, dtype=sensitive_dtype)
        )
        self.gate_bias = nn.Parameter(torch.zeros(7, groups, dtype=sensitive_dtype))
        self.mix_gate = nn.Parameter(
            torch.tensor(float(mix_gate_init), dtype=sensitive_dtype)
        )
        self.refinement = BidirectionalAttentionOnly(
            hidden_size=adapter_size,
            num_heads=attention_heads,
            norm_eps=norm_eps,
            layer_scale_init=layer_scale_init,
            projection_bias=projection_bias,
            linear_dtype=linear_dtype,
        )
        self.output_norm = FP32RMSNorm(adapter_size, norm_eps)
        self.output_projection = nn.Linear(
            adapter_size,
            output_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )

    def forward(
        self,
        qwen_states: torch.Tensor,
        main_token_indices: torch.Tensor,
        main_mask: torch.Tensor,
    ) -> TextConditioningOutput:
        if qwen_states.ndim != 4 or qwen_states.shape[2:] != (7, self.input_size):
            raise ValueError("qwen_states must have shape [B,L,7,input_size]")
        if main_token_indices.shape != main_mask.shape or main_mask.dtype != torch.bool:
            raise ValueError("main indices and mask must have matching [B,M] shapes")
        if main_token_indices.dtype != torch.long:
            raise TypeError("main_token_indices must use torch.long")
        if main_token_indices.shape[0] != qwen_states.shape[0]:
            raise ValueError("main indices batch size differs from Qwen states")
        active = main_token_indices[main_mask]
        if active.numel() and (active.min() < 0 or active.max() >= qwen_states.shape[1]):
            raise ValueError("active main token index is outside the Qwen sequence")

        safe_indices = main_token_indices.clamp(min=0)
        gather_index = safe_indices[:, :, None, None].expand(-1, -1, 7, self.input_size)
        selected = torch.gather(qwen_states.detach(), dim=1, index=gather_index)
        selected = selected * main_mask[:, :, None, None]
        projected = torch.stack(
            tuple(
                self.shared_projection(self.layer_norms[layer](selected[:, :, layer]))
                for layer in range(7)
            ),
            dim=2,
        )
        grouped = projected.view(*projected.shape[:-1], self.groups, self.group_size)
        scores = (
            torch.einsum("bmlgh,lgh->bmlg", grouped.float(), self.gate_weight)
            / math.sqrt(self.group_size)
            + self.gate_bias[None, None]
        )
        weights = scores.softmax(dim=2)
        mixed = (
            (grouped.float() * weights.unsqueeze(-1))
            .sum(dim=2)
            .flatten(-2)
            .to(grouped.dtype)
        )
        deepest = projected[:, :, -1]
        tokens = deepest + self.mix_gate.to(deepest.dtype) * (mixed - deepest)
        tokens = self.refinement(tokens, main_mask)
        tokens = self.output_projection(self.output_norm(tokens))
        tokens = tokens * main_mask.unsqueeze(-1)
        return TextConditioningOutput(tokens=tokens, mask=main_mask, layer_weights=weights)


__all__ = ["TextConditioner", "TextConditioningOutput"]
