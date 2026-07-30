"""Four-query Artist style resampler with learned null tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sakuramoon.conditioning.norm import FP32RMSNorm


@dataclass(frozen=True)
class StyleConditioningOutput:
    tokens: torch.Tensor
    mask: torch.Tensor


class _ResidualSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        norm_eps: float,
        projection_bias: bool,
    ) -> None:
        super().__init__()
        self.norm = FP32RMSNorm(hidden_size, norm_eps)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=projection_bias)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=projection_bias)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=projection_bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tensor)
        return tensor + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class StyleResampler(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        query_count: int,
        attention_heads: int,
        norm_eps: float,
        init_std: float,
        projection_bias: bool,
    ) -> None:
        super().__init__()
        if query_count <= 0 or init_std <= 0.0:
            raise ValueError("query_count and init_std must be positive")
        self.input_size = input_size
        self.query_count = query_count
        self.shared_norm = FP32RMSNorm(input_size, norm_eps)
        self.layer_embedding = nn.Parameter(torch.empty(7, input_size))
        self.input_projection = nn.Linear(input_size, hidden_size, bias=projection_bias)
        self.queries = nn.Parameter(torch.empty(query_count, hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            attention_heads,
            dropout=0.0,
            bias=projection_bias,
            batch_first=True,
        )
        self.style_mlp = _ResidualSwiGLU(
            hidden_size,
            intermediate_size,
            norm_eps,
            projection_bias,
        )
        self.output_projection = nn.Linear(hidden_size, output_size, bias=projection_bias)
        self.null_tokens = nn.Parameter(torch.empty(query_count, output_size))
        nn.init.normal_(self.layer_embedding, std=init_std)
        nn.init.normal_(self.queries, std=init_std)
        nn.init.normal_(self.null_tokens, std=init_std)

    def forward(
        self,
        qwen_states: torch.Tensor,
        artist_token_indices: torch.Tensor,
        artist_mask: torch.Tensor,
        use_null: torch.Tensor,
    ) -> StyleConditioningOutput:
        if qwen_states.ndim != 4 or qwen_states.shape[2:] != (7, self.input_size):
            raise ValueError("qwen_states must have shape [B,L,7,input_size]")
        if artist_token_indices.shape != artist_mask.shape or artist_mask.dtype != torch.bool:
            raise ValueError("Artist indices and mask must have matching [B,A] shapes")
        if artist_token_indices.dtype != torch.long:
            raise TypeError("artist_token_indices must use torch.long")
        batch = qwen_states.shape[0]
        if artist_token_indices.shape[0] != batch or use_null.shape != (batch,):
            raise ValueError("Artist metadata batch shape differs from Qwen states")
        if use_null.dtype != torch.bool:
            raise TypeError("use_null must be boolean")
        active_indices = artist_token_indices[artist_mask]
        if active_indices.numel() and (
            active_indices.min() < 0 or active_indices.max() >= qwen_states.shape[1]
        ):
            raise ValueError("active Artist token index is outside the Qwen sequence")

        tokens = (
            self.null_tokens.to(dtype=qwen_states.dtype, device=qwen_states.device)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .clone()
        )
        active_samples = artist_mask.any(dim=1) & ~use_null
        if active_samples.any():
            safe_indices = artist_token_indices.clamp(min=0)
            gather_index = safe_indices[:, :, None, None].expand(-1, -1, 7, self.input_size)
            selected = torch.gather(qwen_states.detach(), dim=1, index=gather_index)
            selected = self.shared_norm(selected) + self.layer_embedding[None, None]
            memory = self.input_projection(selected).flatten(1, 2)
            memory_mask = artist_mask[:, :, None].expand(-1, -1, 7).flatten(1, 2)
            memory = memory[active_samples]
            memory_mask = memory_mask[active_samples]
            queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
            attended, _ = self.cross_attention(
                queries,
                memory,
                memory,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )
            style = self.style_mlp(queries + attended)
            tokens[active_samples] = self.output_projection(style)

        mask = torch.ones(batch, self.query_count, dtype=torch.bool, device=qwen_states.device)
        return StyleConditioningOutput(tokens=tokens, mask=mask)


__all__ = ["StyleConditioningOutput", "StyleResampler"]
