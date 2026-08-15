"""Fixed-count condition-token encoder with learned null tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sakuramoon.conditioning.norm import FP32RMSNorm


@dataclass(frozen=True)
class ConditionTokenOutput:
    tokens: torch.Tensor
    mask: torch.Tensor


class _ResidualSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        norm_eps: float,
        projection_bias: bool,
        linear_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm = FP32RMSNorm(hidden_size, norm_eps)
        self.gate = nn.Linear(
            hidden_size, intermediate_size, bias=projection_bias, dtype=linear_dtype
        )
        self.up = nn.Linear(
            hidden_size, intermediate_size, bias=projection_bias, dtype=linear_dtype
        )
        self.down = nn.Linear(
            intermediate_size, hidden_size, bias=projection_bias, dtype=linear_dtype
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tensor)
        return tensor + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class ConditionTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        token_count: int,
        attention_heads: int,
        norm_eps: float,
        init_std: float,
        projection_bias: bool,
        linear_dtype: torch.dtype,
        sensitive_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if token_count <= 0 or init_std <= 0.0:
            raise ValueError("token_count and init_std must be positive")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if sensitive_dtype != torch.float32:
            raise ValueError("sensitive condition parameters must use float32")
        self.input_size = input_size
        self.token_count = token_count
        self._artifact_config: dict[str, object] = {
            "attention_heads": attention_heads,
            "hidden_size": hidden_size,
            "init_std": init_std,
            "input_size": input_size,
            "intermediate_size": intermediate_size,
            "linear_dtype": str(linear_dtype).removeprefix("torch."),
            "norm_eps": norm_eps,
            "output_size": output_size,
            "projection_bias": projection_bias,
            "token_count": token_count,
            "sensitive_dtype": str(sensitive_dtype).removeprefix("torch."),
        }
        self.shared_norm = FP32RMSNorm(input_size, norm_eps)
        self.layer_embedding = nn.Parameter(
            torch.empty(7, input_size, dtype=sensitive_dtype)
        )
        self.input_projection = nn.Linear(
            input_size,
            hidden_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )
        self.queries = nn.Parameter(
            torch.empty(token_count, hidden_size, dtype=sensitive_dtype)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            attention_heads,
            dropout=0.0,
            bias=projection_bias,
            batch_first=True,
            dtype=linear_dtype,
        )
        self.condition_mlp = _ResidualSwiGLU(
            hidden_size,
            intermediate_size,
            norm_eps,
            projection_bias,
            linear_dtype,
        )
        self.output_projection = nn.Linear(
            hidden_size,
            output_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )
        self.null_tokens = nn.Parameter(
            torch.empty(token_count, output_size, dtype=sensitive_dtype)
        )
        nn.init.normal_(self.layer_embedding, std=init_std)
        nn.init.normal_(self.queries, std=init_std)
        nn.init.normal_(self.null_tokens, std=init_std)

    def forward(
        self,
        qwen_states: torch.Tensor,
        condition_token_indices: torch.Tensor,
        condition_mask: torch.Tensor,
        use_null_condition: torch.Tensor,
        active_condition_sample_indices: torch.Tensor,
    ) -> ConditionTokenOutput:
        if qwen_states.ndim != 4 or qwen_states.shape[2:] != (7, self.input_size):
            raise ValueError("qwen_states must have shape [B,L,7,input_size]")
        if (
            condition_token_indices.shape != condition_mask.shape
            or condition_mask.dtype != torch.bool
        ):
            raise ValueError(
                "condition indices and mask must have matching [B,C] shapes"
            )
        if condition_token_indices.dtype != torch.long:
            raise TypeError("condition_token_indices must use torch.long")
        batch = qwen_states.shape[0]
        if (
            batch <= 0
            or condition_token_indices.shape[0] != batch
            or use_null_condition.shape != (batch,)
        ):
            raise ValueError("condition metadata batch shape differs from Qwen states")
        if use_null_condition.dtype != torch.bool:
            raise TypeError("use_null_condition must be boolean")
        if (
            condition_token_indices.device != qwen_states.device
            or condition_mask.device != qwen_states.device
            or use_null_condition.device != qwen_states.device
            or active_condition_sample_indices.device != qwen_states.device
        ):
            raise ValueError(
                "Qwen states and condition routing tensors must share one device"
            )
        if (
            active_condition_sample_indices.ndim != 1
            or active_condition_sample_indices.dtype != torch.long
        ):
            raise ValueError(
                "active_condition_sample_indices must be a one-dimensional long tensor"
            )

        tokens = (
            self.null_tokens.to(dtype=qwen_states.dtype, device=qwen_states.device)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .clone()
        )
        sample_in_range = (active_condition_sample_indices >= 0) & (
            active_condition_sample_indices < batch
        )
        safe_active_samples = active_condition_sample_indices.clamp(0, batch - 1)
        expected_active = condition_mask.any(dim=1) & ~use_null_condition
        planned_active = torch.zeros_like(expected_active).scatter(
            0,
            safe_active_samples,
            True,
        )
        plan_matches = (planned_active == expected_active).all() & (
            expected_active.sum() == active_condition_sample_indices.numel()
        )
        active_condition_indices = condition_token_indices.index_select(
            0, safe_active_samples
        )
        active_condition_mask = condition_mask.index_select(0, safe_active_samples)
        indices_in_range = (~active_condition_mask) | (
            (active_condition_indices >= 0)
            & (active_condition_indices < qwen_states.shape[1])
        )
        valid_plan = sample_in_range.all() & plan_matches & indices_in_range.all()
        if qwen_states.is_cuda:
            torch._assert_async(  # pyright: ignore[reportPrivateUsage,reportPrivateImportUsage]
                valid_plan
            )
        elif not bool(valid_plan):
            raise ValueError("active condition sample/index plan is invalid")

        if active_condition_sample_indices.numel():
            safe_indices = active_condition_indices.masked_fill(
                ~active_condition_mask, 0
            )
            gather_index = safe_indices[:, :, None, None].expand(
                -1, -1, 7, self.input_size
            )
            selected = torch.gather(
                qwen_states.detach().index_select(0, safe_active_samples),
                dim=1,
                index=gather_index,
            )
            selected = self.shared_norm(selected) + self.layer_embedding.to(
                selected.dtype
            )[None, None]
            memory = self.input_projection(selected).flatten(1, 2)
            memory_mask = (
                active_condition_mask[:, :, None].expand(-1, -1, 7).flatten(1, 2)
            )
            queries = (
                self.queries.to(memory.dtype)
                .unsqueeze(0)
                .expand(memory.shape[0], -1, -1)
            )
            attended, _ = self.cross_attention(
                queries,
                memory,
                memory,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )
            condition = self.condition_mlp(queries + attended)
            tokens.index_copy_(
                0,
                safe_active_samples,
                self.output_projection(condition),
            )

        mask = torch.ones(
            batch,
            self.token_count,
            dtype=torch.bool,
            device=qwen_states.device,
        )
        return ConditionTokenOutput(tokens=tokens, mask=mask)

    def artifact_config(self) -> dict[str, object]:
        return dict(self._artifact_config)

__all__ = ["ConditionTokenEncoder", "ConditionTokenOutput"]
