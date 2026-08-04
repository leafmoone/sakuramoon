"""Dense reference and FA4 varlen grouped-query attention."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import nn

from sakuramoon.conditioning.packing import (
    ValidatedCuSeqlens,
    build_validated_cu_seqlens,
)
from sakuramoon.conditioning.rope import QKRoPE2D

FA4_QUERY_HEADS = 20
FA4_KV_HEADS = 5
FA4_HEAD_DIM = 128
# flash-attn-4 beta24 faults on SM120 asymmetric varlen batches when this
# layout optimization is enabled. Native 20Q/5KV GQA does not depend on it.
FA4_PACK_GQA = False

_ACCEPTED_BOUNDARY_CAPABILITY = object()


@dataclass(frozen=True, init=False)
class AcceptedCuSeqlens:
    """Private packed-entry capability reused by every production block."""

    sequence_lengths: tuple[int, ...]
    total_tokens: int
    max_seqlen: int
    batch_size: int
    __tensor: torch.Tensor = field(repr=False, compare=False)
    __capability: object = field(repr=False, compare=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("accepted boundaries can only be created at the packed entry")

    @classmethod
    def create_for_entry(
        cls,
        sequence_lengths: tuple[int, ...],
        tensor: torch.Tensor,
        *,
        capability: object,
    ) -> AcceptedCuSeqlens:
        if capability is not _ACCEPTED_BOUNDARY_CAPABILITY:
            raise TypeError("invalid accepted-boundary capability")
        instance = object.__new__(cls)
        object.__setattr__(instance, "sequence_lengths", sequence_lengths)
        object.__setattr__(instance, "total_tokens", sum(sequence_lengths))
        object.__setattr__(instance, "max_seqlen", max(sequence_lengths))
        object.__setattr__(instance, "batch_size", len(sequence_lengths))
        object.__setattr__(instance, "_AcceptedCuSeqlens__tensor", tensor)
        object.__setattr__(instance, "_AcceptedCuSeqlens__capability", capability)
        return instance

    def has_capability(self, *, capability: object) -> bool:
        return self.__capability is capability

    def tensor_for_kernel(self, *, capability: object) -> torch.Tensor:
        if capability is not _ACCEPTED_BOUNDARY_CAPABILITY:
            raise TypeError("invalid accepted-boundary capability")
        return self.__tensor


def _require_accepted_boundaries(
    boundaries: AcceptedCuSeqlens,
) -> AcceptedCuSeqlens:
    if (
        type(boundaries) is not AcceptedCuSeqlens
        or not boundaries.has_capability(
            capability=_ACCEPTED_BOUNDARY_CAPABILITY
        )
    ):
        raise TypeError("FA4 requires boundaries accepted at the packed entry")
    return boundaries


def accept_fa4_boundaries(
    boundaries: ValidatedCuSeqlens,
    *,
    total_tokens: int,
    batch_size: int,
    device: torch.device,
) -> AcceptedCuSeqlens:
    """Validate boundary contents once, then rematerialize private CUDA offsets."""

    if type(boundaries) is not ValidatedCuSeqlens:
        raise TypeError("packed entry requires a ValidatedCuSeqlens input")
    lengths = boundaries.sequence_lengths
    if type(lengths) is not tuple or not lengths or any(
        type(length) is not int or length <= 0 for length in lengths
    ):
        raise ValueError("validated boundaries contain invalid host lengths")
    offsets = (0, *accumulate(lengths))
    if (
        boundaries.batch_size != len(lengths)
        or boundaries.batch_size != batch_size
        or boundaries.total_tokens != offsets[-1]
        or boundaries.total_tokens != total_tokens
        or boundaries.max_seqlen != max(lengths)
    ):
        raise ValueError("validated boundaries contain inconsistent host metadata")
    if (
        boundaries.tensor.ndim != 1
        or boundaries.tensor.shape != (len(offsets),)
        or boundaries.tensor.dtype != torch.int32
        or not boundaries.tensor.is_contiguous()
        or boundaries.tensor.device.type != device.type
        or (
            (device_index := getattr(device, "index", None)) is not None
            and boundaries.tensor.device.index != device_index
        )
    ):
        raise ValueError("validated boundaries contain inconsistent tensor metadata")

    expected_cpu = torch.tensor(offsets, dtype=torch.int32, device="cpu")
    actual_cpu = boundaries.tensor.detach().to(device="cpu", copy=True)
    if not torch.equal(actual_cpu, expected_cpu):
        raise ValueError("CUDA boundary values differ from validated host lengths")

    private_tensor = expected_cpu.to(device=boundaries.tensor.device).contiguous()
    return AcceptedCuSeqlens.create_for_entry(
        lengths,
        private_tensor,
        capability=_ACCEPTED_BOUNDARY_CAPABILITY,
    )


def accepted_sample_indices(boundaries: AcceptedCuSeqlens) -> torch.Tensor:
    """Build token routing from the same accepted host identity used by FA4."""

    accepted = _require_accepted_boundaries(boundaries)
    boundary_tensor = accepted.tensor_for_kernel(
        capability=_ACCEPTED_BOUNDARY_CAPABILITY
    )
    lengths = torch.tensor(
        accepted.sequence_lengths,
        device=boundary_tensor.device,
        dtype=torch.int64,
    )
    return torch.repeat_interleave(
        torch.arange(
            accepted.batch_size,
            device=boundary_tensor.device,
            dtype=torch.int64,
        ),
        lengths,
        output_size=accepted.total_tokens,
    )


class _FA4VarlenCallable(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        pack_gqa: bool,
        deterministic: bool,
        return_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def fa4_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    boundaries: AcceptedCuSeqlens,
) -> torch.Tensor:
    """Run the locked FA4 self-attention kernel on padding-free sequences."""

    total_tokens = query.shape[0] if query.ndim == 3 else -1
    if query.shape != (total_tokens, FA4_QUERY_HEADS, FA4_HEAD_DIM):
        raise ValueError("query must have shape [T,20,128]")
    expected_kv_shape = (total_tokens, FA4_KV_HEADS, FA4_HEAD_DIM)
    if key.shape != expected_kv_shape or value.shape != expected_kv_shape:
        raise ValueError("key and value must have shape [T,5,128]")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise TypeError("FA4 production attention requires BF16 query, key, and value")
    if not all(tensor.is_cuda for tensor in (query, key, value)):
        raise ValueError("FA4 production attention requires CUDA tensors")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key, and value must share one CUDA device")
    accepted = _require_accepted_boundaries(boundaries)
    boundary_tensor = accepted.tensor_for_kernel(
        capability=_ACCEPTED_BOUNDARY_CAPABILITY
    )
    if accepted.total_tokens != total_tokens:
        raise ValueError("validated boundaries do not match the token count")
    if (
        accepted.batch_size <= 0
        or len(accepted.sequence_lengths) != accepted.batch_size
        or sum(accepted.sequence_lengths) != accepted.total_tokens
        or max(accepted.sequence_lengths) != accepted.max_seqlen
        or boundary_tensor.ndim != 1
        or boundary_tensor.shape != (accepted.batch_size + 1,)
        or boundary_tensor.dtype != torch.int32
        or not boundary_tensor.is_contiguous()
    ):
        raise ValueError("accepted boundaries contain inconsistent static metadata")
    if boundary_tensor.device != query.device:
        raise ValueError("cu_seqlens and query must share one CUDA device")
    if not all(tensor.is_contiguous() for tensor in (query, key, value)):
        raise ValueError("FA4 query, key, and value must be contiguous")

    try:
        fa4_module = importlib.import_module("flash_attn.cute")
    except ImportError as exc:
        raise RuntimeError(
            "flash-attn-4 is required for the fa4_varlen backend"
        ) from exc
    flash_attn_varlen_func = cast(
        _FA4VarlenCallable,
        fa4_module.flash_attn_varlen_func,
    )

    output, _lse = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=boundary_tensor,
        cu_seqlens_k=boundary_tensor,
        max_seqlen_q=accepted.max_seqlen,
        max_seqlen_k=accepted.max_seqlen,
        causal=False,
        pack_gqa=FA4_PACK_GQA,
        deterministic=False,
        return_lse=False,
    )
    if output.shape != query.shape or output.dtype != torch.bfloat16:
        raise RuntimeError("FA4 returned an unexpected output shape or dtype")
    return output


def dense_attention_mask(token_mask: torch.Tensor) -> torch.Tensor:
    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a boolean [B,L] tensor")
    return token_mask[:, None, :, None] & token_mask[:, None, None, :]


class DenseGQAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
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
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            hidden_size <= 0
            or q_heads <= 0
            or kv_heads <= 0
            or head_dim <= 0
            or q_heads * head_dim != hidden_size
            or q_heads % kv_heads
        ):
            raise ValueError("attention dimensions violate native GQA")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if projection_bias or dropout != 0.0:
            raise ValueError("DiT attention requires bias=false and dropout=0")
        self.hidden_size = hidden_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden_size,
            q_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            kv_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            kv_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.content_gate = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            dtype=linear_dtype,
        )
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            dtype=linear_dtype,
        )
        self.qk_rope = QKRoPE2D(
            head_dim=head_dim,
            nope_dim=rope_nope_dim,
            y_dim=rope_y_dim,
            x_dim=rope_x_dim,
            position_scale=rope_position_scale,
            theta=rope_theta,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [B,L,hidden_size]")
        batch, length, _ = tokens.shape
        if attention_mask.shape != (batch, 1, length, length):
            raise ValueError("attention_mask must have shape [B,1,L,L]")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean with True meaning allowed")
        if (
            coordinates.shape != (batch, length, 2)
            or coordinates.dtype != torch.float32
        ):
            raise ValueError("coordinates must be FP32 with shape [B,L,2]")

        query = self.q_proj(tokens).view(
            batch,
            length,
            self.q_heads,
            self.head_dim,
        )
        key = self.k_proj(tokens).view(
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        )
        value = self.v_proj(tokens).view(
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        )
        query, key = self.qk_rope(
            query.flatten(0, 1),
            key.flatten(0, 1),
            coordinates.flatten(0, 1),
        )
        query = (
            query.view(batch, length, self.q_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        key = (
            key.view(batch, length, self.kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value = value.transpose(1, 2).contiguous()
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.hidden_size)
        gated = attended * torch.sigmoid(self.content_gate(tokens))
        output = self.out_proj(gated)
        valid_queries = attention_mask.any(dim=-1).squeeze(1)
        return output * valid_queries.unsqueeze(-1)


class FA4VarlenGQAAttention(nn.Module):
    """Production DiT attention over T024 padding-free packed tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
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
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            hidden_size != FA4_QUERY_HEADS * FA4_HEAD_DIM
            or q_heads != FA4_QUERY_HEADS
            or kv_heads != FA4_KV_HEADS
            or head_dim != FA4_HEAD_DIM
        ):
            raise ValueError(
                "FA4 production attention is locked to d=2560, 20Q/5KV, head_dim=128"
            )
        if linear_dtype != torch.bfloat16:
            raise ValueError("FA4 production projections require BF16")
        if projection_bias or dropout != 0.0:
            raise ValueError("DiT attention requires bias=false and dropout=0")
        self.hidden_size = hidden_size
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=linear_dtype
        )
        self.k_proj = nn.Linear(
            hidden_size,
            kv_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            kv_heads * head_dim,
            bias=False,
            dtype=linear_dtype,
        )
        self.content_gate = nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=linear_dtype
        )
        self.out_proj = nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=linear_dtype
        )
        self.qk_rope = QKRoPE2D(
            head_dim=head_dim,
            nope_dim=rope_nope_dim,
            y_dim=rope_y_dim,
            x_dim=rope_x_dim,
            position_scale=rope_position_scale,
            theta=rope_theta,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        boundaries: AcceptedCuSeqlens,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [T,2560]")
        if tokens.dtype != torch.bfloat16 or not tokens.is_cuda:
            raise ValueError("FA4 production tokens must be CUDA BF16")
        if (
            coordinates.shape != (tokens.shape[0], 2)
            or coordinates.dtype != torch.float32
        ):
            raise ValueError("coordinates must be FP32 with shape [T,2]")
        if coordinates.device != tokens.device:
            raise ValueError("coordinates and tokens must share one CUDA device")

        query = self.q_proj(tokens).view(-1, self.q_heads, self.head_dim)
        key = self.k_proj(tokens).view(-1, self.kv_heads, self.head_dim)
        value = self.v_proj(tokens).view(-1, self.kv_heads, self.head_dim)
        query, key = self.qk_rope(query, key, coordinates)
        attended = fa4_varlen_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            boundaries,
        ).reshape(-1, self.hidden_size)
        gated = attended * torch.sigmoid(self.content_gate(tokens))
        return self.out_proj(gated)


__all__ = [
    "FA4_PACK_GQA",
    "DenseGQAAttention",
    "FA4VarlenGQAAttention",
    "ValidatedCuSeqlens",
    "accept_fa4_boundaries",
    "accepted_sample_indices",
    "build_validated_cu_seqlens",
    "dense_attention_mask",
    "fa4_varlen_attention",
]
