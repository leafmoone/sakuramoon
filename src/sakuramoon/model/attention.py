"""Dense reference and DAS FlashAttention-2 varlen grouped-query attention."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

try:
    # DAS provides a Hygon/DCU-compatible FlashAttention 2 wheel.  Keep the
    # import optional so the dense reference path remains usable elsewhere.
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except (ImportError, OSError):
    _flash_attn_varlen_func = None

from sakuramoon.conditioning.packing import (
    ValidatedCuSeqlens,
    build_validated_cu_seqlens,
    validated_cu_seqlens_for_packed_entry,
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
    """Promote constructor-sealed offsets to the private FA4 capability."""

    lengths, tensor = validated_cu_seqlens_for_packed_entry(
        boundaries,
        total_tokens=total_tokens,
        batch_size=batch_size,
        device=device,
    )
    return AcceptedCuSeqlens.create_for_entry(
        lengths,
        tensor,
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


@torch.compiler.disable(
    reason="DAS flash_attn_varlen_func is an explicit PyBind eager boundary"
)
def fa4_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    boundaries: AcceptedCuSeqlens,
) -> torch.Tensor:
    """Run the installed DAS FlashAttention-2 kernel on packed sequences.

    The public function name is retained for checkpoint/test compatibility with
    the original FA4 implementation; the DTK branch deliberately uses the DAS
    ``flash_attn_varlen_func`` ABI instead of ``flash_attn.cute``.
    """

    total_tokens = query.shape[0] if query.ndim == 3 else -1
    if query.shape != (total_tokens, FA4_QUERY_HEADS, FA4_HEAD_DIM):
        raise ValueError("query must have shape [T,20,128]")
    expected_kv_shape = (total_tokens, FA4_KV_HEADS, FA4_HEAD_DIM)
    if key.shape != expected_kv_shape or value.shape != expected_kv_shape:
        raise ValueError("key and value must have shape [T,5,128]")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise TypeError("packed production attention requires BF16 query, key, and value")
    if not all(tensor.is_cuda for tensor in (query, key, value)):
        raise ValueError("packed production attention requires CUDA tensors")
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

    if _flash_attn_varlen_func is None:
        raise RuntimeError(
            "DAS flash-attn with flash_attn_varlen_func is required for packed attention"
        )
    output = _flash_attn_varlen_func(
        query,
        key,
        value,
        boundary_tensor,
        boundary_tensor,
        accepted.max_seqlen,
        accepted.max_seqlen,
        dropout_p=0.0,
        causal=False,
        deterministic=False,
    )
    if output.shape != query.shape or output.dtype != torch.bfloat16:
        raise RuntimeError("DAS FlashAttention returned an unexpected output shape or dtype")
    return output


def dense_attention_mask(token_mask: torch.Tensor) -> torch.Tensor:
    if token_mask.ndim != 2 or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be a boolean [B,L] tensor")
    return token_mask[:, None, :, None] & token_mask[:, None, None, :]


def _outer_mask_lengths(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int] | None:
    """Return valid-token lengths when a dense mask is an outer product.

    FlashAttention varlen is mathematically equivalent to the dense attention
    used by this project for masks produced by :func:`dense_attention_mask`.
    Keep the check here so callers that provide a genuinely arbitrary dense
    mask continue through the reference SDPA implementation.
    """

    query_valid = attention_mask.any(dim=-1).squeeze(1)
    key_valid = attention_mask.any(dim=-2).squeeze(1)
    if not torch.equal(query_valid, key_valid):
        return None

    row_counts = attention_mask.sum(dim=-1).squeeze(1)
    expected_counts = query_valid.to(row_counts.dtype) * key_valid.sum(
        dim=-1,
        keepdim=True,
    )
    if not torch.equal(row_counts, expected_counts):
        return None

    lengths = query_valid.sum(dim=-1, dtype=torch.int32).contiguous()
    if bool((lengths <= 0).any()):
        return None
    return lengths, int(lengths.max().item())


def _flash_varlen_self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_tokens: torch.Tensor,
    lengths: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Run DAS FlashAttention on packed valid tokens and restore BSHD layout."""

    if _flash_attn_varlen_func is None:
        raise RuntimeError("FlashAttention varlen kernel is not installed")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("packed FlashAttention inputs must be BSHD tensors")

    batch, _seqlen, _, _ = query.shape
    cu_seqlens = torch.zeros(
        batch + 1,
        device=query.device,
        dtype=torch.int32,
    )
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    packed_query = query[valid_tokens].contiguous()
    packed_key = key[valid_tokens].contiguous()
    packed_value = value[valid_tokens].contiguous()
    packed_output = _flash_attn_varlen_func(
        packed_query,
        packed_key,
        packed_value,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=0.0,
        causal=False,
    )
    output = torch.zeros_like(query)
    output[valid_tokens] = packed_output
    return output


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
        query = query.view(batch, length, self.q_heads, self.head_dim).contiguous()
        key = key.view(batch, length, self.kv_heads, self.head_dim).contiguous()
        value = value.view(batch, length, self.kv_heads, self.head_dim).contiguous()
        outer_lengths = _outer_mask_lengths(attention_mask)
        use_flash_varlen = (
            _flash_attn_varlen_func is not None
            and outer_lengths is not None
            and query.device.type == "cuda"
            and query.dtype in (torch.float16, torch.bfloat16)
            # On HCU, packing/launch overhead dominates tiny eval batches.
            and batch >= 8
        )
        if use_flash_varlen:
            lengths, max_seqlen = outer_lengths
            valid_tokens = attention_mask.any(dim=-1).squeeze(1)
            attended = _flash_varlen_self_attention(
                query,
                key,
                value,
                valid_tokens,
                lengths,
                max_seqlen,
            )
        else:
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            ).transpose(1, 2).contiguous()
        attended = attended.reshape(batch, length, self.hidden_size)
        gated = attended * torch.sigmoid(self.content_gate(tokens))
        output = self.out_proj(gated)
        valid_queries = attention_mask.any(dim=-1).squeeze(1)
        return output * valid_queries.unsqueeze(-1)


class FA4VarlenGQAAttention(nn.Module):
    """Production DiT attention over padding-free tokens using DAS FA2."""

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
                "packed attention is locked to d=2560, 20Q/5KV, head_dim=128"
            )
        if linear_dtype != torch.bfloat16:
            raise ValueError("packed production projections require BF16")
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
            raise ValueError("packed production tokens must be CUDA BF16")
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
