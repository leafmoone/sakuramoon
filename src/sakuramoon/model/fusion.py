"""Opt-in Inductor fusion boundaries, independent of the attention backend.

No parameters or state_dict keys are introduced. Autograd owns all reductions,
including repeated packed sample indices and shared per-head gamma. Explicit
casts preserve the *source* rounding points, but Inductor may change BF16
rounding, FMA and reduction order; validate on the actual training backend.
Compilation errors propagate: silently falling back is not proof of fusion.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TypeVar, cast

import torch
from torch import nn

F = TypeVar("F", bound=Callable[..., object])


def _norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + eps
    )
    return normalized.to(x.dtype) * weight.to(x.dtype)


def norm_modulation_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    sample_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = _norm(x, weight, eps)
    if sample_indices is None:
        scale = scale.to(x.dtype).unsqueeze(1)
        shift = shift.to(x.dtype).unsqueeze(1)
    else:
        # Gather stays inside the compiled boundary; no eager [tokens, hidden]
        # modulation copies are passed across the call boundary.
        scale = scale.index_select(0, sample_indices).to(x.dtype)
        shift = shift.index_select(0, sample_indices).to(x.dtype)
    return (1.0 + scale) * normalized + shift


def residual_reference(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    growth: float | torch.Tensor,
    sample_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if sample_indices is None:
        gate = gate.to(residual.dtype).unsqueeze(1)
    else:
        gate = gate.index_select(0, sample_indices).to(residual.dtype)
    return residual + growth * gate * branch


def _rotate(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    pairs = values.float().reshape(*values.shape[:-1], -1, 2)
    first, second = pairs.unbind(-1)
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
    return (
        torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine), -1
        )
        .flatten(-2)
        .to(values.dtype)
    )


def qk_rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    coordinates: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
    nope_dim: int,
    y_dim: int,
    x_dim: int,
    position_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_angles = coordinates[:, 0, None] * position_scale * frequencies[None]
    x_angles = coordinates[:, 1, None] * position_scale * frequencies[None]

    def apply(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        nope, y, x = torch.split(
            _norm(values, weight, eps), [nope_dim, y_dim, x_dim], -1
        )
        return torch.cat(
            (nope, _rotate(y, y_angles), _rotate(x, x_angles)), -1
        ).contiguous()

    # Q and K have independent head dimensions; KV is never expanded.
    return apply(query, q_weight), apply(key, k_weight)


@cache
def _compiled(function: Callable[..., object]) -> Callable[..., object]:
    """Lazy compilation; actual device/compiler support is checked on first call."""
    return torch.compile(function, backend="inductor", fullgraph=True)  # pyright: ignore[reportUnknownMemberType]


def compiled(function: F) -> F:
    # An outer compiler already owns this graph: inline the same tensor ops,
    # rather than tracing the cache or creating nested compiled callables.
    if torch.compiler.is_compiling():
        return function
    return cast(F, _compiled(function))


def configure_fusions(model: nn.Module, *, enabled: bool = True) -> int:
    """Enable/disable fusion before DDP/outer compile; return module count.

    This is a runtime optimization switch, not checkpoint architecture metadata.
    Set it again explicitly after loading. The default remains the reference path.
    """
    # Local imports avoid a module import cycle with the call sites.
    from sakuramoon.conditioning.rope import QKRoPE2D
    from sakuramoon.model.block import DiTBlock, PackedDiTBlock

    count = 0
    for module in model.modules():
        if isinstance(module, (DiTBlock, PackedDiTBlock, QKRoPE2D)):
            module.fusions_enabled = enabled
            count += 1
    return count
