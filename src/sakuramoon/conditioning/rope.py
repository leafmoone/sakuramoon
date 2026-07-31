"""Area-normalized image coordinates and 2D rotary embedding."""

from __future__ import annotations

import math

import torch
from torch import nn

from sakuramoon.conditioning.norm import FP32RMSNorm
from sakuramoon.conditioning.packing import PackedSequences


def image_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("image token grid dimensions must be positive")
    y_radius = math.sqrt(height / width)
    x_radius = math.sqrt(width / height)
    y = (2.0 * (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height - 1.0) * y_radius
    x = (2.0 * (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width - 1.0) * x_radius
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_y.flatten(), grid_x.flatten()), dim=-1)


def packed_coordinates(packed: PackedSequences) -> torch.Tensor:
    coordinates = torch.zeros(
        packed.tokens.shape[0],
        2,
        dtype=torch.float32,
        device=packed.tokens.device,
    )
    for spans, (height, width) in zip(packed.spans, packed.image_shapes, strict=True):
        coordinates[spans.image.start : spans.image.end] = image_coordinates(
            height,
            width,
            device=packed.tokens.device,
        )
    return coordinates


def _rotate(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    pairs = values.float().reshape(*values.shape[:-1], -1, 2)
    first, second = pairs.unbind(dim=-1)
    cosine = angles.cos()[:, None]
    sine = angles.sin()[:, None]
    rotated = torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    )
    return rotated.flatten(-2).to(values.dtype)


class QKRoPE2D(nn.Module):
    """Normalize Q/K per head, then apply shared-frequency y/x RoPE."""

    frequencies: torch.Tensor

    def __init__(
        self,
        *,
        head_dim: int,
        nope_dim: int,
        y_dim: int,
        x_dim: int,
        position_scale: float,
        theta: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if nope_dim + y_dim + x_dim != head_dim or y_dim != x_dim or y_dim % 2:
            raise ValueError("RoPE dimensions must partition the head and share even y/x sizes")
        if position_scale <= 0.0 or theta <= 0.0:
            raise ValueError("position_scale and theta must be positive")
        self.head_dim = head_dim
        self.nope_dim = nope_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.position_scale = position_scale
        self.q_norm = FP32RMSNorm(head_dim, norm_eps)
        self.k_norm = FP32RMSNorm(head_dim, norm_eps)
        frequencies = theta ** (
            -torch.arange(0, y_dim, 2, dtype=torch.float32) / float(y_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def _apply_rope(self, values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        nope, y_values, x_values = torch.split(
            values,
            [self.nope_dim, self.y_dim, self.x_dim],
            dim=-1,
        )
        y_angles = coordinates[:, 0, None] * self.position_scale * self.frequencies[None]
        x_angles = coordinates[:, 1, None] * self.position_scale * self.frequencies[None]
        return torch.cat(
            (nope, _rotate(y_values, y_angles), _rotate(x_values, x_angles)),
            dim=-1,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or key.ndim != 3:
            raise ValueError("query and key must have shape [tokens, heads, head_dim]")
        if query.shape[0] != key.shape[0] or query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError("query/key token counts and head dimensions must match")
        if coordinates.shape != (query.shape[0], 2):
            raise ValueError("coordinates must have shape [tokens,2]")
        if query.dtype != key.dtype or query.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("query and key must share float32 or bfloat16 dtype")
        if coordinates.dtype != torch.float32:
            raise TypeError("coordinates must use float32")
        if (
            key.device != query.device
            or coordinates.device != query.device
            or self.frequencies.device != query.device
        ):
            raise ValueError("query, key, coordinates, and RoPE frequencies must share one device")
        normalized_query = self.q_norm(query)
        normalized_key = self.k_norm(key)
        return self._apply_rope(normalized_query, coordinates), self._apply_rope(
            normalized_key, coordinates
        )


__all__ = ["QKRoPE2D", "image_coordinates", "packed_coordinates"]
