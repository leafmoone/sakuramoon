"""Image-span-only clean-latent prediction head."""

from __future__ import annotations

import torch
from torch import nn

from sakuramoon.model.norm import RMSNorm


class FinalOutputHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        out_channels: int,
        norm_eps: float,
        projection_dtype: torch.dtype,
        weight_zero_init: bool,
        bias_zero_init: bool,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or out_channels <= 0:
            raise ValueError("output head dimensions must be positive")
        if projection_dtype != torch.float32:
            raise ValueError("the sensitive output projection must use float32")
        if not weight_zero_init or not bias_zero_init:
            raise ValueError("the output projection weight and bias must start at zero")
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.norm = RMSNorm(hidden_size, norm_eps)
        self.projection = nn.Linear(
            hidden_size,
            out_channels,
            bias=True,
            dtype=projection_dtype,
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        image_hidden: torch.Tensor,
        final_scale: torch.Tensor,
        final_shift: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        if image_hidden.ndim != 3 or image_hidden.shape[-1] != self.hidden_size:
            raise ValueError("image_hidden must have shape [B,H*W,hidden_size]")
        batch, image_tokens, _ = image_hidden.shape
        if final_scale.shape != (batch, self.hidden_size) or final_shift.shape != (
            batch,
            self.hidden_size,
        ):
            raise ValueError("final scale and shift must have shape [B,hidden_size]")
        height, width = image_shape
        if height <= 0 or width <= 0 or image_tokens != height * width:
            raise ValueError("image token count must equal the latent image shape")
        normalized = self.norm(image_hidden)
        conditioned = (
            (1.0 + final_scale.to(normalized.dtype).unsqueeze(1)) * normalized
            + final_shift.to(normalized.dtype).unsqueeze(1)
        )
        prediction = self.projection(conditioned.float())
        return prediction.transpose(1, 2).reshape(
            batch,
            self.out_channels,
            height,
            width,
        )


__all__ = ["FinalOutputHead"]
