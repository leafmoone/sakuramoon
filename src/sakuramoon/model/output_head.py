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
            1.0 + final_scale.to(normalized.dtype).unsqueeze(1)
        ) * normalized + final_shift.to(normalized.dtype).unsqueeze(1)
        prediction = self.projection(conditioned.float())
        return prediction.transpose(1, 2).reshape(
            batch,
            self.out_channels,
            height,
            width,
        )

    def forward_packed(
        self,
        image_hidden: torch.Tensor,
        image_sample_indices: torch.Tensor,
        final_scale: torch.Tensor,
        final_shift: torch.Tensor,
        image_shapes: tuple[tuple[int, int], ...],
    ) -> tuple[torch.Tensor, ...]:
        """Project concatenated image spans and restore each sample grid."""

        if image_hidden.ndim != 2 or image_hidden.shape[-1] != self.hidden_size:
            raise ValueError("packed image_hidden must have shape [I,hidden_size]")
        batch = len(image_shapes)
        if batch == 0:
            raise ValueError("image_shapes must contain at least one sample")
        if final_scale.shape != (batch, self.hidden_size) or final_shift.shape != (
            batch,
            self.hidden_size,
        ):
            raise ValueError("final scale and shift must have shape [B,hidden_size]")
        if (
            image_sample_indices.shape != image_hidden.shape[:1]
            or image_sample_indices.dtype != torch.int64
            or image_sample_indices.device != image_hidden.device
        ):
            raise ValueError(
                "image_sample_indices must be int64 [I] on the image device"
            )
        image_lengths = tuple(height * width for height, width in image_shapes)
        if any(height <= 0 or width <= 0 for height, width in image_shapes):
            raise ValueError("image grid dimensions must be positive")
        if sum(image_lengths) != image_hidden.shape[0]:
            raise ValueError("packed image token count must equal all image grid areas")

        normalized = self.norm(image_hidden)
        scale = final_scale.index_select(0, image_sample_indices).to(normalized.dtype)
        shift = final_shift.index_select(0, image_sample_indices).to(normalized.dtype)
        prediction = self.projection(((1.0 + scale) * normalized + shift).float())
        samples = prediction.split(image_lengths)
        return tuple(
            sample.transpose(0, 1).reshape(self.out_channels, height, width)
            for sample, (height, width) in zip(samples, image_shapes, strict=True)
        )


__all__ = ["FinalOutputHead"]
