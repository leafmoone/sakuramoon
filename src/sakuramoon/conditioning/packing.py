"""Padding-free per-sample sequence packing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int


@dataclass(frozen=True)
class SampleSpans:
    text: TokenSpan
    style: TokenSpan
    image: TokenSpan


@dataclass(frozen=True)
class PackedSequences:
    tokens: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    spans: tuple[SampleSpans, ...]
    image_shapes: tuple[tuple[int, int], ...]


def pack_sequences(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    style_tokens: torch.Tensor,
    image_tokens: tuple[torch.Tensor, ...],
    image_shapes: tuple[tuple[int, int], ...],
) -> PackedSequences:
    if text_tokens.ndim != 3 or text_mask.shape != text_tokens.shape[:2]:
        raise ValueError("text tokens must be [B,L,D] with a matching mask")
    if text_mask.dtype != torch.bool:
        raise TypeError("text_mask must be boolean")
    batch, _, hidden = text_tokens.shape
    if style_tokens.shape != (batch, 4, hidden):
        raise ValueError("style tokens must have shape [B,4,D]")
    if len(image_tokens) != batch or len(image_shapes) != batch or batch == 0:
        raise ValueError("one image token tensor and image shape are required per sample")

    sequences: list[torch.Tensor] = []
    spans: list[SampleSpans] = []
    offsets = [0]
    for sample in range(batch):
        image_height, image_width = image_shapes[sample]
        image = image_tokens[sample]
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image token grid dimensions must be positive")
        if image.shape != (image_height * image_width, hidden):
            raise ValueError("image token count must equal latent height times width")
        if image.device != text_tokens.device or image.dtype != text_tokens.dtype:
            raise ValueError("all packed tokens must share device and dtype")

        text = text_tokens[sample, text_mask[sample]]
        style = style_tokens[sample]
        start = offsets[-1]
        text_end = start + text.shape[0]
        style_end = text_end + 4
        image_end = style_end + image.shape[0]
        sequences.append(torch.cat((text, style, image), dim=0))
        spans.append(
            SampleSpans(
                text=TokenSpan(start, text_end),
                style=TokenSpan(text_end, style_end),
                image=TokenSpan(style_end, image_end),
            )
        )
        offsets.append(image_end)

    lengths = [offsets[index + 1] - offsets[index] for index in range(batch)]
    return PackedSequences(
        tokens=torch.cat(sequences, dim=0),
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=text_tokens.device),
        max_seqlen=max(lengths),
        spans=tuple(spans),
        image_shapes=image_shapes,
    )


def dense_reference_mask(packed: PackedSequences) -> torch.Tensor:
    """Build a block-diagonal mask for correctness tests, not production attention."""

    total = packed.tokens.shape[0]
    mask = torch.zeros(total, total, dtype=torch.bool, device=packed.tokens.device)
    for start, end in zip(packed.cu_seqlens[:-1], packed.cu_seqlens[1:], strict=True):
        mask[int(start.item()) : int(end.item()), int(start.item()) : int(end.item())] = True
    return mask


def canvas_condition(height_px: int, width_px: int) -> tuple[float, float]:
    if height_px <= 0 or width_px <= 0:
        raise ValueError("canvas dimensions must be positive")
    size_scale = 0.5 * math.log2((height_px * width_px) / float(512**2))
    aspect = math.log2(width_px / height_px)
    return size_scale, aspect


__all__ = [
    "PackedSequences",
    "SampleSpans",
    "TokenSpan",
    "canvas_condition",
    "dense_reference_mask",
    "pack_sequences",
]
