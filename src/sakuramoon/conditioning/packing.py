"""Padding-free per-sample sequence packing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate

import torch


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int


@dataclass(frozen=True)
class SampleSpans:
    text: TokenSpan
    condition: TokenSpan
    image: TokenSpan


_VALIDATED_BOUNDARY_CAPABILITY = object()


@dataclass(frozen=True, init=False)
class ValidatedCuSeqlens:
    """Sequence boundaries derived once from validated host-side lengths."""

    __tensor: torch.Tensor
    __capability: object
    sequence_lengths: tuple[int, ...]
    total_tokens: int
    max_seqlen: int
    batch_size: int

    def __init__(
        self,
        sequence_lengths: tuple[int, ...],
        *,
        device: torch.device,
    ) -> None:
        if type(sequence_lengths) is not tuple or not sequence_lengths or any(
            type(length) is not int or length <= 0 for length in sequence_lengths
        ):
            raise ValueError(
                "sequence lengths must be a nonempty tuple of positive integers"
            )
        offsets = (0, *accumulate(sequence_lengths))
        object.__setattr__(
            self,
            "_ValidatedCuSeqlens__tensor",
            torch.tensor(offsets, dtype=torch.int32, device=device),
        )
        object.__setattr__(
            self,
            "_ValidatedCuSeqlens__capability",
            _VALIDATED_BOUNDARY_CAPABILITY,
        )
        object.__setattr__(self, "sequence_lengths", sequence_lengths)
        object.__setattr__(self, "total_tokens", offsets[-1])
        object.__setattr__(self, "max_seqlen", max(sequence_lengths))
        object.__setattr__(self, "batch_size", len(sequence_lengths))

    @property
    def tensor(self) -> torch.Tensor:
        """Return a diagnostic snapshot without exposing mutable kernel state."""

        return self.__tensor.detach().clone()

    def _tensor_for_packed_entry(self) -> torch.Tensor:
        if (
            getattr(self, "_ValidatedCuSeqlens__capability", None)
            is not _VALIDATED_BOUNDARY_CAPABILITY
        ):
            raise TypeError("invalid validated-boundary capability")
        return self.__tensor


def build_validated_cu_seqlens(
    sequence_lengths: tuple[int, ...],
    *,
    device: torch.device,
) -> ValidatedCuSeqlens:
    return ValidatedCuSeqlens(sequence_lengths, device=device)


def validated_cu_seqlens_for_packed_entry(
    boundaries: ValidatedCuSeqlens,
    *,
    total_tokens: int,
    batch_size: int,
    device: torch.device,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Accept only constructor-sealed offsets without synchronizing device values."""

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
    tensor = boundaries._tensor_for_packed_entry()
    device_index = getattr(device, "index", None)
    if (
        tensor.ndim != 1
        or tensor.shape != (len(offsets),)
        or tensor.dtype != torch.int32
        or not tensor.is_contiguous()
        or tensor.device.type != device.type
        or (
            device_index is not None
            and tensor.device.index != device_index
        )
    ):
        raise ValueError("validated boundaries contain inconsistent tensor metadata")
    return lengths, tensor


@dataclass(frozen=True)
class PackedSequences:
    tokens: torch.Tensor
    boundaries: ValidatedCuSeqlens
    spans: tuple[SampleSpans, ...]
    image_shapes: tuple[tuple[int, int], ...]

    @property
    def cu_seqlens(self) -> torch.Tensor:
        return self.boundaries.tensor

    @property
    def max_seqlen(self) -> int:
        return self.boundaries.max_seqlen


def pack_sequences(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
    text_lengths: tuple[int, ...],
    condition_tokens: torch.Tensor,
    condition_token_count: int,
    image_tokens: tuple[torch.Tensor, ...],
    image_shapes: tuple[tuple[int, int], ...],
) -> PackedSequences:
    if text_tokens.ndim != 3 or text_mask.shape != text_tokens.shape[:2]:
        raise ValueError("text tokens must be [B,L,D] with a matching mask")
    if text_mask.dtype != torch.bool:
        raise TypeError("text_mask must be boolean")
    batch, _, hidden = text_tokens.shape
    if type(condition_token_count) is not int or condition_token_count <= 0:
        raise ValueError("condition_token_count must be a positive integer")
    if condition_tokens.shape != (batch, condition_token_count, hidden):
        raise ValueError(
            "condition tokens must have shape [B,condition_token_count,D]"
        )
    if (
        type(text_lengths) is not tuple
        or len(text_lengths) != batch
        or any(
            type(length) is not int
            or length <= 0
            or length > text_tokens.shape[1]
            for length in text_lengths
        )
    ):
        raise ValueError("text_lengths must contain one valid host length per sample")
    if text_mask.device != text_tokens.device:
        raise ValueError("text tokens and mask must share one device")
    if (
        condition_tokens.device != text_tokens.device
        or condition_tokens.dtype != text_tokens.dtype
    ):
        raise ValueError("text and condition tokens must share one device and dtype")
    if not torch.is_floating_point(text_tokens):
        raise TypeError("packed tokens must use a floating dtype")
    if len(image_tokens) != batch or len(image_shapes) != batch or batch == 0:
        raise ValueError("one image token tensor and image shape are required per sample")

    length_tensor = torch.tensor(
        text_lengths,
        dtype=torch.int64,
        device=text_tokens.device,
    )
    expected_mask = (
        torch.arange(text_tokens.shape[1], device=text_tokens.device)[None]
        < length_tensor[:, None]
    )
    mask_matches_lengths = (text_mask == expected_mask).all()
    if text_tokens.is_cuda:
        torch._assert_async(  # pyright: ignore[reportPrivateUsage,reportPrivateImportUsage]
            mask_matches_lengths
        )
    elif not bool(mask_matches_lengths):
        raise ValueError("text_mask must be a contiguous prefix matching text_lengths")

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

        text = text_tokens[sample, : text_lengths[sample]]
        condition = condition_tokens[sample]
        start = offsets[-1]
        text_end = start + text.shape[0]
        condition_end = text_end + condition_token_count
        image_end = condition_end + image.shape[0]
        sequences.append(torch.cat((text, condition, image), dim=0))
        spans.append(
            SampleSpans(
                text=TokenSpan(start, text_end),
                condition=TokenSpan(text_end, condition_end),
                image=TokenSpan(condition_end, image_end),
            )
        )
        offsets.append(image_end)

    lengths = tuple(offsets[index + 1] - offsets[index] for index in range(batch))
    return PackedSequences(
        tokens=torch.cat(sequences, dim=0),
        boundaries=build_validated_cu_seqlens(lengths, device=text_tokens.device),
        spans=tuple(spans),
        image_shapes=image_shapes,
    )


def dense_reference_mask(packed: PackedSequences) -> torch.Tensor:
    """Build a block-diagonal mask for correctness tests, not production attention."""

    total = packed.tokens.shape[0]
    mask = torch.zeros(total, total, dtype=torch.bool, device=packed.tokens.device)
    for spans in packed.spans:
        start = spans.text.start
        end = spans.image.end
        mask[start:end, start:end] = True
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
    "ValidatedCuSeqlens",
    "build_validated_cu_seqlens",
    "canvas_condition",
    "dense_reference_mask",
    "pack_sequences",
    "validated_cu_seqlens_for_packed_entry",
]
