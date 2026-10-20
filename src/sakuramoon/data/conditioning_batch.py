"""Shared tensor batching for serialized caption condition inputs.

The three conditioning entry points (training sampling, evaluation,
inference) build the same nine tensors from ``SerializedCaption`` rows.
These primitives keep the exact pads, dtypes, layout, and row order of
the original implementations; the entry wrappers keep their own error
policies and call timing, so inference still runs the Qwen encoder
between token padding and index materialization.
"""

from __future__ import annotations

import torch

from sakuramoon.data.serialize import SerializedCaption


def token_rows(
    captions: tuple[SerializedCaption, ...],
    *,
    pad_token_id: int,
    device: torch.device,
    on_overflow: tuple[type[Exception], str] | None = None,
    copy_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense padded token grid: long input IDs and a bool attention mask.

    With ``on_overflow`` set, a row longer than the dense length raises the
    given exception at the row's copy position; otherwise the raw torch
    assignment error surfaces.  No pre-scan is performed, so earlier rows
    keep their original fault priority.  ``copy_dtype`` preserves each
    caller's original copy semantics: the training path copied with
    explicit ``torch.long`` while evaluation and inference relied on torch
    dtype inference (including their distinct malformed-element errors);
    the padded grid itself always stays long/bool.
    """
    length = max(item.dense_length for item in captions)
    input_ids = torch.full(
        (len(captions), length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(captions), length), dtype=torch.bool, device=device
    )
    for row, caption in enumerate(captions):
        count = len(caption.input_ids)
        if on_overflow is not None and count > length:
            raise on_overflow[0](on_overflow[1])
        input_ids[row, :count] = torch.tensor(
            caption.input_ids, dtype=copy_dtype, device=device
        )
        attention_mask[row, :count] = True
    return input_ids, attention_mask


def index_rows(
    values: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    copy_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-aligned long index grid with -1 padding and a bool mask.

    ``copy_dtype`` mirrors :func:`token_rows`: the original training caller
    copied with explicit ``torch.long``; evaluation and inference relied on
    torch dtype inference and must keep that error surface.
    """
    width = max((len(item) for item in values), default=0)
    indices = torch.full((len(values), width), -1, dtype=torch.long, device=device)
    mask = torch.zeros((len(values), width), dtype=torch.bool, device=device)
    for row, item in enumerate(values):
        if item:
            indices[row, : len(item)] = torch.tensor(
                item, dtype=copy_dtype, device=device
            )
            mask[row, : len(item)] = True
    return indices, mask


def condition_flags(
    captions: tuple[SerializedCaption, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The use-null flag row and the active (non-null) sample indices."""
    use_null = torch.tensor(
        tuple(item.use_null_condition for item in captions),
        dtype=torch.bool,
        device=device,
    )
    active = torch.tensor(
        tuple(
            index for index, item in enumerate(captions) if not item.use_null_condition
        ),
        dtype=torch.long,
        device=device,
    )
    return use_null, active


def conditioning_batch(
    captions: tuple[SerializedCaption, ...],
    *,
    pad_token_id: int,
    device: torch.device,
    on_overflow: tuple[type[Exception], str] | None = None,
    copy_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """The full nine-tensor conditioning batch in the established order.

    The order is (input_ids, attention_mask, main_indices, main_mask,
    main_lengths, condition_indices, condition_mask, use_null,
    active_indices); main_lengths stays a tuple of ints, not a tensor.
    """
    input_ids, attention_mask = token_rows(
        captions,
        pad_token_id=pad_token_id,
        device=device,
        on_overflow=on_overflow,
        copy_dtype=copy_dtype,
    )
    main_indices, main_mask = index_rows(
        tuple(item.main_token_indices for item in captions),
        device=device,
        copy_dtype=copy_dtype,
    )
    condition_indices, condition_mask = index_rows(
        tuple(item.condition_token_indices for item in captions),
        device=device,
        copy_dtype=copy_dtype,
    )
    use_null, active = condition_flags(captions, device=device)
    return (
        input_ids,
        attention_mask,
        main_indices,
        main_mask,
        tuple(len(item.main_token_indices) for item in captions),
        condition_indices,
        condition_mask,
        use_null,
        active,
    )
