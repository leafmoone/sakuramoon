"""Text-only access to the frozen local Qwen3.5 language model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

from sakuramoon.assets import require_local_qwen
from sakuramoon.data.serialize import CONDITION_BUCKETS, EXPECTED_PREFIX_TOKENS

HIDDEN_STATE_BLOCKS = (2, 4, 8, 12, 16, 20, 24)
QWEN_DENSE_LENGTHS = tuple(
    EXPECTED_PREFIX_TOKENS + condition_bucket
    for condition_bucket in CONDITION_BUCKETS
)
QWEN_DENSE_GROUP_SIZE = 32
_HIDDEN_STATE_INDEX_BY_BLOCK = {
    2: 2,
    4: 4,
    8: 8,
    12: 12,
    16: 16,
    20: 20,
}


class _ModelOutput(Protocol):
    hidden_states: tuple[torch.Tensor, ...] | None


@dataclass(frozen=True)
class QwenEncoderOutput:
    """Seven selected states from one Qwen forward."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class QwenRuntime:
    encoder: FrozenQwenEncoder
    tokenizer: PreTrainedTokenizerBase


class FrozenQwenEncoder(nn.Module):
    """Keep Qwen frozen and expose only the approved seven text states."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        super().train(False)

    def train(self, mode: bool = True) -> FrozenQwenEncoder:
        """A frozen encoder remains in evaluation mode."""

        del mode
        super().train(False)
        self.model.eval()
        return self

    def _forward_selected(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        layers = getattr(self.model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) != 24:
            raise RuntimeError("Qwen must expose exactly 24 decoder layers")
        raw_block_24: list[torch.Tensor] = []

        def capture_raw_block_24(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            block_output: object,
        ) -> None:
            if not isinstance(block_output, torch.Tensor):
                raise TypeError("Qwen block 24 must return one raw tensor")
            raw_block_24.append(block_output)

        handle = layers[-1].register_forward_hook(capture_raw_block_24)
        try:
            output = cast(
                _ModelOutput,
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                ),
            )
        finally:
            handle.remove()
        states = output.hidden_states
        if states is None or len(states) != 25:
            actual = None if states is None else len(states)
            raise RuntimeError(f"Qwen must return 25 hidden-state entries, got {actual}")
        if len(raw_block_24) != 1:
            raise RuntimeError(
                f"Qwen block 24 must run exactly once, got {len(raw_block_24)}"
            )
        if raw_block_24[0].shape != states[24].shape:
            raise RuntimeError("Qwen raw block 24 shape differs from final hidden state")

        selected = torch.stack(
            tuple(
                states[_HIDDEN_STATE_INDEX_BY_BLOCK[block]]
                for block in HIDDEN_STATE_BLOCKS[:-1]
            )
            + (raw_block_24[0],),
            dim=2,
        )
        if selected.shape[-1] != 2048:
            raise RuntimeError(f"Qwen hidden size must be 2048, got {selected.shape[-1]}")
        return selected.detach()

    @staticmethod
    def _group_rows(
        dense_lengths: tuple[int, ...],
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        grouped: dict[int, list[int]] = {}
        for row, length in enumerate(dense_lengths):
            grouped.setdefault(length, []).append(row)
        return tuple(
            (length, tuple(rows)) for length, rows in sorted(grouped.items())
        )

    @staticmethod
    def _chunk_rows(
        dense_lengths: tuple[int, ...],
        group_size: int,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        ordered = sorted(
            range(len(dense_lengths)),
            key=lambda row: (dense_lengths[row], row),
        )
        groups: list[tuple[int, tuple[int, ...]]] = []
        for start in range(0, len(ordered), group_size):
            rows = tuple(ordered[start : start + group_size])
            groups.append((max(dense_lengths[row] for row in rows), rows))
        return tuple(groups)

    @staticmethod
    def _validate_declared_lengths(
        attention_mask: torch.Tensor,
        dense_lengths: tuple[int, ...],
    ) -> None:
        for length, rows in FrozenQwenEncoder._group_rows(dense_lengths):
            row_indices = torch.tensor(
                rows, device=attention_mask.device, dtype=torch.long
            )
            trailing_mask_is_empty = (
                ~attention_mask.index_select(0, row_indices)[:, length:]
            ).all()
            if attention_mask.is_cuda:
                torch._assert_async(  # pyright: ignore[reportPrivateUsage,reportPrivateImportUsage]
                    trailing_mask_is_empty
                )
            elif not bool(trailing_mask_is_empty):
                raise ValueError("dense length truncates a valid Qwen token")

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        dense_lengths: tuple[int, ...] | None = None,
        dense_group_size: int = QWEN_DENSE_GROUP_SIZE,
    ) -> QwenEncoderOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, length]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if attention_mask.dtype != torch.bool:
            raise TypeError(
                "attention_mask must use torch.bool with True meaning valid token"
            )

        if dense_lengths is None:
            selected = self._forward_selected(input_ids, attention_mask)
        else:
            if (
                type(dense_lengths) is not tuple
                or len(dense_lengths) != input_ids.shape[0]
                or any(
                    type(length) is not int
                    or length not in QWEN_DENSE_LENGTHS
                    or length > input_ids.shape[1]
                    for length in dense_lengths
                )
            ):
                raise ValueError(
                    "dense_lengths must contain one supported Qwen bucket per row"
                )
            if type(dense_group_size) is not int or dense_group_size <= 0:
                raise ValueError("dense_group_size must be a positive integer")
            self._validate_declared_lengths(attention_mask, dense_lengths)
            groups = self._chunk_rows(dense_lengths, dense_group_size)
            if len(groups) == 1:
                length, _rows = groups[0]
                selected = self._forward_selected(
                    input_ids[:, :length], attention_mask[:, :length]
                )
            else:
                selected: torch.Tensor | None = None
                for length, rows in groups:
                    row_indices = torch.tensor(
                        rows, device=input_ids.device, dtype=torch.long
                    )
                    group_selected = self._forward_selected(
                        input_ids.index_select(0, row_indices)[:, :length].contiguous(),
                        attention_mask.index_select(0, row_indices)[
                            :, :length
                        ].contiguous(),
                    )
                    if selected is None:
                        selected = torch.zeros(
                            (input_ids.shape[0], input_ids.shape[1], 7, 2048),
                            device=input_ids.device,
                            dtype=group_selected.dtype,
                        )
                    selected[:, :length].index_copy_(
                        0, row_indices, group_selected
                    )
                if selected is None:
                    raise RuntimeError("Qwen dense grouping produced no output")
        return QwenEncoderOutput(
            hidden_states=selected,
            attention_mask=attention_mask,
        )


def load_local_qwen(repository_root: Path, device: torch.device) -> QwenRuntime:
    """Load the fixed local text model without creating the visual tower."""

    if device.type != "cuda":
        raise ValueError("the production Qwen encoder requires a CUDA device")
    model_path = require_local_qwen(repository_root)
    if not is_fast_path_available:
        warnings.warn(
            "Qwen3.5 FLA/causal-conv1d kernels are unavailable; using the "
            "Transformers PyTorch fallback. Training will be slower.",
            RuntimeWarning,
            stacklevel=2,
        )

    config = Qwen3_5TextConfig.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_path, local_files_only=True
    )
    if config.num_hidden_layers != 24 or config.hidden_size != 2048:
        raise RuntimeError("local Qwen text config must be 24 layers with hidden size 2048")
    config.use_cache = False

    causal_lm = Qwen3_5ForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_path,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    text_model = causal_lm.model
    text_model.to(  # pyright: ignore[reportUnknownMemberType, reportArgumentType, reportCallIssue]
        device=device
    )
    encoder = FrozenQwenEncoder(text_model)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            padding_side="right",
        ),
    )
    if tokenizer.pad_token_id != 248044:
        raise RuntimeError("Qwen padding token must be <|endoftext|> (248044)")
    return QwenRuntime(encoder=encoder, tokenizer=tokenizer)


__all__ = [
    "HIDDEN_STATE_BLOCKS",
    "QWEN_DENSE_GROUP_SIZE",
    "QWEN_DENSE_LENGTHS",
    "FrozenQwenEncoder",
    "QwenEncoderOutput",
    "QwenRuntime",
    "load_local_qwen",
]
