"""Text-only access to the frozen local Qwen3.5 language model."""

from __future__ import annotations

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

HIDDEN_STATE_BLOCKS = (2, 4, 8, 12, 16, 20, 24)
_HIDDEN_STATE_INDEX_BY_BLOCK = {
    2: 2,
    4: 4,
    8: 8,
    12: 12,
    16: 16,
    20: 20,
    24: 24,
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

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> QwenEncoderOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, length]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if attention_mask.dtype not in (torch.bool, torch.long):
            raise TypeError("attention_mask must use torch.bool or torch.long")

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
        states = output.hidden_states
        if states is None or len(states) != 25:
            actual = None if states is None else len(states)
            raise RuntimeError(f"Qwen must return 25 hidden-state entries, got {actual}")

        selected = torch.stack(
            tuple(states[_HIDDEN_STATE_INDEX_BY_BLOCK[block]] for block in HIDDEN_STATE_BLOCKS),
            dim=2,
        )
        if selected.shape[-1] != 2048:
            raise RuntimeError(f"Qwen hidden size must be 2048, got {selected.shape[-1]}")
        return QwenEncoderOutput(
            hidden_states=selected.detach(),
            attention_mask=attention_mask.to(dtype=torch.bool),
        )


def load_local_qwen(repository_root: Path, device: torch.device) -> QwenRuntime:
    """Load the fixed local text model without creating the visual tower."""

    if device.type != "cuda":
        raise ValueError("the production Qwen encoder requires a CUDA device")
    model_path = require_local_qwen(repository_root)
    if not is_fast_path_available:
        raise RuntimeError("Qwen3.5 fast linear-attention kernels are unavailable")

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
    "FrozenQwenEncoder",
    "QwenEncoderOutput",
    "QwenRuntime",
    "load_local_qwen",
]
