"""Learned embeddings for text, condition, and image token modalities."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

Modality = Literal["text", "condition", "image"]


class ModalityEmbedding(nn.Module):
    def __init__(self, hidden_size: int, init_std: float) -> None:
        super().__init__()
        if hidden_size <= 0 or init_std <= 0.0:
            raise ValueError("hidden_size and init_std must be positive")
        self.text = nn.Parameter(torch.empty(hidden_size))
        self.condition = nn.Parameter(torch.empty(hidden_size))
        self.image = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.text, std=init_std)
        nn.init.normal_(self.condition, std=init_std)
        nn.init.normal_(self.image, std=init_std)

    def forward(self, tokens: torch.Tensor, modality: Modality) -> torch.Tensor:
        if tokens.ndim < 2 or tokens.shape[-1] != self.text.numel():
            raise ValueError("tokens must end in the configured hidden size")
        if modality not in ("text", "condition", "image"):
            raise ValueError("modality must be text, condition, or image")
        embedding = getattr(self, modality)
        return tokens + embedding.to(dtype=tokens.dtype, device=tokens.device)


__all__ = ["Modality", "ModalityEmbedding"]
