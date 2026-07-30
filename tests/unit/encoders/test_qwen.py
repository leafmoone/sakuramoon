from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from sakuramoon.encoders.qwen import FrozenQwenEncoder


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0
        self.kwargs: dict[str, Any] = {}

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        self.kwargs = kwargs
        batch, length = input_ids.shape
        states = tuple(
            torch.full((batch, length, 2048), float(index)) for index in range(25)
        )
        return SimpleNamespace(hidden_states=states)


def test_selects_seven_states_from_one_frozen_forward() -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    output = encoder(input_ids, mask)

    assert backend.calls == 1
    assert backend.kwargs["use_cache"] is False
    assert backend.kwargs["output_hidden_states"] is True
    assert output.hidden_states.shape == (2, 3, 7, 2048)
    assert output.hidden_states[0, 0, :, 0].tolist() == [2, 4, 8, 12, 16, 20, 24]
    assert output.attention_mask.dtype == torch.bool
    assert not output.hidden_states.requires_grad
    assert not backend.weight.requires_grad
    assert not encoder.training


def test_train_does_not_enable_frozen_backend() -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)

    encoder.train()

    assert not encoder.training
    assert not backend.training


def test_rejects_invalid_inputs() -> None:
    encoder = FrozenQwenEncoder(_FakeQwen())

    with pytest.raises(TypeError, match="torch.long"):
        encoder(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool))
