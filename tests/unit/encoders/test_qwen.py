from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from sakuramoon.encoders.qwen import FrozenQwenEncoder


class _FakeBlock(nn.Module):
    def __init__(self, block: int) -> None:
        super().__init__()
        self.block = block
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states + float(self.block)


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.layers = nn.ModuleList(_FakeBlock(block) for block in range(1, 25))
        self.calls = 0
        self.kwargs: dict[str, Any] = {}
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        self.kwargs = kwargs
        self.input_shapes.append(tuple(input_ids.shape))
        batch, length = input_ids.shape
        hidden_states = input_ids[:, :, None].to(torch.float32).expand(
            batch, length, 2048
        )
        states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            states.append(hidden_states)
        # Transformers 5.14.1 replaces the final captured block output with
        # last_hidden_state after final RMSNorm. Make that semantic visible.
        states[-1] = states[-1] + 1000.0
        return SimpleNamespace(hidden_states=tuple(states))


def test_selects_seven_states_from_one_frozen_forward() -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])

    output = encoder(input_ids, mask)

    assert backend.calls == 1
    assert backend.kwargs["use_cache"] is False
    assert backend.kwargs["output_hidden_states"] is True
    assert output.hidden_states.shape == (2, 3, 7, 2048)
    assert output.hidden_states[0, 0, :, 0].tolist() == [4, 11, 37, 79, 137, 211, 301]
    assert backend.layers[-1].calls == 1
    assert output.attention_mask.dtype == torch.bool
    assert output.attention_mask is mask
    assert not output.hidden_states.requires_grad
    assert not backend.weight.requires_grad
    assert not encoder.training


def test_train_does_not_enable_frozen_backend() -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)

    encoder.train()

    assert not encoder.training
    assert not backend.training


def test_groups_dense_buckets_and_restores_original_row_order() -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)
    input_ids = torch.arange(3 * 162, dtype=torch.long).reshape(3, 162)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[0, :90] = True
    mask[1, :140] = True
    mask[2, :80] = True

    output = encoder(
        input_ids,
        mask,
        dense_lengths=(98, 162, 98),
        dense_group_size=2,
    )

    assert backend.calls == 2
    assert backend.input_shapes == [(2, 98), (1, 162)]
    assert output.hidden_states.shape == (3, 162, 7, 2048)
    assert output.hidden_states[0, 0, :, 0].tolist() == [3, 10, 36, 78, 136, 210, 300]
    assert output.hidden_states[1, 140, :, 0].tolist() == [305, 312, 338, 380, 438, 512, 602]
    assert output.hidden_states[2, 97, :, 0].tolist() == [424, 431, 457, 499, 557, 631, 721]
    assert not bool(output.hidden_states[0, 98:].any())
    assert not bool(output.hidden_states[2, 98:].any())
    assert output.attention_mask is mask


def test_rejects_dense_bucket_that_truncates_valid_tokens() -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)
    input_ids = torch.ones(2, 162, dtype=torch.long)
    mask = torch.ones(2, 162, dtype=torch.bool)

    with pytest.raises(ValueError, match="truncates"):
        encoder(input_ids, mask, dense_lengths=(98, 162))

    assert backend.calls == 0


def test_rejects_invalid_inputs() -> None:
    encoder = FrozenQwenEncoder(_FakeQwen())

    with pytest.raises(TypeError, match="torch.long"):
        encoder(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool))


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[1, -1]], dtype=torch.long),
        torch.tensor([[0.0, 1.0]]),
    ),
)
def test_rejects_non_boolean_attention_mask_before_forward(
    mask: torch.Tensor,
) -> None:
    backend = _FakeQwen()
    encoder = FrozenQwenEncoder(backend)

    with pytest.raises(TypeError, match="torch.bool"):
        encoder(torch.ones(1, 2, dtype=torch.long), mask)

    assert backend.calls == 0
