from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sakuramoon.train.sampling import (
    TrainingSamplingError,
    _condition_representation_diagnostics,  # pyright: ignore[reportPrivateUsage]
)
from sakuramoon.train.step import (
    _condition_encoder_grad_norm,  # pyright: ignore[reportPrivateUsage]
)


def test_condition_representation_diagnostics_separate_active_and_null() -> None:
    tokens = torch.zeros(24, 2, 2)
    tokens[0] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    tokens[1] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    tokens[4] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    values = _condition_representation_diagnostics(tokens)

    assert values["condition_A_rms"] == pytest.approx(math.sqrt(0.5))
    assert values["condition_B_rms"] == pytest.approx(math.sqrt(0.5))
    assert values["condition_active_rms"] == pytest.approx(math.sqrt(0.5))
    assert values["condition_null_rms"] == pytest.approx(1.0)
    assert values["condition_A_B_cosine"] == pytest.approx(0.0)
    assert values["condition_A_null_cosine"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert values["condition_B_null_cosine"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert values["condition_A_B_delta_rms"] == pytest.approx(1.0)
    assert values["condition_A_null_delta_rms"] == pytest.approx(math.sqrt(0.5))


def test_condition_representation_diagnostics_require_cfg_batch() -> None:
    with pytest.raises(TrainingSamplingError, match="24 CFG branches"):
        _condition_representation_diagnostics(torch.zeros(3, 2, 2))


class _Composite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition_tokens = nn.Linear(2, 2, bias=False)
        self.other = nn.Linear(2, 2, bias=False)


def test_condition_encoder_grad_norm_uses_only_condition_parameters() -> None:
    module = _Composite()
    module.condition_tokens.weight.grad = torch.full((2, 2), 3.0)
    module.other.weight.grad = torch.full((2, 2), 100.0)

    value = _condition_encoder_grad_norm(module, device=torch.device("cpu"))

    torch.testing.assert_close(value, torch.tensor(6.0))
