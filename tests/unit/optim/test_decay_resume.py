from __future__ import annotations

import copy
from typing import cast

import pytest
import torch
from torch import nn

from sakuramoon.checkpoint.load import (
    CheckpointError,
    _validate_optimizer_state,  # pyright: ignore[reportPrivateUsage]
)
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.groups import audit_trainable_parameters


class _PolicyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.norm = nn.Parameter(torch.ones(64, dtype=torch.float32))


def _optimizer_wrapper() -> IsolatedAdamW8bit:
    module = _PolicyModule()
    audit = audit_trainable_parameters(
        module,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [module.matrix.weight],
                "param_names": ["matrix.weight"],
                "group_name": "matrix_decay",
                "weight_decay": 0.0,
            },
            {
                "params": [module.norm],
                "param_names": ["norm"],
                "group_name": "sensitive_no_decay",
                "weight_decay": 0.0,
            },
        ],
        lr=5e-5,
        betas=(0.9, 0.95),
    )

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper.optimizer = optimizer
    wrapper.audit = audit
    return cast(IsolatedAdamW8bit, wrapper)


def test_checkpoint_restore_allows_current_weight_decay_override() -> None:
    wrapper = _optimizer_wrapper()
    saved = copy.deepcopy(wrapper.optimizer.state_dict())
    saved["param_groups"][0]["weight_decay"] = 0.01

    _validate_optimizer_state(saved, wrapper, successful_updates=0)


def test_checkpoint_restore_keeps_non_hyperparameter_fields_strict() -> None:
    wrapper = _optimizer_wrapper()
    saved = copy.deepcopy(wrapper.optimizer.state_dict())
    saved["param_groups"][0]["betas"] = (0.8, 0.95)

    with pytest.raises(CheckpointError, match="betas"):
        _validate_optimizer_state(saved, wrapper, successful_updates=0)


def test_checkpoint_restore_allows_lazy_state_for_new_parameters() -> None:
    wrapper = _optimizer_wrapper()
    saved = copy.deepcopy(wrapper.optimizer.state_dict())
    saved["state"][1] = {
        "step": torch.tensor(10.0, dtype=torch.float32),
        "exp_avg": torch.zeros(64, dtype=torch.float32),
        "exp_avg_sq": torch.zeros(64, dtype=torch.float32),
    }

    _validate_optimizer_state(saved, wrapper, successful_updates=10)
    wrapper.optimizer.load_state_dict(saved)

    specs = {spec.name: spec for spec in wrapper.audit.specs}
    assert specs["norm"].parameter in wrapper.optimizer.state
    assert specs["matrix.weight"].parameter not in wrapper.optimizer.state
