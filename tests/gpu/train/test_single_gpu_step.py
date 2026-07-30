from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.optim.adamw8bit import build_adamw8bit
from sakuramoon.train.step import SingleGpuStep, SingleGpuUpdateState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class _MixedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Linear(
            64, 64, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.sensitive = nn.Parameter(torch.ones(64, device="cuda"))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        output = self.matrix(inputs.to(torch.bfloat16)).float() * self.sensitive
        return (output - targets).square().mean(dim=1)


def _optimizer(module: nn.Module, seed: int):
    return build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=seed,
    )


def test_torchao_accumulated_update_matches_merged_batch() -> None:
    torch.manual_seed(1510)  # pyright: ignore[reportUnknownMemberType]
    accumulated = _MixedModule()
    merged = _MixedModule()
    merged.load_state_dict(accumulated.state_dict())
    inputs = torch.randn(4, 64, device="cuda") * 0.1
    targets = torch.randn(4, 64, device="cuda") * 0.1
    accumulated_optimizer = _optimizer(accumulated, 1511)
    merged_optimizer = _optimizer(merged, 1511)
    accumulated_step = SingleGpuStep(
        accumulated,
        accumulated_optimizer,
        accumulation_steps=2,
        state=SingleGpuUpdateState.initial(),
    )
    merged_step = SingleGpuStep(
        merged,
        merged_optimizer,
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
    )

    accumulated_step.backward(accumulated(inputs[:1], targets[:1]))
    accumulated_step.backward(accumulated(inputs[1:], targets[1:]))
    accumulated_result = accumulated_step.finish_update()
    merged_step.backward(merged(inputs, targets))
    merged_result = merged_step.finish_update()

    torch.testing.assert_close(
        accumulated_result.mean_loss,
        merged_result.mean_loss,
        atol=1e-7,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        accumulated_result.clip.pre_clip_norm,
        merged_result.clip.pre_clip_norm,
        atol=2e-4,
        rtol=2e-3,
    )
    torch.testing.assert_close(accumulated.matrix.weight, merged.matrix.weight)
    torch.testing.assert_close(accumulated.sensitive, merged.sensitive)
    assert accumulated_result.state == SingleGpuUpdateState(1, 1, 4)
    assert merged_result.state == SingleGpuUpdateState(1, 1, 4)
    assert torch.equal(
        accumulated_optimizer.sr_rng.state,
        merged_optimizer.sr_rng.state,
    )
    assert all(parameter.grad is None for parameter in accumulated.parameters())
    assert all(parameter.grad is None for parameter in merged.parameters())
