from __future__ import annotations

import math

import torch

from sakuramoon.sampling.heun import heun_final_euler


def test_heun_50_uses_99_evaluations_and_fp32_state() -> None:
    calls = 0

    def exponential_velocity(
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        assert timestep.shape == (state.shape[0],)
        return state

    result = heun_final_euler(
        exponential_velocity,
        torch.ones(1, 1, dtype=torch.bfloat16),
        steps=50,
    )

    assert result.nfe == 99
    assert calls == 99
    assert result.state.dtype == torch.float32
    torch.testing.assert_close(
        result.state,
        torch.tensor([[math.e]]),
        atol=0.002,
        rtol=0,
    )


def test_constant_velocity_is_exact_and_deterministic() -> None:
    def constant_velocity(
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        return torch.full_like(state, 2.0)

    first = heun_final_euler(
        constant_velocity,
        torch.zeros(2, 3),
        steps=50,
    )
    second = heun_final_euler(
        constant_velocity,
        torch.zeros(2, 3),
        steps=50,
    )

    torch.testing.assert_close(first.state, torch.full((2, 3), 2.0))
    torch.testing.assert_close(first.state, second.state)


def test_sampler_disables_autograd_for_all_evaluations() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def parameterized_velocity(
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        assert not torch.is_grad_enabled()
        return state * parameter

    result = heun_final_euler(
        parameterized_velocity,
        torch.ones(1, 1, requires_grad=True),
        steps=2,
    )

    assert result.state.requires_grad is False
    assert result.state.grad_fn is None
