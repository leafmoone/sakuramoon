"""Linear-time Heun solver with a final Euler interval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class VelocityFunction(Protocol):
    def __call__(self, state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class HeunResult:
    state: torch.Tensor
    nfe: int


def _evaluate_velocity(
    velocity_function: VelocityFunction,
    state: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    velocity = velocity_function(state, timestep)
    if velocity.shape != state.shape or velocity.dtype != torch.float32:
        raise ValueError("velocity function must return FP32 with the state shape")
    return velocity


@torch.inference_mode()
def heun_final_euler(
    velocity_function: VelocityFunction,
    initial_noise: torch.Tensor,
    *,
    steps: int,
) -> HeunResult:
    if steps < 2:
        raise ValueError("Heun sampling requires at least two intervals")
    if not initial_noise.is_floating_point() or initial_noise.ndim == 0:
        raise ValueError("initial_noise must be a batched floating-point tensor")
    state = initial_noise.float()
    grid = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        device=state.device,
        dtype=torch.float32,
    )
    nfe = 0
    for index in range(steps):
        timestep = grid[index].expand(state.shape[0])
        next_timestep = grid[index + 1].expand(state.shape[0])
        delta = grid[index + 1] - grid[index]
        first = _evaluate_velocity(velocity_function, state, timestep)
        nfe += 1
        predicted = state + delta * first
        if index == steps - 1:
            state = predicted
            continue
        second = _evaluate_velocity(velocity_function, predicted, next_timestep)
        nfe += 1
        state = state + 0.5 * delta * (first + second)
    if not bool(torch.isfinite(state).all().item()):
        raise FloatingPointError("velocity integration produced a nonfinite state")
    return HeunResult(state=state, nfe=nfe)


__all__ = ["HeunResult", "VelocityFunction", "heun_final_euler"]
