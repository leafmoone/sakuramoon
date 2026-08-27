"""Flow integrators (plan §5.4, §5.5, §17.1).

The velocity field ``v_fn(rt, t)`` maps a residual state (shape (B,128,h,w))
and a scalar timestep to the model's v-prediction. All integrators return the
final *latent* ``z_hat = z_lr + r_final`` (plan §5.4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch

__all__ = [
    "HEUN_TIMESTEPS",
    "VelocityFn",
    "euler_trajectory",
    "four_step_heun",
    "heun_trajectory",
    "one_step",
    "step_euler",
    "step_heun",
]

#: 4-step solver time points (plan §5.5).
HEUN_TIMESTEPS: tuple[float, float, float, float, float] = (0.0, 0.25, 0.5, 0.75, 1.0)

VelocityFn: TypeAlias = Callable[[torch.Tensor, float], torch.Tensor]


def step_euler(r: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """r_{i+1} = r_i + dt * v (plan §5.4 / §5.5 last-step fallback)."""
    return r + dt * v


def step_heun(
    r: torch.Tensor,
    t: float,
    dt: float,
    v_fn: VelocityFn,
) -> torch.Tensor:
    """Heun predictor-corrector over [t, t+dt]; two velocity evaluations."""
    v0 = v_fn(r, t)
    r_pred = step_euler(r, v0, dt)
    v1 = v_fn(r_pred, t + dt)
    return r + 0.5 * dt * (v0 + v1)


def one_step(r0: torch.Tensor, v_fn: VelocityFn, z_lr: torch.Tensor) -> torch.Tensor:
    """One-step inference (plan §5.4): delta_hat = r0 + v(r0, 0); z_hat = z_lr + delta_hat.

    Faithful mode uses sigma=0 so r0=0 and the result is fully deterministic.
    """
    delta_hat = r0 + v_fn(r0, 0.0)
    return z_lr + delta_hat


def four_step_heun(
    r0: torch.Tensor,
    v_fn: VelocityFn,
    z_lr: torch.Tensor,
    last_euler: bool = True,
) -> torch.Tensor:
    """4-step Heun over HEUN_TIMESTEPS (plan §5.5).

    Three Heun intervals [0,.25], [.25,.5], [.5,.75] plus the last interval
    [.75,1.0] which degrades to Euler by default (saves one network evaluation:
    7 evaluations instead of 8).
    """
    r = r0
    for t in HEUN_TIMESTEPS[:3]:
        r = step_heun(r, t, 0.25, v_fn)
    if last_euler:
        r = step_euler(r, v_fn(r, 0.75), 0.25)
    else:
        r = step_heun(r, 0.75, 0.25, v_fn)
    # r is the integrated residual endpoint (= delta_hat); final latent (plan §5.4).
    return z_lr + r


def euler_trajectory(
    r0: torch.Tensor, v_fn: VelocityFn, n_steps: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Generic N-step Euler over [0, 1] (plan §5.4); the sweep generalization.

    Splits [0, 1] into ``n_steps`` equal sub-steps (``dt = 1 / n_steps``).
    Returns ``(r_final, states)`` where ``states[k]`` is the residual state
    after sub-step ``k + 1`` (at ``t = (k + 1) / n_steps``). The caller forms
    the final latent as ``z_lr + r_final`` (plan §5.4). ``n_steps = 1``
    reduces to :func:`one_step` with ``r0`` already the source state.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    dt = 1.0 / n_steps
    r = r0
    states: list[torch.Tensor] = []
    for k in range(n_steps):
        r = step_euler(r, v_fn(r, k * dt), dt)
        states.append(r)
    return r, states


def heun_trajectory(
    r0: torch.Tensor,
    v_fn: VelocityFn,
    n_steps: int,
    last_euler: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Generic N-step Heun over [0, 1] (plan §5.5 generalization, sweep use).

    ``n_steps`` equal sub-steps (``dt = 1 / n_steps``), each a Heun
    predictor-corrector (two velocity evaluations), the last sub-step
    degrading to Euler when ``last_euler`` (saves one evaluation). Returns
    ``(r_final, states)`` with ``states[k]`` at ``t = (k + 1) / n_steps``.
    With ``n_steps = 4, last_euler = True`` this matches :func:`four_step_heun`
    (7 evaluations, states at 0.25 / 0.5 / 0.75 / 1.0).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    dt = 1.0 / n_steps
    r = r0
    states: list[torch.Tensor] = []
    for k in range(n_steps):
        t = k * dt
        v0 = v_fn(r, t)
        r_pred = step_euler(r, v0, dt)
        if k == n_steps - 1 and last_euler:
            r = r_pred  # Euler final sub-step: one evaluation
        else:
            v1 = v_fn(r_pred, t + dt)
            r = r + 0.5 * dt * (v0 + v1)
        states.append(r)
    return r, states
