"""High-level flow sampling (plan §5.4, §5.5, §17.1).

A ``VelocityModel`` is any callable ``v = model(rt, t, sigma, cond) -> v_hat``
that predicts the residual-flow velocity. The sampler binds the conditioning
(LQ latent, pixel features, sigma) and exposes the two inference modes:

* ``one_step``  — Faithful / Balanced deterministic (sigma=0) or seeded (sigma>0)
* ``four_step`` — Quality mode (Heun, last step Euler by default)

Both return the final HR latent ``z_hat = z_lr + delta_hat`` (plan §5.4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from anime_sr.flow.path import sample_source_noise
from anime_sr.flow.solver import four_step_heun, one_step

__all__ = ["FlowSampler", "VelocityModel"]


class VelocityModel(Protocol):
    """``v_hat = model(rt, t, sigma, cond)``; rt/cond shapes are model-specific."""

    def __call__(
        self,
        rt: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        cond: object,
    ) -> torch.Tensor: ...


class FlowSampler:
    """Binds a velocity model + conditioning to the residual-flow solvers."""

    def __init__(self, model: VelocityModel) -> None:
        self.model = model

    def _v_fn(
        self,
        z_lr: torch.Tensor,
        cond: object,
        sigma: torch.Tensor,
    ) -> Callable[[torch.Tensor, float], torch.Tensor]:
        def v_fn(rt: torch.Tensor, t: float) -> torch.Tensor:
            t_batch = torch.full((rt.shape[0],), t, device=rt.device, dtype=rt.dtype)
            return self.model(rt, t_batch, sigma, cond)

        return v_fn

    @staticmethod
    def _sigma_vector(z_lr: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """(B,) sigma vector: pass a tensor through, broadcast a scalar."""
        if isinstance(sigma, torch.Tensor):
            return sigma
        return torch.full((z_lr.shape[0],), sigma, device=z_lr.device, dtype=z_lr.dtype)

    def one_step(
        self,
        z_lr: torch.Tensor,
        cond: object,
        sigma: float | torch.Tensor = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Faithful (sigma=0) / seeded one-step (plan §5.4, §17.1)."""
        sigma_vec = self._sigma_vector(z_lr, sigma)
        r0 = sample_source_noise(
            sigma_vec,
            z_lr.shape,
            generator=generator,
            dtype=z_lr.dtype,
            device=z_lr.device,
        )
        return one_step(r0, self._v_fn(z_lr, cond, sigma_vec), z_lr)

    def four_step(
        self,
        z_lr: torch.Tensor,
        cond: object,
        sigma: float | torch.Tensor = 0.0,
        generator: torch.Generator | None = None,
        last_euler: bool = True,
    ) -> torch.Tensor:
        """Quality 4-step Heun (plan §5.5, §17.1)."""
        sigma_vec = self._sigma_vector(z_lr, sigma)
        r0 = sample_source_noise(
            sigma_vec,
            z_lr.shape,
            generator=generator,
            dtype=z_lr.dtype,
            device=z_lr.device,
        )
        return four_step_heun(
            r0, self._v_fn(z_lr, cond, sigma_vec), z_lr, last_euler=last_euler
        )
