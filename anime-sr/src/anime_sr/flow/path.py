"""LQ-centered residual flow (plan §5): path math and source-state sampling.

Convention (plan §5.1-§5.3, §21):
    delta = z_hr - z_lr                    (residual target, never full z_hr)
    r0    = sigma * epsilon                (source state; sigma=0 -> zero)
    rt    = (1 - t) * r0 + t * delta       (flow path, t in [0, 1])
    v*    = delta - r0                     (target velocity, constant in t)

The residual path endpoint at t=1 is delta itself (not z_lr+delta); the final
latent is z_hr = z_lr + delta. See solver.py for the integrators and sampling.py
for the 1-step / 4-step convenience wrappers.
"""

from __future__ import annotations

import torch

__all__ = [
    "interpolate",
    "sample_sigma",
    "sample_source_noise",
    "target_velocity",
]


def interpolate(r0: torch.Tensor, delta: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    """rt = (1 - t) * r0 + t * delta (plan §5.3)."""
    if isinstance(t, torch.Tensor):
        t = t.to(r0.dtype)
        if t.dim() == 1:
            t = t.view(-1, *([1] * (r0.ndim - 1)))
    return (1.0 - t) * r0 + t * delta


def target_velocity(delta: torch.Tensor, r0: torch.Tensor) -> torch.Tensor:
    """v* = delta - r0 (plan §5.3); constant along the linear path."""
    return delta - r0


def sample_sigma(
    batch: int,
    zero_fraction: float,
    sigma_range: tuple[float, float] = (0.02, 0.15),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-sample sigma mix (plan §5.6): ``zero_fraction`` of samples get 0, the rest
    get Uniform(sigma_range). Two independent Uniform draws: one for the mix, one for
    the noisy value (keeps the draw count fixed regardless of the outcome)."""
    if not 0.0 <= zero_fraction <= 1.0:
        raise ValueError(f"zero_fraction must be in [0, 1], got {zero_fraction}")
    u_mix = torch.rand(batch, device=device, dtype=torch.float32, generator=generator)
    is_zero = u_mix < zero_fraction
    lo, hi = sigma_range
    u_val = torch.rand(batch, device=device, dtype=torch.float32, generator=generator)
    noisy = (lo + (hi - lo) * u_val).to(dtype)
    return torch.where(is_zero, torch.zeros((), device=device, dtype=dtype), noisy)


def sample_source_noise(
    sigma: torch.Tensor,
    shape: tuple[int, ...],
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """r0 = sigma * epsilon, epsilon ~ N(0, I) (plan §5.2).

    sigma: (B,) -> r0: (B, *rest) where rest = shape[1:]."""
    eps = torch.randn(shape[1:], device=device, dtype=dtype, generator=generator).expand(shape)
    return sigma.view(-1, *([1] * (len(shape) - 1))) * eps
