"""Flow matching path/solver tests (plan §5, §17.1).

Endpoint contracts:
    * v* = delta - r0 is constant in t -> both the 1-step endpoint and the
      4-step Heun endpoint must land EXACTLY on z_lr + delta (fp tolerance),
      i.e. "4-step not worse than 1-step" for the exact field.
    * v = 0 (model returns nothing) -> 1-step gives z_lr + r0; with the
      Faithful sigma=0 mix (r0 = 0) the whole path is the identity.
    * sigma sampling: 75% zero / 25% U(0.02, 0.15) (plan §5.6).
"""

from __future__ import annotations

import pytest
import torch
from anime_sr.flow.path import (
    interpolate,
    sample_sigma,
    sample_source_noise,
    target_velocity,
)
from anime_sr.flow.solver import (
    HEUN_TIMESTEPS,
    euler_trajectory,
    four_step_heun,
    heun_trajectory,
    one_step,
    step_heun,
)

SHAPE = (2, 4, 64, 64)  # (B, C, h, w) residual state


def test_interpolate_endpoints() -> None:
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    assert torch.allclose(interpolate(r0, delta, 0.0), r0)
    assert torch.allclose(interpolate(r0, delta, 1.0), delta)
    mid = interpolate(r0, delta, 0.5)
    assert torch.allclose(mid, 0.5 * (r0 + delta))
    # per-sample t (B,1,1,1) broadcast
    t = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
    out = interpolate(r0, delta, t)
    assert torch.allclose(out[0], r0[0]) and torch.allclose(out[1], delta[1])


def test_target_velocity_constant() -> None:
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    v = target_velocity(delta, r0)
    assert torch.allclose(v, delta - r0)


def test_one_step_exact_field() -> None:
    """1-step endpoint with v_fn = v* (constant) lands on z_lr + delta exactly."""
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    z_lr = torch.randn(SHAPE)
    v_fn = lambda rt, t: target_velocity(delta, r0)
    z_hat = one_step(r0, v_fn, z_lr)
    assert torch.allclose(z_hat, z_lr + delta, atol=1e-5)


def test_four_step_heun_exact_field() -> None:
    """4-step Heun with the exact (constant) field == 1-step endpoint."""
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    z_lr = torch.randn(SHAPE)
    v_fn = lambda rt, t: target_velocity(delta, r0)
    z_4 = four_step_heun(r0, v_fn, z_lr)
    z_1 = one_step(r0, v_fn, z_lr)
    assert torch.allclose(z_4, z_1, atol=1e-5)
    assert torch.allclose(z_4, z_lr + delta, atol=1e-5)


def test_zero_model_identity_fairthful() -> None:
    """v = 0 + Faithful (sigma=0 -> r0=0) => output is exactly z_lr."""
    z_lr = torch.randn(SHAPE)
    r0 = torch.zeros(SHAPE)  # sigma = 0
    v_zero = lambda rt, t: torch.zeros_like(rt)
    assert torch.allclose(one_step(r0, v_zero, z_lr), z_lr)
    assert torch.allclose(four_step_heun(r0, v_zero, z_lr), z_lr)


def test_zero_model_noisy() -> None:
    """v = 0 with r0 = sigma*eps: 1-step returns z_lr + r0 (plan §5.4)."""
    sigma = torch.tensor([0.05, 0.1])
    r0 = sample_source_noise(sigma, SHAPE, generator=torch.Generator().manual_seed(3))
    z_lr = torch.randn(SHAPE)
    v_zero = lambda rt, t: torch.zeros_like(rt)
    assert torch.allclose(one_step(r0, v_zero, z_lr), z_lr + r0)
    assert torch.allclose(four_step_heun(r0, v_zero, z_lr), z_lr + r0, atol=1e-6)


def test_step_heun_matches_euler_for_constant_field() -> None:
    r = torch.randn(SHAPE)
    v_const = torch.full(SHAPE, 1.0)
    dt = 0.25
    v_fn = lambda rt, t: v_const
    out = step_heun(r, 0.0, dt, v_fn)
    assert torch.allclose(out, r + dt * v_const, atol=1e-6)


def test_heun_timesteps() -> None:
    assert HEUN_TIMESTEPS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_sample_sigma_mix() -> None:
    gen = torch.Generator().manual_seed(7)
    sigma = sample_sigma(10000, zero_fraction=0.75, generator=gen)
    frac_zero = (sigma == 0).float().mean().item()
    assert 0.70 <= frac_zero <= 0.80, f"zero fraction {frac_zero}"
    noisy = sigma[sigma > 0]
    assert (noisy >= 0.02).all() and (noisy <= 0.15).all()
    # two independent draws: re-rolling the mix gives different zero pattern
    gen2 = torch.Generator().manual_seed(7)
    again = sample_sigma(10000, zero_fraction=0.75, generator=gen2)
    assert torch.allclose(sigma, again)  # determinism per seed


def test_sample_source_noise_zero_sigma() -> None:
    sigma = torch.zeros(2)
    r0 = sample_source_noise(sigma, SHAPE, generator=torch.Generator().manual_seed(1))
    assert r0.shape == SHAPE
    assert r0.abs().max().item() == 0.0


def test_sample_source_noise_unit_scale() -> None:
    sigma = torch.ones(512)
    gen = torch.Generator().manual_seed(5)
    r0 = sample_source_noise(sigma, (512, 4, 8, 8), generator=gen)
    assert torch.abs(r0.std() - 1.0) < 0.1
    assert torch.abs(r0.mean()) < 0.1


def test_sample_source_noise_per_sample_epsilon() -> None:
    """P0: epsilon is drawn per sample (full-batch randn), not broadcast.

    With the old spatial-only draw + expand, r0[0] == r0[1] exactly; the
    per-sample draw makes distinct batch elements (and per-sigma scaling)
    independent."""
    sigma = torch.tensor([0.05, 0.1])
    r0 = sample_source_noise(sigma, SHAPE, generator=torch.Generator().manual_seed(11))
    assert r0.shape == SHAPE
    assert not torch.equal(r0[0], r0[1])
    # per-sample scaling: r0 / sigma is the unit epsilon field (std ~ 1)
    eps = r0 / sigma.view(2, 1, 1, 1)
    assert abs(eps.std().item() - 1.0) < 0.2
    assert abs(eps.mean().item()) < 0.2
    # deterministic per seed (resume/test reproducibility)
    again = sample_source_noise(
        sigma, SHAPE, generator=torch.Generator().manual_seed(11)
    )
    assert torch.equal(r0, again)


def test_euler_trajectory_n1_equals_one_step() -> None:
    """N=1 Euler trajectory is one_step minus the z_lr offset (plan §5.4)."""
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    z_lr = torch.randn(SHAPE)
    v_fn = lambda rt, t: target_velocity(delta, r0)
    r1, states = euler_trajectory(r0, v_fn, 1)
    assert len(states) == 1
    assert torch.allclose(states[0], r1)
    assert torch.allclose(z_lr + r1, one_step(r0, v_fn, z_lr))


def test_heun_trajectory_n4_equals_four_step_heun() -> None:
    """N=4 last-euler Heun trajectory == four_step_heun (7 evaluations)."""
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    z_lr = torch.randn(SHAPE)
    v_fn = lambda rt, t: target_velocity(delta, r0)
    r4, states = heun_trajectory(r0, v_fn, 4, last_euler=True)
    assert len(states) == 4
    z_hat_ref = four_step_heun(r0, v_fn, z_lr)
    assert torch.allclose(z_lr + r4, z_hat_ref, atol=1e-6)
    assert torch.allclose(states[-1], r4)


def test_trajectory_states_land_on_exact_path() -> None:
    """With v = v* (constant) the exact path is r_t = r0 + t*v*; every
    sub-step state must land on it (Euler and last-euler Heun)."""
    r0 = torch.randn(SHAPE)
    delta = torch.randn(SHAPE)
    v_star = target_velocity(delta, r0)
    v_fn = lambda rt, t: v_star
    for solver, n in (("euler", 2), ("euler", 8), ("heun", 2), ("heun", 8)):
        if solver == "euler":
            r_final, states = euler_trajectory(r0, v_fn, n)
        else:
            r_final, states = heun_trajectory(r0, v_fn, n, last_euler=True)
        assert len(states) == n
        for k, s in enumerate(states):
            expect = r0 + ((k + 1) / n) * v_star
            assert torch.allclose(s, expect, atol=1e-5), f"{solver} n={n} k={k}"
        # endpoint: the full path r0 + v* (plan §5.4 "z_hat = z_lr + delta")
        assert torch.allclose(r_final, r0 + v_star, atol=1e-5)


def test_trajectory_zero_model() -> None:
    """v = 0: every state stays at r0 (nothing moves)."""
    r0 = torch.randn(SHAPE)
    v_zero = lambda rt, t: torch.zeros_like(rt)
    r_final, states = euler_trajectory(r0, v_zero, 4)
    assert torch.allclose(r_final, r0)
    assert all(torch.allclose(s, r0) for s in states)
    r_final, states = heun_trajectory(r0, v_zero, 3, last_euler=False)
    assert torch.allclose(r_final, r0)
    assert all(torch.allclose(s, r0) for s in states)


def test_trajectory_rejects_bad_n_steps() -> None:
    v_zero = lambda rt, t: torch.zeros(SHAPE)
    r0 = torch.zeros(SHAPE)
    with pytest.raises(ValueError):
        euler_trajectory(r0, v_zero, 0)
    with pytest.raises(ValueError):
        heun_trajectory(r0, v_zero, 0)
