from __future__ import annotations

import torch

from sakuramoon.objective.flow import (
    flow_matching_loss,
    guided_velocity,
    interpolate_state,
    sample_jlt_timesteps,
    sample_noise,
    x_prediction_to_velocity,
)


def test_jlt_and_noise_sampling_are_explicit_and_deterministic() -> None:
    generator_a = torch.Generator(device="cpu")
    generator_a.manual_seed(123)
    generator_b = torch.Generator(device="cpu")
    generator_b.manual_seed(123)

    timestep_a = sample_jlt_timesteps(
        8,
        p_mean=-0.8,
        p_std=0.8,
        device=torch.device("cpu"),
        generator=generator_a,
    )
    timestep_b = sample_jlt_timesteps(
        8,
        p_mean=-0.8,
        p_std=0.8,
        device=torch.device("cpu"),
        generator=generator_b,
    )
    clean = torch.zeros(2, 3)
    noise = sample_noise(clean, noise_scale=1.0, generator=generator_a)

    torch.testing.assert_close(timestep_a, timestep_b)
    assert ((timestep_a > 0.0) & (timestep_a < 1.0)).all()
    assert noise.shape == clean.shape


def test_interpolation_endpoints_are_noise_then_clean() -> None:
    clean = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
    noise = torch.tensor([[-2.0, -4.0], [-3.0, -5.0]])

    state = interpolate_state(
        clean,
        noise,
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )

    torch.testing.assert_close(state[0], noise[0])
    torch.testing.assert_close(state[1], clean[1])


def test_exact_x_prediction_has_zero_velocity_loss() -> None:
    clean = torch.randn(2, 3, 4, 4)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0.25, 0.9], dtype=torch.float32)
    state = interpolate_state(clean, noise, timestep)
    denominator = (1.0 - timestep).clamp_min(0.05).reshape(2, 1, 1, 1)
    exact_prediction = state + denominator * (clean - noise)

    result = flow_matching_loss(
        exact_prediction,
        state,
        clean,
        noise,
        timestep,
        t_eps=0.05,
    )

    assert result.loss.dtype == torch.float32
    torch.testing.assert_close(result.per_sample, torch.zeros(2), atol=1e-12, rtol=0)


def test_velocity_loss_uses_t_eps_above_point_nine_five() -> None:
    clean = torch.tensor([[3.0, -1.0]])
    noise = torch.tensor([[-2.0, 4.0]])
    timestep = torch.tensor([0.99], dtype=torch.float32)
    state = interpolate_state(clean, noise, timestep)
    exact_prediction = state + 0.05 * (clean - noise)

    result = flow_matching_loss(
        exact_prediction,
        state,
        clean,
        noise,
        timestep,
        t_eps=0.05,
    )

    torch.testing.assert_close(result.loss, torch.tensor(0.0), atol=1e-10, rtol=0)


def test_loss_reduces_each_sample_before_global_mean() -> None:
    state = torch.zeros(2, 1, 1, 2)
    prediction = torch.tensor([[[[1.0, 1.0]]], [[[2.0, 2.0]]]])
    clean = torch.zeros_like(state)
    noise = torch.zeros_like(state)
    timestep = torch.zeros(2, dtype=torch.float32)

    result = flow_matching_loss(
        prediction,
        state,
        clean,
        noise,
        timestep,
        t_eps=0.05,
    )

    torch.testing.assert_close(result.per_sample, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(result.loss, torch.tensor(2.5))


def test_cfg_converts_each_x_prediction_before_guidance() -> None:
    state = torch.tensor([[1.0, 2.0]])
    timestep = torch.tensor([0.99], dtype=torch.float32)
    conditional = torch.tensor([[2.0, 4.0]])
    unconditional = torch.tensor([[0.0, 1.0]])

    guided = guided_velocity(
        conditional,
        unconditional,
        state,
        timestep,
        t_eps=0.05,
        guidance_scale=2.9,
    )
    conditional_velocity = x_prediction_to_velocity(
        conditional,
        state,
        timestep,
        t_eps=0.05,
    )
    unconditional_velocity = x_prediction_to_velocity(
        unconditional,
        state,
        timestep,
        t_eps=0.05,
    )

    torch.testing.assert_close(
        guided,
        unconditional_velocity + 2.9 * (conditional_velocity - unconditional_velocity),
    )
