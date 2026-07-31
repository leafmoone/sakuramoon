from __future__ import annotations

from collections.abc import Callable

import pytest
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

    torch.testing.assert_close(
        timestep_a,
        torch.tensor(
            [
                0.2912803888320923,
                0.3309902548789978,
                0.25054502487182617,
                0.27045124769210815,
                0.14709876477718353,
                0.34692472219467163,
                0.17109538614749908,
                0.19717638194561005,
            ]
        ),
    )
    torch.testing.assert_close(timestep_a, timestep_b)
    torch.testing.assert_close(
        noise,
        torch.tensor(
            [
                [0.32390275597572327, -0.10852263122797012, 0.21033115684986115],
                [-0.39084282517433167, 0.23497341573238373, 0.6652604341506958],
            ]
        ),
    )
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

    result = flow_matching_loss(
        clean,
        state,
        clean,
        timestep,
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )

    assert result.loss.dtype == torch.float32
    torch.testing.assert_close(result.per_sample, torch.zeros(2), atol=1e-12, rtol=0)


def test_velocity_loss_uses_t_eps_above_point_nine_five() -> None:
    clean = torch.tensor([[3.0, -1.0]])
    noise = torch.tensor([[-2.0, 4.0]])
    timestep = torch.tensor([0.99], dtype=torch.float32)
    state = interpolate_state(clean, noise, timestep)

    result = flow_matching_loss(
        clean,
        state,
        clean,
        timestep,
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )

    torch.testing.assert_close(result.loss, torch.tensor(0.0), atol=1e-10, rtol=0)


def test_loss_reduces_each_sample_before_global_mean() -> None:
    state = torch.zeros(2, 1, 1, 2)
    prediction = torch.tensor([[[[1.0, 1.0]]], [[[2.0, 2.0]]]])
    clean = torch.zeros_like(state)
    timestep = torch.zeros(2, dtype=torch.float32)

    result = flow_matching_loss(
        prediction,
        state,
        clean,
        timestep,
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )

    torch.testing.assert_close(result.per_sample, torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(result.loss, torch.tensor(2.5))


def test_strict_jlt_loss_is_x_error_times_inverse_square_weight() -> None:
    clean = torch.tensor([[2.0, -1.0], [3.0, 4.0]])
    noise = torch.tensor([[-2.0, 5.0], [1.0, -3.0]])
    timestep = torch.tensor([0.25, 0.99], dtype=torch.float32)
    state = interpolate_state(clean, noise, timestep)
    prediction = clean + torch.tensor([[0.75, -1.5], [0.05, -0.10]])
    denominator = (1.0 - timestep).clamp_min(0.05).reshape(2, 1)

    result = flow_matching_loss(
        prediction,
        state,
        clean,
        timestep,
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )
    expected = ((prediction - clean).float() / denominator).square().mean(dim=1)

    torch.testing.assert_close(result.per_sample, expected)
    torch.testing.assert_close(result.loss, expected.mean())
    torch.testing.assert_close(result.per_sample[1], torch.tensor(2.5))


def test_endpoint_weight_is_clamped_to_four_hundred() -> None:
    clean = torch.zeros(1, 2)
    state = torch.zeros_like(clean)
    prediction = torch.ones_like(clean)

    result = flow_matching_loss(
        prediction,
        state,
        clean,
        torch.tensor([0.99], dtype=torch.float32),
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )

    torch.testing.assert_close(result.loss, torch.tensor(400.0))


def test_noise_observation_boundary_is_point_nine_five_with_sum_count_outputs() -> None:
    clean = torch.zeros(2, 1)
    state = torch.zeros_like(clean)
    timestep = torch.tensor([0.94, 0.95], dtype=torch.float32)
    prediction = torch.tensor([[0.12], [0.15]])

    result = flow_matching_loss(
        prediction,
        state,
        clean,
        timestep,
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )

    torch.testing.assert_close(result.per_sample, torch.tensor([4.0, 9.0]))
    torch.testing.assert_close(result.loss, torch.tensor(6.5))
    torch.testing.assert_close(result.high_noise_loss_sum, torch.tensor(4.0))
    torch.testing.assert_close(result.low_noise_loss_sum, torch.tensor(9.0))
    assert result.high_noise_sample_count.item() == 1
    assert result.low_noise_sample_count.item() == 1


@pytest.mark.parametrize(
    "operation",
    [
        lambda: sample_jlt_timesteps(
            1,
            p_mean=-0.7,
            p_std=0.8,
            device=torch.device("cpu"),
            generator=torch.Generator(),
        ),
        lambda: sample_jlt_timesteps(
            1,
            p_mean=-0.8,
            p_std=1,
            device=torch.device("cpu"),
            generator=torch.Generator(),
        ),
        lambda: sample_noise(
            torch.zeros(1, 1), noise_scale=True, generator=torch.Generator()
        ),
        lambda: x_prediction_to_velocity(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.float32),
            t_eps=0.1,
        ),
        lambda: guided_velocity(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.float32),
            t_eps=0.05,
            guidance_scale=3.0,
        ),
        lambda: flow_matching_loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.float32),
            t_eps=0.05,
            noise_observation_boundary=0.5,
        ),
    ],
)
def test_objective_helpers_reject_noncanonical_semantics(
    operation: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="locked TOML float"):
        operation()


def _integer_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.int64)


def _rank_one_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1)


def _wrong_shape_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.expand(2, 2)


@pytest.mark.parametrize(
    "mutation",
    [
        _integer_tensor,
        _rank_one_tensor,
        _wrong_shape_tensor,
    ],
)
def test_flow_loss_rejects_invalid_tensor_contracts(
    mutation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    clean = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        flow_matching_loss(
            mutation(clean),
            clean,
            clean,
            torch.zeros(1, dtype=torch.float32),
            t_eps=0.05,
            noise_observation_boundary=0.95,
        )


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
