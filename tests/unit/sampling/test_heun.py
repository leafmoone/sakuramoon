from __future__ import annotations

import math

import pytest
import torch

from sakuramoon.sampling.heun import euler, heun_final_euler
from sakuramoon.sampling.profiles import (
    SAMPLING_PROFILES,
    SamplingProfile,
    SamplingProfileName,
    resolve_sampling_profile,
)
from sakuramoon.sampling.sampler import (
    GenerationMetadata,
    build_generation_metadata,
    sample_profile,
)

PROFILE_NAMES: tuple[SamplingProfileName, ...] = (
    "preview",
    "balanced",
    "reference",
)


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


def test_euler_is_fp32_exact_for_constant_velocity_and_uses_one_nfe_per_step() -> None:
    calls = 0

    def constant_velocity(
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        del timestep
        return torch.full_like(state, 3.0)

    result = euler(
        constant_velocity,
        torch.zeros(2, 3, dtype=torch.bfloat16),
        steps=28,
    )

    assert calls == result.nfe == 28
    assert result.state.dtype == torch.float32
    torch.testing.assert_close(result.state, torch.full((2, 3), 3.0))


def test_profile_registry_is_exact_and_nfe_is_derived() -> None:
    assert {
        name: (profile.solver, profile.steps, profile.nfe, profile.time_schedule)
        for name, profile in SAMPLING_PROFILES.items()
    } == {
        "preview": ("euler", 28, 28, "linear"),
        "balanced": ("heun_final_euler", 25, 49, "linear"),
        "reference": ("heun_final_euler", 50, 99, "linear"),
    }

    with pytest.raises(ValueError, match="unknown sampling profile"):
        resolve_sampling_profile("custom")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="canonical registry"):
        SamplingProfile("preview", "euler", 29, "linear")
    with pytest.raises(ValueError, match="must be a string"):
        resolve_sampling_profile(())  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_profiles_use_declared_nfe_without_evaluating_clean_endpoint(
    profile_name: SamplingProfileName,
) -> None:
    timesteps: list[float] = []

    def zero_velocity(
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        timesteps.append(float(timestep[0]))
        return torch.zeros_like(state)

    result = sample_profile(
        zero_velocity,
        torch.ones(1, 2, dtype=torch.bfloat16),
        profile=profile_name,
    )

    assert result.nfe == result.profile.nfe == len(timesteps)
    assert result.state.dtype == torch.float32
    assert min(timesteps) == 0.0
    assert max(timesteps) < 1.0


def test_generation_metadata_records_sampling_and_explicit_legacy_provenance() -> None:
    sampled = sample_profile(
        lambda state, timestep: torch.zeros_like(state),
        torch.zeros(1, 1),
        profile="balanced",
    )
    metadata = build_generation_metadata(
        sampled,
        checkpoint_id="legacy-model-only",
        checkpoint_kind="model-only",
        objective_provenance="pre_fix",
        cfg_scale=2.9,
    ).as_mapping()

    assert metadata == {
        "cfg_scale": 2.9,
        "checkpoint_id": "legacy-model-only",
        "checkpoint_kind": "model-only",
        "nfe": 49,
        "noise_scale": 1.0,
        "objective_provenance": "pre_fix",
        "prediction_type": "x",
        "profile": "balanced",
        "schema_version": 1,
        "solver": "heun_final_euler",
        "state_dtype": "float32",
        "steps": 25,
        "t_eps": 0.05,
        "time_schedule": "linear",
    }

    with pytest.raises(ValueError, match="only valid as model-only"):
        GenerationMetadata(
            checkpoint_id="raw-checkpoint",
            checkpoint_kind="raw",
            objective_provenance="pre_fix",
            profile="reference",
            cfg_scale=2.9,
        )
