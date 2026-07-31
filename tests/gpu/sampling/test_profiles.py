from __future__ import annotations

import pytest
import torch

from sakuramoon.sampling.profiles import SamplingProfileName
from sakuramoon.sampling.sampler import build_generation_metadata, sample_profile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    ("profile", "expected_nfe"),
    [("preview", 28), ("balanced", 49), ("reference", 99)],
)
def test_fixed_profile_executes_on_cuda_without_clean_endpoint_evaluation(
    profile: SamplingProfileName,
    expected_nfe: int,
) -> None:
    timesteps: list[float] = []

    def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        assert state.is_cuda and timestep.is_cuda
        assert state.dtype == timestep.dtype == torch.float32
        timesteps.append(float(timestep[0]))
        return torch.ones_like(state)

    sampled = sample_profile(
        velocity,
        torch.zeros(1, 8, device="cuda", dtype=torch.bfloat16),
        profile=profile,
    )
    metadata = build_generation_metadata(
        sampled,
        checkpoint_id="gpu-smoke",
        checkpoint_kind="raw",
        objective_provenance="strict_jlt",
        cfg_scale=2.9,
    ).as_mapping()

    assert sampled.nfe == len(timesteps) == expected_nfe
    assert sampled.state.dtype == torch.float32
    assert sampled.state.requires_grad is False
    assert max(timesteps) < 1.0
    assert metadata["profile"] == profile
    assert metadata["nfe"] == expected_nfe
