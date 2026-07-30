from __future__ import annotations

import pytest
import torch

from sakuramoon.objective.flow import (
    flow_matching_loss,
    guided_velocity,
    interpolate_state,
    sample_noise,
)
from sakuramoon.sampling.heun import heun_final_euler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_bf16_flow_loss_backward_update_cfg_and_heun_on_cuda() -> None:
    torch.manual_seed(911)  # pyright: ignore[reportUnknownMemberType]
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(912)
    clean = torch.randn(2, 128, 4, 4, device=device, dtype=torch.bfloat16)
    noise = sample_noise(clean, noise_scale=1.0, generator=generator)
    timestep = torch.tensor([0.25, 0.99], device=device, dtype=torch.float32)
    state = interpolate_state(clean, noise, timestep)
    prediction = torch.nn.Parameter(torch.randn_like(clean, dtype=torch.bfloat16))
    optimizer = torch.optim.SGD((prediction,), lr=0.01)
    before = prediction.detach().clone()

    loss_output = flow_matching_loss(
        prediction,
        state,
        clean,
        noise,
        timestep,
        t_eps=0.05,
    )
    loss_output.loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    assert loss_output.loss.dtype == torch.float32
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert not torch.equal(before, prediction)

    conditional = state + 0.05 * (clean.float() - noise.float())
    unconditional = state.float()
    guided = guided_velocity(
        conditional.to(torch.bfloat16),
        unconditional.to(torch.bfloat16),
        state,
        timestep,
        t_eps=0.05,
        guidance_scale=2.9,
    )
    assert guided.dtype == torch.float32
    assert torch.isfinite(guided).all()

    def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        conditional_x = state + 0.05 * torch.ones_like(state)
        return guided_velocity(
            conditional_x.to(torch.bfloat16),
            state.to(torch.bfloat16),
            state,
            timestep,
            t_eps=0.05,
            guidance_scale=2.9,
        )

    sampled = heun_final_euler(
        velocity,
        torch.zeros(2, 8, device=device, dtype=torch.bfloat16),
        steps=50,
    )

    assert sampled.nfe == 99
    assert sampled.state.dtype == torch.float32
    assert sampled.state.requires_grad is False
    assert torch.isfinite(sampled.state).all()
