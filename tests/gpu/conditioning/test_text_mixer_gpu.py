from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.text_mixer import TextConditioner

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_text_mixer_production_shape_bf16_forward_backward_on_one_gpu() -> None:
    device = torch.device("cuda", 0)
    with torch.device(device):
        module = TextConditioner.for_production(
            attention_heads=8,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
        )

    states = torch.randn(
        2,
        32,
        7,
        2048,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    indices = torch.arange(32, device=device).expand(2, -1).clone()
    mask = torch.ones(2, 32, device=device, dtype=torch.bool)
    mask[1, 24:] = False
    indices[1, 24:] = torch.iinfo(torch.long).max

    output = module(states, indices, mask)
    output.tokens.float().square().mean().backward()

    assert output.tokens.shape == (2, 32, 2560)
    assert output.tokens.dtype == torch.bfloat16
    assert torch.isfinite(output.tokens).all()
    assert torch.count_nonzero(output.tokens[1, 24:]) == 0
    assert states.grad is None
    reached_gradients = tuple(
        parameter.grad for parameter in module.parameters() if parameter.grad is not None
    )
    assert reached_gradients
    assert all(torch.isfinite(gradient).all() for gradient in reached_gradients)
