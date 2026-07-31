from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.style_resampler import StyleResampler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_production_style_resampler_skips_null_samples_on_one_gpu() -> None:
    device = torch.device("cuda", 0)
    module = StyleResampler(
        input_size=2048,
        hidden_size=1024,
        intermediate_size=2048,
        output_size=2560,
        query_count=4,
        attention_heads=16,
        norm_eps=1e-6,
        init_std=0.02,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    ).to(device)
    states = torch.randn(
        3,
        6,
        7,
        2048,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    projected_batch_sizes: list[int] = []

    def record_projected_batch(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        projected_batch_sizes.append(inputs[0].shape[0])

    handle = module.input_projection.register_forward_pre_hook(record_projected_batch)
    try:
        output = module(
            states,
            torch.tensor(
                [[1, 9999], [9999, 9999], [2, 3]],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [[True, False], [True, True], [True, True]],
                dtype=torch.bool,
                device=device,
            ),
            torch.tensor([False, True, False], dtype=torch.bool, device=device),
        )
    finally:
        handle.remove()

    assert output.tokens.shape == (3, 4, 2560)
    assert output.tokens.dtype == torch.bfloat16
    assert torch.isfinite(output.tokens).all()
    assert projected_batch_sizes == [2]
    torch.testing.assert_close(
        output.tokens[1], module.null_tokens.to(dtype=torch.bfloat16)
    )

    output.tokens.float().square().mean().backward()
    assert states.grad is None
    assert module.input_projection.weight.grad is not None
    assert module.null_tokens.grad is not None
