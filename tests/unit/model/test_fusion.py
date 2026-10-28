"""Compiled boundaries must preserve input and shared-parameter gradients."""

# pyright: reportUnknownMemberType=false
from __future__ import annotations

import copy

import pytest
import torch

from sakuramoon.conditioning.norm import FP32RMSNorm
from sakuramoon.conditioning.rope import QKRoPE2D
from sakuramoon.model.fusion import (
    compiled,
    configure_fusions,
    norm_modulation_reference,
    residual_reference,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qk_compiled_gqa_gamma_backward(dtype: torch.dtype) -> None:
    torch.manual_seed(91)
    reference = QKRoPE2D(
        head_dim=16,
        nope_dim=8,
        y_dim=4,
        x_dim=4,
        position_scale=3.2,
        theta=10000,
        norm_eps=1e-6,
    )
    with torch.no_grad():
        reference.q_norm.weight.uniform_(0.4, 1.6)
        reference.k_norm.weight.uniform_(0.5, 1.5)
    optimized = copy.deepcopy(reference)
    assert configure_fusions(optimized) == 1
    assert tuple(reference.state_dict()) == tuple(optimized.state_dict())
    q = torch.randn(7, 4, 16, dtype=dtype, requires_grad=True)
    k = torch.randn(7, 2, 16, dtype=dtype, requires_grad=True)
    qc, kc = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    coords = torch.randn(7, 2)
    expected = reference(q, k, coords)
    actual = optimized(qc, kc, coords)
    probes = tuple(torch.randn_like(value) for value in expected)
    torch.stack(
        [
            (value.float() * probe).sum()
            for value, probe in zip(expected, probes, strict=True)
        ]
    ).sum().backward()
    torch.stack(
        [
            (value.float() * probe).sum()
            for value, probe in zip(actual, probes, strict=True)
        ]
    ).sum().backward()
    tolerance = 0.06 if dtype == torch.bfloat16 else 2e-5
    for left, right in zip(expected, actual, strict=True):
        assert right.is_contiguous()
        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
    for left, right in (
        (q.grad, qc.grad),
        (k.grad, kc.grad),
        (reference.q_norm.weight.grad, optimized.q_norm.weight.grad),
        (reference.k_norm.weight.grad, optimized.k_norm.weight.grad),
    ):
        torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_modulation_residual_compiled_sample_reductions(
    packed: bool, dtype: torch.dtype
) -> None:
    torch.manual_seed(32)
    x = torch.randn((9, 24) if packed else (3, 3, 24), dtype=dtype, requires_grad=True)
    norm = FP32RMSNorm(24, 1e-6)
    with torch.no_grad():
        norm.weight.uniform_(0.2, 1.8)
    scale, shift, gate = (torch.randn(3, 24, requires_grad=True) for _ in range(3))
    indices = torch.tensor([2, 0, 2, 1, 1, 2, 0, 2, 2]) if packed else None
    inputs = (x, norm.weight, scale, shift, gate)
    clones = tuple(value.detach().clone().requires_grad_() for value in inputs)
    xc, wc, sc, shc, gc = clones
    # Independent original math is the oracle, including original FP32RMSNorm.
    s, sh, g = (
        (
            value.index_select(0, indices)
            if indices is not None
            else value.unsqueeze(1)
        ).to(dtype)
        for value in (scale, shift, gate)
    )
    branch = (1.0 + s) * norm(x) + sh
    expected = x + 0.7 * g * branch
    branchc = compiled(norm_modulation_reference)(xc, wc, sc, shc, 1e-6, indices)
    actual = compiled(residual_reference)(xc, branchc, gc, 0.7, indices)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    tolerance = 0.06 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(expected, actual, atol=tolerance, rtol=tolerance)
    for left, right in zip(inputs, clones, strict=True):
        torch.testing.assert_close(
            left.grad, right.grad, atol=tolerance, rtol=tolerance
        )


def test_compiled_boundary_inside_outer_compile_and_checkpoint() -> None:
    from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]

    torch.manual_seed(5)
    x = torch.randn(2, 3, 16, requires_grad=True)
    weight = torch.randn(16, requires_grad=True)
    scale = torch.randn(2, 16, requires_grad=True)
    shift = torch.randn(2, 16, requires_grad=True)
    inputs = (x, weight, scale, shift)
    clones = tuple(value.detach().clone().requires_grad_() for value in inputs)

    def operation(
        x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, sh: torch.Tensor
    ) -> torch.Tensor:
        return compiled(norm_modulation_reference)(x, w, s, sh, 1e-6)

    outer = torch.compile(operation, backend="inductor", fullgraph=True)
    expected = norm_modulation_reference(*inputs, 1e-6)
    actual = checkpoint(outer, *clones, use_reentrant=False)  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(actual, torch.Tensor)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
    for left, right in zip(inputs, clones, strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=3e-5, rtol=3e-5)


def test_dense_block_fusion_wiring_and_all_parameter_gradients() -> None:
    from sakuramoon.conditioning.global_condition import BlockModulation
    from sakuramoon.model.attention import dense_attention_mask
    from sakuramoon.model.block import DiTBlock

    torch.manual_seed(31)
    reference = DiTBlock(
        hidden_size=32,
        intermediate_size=48,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
        rope_nope_dim=0,
        rope_y_dim=4,
        rope_x_dim=4,
        rope_position_scale=2.5,
        rope_theta=10000,
        norm_eps=1e-6,
        linear_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0,
        mlp_dropout=0,
    )
    optimized = copy.deepcopy(reference)
    assert configure_fusions(optimized) == 2
    x = torch.randn(2, 3, 32, requires_grad=True)
    xc = x.detach().clone().requires_grad_()
    values = tuple(torch.randn(2, 32, requires_grad=True) for _ in range(6))
    clones = tuple(value.detach().clone().requires_grad_() for value in values)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    coords = torch.randn(2, 3, 2)
    expected = reference(
        x,
        mask,
        dense_attention_mask(mask),
        coords,
        BlockModulation(*values),
        attention_growth=0.7,
        mlp_growth=0.4,
    )
    actual = optimized(
        xc,
        mask,
        dense_attention_mask(mask),
        coords,
        BlockModulation(*clones),
        attention_growth=0.7,
        mlp_growth=0.4,
    )
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
    for left, right in zip(
        (x, *values, *reference.parameters()),
        (xc, *clones, *optimized.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(left.grad, right.grad, atol=3e-5, rtol=3e-5)
