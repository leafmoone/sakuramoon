"""Independent token -> spatial layout regression tests for IRepaAlignment.

Spatial contract of the iREPA v1 projector::

    input   tokens : [B, T, D], T = H * W, row-major (token t = y * W + x)
    layout  spatial[b, c, y, x] == tokens[b, y * W + x, c]   (channel-first)
    output  [B, T, 768], token t = y * W + x (row-major)

The reference in these tests is an INDEPENDENT index oracle: the expected
spatial mapping is constructed directly from the index definition (nested
loops for small grids, index tensors for the production width), never by
reshaping the input with the production expression.  Selector convolutions
(a single active kernel tap, zero elsewhere, zero bias) make the layout
assertions exact: with 0/1 BF16 pulse inputs no floating-point tolerance is
needed, and a wrong token/channel assignment cannot hide inside rounding.
FP32 is used only for the one arbitrary-weight reference derivation.

Every test drives the real production module (IRepaAlignment with the
locked out=768 / kernel=3 / stride=1 / padding=1 contract).
"""

from __future__ import annotations

import torch

from sakuramoon.model.irepa import (
    IREPA_TEACHER_FEATURE_WIDTH,
    IRepaAlignment,
)

OUT_C = IREPA_TEACHER_FEATURE_WIDTH  # 768, locked v1 contract


def _index_oracle(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Channel-first spatial tensor built directly from the index definition.

    Test oracle only; the production path never calls it::

        for b in range(B):
            for y in range(H):
                for x in range(W):
                    for c in range(D):
                        expected[b, c, y, x] = tokens[b, y * width + x, c]
    """

    batch, _, channels = tokens.shape
    expected = torch.empty(batch, channels, height, width, dtype=tokens.dtype)
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    expected[b, c, y, x] = tokens[b, y * width + x, c]
    return expected


def _pulse_tokens(batch: int, height: int, width: int, channels: int) -> torch.Tensor:
    """Deterministic 0/1 pattern, exactly representable in BF16.

    Values come from modular integer arithmetic mapped to 0.0/1.0: no large
    BF16 arange, so no rounding collision can mask a layout error.
    """

    tokens = torch.zeros(batch, height * width, channels, dtype=torch.bfloat16)
    for b in range(batch):
        for t in range(height * width):
            for c in range(channels):
                if (b * 13 + t * 31 + c * 7) % 5 < 2:
                    tokens[b, t, c] = 1.0
    return tokens


def _select_single_tap(
    module: IRepaAlignment,
    *,
    out_channel: int,
    in_channel: int,
    dy: int,
    dx: int,
) -> None:
    """Zero weight/bias, enable exactly one kernel tap (0/1, exact in BF16)."""

    weight = module.projector.weight
    bias = module.projector.bias
    assert weight is not None and bias is not None
    with torch.no_grad():
        weight.zero_()
        bias.zero_()
        weight[out_channel, in_channel, dy, dx] = 1.0


def _center_selector_reference(
    tokens: torch.Tensor,
    height: int,
    width: int,
    in_channel: int,
    out_channel: int,
) -> torch.Tensor:
    """[B, T, 768] reference for a center tap, by index definition."""

    batch, _, _ = tokens.shape
    expected = torch.zeros(batch, height * width, OUT_C, dtype=tokens.dtype)
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                expected[b, y * width + x, out_channel] = tokens[
                    b, y * width + x, in_channel
                ]
    return expected


def _neighbor_selector_reference(
    tokens: torch.Tensor,
    height: int,
    width: int,
    in_channel: int,
    out_channel: int,
    dy: int,
    dx: int,
) -> torch.Tensor:
    """[B, T, 768] reference for a tap at (dy, dx), zero-padding outside."""

    batch, _, _ = tokens.shape
    expected = torch.zeros(batch, height * width, OUT_C, dtype=tokens.dtype)
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                sy, sx = y + dy - 1, x + dx - 1
                if 0 <= sy < height and 0 <= sx < width:
                    expected[b, y * width + x, out_channel] = tokens[
                        b, sy * width + sx, in_channel
                    ]
    return expected


def test_center_tap_reads_exact_token_channel_square() -> None:
    # A: square grid, token/channel encoding must not be swapped.
    height, width, channels = 4, 4, 3
    tokens = _pulse_tokens(1, height, width, channels)
    module = IRepaAlignment(channels)

    for in_channel, out_channel in ((2, 5), (0, OUT_C - 1)):
        _select_single_tap(
            module, out_channel=out_channel, in_channel=in_channel, dy=1, dx=1
        )
        output = module(tokens, (height, width))
        expected = _center_selector_reference(
            tokens, height, width, in_channel, out_channel
        )
        assert output.shape == (1, height * width, OUT_C)
        assert torch.equal(output, expected), (
            "center tap (1,1) did not read tokens[b, y*W+x, c] at (y, x); "
            "spatial position and channel are mixed"
        )


def test_center_tap_reads_exact_token_channel_non_square() -> None:
    # A: non-square grid; a transposed (x*H+y) interpretation fails here.
    height, width, channels = 3, 7, 5
    tokens = _pulse_tokens(2, height, width, channels)
    module = IRepaAlignment(channels)

    _select_single_tap(module, out_channel=3, in_channel=4, dy=1, dx=1)
    output = module(tokens, (height, width))
    expected = _center_selector_reference(tokens, height, width, 4, 3)
    assert torch.equal(output, expected), (
        "non-square grid: output does not satisfy "
        "spatial[b,c,y,x] == tokens[b, y*W+x, c]"
    )


def test_center_tap_multi_batch_isolation() -> None:
    # A: multi-batch; every batch must map independently by the same index.
    height, width, channels = 5, 4, 3
    tokens = _pulse_tokens(3, height, width, channels)
    module = IRepaAlignment(channels)

    _select_single_tap(module, out_channel=11, in_channel=1, dy=1, dx=1)
    output = module(tokens, (height, width))
    expected = _center_selector_reference(tokens, height, width, 1, 11)
    assert torch.equal(output, expected), (
        "multi-batch: at least one batch violates the index contract"
    )
    # the oracle spatial tensor itself (independent of the conv) is checked
    # against the definition for every batch
    spatial = _index_oracle(tokens, height, width)
    assert spatial.shape == (3, channels, height, width)
    for b in range(3):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    assert spatial[b, c, y, x].item() == tokens[
                        b, y * width + x, c
                    ].item()


def test_center_tap_non_contiguous_input() -> None:
    # B: a legitimate strided [B, T, D] view must satisfy the same relation.
    height, width, channels = 4, 5, 3
    base = torch.zeros(2, 2 * height * width, channels, dtype=torch.bfloat16)
    for b in range(2):
        for t in range(2 * height * width):
            for c in range(channels):
                if (b * 11 + t * 17 + c * 3) % 4 < 2:
                    base[b, t, c] = 1.0
    tokens = base[:, ::2, :]
    assert not tokens.is_contiguous()
    module = IRepaAlignment(channels)

    _select_single_tap(module, out_channel=10, in_channel=2, dy=1, dx=1)
    output = module(tokens, (height, width))
    expected = _center_selector_reference(tokens, height, width, 2, 10)
    assert torch.equal(output, expected), (
        "non-contiguous [B, T, D] input: index contract not preserved"
    )


def test_neighbor_tap_reads_spatial_neighbor_with_zero_padding() -> None:
    # E: a non-center tap must read the true spatial neighbor (token),
    # not the adjacent feature channel, and zero-pad at the boundary.
    height, width, channels = 4, 5, 3
    tokens = _pulse_tokens(1, height, width, channels)
    module = IRepaAlignment(channels)

    for dy, dx in ((0, 0), (2, 2)):
        _select_single_tap(
            module, out_channel=21, in_channel=2, dy=dy, dx=dx
        )
        output = module(tokens, (height, width))
        expected = _neighbor_selector_reference(
            tokens, height, width, 2, 21, dy, dx
        )
        assert torch.equal(output, expected), (
            f"neighbor tap ({dy},{dx}): reads a wrong spatial token or a "
            "channel instead of a token; zero-padding boundary violated"
        )


def test_gradient_lands_on_exact_input_token_channel() -> None:
    # F: d L / d tokens for one output token/channel must land on the
    # corresponding input token/channel only; other batches unaffected.
    height, width, channels = 4, 4, 3
    # probe token (b0=0, y0=2, x0=3) -> t0 = 11; pulse value at
    # (0, 11, in_channel=0) is (11*31) % 5 = 1 < 2 -> exactly 1.0
    tokens = _pulse_tokens(2, height, width, channels).requires_grad_()
    module = IRepaAlignment(channels)
    in_channel, out_channel = 0, 7
    _select_single_tap(module, out_channel=out_channel, in_channel=in_channel, dy=1, dx=1)
    output = module(tokens, (height, width))

    b0, y0, x0 = 0, 2, 3
    t0 = y0 * width + x0
    assert tokens[b0, t0, in_channel].item() == 1.0  # exact 0/1 pulse premise

    output[b0, t0, out_channel].float().backward()

    grad = tokens.grad
    assert grad is not None and grad.dtype is torch.bfloat16
    assert torch.equal(
        grad[b0, t0],
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.bfloat16),
    ), "gradient did not land on the exact input token/channel"
    rest = grad.clone()
    rest[b0, t0] = 0.0
    assert torch.equal(rest, torch.zeros_like(rest)), (
        "gradient leaked to other tokens or to the other batch"
    )
    weight = module.projector.weight
    bias = module.projector.bias
    assert weight is not None and bias is not None
    wgrad = weight.grad
    bgrad = bias.grad
    assert wgrad is not None and bgrad is not None
    # Weight gradient carries the full index relation: for every channel ci
    # and kernel offset (dy, dx), dL/dw[7, ci, dy, dx] = the pulse value of
    # the input token that output (y0, x0) reads at that offset,
    # tokens[b0, (y0+dy-1)*W + (x0+dx-1), ci] (0/1, exact); zero-padding
    # offsets outside the grid stay exactly zero.
    expected_wgrad = torch.zeros_like(wgrad)
    for ci in range(channels):
        for dy in range(3):
            for dx in range(3):
                sy, sx = y0 + dy - 1, x0 + dx - 1
                if 0 <= sy < height and 0 <= sx < width:
                    expected_wgrad[
                        out_channel, ci, dy, dx
                    ] = tokens[b0, sy * width + sx, ci]
    assert torch.equal(wgrad, expected_wgrad), (
        "weight gradient must mirror exactly the spatial tokens read by "
        "the probed output position, zero elsewhere"
    )
    expected_bgrad = torch.zeros_like(bgrad)
    expected_bgrad[out_channel] = 1.0
    assert torch.equal(bgrad, expected_bgrad), (
        "bias gradient must be exactly the selected output channel"
    )


def test_production_width_2560_index_contract() -> None:
    # D=2560 (production), square 16x16: exact 0/1 index contract.
    height, width, channels = 16, 16, 2560
    b = torch.arange(1).view(1, 1, 1)
    t = torch.arange(height * width).view(1, height * width, 1)
    c = torch.arange(channels).view(1, 1, channels)
    tokens = (((b * 13 + t * 31 + c * 7) % 5) < 2).to(torch.bfloat16)
    module = IRepaAlignment(channels)
    in_channel, out_channel = 100, OUT_C - 1
    _select_single_tap(
        module, out_channel=out_channel, in_channel=in_channel, dy=1, dx=1
    )
    output = module(tokens, (height, width))
    expected = _center_selector_reference(tokens, height, width, in_channel, out_channel)
    assert torch.equal(output, expected), (
        "production width D=2560: index contract violated"
    )


def test_random_weight_matches_independent_reference() -> None:
    # Arbitrary-weight conv: reference = conv applied to the spatial tensor
    # built by the independent index oracle, compared in FP32 with the
    # BF16 rounding tolerance (layout errors are O(1), far above it).
    height, width, channels = 6, 7, 4
    tokens = _pulse_tokens(2, height, width, channels)
    module = IRepaAlignment(channels)
    weight = module.projector.weight
    bias = module.projector.bias
    assert weight is not None and bias is not None
    # Deterministic bounded arbitrary weights (modular pattern, no global
    # seed): the layout error is O(1), far above the BF16 tolerance.
    with torch.no_grad():
        w_idx = torch.arange(
            weight.numel(), dtype=torch.float32
        ).reshape(weight.shape)
        weight.copy_(((w_idx % 7) - 3) * 0.05)
        b_idx = torch.arange(bias.numel(), dtype=torch.float32)
        bias.copy_(((b_idx % 5) - 2) * 0.02)

    output = module(tokens, (height, width)).float()
    spatial_ref = _index_oracle(tokens, height, width).float()
    expected = torch.nn.functional.conv2d(
        spatial_ref,
        weight.float(),
        bias.float(),
        stride=module.projector.stride,
        padding=module.projector.padding,
        dilation=module.projector.dilation,
        groups=module.projector.groups,
    )
    expected = expected.flatten(2).transpose(1, 2)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=1e-2)
    assert output.shape == (2, height * width, OUT_C)
