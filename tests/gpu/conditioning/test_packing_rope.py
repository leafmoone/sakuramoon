from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.packing import pack_sequences
from sakuramoon.conditioning.rope import QKRoPE2D, packed_coordinates

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_cuda_packing_boundaries_and_rope_preserve_locked_types() -> None:
    device = torch.device("cuda", 0)
    text = torch.randn(2, 3, 2560, dtype=torch.bfloat16, device=device)
    text_mask = torch.tensor(
        [[True, False, True], [True, False, False]],
        dtype=torch.bool,
        device=device,
    )
    style = torch.randn(2, 4, 2560, dtype=torch.bfloat16, device=device)
    images = (
        torch.randn(4, 2560, dtype=torch.bfloat16, device=device),
        torch.randn(2, 2560, dtype=torch.bfloat16, device=device),
    )

    packed = pack_sequences(text, text_mask, style, images, ((2, 2), (1, 2)))
    coordinates = packed_coordinates(packed)
    rope = QKRoPE2D(
        head_dim=128,
        nope_dim=32,
        y_dim=48,
        x_dim=48,
        position_scale=16.0,
        theta=1000.0,
        norm_eps=1e-6,
    ).to(device)
    query = torch.randn(
        packed.tokens.shape[0],
        20,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        packed.tokens.shape[0],
        5,
        128,
        dtype=torch.bfloat16,
        device=device,
    )

    query_out, key_out = rope(query, key, coordinates)

    assert packed.boundaries.total_tokens == 17
    assert packed.boundaries.max_seqlen == 10
    assert packed.boundaries.batch_size == 2
    assert torch.equal(
        packed.cu_seqlens,
        torch.tensor([0, 10, 17], dtype=torch.int32, device=device),
    )
    assert packed.tokens.dtype == torch.bfloat16
    assert packed.tokens.device == device
    assert coordinates.dtype == torch.float32
    assert coordinates.device == device
    assert query_out.shape == query.shape and query_out.dtype == torch.bfloat16
    assert key_out.shape == key.shape and key_out.dtype == torch.bfloat16
    assert torch.isfinite(query_out).all() and torch.isfinite(key_out).all()
