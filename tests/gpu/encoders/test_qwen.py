from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.encoders.qwen import load_local_qwen

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_real_local_qwen_uses_boolean_mask_on_one_gpu() -> None:
    repository_root = Path(__file__).parents[3]
    device = torch.device("cuda", 0)
    runtime = load_local_qwen(repository_root, device)
    input_ids = torch.arange(98, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    final_norm = getattr(runtime.encoder.model, "norm", None)
    assert isinstance(final_norm, nn.Module)
    norm_boundary: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_norm_boundary(
        _module: nn.Module,
        inputs: tuple[object, ...],
        norm_output: object,
    ) -> None:
        assert len(inputs) == 1
        assert isinstance(inputs[0], torch.Tensor)
        assert isinstance(norm_output, torch.Tensor)
        norm_boundary.append((inputs[0].detach(), norm_output.detach()))

    handle = final_norm.register_forward_hook(capture_norm_boundary)
    try:
        output = runtime.encoder(input_ids, attention_mask)
    finally:
        handle.remove()

    assert output.hidden_states.shape == (1, 98, 7, 2048)
    assert output.hidden_states.device == device
    assert torch.isfinite(output.hidden_states).all()
    assert output.attention_mask is attention_mask
    assert len(norm_boundary) == 1
    raw_block_24, normalized_block_24 = norm_boundary[0]
    selected_block_24 = output.hidden_states[:, :, -1]
    assert torch.equal(selected_block_24, raw_block_24)
    max_abs_diff = (selected_block_24.float() - normalized_block_24.float()).abs().max()
    assert max_abs_diff.item() == pytest.approx(25.15625, abs=0.01)
    assert not torch.equal(selected_block_24, normalized_block_24)

    invalid_mask = attention_mask.to(dtype=torch.long)
    invalid_mask[0, -1] = 2
    with pytest.raises(TypeError, match="torch.bool"):
        runtime.encoder(input_ids, invalid_mask)
