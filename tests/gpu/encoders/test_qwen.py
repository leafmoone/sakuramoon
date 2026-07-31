from __future__ import annotations

from pathlib import Path

import pytest
import torch

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

    output = runtime.encoder(input_ids, attention_mask)

    assert output.hidden_states.shape == (1, 98, 7, 2048)
    assert output.hidden_states.device == device
    assert torch.isfinite(output.hidden_states).all()
    assert output.attention_mask is attention_mask

    invalid_mask = attention_mask.to(dtype=torch.long)
    invalid_mask[0, -1] = 2
    with pytest.raises(TypeError, match="torch.bool"):
        runtime.encoder(input_ids, invalid_mask)
