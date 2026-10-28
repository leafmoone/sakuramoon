"""Final-norm architecture change: strict named state and finite isolated updates."""
# pyright: reportPrivateUsage=false
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import sakuramoon.checkpoint.load as loader
from sakuramoon.checkpoint.artifact import (
    architectures_share_parameter_contract,
    build_trainable_composite,
    export_trainable_composite,
)
from sakuramoon.checkpoint.final_norm import (
    FINAL_NORM_FQN,
    remove_final_norm_named_state,
)
from sakuramoon.checkpoint.save import _write_model
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
)
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.output_head import FinalOutputHead
from sakuramoon.train.step import TrainableComposite


def _head(norm: str = "none") -> FinalOutputHead:
    return FinalOutputHead(
        hidden_size=8, out_channels=4, norm_eps=1e-6,
        projection_dtype=torch.float32, weight_zero_init=True, bias_zero_init=True,
        final_output_norm=norm,
    )


def _composite(norm: str) -> TrainableComposite:
    return TrainableComposite(
        dit=DenseDiT(
            depth=1, active_slot_ids=(0,), input_channels=8, hidden_size=8,
            intermediate_size=16, q_heads=2, kv_heads=1, head_dim=4,
            rope_nope_dim=0, rope_y_dim=2, rope_x_dim=2,
            rope_position_scale=1.0, rope_theta=10.0, norm_eps=1e-6,
            timestep_dim=64, size_dim=64, aspect_dim=64, condition_hidden_size=8,
            stable_slot_count=1, modulation_chunks=6, final_modulation_size=16,
            out_channels=8, condition_token_count=2, modality_init_std=0.02,
            linear_dtype=torch.float32, sensitive_dtype=torch.float32,
            projection_bias=False, attention_dropout=0.0, mlp_dropout=0.0,
            output_weight_zero_init=True, output_bias_zero_init=True,
            final_output_norm=norm,
        ),
        text=TextConditioner(
            input_size=8, adapter_size=8, output_size=8, groups=2,
            attention_heads=2, norm_eps=1e-6, mix_gate_init=0.1,
            layer_scale_init=1.0, projection_bias=False,
            linear_dtype=torch.float32, sensitive_dtype=torch.float32,
        ),
        condition_tokens=ConditionTokenEncoder(
            input_size=8, hidden_size=8, intermediate_size=16, output_size=8,
            token_count=2, attention_heads=2, norm_eps=1e-6, init_std=0.02,
            projection_bias=False, linear_dtype=torch.float32,
            sensitive_dtype=torch.float32,
        ),
    )


def test_dense_packed_head_is_direct_modulation_and_fp32_projection() -> None:
    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
    old, new = _head("rmsnorm"), _head()
    with torch.no_grad():
        old.projection.weight.normal_()
        old.projection.bias.normal_()
        old.norm.weight.uniform_(0.4, 1.8)
    new.projection.load_state_dict(old.projection.state_dict())
    assert set(old.state_dict()) - set(new.state_dict()) == {"norm.weight"}
    x = (torch.randn(2, 3, 8) * 3).requires_grad_()
    scale = torch.randn(2, 8, requires_grad=True)
    shift = torch.randn(2, 8, requires_grad=True)
    expected = new.projection(((1 + scale[:, None]) * x + shift[:, None]).float())
    dense = new(x, scale, shift, (1, 3))
    torch.testing.assert_close(dense, expected.transpose(1, 2).reshape(2, 4, 1, 3))
    packed = new.forward_packed(x.flatten(0, 1), torch.tensor([0, 0, 0, 1, 1, 1]), scale, shift, ((1, 3), (1, 3)))
    torch.testing.assert_close(torch.stack(packed), dense)
    assert not torch.allclose(old(x, scale, shift, (1, 3)), dense)
    dense.square().mean().backward()
    for tensor in (x, scale, shift, *new.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    optimizer = torch.optim.AdamW(new.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad()
        loss = new(x.detach(), scale.detach(), shift.detach(), (1, 3)).square().mean()
        loss.backward()
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        assert torch.isfinite(loss)
    fresh = _head()
    fresh.load_state_dict(new.state_dict(), strict=True)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
    fresh_optimizer.load_state_dict(optimizer.state_dict())
    torch.testing.assert_close(fresh(x, scale, shift, (1, 3)), new(x, scale, shift, (1, 3)))


def test_artifact_named_drop_and_fresh_roundtrip(tmp_path: Path) -> None:
    source, target = _composite("rmsnorm"), _composite("none")
    identity = CheckpointIdentity(update=1, checkpoint_id="synthetic")
    _write_model(tmp_path / "old", source, identity, CheckpointKind.MODEL_ONLY, 2**30)
    old_dir = tmp_path / "old" / "model"
    weight_map, current, dropped = loader._validate_model_tensors(old_dir, target, identity, CheckpointKind.MODEL_ONLY)
    assert dropped == {FINAL_NORM_FQN}
    loader._apply_model(old_dir, weight_map, current, dropped)
    source_state = source.state_dict()
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, source_state[name])
    _write_model(tmp_path / "new", target, identity, CheckpointKind.MODEL_ONLY, 2**30)
    restored, _, _ = loader.load_model_directory(tmp_path / "new" / "model", device="cpu")
    assert FINAL_NORM_FQN not in restored.state_dict()
    assert all(torch.equal(tensor, restored.state_dict()[name]) for name, tensor in target.state_dict().items())
    legacy = export_trainable_composite(source)
    cast(dict[str, Any], legacy["dit"]).pop("final_output_norm")
    rebuilt = build_trainable_composite(legacy, device="cpu")
    assert FINAL_NORM_FQN in rebuilt.state_dict()
    assert "final_output_norm" not in cast(dict[str, Any], legacy["dit"])


def test_architecture_drop_is_directional_and_other_fields_strict() -> None:
    source = export_trainable_composite(_composite("rmsnorm"))
    target = export_trainable_composite(_composite("none"))
    assert not architectures_share_parameter_contract(target, source)
    assert architectures_share_parameter_contract(target, source, allow_final_norm_drop=True)
    assert not architectures_share_parameter_contract(source, target, allow_final_norm_drop=True)
    broken = copy.deepcopy(source)
    cast(dict[str, Any], broken["dit"])["norm_eps"] = 0.1
    assert not architectures_share_parameter_contract(target, broken, allow_final_norm_drop=True)


def test_named_ema_shadow_preserves_objects_and_rejects_other_deltas() -> None:
    source = _composite("rmsnorm").state_dict()
    target = _composite("none").state_dict()
    migrated = remove_final_norm_named_state(source, target)
    assert all(migrated[name] is source[name] for name in target)
    with pytest.raises(CheckpointError, match="delta"):
        remove_final_norm_named_state({**source, "unexpected.weight": torch.ones(1)}, target)
    with pytest.raises(CheckpointError, match="shape or dtype"):
        remove_final_norm_named_state({**source, "dit.output_head.projection.bias": torch.ones(9)}, target)


@pytest.mark.parametrize("case", ["shape", "dtype", "extra", "missing"])
def test_deleted_norm_does_not_hide_corrupt_tensors(tmp_path: Path, case: str) -> None:
    import json

    from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]

    source, target = _composite("rmsnorm"), _composite("none")
    identity = CheckpointIdentity("synthetic", 1)
    _write_model(tmp_path, source, identity, CheckpointKind.MODEL_ONLY, 2**30)
    model_dir = tmp_path / "model"
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard = model_dir / index["weight_map"][FINAL_NORM_FQN]
    tensors = load_file(shard)
    if case == "shape":
        tensors[FINAL_NORM_FQN] = torch.ones(9)
    elif case == "dtype":
        tensors[FINAL_NORM_FQN] = tensors[FINAL_NORM_FQN].to(torch.bfloat16)
    elif case == "extra":
        tensors["unexpected.weight"] = torch.ones(1)
        index["weight_map"]["unexpected.weight"] = shard.name
    else:
        del tensors["dit.output_head.projection.bias"]
        del index["weight_map"]["dit.output_head.projection.bias"]
    save_file(tensors, shard)
    index_path.write_text(json.dumps(index))
    with pytest.raises(CheckpointError):
        loader._validate_model_tensors(model_dir, target, identity, CheckpointKind.MODEL_ONLY)
