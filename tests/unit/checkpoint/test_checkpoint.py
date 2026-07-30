from __future__ import annotations

import importlib
import json
import random
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sakuramoon.checkpoint import (
    CheckpointIdentity,
    CheckpointKind,
    GrowthCheckpointState,
    discover_complete_checkpoints,
    load_model_directory,
    load_model_only,
    save_model_only,
)
from sakuramoon.checkpoint.rng import (
    capture_rank_rng,
    restore_rank_rng,
    validate_rank_rng,
)
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    RawCheckpointState,
    raw_state_from_dicts,
    raw_state_to_dict,
)
from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.data.state import ShardRunState
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.optim.groups import audit_trainable_parameters
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _tiny_composite() -> TrainableComposite:
    return TrainableComposite(
        dit=DenseDiT(  # pyright: ignore[reportArgumentType]
            depth=16,
            input_channels=128,
            hidden_size=8,
            intermediate_size=8,
            q_heads=1,
            kv_heads=1,
            head_dim=8,
            rope_nope_dim=0,
            rope_y_dim=4,
            rope_x_dim=4,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=16,
            out_channels=128,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        ),
        text=TextConditioner(
            input_size=8,
            adapter_size=8,
            output_size=8,
            groups=1,
            attention_heads=1,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
        style=StyleResampler(
            input_size=8,
            hidden_size=8,
            intermediate_size=8,
            output_size=8,
            query_count=4,
            attention_heads=1,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
    )


@cache
def _tiny_parameter_schema_sha256() -> str:
    return audit_trainable_parameters(
        _tiny_composite(),
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    ).schema_sha256


def _identity(checkpoint_id: str = "unit", update: int = 12) -> CheckpointIdentity:
    return CheckpointIdentity(
        checkpoint_id=checkpoint_id,
        update=update,
        config_sha256=_HASH_A,
        dependency_sha256=_HASH_B,
        parameter_schema_sha256=_tiny_parameter_schema_sha256(),
    )


def test_model_only_round_trip_uses_deterministic_fqn_shards(tmp_path: Path) -> None:
    source = _tiny_composite()
    expected = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    shard_limit = 6 * 1024**2
    result = save_model_only(
        tmp_path, _identity(), source, max_shard_bytes=shard_limit
    )
    index = json.loads((result.path / "model/model.safetensors.index.json").read_bytes())

    assert result.path.name == "model_12_unit"
    assert list(index["weight_map"]) == sorted(expected)
    assert len(set(index["weight_map"].values())) > 1
    assert all(
        path.stat().st_size <= shard_limit
        for path in (result.path / "model").glob("*.safetensors")
    )
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in expected.values()
    )
    assert discover_complete_checkpoints(tmp_path) == (result.path,)

    restored = _tiny_composite()
    with torch.no_grad():
        for tensor in restored.state_dict(keep_vars=True).values():
            tensor.zero_()
    load_model_only(result.path, restored, _identity())
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)


def test_checksum_failure_happens_before_model_changes(tmp_path: Path) -> None:
    source = _tiny_composite()
    result = save_model_only(tmp_path, _identity(), source)
    shard = next((result.path / "model").glob("*.safetensors"))
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(body)
    target = _tiny_composite()
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointError, match="checksum"):
        load_model_only(result.path, target, _identity())

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_manifest_file_set_is_strict(tmp_path: Path, mutation: str) -> None:
    result = save_model_only(tmp_path, _identity(mutation), _tiny_composite())
    if mutation == "missing":
        (result.path / "model/config.json").unlink()
    else:
        (result.path / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(CheckpointError, match="file set"):
        load_model_only(result.path, _tiny_composite(), _identity(mutation))


@pytest.mark.parametrize(
    "expected",
    [
        replace(_identity(), checkpoint_id="other"),
        replace(_identity(), update=13),
        replace(_identity(), config_sha256="d" * 64),
        replace(_identity(), dependency_sha256="e" * 64),
        replace(_identity(), parameter_schema_sha256="f" * 64),
    ],
)
def test_identity_failure_happens_before_model_changes(
    tmp_path: Path, expected: CheckpointIdentity
) -> None:
    result = save_model_only(tmp_path, _identity(), _tiny_composite())
    target = _tiny_composite()
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointError, match="identity"):
        load_model_only(result.path, target, expected)

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_kind_failure_happens_before_model_changes(tmp_path: Path) -> None:
    result = save_model_only(tmp_path, _identity(), _tiny_composite())
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["kind"] = CheckpointKind.PMA.value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = _tiny_composite()
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(CheckpointError, match="kind"):
        load_model_only(result.path, target, _identity())

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_incomplete_directories_are_not_discovered(tmp_path: Path) -> None:
    incomplete = tmp_path / "ckpt_3_incomplete"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text("{}", encoding="utf-8")

    assert discover_complete_checkpoints(tmp_path) == ()
    with pytest.raises(CheckpointError, match="incomplete"):
        load_model_only(incomplete, _tiny_composite(), _identity())


def test_unpublished_complete_temporary_directory_is_not_discovered(
    tmp_path: Path,
) -> None:
    result = save_model_only(tmp_path, _identity(), _tiny_composite())
    temporary = tmp_path / f".{result.path.name}.deadbeef.tmp"
    result.path.rename(temporary)

    assert discover_complete_checkpoints(tmp_path) == ()


def test_failed_save_removes_only_its_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = save_model_only(tmp_path, _identity("complete"), _tiny_composite()).path
    save_module = importlib.import_module("sakuramoon.checkpoint.save")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(save_module, "_write_model", fail_write)
    with pytest.raises(OSError, match="injected"):
        save_model_only(tmp_path, _identity("failed"), _tiny_composite())

    assert discover_complete_checkpoints(tmp_path) == (complete,)
    assert [path.name for path in tmp_path.iterdir()] == [complete.name]


def test_rank_rng_round_trip_restores_python_numpy_and_torch() -> None:
    random.seed(1201)
    np.random.seed(1202)
    torch.manual_seed(1203)  # pyright: ignore[reportUnknownMemberType]
    state = capture_rank_rng()
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)  # pyright: ignore[reportUnknownMemberType]
    restore_rank_rng(state)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


def test_rank_rng_rejects_state_that_cannot_be_restored() -> None:
    state = capture_rank_rng()
    state["python_internal"] = state["python_internal"][:-1]

    with pytest.raises(CheckpointError, match="not restorable"):
        validate_rank_rng(state)


def test_raw_training_state_is_strict_and_round_trips() -> None:
    state = RawCheckpointState(
        trainer=SingleGpuUpdateState(3, 2, 11),
        data=ShardRunState(("a.tar",), "b.tar", 1, 7),
        growth=GrowthCheckpointState(
            BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None
        ),
    )
    documents = raw_state_to_dict(state)

    assert raw_state_from_dicts(*documents) == state
    invalid = dict(documents[0])
    invalid["unexpected"] = True
    with pytest.raises(CheckpointError, match="unknown or missing"):
        raw_state_from_dicts(invalid, documents[1], documents[2])

    invalid = dict(documents[0])
    invalid["schema_version"] = True
    with pytest.raises(CheckpointError, match="schema version"):
        raw_state_from_dicts(invalid, documents[1], documents[2])


def test_manifest_rejects_boolean_schema_version(tmp_path: Path) -> None:
    result = save_model_only(tmp_path, _identity(), _tiny_composite())
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["schema_version"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="schema version"):
        load_model_only(result.path, _tiny_composite(), _identity())


def test_invalid_identity_and_shard_limit_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint_id"):
        _identity("../escape")
    with pytest.raises(ValueError, match="shard"):
        save_model_only(tmp_path, _identity(), _tiny_composite(), max_shard_bytes=8)

    with pytest.raises(ValueError, match="shard"):
        save_model_only(tmp_path, _identity("header"), _tiny_composite(), max_shard_bytes=64)

    with pytest.raises(ValueError, match="parameter schema"):
        save_model_only(
            tmp_path,
            replace(_identity("schema"), parameter_schema_sha256="c" * 64),
            _tiny_composite(),
        )


def test_model_directory_is_self_describing_and_rejects_arbitrary_module(
    tmp_path: Path,
) -> None:
    source = _tiny_composite()
    result = save_model_only(tmp_path, _identity(), source)

    restored, identity, kind = load_model_directory(result.path / "model", device="cpu")

    assert identity == _identity()
    assert kind is CheckpointKind.MODEL_ONLY
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(TypeError, match="TrainableComposite"):
        save_model_only(tmp_path, _identity("bad"), nn.Linear(8, 8))


def test_parent_fsync_failure_rolls_back_published_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_module = importlib.import_module("sakuramoon.checkpoint.save")
    real_fsync = save_module._fsync_directory

    def fail_parent(path: Path) -> None:
        if path == tmp_path:
            raise OSError("injected parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(save_module, "_fsync_directory", fail_parent)
    with pytest.raises(OSError, match="parent fsync"):
        save_model_only(tmp_path, _identity(), _tiny_composite())

    assert list(tmp_path.iterdir()) == []
    assert discover_complete_checkpoints(tmp_path) == ()


def test_growth_state_rejects_noncanonical_range_slots() -> None:
    with pytest.raises(ValueError, match="canonical"):
        GrowthCheckpointState(tuple(range(16)), 1.0, "S0", 1, 256, None, None)


def test_growth_state_rejects_alpha_that_differs_from_persisted_progress() -> None:
    with pytest.raises(ValueError, match="differs from persisted ramp progress"):
        RawCheckpointState(
            trainer=SingleGpuUpdateState(501, 501, 1),
            data=ShardRunState.empty(),
            growth=GrowthCheckpointState(
                BASE_SLOT_IDS, 0.25, "G1", 4, 256, 1, 1000
            ),
        )
