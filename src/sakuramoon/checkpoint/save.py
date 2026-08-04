"""Atomic deterministic single-GPU checkpoint publication."""

from __future__ import annotations

import json
import os
import shutil
import tomllib
import uuid
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from sakuramoon.checkpoint.artifact import (
    active_slot_ids_from_module,
    export_trainable_composite,
    validate_optimizer_coverage,
)
from sakuramoon.checkpoint.rng import capture_rank_rng
from sakuramoon.checkpoint.schema import (
    MAX_MODEL_SHARD_BYTES,
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    CheckpointSaveResult,
    FileRecord,
    RawCheckpointState,
    identity_to_dict,
    manifest_to_dict,
    raw_state_to_dict,
)
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit

_SAFETENSORS_HEADER_RESERVE_BYTES = 1024 * 1024


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, _json_bytes(value))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _target_name(kind: CheckpointKind, identity: CheckpointIdentity) -> str:
    prefixes = {
        CheckpointKind.RAW: "ckpt",
        CheckpointKind.MODEL_ONLY: "model",
        CheckpointKind.PMA: "pma",
        CheckpointKind.RELEASE: "release",
    }
    prefix = prefixes[kind]
    return f"{prefix}_{identity.update}_{identity.checkpoint_id}"


def _write_model(
    temporary: Path,
    module: nn.Module,
    identity: CheckpointIdentity,
    kind: CheckpointKind,
    max_shard_bytes: int,
) -> None:
    if type(max_shard_bytes) is not int or not 0 < max_shard_bytes <= MAX_MODEL_SHARD_BYTES:
        raise ValueError("max_shard_bytes must be in (0, 2 GiB]")
    architecture = export_trainable_composite(module)
    state = module.state_dict()
    names = tuple(sorted(state))
    if not names:
        raise ValueError("checkpoint model has no state tensors")
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    total_bytes = 0
    payload_budget = max(1, max_shard_bytes - _SAFETENSORS_HEADER_RESERVE_BYTES)
    for name in names:
        tensor = state[name]
        if tensor.device.type == "meta":
            raise ValueError(f"cannot checkpoint a meta tensor: {name}")
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes > max_shard_bytes:
            raise ValueError(f"model tensor exceeds shard limit: {name}")
        if current and current_bytes + tensor_bytes > payload_budget:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += tensor_bytes
        total_bytes += tensor_bytes
    shards.append(current)

    model_dir = temporary / "model"
    model_dir.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for index, shard_names in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        tensors = {
            name: state[name].detach().contiguous().cpu().clone() for name in shard_names
        }
        save_file(tensors, str(model_dir / filename))
        _fsync_file(model_dir / filename)
        if (model_dir / filename).stat().st_size > max_shard_bytes:
            raise ValueError(f"serialized model shard exceeds shard limit: {filename}")
        weight_map.update({name: filename for name in shard_names})
    _write_json(
        model_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
    )
    _write_json(
        model_dir / "config.json",
        {
            "identity": identity_to_dict(identity),
            "kind": kind.value,
            "architecture": architecture,
            "out_channels": 128,
            "prediction_type": "x",
            "schema_version": 1,
        },
    )
    model_records = _payload_records(model_dir)
    _write_json(
        model_dir / "manifest.json",
        {
            "files": [
                {"path": record.path, "size": record.size}
                for record in model_records
            ],
            "schema_version": 1,
        },
    )
    _fsync_directory(model_dir)


def _optimizer_schema(optimizer: IsolatedAdamW8bit) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for group in optimizer.optimizer.param_groups:
        group_name = group.get("group_name")
        param_names = group.get("param_names")
        if not isinstance(group_name, str) or not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in cast(list[object], param_names)
        ):
            raise CheckpointError("optimizer parameter groups lack canonical names")
        groups.append({"group_name": group_name, "param_names": param_names})
    return {
        "groups": groups,
        "schema_version": 1,
    }


def _write_raw_sidecars(
    temporary: Path,
    optimizer: IsolatedAdamW8bit,
    state: RawCheckpointState,
    resolved_config: bytes,
) -> None:
    _write_bytes(temporary / "resolved_config.toml", resolved_config)
    train_state = temporary / "train_state"
    train_state.mkdir()
    torch.save(optimizer.optimizer.state_dict(), train_state / "optimizer.pt")
    _fsync_file(train_state / "optimizer.pt")
    _write_json(train_state / "optimizer_schema.json", _optimizer_schema(optimizer))
    trainer, growth = raw_state_to_dict(state)
    _write_json(train_state / "trainer_state.json", trainer)
    _write_json(train_state / "growth_state.json", growth)
    rng_dir = train_state / "rng"
    rng_dir.mkdir()
    save_file(capture_rank_rng(), str(rng_dir / "rank-0.safetensors"))
    _fsync_file(rng_dir / "rank-0.safetensors")
    sr_state = optimizer.sr_rng.state_dict()
    sr_tensor = sr_state.get("state")
    if not isinstance(sr_tensor, torch.Tensor):
        raise CheckpointError("optimizer SR RNG state is invalid")
    save_file(
        {
            "device_index": torch.tensor(
                -1 if sr_state.get("device_index") is None else sr_state["device_index"],
                dtype=torch.int64,
            ),
            "state": sr_tensor,
        },
        str(rng_dir / "optimizer_sr.safetensors"),
    )
    _fsync_file(rng_dir / "optimizer_sr.safetensors")
    _fsync_directory(rng_dir)
    _fsync_directory(train_state)


def _payload_records(temporary: Path) -> tuple[FileRecord, ...]:
    records: list[FileRecord] = []
    for path in sorted(temporary.rglob("*")):
        relative = path.relative_to(temporary).as_posix()
        if path.is_file() and relative not in {"manifest.json", "COMPLETE"}:
            records.append(FileRecord(relative, path.stat().st_size))
    return tuple(records)


def _save(
    destination_root: Path,
    identity: CheckpointIdentity,
    kind: CheckpointKind,
    module: nn.Module,
    *,
    optimizer: IsolatedAdamW8bit | None,
    state: RawCheckpointState | None,
    resolved_config: bytes | None,
    max_shard_bytes: int,
) -> CheckpointSaveResult:
    if kind not in {CheckpointKind.RAW, CheckpointKind.MODEL_ONLY}:
        raise ValueError("checkpoint artifact kind is unsupported")
    if (kind is CheckpointKind.RAW) != (optimizer is not None and state is not None):
        raise ValueError("raw checkpoint requires optimizer and training state sidecars")
    if kind is CheckpointKind.RAW:
        if type(resolved_config) is not bytes:
            raise TypeError("raw checkpoint requires resolved config bytes")
        try:
            parsed_config = tomllib.loads(resolved_config.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            raise ValueError("resolved config must be valid UTF-8 TOML") from None
        if not parsed_config:
            raise ValueError("resolved config must be nonempty")
    elif resolved_config is not None:
        raise ValueError("non-raw artifacts cannot contain resolved config sidecars")
    export_trainable_composite(module)
    if state is not None and identity.update != state.trainer.successful_updates:
        raise ValueError("checkpoint update must equal successful optimizer updates")
    if optimizer is not None:
        validate_optimizer_coverage(
            module,
            tuple((spec.name, spec.parameter) for spec in optimizer.audit.specs),
        )
    if state is not None and state.growth.active_slot_ids != active_slot_ids_from_module(module):
        raise ValueError("checkpoint growth state differs from model active slots")
    destination_root.mkdir(parents=True, exist_ok=True)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ValueError("checkpoint destination must be a real directory")
    target = destination_root / _target_name(kind, identity)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"checkpoint target already exists: {target.name}")
    temporary = destination_root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        _write_model(temporary, module, identity, kind, max_shard_bytes)
        if optimizer is not None and state is not None and resolved_config is not None:
            _write_raw_sidecars(temporary, optimizer, state, resolved_config)
        records = _payload_records(temporary)
        manifest = CheckpointManifest(kind=kind, identity=identity, files=records)
        _write_json(temporary / "manifest.json", manifest_to_dict(manifest))
        _write_bytes(temporary / "COMPLETE", b"complete\n")
        _fsync_directory(temporary)
        os.replace(temporary, target)
        try:
            _fsync_directory(destination_root)
        except BaseException:
            rollback = destination_root / f".{target.name}.{uuid.uuid4().hex}.rollback.tmp"
            if target.exists():
                os.replace(target, rollback)
            if rollback.exists():
                shutil.rmtree(rollback)
            raise
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return CheckpointSaveResult(
        path=target,
        kind=kind,
        payload_bytes=sum(record.size for record in records),
        files=len(records),
    )


def save_raw_checkpoint(
    destination_root: Path,
    identity: CheckpointIdentity,
    module: nn.Module,
    optimizer: IsolatedAdamW8bit,
    state: RawCheckpointState,
    *,
    resolved_config: bytes,
    max_shard_bytes: int = MAX_MODEL_SHARD_BYTES,
) -> CheckpointSaveResult:
    return _save(
        destination_root,
        identity,
        CheckpointKind.RAW,
        module,
        optimizer=optimizer,
        state=state,
        resolved_config=resolved_config,
        max_shard_bytes=max_shard_bytes,
    )


def save_model_only(
    destination_root: Path,
    identity: CheckpointIdentity,
    module: nn.Module,
    *,
    max_shard_bytes: int = MAX_MODEL_SHARD_BYTES,
) -> CheckpointSaveResult:
    return _save(
        destination_root,
        identity,
        CheckpointKind.MODEL_ONLY,
        module,
        optimizer=None,
        state=None,
        resolved_config=None,
        max_shard_bytes=max_shard_bytes,
    )


__all__ = [
    "save_model_only",
    "save_raw_checkpoint",
]
