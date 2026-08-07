"""Fail-closed checkpoint validation and same-topology restore."""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Protocol, Self, cast

import torch
from safetensors import safe_open
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from sakuramoon.checkpoint.artifact import (
    active_slot_ids_from_module,
    architectures_share_parameter_contract,
    build_trainable_composite,
    export_trainable_composite,
    validate_optimizer_coverage,
)
from sakuramoon.checkpoint.rng import restore_rank_rng, validate_rank_rng
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    RawCheckpointState,
    identity_from_dict,
    manifest_from_dict,
    raw_state_from_dicts,
)
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.groups import ParameterSpec

_TORCH_TO_SAFE_DTYPE = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
_FINAL_CHECKPOINT_NAME = re.compile(
    r"(?:ckpt|model|pma|release)_[0-9]+_[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
)
_RAW_SIDECARS = {
    "resolved_config.toml",
    "train_state/growth_state.json",
    "train_state/optimizer.pt",
    "train_state/optimizer_schema.json",
    "train_state/rng/optimizer_sr.safetensors",
    "train_state/rng/rank-0.safetensors",
    "train_state/trainer_state.json",
}


class _SafeSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> list[int]: ...


class _SafeHandle(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def keys(self) -> list[str]: ...

    def get_slice(self, name: str) -> _SafeSlice: ...


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError(f"{name} is unreadable or invalid JSON") from None


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(document: dict[str, Any], expected: set[str], name: str) -> None:
    if set(document) != expected:
        raise CheckpointError(f"{name} has unknown or missing fields")


def read_checkpoint_manifest(path: Path) -> CheckpointManifest:
    if path.is_symlink() or not path.is_dir():
        raise CheckpointError("checkpoint path must be a real directory")
    complete = path / "COMPLETE"
    try:
        if complete.is_symlink() or complete.read_bytes() != b"complete\n":
            raise CheckpointError("checkpoint is incomplete")
    except OSError:
        raise CheckpointError("checkpoint is incomplete") from None
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CheckpointError("checkpoint manifest must be a regular file")
    manifest = manifest_from_dict(_read_json(manifest_path, "manifest"))
    expected_paths = {record.path for record in manifest.files}
    allowed_files = expected_paths | {"manifest.json", "COMPLETE"}
    allowed_directories = {
        parent.as_posix()
        for relative in allowed_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    actual_payload_paths: set[str] = set()
    for item in path.rglob("*"):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise CheckpointError("checkpoint tree contains a symbolic link")
        if item.is_dir():
            if relative not in allowed_directories:
                raise CheckpointError("checkpoint payload file set does not match manifest")
            continue
        if not item.is_file() or relative not in allowed_files:
            raise CheckpointError("checkpoint payload file set does not match manifest")
        if relative not in {"manifest.json", "COMPLETE"}:
            actual_payload_paths.add(relative)
    if actual_payload_paths != expected_paths:
        raise CheckpointError("checkpoint payload file set does not match manifest")
    for record in manifest.files:
        payload = path / record.path
        if payload.is_symlink() or not payload.is_file():
            raise CheckpointError(f"checkpoint payload is missing: {record.path}")
        try:
            if payload.stat().st_size != record.size:
                raise CheckpointError(f"checkpoint payload size differs: {record.path}")
        except OSError:
            raise CheckpointError(f"checkpoint payload is unreadable: {record.path}") from None
    return manifest


def read_raw_checkpoint_state(
    checkpoint: Path,
) -> tuple[CheckpointManifest, RawCheckpointState]:
    manifest = read_checkpoint_manifest(checkpoint)
    if manifest.kind is not CheckpointKind.RAW:
        raise CheckpointError("raw checkpoint state accepts raw checkpoints only")
    _validate_raw_sidecars(checkpoint, manifest)
    train_state = checkpoint / "train_state"
    state = raw_state_from_dicts(
        _read_json(train_state / "trainer_state.json", "trainer state"),
        _read_json(train_state / "growth_state.json", "growth state"),
    )
    if manifest.identity.update != state.trainer.successful_updates:
        raise CheckpointError("checkpoint update differs from trainer successful updates")
    return manifest, state


def _validate_raw_sidecars(
    checkpoint: Path, manifest: CheckpointManifest
) -> None:
    sidecars = {
        record.path
        for record in manifest.files
        if not record.path.startswith("model/")
    }
    if "train_state/data_state.json" in sidecars:
        raise CheckpointError("legacy raw data sidecar is unsupported")
    if sidecars != _RAW_SIDECARS:
        raise CheckpointError("raw checkpoint sidecars are unknown or missing")
    model_records = _model_manifest_records(checkpoint / "model")
    outer_model = {
        record.path.removeprefix("model/"): record
        for record in manifest.files
        if record.path.startswith("model/")
    }
    expected_model_paths = {
        "manifest.json",
        *(record.path for record in model_records),
    }
    if set(outer_model) != expected_model_paths:
        raise CheckpointError("raw checkpoint model payload set is invalid")
    for record in model_records:
        outer = outer_model[record.path]
        if outer.size != record.size:
            raise CheckpointError("raw checkpoint model manifests differ")
    try:
        resolved_config = (checkpoint / "resolved_config.toml").read_bytes()
        parsed_config = tomllib.loads(resolved_config.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise CheckpointError("resolved config is unreadable") from None
    if not parsed_config:
        raise CheckpointError("resolved config is unreadable")


def _validate_identity(
    manifest: CheckpointManifest,
    expected: CheckpointIdentity,
    kind: CheckpointKind,
) -> None:
    if manifest.kind is not kind:
        raise CheckpointError(f"checkpoint kind must be {kind.value}")
    if manifest.identity != expected:
        raise CheckpointError("checkpoint identity does not match the requested run")


def _model_index(path: Path) -> tuple[dict[str, str], int]:
    document = _mapping(_read_json(path, "model index"), "model index")
    _exact_keys(document, {"metadata", "weight_map"}, "model index")
    metadata = _mapping(document["metadata"], "model index metadata")
    weight_map = _mapping(document["weight_map"], "model weight map")
    _exact_keys(metadata, {"total_size"}, "model index metadata")
    total_size = metadata["total_size"]
    if type(total_size) is not int or total_size < 0:
        raise CheckpointError("model total size is invalid")
    if not weight_map or not all(isinstance(filename, str) for filename in weight_map.values()):
        raise CheckpointError("model weight map is invalid")
    return cast(dict[str, str], weight_map), total_size


def _validate_model_config(
    path: Path,
    expected: CheckpointIdentity,
    kind: CheckpointKind,
) -> object:
    document = _mapping(_read_json(path, "model config"), "model config")
    _exact_keys(
        document,
        {
            "architecture",
            "schema_version",
            "kind",
            "identity",
            "prediction_type",
            "out_channels",
        },
        "model config",
    )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["kind"] != kind.value
        or document["prediction_type"] != "x"
        or type(document["out_channels"]) is not int
        or document["out_channels"] != 128
        or identity_from_dict(document["identity"]) != expected
    ):
        raise CheckpointError("model config is incompatible")
    return document["architecture"]


def _validate_model_tensors(
    model_dir: Path,
    module: nn.Module,
    expected: CheckpointIdentity,
    kind: CheckpointKind,
) -> tuple[dict[str, str], dict[str, torch.Tensor]]:
    architecture = _validate_model_config(model_dir / "config.json", expected, kind)
    try:
        if not architectures_share_parameter_contract(
            export_trainable_composite(module), architecture
        ):
            raise CheckpointError("model architecture differs from artifact")
    except (TypeError, ValueError):
        raise CheckpointError("target is not the artifact trainable composite") from None
    weight_map, declared_size = _model_index(model_dir / "model.safetensors.index.json")
    current = module.state_dict(keep_vars=True)
    if set(weight_map) != set(current):
        raise CheckpointError("checkpoint model FQNs do not match current model")
    shard_names = set(weight_map.values())
    if any("/" in name or not name.endswith(".safetensors") for name in shard_names):
        raise CheckpointError("model shard name is invalid")
    observed_names: set[str] = set()
    observed_size = 0
    for shard_name in sorted(shard_names):
        shard_path = model_dir / shard_name
        try:
            reader = cast(
                _SafeHandle,
                safe_open(shard_path, framework="pt", device="cpu"),
            )
            with reader as handle:
                shard_keys = handle.keys()
                for name in shard_keys:
                    if name in observed_names or weight_map.get(name) != shard_name:
                        raise CheckpointError("model shard keys do not match weight map")
                    target = current.get(name)
                    if target is None:
                        raise CheckpointError("model shard contains an unknown FQN")
                    view = handle.get_slice(name)
                    expected_dtype = _TORCH_TO_SAFE_DTYPE.get(target.dtype)
                    if expected_dtype is None or view.get_dtype() != expected_dtype:
                        raise CheckpointError(f"model tensor dtype differs: {name}")
                    if tuple(view.get_shape()) != tuple(target.shape):
                        raise CheckpointError(f"model tensor shape differs: {name}")
                    observed_names.add(name)
                    observed_size += target.numel() * target.element_size()
        except CheckpointError:
            raise
        except Exception:  # noqa: BLE001 - normalize backend deserialization errors
            raise CheckpointError(f"model shard is unreadable: {shard_name}") from None
    if observed_names != set(current) or observed_size != declared_size:
        raise CheckpointError("model index size or tensor set is inconsistent")
    return weight_map, current


def _apply_model(
    model_dir: Path,
    weight_map: dict[str, str],
    current: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for shard_name in sorted(set(weight_map.values())):
            tensors = load_file(model_dir / shard_name, device="cpu")
            for name, tensor in tensors.items():
                current[name].copy_(tensor, non_blocking=False)


def _validate_optimizer_schema(
    value: object,
    optimizer: IsolatedAdamW8bit,
    expected: CheckpointIdentity,
) -> None:
    del expected
    document = _mapping(value, "optimizer schema")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise CheckpointError("optimizer parameter schema version is invalid")
    groups = document["groups"]
    if not isinstance(groups, list):
        raise CheckpointError("optimizer groups must be an array")
    saved: list[tuple[str, tuple[str, ...]]] = []
    for raw_group in cast(list[object], groups):
        group = _mapping(raw_group, "optimizer group")
        _exact_keys(group, {"group_name", "param_names"}, "optimizer group")
        names = group["param_names"]
        if not isinstance(group["group_name"], str) or not isinstance(names, list) or not all(
            isinstance(name, str) for name in cast(list[object], names)
        ):
            raise CheckpointError("optimizer group is invalid")
        saved.append((group["group_name"], tuple(cast(list[str], names))))
    current: list[tuple[str, tuple[str, ...]]] = []
    for group in optimizer.optimizer.param_groups:
        group_name = group.get("group_name")
        names = group.get("param_names")
        if not isinstance(group_name, str) or not isinstance(names, list):
            raise CheckpointError("current optimizer lacks canonical parameter names")
        current.append((group_name, tuple(cast(list[str], names))))
    if saved != current:
        raise CheckpointError("optimizer canonical parameter groups do not match")


def _load_optimizer_state(path: Path) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - normalize torch safe-loader errors
        raise CheckpointError("optimizer state is not safe-loadable") from None
    if not isinstance(value, dict):
        raise CheckpointError("optimizer state has invalid top-level fields")
    document = cast(dict[str, object], value)
    if set(document) != {"state", "param_groups"}:
        raise CheckpointError("optimizer state has invalid top-level fields")
    return document


def _locked_value_equal(saved: object, current: object) -> bool:
    if isinstance(saved, torch.Tensor) or isinstance(current, torch.Tensor):
        return (
            isinstance(saved, torch.Tensor)
            and isinstance(current, torch.Tensor)
            and saved.dtype == current.dtype
            and saved.shape == current.shape
            and torch.equal(saved.cpu(), current.cpu())
        )
    if isinstance(saved, list) and isinstance(current, list):
        saved_items = cast(list[object], saved)
        current_items = cast(list[object], current)
        return len(saved_items) == len(current_items) and all(
            _locked_value_equal(left, right)
            for left, right in zip(saved_items, current_items, strict=True)
        )
    if isinstance(saved, tuple) and isinstance(current, tuple):
        saved_items = cast(tuple[object, ...], saved)
        current_items = cast(tuple[object, ...], current)
        return len(saved_items) == len(current_items) and all(
            _locked_value_equal(left, right)
            for left, right in zip(saved_items, current_items, strict=True)
        )
    if isinstance(saved, (list, tuple)) or isinstance(current, (list, tuple)):
        return False
    return type(saved) is type(current) and saved == current


def _validate_optimizer_moment(
    value: object,
    parameter: torch.Tensor,
    *,
    quantized: bool,
    signed: bool,
) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise CheckpointError("optimizer moment must be a CPU tensor")
    if tuple(value.shape) != tuple(parameter.shape):
        raise CheckpointError("optimizer moment shape does not match parameter")
    if quantized:
        if type(value).__name__ != "OptimState8bit":
            raise CheckpointError("optimizer moment class is invalid")
        moment = cast(Any, value)
        if (
            moment.block_size != 256
            or moment.signed is not signed
            or moment.codes.dtype != torch.uint8
            or tuple(moment.codes.shape) != tuple(parameter.shape)
            or moment.scale.dtype != torch.float32
            or moment.scale.ndim != 1
            or moment.scale.numel() != parameter.numel() // 256
            or moment.qmap.dtype != torch.float32
            or moment.qmap.shape != (256,)
        ):
            raise CheckpointError("quantized optimizer moment schema is invalid")
        return
    if type(value) is not torch.Tensor or value.dtype != parameter.dtype:
        raise CheckpointError("optimizer moment dtype or class is invalid")


def _validate_optimizer_state(
    document: dict[str, object],
    optimizer: IsolatedAdamW8bit,
    successful_updates: int,
) -> None:
    saved_groups = document["param_groups"]
    saved_state = document["state"]
    current_document = cast(dict[str, object], optimizer.optimizer.state_dict())
    current_groups_raw = current_document["param_groups"]
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise CheckpointError("optimizer state groups or state mapping is invalid")
    if not isinstance(current_groups_raw, list):
        raise CheckpointError("current optimizer parameter groups are invalid")
    current_groups = cast(list[object], current_groups_raw)
    saved_group_items = cast(list[object], saved_groups)
    saved_state_items = cast(dict[object, object], saved_state)
    if len(saved_group_items) != len(current_groups):
        raise CheckpointError("optimizer state group count does not match")

    audit_by_name = {spec.name: spec for spec in optimizer.audit.specs}
    expected_by_id: dict[int, ParameterSpec] = {}
    for saved_raw, current_raw in zip(
        saved_group_items, current_groups, strict=True
    ):
        if not isinstance(saved_raw, dict) or not isinstance(current_raw, dict):
            raise CheckpointError("optimizer parameter group is invalid")
        saved_group = cast(dict[str, object], saved_raw)
        current_group = cast(dict[str, object], current_raw)
        if set(saved_group) != set(current_group):
            raise CheckpointError("optimizer parameter group fields do not match")
        for key in current_group:
            # Learning rate and weight decay are runtime-controlled
            # hyperparameters. Keep current config values while restoring
            # moments and canonical parameter identity from the checkpoint.
            if key in {"lr", "weight_decay"}:
                continue
            if not _locked_value_equal(saved_group[key], current_group[key]):
                raise CheckpointError(f"optimizer parameter group differs: {key}")
        raw_ids = current_group.get("params")
        raw_names = current_group.get("param_names")
        if not isinstance(raw_ids, list) or not isinstance(raw_names, list):
            raise CheckpointError("optimizer canonical parameter IDs are invalid")
        id_items = cast(list[object], raw_ids)
        name_items = cast(list[object], raw_names)
        if (
            not all(type(item) is int for item in id_items)
            or not all(isinstance(item, str) for item in name_items)
            or len(id_items) != len(name_items)
        ):
            raise CheckpointError("optimizer canonical parameter IDs are invalid")
        for parameter_id, name in zip(
            cast(list[int], id_items), cast(list[str], name_items), strict=True
        ):
            spec = audit_by_name.get(name)
            if spec is None or parameter_id in expected_by_id:
                raise CheckpointError("optimizer parameter identity is invalid")
            expected_by_id[parameter_id] = spec
    if set(audit_by_name) != {spec.name for spec in expected_by_id.values()}:
        raise CheckpointError("optimizer parameter state omits canonical parameters")

    state_ids = set(saved_state_items)
    if not state_ids <= set(expected_by_id) or (successful_updates == 0 and state_ids):
        raise CheckpointError("optimizer initialized state set does not match update")
    for parameter_id, raw_parameter_state in saved_state_items.items():
        if type(parameter_id) is not int or not isinstance(raw_parameter_state, dict):
            raise CheckpointError("optimizer per-parameter state is invalid")
        parameter_state = cast(dict[str, object], raw_parameter_state)
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise CheckpointError("optimizer per-parameter state fields are invalid")
        step = parameter_state["step"]
        if (
            type(step) is not torch.Tensor
            or step.device.type != "cpu"
            or step.shape != ()
            or step.dtype != torch.float32
            or not math.isfinite(float(step.item()))
            or not float(step.item()).is_integer()
            or not 1 <= int(step.item()) <= successful_updates
        ):
            raise CheckpointError("optimizer step is outside successful update history")
        spec = expected_by_id[parameter_id]
        parameter = spec.parameter
        quantized = parameter.numel() >= 4096 and parameter.numel() % 256 == 0
        _validate_optimizer_moment(
            parameter_state["exp_avg"], parameter, quantized=quantized, signed=True
        )
        _validate_optimizer_moment(
            parameter_state["exp_avg_sq"], parameter, quantized=quantized, signed=False
        )


def _load_sr_rng(path: Path, optimizer: IsolatedAdamW8bit) -> dict[str, object]:
    try:
        tensors = load_file(path, device="cpu")
    except Exception:  # noqa: BLE001 - normalize Safetensors loader errors
        raise CheckpointError("optimizer SR RNG file is unreadable") from None
    if set(tensors) != {"device_index", "state"}:
        raise CheckpointError("optimizer SR RNG tensors are invalid")
    device_index = tensors["device_index"]
    state = tensors["state"]
    if device_index.shape != () or device_index.dtype != torch.int64:
        raise CheckpointError("optimizer SR RNG device is invalid")
    if state.dtype != torch.uint8 or state.ndim != 1:
        raise CheckpointError("optimizer SR RNG state is invalid")
    saved_index = int(device_index.item())
    current_index = optimizer.sr_rng.device.index
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    remapping_rank_zero = (
        local_rank > 0 and saved_index == 0 and current_index == local_rank
    )
    if saved_index != current_index and not remapping_rank_zero:
        raise CheckpointError("optimizer SR RNG device does not match")
    if state.shape != optimizer.sr_rng.state.shape:
        raise CheckpointError("optimizer SR RNG shape does not match")
    return {"device_type": "cuda", "device_index": current_index, "state": state}


def _model_manifest_records(model_dir: Path) -> tuple[FileRecord, ...]:
    document = _mapping(
        _read_json(model_dir / "manifest.json", "model manifest"), "model manifest"
    )
    _exact_keys(document, {"schema_version", "files"}, "model manifest")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["files"], list)
    ):
        raise CheckpointError("model manifest is invalid")
    records: list[FileRecord] = []
    try:
        for item in cast(list[object], document["files"]):
            record = _mapping(item, "model file record")
            records.append(FileRecord(record["path"], record["size"]))
    except (KeyError, TypeError, ValueError):
        raise CheckpointError("model manifest is invalid") from None
    expected = {record.path for record in records}
    if not expected or len(expected) != len(records):
        raise CheckpointError("model manifest file set is invalid")
    weight_map, _declared_size = _model_index(
        model_dir / "model.safetensors.index.json"
    )
    shard_names = set(weight_map.values())
    if any("/" in name or not name.endswith(".safetensors") for name in shard_names):
        raise CheckpointError("model shard name is invalid")
    if expected != {
        "config.json",
        "model.safetensors.index.json",
        *shard_names,
    }:
        raise CheckpointError("model manifest contains an unknown payload")
    return tuple(records)


def _validate_standalone_model_manifest(model_dir: Path) -> None:
    records = _model_manifest_records(model_dir)
    expected = {record.path for record in records}
    actual: set[str] = set()
    for path in model_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise CheckpointError("model directory contains an invalid entry")
        if path.name != "manifest.json":
            actual.add(path.name)
    if actual != expected:
        raise CheckpointError("model directory file set does not match manifest")
    for record in records:
        path = model_dir / record.path
        try:
            if path.stat().st_size != record.size:
                raise CheckpointError(f"model directory size differs: {record.path}")
        except OSError:
            raise CheckpointError(f"model directory file is unreadable: {record.path}") from None


def _config_identity_kind(model_dir: Path) -> tuple[CheckpointIdentity, CheckpointKind, object]:
    document = _mapping(_read_json(model_dir / "config.json", "model config"), "model config")
    try:
        identity = identity_from_dict(document.get("identity"))
        kind = CheckpointKind(document.get("kind"))
    except (CheckpointError, TypeError, ValueError):
        raise CheckpointError("model config identity or kind is invalid") from None
    architecture = _validate_model_config(model_dir / "config.json", identity, kind)
    return identity, kind, architecture


def _load_model_kind(
    checkpoint: Path,
    module: nn.Module,
    expected: CheckpointIdentity,
    kind: CheckpointKind,
) -> None:
    manifest = read_checkpoint_manifest(checkpoint)
    _validate_identity(manifest, expected, kind)
    model_dir = checkpoint / "model"
    weight_map, current = _validate_model_tensors(model_dir, module, expected, kind)
    _apply_model(model_dir, weight_map, current)


def load_model_only(
    checkpoint: Path,
    module: nn.Module,
    expected: CheckpointIdentity,
) -> None:
    _load_model_kind(checkpoint, module, expected, CheckpointKind.MODEL_ONLY)


def load_model_directory(
    model_dir: Path,
    *,
    device: torch.device | str,
) -> tuple[nn.Module, CheckpointIdentity, CheckpointKind]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise CheckpointError("model directory must be a real directory")
    _validate_standalone_model_manifest(model_dir)
    identity, kind, architecture = _config_identity_kind(model_dir)
    try:
        module = build_trainable_composite(architecture, device=device)
    except (TypeError, ValueError):
        raise CheckpointError("model architecture is invalid") from None
    weight_map, current = _validate_model_tensors(
        model_dir, module, identity, kind
    )
    _apply_model(model_dir, weight_map, current)
    return module, identity, kind


def load_inference_artifact(
    checkpoint: Path,
    expected: CheckpointIdentity,
    *,
    device: torch.device | str,
) -> nn.Module:
    manifest = read_checkpoint_manifest(checkpoint)
    if manifest.identity != expected:
        raise CheckpointError("checkpoint identity does not match the requested run")
    model_dir = checkpoint / "model"
    architecture = _validate_model_config(
        model_dir / "config.json", expected, manifest.kind
    )
    try:
        module = build_trainable_composite(architecture, device=device)
    except (TypeError, ValueError):
        raise CheckpointError("model architecture is invalid") from None
    weight_map, current = _validate_model_tensors(
        model_dir, module, expected, manifest.kind
    )
    _apply_model(model_dir, weight_map, current)
    return module


def load_raw_checkpoint(
    checkpoint: Path,
    module: nn.Module,
    optimizer: IsolatedAdamW8bit,
    expected: CheckpointIdentity,
) -> RawCheckpointState:
    manifest = read_checkpoint_manifest(checkpoint)
    _validate_identity(manifest, expected, CheckpointKind.RAW)
    _validate_raw_sidecars(checkpoint, manifest)
    try:
        validate_optimizer_coverage(
            module,
            tuple((spec.name, spec.parameter) for spec in optimizer.audit.specs),
        )
    except (TypeError, ValueError):
        raise CheckpointError("checkpoint module and optimizer boundary differ") from None
    weight_map, current_model = _validate_model_tensors(
        checkpoint / "model", module, expected, CheckpointKind.RAW
    )
    train_state = checkpoint / "train_state"
    _validate_optimizer_schema(
        _read_json(train_state / "optimizer_schema.json", "optimizer schema"),
        optimizer,
        expected,
    )
    optimizer_state = _load_optimizer_state(train_state / "optimizer.pt")
    state = raw_state_from_dicts(
        _read_json(train_state / "trainer_state.json", "trainer state"),
        _read_json(train_state / "growth_state.json", "growth state"),
    )
    if expected.update != state.trainer.successful_updates:
        raise CheckpointError("checkpoint update differs from trainer successful updates")
    if state.growth.active_slot_ids != active_slot_ids_from_module(module):
        raise CheckpointError("checkpoint growth state differs from model active slots")
    _validate_optimizer_state(
        optimizer_state, optimizer, state.trainer.successful_updates
    )
    try:
        rank_rng = load_file(train_state / "rng" / "rank-0.safetensors", device="cpu")
    except Exception:  # noqa: BLE001 - normalize Safetensors loader errors
        raise CheckpointError("rank RNG file is unreadable") from None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank > 0 and torch.cuda.is_available():
        rank_rng = dict(rank_rng)
        rank_rng["cuda_device_index"] = torch.tensor(
            torch.cuda.current_device(), dtype=torch.int64
        )
    validate_rank_rng(rank_rng)
    sr_rng = _load_sr_rng(train_state / "rng" / "optimizer_sr.safetensors", optimizer)

    _apply_model(checkpoint / "model", weight_map, current_model)
    current_groups = cast(
        list[object],
        cast(dict[str, object], optimizer.optimizer.state_dict())["param_groups"],
    )
    saved_groups = cast(list[object], optimizer_state["param_groups"])
    if len(saved_groups) != len(current_groups):
        raise CheckpointError("optimizer state group count does not match")
    runtime_optimizer_state = dict(optimizer_state)
    runtime_optimizer_state["param_groups"] = [
        {
            **cast(dict[str, object], saved_group),
            "lr": cast(dict[str, object], current_group)["lr"],
            "weight_decay": cast(dict[str, object], current_group)[
                "weight_decay"
            ],
        }
        for saved_group, current_group in zip(
            saved_groups, current_groups, strict=True
        )
    ]
    optimizer.load_state_dict(
        {
            "optimizer": runtime_optimizer_state,
            "sr_rng": sr_rng,
        }
    )
    restore_rank_rng(rank_rng)
    return state


def discover_complete_checkpoints(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    complete: list[Path] = []
    for path in sorted(root.iterdir()):
        if (
            path.is_symlink()
            or not path.is_dir()
            or _FINAL_CHECKPOINT_NAME.fullmatch(path.name) is None
        ):
            continue
        try:
            marker = path / "COMPLETE"
            manifest_path = path / "manifest.json"
            if marker.is_symlink() or marker.read_bytes() != b"complete\n":
                continue
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            manifest = manifest_from_dict(_read_json(manifest_path, "manifest"))
        except (CheckpointError, OSError):
            continue
        prefixes = {
            CheckpointKind.RAW: "ckpt",
            CheckpointKind.MODEL_ONLY: "model",
            CheckpointKind.PMA: "pma",
            CheckpointKind.RELEASE: "release",
        }
        prefix = prefixes[manifest.kind]
        expected_name = (
            f"{prefix}_{manifest.identity.update}_{manifest.identity.checkpoint_id}"
        )
        if path.name == expected_name:
            complete.append(path)
    return tuple(complete)


__all__ = [
    "discover_complete_checkpoints",
    "load_inference_artifact",
    "load_model_directory",
    "load_model_only",
    "load_raw_checkpoint",
    "read_checkpoint_manifest",
    "read_raw_checkpoint_state",
]
