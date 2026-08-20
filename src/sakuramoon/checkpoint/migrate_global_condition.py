"""Explicitly migrate one RAW architecture-v2 checkpoint to architecture v3.

The normal loader remains fail-closed. Migration copies the source checkpoint,
adds the zero-initialized global-condition parameters in a new safetensors
shard, remaps optimizer parameter IDs by canonical FQN, and atomically
publishes a distinct destination directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from sakuramoon.checkpoint.artifact import build_trainable_composite
from sakuramoon.checkpoint.load import read_checkpoint_manifest
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    identity_from_dict,
    manifest_to_dict,
)
from sakuramoon.optim.groups import audit_trainable_parameters

SOURCE_ARCHITECTURE_SCHEMA_VERSION = 2
TARGET_ARCHITECTURE_SCHEMA_VERSION = 3
GLOBAL_CONDITION_SHARD = "model-global-condition.safetensors"
_NEW_PARAMETERS = frozenset(
    {
        "dit.conditioner.condition_global_norm.weight",
        "dit.conditioner.condition_global_projection.weight",
    }
)
_MODEL_CONFIG_KEYS = {
    "architecture",
    "schema_version",
    "kind",
    "identity",
    "prediction_type",
    "out_channels",
}
_GROUP_ORDER = ("matrix_decay", "sensitive_no_decay")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError(f"{name} is unreadable or invalid JSON") from None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _replace_bytes(path: Path, value: bytes) -> None:
    if path.is_symlink() or not path.is_file():
        raise CheckpointError(f"migration target file is invalid: {path.name}")
    path.write_bytes(value)


def _records(root: Path, *, recursive: bool) -> tuple[FileRecord, ...]:
    paths = root.rglob("*") if recursive else root.iterdir()
    records: list[FileRecord] = []
    for path in sorted(paths):
        if path.is_symlink():
            raise CheckpointError("migration output contains a symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "COMPLETE"}:
            continue
        records.append(FileRecord(relative, path.stat().st_size))
    return tuple(records)


def migrate_architecture(value: object) -> dict[str, object]:
    document = _mapping(value, "source model architecture")
    if (
        set(document)
        != {"schema_version", "class", "dit", "text", "condition_tokens"}
        or document.get("schema_version") != SOURCE_ARCHITECTURE_SCHEMA_VERSION
        or document.get("class") != "TrainableComposite"
    ):
        raise CheckpointError("source architecture is not canonical schema v2")
    migrated = dict(document)
    migrated["schema_version"] = TARGET_ARCHITECTURE_SCHEMA_VERSION
    return cast(dict[str, object], migrated)


def _new_model_tensors(architecture: object) -> dict[str, torch.Tensor]:
    document = _mapping(architecture, "target model architecture")
    dit = _mapping(document.get("dit"), "target DiT architecture")
    model_dim = dit.get("hidden_size")
    hidden_dim = dit.get("condition_hidden_size")
    if (
        type(model_dim) is not int
        or type(hidden_dim) is not int
        or model_dim <= 0
        or hidden_dim <= 0
    ):
        raise CheckpointError("target global-condition dimensions are invalid")
    return {
        "dit.conditioner.condition_global_norm.weight": torch.ones(
            model_dim, dtype=torch.float32
        ),
        "dit.conditioner.condition_global_projection.weight": torch.zeros(
            hidden_dim, model_dim, dtype=torch.float32
        ),
    }


def _optimizer_groups(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise CheckpointError(f"{name} must be an array")
    groups: list[dict[str, object]] = []
    for index, raw_group in enumerate(cast(list[object], value)):
        group = _mapping(raw_group, f"{name}[{index}]")
        names = group.get("param_names")
        ids = group.get("params")
        if (
            not isinstance(names, list)
            or not isinstance(ids, list)
            or len(names) != len(ids)
            or not all(isinstance(item, str) for item in names)
            or not all(type(item) is int for item in ids)
        ):
            raise CheckpointError(f"{name}[{index}] parameter identity is invalid")
        groups.append(group)
    return groups


def _target_group_names(architecture: object) -> dict[str, list[str]]:
    try:
        module = build_trainable_composite(architecture, device="meta")
        audit = audit_trainable_parameters(
            module,
            matrix_weight_decay=0.0,
            sensitive_weight_decay=0.0,
        )
    except (TypeError, ValueError):
        raise CheckpointError("target architecture cannot produce canonical groups") from None
    return {
        "matrix_decay": [spec.name for spec in audit.decay],
        "sensitive_no_decay": [spec.name for spec in audit.sensitive],
    }


def migrate_optimizer_state(
    optimizer_state: object,
    optimizer_schema: object,
    target_architecture: object,
) -> tuple[dict[str, object], dict[str, object]]:
    state_document = _mapping(optimizer_state, "source optimizer state")
    if set(state_document) != {"state", "param_groups"}:
        raise CheckpointError("source optimizer state fields are invalid")
    raw_state = state_document["state"]
    if not isinstance(raw_state, dict):
        raise CheckpointError("source optimizer parameter state is invalid")
    state_by_id = cast(dict[object, object], raw_state)

    schema_document = _mapping(optimizer_schema, "source optimizer schema")
    if (
        set(schema_document) != {"groups", "schema_version"}
        or schema_document.get("schema_version") != 1
        or not isinstance(schema_document.get("groups"), list)
    ):
        raise CheckpointError("source optimizer schema is invalid")
    state_groups = _optimizer_groups(
        state_document["param_groups"], "source optimizer state groups"
    )
    schema_groups = cast(list[object], schema_document["groups"])
    if len(state_groups) != len(schema_groups):
        raise CheckpointError("source optimizer group counts differ")

    state_group_by_name: dict[str, dict[str, object]] = {}
    old_id_by_name: dict[str, int] = {}
    observed_ids: set[int] = set()
    for index, (state_group, raw_schema_group) in enumerate(
        zip(state_groups, schema_groups, strict=True)
    ):
        schema_group = _mapping(raw_schema_group, f"source optimizer schema group {index}")
        if set(schema_group) != {"group_name", "param_names"}:
            raise CheckpointError("source optimizer schema group fields are invalid")
        group_name = state_group.get("group_name")
        state_names = cast(list[str], state_group["param_names"])
        state_ids = cast(list[int], state_group["params"])
        if (
            not isinstance(group_name, str)
            or group_name in state_group_by_name
            or schema_group.get("group_name") != group_name
            or schema_group.get("param_names") != state_names
        ):
            raise CheckpointError("source optimizer state and schema groups differ")
        state_group_by_name[group_name] = state_group
        for name, parameter_id in zip(state_names, state_ids, strict=True):
            if name in old_id_by_name or parameter_id in observed_ids:
                raise CheckpointError("source optimizer parameter identity is duplicated")
            old_id_by_name[name] = parameter_id
            observed_ids.add(parameter_id)
    if set(state_group_by_name) != set(_GROUP_ORDER):
        raise CheckpointError("source optimizer group names are invalid")
    if not set(state_by_id) <= observed_ids:
        raise CheckpointError("source optimizer state references an unknown parameter")

    target_groups = _target_group_names(target_architecture)
    target_names = {name for names in target_groups.values() for name in names}
    old_names = set(old_id_by_name)
    if old_names - target_names or target_names - old_names != set(_NEW_PARAMETERS):
        raise CheckpointError("target optimizer parameter delta is not the global path")

    next_id = 0
    migrated_state: dict[int, object] = {}
    migrated_groups: list[dict[str, object]] = []
    migrated_schema_groups: list[dict[str, object]] = []
    for group_name in _GROUP_ORDER:
        names = target_groups[group_name]
        ids = list(range(next_id, next_id + len(names)))
        next_id += len(names)
        group = dict(state_group_by_name[group_name])
        group["param_names"] = names
        group["params"] = ids
        migrated_groups.append(group)
        migrated_schema_groups.append(
            {"group_name": group_name, "param_names": names}
        )
        for name, new_id in zip(names, ids, strict=True):
            old_id = old_id_by_name.get(name)
            if old_id is not None and old_id in state_by_id:
                migrated_state[new_id] = state_by_id[old_id]
    return (
        {"state": migrated_state, "param_groups": migrated_groups},
        {"groups": migrated_schema_groups, "schema_version": 1},
    )


def _migrate_model_directory(
    model_dir: Path,
    *,
    manifest: CheckpointManifest,
) -> dict[str, object]:
    config_path = model_dir / "config.json"
    config = _mapping(_read_json(config_path, "source model config"), "source model config")
    if set(config) != _MODEL_CONFIG_KEYS:
        raise CheckpointError("source model config fields are invalid")
    if (
        config.get("schema_version") != 1
        or config.get("kind") != CheckpointKind.RAW.value
        or config.get("prediction_type") != "x"
        or config.get("out_channels") != 128
        or identity_from_dict(config.get("identity")) != manifest.identity
    ):
        raise CheckpointError("source model config is incompatible")
    architecture = migrate_architecture(config["architecture"])
    new_tensors = _new_model_tensors(architecture)

    index_path = model_dir / "model.safetensors.index.json"
    index = _mapping(_read_json(index_path, "source model index"), "source model index")
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointError("source model index fields are invalid")
    metadata = _mapping(index["metadata"], "source model index metadata")
    weight_map = _mapping(index["weight_map"], "source model weight map")
    total_size = metadata.get("total_size")
    if (
        set(metadata) != {"total_size"}
        or type(total_size) is not int
        or total_size <= 0
        or not weight_map
        or not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items())
        or set(weight_map) & _NEW_PARAMETERS
        or GLOBAL_CONDITION_SHARD in set(weight_map.values())
        or (model_dir / GLOBAL_CONDITION_SHARD).exists()
    ):
        raise CheckpointError("source model index is incompatible")
    save_file(new_tensors, str(model_dir / GLOBAL_CONDITION_SHARD))
    migrated_weight_map = cast(dict[str, str], dict(weight_map))
    migrated_weight_map.update(
        {name: GLOBAL_CONDITION_SHARD for name in sorted(new_tensors)}
    )
    added_size = sum(tensor.numel() * tensor.element_size() for tensor in new_tensors.values())
    config["architecture"] = architecture
    _replace_bytes(config_path, _json_bytes(config))
    _replace_bytes(
        index_path,
        _json_bytes(
            {
                "metadata": {"total_size": total_size + added_size},
                "weight_map": migrated_weight_map,
            }
        ),
    )
    model_records = _records(model_dir, recursive=False)
    _replace_bytes(
        model_dir / "manifest.json",
        _json_bytes(
            {
                "files": [
                    {"path": record.path, "size": record.size}
                    for record in model_records
                ],
                "schema_version": 1,
            }
        ),
    )
    return architecture


def _migrate_raw_optimizer(checkpoint: Path, architecture: object) -> None:
    train_state = checkpoint / "train_state"
    optimizer_path = train_state / "optimizer.pt"
    schema_path = train_state / "optimizer_schema.json"
    try:
        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - backend normalization boundary
        raise CheckpointError("source optimizer state is not safe-loadable") from None
    migrated_optimizer, migrated_schema = migrate_optimizer_state(
        optimizer,
        _read_json(schema_path, "source optimizer schema"),
        architecture,
    )
    torch.save(migrated_optimizer, optimizer_path)
    _replace_bytes(schema_path, _json_bytes(migrated_schema))


def migrate_checkpoint(source: Path, destination: Path) -> Path:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    manifest = read_checkpoint_manifest(source)
    if manifest.kind is not CheckpointKind.RAW:
        raise CheckpointError("global-condition migration accepts RAW checkpoints only")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"migration destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise CheckpointError("migration destination parent must be a real directory")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"migration temporary path already exists: {temporary}")
    try:
        shutil.copytree(source, temporary, symlinks=False)
        architecture = _migrate_model_directory(temporary / "model", manifest=manifest)
        _migrate_raw_optimizer(temporary, architecture)
        records = _records(temporary, recursive=True)
        migrated_manifest = CheckpointManifest(
            kind=manifest.kind,
            identity=manifest.identity,
            files=records,
        )
        _replace_bytes(
            temporary / "manifest.json",
            _json_bytes(manifest_to_dict(migrated_manifest)),
        )
        if (temporary / "COMPLETE").read_bytes() != b"complete\n":
            raise CheckpointError("migration output lost the completion marker")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    read_checkpoint_manifest(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(migrate_checkpoint(args.source, args.destination))


if __name__ == "__main__":
    main()


__all__ = [
    "GLOBAL_CONDITION_SHARD",
    "SOURCE_ARCHITECTURE_SCHEMA_VERSION",
    "TARGET_ARCHITECTURE_SCHEMA_VERSION",
    "migrate_architecture",
    "migrate_checkpoint",
    "migrate_optimizer_state",
]
