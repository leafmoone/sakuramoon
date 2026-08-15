"""Explicitly migrate a four-style-token checkpoint to eight condition tokens.

This module is intentionally never called by the training loader. A legacy
checkpoint must be converted to a new directory before production training can
resume; incompatible artifacts continue to fail closed in the normal loader.
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
from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]

from sakuramoon.checkpoint.load import read_checkpoint_manifest
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    identity_from_dict,
    manifest_to_dict,
)

LEGACY_TOKEN_COUNT = 4
TARGET_TOKEN_COUNT = 8
TARGET_ARCHITECTURE_SCHEMA_VERSION = 2
_MODEL_CONFIG_KEYS = {
    "architecture",
    "schema_version",
    "kind",
    "identity",
    "prediction_type",
    "out_channels",
}
_EXPANDED_PARAMETERS = frozenset(
    {
        "condition_tokens.queries",
        "condition_tokens.null_tokens",
    }
)


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


def _rename_parameter(name: str) -> str:
    if name == "dit.modality.style":
        return "dit.modality.condition"
    if name.startswith("style."):
        suffix = name.removeprefix("style.")
        if suffix.startswith("style_mlp."):
            suffix = "condition_mlp." + suffix.removeprefix("style_mlp.")
        return "condition_tokens." + suffix
    return name


def migrate_architecture(value: object) -> dict[str, object]:
    """Convert the exact legacy architecture document without fallback."""

    document = _mapping(value, "legacy model architecture")
    if set(document) != {"class", "dit", "text", "style"}:
        raise CheckpointError("legacy model architecture fields are invalid")
    if document.get("class") != "TrainableComposite":
        raise CheckpointError("legacy model class is invalid")
    legacy_style = _mapping(document["style"], "legacy style architecture")
    if legacy_style.get("query_count") != LEGACY_TOKEN_COUNT:
        raise CheckpointError("legacy style query count must be exactly four")
    condition_tokens = dict(legacy_style)
    del condition_tokens["query_count"]
    condition_tokens["token_count"] = TARGET_TOKEN_COUNT

    dit = dict(_mapping(document["dit"], "legacy DiT architecture"))
    if "condition_token_count" in dit:
        raise CheckpointError("legacy DiT unexpectedly declares condition tokens")
    dit["condition_token_count"] = TARGET_TOKEN_COUNT
    return {
        "schema_version": TARGET_ARCHITECTURE_SCHEMA_VERSION,
        "class": "TrainableComposite",
        "dit": dit,
        "text": document["text"],
        "condition_tokens": condition_tokens,
    }


def _expand_parameter(
    name: str,
    tensor: torch.Tensor,
    *,
    init_std: float,
) -> torch.Tensor:
    if name not in {"style.queries", "style.null_tokens"}:
        return tensor
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.ndim != 2
        or tensor.shape[0] != LEGACY_TOKEN_COUNT
    ):
        raise CheckpointError(f"legacy expandable tensor is invalid: {name}")
    seed = 2026081501 if name == "style.queries" else 2026081502
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    extra = torch.empty(
        TARGET_TOKEN_COUNT - LEGACY_TOKEN_COUNT,
        tensor.shape[1],
        dtype=tensor.dtype,
        device="cpu",
    )
    extra.normal_(mean=0.0, std=init_std, generator=generator)
    return torch.cat((tensor, extra), dim=0).contiguous()


def migrate_model_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    init_std: float,
) -> dict[str, torch.Tensor]:
    """Rename one shard and deterministically initialize the four new rows."""

    if not tensors:
        raise CheckpointError("legacy model shard is empty")
    migrated: dict[str, torch.Tensor] = {}
    for name, tensor in sorted(tensors.items()):
        if type(name) is not str or not isinstance(tensor, torch.Tensor):
            raise CheckpointError("legacy model shard entries are invalid")
        target_name = _rename_parameter(name)
        if target_name in migrated:
            raise CheckpointError(f"migrated model FQN collides: {target_name}")
        migrated[target_name] = _expand_parameter(
            name,
            tensor,
            init_std=init_std,
        )
    return migrated


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
            or not all(isinstance(item, str) for item in cast(list[object], names))
            or not isinstance(ids, list)
            or not all(type(item) is int for item in cast(list[object], ids))
            or len(names) != len(ids)
        ):
            raise CheckpointError(f"{name}[{index}] parameter identity is invalid")
        groups.append(cast(dict[str, object], group))
    return groups


def migrate_optimizer_state(
    optimizer_state: object,
    optimizer_schema: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Rename canonical FQNs and drop moments only for the expanded tensors."""

    state_document = _mapping(optimizer_state, "legacy optimizer state")
    if set(state_document) != {"state", "param_groups"}:
        raise CheckpointError("legacy optimizer state fields are invalid")
    saved_state = state_document["state"]
    if not isinstance(saved_state, dict):
        raise CheckpointError("legacy optimizer parameter state is invalid")
    state_by_id = cast(dict[object, object], saved_state)

    schema_document = _mapping(optimizer_schema, "legacy optimizer schema")
    if set(schema_document) != {"groups", "schema_version"}:
        raise CheckpointError("legacy optimizer schema fields are invalid")
    if schema_document["schema_version"] != 1:
        raise CheckpointError("legacy optimizer schema version is invalid")
    state_groups = _optimizer_groups(
        state_document["param_groups"], "legacy optimizer state groups"
    )
    raw_schema_groups = schema_document["groups"]
    if not isinstance(raw_schema_groups, list):
        raise CheckpointError("legacy optimizer schema groups must be an array")
    schema_groups = cast(list[object], raw_schema_groups)
    if len(state_groups) != len(schema_groups):
        raise CheckpointError("legacy optimizer group counts differ")

    old_state_ids: set[int] = set()
    next_parameter_id = 0
    migrated_state: dict[int, object] = {}
    migrated_groups: list[dict[str, object]] = []
    migrated_schema_groups: list[dict[str, object]] = []
    for index, (state_group, raw_schema_group) in enumerate(
        zip(state_groups, schema_groups, strict=True)
    ):
        schema_group = _mapping(raw_schema_group, f"legacy optimizer schema group {index}")
        if set(schema_group) != {"group_name", "param_names"}:
            raise CheckpointError("legacy optimizer schema group fields are invalid")
        state_group_name = state_group.get("group_name")
        schema_group_name = schema_group.get("group_name")
        state_names = cast(list[str], state_group["param_names"])
        schema_names = schema_group.get("param_names")
        if (
            not isinstance(state_group_name, str)
            or state_group_name != schema_group_name
            or not isinstance(schema_names, list)
            or state_names != cast(list[object], schema_names)
        ):
            raise CheckpointError("legacy optimizer state and schema groups differ")
        old_ids = cast(list[int], state_group["params"])
        renamed = sorted(
            (_rename_parameter(name), old_id)
            for name, old_id in zip(state_names, old_ids, strict=True)
        )
        if len({name for name, _old_id in renamed}) != len(renamed):
            raise CheckpointError("migrated optimizer names collide")
        target_names: list[str] = []
        target_ids: list[int] = []
        for target_name, old_id in renamed:
            if old_id in old_state_ids:
                raise CheckpointError("legacy optimizer parameter IDs are duplicated")
            old_state_ids.add(old_id)
            target_id = next_parameter_id
            next_parameter_id += 1
            target_names.append(target_name)
            target_ids.append(target_id)
            if target_name not in _EXPANDED_PARAMETERS and old_id in state_by_id:
                migrated_state[target_id] = state_by_id[old_id]
        migrated_group = dict(state_group)
        migrated_group["param_names"] = target_names
        migrated_group["params"] = target_ids
        migrated_groups.append(migrated_group)
        migrated_schema_groups.append(
            {"group_name": state_group_name, "param_names": target_names}
        )

    if not set(state_by_id) <= old_state_ids:
        raise CheckpointError("legacy optimizer state references an unknown parameter")
    return (
        {"state": migrated_state, "param_groups": migrated_groups},
        {"groups": migrated_schema_groups, "schema_version": 1},
    )


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


def _migrate_model_directory(
    model_dir: Path,
    *,
    manifest: CheckpointManifest,
) -> None:
    config_path = model_dir / "config.json"
    config = _mapping(_read_json(config_path, "legacy model config"), "legacy model config")
    if set(config) != _MODEL_CONFIG_KEYS:
        raise CheckpointError("legacy model config fields are invalid")
    if (
        config.get("schema_version") != 1
        or config.get("kind") != manifest.kind.value
        or config.get("prediction_type") != "x"
        or config.get("out_channels") != 128
        or identity_from_dict(config.get("identity")) != manifest.identity
    ):
        raise CheckpointError("legacy model config is incompatible")
    architecture = migrate_architecture(config["architecture"])
    condition_config = _mapping(
        architecture["condition_tokens"], "migrated condition-token architecture"
    )
    init_std = condition_config.get("init_std")
    if type(init_std) is not float or not 0.0 < init_std <= 1.0:
        raise CheckpointError("legacy condition initialization scale is invalid")

    index_path = model_dir / "model.safetensors.index.json"
    index = _mapping(_read_json(index_path, "legacy model index"), "legacy model index")
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointError("legacy model index fields are invalid")
    weight_map = _mapping(index["weight_map"], "legacy model weight map")
    if not weight_map or not all(isinstance(value, str) for value in weight_map.values()):
        raise CheckpointError("legacy model weight map is invalid")
    shard_names = sorted(set(cast(dict[str, str], weight_map).values()))
    if any("/" in name or not name.endswith(".safetensors") for name in shard_names):
        raise CheckpointError("legacy model shard name is invalid")

    migrated_weight_map: dict[str, str] = {}
    observed_old_names: set[str] = set()
    total_size = 0
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        try:
            shard = cast(dict[str, torch.Tensor], load_file(shard_path, device="cpu"))
        except Exception:  # noqa: BLE001 - backend normalization boundary
            raise CheckpointError(f"legacy model shard is unreadable: {shard_name}") from None
        if any(
            name in observed_old_names or weight_map.get(name) != shard_name
            for name in shard
        ):
            raise CheckpointError("legacy model shard keys differ from the index")
        observed_old_names.update(shard)
        migrated = migrate_model_tensors(shard, init_std=init_std)
        for name, tensor in migrated.items():
            if name in migrated_weight_map:
                raise CheckpointError(f"migrated model FQN is duplicated: {name}")
            migrated_weight_map[name] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        save_file(migrated, str(shard_path))
    if observed_old_names != set(weight_map):
        raise CheckpointError("legacy model index omits or invents tensor names")
    if not _EXPANDED_PARAMETERS <= set(migrated_weight_map):
        raise CheckpointError("legacy checkpoint lacks expandable condition parameters")

    config["architecture"] = architecture
    _replace_bytes(config_path, _json_bytes(config))
    _replace_bytes(
        index_path,
        _json_bytes(
            {
                "metadata": {"total_size": total_size},
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


def _migrate_raw_optimizer(checkpoint: Path) -> None:
    train_state = checkpoint / "train_state"
    optimizer_path = train_state / "optimizer.pt"
    schema_path = train_state / "optimizer_schema.json"
    try:
        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - backend normalization boundary
        raise CheckpointError("legacy optimizer state is not safe-loadable") from None
    migrated_optimizer, migrated_schema = migrate_optimizer_state(
        optimizer,
        _read_json(schema_path, "legacy optimizer schema"),
    )
    torch.save(migrated_optimizer, optimizer_path)
    _replace_bytes(schema_path, _json_bytes(migrated_schema))


def migrate_checkpoint(source: Path, destination: Path) -> Path:
    """Atomically publish a strict migrated copy and never mutate ``source``."""

    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    manifest = read_checkpoint_manifest(source)
    if manifest.kind not in {CheckpointKind.RAW, CheckpointKind.MODEL_ONLY}:
        raise CheckpointError("only raw and model-only checkpoints can be migrated")
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
        _migrate_model_directory(temporary / "model", manifest=manifest)
        if manifest.kind is CheckpointKind.RAW:
            _migrate_raw_optimizer(temporary)
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
    migrated = migrate_checkpoint(args.source, args.destination)
    print(migrated)


if __name__ == "__main__":
    main()


__all__ = [
    "LEGACY_TOKEN_COUNT",
    "TARGET_TOKEN_COUNT",
    "migrate_architecture",
    "migrate_checkpoint",
    "migrate_model_tensors",
    "migrate_optimizer_state",
]
