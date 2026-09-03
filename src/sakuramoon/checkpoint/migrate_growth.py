"""Atomic, fail-closed 16-layer S0 -> 20-layer G1 RAW migration.

The migrator never mutates the source.  Existing model shards and optimizer
states are copied by identity; only the four approved stable slots and their
conditioner biases are added to a new growth shard.  A target is published
only after all FQN, topology, trainer-state and optimizer-contract checks pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from sakuramoon.checkpoint.artifact import build_trainable_composite
from sakuramoon.checkpoint.load import (
    read_checkpoint_manifest,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
    manifest_to_dict,
    raw_state_to_dict,
)
from sakuramoon.model.block import DiTBlock, PackedDiTBlock
from sakuramoon.model.growth import (
    G1_NEW_SLOT_IDS,
    active_slot_ids,
    growth_ramp_updates,
    slot_name,
)
from sakuramoon.optim.groups import audit_trainable_parameters

G1_TARGET_STAGE = "G1"
G1_TARGET_DEPTH = 20
G1_TARGET_WORLD_SIZE = 2
G1_TARGET_RESOLUTION = 256
GROWTH_SHARD = "model-growth.safetensors"
GROWTH_MIGRATION_SIDECAR = "train_state/growth_migration.json"
_GROUP_ORDER = ("matrix_decay", "sensitive_no_decay")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError(f"{name} is unreadable or invalid JSON") from None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))


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


def _exact_complete_source(source: Path) -> Path:
    if not source.is_absolute() or source.is_symlink():
        raise CheckpointError("source must be an absolute, non-symlink COMPLETE path")
    resolved = source.resolve(strict=True)
    if resolved != source or not resolved.is_dir():
        raise CheckpointError("source must be an exact checkpoint directory")
    manifest = read_checkpoint_manifest(source)
    if manifest.kind is not CheckpointKind.RAW:
        raise CheckpointError("growth migration accepts RAW checkpoints only")
    return source


def _rewrite_resolved_config(source: bytes, *, planned_updates: int) -> bytes:
    """Rewrite only the governed S0/G1 stage bindings, preserving TOML layout."""

    if type(planned_updates) is not int or planned_updates <= 0:
        raise ValueError("planned_updates must be a positive integer")
    lines = source.decode("utf-8").splitlines()
    section = ""
    replacements: dict[tuple[str, str], str] = {
        ("run", "stage"): 'stage = "G1"',
        ("distributed", "world_size"): "world_size = 2",
        ("growth", "enabled"): "enabled = true",
        ("stage", "name"): 'name = "G1"',
        ("stage", "predecessor"): 'predecessor = "S0"',
        ("stage", "world_size"): "world_size = 2",
        ("stage", "depth"): "depth = 20",
        ("stage", "resolution"): "resolution = 256",
        ("stage", "planned_updates"): f"planned_updates = {planned_updates}",
    }
    seen: set[tuple[str, str]] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        replaced = False
        for (expected_section, key), replacement in replacements.items():
            if section == expected_section and stripped.startswith(f"{key} ="):
                indent = line[: len(line) - len(line.lstrip())]
                result.append(indent + replacement)
                seen.add((expected_section, key))
                replaced = True
                break
        if not replaced:
            result.append(line)
    if seen != set(replacements):
        missing = sorted(set(replacements) - seen)
        raise CheckpointError(f"resolved config is missing governed keys: {missing}")
    return ("\n".join(result) + "\n").encode("utf-8")


def _target_architecture(source_config: dict[str, Any]) -> dict[str, object]:
    if set(source_config) != {
        "architecture",
        "schema_version",
        "kind",
        "identity",
        "prediction_type",
        "out_channels",
    }:
        raise CheckpointError("source model config fields are invalid")
    if source_config.get("schema_version") != 1 or source_config.get("kind") != "raw":
        raise CheckpointError("source model config is not a RAW artifact")
    architecture = copy.deepcopy(_mapping(source_config["architecture"], "architecture"))
    if architecture.get("schema_version") != 3:
        raise CheckpointError("growth migration requires architecture schema v3")
    dit = _mapping(architecture.get("dit"), "DiT architecture")
    if dit.get("depth") != 16 or dit.get("active_slot_ids") != list(active_slot_ids(16)):
        raise CheckpointError("source architecture is not the canonical 16-slot S0")
    dit["depth"] = G1_TARGET_DEPTH
    dit["active_slot_ids"] = list(active_slot_ids(G1_TARGET_DEPTH))
    architecture["dit"] = dit
    return architecture


def _new_block_tensors(architecture: dict[str, object], seed: int) -> dict[str, torch.Tensor]:
    dit = _mapping(architecture["dit"], "target DiT architecture")
    backend = dit.get("attention_backend")
    if backend == "dense_sdpa":
        block_type = DiTBlock
    elif backend == "das_fa2_varlen":
        block_type = PackedDiTBlock
    else:
        raise CheckpointError("target DiT attention backend is unsupported")
    dtype_names = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    block_keys = (
        "hidden_size",
        "intermediate_size",
        "q_heads",
        "kv_heads",
        "head_dim",
        "rope_nope_dim",
        "rope_y_dim",
        "rope_x_dim",
        "rope_position_scale",
        "rope_theta",
        "norm_eps",
        "linear_dtype",
        "projection_bias",
        "attention_dropout",
        "mlp_dropout",
    )
    try:
        kwargs = {
            key: dtype_names.get(value, value)
            for key, value in ((key, dit[key]) for key in block_keys)
        }
    except KeyError:
        raise CheckpointError("target DiT block constructor fields are incomplete") from None
    if type(seed) is not int or seed < 0:
        raise ValueError("migration seed must be a nonnegative integer")
    tensors: dict[str, torch.Tensor] = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for slot_id in G1_NEW_SLOT_IDS:
            block = block_type(**kwargs)  # pyright: ignore[arg-type]
            for name, tensor in block.state_dict().items():
                tensors[f"dit.blocks.{slot_name(slot_id)}.{name}"] = (
                    tensor.detach().cpu().contiguous()
                )
    modulation_size = int(dit["modulation_chunks"]) * int(dit["hidden_size"])
    for slot_id in G1_NEW_SLOT_IDS:
        tensors[f"dit.conditioner.block_biases.{slot_name(slot_id)}"] = torch.zeros(
            modulation_size, dtype=torch.float32
        )
    return tensors


def _target_group_names(architecture: dict[str, object]) -> dict[str, list[str]]:
    try:
        module = build_trainable_composite(architecture, device="meta")
        audit = audit_trainable_parameters(
            module, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
        )
    except (TypeError, ValueError):
        raise CheckpointError("target architecture cannot produce optimizer groups") from None
    return {
        "matrix_decay": [spec.name for spec in audit.decay],
        "sensitive_no_decay": [spec.name for spec in audit.sensitive],
    }


def _migrate_model(
    temporary: Path,
    *,
    identity: CheckpointIdentity,
    migration_seed: int,
) -> tuple[dict[str, object], frozenset[str]]:
    model_dir = temporary / "model"
    config_path = model_dir / "config.json"
    config = _mapping(_read_json(config_path, "source model config"), "source model config")
    architecture = _target_architecture(config)
    new_tensors = _new_block_tensors(architecture, migration_seed)
    index = _mapping(
        _read_json(model_dir / "model.safetensors.index.json", "source model index"),
        "source model index",
    )
    weight_map = _mapping(index.get("weight_map"), "source model weight map")
    total_size = _mapping(index.get("metadata"), "source model metadata").get("total_size")
    if type(total_size) is not int or total_size <= 0:
        raise CheckpointError("source model index total_size is invalid")
    old_names = frozenset(cast(str, name) for name in weight_map)
    target_module = build_trainable_composite(architecture, device="meta")
    target_names = frozenset(target_module.state_dict())
    expected_new = target_names - old_names
    if expected_new != frozenset(new_tensors) or old_names - target_names:
        raise CheckpointError("target model FQN delta is not exactly the approved G1 growth")
    if any(name not in old_names and not (
        name.startswith(
            (
                "dit.blocks.slot_02.",
                "dit.blocks.slot_08.",
                "dit.blocks.slot_14.",
                "dit.blocks.slot_20.",
            )
        )
        or name in {
            "dit.conditioner.block_biases.slot_02",
            "dit.conditioner.block_biases.slot_08",
            "dit.conditioner.block_biases.slot_14",
            "dit.conditioner.block_biases.slot_20",
        }
    ) for name in expected_new):
        raise CheckpointError("unexpected FQN appeared in target model")
    save_file(new_tensors, str(model_dir / GROWTH_SHARD))
    migrated_weight_map = {str(name): str(shard) for name, shard in weight_map.items()}
    migrated_weight_map.update({name: GROWTH_SHARD for name in sorted(new_tensors)})
    config["architecture"] = architecture
    config["identity"] = {"checkpoint_id": identity.checkpoint_id, "update": identity.update}
    _write_json(config_path, config)
    _write_json(
        model_dir / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": total_size
                + sum(tensor.numel() * tensor.element_size() for tensor in new_tensors.values())
            },
            "weight_map": dict(sorted(migrated_weight_map.items())),
        },
    )
    _write_json(
        model_dir / "manifest.json",
        {
            "files": [
                {"path": record.path, "size": record.size}
                for record in _records(model_dir, recursive=False)
            ],
            "schema_version": 1,
        },
    )
    return architecture, old_names


def _migrate_optimizer(
    temporary: Path,
    *,
    architecture: dict[str, object],
) -> None:
    train_state = temporary / "train_state"
    try:
        optimizer = torch.load(
            train_state / "optimizer.pt", map_location="cpu", weights_only=True
        )
    except Exception:  # noqa: BLE001
        raise CheckpointError("source optimizer state is not safe-loadable") from None
    state_document = _mapping(optimizer, "source optimizer state")
    state_by_id = _mapping(state_document.get("state"), "source optimizer state map")
    raw_groups = state_document.get("param_groups")
    if not isinstance(raw_groups, list):
        raise CheckpointError("source optimizer groups are invalid")
    schema = _mapping(
        _read_json(train_state / "optimizer_schema.json", "optimizer schema"),
        "optimizer schema",
    )
    schema_groups = schema.get("groups")
    if schema.get("schema_version") != 1 or not isinstance(schema_groups, list):
        raise CheckpointError("source optimizer schema is invalid")
    by_group: dict[str, dict[str, object]] = {}
    old_id_by_name: dict[str, int] = {}
    for index, raw_group in enumerate(cast(list[object], raw_groups)):
        group = _mapping(raw_group, f"optimizer group {index}")
        name = group.get("group_name")
        names = group.get("param_names")
        ids = group.get("params")
        if (
            not isinstance(name, str)
            or name in by_group
            or not isinstance(names, list)
            or not isinstance(ids, list)
            or len(names) != len(ids)
            or not all(isinstance(value, str) for value in names)
            or not all(type(value) is int for value in ids)
        ):
            raise CheckpointError("source optimizer parameter identities are invalid")
        by_group[name] = group
        for param_name, param_id in zip(cast(list[str], names), cast(list[int], ids), strict=True):
            if param_name in old_id_by_name:
                raise CheckpointError("source optimizer FQN is duplicated")
            old_id_by_name[param_name] = param_id
    if set(by_group) != set(_GROUP_ORDER):
        raise CheckpointError("source optimizer group names are invalid")
    target_groups = _target_group_names(architecture)
    target_names = frozenset(name for names in target_groups.values() for name in names)
    added_names = target_names - frozenset(old_id_by_name)
    if not added_names or any(
        not (
            name.startswith(
                (
                    "dit.blocks.slot_02.",
                    "dit.blocks.slot_08.",
                    "dit.blocks.slot_14.",
                    "dit.blocks.slot_20.",
                )
            )
            or name in {
                "dit.conditioner.block_biases.slot_02",
                "dit.conditioner.block_biases.slot_08",
                "dit.conditioner.block_biases.slot_14",
                "dit.conditioner.block_biases.slot_20",
            }
        )
        for name in added_names
    ):
        raise CheckpointError("target optimizer FQN delta is not approved G1 growth")
    if frozenset(old_id_by_name) - target_names:
        raise CheckpointError("target optimizer lost an existing parameter")
    # Optimizer IDs are runtime-canonical: torch assigns them in target group
    # order, so preserving S0's numeric IDs would fail strict restore after
    # inserting the four new slots into the 20-layer parameter ordering.
    next_id = 0
    migrated_groups: list[dict[str, object]] = []
    migrated_schema_groups: list[dict[str, object]] = []
    migrated_state: dict[int, object] = {}
    for group_name in _GROUP_ORDER:
        source_group = dict(by_group[group_name])
        target_param_names = target_groups[group_name]
        target_ids: list[int] = []
        for name in target_param_names:
            target_id = next_id
            next_id += 1
            target_ids.append(target_id)
            if name in old_id_by_name:
                source_id = old_id_by_name[name]
                if source_id in state_by_id:
                    migrated_state[target_id] = state_by_id[source_id]
        source_group["param_names"] = target_param_names
        source_group["params"] = target_ids
        migrated_groups.append(source_group)
        migrated_schema_groups.append(
            {"group_name": group_name, "param_names": target_param_names}
        )
    _write_json(
        train_state / "optimizer_schema.json",
        {"groups": migrated_schema_groups, "schema_version": 1},
    )
    torch.save(
        {"state": migrated_state, "param_groups": migrated_groups},
        train_state / "optimizer.pt",
    )


def migrate_checkpoint(
    source: Path,
    destination: Path,
    *,
    target_stage: str,
    planned_updates: int,
    migration_seed: int,
    manual_approval: bool,
) -> Path:
    if target_stage != G1_TARGET_STAGE:
        raise ValueError("this migrator only supports target-stage G1")
    if not manual_approval:
        raise PermissionError("G1 migration requires explicit --manual-approval")
    source = _exact_complete_source(source)
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(f"migration destination already exists: {destination}")
    if not destination.is_absolute():
        raise ValueError("migration destination must be absolute")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise CheckpointError("migration destination parent must be a real directory")
    source_manifest, source_state = read_raw_checkpoint_state(source)
    if (
        source_state.growth.stage != "S0"
        or source_state.growth.world_size != G1_TARGET_WORLD_SIZE
        or source_state.growth.resolution != G1_TARGET_RESOLUTION
        or source_state.growth.active_slot_ids != active_slot_ids(16)
        or source_state.growth.alpha != 1.0
    ):
        raise CheckpointError("source checkpoint is not the live canonical S0 topology")
    transition_update = source_state.trainer.successful_updates
    # planned_updates is the absolute successful-update terminal of the target
    # stage; the growth ramp scales with the stage length, not the terminal.
    ramp_updates = growth_ramp_updates(planned_updates - transition_update)
    identity = CheckpointIdentity(
        f"raw-{transition_update}-post-transition", transition_update
    )
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"migration temporary path already exists: {temporary}")
    try:
        shutil.copytree(source, temporary, symlinks=False)
        architecture, _ = _migrate_model(
            temporary,
            identity=identity,
            migration_seed=migration_seed,
        )
        _migrate_optimizer(
            temporary,
            architecture=architecture,
        )
        target_state = RawCheckpointState(
            trainer=source_state.trainer,
            growth=GrowthCheckpointState(
                active_slot_ids=active_slot_ids(G1_TARGET_DEPTH),
                alpha=0.0,
                stage=G1_TARGET_STAGE,
                world_size=G1_TARGET_WORLD_SIZE,
                resolution=G1_TARGET_RESOLUTION,
                ramp_start_successful_update=transition_update,
                ramp_updates=ramp_updates,
            ),
            stage_budget=StageBudgetCheckpointState(
                transition_update, planned_updates
            ),
            checkpoint_cadence=source_state.checkpoint_cadence,
        )
        trainer_document, growth_document = raw_state_to_dict(target_state)
        _write_json(temporary / "train_state" / "trainer_state.json", trainer_document)
        _write_json(temporary / "train_state" / "growth_state.json", growth_document)
        resolved_config = (temporary / "resolved_config.toml").read_bytes()
        (temporary / "resolved_config.toml").write_bytes(
            _rewrite_resolved_config(resolved_config, planned_updates=planned_updates)
        )
        _write_json(
            temporary / GROWTH_MIGRATION_SIDECAR,
            {
                "schema_version": 1,
                "source_checkpoint_id": source_manifest.identity.checkpoint_id,
                "source_update": transition_update,
                "target_stage": G1_TARGET_STAGE,
                "target_depth": G1_TARGET_DEPTH,
                "target_world_size": G1_TARGET_WORLD_SIZE,
                "target_resolution": G1_TARGET_RESOLUTION,
                "planned_updates": planned_updates,
                "ramp_updates": ramp_updates,
                "alpha": 0.0,
                "random_new_slots": True,
                "copy_old_slots": False,
                "migration_seed": migration_seed,
                "strategy": "stable-slot-random-init-v1",
            },
        )
        records = _records(temporary, recursive=True)
        migrated_manifest = CheckpointManifest(
            kind=source_manifest.kind,
            identity=identity,
            files=records,
        )
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest_to_dict(migrated_manifest)))
        (temporary / "COMPLETE").write_bytes(b"complete\n")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    read_raw_checkpoint_state(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-stage", required=True)
    parser.add_argument(
        "--planned-updates",
        type=int,
        required=True,
        help="absolute successful-update terminal of the target stage",
    )
    parser.add_argument("--migration-seed", type=int, required=True)
    parser.add_argument("--manual-approval", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        migrate_checkpoint(
            args.source,
            args.target,
            target_stage=args.target_stage,
            planned_updates=args.planned_updates,
            migration_seed=args.migration_seed,
            manual_approval=args.manual_approval,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["GROWTH_SHARD", "migrate_checkpoint"]
