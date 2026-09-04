"""Explicitly migrate one complete no-iREPA (v3) RAW checkpoint to iREPA (v4).

The normal loader remains fail-closed for an iREPA-enabled configuration
holding a v3 checkpoint.  Migration copies the source checkpoint, adds the
deterministic fresh iREPA projector in a new safetensors shard, remaps the
hybrid-optimizer AdamW parameter IDs by canonical FQN (the projector is the
last canonical FQN in each AdamW group, so every pre-existing parameter keeps
its exact numeric ID and state), records the persistent iREPA schedule anchor
(the first iREPA attempted update binds lambda exactly to zero), and atomically
publishes a distinct destination directory.

The source checkpoint is never modified.  The destination must be empty.  A
failure at any point leaves neither a source nor a destination artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from sakuramoon.checkpoint.artifact import build_trainable_composite
from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    identity_from_dict,
    manifest_to_dict,
)
from sakuramoon.config.schema import IRepaConfig
from sakuramoon.model.irepa import (
    IRepaAlignment,
    irepa_alignment_metadata,
)
from sakuramoon.optim.cmuon import route_cmuon_parameters

SOURCE_ARCHITECTURE_SCHEMA_VERSION = 3
TARGET_ARCHITECTURE_SCHEMA_VERSION = 4
PROJECTOR_SHARD = "model-irepa-projector.safetensors"
IREPA_STATE_FILE = "train_state/irepa_state.json"
IREPA_STATE_SCHEMA_VERSION = 1
IREPA_ALIGNMENT_PREFIX = "irepa_alignment."
PROJECTOR_WEIGHT_FQN = "irepa_alignment.projector.weight"
PROJECTOR_BIAS_FQN = "irepa_alignment.projector.bias"
_GROUP_ORDER = ("matrix_decay", "sensitive_no_decay")
_MODEL_CONFIG_KEYS = {
    "architecture",
    "schema_version",
    "kind",
    "identity",
    "prediction_type",
    "out_channels",
}


@dataclass(frozen=True)
class IRepaMigrationPlan:
    """The read-only outcome of a --dry-run migration."""

    source: Path
    destination: Path
    source_update: int
    anchor: int
    projector_in_channels: int
    migration_seed: int
    added_fqns: tuple[str, ...]
    added_optimizer_parameters: int
    file_count: int


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


def _exact_complete_source(source: Path) -> Path:
    resolved = source.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise CheckpointError("source must be a real directory")
    marker = resolved / "COMPLETE"
    try:
        if marker.is_symlink() or marker.read_bytes() != b"complete\n":
            raise CheckpointError("source checkpoint is incomplete")
    except OSError:
        raise CheckpointError("source checkpoint is incomplete") from None
    return resolved


def _validate_destination(destination: Path) -> Path:
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(f"migration destination already exists: {destination}")
    if not destination.is_absolute():
        raise ValueError("migration destination must be absolute")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise CheckpointError("migration destination parent must be a real directory")
    return destination


def _routing_names(routing: dict[str, object], key: str) -> frozenset[str]:
    """The FQN set of one side of a routing manifest (fail-closed).

    A well-formed manifest side is a list of objects, each carrying a
    non-empty ``name``; anything else is a corrupted checkpoint and raises.
    """
    specs = routing.get(key)
    if not isinstance(specs, list):
        raise CheckpointError(f"routing manifest {key} is not a list")
    names: set[str] = set()
    for raw_spec in cast("list[object]", specs):
        if not isinstance(raw_spec, dict):
            raise CheckpointError(f"routing manifest {key} spec is not an object")
        name = cast("dict[str, object]", raw_spec).get("name")
        if not isinstance(name, str) or not name:
            raise CheckpointError(f"routing manifest {key} spec is missing a name")
        names.add(name)
    return frozenset(names)


def _deterministic_projector(
    in_channels: int, migration_seed: int
) -> IRepaAlignment:
    """Build the canonical projector with an isolated, seed-bound CPU RNG.

    The migration seed must not pollute any global RNG: the CPU torch RNG is
    saved and restored around construction, so repeated migrations with the
    same seed are deterministic while the caller's RNG state is untouched.
    """

    if type(migration_seed) is not int or migration_seed < 0:
        raise ValueError("migration_seed must be a nonnegative integer")
    saved = torch.get_rng_state()
    try:
        torch.manual_seed(migration_seed)  # pyright: ignore[reportUnknownMemberType]
        alignment = IRepaAlignment(in_channels)
    finally:
        torch.set_rng_state(saved)
    return alignment


def _projector_tensors(alignment: IRepaAlignment) -> dict[str, torch.Tensor]:
    weight = alignment.projector.weight.detach().cpu().clone()
    if weight.dtype is not torch.bfloat16 or tuple(weight.shape) != (
        768,
        alignment.projector.in_channels,
        3,
        3,
    ):
        raise CheckpointError("projector weight shape or dtype is not locked")
    bias = alignment.projector.bias
    if bias is None:
        raise CheckpointError("projector bias is missing")
    bias = bias.detach().cpu().clone()
    if bias.dtype is not torch.float32 or tuple(bias.shape) != (768,):
        raise CheckpointError("projector bias shape or dtype is not locked")
    return {
        PROJECTOR_WEIGHT_FQN: weight,
        PROJECTOR_BIAS_FQN: bias,
    }


def _migrate_architecture_v4(document: object) -> dict[str, object]:
    architecture = _mapping(document, "source model architecture")
    if (
        set(architecture)
        != {"schema_version", "class", "dit", "text", "condition_tokens"}
        or architecture.get("schema_version") != SOURCE_ARCHITECTURE_SCHEMA_VERSION
        or architecture.get("class") != "TrainableComposite"
    ):
        raise CheckpointError(
            "source architecture is not a no-iREPA canonical schema v3 document"
        )
    dit = _mapping(architecture.get("dit"), "source DiT architecture")
    in_channels = dit.get("hidden_size")
    if type(in_channels) is not int or in_channels <= 0:
        raise CheckpointError("source DiT hidden_size is invalid")
    migrated = dict(architecture)
    migrated["schema_version"] = TARGET_ARCHITECTURE_SCHEMA_VERSION
    migrated["training_auxiliaries"] = {
        "irepa": irepa_alignment_metadata(in_channels)
    }
    return cast(dict[str, object], migrated)


def _migrate_model_directory(
    model_dir: Path,
    *,
    manifest: CheckpointManifest,
    migration_seed: int,
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
    architecture = _migrate_architecture_v4(config["architecture"])
    dit = _mapping(architecture.get("dit"), "target DiT architecture")
    in_channels = cast(int, dit.get("hidden_size"))
    projector = _deterministic_projector(in_channels, migration_seed)
    new_tensors = _projector_tensors(projector)

    index_path = model_dir / "model.safetensors.index.json"
    index = _mapping(_read_json(index_path, "source model index"), "source model index")
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointError("source model index fields are invalid")
    metadata = _mapping(index["metadata"], "source model index metadata")
    weight_map = _mapping(index["weight_map"], "source model weight map")
    total_size = metadata.get("total_size")
    old_names = frozenset(weight_map)
    if (
        set(metadata) != {"total_size"}
        or type(total_size) is not int
        or total_size <= 0
        or not weight_map
        or not all(isinstance(shard, str) for shard in weight_map.values())
        or set(weight_map) & frozenset(new_tensors)
        or PROJECTOR_SHARD in set(weight_map.values())
        or (model_dir / PROJECTOR_SHARD).exists()
    ):
        raise CheckpointError("source model index is incompatible")

    target_module = build_trainable_composite(architecture, device="meta")
    target_names = frozenset(target_module.state_dict())
    if old_names - target_names:
        raise CheckpointError("source model lost a parameter in the target")
    if target_names - old_names != frozenset(new_tensors):
        raise CheckpointError(
            "target model FQN delta is not exactly the iREPA projector"
        )

    save_file(new_tensors, str(model_dir / PROJECTOR_SHARD))
    migrated_weight_map = cast(dict[str, str], dict(weight_map))
    migrated_weight_map.update(
        {name: PROJECTOR_SHARD for name in sorted(new_tensors)}
    )
    added_size = sum(
        tensor.numel() * tensor.element_size() for tensor in new_tensors.values()
    )
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


def _target_adamw_group_names(architecture: object) -> dict[str, list[str]]:
    """The canonical AdamW (non-CMuon) group FQNs of the target composite.

    A hybrid checkpoint's AdamW optimizer holds ONLY the parameters the CMuon
    allowlist did not claim; the iREPA projector is allowlist-free, so it
    joins the AdamW groups (as the last canonical FQN of each group).  Using
    the full decay/sensitive audit (every parameter) would miscount every
    CMuon parameter as a migration delta and reject real production
    checkpoints.
    """

    try:
        module = build_trainable_composite(architecture, device="meta")
        routing = route_cmuon_parameters(
            module,
            matrix_weight_decay=0.0,
            sensitive_weight_decay=0.0,
        )
    except (TypeError, ValueError):
        raise CheckpointError("target architecture cannot produce canonical groups") from None
    return {
        "matrix_decay": [
            spec.name for spec in routing.adamw_specs if spec.group == "matrix_decay"
        ],
        "sensitive_no_decay": [
            spec.name
            for spec in routing.adamw_specs
            if spec.group == "sensitive_no_decay"
        ],
    }


def _migrate_irepa_optimizer(
    checkpoint: Path,
    *,
    architecture: dict[str, object],
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
) -> None:
    train_state = checkpoint / "train_state"
    optimizer_path = train_state / "optimizer.pt"
    schema_path = train_state / "optimizer_schema.json"
    try:
        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - backend normalization boundary
        raise CheckpointError("source optimizer state is not safe-loadable") from None
    hybrid = _mapping(optimizer, "source hybrid optimizer state")
    if (
        hybrid.get("hybrid_cmuon_schema_version") != 1
        or not isinstance(hybrid.get("cmuon"), dict)
        or not isinstance(hybrid.get("routing"), dict)
        or not isinstance(hybrid.get("transition"), (dict, type(None)))
        or not isinstance(hybrid.get("sr_rng"), dict)
    ):
        raise CheckpointError("source optimizer is not a schema-v1 hybrid CMuon state")
    inner = _mapping(hybrid.get("optimizer"), "source hybrid AdamW state")
    if set(inner) != {"state", "param_groups"}:
        raise CheckpointError("source hybrid AdamW state fields are invalid")
    # torch.load restores the optimizer state with INT parameter ids as keys
    # (the JSON round-trip that would stringify them never happens for the
    # .pt payload), so the state map is int-keyed by construction.
    state_by_id = cast("dict[int, dict[str, object]]", inner.get("state"))
    raw_groups = inner.get("param_groups")
    if not isinstance(raw_groups, list):
        raise CheckpointError("source AdamW groups are invalid")
    raw_groups = cast("list[object]", raw_groups)

    schema_document = _mapping(
        _read_json(schema_path, "source optimizer schema"), "source optimizer schema"
    )
    # The production save path writes a schema-v2 document for hybrid
    # optimizers (the AdamW `groups` plus the `hybrid_cmuon` algorithm block
    # and, for the guarded canonical candidate, a `guarded_canonical` block).
    # Schema v1 (groups only) is the legacy pure-AdamW form.  The migration
    # consumes only `groups`; the algorithm blocks are preserved verbatim so
    # the migrated checkpoint stays loadable.
    if not isinstance(schema_document.get("groups"), list):
        raise CheckpointError("source optimizer schema is invalid")
    schema_version = schema_document.get("schema_version")
    if schema_version == 1:
        if set(schema_document) != {"groups", "schema_version"}:
            raise CheckpointError("source optimizer schema is invalid")
    elif schema_version == 2:
        expected_keys = {"schema_version", "groups", "hybrid_cmuon"}
        if "guarded_canonical" in schema_document:
            expected_keys.add("guarded_canonical")
        if (
            set(schema_document) != expected_keys
            or not isinstance(schema_document.get("hybrid_cmuon"), dict)
            or (
                "guarded_canonical" in schema_document
                and not isinstance(schema_document.get("guarded_canonical"), dict)
            )
        ):
            raise CheckpointError("source optimizer schema is invalid")
    else:
        raise CheckpointError("source optimizer schema is invalid")
    schema_groups = cast(list[object], schema_document["groups"])

    state_group_by_name: dict[str, dict[str, object]] = {}
    old_id_by_name: dict[str, int] = {}
    observed_ids: set[int] = set()
    for index, raw_group in enumerate(raw_groups):
        group = _mapping(raw_group, f"AdamW group {index}")
        name = group.get("group_name")
        raw_names = group.get("param_names")
        raw_ids = group.get("params")
        if (
            not isinstance(name, str)
            or name in state_group_by_name
            or not isinstance(raw_names, list)
            or not isinstance(raw_ids, list)
        ):
            raise CheckpointError(f"AdamW group {index} parameter identity is invalid")
        # The elements are validated at runtime below (fail-closed); the
        # casts declare the validated shape for the typed uses that follow.
        names: list[str] = cast(list[str], group.get("param_names"))
        ids: list[int] = cast(list[int], group.get("params"))
        if (
            len(names) != len(ids)
            or not all(
                isinstance(item, str)  # pyright: ignore[reportUnnecessaryIsInstance]
                for item in names
            )
            or not all(type(item) is int for item in ids)
        ):
            raise CheckpointError(f"AdamW group {index} parameter identity is invalid")
        state_group_by_name[name] = group
        for param_name, param_id in zip(names, ids, strict=True):
            if param_name in old_id_by_name or param_id in observed_ids:
                raise CheckpointError("source AdamW parameter identity is duplicated")
            old_id_by_name[param_name] = param_id
            observed_ids.add(param_id)
    if set(state_group_by_name) != set(_GROUP_ORDER):
        raise CheckpointError("source AdamW group names are invalid")
    if len(schema_groups) != len(raw_groups):
        raise CheckpointError("source AdamW state and schema group counts differ")
    for index, (raw_group, raw_schema_group) in enumerate(
        zip(raw_groups, schema_groups, strict=True)
    ):
        group = _mapping(raw_group, f"AdamW state group {index}")
        schema_group = _mapping(raw_schema_group, f"AdamW schema group {index}")
        if (
            set(schema_group) != {"group_name", "param_names"}
            or schema_group.get("group_name") != group.get("group_name")
            or schema_group.get("param_names") != group.get("param_names")
        ):
            raise CheckpointError("source AdamW state and schema groups differ")
    if not set(state_by_id) <= observed_ids:
        raise CheckpointError("source AdamW state references an unknown parameter")

    target_groups = _target_adamw_group_names(architecture)
    target_names = frozenset(name for names in target_groups.values() for name in names)
    old_names = frozenset(old_id_by_name)
    added_names = target_names - old_names
    if added_names != frozenset({PROJECTOR_WEIGHT_FQN, PROJECTOR_BIAS_FQN}):
        raise CheckpointError("target optimizer FQN delta is not exactly the projector")
    if old_names - target_names:
        raise CheckpointError("target optimizer lost an existing parameter")

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

    routing = _mapping(hybrid.get("routing"), "source routing manifest")
    if set(routing) != {"cmuon", "adamw", "counts"}:
        raise CheckpointError("source routing manifest fields are invalid")
    try:
        module = build_trainable_composite(architecture, device="meta")
        routing_after = route_cmuon_parameters(
            module,
            matrix_weight_decay=matrix_weight_decay,
            sensitive_weight_decay=sensitive_weight_decay,
        )
    except (TypeError, ValueError):
        raise CheckpointError("target architecture cannot produce a routing") from None
    manifest_after = routing_after.routing_manifest()
    source_cmuon_names = _routing_names(routing, "cmuon")
    target_cmuon_names = _routing_names(manifest_after, "cmuon")
    # The iREPA projector is not in the CMuon allowlist, so migration must
    # leave the CMuon set untouched: source and target CMuon FQN sets are
    # equal (the projector therefore appears in neither, by construction).
    if source_cmuon_names != target_cmuon_names or (
        source_cmuon_names & added_names
    ):
        raise CheckpointError("source CMuon routing is not the unchanged pre-iREPA set")
    if frozenset(old_id_by_name) != _routing_names(routing, "adamw"):
        raise CheckpointError("source routing AdamW set differs from optimizer state")
    migrated_routing = manifest_after

    migrated_hybrid = {
        **hybrid,
        "optimizer": {"state": migrated_state, "param_groups": migrated_groups},
        "routing": migrated_routing,
    }
    torch.save(migrated_hybrid, optimizer_path)
    # Preserve the source schema version and the CMuon algorithm blocks
    # (hybrid_cmuon / guarded_canonical) verbatim; only the AdamW `groups`
    # list changes (the projector joins the canonical groups).  Rewriting a
    # v2 document as v1 would make the migrated checkpoint unloadable.
    migrated_schema = {**schema_document, "groups": migrated_schema_groups}
    _replace_bytes(schema_path, _json_bytes(migrated_schema))


def _irepa_state_document(
    *,
    source: CheckpointManifest,
    successful_updates: int,
    migration_seed: int,
) -> dict[str, object]:
    if type(successful_updates) is not int or successful_updates < 0:
        raise CheckpointError("source successful updates are invalid")
    return {
        "schema_version": IREPA_STATE_SCHEMA_VERSION,
        "start_successful_update": successful_updates + 1,
        "source_checkpoint_id": source.identity.checkpoint_id,
        "source_update": successful_updates,
        "migration_seed": migration_seed,
    }


def validate_irepa_state_document(document: object) -> dict[str, object]:
    """Validate a parsed iREPA state sidecar document (fail-closed).

    The sidecar carries the production schedule anchor: ``start_successful_update``
    is the first iREPA attempted update (source ``successful_updates + 1``), so
    the first iREPA update binds lambda exactly to zero.  Both the production
    readiness gate and the resume lambda binding consume this.
    """

    document = _mapping(document, "irepa state")
    expected = {
        "schema_version",
        "start_successful_update",
        "source_checkpoint_id",
        "source_update",
        "migration_seed",
    }
    if set(document) != expected:
        raise CheckpointError("iREPA state has unknown or missing fields")
    if document.get("schema_version") != IREPA_STATE_SCHEMA_VERSION:
        raise CheckpointError("iREPA state schema version differs")
    anchor = document.get("start_successful_update")
    if type(anchor) is not int or anchor < 1:
        raise CheckpointError("iREPA state schedule anchor is invalid")
    source_update = document.get("source_update")
    if type(source_update) is not int or source_update < 0:
        raise CheckpointError("iREPA state source update is invalid")
    if anchor != source_update + 1:
        raise CheckpointError("iREPA state anchor differs from source update + 1")
    seed = document.get("migration_seed")
    if type(seed) is not int or seed < 0:
        raise CheckpointError("iREPA state migration seed is invalid")
    if not isinstance(document.get("source_checkpoint_id"), str) or not cast(
        str, document.get("source_checkpoint_id")
    ):
        raise CheckpointError("iREPA state source checkpoint id is invalid")
    return document


def read_irepa_state(checkpoint: Path) -> dict[str, object]:
    """Read and validate the persistent iREPA state sidecar (fail-closed)."""

    path = checkpoint / IREPA_STATE_FILE
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError("iREPA state sidecar is missing or unreadable") from None
    return validate_irepa_state_document(raw)


def irepa_state_anchor(checkpoint: Path) -> int:
    """The persisted first-iREPA-update anchor for one migrated checkpoint."""

    return cast(int, read_irepa_state(checkpoint)["start_successful_update"])


def _validate_source_no_irepa(checkpoint: Path) -> None:
    model_dir = checkpoint / "model"
    index = _mapping(_read_json(model_dir / "model.safetensors.index.json", "model index"), "model index")
    weight_map = _mapping(index.get("weight_map"), "model weight map")
    if IREPA_ALIGNMENT_PREFIX in "".join(cast(list[str], weight_map)):
        raise CheckpointError("source checkpoint already contains iREPA parameters")
    if PROJECTOR_SHARD in set(weight_map.values()) or (model_dir / PROJECTOR_SHARD).exists():
        raise CheckpointError("source checkpoint already contains an iREPA projector shard")
    if (checkpoint / IREPA_STATE_FILE).exists():
        raise CheckpointError("source checkpoint already contains an iREPA state sidecar")
    config = _mapping(_read_json(model_dir / "config.json", "model config"), "model config")
    architecture = config.get("architecture")
    if not isinstance(architecture, dict) or "training_auxiliaries" in architecture:
        raise CheckpointError("source checkpoint is not a no-iREPA v3 checkpoint")


def migrate_irepa_checkpoint(
    source: Path,
    destination: Path,
    *,
    irepa: IRepaConfig,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
    migration_seed: int,
    dry_run: bool = False,
) -> Path | IRepaMigrationPlan:
    """Migrate one complete no-iREPA RAW checkpoint to iREPA (see module docstring)."""

    if not irepa.enabled:
        raise ValueError("iREPA migration requires an enabled iREPA configuration")
    source = _exact_complete_source(source)
    destination = destination.resolve(strict=False)
    manifest, state = read_raw_checkpoint_state(source)
    if manifest.kind is not CheckpointKind.RAW:
        raise CheckpointError("iREPA migration accepts RAW checkpoints only")
    _validate_source_no_irepa(source)
    if state.trainer.successful_updates != manifest.identity.update:
        raise CheckpointError(
            "checkpoint update differs from trainer successful updates"
        )

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    if dry_run:
        return _dry_run_plan(
            source=source,
            destination=destination,
            manifest=manifest,
            successful_updates=state.trainer.successful_updates,
            migration_seed=migration_seed,
        )
    _validate_destination(destination)
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"migration temporary path already exists: {temporary}")
    try:
        shutil.copytree(source, temporary, symlinks=False)
        architecture = _migrate_model_directory(
            temporary / "model",
            manifest=manifest,
            migration_seed=migration_seed,
        )
        _migrate_irepa_optimizer(
            temporary,
            architecture=architecture,
            matrix_weight_decay=matrix_weight_decay,
            sensitive_weight_decay=sensitive_weight_decay,
        )
        irepa_state = _irepa_state_document(
            source=manifest,
            successful_updates=state.trainer.successful_updates,
            migration_seed=migration_seed,
        )
        # A NEW sidecar (absent from every v3 source): write it directly with
        # fsync, not _replace_bytes, which is for in-place source files.
        irepa_state_path = temporary / "train_state" / "irepa_state.json"
        irepa_state_path.write_bytes(_json_bytes(irepa_state))
        with irepa_state_path.open("rb") as handle:
            os.fsync(handle.fileno())
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
    read_raw_checkpoint_state(destination)
    return destination


def _dry_run_plan(
    *,
    source: Path,
    destination: Path,
    manifest: CheckpointManifest,
    successful_updates: int,
    migration_seed: int,
) -> IRepaMigrationPlan:
    model_dir = source / "model"
    index = _mapping(_read_json(model_dir / "model.safetensors.index.json", "model index"), "model index")
    weight_map = _mapping(index.get("weight_map"), "model weight map")
    old_names = frozenset(weight_map)
    config = _mapping(_read_json(model_dir / "config.json", "model config"), "model config")
    architecture = _migrate_architecture_v4(config["architecture"])
    dit = _mapping(architecture.get("dit"), "target DiT architecture")
    in_channels = cast(int, dit.get("hidden_size"))
    target_module = build_trainable_composite(architecture, device="meta")
    target_names = frozenset(target_module.state_dict())
    added = tuple(sorted(target_names - old_names))
    if set(added) != {PROJECTOR_WEIGHT_FQN, PROJECTOR_BIAS_FQN}:
        raise CheckpointError("dry-run FQN delta is not exactly the iREPA projector")
    return IRepaMigrationPlan(
        source=source,
        destination=destination,
        source_update=successful_updates,
        anchor=successful_updates + 1,
        projector_in_channels=in_channels,
        migration_seed=migration_seed,
        added_fqns=added,
        added_optimizer_parameters=len(added),
        file_count=len(_records(source, recursive=True)) + 1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--migration-seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    import tomllib

    from sakuramoon.config.schema import RuntimeConfig

    args = _parser().parse_args()
    try:
        payload = tomllib.loads(args.config.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SystemExit(f"cannot read migration config: {exc}") from None
    try:
        config = RuntimeConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise SystemExit(f"migration config is invalid: {exc}") from None
    irepa = config.irepa
    if irepa is None:
        raise SystemExit("the migration config must contain an enabled [irepa] table")
    optimizer = config.optimizer
    result = migrate_irepa_checkpoint(
        args.source_checkpoint,
        args.destination,
        irepa=irepa,
        matrix_weight_decay=optimizer.matrix_weight_decay,
        sensitive_weight_decay=optimizer.sensitive_weight_decay,
        migration_seed=args.migration_seed,
        dry_run=args.dry_run,
    )
    if isinstance(result, IRepaMigrationPlan):
        print("DRY RUN: no files written")
        print(f"  source:            {result.source}")
        print(f"  destination:       {result.destination}")
        print(f"  source update:     {result.source_update}")
        print(f"  irepa anchor:      {result.anchor}")
        print(f"  projector in_ch:   {result.projector_in_channels}")
        print(f"  migration seed:    {result.migration_seed}")
        for fqn in result.added_fqns:
            print(f"  + {fqn}")
        print(f"  new optimizer params: {result.added_optimizer_parameters}")
    else:
        print(result)


if __name__ == "__main__":
    main()


__all__ = [
    "IREPA_STATE_FILE",
    "IREPA_STATE_SCHEMA_VERSION",
    "PROJECTOR_SHARD",
    "SOURCE_ARCHITECTURE_SCHEMA_VERSION",
    "TARGET_ARCHITECTURE_SCHEMA_VERSION",
    "IRepaMigrationPlan",
    "migrate_irepa_checkpoint",
]
