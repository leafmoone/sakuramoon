"""Streaming PMA-10 and manual release artifact publication."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import (
    load_file,  # pyright: ignore[reportUnknownVariableType]
    save_file,  # pyright: ignore[reportUnknownVariableType]
)

from sakuramoon.checkpoint.load import (
    read_checkpoint_manifest,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.save import (
    _fsync_directory,  # pyright: ignore[reportPrivateUsage]
    _fsync_file,  # pyright: ignore[reportPrivateUsage]
    _payload_records,  # pyright: ignore[reportPrivateUsage]
    _target_name,  # pyright: ignore[reportPrivateUsage]
    _write_bytes,  # pyright: ignore[reportPrivateUsage]
    _write_json,  # pyright: ignore[reportPrivateUsage]
)
from sakuramoon.checkpoint.schema import (
    MAX_MODEL_SHARD_BYTES,
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    CheckpointSaveResult,
    RawCheckpointState,
    identity_from_dict,
    identity_to_dict,
    manifest_to_dict,
)

PMA_WINDOW = 10
_RAW_SIDECARS = {
    "train_state/data_state.json",
    "train_state/growth_state.json",
    "train_state/optimizer.pt",
    "train_state/optimizer_schema.json",
    "train_state/rng/optimizer_sr.safetensors",
    "train_state/rng/rank-0.safetensors",
    "train_state/trainer_state.json",
}


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    architecture: object
    weight_map: dict[str, str]
    total_size: int


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError(f"{name} is unreadable or invalid JSON") from None
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _model_spec(
    checkpoint: Path,
    identity: CheckpointIdentity,
    kind: CheckpointKind,
) -> _ModelSpec:
    config = _read_json(checkpoint / "model" / "config.json", "model config")
    if set(config) != {
        "architecture",
        "identity",
        "kind",
        "out_channels",
        "prediction_type",
        "schema_version",
    }:
        raise CheckpointError("model config has unknown or missing fields")
    if (
        type(config["schema_version"]) is not int
        or config["schema_version"] != 1
        or config["kind"] != kind.value
        or config["prediction_type"] != "x"
        or type(config["out_channels"]) is not int
        or config["out_channels"] != 128
        or identity_from_dict(config["identity"]) != identity
    ):
        raise CheckpointError("model config is incompatible")
    index = _read_json(
        checkpoint / "model" / "model.safetensors.index.json", "model index"
    )
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointError("model index has unknown or missing fields")
    metadata_value = index["metadata"]
    weight_map_value = index["weight_map"]
    if not isinstance(metadata_value, dict) or not isinstance(weight_map_value, dict):
        raise CheckpointError("model index is invalid")
    metadata = cast(dict[str, object], metadata_value)
    weight_map_values = cast(dict[object, object], weight_map_value)
    if (
        set(metadata) != {"total_size"}
        or type(metadata["total_size"]) is not int
        or metadata["total_size"] <= 0
        or not weight_map_values
        or not all(
            type(name) is str
            and type(filename) is str
            and "/" not in filename
            and filename.endswith(".safetensors")
            for name, filename in weight_map_values.items()
        )
    ):
        raise CheckpointError("model index is invalid")
    weight_map = cast(dict[str, str], weight_map_values)
    return _ModelSpec(
        architecture=config["architecture"],
        weight_map=weight_map,
        total_size=metadata["total_size"],
    )


def _publish(
    destination_root: Path,
    identity: CheckpointIdentity,
    kind: CheckpointKind,
    populate: Callable[[Path], None],
) -> CheckpointSaveResult:
    destination_root.mkdir(parents=True, exist_ok=True)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ValueError("artifact destination must be a real directory")
    target = destination_root / _target_name(kind, identity)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"checkpoint target already exists: {target.name}")
    temporary = destination_root / f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        populate(temporary)
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


def _write_model_metadata(
    model_dir: Path,
    *,
    identity: CheckpointIdentity,
    kind: CheckpointKind,
    spec: _ModelSpec,
) -> None:
    _write_json(
        model_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": spec.total_size}, "weight_map": spec.weight_map},
    )
    _write_json(
        model_dir / "config.json",
        {
            "architecture": spec.architecture,
            "identity": identity_to_dict(identity),
            "kind": kind.value,
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
                {"path": item.path, "sha256": item.sha256, "size": item.size}
                for item in model_records
            ],
            "schema_version": 1,
        },
    )
    _fsync_directory(model_dir)


def save_pma10(
    destination_root: Path,
    identity: CheckpointIdentity,
    sources: tuple[Path, ...],
    *,
    max_shard_bytes: int = MAX_MODEL_SHARD_BYTES,
) -> CheckpointSaveResult:
    """Average exactly ten complete, same-topology raw checkpoints in FP32."""

    if len(sources) != PMA_WINDOW or len({path.resolve() for path in sources}) != PMA_WINDOW:
        raise ValueError("PMA requires exactly ten unique raw checkpoints")
    if type(max_shard_bytes) is not int or not 0 < max_shard_bytes <= MAX_MODEL_SHARD_BYTES:
        raise ValueError("max_shard_bytes must be in (0, 2 GiB]")
    manifests: list[CheckpointManifest] = []
    states: list[RawCheckpointState] = []
    specs: list[_ModelSpec] = []
    for source in sources:
        manifest, state = read_raw_checkpoint_state(source)
        if manifest.kind is not CheckpointKind.RAW:
            raise CheckpointError("PMA inputs must be raw checkpoints")
        sidecars = {
            record.path
            for record in manifest.files
            if record.path.startswith("train_state/")
        }
        if sidecars != _RAW_SIDECARS:
            raise CheckpointError("PMA source raw checkpoint sidecars are incomplete")
        manifests.append(manifest)
        states.append(state)
        specs.append(_model_spec(source, manifest.identity, CheckpointKind.RAW))
    updates = tuple(item.identity.update for item in manifests)
    if updates != tuple(sorted(set(updates))):
        raise ValueError("PMA source updates must be unique and strictly increasing")
    first_identity = manifests[0].identity
    identity_tail = (
        first_identity.config_sha256,
        first_identity.dependency_sha256,
        first_identity.parameter_schema_sha256,
    )
    if any(
        (
            item.identity.config_sha256,
            item.identity.dependency_sha256,
            item.identity.parameter_schema_sha256,
        )
        != identity_tail
        for item in manifests[1:]
    ):
        raise ValueError("PMA source identities differ")
    if (
        identity.update != updates[-1]
        or (
            identity.config_sha256,
            identity.dependency_sha256,
            identity.parameter_schema_sha256,
        )
        != identity_tail
    ):
        raise ValueError("PMA output identity differs from its source window")
    first_state = states[0].growth
    topology = (
        first_state.stage,
        first_state.world_size,
        first_state.resolution,
        first_state.active_slot_ids,
    )
    if first_state.alpha != 1.0 or any(
        state.growth.alpha != 1.0
        or (
            state.growth.stage,
            state.growth.world_size,
            state.growth.resolution,
            state.growth.active_slot_ids,
        )
        != topology
        for state in states[1:]
    ):
        raise ValueError("PMA sources must share one completed stable topology")
    first_spec = specs[0]
    if any(spec != first_spec for spec in specs[1:]):
        raise ValueError("PMA source model architecture or shard layout differs")

    shard_names = sorted(set(first_spec.weight_map.values()))
    names_by_shard = {
        shard: tuple(
            sorted(name for name, filename in first_spec.weight_map.items() if filename == shard)
        )
        for shard in shard_names
    }

    def populate(temporary: Path) -> None:
        model_dir = temporary / "model"
        model_dir.mkdir()
        for shard in shard_names:
            expected_names = names_by_shard[shard]
            sums: dict[str, torch.Tensor] = {}
            source_dtypes: dict[str, torch.dtype] = {}
            for source in sources:
                tensors = load_file(source / "model" / shard, device="cpu")
                if tuple(sorted(tensors)) != expected_names:
                    raise CheckpointError("PMA source shard keys differ from the model index")
                for name in expected_names:
                    tensor = tensors[name]
                    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item()):
                        raise ValueError("PMA source tensors must be finite floating point")
                    if name not in sums:
                        sums[name] = tensor.float()
                        source_dtypes[name] = tensor.dtype
                    else:
                        if tensor.dtype != source_dtypes[name] or tensor.shape != sums[name].shape:
                            raise ValueError("PMA source tensor dtype or shape differs")
                        sums[name].add_(tensor.float())
            averaged = {
                name: (sums[name] / float(PMA_WINDOW)).to(source_dtypes[name])
                for name in expected_names
            }
            if not all(bool(torch.isfinite(tensor).all().item()) for tensor in averaged.values()):
                raise FloatingPointError("PMA average is nonfinite")
            save_file(averaged, str(model_dir / shard))
            _fsync_file(model_dir / shard)
            if (model_dir / shard).stat().st_size > max_shard_bytes:
                raise ValueError("serialized PMA model shard exceeds shard limit")
        _write_model_metadata(
            model_dir,
            identity=identity,
            kind=CheckpointKind.PMA,
            spec=first_spec,
        )
        _write_json(
            temporary / "pma_sources.json",
            {
                "active_slot_ids": list(topology[3]),
                "resolution": topology[2],
                "schema_version": 1,
                "sources": [identity_to_dict(item.identity) for item in manifests],
                "stage": topology[0],
                "window": PMA_WINDOW,
                "world_size": topology[1],
            },
        )

    return _publish(destination_root, identity, CheckpointKind.PMA, populate)


def save_release(
    destination_root: Path,
    identity: CheckpointIdentity,
    pma_checkpoint: Path,
) -> CheckpointSaveResult:
    """Manually promote one validated PMA without creating continuation state."""

    manifest = read_checkpoint_manifest(pma_checkpoint)
    if manifest.kind is not CheckpointKind.PMA:
        raise ValueError("release source must be a PMA artifact")
    source_identity = manifest.identity
    if (
        identity.update != source_identity.update
        or identity.config_sha256 != source_identity.config_sha256
        or identity.dependency_sha256 != source_identity.dependency_sha256
        or identity.parameter_schema_sha256 != source_identity.parameter_schema_sha256
    ):
        raise ValueError("release identity differs from its PMA source")
    spec = _model_spec(pma_checkpoint, source_identity, CheckpointKind.PMA)

    def populate(temporary: Path) -> None:
        model_dir = temporary / "model"
        model_dir.mkdir()
        for shard in sorted(set(spec.weight_map.values())):
            source = pma_checkpoint / "model" / shard
            target = model_dir / shard
            shutil.copyfile(source, target)
            _fsync_file(target)
        _write_model_metadata(
            model_dir,
            identity=identity,
            kind=CheckpointKind.RELEASE,
            spec=spec,
        )
        _write_json(
            temporary / "release_source.json",
            {
                "automatic_release": False,
                "schema_version": 1,
                "source": identity_to_dict(source_identity),
            },
        )

    return _publish(destination_root, identity, CheckpointKind.RELEASE, populate)


__all__ = ["PMA_WINDOW", "save_pma10", "save_release"]
