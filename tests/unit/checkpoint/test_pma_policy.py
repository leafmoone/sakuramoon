from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import (
    load_file,  # pyright: ignore[reportUnknownVariableType]
    save_file,  # pyright: ignore[reportUnknownVariableType]
)

from sakuramoon.checkpoint import (
    CheckpointCadence,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointReason,
    apply_raw_retention,
    plan_raw_retention,
    save_pma10,
    save_release,
)
from sakuramoon.checkpoint.load import read_checkpoint_manifest
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointManifest,
    FileRecord,
    identity_to_dict,
    manifest_to_dict,
)
from sakuramoon.model.growth import BASE_SLOT_IDS


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(root: Path) -> tuple[FileRecord, ...]:
    return tuple(
        FileRecord(
            path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path)
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"manifest.json", "COMPLETE"}
    )


def _identity(update: int, checkpoint_id: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        checkpoint_id=checkpoint_id,
        update=update,
        config_sha256="a" * 64,
        dependency_sha256="b" * 64,
        parameter_schema_sha256="c" * 64,
    )


def _raw_fixture(root: Path, update: int, value: float) -> Path:
    identity = _identity(update, f"source-{update:02d}")
    checkpoint = root / f"ckpt_{update}_{identity.checkpoint_id}"
    model = checkpoint / "model"
    train_state = checkpoint / "train_state"
    rng = train_state / "rng"
    rng.mkdir(parents=True)
    model.mkdir()
    shard = "model-00001-of-00001.safetensors"
    tensor = torch.full((4,), value, dtype=torch.bfloat16)
    save_file({"weight": tensor}, str(model / shard))
    (model / "model.safetensors.index.json").write_bytes(
        _json_bytes(
            {
                "metadata": {"total_size": tensor.numel() * tensor.element_size()},
                "weight_map": {"weight": shard},
            }
        )
    )
    (model / "config.json").write_bytes(
        _json_bytes(
            {
                "architecture": {"class": "synthetic-test-only"},
                "identity": identity_to_dict(identity),
                "kind": "raw",
                "out_channels": 128,
                "prediction_type": "x",
                "schema_version": 1,
            }
        )
    )
    model_records = _records(model)
    (model / "manifest.json").write_bytes(
        _json_bytes(
            {
                "files": [
                    {"path": item.path, "sha256": item.sha256, "size": item.size}
                    for item in model_records
                ],
                "schema_version": 1,
            }
        )
    )
    (train_state / "trainer_state.json").write_bytes(
        _json_bytes(
            {
                "attempted_updates": update,
                "effective_samples": update,
                "schema_version": 1,
                "successful_updates": update,
            }
        )
    )
    (train_state / "data_state.json").write_bytes(
        _json_bytes(
            {
                "active": None,
                "completed": [],
                "replayed_samples": 0,
                "replayed_shards": 0,
                "schema_version": 1,
            }
        )
    )
    (train_state / "growth_state.json").write_bytes(
        _json_bytes(
            {
                "active_slot_ids": list(BASE_SLOT_IDS),
                "alpha": 1.0,
                "ramp_start_successful_update": None,
                "ramp_updates": None,
                "resolution": 256,
                "schema_version": 1,
                "stage": "S0",
                "world_size": 1,
            }
        )
    )
    (train_state / "optimizer.pt").write_bytes(b"test-only\n")
    (train_state / "optimizer_schema.json").write_bytes(_json_bytes({}))
    (rng / "optimizer_sr.safetensors").write_bytes(b"test-only\n")
    (rng / "rank-0.safetensors").write_bytes(b"test-only\n")
    manifest = CheckpointManifest(CheckpointKind.RAW, identity, _records(checkpoint))
    (checkpoint / "manifest.json").write_bytes(_json_bytes(manifest_to_dict(manifest)))
    (checkpoint / "COMPLETE").write_bytes(b"complete\n")
    return checkpoint


def test_pma10_streams_simple_mean_and_release_is_non_resumable(tmp_path: Path) -> None:
    sources = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 11))

    pma = save_pma10(tmp_path, _identity(10, "pma-window"), sources)
    averaged = load_file(
        pma.path / "model" / "model-00001-of-00001.safetensors", device="cpu"
    )["weight"]

    torch.testing.assert_close(averaged, torch.full_like(averaged, 5.5), atol=0, rtol=0)
    assert read_checkpoint_manifest(pma.path).kind is CheckpointKind.PMA
    assert not (pma.path / "train_state").exists()
    release = save_release(tmp_path, _identity(10, "manual-release"), pma.path)
    assert read_checkpoint_manifest(release.path).kind is CheckpointKind.RELEASE
    release_source = json.loads((release.path / "release_source.json").read_bytes())
    assert release_source["automatic_release"] is False
    assert not (release.path / "train_state").exists()


def test_pma_rejects_wrong_window_order_topology_and_missing_sidecars(
    tmp_path: Path,
) -> None:
    sources = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 11))
    with pytest.raises(ValueError, match="exactly ten"):
        save_pma10(tmp_path, _identity(9, "short"), sources[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        save_pma10(tmp_path, _identity(10, "reordered"), sources[::-1])

    sidecar = sources[0] / "train_state" / "optimizer.pt"
    sidecar.unlink()
    manifest_path = sources[0] / "manifest.json"
    manifest = CheckpointManifest(
        CheckpointKind.RAW, _identity(1, "source-01"), _records(sources[0])
    )
    manifest_path.write_bytes(_json_bytes(manifest_to_dict(manifest)))
    with pytest.raises(CheckpointError, match="sidecars"):
        save_pma10(tmp_path, _identity(10, "missing-sidecar"), sources)


def test_checkpoint_cadence_advances_only_after_matching_commit() -> None:
    cadence = CheckpointCadence(0, 100.0)
    assert cadence.due(successful_update=999, monotonic_seconds=101.0) is None
    assert cadence.due(successful_update=1000, monotonic_seconds=101.0) is (
        CheckpointReason.UPDATE_CADENCE
    )
    with pytest.raises(ValueError, match="reason"):
        cadence.committed(
            successful_update=1000,
            monotonic_seconds=101.0,
            reason=CheckpointReason.WALL_CADENCE,
        )
    cadence = cadence.committed(
        successful_update=1000,
        monotonic_seconds=101.0,
        reason=CheckpointReason.UPDATE_CADENCE,
    )
    assert cadence.due(
        successful_update=1001,
        monotonic_seconds=101.0 + 6.0 * 3600.0,
    ) is CheckpointReason.WALL_CADENCE
    assert cadence.due(
        successful_update=1001,
        monotonic_seconds=102.0,
        forced=CheckpointReason.PRE_DECAY,
    ) is CheckpointReason.PRE_DECAY


def test_raw_retention_keeps_two_rolling_and_every_accepted(tmp_path: Path) -> None:
    paths = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 6))
    plan = plan_raw_retention(
        tmp_path, accepted_checkpoint_ids=frozenset({"source-02"})
    )

    assert plan.keep == (paths[1], paths[3], paths[4])
    assert plan.remove == (paths[0], paths[2])
    apply_raw_retention(tmp_path, plan)
    assert tuple(path.exists() for path in paths) == (False, True, False, True, True)
