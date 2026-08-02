from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

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
from sakuramoon.checkpoint.load import (
    read_checkpoint_manifest,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.schema import (
    RAW_SCHEMA_VERSION,
    CheckpointError,
    CheckpointManifest,
    FileRecord,
    identity_from_dict,
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


def _refresh_manifest(
    root: Path, *, identity: CheckpointIdentity | None = None
) -> None:
    existing = json.loads((root / "manifest.json").read_bytes())
    manifest = CheckpointManifest(
        CheckpointKind.RAW,
        identity_from_dict(existing["identity"]) if identity is None else identity,
        _records(root),
    )
    (root / "manifest.json").write_bytes(_json_bytes(manifest_to_dict(manifest)))


def _identity(update: int, checkpoint_id: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        checkpoint_id=checkpoint_id,
        update=update,
        config_sha256=hashlib.sha256(_RESOLVED_CONFIG).hexdigest(),
        dependency_sha256="b" * 64,
        parameter_schema_sha256="c" * 64,
    )


_RESOLVED_CONFIG = b'[run]\nname = "unit"\n'


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
                "checkpoint_cadence": {
                    "every_successful_updates": 1000,
                    "last_successful_update": update,
                    "last_wall_clock_unix_seconds": 1_800_000_000.0 + update,
                },
                "effective_samples": update,
                "schema_version": RAW_SCHEMA_VERSION,
                "stage_budget": {
                    "start_successful_update": 0,
                    "terminal_successful_update": 1000,
                },
                "successful_updates": update,
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
                "schema_version": RAW_SCHEMA_VERSION,
                "stage": "S0",
                "world_size": 1,
            }
        )
    )
    (train_state / "optimizer.pt").write_bytes(b"test-only\n")
    (train_state / "optimizer_schema.json").write_bytes(_json_bytes({}))
    (rng / "optimizer_sr.safetensors").write_bytes(b"test-only\n")
    (rng / "rank-0.safetensors").write_bytes(b"test-only\n")
    (checkpoint / "resolved_config.toml").write_bytes(_RESOLVED_CONFIG)
    manifest = CheckpointManifest(CheckpointKind.RAW, identity, _records(checkpoint))
    (checkpoint / "manifest.json").write_bytes(_json_bytes(manifest_to_dict(manifest)))
    (checkpoint / "COMPLETE").write_bytes(b"complete\n")
    return checkpoint


def test_pma10_streams_simple_mean_and_release_is_non_resumable(tmp_path: Path) -> None:
    sources = tuple(
        _raw_fixture(tmp_path, index, float(index)) for index in range(1, 11)
    )

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
    sources = tuple(
        _raw_fixture(tmp_path, index, float(index)) for index in range(1, 11)
    )
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
    with pytest.raises(CheckpointError, match="sidecars|legacy raw data sidecar"):
        save_pma10(tmp_path, _identity(10, "missing-sidecar"), sources)


@pytest.mark.parametrize("legacy_schema", [1, 2, 3])
def test_legacy_raw_manifest_is_rejected_before_pma_reads_state(
    tmp_path: Path, legacy_schema: int
) -> None:
    source = _raw_fixture(tmp_path, 1, 1.0)
    manifest = json.loads((source / "manifest.json").read_bytes())
    manifest["schema_version"] = legacy_schema
    (source / "manifest.json").write_bytes(_json_bytes(manifest))

    with pytest.raises(CheckpointError, match="legacy raw checkpoint schema"):
        read_raw_checkpoint_state(source)


@pytest.mark.parametrize(
    "extra_path",
    [
        "train_state/data_state.json",
        "opaque.bin",
        "model/data_state.json",
        "model/opaque.bin",
    ],
)
def test_raw_unknown_sidecars_are_rejected(tmp_path: Path, extra_path: str) -> None:
    source = _raw_fixture(tmp_path, 1, 1.0)
    extra = source / extra_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"legacy\n")
    if extra_path.startswith("model/"):
        model = source / "model"
        model_records = tuple(
            record for record in _records(model) if record.path != "manifest.json"
        )
        (model / "manifest.json").write_bytes(
            _json_bytes(
                {
                    "files": [
                        {"path": row.path, "sha256": row.sha256, "size": row.size}
                        for row in model_records
                    ],
                    "schema_version": 1,
                }
            )
        )
    _refresh_manifest(source)

    with pytest.raises(
        CheckpointError,
        match="sidecars|legacy raw data sidecar|model manifest|model payload",
    ):
        read_raw_checkpoint_state(source)


@pytest.mark.parametrize("mutation", ["missing", "invalid", "hash"])
def test_raw_resolved_config_is_fail_closed(tmp_path: Path, mutation: str) -> None:
    source = _raw_fixture(tmp_path, 1, 1.0)
    config = source / "resolved_config.toml"
    if mutation == "missing":
        config.unlink()
    elif mutation == "invalid":
        config.write_bytes(b"[run\n")
        _refresh_manifest(source)
    else:
        config.write_bytes(b'[run]\nname = "different"\n')
        _refresh_manifest(source)

    with pytest.raises(CheckpointError, match="(?:payload|resolved config)"):
        read_raw_checkpoint_state(source)


def test_checkpoint_cadence_advances_only_after_matching_commit() -> None:
    cadence = CheckpointCadence(0, 1_800_000_000.0, 7)
    assert (
        cadence.due(successful_update=6, wall_clock_unix_seconds=1_800_000_001.0)
        is None
    )
    assert cadence.due(
        successful_update=7, wall_clock_unix_seconds=1_800_000_001.0
    ) is (CheckpointReason.UPDATE_CADENCE)
    with pytest.raises(ValueError, match="forced checkpoint"):
        cadence.due(
            successful_update=6,
            wall_clock_unix_seconds=1_800_000_001.0,
            forced=CheckpointReason.WALL_CADENCE,
        )
    with pytest.raises(ValueError, match="reason"):
        cadence.committed(
            successful_update=7,
            wall_clock_unix_seconds=1_800_000_001.0,
            reason=CheckpointReason.WALL_CADENCE,
        )
    cadence = cadence.committed(
        successful_update=7,
        wall_clock_unix_seconds=1_800_000_001.0,
        reason=CheckpointReason.UPDATE_CADENCE,
    )
    assert (
        cadence.due(
            successful_update=8,
            wall_clock_unix_seconds=1_800_000_001.0 + 6.0 * 3600.0,
        )
        is None
    )
    assert (
        cadence.due(
            successful_update=8,
            wall_clock_unix_seconds=1_800_000_002.0,
            forced=CheckpointReason.PRE_DECAY,
        )
        is CheckpointReason.PRE_DECAY
    )


def test_checkpoint_cadence_interval_is_required() -> None:
    with pytest.raises(TypeError):
        CheckpointCadence(0, 0.0)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("interval", [True, 0, -1, 1.0])
def test_checkpoint_cadence_rejects_non_strict_positive_interval(
    interval: object,
) -> None:
    with pytest.raises(ValueError, match="fields are invalid"):
        CheckpointCadence(
            0,
            0.0,
            cast(int, interval),
        )


def test_checkpoint_audit_time_survives_restart_and_rejects_clock_rollback() -> None:
    cadence = CheckpointCadence(20, 1_800_000_000.0, 1000)
    restored = CheckpointCadence(
        cadence.last_successful_update,
        cadence.last_wall_clock_unix_seconds,
        cadence.every_successful_updates,
    )

    assert (
        restored.due(
            successful_update=21,
            wall_clock_unix_seconds=1_800_000_000.0 + 6.0 * 3600.0,
        )
        is None
    )
    with pytest.raises(ValueError, match="cadence input"):
        restored.due(
            successful_update=21,
            wall_clock_unix_seconds=1_799_999_999.0,
        )


def test_raw_retention_keeps_two_rolling_and_every_accepted(tmp_path: Path) -> None:
    paths = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 6))
    plan = plan_raw_retention(
        tmp_path,
        accepted_checkpoint_ids=frozenset({"source-02"}),
        rolling_slots=2,
    )

    assert plan.keep == (paths[1], paths[3], paths[4])
    assert plan.remove == (paths[0], paths[2])
    apply_raw_retention(
        tmp_path,
        plan,
        accepted_checkpoint_ids=frozenset({"source-02"}),
        rolling_slots=2,
    )
    assert tuple(path.exists() for path in paths) == (False, True, False, True, True)


def test_raw_retention_uses_explicit_rolling_slot_count(tmp_path: Path) -> None:
    paths = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 6))

    plan = plan_raw_retention(
        tmp_path,
        accepted_checkpoint_ids=frozenset(),
        rolling_slots=3,
    )

    assert plan.rolling_slots == 3
    assert plan.keep == paths[2:]
    assert plan.remove == paths[:2]


@pytest.mark.parametrize("rolling_slots", [True, 0, -1, 1.0])
def test_raw_retention_rejects_non_strict_slot_count(
    tmp_path: Path, rolling_slots: int
) -> None:
    with pytest.raises(ValueError, match="rolling slots"):
        plan_raw_retention(
            tmp_path,
            accepted_checkpoint_ids=frozenset(),
            rolling_slots=rolling_slots,
        )


def test_raw_retention_rejects_forged_or_stale_plan(tmp_path: Path) -> None:
    paths = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 6))
    accepted = frozenset({"source-02"})
    plan = plan_raw_retention(
        tmp_path, accepted_checkpoint_ids=accepted, rolling_slots=2
    )
    forged = replace(
        plan,
        keep=plan.keep[1:],
        remove=(paths[1],) + plan.remove,
    )

    with pytest.raises(ValueError, match="stale|policy"):
        apply_raw_retention(
            tmp_path, forged, accepted_checkpoint_ids=accepted, rolling_slots=2
        )
    with pytest.raises(ValueError, match="stale|policy"):
        apply_raw_retention(
            tmp_path,
            plan,
            accepted_checkpoint_ids=frozenset(),
            rolling_slots=2,
        )
    with pytest.raises(ValueError, match="stale|policy"):
        apply_raw_retention(
            tmp_path, plan, accepted_checkpoint_ids=accepted, rolling_slots=1
        )
    assert all(path.exists() for path in paths)

    manifest_path = paths[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["identity"]["update"] = 100
    manifest_path.write_bytes(_json_bytes(manifest))
    with pytest.raises(ValueError, match="stale|policy"):
        apply_raw_retention(
            tmp_path, plan, accepted_checkpoint_ids=accepted, rolling_slots=2
        )
    assert all(path.exists() for path in paths)


def test_raw_retention_does_not_rehash_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 6))

    def fail_hash(_path: Path) -> str:
        raise AssertionError("retention must not hash checkpoint payloads")

    monkeypatch.setattr("sakuramoon.checkpoint.load._sha256", fail_hash)
    plan = plan_raw_retention(
        tmp_path, accepted_checkpoint_ids=frozenset(), rolling_slots=2
    )
    apply_raw_retention(
        tmp_path,
        plan,
        accepted_checkpoint_ids=frozenset(),
        rolling_slots=2,
    )

    assert tuple(path.exists() for path in paths) == (False, False, False, True, True)


@pytest.mark.parametrize("mutation", ["payload-size", "extra-file", "symlink", "name"])
def test_raw_retention_rejects_physical_tree_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = tuple(_raw_fixture(tmp_path, index, float(index)) for index in range(1, 6))
    plan = plan_raw_retention(
        tmp_path, accepted_checkpoint_ids=frozenset(), rolling_slots=2
    )

    if mutation == "payload-size":
        payload = paths[0] / "train_state" / "optimizer.pt"
        payload.write_bytes(payload.read_bytes() + b"changed\n")
    elif mutation == "extra-file":
        (paths[0] / "unexpected.bin").write_bytes(b"unexpected\n")
    elif mutation == "symlink":
        (paths[0] / "payload-link").symlink_to("train_state/optimizer.pt")
    else:
        paths[0].rename(tmp_path / "renamed-raw")

    with pytest.raises(ValueError, match="stale|policy"):
        apply_raw_retention(
            tmp_path,
            plan,
            accepted_checkpoint_ids=frozenset(),
            rolling_slots=2,
        )
    assert paths[1].exists()
    assert paths[2].exists()
