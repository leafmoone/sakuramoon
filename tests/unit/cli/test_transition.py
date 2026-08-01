from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
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
from sakuramoon.cli.transition import main, transition_plan, write_transition_plan
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.stage import StageTransitionRequest
from sakuramoon.train.step import SingleGpuUpdateState

_RESOLVED_CONFIG = b"[run]\nname = \"transition-unit\"\n"


def _checkpoint(
    root: Path,
    kind: CheckpointKind,
    *,
    stage: str = "S1",
    world_size: int = 4,
) -> Path:
    checkpoint = root / "ckpt_42_source"
    checkpoint.mkdir()
    identity = CheckpointIdentity(
        "source",
        42,
        hashlib.sha256(_RESOLVED_CONFIG).hexdigest(),
        "b" * 64,
        "c" * 64,
    )
    model = checkpoint / "model"
    model.mkdir()
    payload = model / "model-00001-of-00001.safetensors"
    payload.write_bytes(b"x")
    model_index = model / "model.safetensors.index.json"
    model_index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"test.weight": payload.name},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model_config = model / "config.json"
    model_config.write_text(
        json.dumps(
            {
                "architecture": {"class": "transition-test-only"},
                "identity": {
                    "checkpoint_id": identity.checkpoint_id,
                    "config_sha256": identity.config_sha256,
                    "dependency_sha256": identity.dependency_sha256,
                    "parameter_schema_sha256": identity.parameter_schema_sha256,
                    "update": identity.update,
                },
                "kind": kind.value,
                "out_channels": 128,
                "prediction_type": "x",
                "schema_version": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model_records = tuple(
        FileRecord(
            path.name,
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted((payload, model_config, model_index))
    )
    model_manifest = model / "manifest.json"
    model_manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": row.path, "sha256": row.sha256, "size": row.size}
                    for row in model_records
                ],
                "schema_version": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payloads = [payload, model_config, model_index, model_manifest]
    if kind is CheckpointKind.RAW:
        state = RawCheckpointState(
            trainer=SingleGpuUpdateState(42, 42, 100),
            growth=GrowthCheckpointState(
                active_slot_ids(16),
                1.0,
                stage,
                world_size,
                256,
                None,
                None,
            ),
            stage_budget=StageBudgetCheckpointState(0, 1000),
            checkpoint_cadence=CheckpointCadence(42, 1_800_000_042.0),
        )
        train_state = checkpoint / "train_state"
        train_state.mkdir()
        documents = raw_state_to_dict(state)
        for name, document in zip(
            ("trainer_state.json", "growth_state.json"),
            documents,
            strict=True,
        ):
            state_file = train_state / name
            state_file.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            payloads.append(state_file)
        config = checkpoint / "resolved_config.toml"
        config.write_bytes(_RESOLVED_CONFIG)
        payloads.append(config)
        optimizer = train_state / "optimizer.pt"
        optimizer.write_bytes(b"test-only\n")
        payloads.append(optimizer)
        optimizer_schema = train_state / "optimizer_schema.json"
        optimizer_schema.write_bytes(b"{}\n")
        payloads.append(optimizer_schema)
        rng = train_state / "rng"
        rng.mkdir()
        for name in ("optimizer_sr.safetensors", "rank-0.safetensors"):
            rng_file = rng / name
            rng_file.write_bytes(b"test-only\n")
            payloads.append(rng_file)
    manifest = CheckpointManifest(
        kind=kind,
        identity=identity,
        files=tuple(
            FileRecord(
                path.relative_to(checkpoint).as_posix(),
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(payloads)
        ),
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest_to_dict(manifest), sort_keys=True), encoding="utf-8"
    )
    (checkpoint / "COMPLETE").write_bytes(b"complete\n")
    return checkpoint


def _request(checkpoint: Path) -> StageTransitionRequest:
    return StageTransitionRequest(
        source_stage="S1",
        target_stage="G1",
        source_checkpoint=checkpoint,
        source_checkpoint_id="source",
        planned_updates=50_000,
        manual_approval=True,
    )


def test_transition_plan_is_raw_only_and_records_manual_rollback(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, CheckpointKind.RAW)
    plan = transition_plan(_request(checkpoint))
    assert plan["automatic_retry"] is False
    assert plan["automatic_transition"] is False
    assert plan["rollback_checkpoint"] == str(checkpoint)
    assert plan["ramp_updates"] == 1000
    assert set(plan) == {
        "automatic_retry",
        "automatic_transition",
        "manual_approval",
        "planned_updates",
        "ramp_updates",
        "rollback_checkpoint",
        "schema_version",
        "source",
        "target",
    }
    serialized = json.dumps(plan, sort_keys=True)
    assert not any(
        forbidden in serialized
        for forbidden in (
            "mainset",
            "next_pass",
            "next_seed",
            "tar_cursor",
            "tar_order",
        )
    )


def test_transition_plan_rejects_nonraw_artifact(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, CheckpointKind.MODEL_ONLY)
    with pytest.raises(CheckpointError, match="raw checkpoints only"):
        transition_plan(_request(checkpoint))


def test_transition_plan_rejects_caller_relabelled_checkpoint_stage(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(
        tmp_path, CheckpointKind.RAW, stage="S0", world_size=1
    )
    with pytest.raises(ValueError, match="axes do not match"):
        transition_plan(_request(checkpoint))


def test_cli_requires_manual_approval_and_never_overwrites(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, CheckpointKind.RAW)
    output = tmp_path / "plan.json"
    arguments = [
        "--source-stage",
        "S1",
        "--target-stage",
        "G1",
        "--source-checkpoint",
        str(checkpoint),
        "--source-checkpoint-id",
        "source",
        "--planned-updates",
        "50000",
        "--manual-approval",
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert json.loads(output.read_bytes())["manual_approval"] is True
    with pytest.raises(FileExistsError, match="already exists"):
        main(arguments)


def test_transition_plan_publication_is_atomic_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakuramoon.cli.transition as transition_module

    output = tmp_path / "plan.json"
    barrier = threading.Barrier(2)
    real_link = os.link

    def synchronized_link(
        source: os.PathLike[str],
        target: os.PathLike[str],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        barrier.wait()
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(transition_module.os, "link", synchronized_link)
    outcomes: list[str] = []

    def writer(value: int) -> None:
        try:
            write_transition_plan(output, {"writer": value})
            outcomes.append(f"success:{value}")
        except FileExistsError:
            outcomes.append(f"exists:{value}")

    threads = [threading.Thread(target=writer, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcome.split(":")[0] for outcome in outcomes) == [
        "exists",
        "success",
    ]
    winner = int(next(item.split(":")[1] for item in outcomes if item.startswith("success")))
    assert json.loads(output.read_bytes()) == {"writer": winner}
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_parent_fsync_failure_does_not_unlink_another_writers_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sakuramoon.cli.transition as transition_module

    output = tmp_path / "plan.json"
    real_fsync = os.fsync
    calls = 0

    def fail_after_replacement(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            output.unlink()
            output.write_text('{"writer":"other"}\n', encoding="utf-8")
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(transition_module.os, "fsync", fail_after_replacement)
    with pytest.raises(OSError, match="parent fsync"):
        write_transition_plan(output, {"writer": "first"})

    assert json.loads(output.read_bytes()) == {"writer": "other"}
    assert not tuple(tmp_path.glob(".*.tmp"))
