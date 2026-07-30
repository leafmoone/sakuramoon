from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    GrowthCheckpointState,
    RawCheckpointState,
    manifest_to_dict,
    raw_state_to_dict,
)
from sakuramoon.cli.transition import main, transition_plan, write_transition_plan
from sakuramoon.data.state import ShardRunState
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.stage import StageTransitionRequest
from sakuramoon.train.step import SingleGpuUpdateState


def _checkpoint(
    root: Path,
    kind: CheckpointKind,
    *,
    stage: str = "S1",
    world_size: int = 4,
) -> Path:
    checkpoint = root / "ckpt_42_source"
    checkpoint.mkdir()
    payload = checkpoint / "payload.bin"
    payload.write_bytes(b"x")
    payloads = [payload]
    if kind is CheckpointKind.RAW:
        state = RawCheckpointState(
            trainer=SingleGpuUpdateState(42, 42, 100),
            data=ShardRunState.empty(),
            growth=GrowthCheckpointState(
                active_slot_ids(16),
                1.0,
                stage,
                world_size,
                256,
                None,
                None,
            ),
        )
        train_state = checkpoint / "train_state"
        train_state.mkdir()
        documents = raw_state_to_dict(state)
        for name, document in zip(
            ("trainer_state.json", "data_state.json", "growth_state.json"),
            documents,
            strict=True,
        ):
            state_file = train_state / name
            state_file.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            payloads.append(state_file)
    identity = CheckpointIdentity("source", 42, "a" * 64, "b" * 64, "c" * 64)
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
        next_pass_index=1,
        next_seed=2,
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
        "--next-pass-index",
        "1",
        "--next-seed",
        "2",
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
