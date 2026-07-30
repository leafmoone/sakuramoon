"""Validate and record one explicitly approved stage transition."""

from __future__ import annotations

import argparse
import json
import os
import stat
import uuid
from collections.abc import Sequence
from pathlib import Path

from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.schema import CheckpointError
from sakuramoon.train.stage import (
    StageTransitionRequest,
    stage_spec,
    validate_checkpoint_stage,
)


def transition_plan(request: StageTransitionRequest) -> dict[str, object]:
    manifest, state = read_raw_checkpoint_state(request.source_checkpoint)
    if manifest.identity.checkpoint_id != request.source_checkpoint_id:
        raise CheckpointError("source checkpoint ID differs from the transition request")
    source = stage_spec(request.source_stage)
    target = stage_spec(request.target_stage)
    validate_checkpoint_stage(state, source)
    return {
        "automatic_retry": False,
        "automatic_transition": False,
        "manual_approval": True,
        "next_pass_index": request.next_pass_index,
        "next_seed": request.next_seed,
        "planned_updates": request.planned_updates,
        "ramp_updates": request.ramp_updates,
        "rollback_checkpoint": str(request.source_checkpoint),
        "schema_version": 1,
        "source": {
            "checkpoint_id": manifest.identity.checkpoint_id,
            "depth": source.depth,
            "resolution": source.resolution,
            "stage": source.name,
            "update": manifest.identity.update,
            "world_size": source.world_size,
        },
        "target": {
            "depth": target.depth,
            "resolution": target.resolution,
            "stage": target.name,
            "world_size": target.world_size,
        },
    }


def write_transition_plan(path: Path, plan: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    body = (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            published = False
            try:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
                try:
                    os.link(temporary, path, follow_symlinks=False)
                except FileExistsError:
                    raise FileExistsError("transition plan already exists") from None
                published = True
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise RuntimeError("published transition plan is not a regular file")
                descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                    temporary.unlink()
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except BaseException:
                if published:
                    owned = os.fstat(handle.fileno())
                    try:
                        current = path.stat(follow_symlinks=False)
                        if (current.st_dev, current.st_ino) == (
                            owned.st_dev,
                            owned.st_ino,
                        ):
                            path.unlink()
                    except FileNotFoundError:
                        pass
                temporary.unlink(missing_ok=True)
                raise
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", required=True)
    parser.add_argument("--target-stage", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-id", required=True)
    parser.add_argument("--next-pass-index", type=int, required=True)
    parser.add_argument("--next-seed", type=int, required=True)
    parser.add_argument("--planned-updates", type=int, required=True)
    parser.add_argument("--manual-approval", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = StageTransitionRequest(
        source_stage=args.source_stage,
        target_stage=args.target_stage,
        source_checkpoint=args.source_checkpoint,
        source_checkpoint_id=args.source_checkpoint_id,
        next_pass_index=args.next_pass_index,
        next_seed=args.next_seed,
        planned_updates=args.planned_updates,
        manual_approval=args.manual_approval,
    )
    write_transition_plan(args.output, transition_plan(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "transition_plan", "write_transition_plan"]
