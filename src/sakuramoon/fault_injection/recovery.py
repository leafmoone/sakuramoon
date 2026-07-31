"""Explicit raw-checkpoint parent selection for fault recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sakuramoon.checkpoint.load import (
    discover_complete_checkpoints,
    read_checkpoint_manifest,
)
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
)


@dataclass(frozen=True, slots=True)
class ResumeParent:
    path: Path
    identity: CheckpointIdentity


def select_complete_raw_parent(
    root: Path, *, checkpoint_id: str, successful_update: int
) -> ResumeParent:
    """Select one exact COMPLETE raw checkpoint; never fall back to another point."""

    if not checkpoint_id or type(successful_update) is not int or successful_update < 0:
        raise ValueError("resume parent identity is invalid")
    matches: list[ResumeParent] = []
    for path in discover_complete_checkpoints(root):
        manifest = read_checkpoint_manifest(path)
        if (
            manifest.kind is CheckpointKind.RAW
            and manifest.identity.checkpoint_id == checkpoint_id
            and manifest.identity.update == successful_update
        ):
            matches.append(ResumeParent(path, manifest.identity))
    if len(matches) != 1:
        raise CheckpointError("exact COMPLETE raw recovery parent was not found")
    return matches[0]


__all__ = ["ResumeParent", "select_complete_raw_parent"]
