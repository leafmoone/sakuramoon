"""Successful-update checkpoint cadence and fail-closed raw retention."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sakuramoon.checkpoint.load import read_checkpoint_manifest
from sakuramoon.checkpoint.schema import CheckpointError, CheckpointKind


class CheckpointReason(StrEnum):
    UPDATE_CADENCE = "update-cadence"
    WALL_CADENCE = "wall-cadence"
    STAGE_FINALIZE = "stage-finalize"
    PRE_GROWTH = "pre-growth"
    POST_GROWTH = "post-growth"
    RAMP_MIDPOINT = "ramp-midpoint"
    RAMP_END = "ramp-end"
    PRE_DECAY = "pre-decay"


FORCED_CHECKPOINT_REASONS = frozenset(
    {
        CheckpointReason.STAGE_FINALIZE,
        CheckpointReason.PRE_GROWTH,
        CheckpointReason.POST_GROWTH,
        CheckpointReason.RAMP_MIDPOINT,
        CheckpointReason.RAMP_END,
        CheckpointReason.PRE_DECAY,
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointCadence:
    """Track the last durable raw checkpoint without advancing on failed saves."""

    last_successful_update: int
    last_monotonic_seconds: float
    every_successful_updates: int = 1000
    every_hours: float = 6.0

    def __post_init__(self) -> None:
        if (
            type(self.last_successful_update) is not int
            or self.last_successful_update < 0
            or type(self.last_monotonic_seconds) is not float
            or not math.isfinite(self.last_monotonic_seconds)
            or self.last_monotonic_seconds < 0.0
            or type(self.every_successful_updates) is not int
            or self.every_successful_updates != 1000
            or type(self.every_hours) is not float
            or self.every_hours != 6.0
        ):
            raise ValueError("checkpoint cadence differs from the locked policy")

    def due(
        self,
        *,
        successful_update: int,
        monotonic_seconds: float,
        forced: CheckpointReason | None = None,
    ) -> CheckpointReason | None:
        if (
            type(successful_update) is not int
            or successful_update < self.last_successful_update
            or type(monotonic_seconds) is not float
            or not math.isfinite(monotonic_seconds)
            or monotonic_seconds < self.last_monotonic_seconds
        ):
            raise ValueError("checkpoint cadence input is invalid")
        if forced is not None:
            if forced not in FORCED_CHECKPOINT_REASONS:
                raise ValueError("checkpoint reason is not a forced checkpoint")
            return forced
        if (
            successful_update > self.last_successful_update
            and successful_update % self.every_successful_updates == 0
        ):
            return CheckpointReason.UPDATE_CADENCE
        if monotonic_seconds - self.last_monotonic_seconds >= self.every_hours * 3600.0:
            return CheckpointReason.WALL_CADENCE
        return None

    def committed(
        self,
        *,
        successful_update: int,
        monotonic_seconds: float,
        reason: CheckpointReason,
    ) -> CheckpointCadence:
        if reason != self.due(
            successful_update=successful_update,
            monotonic_seconds=monotonic_seconds,
            forced=reason if reason in FORCED_CHECKPOINT_REASONS else None,
        ):
            raise ValueError("committed checkpoint reason differs from the due reason")
        return CheckpointCadence(successful_update, monotonic_seconds)


@dataclass(frozen=True, slots=True)
class RawRetentionPlan:
    keep: tuple[Path, ...]
    remove: tuple[Path, ...]


def plan_raw_retention(
    root: Path,
    *,
    accepted_checkpoint_ids: frozenset[str],
) -> RawRetentionPlan:
    """Keep all accepted raw checkpoints plus the two newest rolling raws."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint retention root must be a real directory")
    if any(type(value) is not str or not value for value in accepted_checkpoint_ids):
        raise ValueError("accepted checkpoint IDs must be nonempty strings")
    raw: list[tuple[int, str, Path]] = []
    accepted: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            manifest = read_checkpoint_manifest(path)
        except CheckpointError:
            continue
        if manifest.kind is not CheckpointKind.RAW:
            continue
        item = (manifest.identity.update, manifest.identity.checkpoint_id, path)
        if manifest.identity.checkpoint_id in accepted_checkpoint_ids:
            accepted.append(path)
        else:
            raw.append(item)
    accepted.sort()
    raw.sort(key=lambda item: (item[0], item[1], item[2].name))
    rolling_keep = tuple(item[2] for item in raw[-2:])
    remove = tuple(item[2] for item in raw[:-2])
    return RawRetentionPlan(tuple(accepted) + rolling_keep, remove)


def apply_raw_retention(root: Path, plan: RawRetentionPlan) -> None:
    """Apply a previously resolved plan, revalidating every deletion target."""

    resolved_root = root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint retention root must be a real directory")
    if set(plan.keep) & set(plan.remove):
        raise ValueError("retention keep/remove sets overlap")
    for path in plan.remove:
        if (
            path.parent.resolve(strict=True) != resolved_root
            or path.is_symlink()
            or not path.is_dir()
        ):
            raise ValueError("retention target is outside the checkpoint root")
        manifest = read_checkpoint_manifest(path)
        if manifest.kind is not CheckpointKind.RAW:
            raise ValueError("retention may remove complete raw checkpoints only")
    for path in plan.remove:
        shutil.rmtree(path)
        descriptor = os.open(resolved_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "FORCED_CHECKPOINT_REASONS",
    "CheckpointCadence",
    "CheckpointReason",
    "RawRetentionPlan",
    "apply_raw_retention",
    "plan_raw_retention",
]
