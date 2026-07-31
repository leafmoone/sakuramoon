"""Successful-update checkpoint cadence and fail-closed raw retention."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    manifest_from_dict,
)


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
    root: Path
    accepted_checkpoint_ids: frozenset[str]
    keep: tuple[Path, ...]
    remove: tuple[Path, ...]
    identities: tuple[tuple[Path, CheckpointIdentity], ...]

    def __post_init__(self) -> None:
        paths = self.keep + self.remove
        if (
            not self.root.is_absolute()
            or any(type(value) is not str or not value for value in self.accepted_checkpoint_ids)
            or len(paths) != len(set(paths))
            or tuple(path for path, _ in self.identities) != tuple(sorted(paths))
            or len(self.identities) != len(paths)
        ):
            raise ValueError("raw retention plan is invalid")


def _retention_manifest(path: Path) -> CheckpointManifest:
    """Validate complete checkpoint metadata without rehashing multi-GB payloads."""

    if path.is_symlink() or not path.is_dir():
        raise CheckpointError("retention checkpoint must be a real directory")
    complete = path / "COMPLETE"
    manifest_path = path / "manifest.json"
    try:
        if complete.is_symlink() or complete.read_bytes() != b"complete\n":
            raise CheckpointError("retention checkpoint is incomplete")
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise CheckpointError("retention checkpoint manifest is invalid")
        manifest = manifest_from_dict(json.loads(manifest_path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError("retention checkpoint metadata is unreadable") from None
    if manifest.kind is not CheckpointKind.RAW:
        raise CheckpointError("retention checkpoint is not raw")
    expected_name = (
        f"ckpt_{manifest.identity.update}_{manifest.identity.checkpoint_id}"
    )
    if path.name != expected_name:
        raise CheckpointError("retention checkpoint name differs from its identity")

    expected_files = {record.path: record.size for record in manifest.files}
    allowed_files = set(expected_files) | {"manifest.json", "COMPLETE"}
    allowed_directories = {
        parent.as_posix()
        for relative in allowed_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    actual_payloads: dict[str, int] = {}
    try:
        for item in path.rglob("*"):
            relative = item.relative_to(path).as_posix()
            if item.is_symlink():
                raise CheckpointError("retention checkpoint contains a symbolic link")
            if item.is_dir():
                if relative not in allowed_directories:
                    raise CheckpointError("retention checkpoint file set is invalid")
                continue
            if not item.is_file() or relative not in allowed_files:
                raise CheckpointError("retention checkpoint file set is invalid")
            if relative in expected_files:
                actual_payloads[relative] = item.stat().st_size
    except OSError:
        raise CheckpointError("retention checkpoint metadata is unreadable") from None
    if actual_payloads != expected_files:
        raise CheckpointError("retention checkpoint payload sizes differ from manifest")
    return manifest


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
    resolved_root = root.resolve(strict=True)
    raw: list[tuple[int, str, Path, CheckpointIdentity]] = []
    accepted: list[Path] = []
    identities: list[tuple[Path, CheckpointIdentity]] = []
    for unresolved_path in sorted(root.iterdir()):
        path = resolved_root / unresolved_path.name
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            manifest = _retention_manifest(path)
        except CheckpointError:
            continue
        if manifest.kind is not CheckpointKind.RAW:
            continue
        identities.append((path, manifest.identity))
        item = (
            manifest.identity.update,
            manifest.identity.checkpoint_id,
            path,
            manifest.identity,
        )
        if manifest.identity.checkpoint_id in accepted_checkpoint_ids:
            accepted.append(path)
        else:
            raw.append(item)
    accepted.sort()
    raw.sort(key=lambda item: (item[0], item[1], item[2].name))
    rolling_keep = tuple(item[2] for item in raw[-2:])
    remove = tuple(item[2] for item in raw[:-2])
    return RawRetentionPlan(
        root=resolved_root,
        accepted_checkpoint_ids=accepted_checkpoint_ids,
        keep=tuple(accepted) + rolling_keep,
        remove=remove,
        identities=tuple(sorted(identities)),
    )


def apply_raw_retention(
    root: Path,
    plan: RawRetentionPlan,
    *,
    accepted_checkpoint_ids: frozenset[str],
) -> None:
    """Apply a previously resolved plan, revalidating every deletion target."""

    current = plan_raw_retention(
        root, accepted_checkpoint_ids=accepted_checkpoint_ids
    )
    if current != plan:
        raise ValueError("retention plan is stale or was not produced for this policy")
    resolved_root = current.root
    expected_identities = dict(current.identities)
    for path in plan.remove:
        if (
            path.parent.resolve(strict=True) != resolved_root
            or path.is_symlink()
            or not path.is_dir()
        ):
            raise ValueError("retention target is outside the checkpoint root")
        manifest = _retention_manifest(path)
        if (
            manifest.kind is not CheckpointKind.RAW
            or manifest.identity != expected_identities[path]
        ):
            raise ValueError("retention target changed after planning")
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
