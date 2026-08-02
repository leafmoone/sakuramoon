"""Successful-update checkpoint cadence and fail-closed raw retention."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from sakuramoon.checkpoint.schema import (
    FORCED_CHECKPOINT_REASONS,
    CheckpointCadence,
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    CheckpointReason,
    manifest_from_dict,
)


@dataclass(frozen=True, slots=True)
class RawRetentionPlan:
    root: Path
    accepted_checkpoint_ids: frozenset[str]
    rolling_slots: int
    keep: tuple[Path, ...]
    remove: tuple[Path, ...]
    identities: tuple[tuple[Path, CheckpointIdentity], ...]

    def __post_init__(self) -> None:
        paths = self.keep + self.remove
        if (
            not self.root.is_absolute()
            or type(self.rolling_slots) is not int
            or self.rolling_slots <= 0
            or any(
                type(value) is not str or not value
                for value in self.accepted_checkpoint_ids
            )
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
    expected_name = f"ckpt_{manifest.identity.update}_{manifest.identity.checkpoint_id}"
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
    rolling_slots: int,
) -> RawRetentionPlan:
    """Keep all accepted raw checkpoints plus the configured rolling raws."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint retention root must be a real directory")
    if type(rolling_slots) is not int or rolling_slots <= 0:
        raise ValueError("raw checkpoint rolling slots must be a positive integer")
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
    rolling_keep = tuple(item[2] for item in raw[-rolling_slots:])
    remove = tuple(item[2] for item in raw[:-rolling_slots])
    return RawRetentionPlan(
        root=resolved_root,
        accepted_checkpoint_ids=accepted_checkpoint_ids,
        rolling_slots=rolling_slots,
        keep=tuple(accepted) + rolling_keep,
        remove=remove,
        identities=tuple(sorted(identities)),
    )


def apply_raw_retention(
    root: Path,
    plan: RawRetentionPlan,
    *,
    accepted_checkpoint_ids: frozenset[str],
    rolling_slots: int,
) -> None:
    """Apply a previously resolved plan, revalidating every deletion target."""

    current = plan_raw_retention(
        root,
        accepted_checkpoint_ids=accepted_checkpoint_ids,
        rolling_slots=rolling_slots,
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
