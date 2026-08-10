"""Filesystem helpers shared by the data service and training commands."""

from __future__ import annotations

import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from sakuramoon.config.schema import RuntimeConfig


class StorageValidationError(RuntimeError):
    """A configured path is unsafe or unavailable."""


@dataclass(frozen=True, slots=True)
class MountIdentity:
    mount_point: Path
    filesystem: str
    source: str
    options: frozenset[str]


@dataclass(frozen=True, slots=True)
class StorageCapacity:
    mount: MountIdentity
    free_bytes: int
    required_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ServerBackedStorageReport:
    persistent_mount: MountIdentity
    runtime_mount: MountIdentity
    capacities: tuple[StorageCapacity, ...]
    probed_directories: tuple[Path, ...]


_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


def _decode_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def mount_identity(path: Path) -> MountIdentity:
    """Return the longest mountinfo entry containing an existing path."""

    try:
        resolved = path.resolve(strict=True)
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StorageValidationError("mount identity is unreadable") from exc
    candidates: list[MountIdentity] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_point = Path(_decode_mount_field(fields[4]))
            identity = MountIdentity(
                mount_point=mount_point,
                filesystem=fields[separator + 1],
                source=_decode_mount_field(fields[separator + 2]),
                options=frozenset(
                    option
                    for group in (fields[5], fields[separator + 3])
                    for option in group.split(",")
                ),
            )
        except (IndexError, ValueError):
            raise StorageValidationError("mount identity is malformed") from None
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append(identity)
    if not candidates:
        raise StorageValidationError("configured path has no mount identity")
    return max(candidates, key=lambda item: len(item.mount_point.parts))


def repository_directory(
    root: Path, configured: str, *, allow_absolute: bool = False
) -> Path:
    """Resolve and create one repository-relative directory."""

    try:
        base = root.resolve(strict=True)
        relative = Path(configured)
        if relative.is_absolute():
            if not allow_absolute:
                raise ValueError
            resolved = relative.resolve(strict=False)
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved.resolve(strict=True)
        if ".." in relative.parts:
            raise ValueError
        candidate = base / relative
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError):
        raise StorageValidationError("configured repository path is invalid") from None
    return resolved


def repository_file_parent(
    root: Path, configured: str, *, allow_absolute: bool = False
) -> Path:
    relative = Path(configured)
    if not relative.name:
        raise StorageValidationError("configured repository file is invalid")
    return repository_directory(
        root, str(relative.parent), allow_absolute=allow_absolute
    )


def require_host_local_runtime(socket_path: Path, ownership_path: Path) -> MountIdentity:
    """Create the data-service runtime directory without fixing it to `/run`."""

    if socket_path.parent != ownership_path.parent:
        raise StorageValidationError("data service runtime files must share a directory")
    runtime = socket_path.parent
    try:
        if runtime.exists() and runtime.is_symlink():
            raise StorageValidationError("data service runtime directory is a symlink")
        runtime.mkdir(parents=True, exist_ok=True)
        resolved = runtime.resolve(strict=True)
        for candidate in (socket_path, ownership_path):
            if candidate.is_symlink():
                raise StorageValidationError("data service runtime path is a symlink")
    except OSError as exc:
        raise StorageValidationError("data service runtime path is unavailable") from exc
    return mount_identity(resolved)


def probe_atomic_publication(directory: Path) -> None:
    """Verify the rename primitive used to publish service state."""

    token = secrets.token_hex(16)
    temporary = directory / f".sakuramoon-probe-{os.getpid()}-{token}.tmp"
    published = directory / f".sakuramoon-probe-{os.getpid()}-{token}.published"
    payload = token.encode("ascii")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, published)
        if published.read_bytes() != payload:
            raise StorageValidationError("atomic publication readback differs")
    except OSError as exc:
        raise StorageValidationError("atomic publication probe failed") from exc
    finally:
        temporary.unlink(missing_ok=True)
        published.unlink(missing_ok=True)


def require_data_service_storage(
    config: RuntimeConfig, repository_root: Path
) -> ServerBackedStorageReport:
    """Prepare writable data-service paths on local or network storage."""

    directories = tuple(
        dict.fromkeys(
            (
                repository_directory(repository_root, config.paths.cache_dir),
                repository_file_parent(
                    repository_root,
                    config.data.service.mainset_path,
                    allow_absolute=True,
                ),
            )
        )
    )
    if config.storage.atomic_publish_probe:
        for directory in directories:
            probe_atomic_publication(directory)
    persistent_mount = mount_identity(directories[0])
    free_bytes = min(shutil.disk_usage(path).free for path in directories)
    runtime_mount = require_host_local_runtime(
        Path(config.data.service.socket_path),
        Path(config.data.service.ownership_lock_path),
    )
    return ServerBackedStorageReport(
        persistent_mount=persistent_mount,
        runtime_mount=runtime_mount,
        capacities=(StorageCapacity(persistent_mount, free_bytes),),
        probed_directories=directories,
    )


__all__ = [
    "MountIdentity",
    "ServerBackedStorageReport",
    "StorageCapacity",
    "StorageValidationError",
    "mount_identity",
    "probe_atomic_publication",
    "repository_directory",
    "repository_file_parent",
    "require_data_service_storage",
    "require_host_local_runtime",
]
