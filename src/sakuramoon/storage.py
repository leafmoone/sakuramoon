"""Governed server-backed storage identity, capacity, and publication probes."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sakuramoon.config.schema import (
    EvaluationEnabledConfig,
    RuntimeConfig,
    StorageConfig,
)


class StorageValidationError(RuntimeError):
    """Configured storage does not match the governed server-backed identity."""


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
    required_bytes: int


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
    """Resolve the longest mountinfo entry containing an existing path."""

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
            filesystem = fields[separator + 1]
            source = _decode_mount_field(fields[separator + 2])
            options = frozenset(
                option
                for group in (fields[5], fields[separator + 3])
                for option in group.split(",")
            )
        except (IndexError, ValueError):
            raise StorageValidationError("mount identity is malformed") from None
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append(MountIdentity(mount_point, filesystem, source, options))
    if not candidates:
        raise StorageValidationError("configured path has no mount identity")
    return max(candidates, key=lambda identity: len(identity.mount_point.parts))


def repository_directory(root: Path, configured: str) -> Path:
    """Resolve and create one repository-relative directory without symlink escape."""

    try:
        base = root.resolve(strict=True)
        relative = Path(configured)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError
        candidate = base / relative
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError):
        raise StorageValidationError("configured persistent path is invalid") from None
    return resolved


def repository_file_parent(root: Path, configured: str) -> Path:
    """Resolve and create the parent of one repository-relative file path."""

    relative = Path(configured)
    if not relative.name:
        raise StorageValidationError("configured persistent file path is invalid")
    return repository_directory(root, str(relative.parent))


def require_shared_mount(path: Path, expected: StorageConfig) -> MountIdentity:
    identity = mount_identity(path)
    expected_version = f"vers={expected.nfs_version}"
    if (
        expected.mode != "server_backed"
        or identity.filesystem != expected.shared_filesystem
        or identity.source != expected.shared_mount_source
        or expected_version not in identity.options
        or (expected.hard_mount and "hard" not in identity.options)
        or "soft" in identity.options
    ):
        raise StorageValidationError(
            "persistent path differs from the governed server-backed mount"
        )
    return identity


def require_host_local_runtime(
    socket_path: Path, ownership_path: Path
) -> MountIdentity:
    if socket_path != Path(
        "/run/sakuramoon/data-service.sock"
    ) or ownership_path != Path("/run/sakuramoon/data-service.lock"):
        raise StorageValidationError("data service runtime paths are not governed")
    try:
        runtime = socket_path.parent
        if runtime.exists() and runtime.is_symlink():
            raise StorageValidationError("data service runtime directory is a symlink")
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(runtime, 0o700)
        resolved = runtime.resolve(strict=True)
        if resolved != Path("/run/sakuramoon"):
            raise StorageValidationError("data service runtime directory escaped /run")
        for candidate in (socket_path, ownership_path):
            if candidate.is_symlink():
                raise StorageValidationError("data service runtime path is a symlink")
        if ownership_path.exists() and not stat.S_ISREG(ownership_path.stat().st_mode):
            raise StorageValidationError("data service ownership path is not a file")
    except OSError as exc:
        raise StorageValidationError(
            "data service runtime path is unavailable"
        ) from exc
    identity = mount_identity(resolved)
    if identity.filesystem in {"nfs", "nfs4"}:
        raise StorageValidationError("data service runtime path must not be on NFS")
    return identity


def probe_atomic_publication(directory: Path) -> None:
    """Exercise the publication primitives required by state and checkpoints."""

    token = secrets.token_hex(16)
    temporary = directory / f".sakuramoon-storage-probe-{os.getpid()}-{token}.tmp"
    published = directory / f".sakuramoon-storage-probe-{os.getpid()}-{token}.published"
    payload = f"sakuramoon-storage-probe:{token}\n".encode("ascii")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, published)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if published.read_bytes() != payload:
            raise StorageValidationError("server-backed publication readback differs")
    except (OSError, StorageValidationError) as exc:
        if isinstance(exc, StorageValidationError):
            raise
        raise StorageValidationError(
            "server-backed atomic publication probe failed"
        ) from exc
    finally:
        cleanup_error: OSError | None = None
        for residue in (temporary, published):
            try:
                residue.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise StorageValidationError(
                "server-backed publication probe cleanup failed"
            )


def _validate_persistent_directories(
    directories: tuple[Path, ...], expected: StorageConfig
) -> tuple[MountIdentity, tuple[Path, ...]]:
    identities = tuple(require_shared_mount(path, expected) for path in directories)
    canonical = identities[0]
    if any(identity != canonical for identity in identities[1:]):
        raise StorageValidationError("persistent paths do not share one mount identity")
    unique_directories = tuple(dict.fromkeys(directories))
    if expected.atomic_publish_probe:
        for directory in unique_directories:
            probe_atomic_publication(directory)
    return canonical, unique_directories


def _capacity_report(
    mount: MountIdentity,
    paths: tuple[Path, ...],
    required_bytes: int,
) -> StorageCapacity:
    free = min(shutil.disk_usage(path).free for path in paths)
    if free < required_bytes:
        raise StorageValidationError(
            "server-backed free space cannot cover governed storage reservations"
        )
    return StorageCapacity(mount, free, required_bytes)


def require_data_service_storage(
    config: RuntimeConfig, repository_root: Path
) -> ServerBackedStorageReport:
    cache = repository_directory(repository_root, config.paths.cache_dir)
    mainset = repository_file_parent(repository_root, config.data.service.mainset_path)
    persistent_mount, probed = _validate_persistent_directories(
        (cache, mainset), config.storage
    )
    reserve = config.storage.minimum_free_gib * 1024**3
    cache_bytes = config.data.cache.high_watermark_gib * 1024**3
    capacity = _capacity_report(persistent_mount, probed, reserve + cache_bytes)
    runtime_mount = require_host_local_runtime(
        Path(config.data.service.socket_path),
        Path(config.data.service.ownership_lock_path),
    )
    return ServerBackedStorageReport(
        persistent_mount, runtime_mount, (capacity,), probed
    )


def _checkpoint_copy_count(config: RuntimeConfig) -> int:
    slots = config.checkpoint.slots
    copies = config.storage.checkpoint_copies
    if (
        type(slots) is not int
        or slots <= 0
        or type(copies) is not int
        or copies != slots + 1
    ):
        raise StorageValidationError(
            "checkpoint storage copies must equal retention slots plus one publishing copy"
        )
    return slots + 1


def require_training_storage(
    config: RuntimeConfig,
    repository_root: Path,
    *,
    checkpoint_payload_bytes: int,
) -> ServerBackedStorageReport:
    if type(checkpoint_payload_bytes) is not int or checkpoint_payload_bytes <= 0:
        raise StorageValidationError(
            "measured raw checkpoint bytes must be a positive integer"
        )
    checkpoint_copies = _checkpoint_copy_count(config)
    run = repository_directory(repository_root, config.paths.run_dir)
    cache = repository_directory(repository_root, config.paths.cache_dir)
    checkpoint = repository_directory(repository_root, config.paths.checkpoint_dir)
    artifact = repository_directory(repository_root, config.paths.artifact_dir)
    mainset = repository_file_parent(repository_root, config.data.service.mainset_path)
    persistent = (run, cache, checkpoint, artifact, mainset)
    persistent_mount, probed = _validate_persistent_directories(
        persistent, config.storage
    )
    reserve = config.storage.minimum_free_gib * 1024**3
    governed_checkpoint_bytes = max(
        checkpoint_payload_bytes,
        config.storage.measured_raw_checkpoint_bytes,
    )
    required = (
        reserve
        + config.data.cache.high_watermark_gib * 1024**3
        + checkpoint_copies * governed_checkpoint_bytes
    )
    capacity = _capacity_report(persistent_mount, probed, required)
    runtime_mount = require_host_local_runtime(
        Path(config.data.service.socket_path),
        Path(config.data.service.ownership_lock_path),
    )
    return ServerBackedStorageReport(
        persistent_mount, runtime_mount, (capacity,), probed
    )


def require_evaluation_storage(
    config: RuntimeConfig,
    repository_root: Path,
) -> ServerBackedStorageReport:
    """Validate evaluator persistence and its explicit output-space reservation."""

    checkpoint_copies = _checkpoint_copy_count(config)
    evaluation = config.evaluation
    if not evaluation.enabled:
        raise StorageValidationError("evaluation storage requires enabled evaluation")
    enabled_evaluation = cast(EvaluationEnabledConfig, evaluation)
    run = repository_directory(repository_root, config.paths.run_dir)
    cache = repository_directory(repository_root, config.paths.cache_dir)
    checkpoint = repository_directory(repository_root, config.paths.checkpoint_dir)
    artifact = repository_directory(repository_root, config.paths.artifact_dir)
    mainset = repository_file_parent(repository_root, config.data.service.mainset_path)
    persistent = (run, cache, checkpoint, artifact, mainset)
    persistent_mount, probed = _validate_persistent_directories(
        persistent, config.storage
    )
    required = (
        config.storage.minimum_free_gib * 1024**3
        + config.data.cache.high_watermark_gib * 1024**3
        + checkpoint_copies * config.storage.measured_raw_checkpoint_bytes
        + enabled_evaluation.output_reserve_gib * 1024**3
    )
    capacity = _capacity_report(persistent_mount, probed, required)
    runtime_mount = require_host_local_runtime(
        Path(config.data.service.socket_path),
        Path(config.data.service.ownership_lock_path),
    )
    return ServerBackedStorageReport(
        persistent_mount, runtime_mount, (capacity,), probed
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
    "require_evaluation_storage",
    "require_host_local_runtime",
    "require_shared_mount",
    "require_training_storage",
]
