from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sakuramoon import storage as storage_module
from sakuramoon.config.schema import StorageConfig
from sakuramoon.storage import (
    MountIdentity,
    StorageValidationError,
    mount_identity,
    probe_atomic_publication,
    require_host_local_runtime,
    require_shared_mount,
)


def _config() -> StorageConfig:
    return StorageConfig.model_validate(
        {
            "mode": "server_backed",
            "shared_filesystem": "nfs",
            "shared_mount_source": "server.example:/governed/export",
            "nfs_version": 3,
            "hard_mount": True,
            "minimum_free_gib": 8,
            "checkpoint_copies": 3,
            "atomic_publish_probe": True,
        }
    )


def test_mount_identity_uses_longest_entry_and_decodes_mount_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "shared path"
    nested.mkdir()
    escaped = str(nested).replace(" ", "\\040")
    mountinfo = (
        "1 0 0:1 / / rw - fuse.root root rw\n"
        f"2 1 0:2 / {escaped} rw,relatime - nfs "
        "server.example:/governed/export rw,vers=3,hard,proto=tcp\n"
    )
    original = Path.read_text

    def read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == Path("/proc/self/mountinfo"):
            return mountinfo
        return original(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)

    identity = mount_identity(nested)

    assert identity.mount_point == nested
    assert identity.filesystem == "nfs"
    assert identity.source == "server.example:/governed/export"
    assert {"vers=3", "hard", "proto=tcp"} <= identity.options


@pytest.mark.parametrize(
    "identity",
    [
        MountIdentity(Path("/shared"), "nfs4", "server.example:/governed/export", frozenset({"vers=3", "hard"})),
        MountIdentity(Path("/shared"), "nfs", "other:/export", frozenset({"vers=3", "hard"})),
        MountIdentity(Path("/shared"), "nfs", "server.example:/governed/export", frozenset({"vers=4", "hard"})),
        MountIdentity(Path("/shared"), "nfs", "server.example:/governed/export", frozenset({"vers=3", "soft"})),
    ],
)
def test_shared_mount_identity_drift_is_a_hard_failure(
    tmp_path: Path,
    identity: MountIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def observed_identity(_path: Path) -> MountIdentity:
        return identity

    monkeypatch.setattr(storage_module, "mount_identity", observed_identity)

    with pytest.raises(StorageValidationError, match="governed"):
        require_shared_mount(tmp_path, _config())


def test_atomic_publication_probe_fsync_replace_readback_and_cleans_residue(
    tmp_path: Path,
) -> None:
    probe_atomic_publication(tmp_path)

    assert not tuple(tmp_path.iterdir())


def test_atomic_publication_probe_failure_cleans_only_its_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retained = tmp_path / "retained"
    retained.write_text("user-owned", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(StorageValidationError, match="probe failed"):
        probe_atomic_publication(tmp_path)

    assert retained.read_text(encoding="utf-8") == "user-owned"
    assert tuple(tmp_path.iterdir()) == (retained,)


def test_runtime_ipc_rejects_any_non_governed_path(tmp_path: Path) -> None:
    with pytest.raises(StorageValidationError, match="not governed"):
        require_host_local_runtime(
            (tmp_path / "data-service.sock").absolute(),
            (tmp_path / "data-service.lock").absolute(),
        )


def test_capacity_requires_cache_three_checkpoints_and_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = MountIdentity(
        tmp_path,
        "nfs",
        "server.example:/governed/export",
        frozenset({"rw", "vers=3", "hard"}),
    )
    def shared_identity(_path: Path) -> MountIdentity:
        return mount

    def no_probe(_path: Path) -> None:
        return None

    def local_runtime(_socket: Path, _lock: Path) -> MountIdentity:
        return MountIdentity(Path("/"), "tmpfs", "tmpfs", frozenset({"rw"}))

    monkeypatch.setattr(storage_module, "mount_identity", shared_identity)
    monkeypatch.setattr(storage_module, "probe_atomic_publication", no_probe)
    monkeypatch.setattr(
        storage_module,
        "require_host_local_runtime",
        local_runtime,
    )
    config = SimpleNamespace(
        storage=_config(),
        paths=SimpleNamespace(
            run_dir="runs/test",
            cache_dir="cache/test",
            checkpoint_dir="checkpoints/test",
            artifact_dir="artifacts/test",
        ),
        data=SimpleNamespace(
            cache=SimpleNamespace(high_watermark_gib=16),
            service=SimpleNamespace(
                mainset_path="cache/mainset.json",
                socket_path="/run/sakuramoon/data-service.sock",
                ownership_lock_path="/run/sakuramoon/data-service.lock",
            ),
        ),
    )
    required = (8 + 16) * 1024**3 + 3 * 1024
    def insufficient_space(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=required - 1)

    monkeypatch.setattr(storage_module.shutil, "disk_usage", insufficient_space)

    with pytest.raises(StorageValidationError, match="free space"):
        storage_module.require_training_storage(
            config,  # pyright: ignore[reportArgumentType]
            tmp_path,
            checkpoint_payload_bytes=1024,
        )

    def exact_space(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=required)

    monkeypatch.setattr(storage_module.shutil, "disk_usage", exact_space)
    report = storage_module.require_training_storage(
        config,  # pyright: ignore[reportArgumentType]
        tmp_path,
        checkpoint_payload_bytes=1024,
    )
    assert report.capacities[0].required_bytes == required
    assert report.capacities[0].free_bytes == required
