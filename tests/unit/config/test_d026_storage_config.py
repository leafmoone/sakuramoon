from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import secrets
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError

import sakuramoon.data.production as production_module
from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.cli import data_service as data_service_cli
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.production import (
    ConfiguredDataLoader,
    ProductionBatchStreamIdentity,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.storage import StorageValidationError
from sakuramoon.train import preflight as preflight_module
from sakuramoon.train.preflight import (
    RestoredSingleGpuCheckpoint,
)
from sakuramoon.train.preflight import (
    _build_single_gpu_preflight_checks as build_single_gpu_preflight_checks,
)
from sakuramoon.train.step import SingleGpuUpdateState

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


class _UnusedCheckpointPublisher:
    def publish_preflight(
        self, identity: CheckpointIdentity, state: RawCheckpointState
    ) -> Path:
        del identity, state
        return Path("unused")

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        del state, reason, cadence
        return Path("unused")

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: CheckpointManifest,
        state: RawCheckpointState,
    ) -> None:
        del checkpoint, manifest, state

    def discard_preflight(self, checkpoint: Path) -> None:
        del checkpoint


def _restored_checkpoint(
    module: torch.nn.Module, optimizer: object
) -> RestoredSingleGpuCheckpoint:
    restored = object.__new__(RestoredSingleGpuCheckpoint)
    restored._manifest = CheckpointManifest(
        CheckpointKind.RAW,
        CheckpointIdentity("d026-unit", 0, _HASH_A, _HASH_B, _HASH_C),
        (FileRecord("payload", 123456, _HASH_D),),
    )
    restored._module = module
    restored._optimizer = optimizer
    restored._owner_pid = preflight_module.os.getpid()
    restored._path = Path("/test/d026-checkpoint")
    restored._payload_bytes = 123456
    restored._state = RawCheckpointState(
        SingleGpuUpdateState.initial(),
        GrowthCheckpointState(BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None),
        StageBudgetCheckpointState(0, 1),
        CheckpointCadence(0, 0.0),
    )
    restored._token = secrets.token_hex(32)
    preflight_module._RESTORED_CHECKPOINTS[restored._token] = restored
    return restored


def test_server_backed_storage_fields_are_explicit_and_small_cache_is_valid(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.storage.mode == "server_backed"
    assert config.storage.shared_filesystem == "nfs"
    assert config.storage.nfs_version == 3
    assert config.storage.hard_mount is True
    assert config.storage.measured_raw_checkpoint_bytes == 5143061370
    assert config.storage.checkpoint_copies == 3
    assert config.data.cache.low_watermark_gib == 8
    assert config.data.cache.high_watermark_gib == 16
    assert config.data.service.socket_path == "/run/sakuramoon/data-service.sock"
    assert config.data.service.ownership_lock_path == (
        "/run/sakuramoon/data-service.lock"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("storage", "mode"), "local_nvme"),
        (("storage", "shared_filesystem"), "ext4"),
        (("storage", "nfs_version"), 4),
        (("storage", "hard_mount"), False),
        (("storage", "checkpoint_copies"), 2),
        (("storage", "atomic_publish_probe"), False),
        (("data", "service", "socket_path"), "cache/data-service.sock"),
        (("data", "service", "ownership_lock_path"), "cache/service.lock"),
    ],
)
def test_server_backed_identity_and_runtime_path_drift_is_rejected(
    valid_payload: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    payload = copy.deepcopy(valid_payload)
    current = payload
    for part in path[:-1]:
        current = cast(dict[str, Any], current[part])
    current[path[-1]] = value

    with pytest.raises(ValidationError, match="literal_error"):
        RuntimeConfig.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    [
        ("storage",),
        ("storage", "shared_mount_source"),
        ("storage", "minimum_free_gib"),
        ("storage", "measured_raw_checkpoint_bytes"),
        ("data", "service", "ownership_lock_path"),
    ],
)
def test_server_backed_storage_fields_have_no_defaults(
    valid_payload: dict[str, Any], path: tuple[str, ...]
) -> None:
    payload = copy.deepcopy(valid_payload)
    current = payload
    for part in path[:-1]:
        current = cast(dict[str, Any], current[part])
    current.pop(path[-1])

    with pytest.raises(ValidationError, match="missing"):
        RuntimeConfig.model_validate(payload)


@pytest.mark.parametrize("low,high", [(0, 1), (8, 16), (128, 256)])
def test_explicit_bounded_server_cache_has_no_300_gib_floor(
    valid_payload: dict[str, Any], low: int, high: int
) -> None:
    valid_payload["data"]["cache"]["low_watermark_gib"] = low
    valid_payload["data"]["cache"]["high_watermark_gib"] = high

    config = RuntimeConfig.model_validate(valid_payload)

    assert config.data.cache.low_watermark_gib == low
    assert config.data.cache.high_watermark_gib == high


def test_training_preflight_calls_governed_storage_with_measured_checkpoint(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("[run]\n", encoding="utf-8")
    loaded = LoadedConfig(
        config,
        (),
        resolved.read_text(encoding="utf-8"),
        _HASH_A,
    )
    observed: list[tuple[RuntimeConfig, Path, int]] = []

    def require_storage(
        candidate: RuntimeConfig,
        root: Path,
        *,
        checkpoint_payload_bytes: int,
    ) -> None:
        observed.append((candidate, root, checkpoint_payload_bytes))

    monkeypatch.setattr(preflight_module, "require_training_storage", require_storage)
    module = torch.nn.Linear(1, 1)
    optimizer = object()
    restored = _restored_checkpoint(module, optimizer)
    session = DataServiceSessionIdentity(config.data.manifest.sha256, 2)
    data_client = SimpleNamespace(identity=session)
    stream_identity = ProductionBatchStreamIdentity(
        _HASH_A,
        ConfiguredDataLoader.from_config(config),
        config.data.manifest.sha256,
        session.sha256,
        _HASH_D,
    )
    batches = production_module._issue_batch_stream(
        cast(Iterator[TrainingBatch], iter(())), stream_identity
    )
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)
    plan = build_single_gpu_preflight_checks(
        loaded,
        repository_root=tmp_path,
        resolved_config_path=resolved,
        data_client=cast(Any, data_client),
        batches=batches,
        runtime=cast(Any, runtime),
        qwen=qwen,
        vae=vae,
        trainable_module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        workload=preflight_module._issue_verified_preflight_workload(
            {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
            config=config,
            runtime=runtime,
            trainable_module=module,
            optimizer=optimizer,
        ),
        checkpoint_publisher=_UnusedCheckpointPublisher(),
    )

    dict(plan._checks)["storage_capacity"]()

    assert observed == [(config, tmp_path, 123456)]


def test_data_service_cli_hard_fails_before_manifest_when_storage_drifts(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    loaded = cast(LoadedConfig, SimpleNamespace(config=config))
    observed: list[tuple[RuntimeConfig, Path]] = []

    def load_config(_path: Path, *, config_root: Path) -> LoadedConfig:
        del config_root
        return loaded

    monkeypatch.setattr(data_service_cli, "load_config", load_config)

    def reject_storage(candidate: RuntimeConfig, root: Path) -> None:
        observed.append((candidate, root))
        raise StorageValidationError("injected mount drift")

    monkeypatch.setattr(
        data_service_cli, "require_data_service_storage", reject_storage
    )

    result = data_service_cli.main(
        ["--config", str(tmp_path / "config.toml"), "--root", str(tmp_path)]
    )

    assert result == 1
    assert observed == [(config, tmp_path.resolve())]
    assert json.loads(capsys.readouterr().out) == {
        "error": "data_service_failed",
        "ok": False,
    }
