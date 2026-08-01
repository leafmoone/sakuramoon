from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from sakuramoon.cli import data_service as data_service_cli
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.storage import StorageValidationError
from sakuramoon.train import preflight as preflight_module
from sakuramoon.train.preflight import build_single_gpu_preflight_checks


def test_server_backed_storage_fields_are_explicit_and_small_cache_is_valid(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.storage.mode == "server_backed"
    assert config.storage.shared_filesystem == "nfs"
    assert config.storage.nfs_version == 3
    assert config.storage.hard_mount is True
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
    loaded = cast(
        LoadedConfig,
        SimpleNamespace(config=config, resolved_toml=resolved.read_text(encoding="utf-8")),
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
    checks = build_single_gpu_preflight_checks(
        loaded,
        repository_root=tmp_path,
        resolved_config_path=resolved,
        data_client=cast(Any, object()),
        qwen=object(),
        vae=object(),
        trainable_module=cast(Any, object()),
        parameter_schema=lambda: None,
        image_shapes=lambda: None,
        text_shapes=lambda: None,
        zero_update_loss=lambda: None,
        optimizer_step=lambda: None,
        sample=lambda: None,
        checkpoint_round_trip=lambda: None,
        checkpoint_payload_bytes=123456,
    )

    checks["storage_capacity"]()

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
