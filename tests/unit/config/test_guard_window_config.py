"""Resolved TOML contracts for the opt-in guard window stream."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w
from pydantic import ValidationError

from sakuramoon.config.load import LoadedConfig, load_config
from sakuramoon.config.schema import GuardWindowConfig


def _load(
    tmp_path: Path,
    payload: dict[str, Any],
    environment: dict[str, str],
    section: dict[str, object] | None,
) -> LoadedConfig:
    candidate = copy.deepcopy(payload)
    if section is not None:
        candidate["guard_window"] = section
    (tmp_path / "run.toml").write_text(tomli_w.dumps(candidate), encoding="utf-8")
    return load_config(Path("run.toml"), config_root=tmp_path, environment=environment)


def test_absent_guard_window_adds_no_resolved_config_bytes(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(tmp_path, valid_payload, secret_environment, None)
    assert loaded.config.guard_window is None
    assert "guard_window" not in tomllib.loads(loaded.resolved_toml)
    assert "guard_window" not in loaded.config.model_dump(exclude_none=True)


def test_disabled_guard_window_preserves_optimizer_and_training_config(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    baseline = _load(tmp_path, valid_payload, secret_environment, None)
    disabled = _load(tmp_path, valid_payload, secret_environment, {"enabled": False})
    assert disabled.config.guard_window == GuardWindowConfig()
    assert baseline.config.optimizer == disabled.config.optimizer
    assert baseline.config.train == disabled.config.train
    assert baseline.config.checkpoint == disabled.config.checkpoint
    # The opt-in block itself can be serialized; no unrelated config changes.
    before = tomllib.loads(baseline.resolved_toml)
    after = tomllib.loads(disabled.resolved_toml)
    assert after.pop("guard_window")["enabled"] is False
    assert after == before


def test_enabled_guard_window_roundtrips_session_and_cadence(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(
        tmp_path,
        valid_payload,
        secret_environment,
        {"enabled": True, "session_id": "guard-session-2", "window_updates": 3},
    )
    section = loaded.config.guard_window
    assert isinstance(section, GuardWindowConfig)
    assert section.enabled is True
    assert section.session_id == "guard-session-2"
    assert section.window_updates == 3
    assert tomllib.loads(loaded.resolved_toml)["guard_window"] == {
        "enabled": True,
        "session_id": "guard-session-2",
        "window_updates": 3,
    }


@pytest.mark.parametrize("cadence", [0, -1])
def test_nonpositive_guard_window_cadence_is_rejected(cadence: int) -> None:
    with pytest.raises(ValidationError):
        GuardWindowConfig(window_updates=cadence)


def test_enabled_guard_window_requires_explicit_session() -> None:
    with pytest.raises(ValidationError, match="session_id"):
        GuardWindowConfig(enabled=True)
