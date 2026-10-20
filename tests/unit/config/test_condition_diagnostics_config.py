from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from sakuramoon.config.load import load_config
from sakuramoon.config.schema import ConditionDiagnosticsConfig


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def _load(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
    *,
    section: dict[str, Any] | None = None,
):
    payload = copy.deepcopy(valid_payload)
    if section is not None:
        payload["condition_diagnostics"] = section
    _write_toml(tmp_path / "run.toml", payload)
    return load_config(
        Path("run.toml"),
        config_root=tmp_path,
        environment=secret_environment,
    )


def test_resolved_toml_omits_condition_diagnostics_by_default(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    """Gate 2 (no new config bytes when off): with the section absent the
    resolved TOML carries no [condition_diagnostics] table and the dumped
    config omits the key entirely."""

    loaded = _load(tmp_path, valid_payload, secret_environment)

    assert loaded.config.condition_diagnostics is None
    assert "condition_diagnostics" not in tomllib.loads(loaded.resolved_toml)
    dumped = loaded.config.model_dump(
        mode="python", by_alias=True, exclude_none=True
    )
    assert "condition_diagnostics" not in dumped


def test_resolved_toml_carries_enabled_condition_diagnostics(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(
        tmp_path,
        valid_payload,
        secret_environment,
        section={"enabled": True, "log_every_n": 4, "flush_every_records": 25},
    )

    section = loaded.config.condition_diagnostics
    assert isinstance(section, ConditionDiagnosticsConfig)
    assert section.enabled is True
    assert section.log_every_n == 4
    assert section.flush_every_records == 25
    assert "[condition_diagnostics]" in loaded.resolved_toml
