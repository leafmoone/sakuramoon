from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from sakuramoon.config.redact import REDACTED, redact_mapping
from sakuramoon.config.resolve import (
    resolved_config_bytes,
    resolved_config_sha256,
    write_resolved_config,
)
from sakuramoon.config.schema import RuntimeConfig


def test_redaction_recurses_but_keeps_environment_variable_names() -> None:
    payload = {
        "token": "raw-token",
        "nested": [{"password": "raw-password"}],
        "api_key_env": "WANDB_API_KEY",
        "content_sha256": "1" * 64,
        "secret_object": SecretStr("raw-secret"),
    }

    redacted = redact_mapping(payload)

    assert redacted["token"] == REDACTED
    assert redacted["nested"] == [{"password": REDACTED}]
    assert redacted["api_key_env"] == "WANDB_API_KEY"
    assert redacted["content_sha256"] == "1" * 64
    assert redacted["secret_object"] == REDACTED


def test_resolved_hash_matches_exact_bytes(valid_payload: dict[str, Any]) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    payload = resolved_config_bytes(config)

    assert b"DECISION_REQUIRED" not in payload
    assert b"model/qwen_3.5_2B" in payload
    assert (
        resolved_config_sha256(config)
        == __import__("hashlib").sha256(payload).hexdigest()
    )


def test_resolved_config_records_selected_sampling_identity(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    resolved = tomllib.loads(resolved_config_bytes(config).decode())

    assert resolved["sampling"]["profile"] == "balanced"
    assert resolved["sampling"]["solver"] == "heun_final_euler"
    assert resolved["sampling"]["steps"] == 25
    assert resolved["sampling"]["nfe"] == 49
    assert resolved["cfg"]["scale"] == 2.9
    assert resolved["evaluation"]["sampling"] == {
        "profile": "reference",
        "solver": "heun_final_euler",
        "steps": 50,
        "nfe": 99,
        "time_schedule": "linear",
    }


def test_atomic_writer_creates_normal_parent_chain(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    destination = tmp_path / "new" / "nested" / "resolved_config.toml"

    digest = write_resolved_config(config, destination)

    assert destination.read_bytes() == resolved_config_bytes(config)
    assert digest == resolved_config_sha256(config)
    assert not list(destination.parent.glob("*.tmp"))


def test_writer_rejects_symlinked_parent_before_creating_outside_directory(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="parent may not be a symlink"):
        write_resolved_config(
            config, tmp_path / "link" / "must-not-exist" / "resolved.toml"
        )
    assert not (outside / "must-not-exist").exists()


def test_writer_rejects_symlink_destination(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    target = tmp_path / "target.toml"
    target.write_text("untouched", encoding="utf-8")
    link = tmp_path / "resolved.toml"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="refusing to replace symlink"):
        write_resolved_config(config, link)
    assert target.read_text(encoding="utf-8") == "untouched"
