from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from sakuramoon.config import ConfigurationError, RuntimeConfig, load_config

REPOSITORY_ROOT = Path(__file__).parents[3]


def test_documentation_example_is_parseable_but_not_runtime_acceptable() -> None:
    with pytest.raises(ConfigurationError, match="unresolved.*caption.dropout"):
        load_config(
            Path("examples/all_options.example.toml"),
            config_root=REPOSITORY_ROOT / "config",
            environment={
                "MODELSCOPE_API_TOKEN": "not-used-before-placeholder-failure",
                "WANDB_API_KEY": "not-used-before-placeholder-failure",
            },
        )


def test_public_contract_exposes_only_validated_runtime_objects() -> None:
    assert RuntimeConfig.model_config.get("strict") is True
    assert RuntimeConfig.model_config.get("extra") == "forbid"
    assert RuntimeConfig.model_config.get("frozen") is True
    assert "require_secrets" not in inspect.signature(load_config).parameters


def test_no_runtime_config_file_exists_outside_documentation_example() -> None:
    runtime_toml = [
        path
        for path in (REPOSITORY_ROOT / "config").rglob("*.toml")
        if path.name != "all_options.example.toml"
    ]
    assert runtime_toml == []
