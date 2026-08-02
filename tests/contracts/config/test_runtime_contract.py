from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from sakuramoon.config import ConfigurationError, RuntimeConfig, load_config

REPOSITORY_ROOT = Path(__file__).parents[3]


def test_documentation_example_is_parseable_but_not_runtime_acceptable() -> None:
    with pytest.raises(ConfigurationError, match="unresolved decision/benchmark"):
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


def test_runtime_config_inventory_is_exact() -> None:
    config_root = REPOSITORY_ROOT / "config"
    observed = {
        path.relative_to(config_root)
        for path in config_root.rglob("*.toml")
    }
    assert observed == {
        Path("base.toml"),
        Path("engineering_capacity_s000.toml"),
        Path("engineering_capacity_s000_w1_b1.toml"),
        Path("engineering_capacity_s000_w1_b2.toml"),
        Path("engineering_capacity_s000_w2_b2.toml"),
        Path("engineering_capacity_s000_w3_b1.toml"),
        Path("engineering_capacity_s000_w3_b2.toml"),
        Path("engineering_eval_s000.toml"),
        Path("engineering_resume_s000.toml"),
        Path("engineering_smoke_s000.toml"),
        Path("eval.toml"),
        Path("examples/all_options.example.toml"),
        Path("sample.toml"),
        Path("train_g1.toml"),
        Path("train_g2.toml"),
        Path("train_h1.toml"),
        Path("train_h2.toml"),
        Path("train_s0.toml"),
        Path("train_s1.toml"),
        Path("train_s2.toml"),
        Path("train_s3.toml"),
    }
