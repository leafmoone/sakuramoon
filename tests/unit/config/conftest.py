from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sakuramoon.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[3]


@pytest.fixture
def secret_environment() -> dict[str, str]:
    return {
        "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
        "WANDB_API_KEY": "synthetic-wandb-secret",
    }


@pytest.fixture
def valid_payload(secret_environment: dict[str, str]) -> dict[str, Any]:
    """Use the real S0 config as the loader-test fixture."""

    loaded = load_config(
        Path("train_s0.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment=secret_environment,
    )
    return loaded.config.model_dump(
        mode="python",
        by_alias=True,
        exclude_computed_fields=True,
        exclude_none=True,
    )
