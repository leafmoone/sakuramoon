from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from sakuramoon.config import load_config
from sakuramoon.train.production import (
    ProductionReadinessError,
    require_production_irepa_readiness,
)

REPOSITORY_ROOT = Path(__file__).parents[3]

SECRET_ENVIRONMENT = {
    "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
    "WANDB_API_KEY": "synthetic-wandb-secret",
}

IREPA_REQUIRED = {
    "teacher_id": "facebook/PE-Spatial-B16-512",
    "tap_slot": 8,
    "projector_kernel_size": 3,
    "spatial_norm": "zscore",
    "loss": "cosine",
}


def _irepa_table(**overrides: Any) -> dict[str, Any]:
    table = {"enabled": True, **IREPA_REQUIRED}
    table.update(overrides)
    return table


def _load(tmp_path: Path, *, irepa: dict[str, Any] | None) -> Any:
    """Load the real S0 config (same fixture source as the config tests),
    optionally with an [irepa] table appended."""

    payload: dict[str, Any] = load_config(
        Path("train_s0.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment=SECRET_ENVIRONMENT,
    ).config.model_dump(
        mode="python",
        by_alias=True,
        exclude_computed_fields=True,
        exclude_none=True,
    )
    if irepa is not None:
        payload = copy.deepcopy(payload)
        payload["irepa"] = irepa
    (tmp_path / "run.toml").write_text(
        tomli_w.dumps(payload), encoding="utf-8"
    )
    return load_config(
        Path("run.toml"),
        config_root=tmp_path,
        environment=SECRET_ENVIRONMENT,
    )


def test_enabled_irepa_fails_production_readiness(tmp_path: Path) -> None:
    loaded = _load(tmp_path, irepa=_irepa_table())

    with pytest.raises(ProductionReadinessError) as excinfo:
        require_production_irepa_readiness(loaded.config)
    assert excinfo.value.blockers == (
        (
            "iREPA runtime graph is implemented, but no-iREPA→iREPA "
            "checkpoint/optimizer migration is not installed"
        ),
    )


def test_absent_irepa_passes_production_readiness(tmp_path: Path) -> None:
    loaded = _load(tmp_path, irepa=None)

    assert loaded.config.irepa is None
    require_production_irepa_readiness(loaded.config)


def test_disabled_irepa_passes_production_readiness(tmp_path: Path) -> None:
    loaded = _load(tmp_path, irepa=_irepa_table(enabled=False))

    assert loaded.config.irepa is not None
    assert loaded.config.irepa.enabled is False
    require_production_irepa_readiness(loaded.config)
