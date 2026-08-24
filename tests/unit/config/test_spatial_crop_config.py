"""P2 strict-table tests for the ``[data.spatial_crop]`` config surface.

The shifted-bucket policy is a strict, exact-type table: unknown keys,
integer literals in float positions, zoom-range inversions, and the
enabled/probability coupling must all fail before any training parameter
is derived.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from sakuramoon.config.load import ConfigurationError, load_config


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def _load(
    tmp_path: Path,
    payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> Any:
    _write_toml(tmp_path / "run.toml", payload)
    return load_config(
        Path("run.toml"),
        config_root=tmp_path,
        environment=secret_environment,
    )


def test_disabled_spatial_crop_defaults_load(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(tmp_path, valid_payload, secret_environment)
    policy = loaded.config.data.spatial_crop
    assert policy.enabled is False
    assert policy.probability == 0.0
    assert policy.mode == "shifted_bucket"
    assert policy.min_equivalent_zoom < policy.max_equivalent_zoom


def test_enabled_p25_policy_loads(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["enabled"] = True
    payload["data"]["spatial_crop"]["probability"] = 0.25
    loaded = _load(tmp_path, payload, secret_environment)
    assert loaded.config.data.spatial_crop.probability == 0.25


def test_unknown_spatial_crop_key_is_rejected(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["bogus_key"] = 1.0
    with pytest.raises(ConfigurationError, match="Extra inputs are not permitted"):
        _load(tmp_path, payload, secret_environment)


def test_float_positions_reject_toml_integers(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["probability"] = 1
    with pytest.raises(ConfigurationError, match="TOML float syntax"):
        _load(tmp_path, payload, secret_environment)


def test_min_zoom_must_stay_below_max_zoom(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["min_equivalent_zoom"] = 1.10
    with pytest.raises(
        ConfigurationError, match="min_equivalent_zoom must be below"
    ):
        _load(tmp_path, payload, secret_environment)


def test_enabled_requires_positive_probability(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["enabled"] = True
    payload["data"]["spatial_crop"]["probability"] = 0.0
    with pytest.raises(
        ConfigurationError, match="probability must be positive when enabled"
    ):
        _load(tmp_path, payload, secret_environment)


def test_disabled_requires_zero_probability(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["probability"] = 0.25
    with pytest.raises(
        ConfigurationError, match="probability must be zero when disabled"
    ):
        _load(tmp_path, payload, secret_environment)


def test_max_zoom_violating_crop_retention_is_rejected(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["data"]["spatial_crop"]["enabled"] = True
    payload["data"]["spatial_crop"]["probability"] = 0.25
    # 1/1.5**2 == 0.444 < 0.8, so the zoom overshoots the retention guard.
    payload["data"]["spatial_crop"]["max_equivalent_zoom"] = 1.5
    with pytest.raises(ConfigurationError, match="min_crop_retention guard"):
        _load(tmp_path, payload, secret_environment)
