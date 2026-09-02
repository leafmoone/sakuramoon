from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from sakuramoon.config.assembly import trainable_composite_spec
from sakuramoon.config.load import ConfigurationError, load_config

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


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def cast_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected a dict")
    return value


def _load(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
    *,
    irepa: dict[str, Any] | None = None,
):
    payload = copy.deepcopy(valid_payload)
    if irepa is not None:
        payload["irepa"] = irepa
    _write_toml(tmp_path / "run.toml", payload)
    return load_config(
        Path("run.toml"),
        config_root=tmp_path,
        environment=secret_environment,
    )


def test_legacy_config_without_irepa_parses_as_absent(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(tmp_path, valid_payload, secret_environment)

    assert loaded.config.irepa is None
    assert "irepa" not in loaded.resolved_toml
    dumped = loaded.config.model_dump(
        mode="python", by_alias=True, exclude_none=True
    )
    assert "irepa" not in dumped
    spec = trainable_composite_spec(loaded.config)
    assert spec["schema_version"] == 3
    assert "training_auxiliaries" not in spec


def test_explicit_disabled_irepa_parses_and_stays_legacy(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(
        tmp_path,
        valid_payload,
        secret_environment,
        irepa=_irepa_table(enabled=False),
    )

    assert loaded.config.irepa is not None
    assert loaded.config.irepa.enabled is False
    assert "[irepa]" in loaded.resolved_toml
    spec = trainable_composite_spec(loaded.config)
    assert spec["schema_version"] == 3
    assert "training_auxiliaries" not in spec


def test_enabled_irepa_parses_with_v1_defaults(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(tmp_path, valid_payload, secret_environment, irepa=_irepa_table())
    irepa = loaded.config.irepa
    assert irepa is not None and irepa.enabled is True
    assert irepa.teacher_local_path == "model/pe_spatial_b16_512"
    assert irepa.spatial_norm_gamma == 0.6
    assert irepa.spatial_norm_eps == 1e-6
    assert irepa.weight == 0.5
    assert irepa.ramp_in_updates == 1000
    assert irepa.ramp_out_after_updates is None
    assert irepa.ramp_out_updates == 1000

    spec = trainable_composite_spec(loaded.config)
    assert spec["schema_version"] == 4
    auxiliaries = spec.get("training_auxiliaries")
    assert type(auxiliaries) is dict
    assert set(auxiliaries) == {"irepa"}
    irepa_meta = cast_dict(auxiliaries)["irepa"]
    assert irepa_meta["in_channels"] == 2560
    assert irepa_meta["out_channels"] == 768
    assert irepa_meta["kernel_size"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("teacher_id", "facebook/PE-Spatial-B16-256"),
        ("tap_slot", 4),
        ("projector_kernel_size", 1),
        ("spatial_norm", "rms"),
        ("spatial_norm_eps", 1e-5),
        ("spatial_norm_gamma", 0.5),
        ("loss", "l2"),
        ("weight", -0.5),
        ("ramp_in_updates", 0),
        ("ramp_out_updates", -1),
        ("teacher_local_path", "../model/pe_spatial_b16_512"),
        ("teacher_local_path", "/absolute/model/path"),
        ("teacher_local_path", "model\\pe_spatial"),
        ("teacher_local_path", " model/pe_spatial_b16_512"),
    ],
)
def test_invalid_irepa_field_fails_closed(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ConfigurationError):
        _load(
            tmp_path,
            valid_payload,
            secret_environment,
            irepa=_irepa_table(**{field: value}),
        )


def test_unknown_irepa_key_fails_closed(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    with pytest.raises(ConfigurationError):
        _load(
            tmp_path,
            valid_payload,
            secret_environment,
            irepa=_irepa_table(experimental=True),
        )


def test_nonfinite_irepa_weight_fails_closed(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["irepa"] = _irepa_table(weight=0.5)
    toml_text = tomli_w.dumps(payload).replace("weight = 0.5", "weight = inf")
    assert "weight = inf" in toml_text
    (tmp_path / "run.toml").write_text(toml_text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(
            Path("run.toml"),
            config_root=tmp_path,
            environment=secret_environment,
        )


@pytest.mark.parametrize(
    "ramp_out_after_updates",
    [1000, 500, 1],
)
def test_invalid_ramp_schedule_fails_closed(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
    ramp_out_after_updates: int,
) -> None:
    with pytest.raises(ConfigurationError):
        _load(
            tmp_path,
            valid_payload,
            secret_environment,
            irepa=_irepa_table(ramp_out_after_updates=ramp_out_after_updates),
        )


def test_valid_ramp_schedule_parses(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    loaded = _load(
        tmp_path,
        valid_payload,
        secret_environment,
        irepa=_irepa_table(ramp_out_after_updates=10000),
    )
    assert loaded.config.irepa is not None
    assert loaded.config.irepa.ramp_out_after_updates == 10000
