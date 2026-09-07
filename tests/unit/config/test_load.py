from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, cast

import pytest
import tomli_w

from sakuramoon.config.load import ConfigurationError, load_config, resolve_secret


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")


def test_recursive_merge_is_deterministic(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    _write_toml(tmp_path / "base.toml", valid_payload)
    _write_toml(
        tmp_path / "overlay.toml",
        {"extends": ["base.toml"], "run": {"run_id": "merged"}},
    )

    first = load_config(
        Path("overlay.toml"),
        config_root=tmp_path,
        environment=secret_environment,
    )
    second = load_config(
        Path("overlay.toml"),
        config_root=tmp_path,
        environment=secret_environment,
    )

    assert first.config.run.run_id == "merged"
    assert first.config.caption.condition_mode == "artist_or_character"
    assert first.config.caption.dropout.condition_route == 0.0
    assert first.config.caption.dropout.condition_only == 0.0
    assert first.config.model.condition_tokens.token_count == 8
    assert first.resolved_toml == second.resolved_toml
    assert [item.path for item in first.inputs] == ["base.toml", "overlay.toml"]


def test_recursive_merge_replaces_arrays_before_strict_validation(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    _write_toml(tmp_path / "base.toml", valid_payload)
    base_slots = list(valid_payload["model"]["dit"]["active_slot_ids"])
    broken_slots = list(base_slots)
    broken_slots[1] = broken_slots[0]  # duplicate id -> schema failure
    overlay = {
        "extends": ["base.toml"],
        "run": {"run_id": "merged"},
        "model": {"dit": {"active_slot_ids": broken_slots}},
    }
    _write_toml(tmp_path / "overlay.toml", overlay)

    with pytest.raises(ConfigurationError, match="unique non-negative ids") as first:
        load_config(
            Path("overlay.toml"),
            config_root=tmp_path,
            environment=secret_environment,
        )
    with pytest.raises(ConfigurationError) as second:
        load_config(
            Path("overlay.toml"),
            config_root=tmp_path,
            environment=secret_environment,
        )

    assert str(first.value) == str(second.value)


def test_weight_decay_is_a_bounded_runtime_hyperparameter(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["optimizer"]["matrix_weight_decay"] = 0.125
    payload["optimizer"]["sensitive_weight_decay"] = 0.25
    _write_toml(tmp_path / "run.toml", payload)

    loaded = load_config(
        Path("run.toml"), config_root=tmp_path, environment=secret_environment
    )

    assert loaded.config.optimizer.matrix_weight_decay == 0.125
    assert loaded.config.optimizer.sensitive_weight_decay == 0.25


def test_jlt_learning_rate_scales_with_effective_global_batch(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    _write_toml(tmp_path / "run.toml", valid_payload)

    loaded = load_config(
        Path("run.toml"), config_root=tmp_path, environment=secret_environment
    )

    optimizer = valid_payload["optimizer"]
    train = valid_payload["train"]
    distributed = valid_payload["distributed"]
    expected_global_batch = (
        train["local_batch"]
        * train["accumulation"]
        * distributed["world_size"]
    )
    expected_learning_rate = (
        optimizer["base_lr"]
        * expected_global_batch
        / optimizer["reference_batch"]
    )

    assert loaded.config.effective_global_batch() == expected_global_batch
    assert loaded.config.scaled_learning_rate() == expected_learning_rate


@pytest.mark.parametrize(
    ("files", "entry", "expected"),
    [
        (
            {"a.toml": {"extends": ["b.toml"]}, "b.toml": {"extends": ["a.toml"]}},
            "a.toml",
            "extends cycle",
        ),
        (
            {"a.toml": {"extends": ["b.toml", "b.toml"]}, "b.toml": {}},
            "a.toml",
            "duplicate paths",
        ),
        (
            {"a.toml": {"extends": ["../outside.toml"]}},
            "a.toml",
            "config file does not exist",
        ),
        (
            {"a.toml": {"run": "bad"}, "b.toml": {"extends": ["a.toml"], "run": {}}},
            "b.toml",
            "table/scalar merge conflict",
        ),
    ],
)
def test_invalid_include_graphs_fail_before_schema_validation(
    tmp_path: Path,
    files: dict[str, dict[str, Any]],
    entry: str,
    expected: str,
) -> None:
    for name, payload in files.items():
        _write_toml(tmp_path / name, payload)

    with pytest.raises(ConfigurationError, match=expected):
        load_config(Path(entry), config_root=tmp_path, environment={})


def _symlink_available() -> bool:
    probe = Path(__file__).with_suffix(".probe")
    try:
        os.symlink(__file__, str(probe))
    except (OSError, NotImplementedError):
        return False
    probe.unlink(missing_ok=True)
    return True


@pytest.mark.skipif(not _symlink_available(), reason="symlink creation unavailable")
def test_symlinked_config_file_is_trusted_and_loads(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    """Config inputs are trusted paths: a symlinked .toml loads like a file."""

    target = tmp_path / "target.toml"
    _write_toml(target, valid_payload)
    (tmp_path / "link.toml").symlink_to(target)

    loaded = load_config(
        Path("link.toml"),
        config_root=tmp_path,
        environment=secret_environment,
    )
    assert loaded.config.run.run_id == valid_payload["run"]["run_id"]


def test_absolute_config_path_is_used_as_given(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    """A top-level config path may be absolute and is used as given."""

    outside = tmp_path.parent / f"{tmp_path.name}-outside.toml"
    try:
        _write_toml(outside, valid_payload)
        loaded = load_config(
            outside, config_root=tmp_path, environment=secret_environment
        )
        assert loaded.config.run.run_id == valid_payload["run"]["run_id"]
    finally:
        outside.unlink(missing_ok=True)


def test_loader_never_reads_dotenv(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    _write_toml(tmp_path / "run.toml", valid_payload)
    (tmp_path / ".env").write_text(
        "MODELSCOPE_API_TOKEN=must-not-be-loaded\nWANDB_API_KEY=must-not-be-loaded\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="MODELSCOPE_API_TOKEN"):
        load_config(Path("run.toml"), config_root=tmp_path, environment={})


def test_explicit_offline_load_skips_only_secret_presence_validation(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    _write_toml(tmp_path / "run.toml", valid_payload)

    loaded = load_config(
        Path("run.toml"),
        config_root=tmp_path,
        environment={},
        validate_secrets=False,
    )

    assert loaded.config.run.run_id == valid_payload["run"]["run_id"]
    with pytest.raises(TypeError, match="validate_secrets"):
        load_config(
            Path("run.toml"),
            config_root=tmp_path,
            environment={},
            validate_secrets=cast(Any, 0),
        )


def test_environment_values_do_not_appear_in_result(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    _write_toml(tmp_path / "run.toml", valid_payload)

    loaded = load_config(
        Path("run.toml"), config_root=tmp_path, environment=secret_environment
    )

    assert "MODELSCOPE_API_TOKEN" in loaded.resolved_toml
    assert "WANDB_API_KEY" in loaded.resolved_toml
    for value in secret_environment.values():
        assert value not in loaded.resolved_toml
        assert value not in repr(loaded)


def test_secret_resolution_returns_masked_transient_value() -> None:
    secret = resolve_secret("TOKEN_NAME", {"TOKEN_NAME": "sensitive-value"})

    assert secret.get_secret_value() == "sensitive-value"
    assert "sensitive-value" not in repr(secret)
    assert "sensitive-value" not in str(secret)


def test_validation_error_omits_input_secret_value(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    secret = "do-not-render-this-secret"
    payload = copy.deepcopy(valid_payload)
    payload["security"]["wandb_api_key"] = secret
    _write_toml(tmp_path / "bad.toml", payload)

    with pytest.raises(ConfigurationError) as captured:
        load_config(Path("bad.toml"), config_root=tmp_path, environment={})
    assert secret not in str(captured.value)


def test_no_process_environment_mutation(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    _write_toml(tmp_path / "run.toml", valid_payload)
    before = dict(os.environ)
    load_config(Path("run.toml"), config_root=tmp_path, environment=secret_environment)
    assert dict(os.environ) == before
