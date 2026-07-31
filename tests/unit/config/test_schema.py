from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import RuntimeConfig


def _add_unknown(data: dict[str, Any]) -> None:
    data["run"]["unexpected"] = True


def _remove_required(data: dict[str, Any]) -> None:
    data["run"].pop("seed")


def _wrong_type(data: dict[str, Any]) -> None:
    data["run"]["seed"] = "1"


def _out_of_range(data: dict[str, Any]) -> None:
    data["caption"]["dropout"]["general"] = 1.01


def _changed_fixed_nl(data: dict[str, Any]) -> None:
    data["caption"]["dropout"]["nl"]["nl3"] = 0.2


def _fixed_architecture_change(data: dict[str, Any]) -> None:
    data["model"]["dit"]["hidden_size"] = 2048


def _invalid_transition(data: dict[str, Any]) -> None:
    data["stage"]["predecessor"] = "S0"


def _world_size_mismatch(data: dict[str, Any]) -> None:
    data["distributed"]["world_size"] = 4


INVALID_MUTATIONS: list[tuple[Callable[[dict[str, Any]], None], str]] = [
    (_add_unknown, "extra_forbidden"),
    (_remove_required, "missing"),
    (_wrong_type, "int_type"),
    (_out_of_range, "less_than_equal"),
    (_changed_fixed_nl, "greater_than_equal"),
    (_fixed_architecture_change, "literal_error"),
    (_invalid_transition, "approved transition graph"),
    (_world_size_mismatch, "distributed and stage world_size"),
]


def test_valid_synthetic_fixture_is_strict_and_frozen(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.caption.dropout.all_condition == 0.1
    assert config.caption.dropout.general == 0.1
    assert config.caption.dropout.artist == 0.1
    assert config.caption.dropout.character == 0.2
    assert config.caption.dropout.copyright == 0.1
    assert config.caption.dropout.nsfw == 0.1
    assert config.caption.dropout.candidate_source == 0.3
    assert set(config.caption.dropout.nl.model_dump().values()) == {0.3}
    assert config.caption.condition_buckets[-1] == 512
    assert config.model.packing.modality_init_std == 0.02
    with pytest.raises(ValidationError, match="frozen"):
        config.run.seed = 5


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("caption", "dropout", "all_condition"), 0.2),
        (("caption", "dropout", "general"), 0.2),
        (("caption", "dropout", "artist"), 0.2),
        (("caption", "dropout", "character"), 0.1),
        (("caption", "dropout", "copyright"), 0.2),
        (("caption", "dropout", "nsfw"), 0.2),
        (("caption", "dropout", "candidate_source"), 0.2),
        (("caption", "dropout", "nl", "long_names"), 0.2),
        (("caption", "dropout", "nl", "long_no_names"), 0.2),
        (("caption", "dropout", "nl", "short_vibes"), 0.2),
        (("caption", "dropout", "nl", "nl2"), 0.2),
        (("caption", "dropout", "nl", "nl3"), 0.2),
    ],
)
def test_caption_dropout_values_are_exactly_locked(
    valid_payload: dict[str, Any], path: tuple[str, ...], value: float
) -> None:
    current = valid_payload
    for component in path[:-1]:
        child = current[component]
        assert isinstance(child, dict)
        current = cast(dict[str, Any], child)
    current[path[-1]] = value

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    INVALID_MUTATIONS,
)
def test_schema_hard_fails_invalid_values(
    valid_payload: dict[str, Any],
    mutation: Any,
    expected: str,
) -> None:
    payload = copy.deepcopy(valid_payload)
    mutation(payload)

    with pytest.raises(ValidationError, match=expected):
        RuntimeConfig.model_validate(payload)


def test_stage_backend_and_growth_are_cross_checked(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["growth"]["enabled"] = True
    with pytest.raises(ValidationError, match="growth is enabled only"):
        RuntimeConfig.model_validate(valid_payload)


def test_selected_stage_must_be_enabled(valid_payload: dict[str, Any]) -> None:
    valid_payload["stage"]["enabled"] = False

    with pytest.raises(ValidationError, match="selected stage must be enabled"):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("caption", "dropout", "general"), 1),
        (("model", "rope", "position_scale"), 16),
        (("optimizer", "betas"), [0.9, 1]),
        (("compile", "minimum_end_to_end_gain_percent"), 3),
        (("caption", "dropout", "general"), float("inf")),
        (("caption", "dropout", "general"), float("nan")),
    ],
)
def test_float_fields_require_toml_float_syntax_and_finite_values(
    valid_payload: dict[str, Any], path: tuple[str, ...], value: object
) -> None:
    current = valid_payload
    for component in path[:-1]:
        child = current[component]
        assert isinstance(child, dict)
        current = cast(dict[str, Any], child)
    current[path[-1]] = value

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(valid_payload)


def test_acceptance_sample_count_is_explicit_but_benchmark_configurable(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["evaluation"]["fid"]["acceptance_samples"] = 25000
    valid_payload["evaluation"]["is"]["acceptance_samples"] = 25000

    config = RuntimeConfig.model_validate(valid_payload)

    assert config.evaluation.fid.acceptance_samples == 25000
    assert config.evaluation.is_.acceptance_samples == 25000


@pytest.mark.parametrize("metric", ["fid", "is"])
def test_acceptance_sample_count_is_required(
    valid_payload: dict[str, Any], metric: str
) -> None:
    valid_payload["evaluation"][metric].pop("acceptance_samples")

    with pytest.raises(ValidationError, match="acceptance_samples"):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("table", "value"),
    [
        ("qwen", "model/another-qwen"),
        ("vae", "model/another-vae"),
    ],
)
def test_runtime_model_paths_are_fixed_local_directories(
    valid_payload: dict[str, Any], table: str, value: str
) -> None:
    valid_payload["assets"][table]["local_path"] = value

    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    "revision",
    [
        "master",
        "A" * 40,
        "0" * 39,
        "0" * 41,
    ],
)
def test_toml_dataset_revision_requires_lowercase_commit(
    valid_payload: dict[str, Any], revision: str
) -> None:
    valid_payload["data"]["source"]["revision"] = revision

    with pytest.raises(ValidationError, match="revision"):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connect_timeout_seconds", 0.0),
        ("connect_timeout_seconds", 1),
        ("read_timeout_seconds", 301.0),
        ("max_retries", -1),
        ("max_retries", True),
        ("retry_backoff_seconds", 1),
        ("stream_chunk_bytes", 65535),
    ],
)
def test_dataset_http_policy_is_strict_and_bounded(
    valid_payload: dict[str, Any], field: str, value: object
) -> None:
    valid_payload["data"]["transport"][field] = value

    with pytest.raises(ValidationError, match=field):
        RuntimeConfig.model_validate(valid_payload)


def test_higher_resolution_stages_cannot_be_enabled(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["run"]["stage"] = "H1"
    valid_payload["stage"].update(
        {
            "name": "H1",
            "enabled": True,
            "predecessor": "S3",
            "world_size": 4,
            "depth": 24,
            "resolution": 768,
        }
    )
    valid_payload["distributed"].update({"backend": "ddp", "world_size": 4})

    with pytest.raises(ValidationError, match="H1/H2 must remain disabled"):
        RuntimeConfig.model_validate(valid_payload)


def test_every_schema_table_requires_every_declared_property() -> None:
    schema = RuntimeConfig.model_json_schema(by_alias=True)
    tables = [schema, *schema["$defs"].values()]

    for table in tables:
        properties = table.get("properties")
        if properties:
            assert set(table.get("required", [])) == set(properties)
