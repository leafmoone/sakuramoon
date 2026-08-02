from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import (
    EvaluationEnabledConfig,
    FidEnabledConfig,
    IsEnabledConfig,
    RuntimeConfig,
)


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
    assert config.sampling.profile == "balanced"
    assert (
        config.sampling.solver,
        config.sampling.steps,
        config.sampling.nfe,
        config.sampling.time_schedule,
    ) == ("heun_final_euler", 25, 49, "linear")
    assert isinstance(config.evaluation, EvaluationEnabledConfig)
    assert config.evaluation.sampling.profile == "reference"
    assert config.evaluation.sampling.nfe == 99
    with pytest.raises(ValidationError, match="frozen"):
        config.run.seed = 5


def test_data_service_fields_are_required_and_range_workers_is_rejected(
    valid_payload: dict[str, Any],
) -> None:
    missing_lookahead = copy.deepcopy(valid_payload)
    missing_lookahead["data"]["cache"].pop("verified_shard_lookahead")
    with pytest.raises(
        ValidationError, match=r"(?s)verified_shard_lookahead.*missing"
    ):
        RuntimeConfig.model_validate(missing_lookahead)

    missing_service = copy.deepcopy(valid_payload)
    missing_service["data"].pop("service")
    with pytest.raises(ValidationError, match=r"(?s)service.*missing"):
        RuntimeConfig.model_validate(missing_service)

    stale = copy.deepcopy(valid_payload)
    stale["data"]["cache"]["range_workers"] = 2
    with pytest.raises(ValidationError, match=r"(?s)range_workers.*extra_forbidden"):
        RuntimeConfig.model_validate(stale)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("cache", "verified_shard_lookahead", 1),
        ("cache", "ready_batches_per_rank", 1),
        ("cache", "ready_batches_per_rank", 3),
        ("service", "lease_channel_capacity", 1),
        ("service", "ack_channel_capacity", 1),
    ],
)
def test_data_service_channels_cover_exact_worker_topology(
    valid_payload: dict[str, Any], section: str, field: str, value: int
) -> None:
    valid_payload["data"][section][field] = value
    with pytest.raises(ValidationError, match="exact worker topology"):
        RuntimeConfig.model_validate(valid_payload)


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
    ("path", "value"),
    [
        (("objective", "prediction_type"), "v"),
        (("objective", "loss"), "velocity_mse"),
        (("objective", "target_velocity"), "clean-noise"),
        (("objective", "endpoint_weighting"), "none"),
        (("logging", "noise_observation_boundary"), 0.5),
        (("timestep", "p_mean"), -0.7),
        (("timestep", "p_std"), 0.9),
        (("timestep", "noise_scale"), 0.9),
        (("timestep", "t_eps"), 0.1),
        (("cfg", "scale"), 3.0),
    ],
)
def test_strict_jlt_objective_identity_is_exactly_locked(
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


@pytest.mark.parametrize(
    ("profile", "field", "value"),
    [
        ("preview", "solver", "heun_final_euler"),
        ("preview", "steps", 27),
        ("balanced", "steps", 50),
        ("reference", "solver", "euler"),
        ("reference", "time_schedule", "cosine"),
    ],
)
def test_sampling_profile_registry_rejects_unapproved_combinations(
    valid_payload: dict[str, Any], profile: str, field: str, value: object
) -> None:
    valid_payload["sampling"]["profiles"][profile][field] = value

    with pytest.raises(ValidationError, match="sampling profile|literal"):
        RuntimeConfig.model_validate(valid_payload)


@pytest.mark.parametrize("profile", ["preview", "balanced", "reference"])
def test_runtime_sampling_profile_selection_is_explicit(
    valid_payload: dict[str, Any], profile: str
) -> None:
    valid_payload["sampling"]["profile"] = profile

    config = RuntimeConfig.model_validate(valid_payload)

    selected = config.sampling.profiles.model_dump()[profile]
    assert config.sampling.solver == selected["solver"]
    assert config.sampling.steps == selected["steps"]
    assert (
        config.sampling.nfe
        == {
            "preview": 28,
            "balanced": 49,
            "reference": 99,
        }[profile]
    )


def test_sampling_rejects_missing_unknown_or_user_supplied_nfe(
    valid_payload: dict[str, Any],
) -> None:
    missing = copy.deepcopy(valid_payload)
    missing["sampling"].pop("profile")
    with pytest.raises(ValidationError, match="profile"):
        RuntimeConfig.model_validate(missing)

    unknown = copy.deepcopy(valid_payload)
    unknown["sampling"]["profile"] = "custom"
    with pytest.raises(ValidationError, match="literal"):
        RuntimeConfig.model_validate(unknown)

    supplied = copy.deepcopy(valid_payload)
    supplied["sampling"]["nfe"] = 49
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig.model_validate(supplied)

    supplied_to_profile = copy.deepcopy(valid_payload)
    supplied_to_profile["sampling"]["profiles"]["preview"]["nfe"] = 28
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig.model_validate(supplied_to_profile)


def test_formal_evaluation_requires_reference_profile(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["evaluation"]["sampling"]["profile"] = "balanced"

    with pytest.raises(ValidationError, match="literal"):
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

    assert isinstance(config.evaluation, EvaluationEnabledConfig)
    assert isinstance(config.evaluation.fid, FidEnabledConfig)
    assert isinstance(config.evaluation.is_, IsEnabledConfig)
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


@pytest.mark.parametrize("revision", ["main", "A" * 40, "0" * 40, "master "])
def test_toml_dataset_revision_requires_master_branch(
    valid_payload: dict[str, Any], revision: str
) -> None:
    valid_payload["data"]["source"]["revision"] = revision

    with pytest.raises(ValidationError, match="revision"):
        RuntimeConfig.model_validate(valid_payload)


def test_training_manifest_is_automatic_and_rejects_external_hash_binding(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.data.manifest.path == "synthetic/train-manifest.json"
    assert config.data.manifest.initialize_if_missing is True
    assert config.data.manifest.refresh_existing is False

    for field, value in (
        ("path", "/tmp/train-manifest.json"),
        ("path", "../train-manifest.json"),
        ("path", "synthetic\\train-manifest.json"),
        ("initialize_if_missing", False),
        ("refresh_existing", True),
    ):
        candidate = copy.deepcopy(valid_payload)
        candidate["data"]["manifest"][field] = value
        with pytest.raises(ValidationError, match=field):
            RuntimeConfig.model_validate(candidate)

    stale = copy.deepcopy(valid_payload)
    stale["data"]["manifest"]["sha256"] = "3" * 64
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig.model_validate(stale)


def test_validation_uses_two_persistent_shards_without_external_manifest_hash(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.data.validation.selection_path == (
        "synthetic/validation-selection.json"
    )
    assert config.data.validation.shard_root == "synthetic/validation-shards"
    assert config.data.validation.shard_count == 2

    for field, value in (
        ("selection_path", "/tmp/selection.json"),
        ("selection_path", "../selection.json"),
        ("shard_root", "synthetic\\validation-shards"),
        ("shard_count", 1),
    ):
        candidate = copy.deepcopy(valid_payload)
        candidate["data"]["validation"][field] = value
        with pytest.raises(ValidationError, match=field):
            RuntimeConfig.model_validate(candidate)

    stale = copy.deepcopy(valid_payload)
    stale["data"]["validation"]["manifest_sha256"] = "4" * 64
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig.model_validate(stale)


def test_lr_schedule_is_linear_warmup_then_constant(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.scheduler.name == "linear_warmup_constant"
    assert config.scheduler.warmup_updates == 1000
    assert config.scheduler.max_lr == 0.00002
    assert config.scheduler.after_warmup == "constant"

    configurable = copy.deepcopy(valid_payload)
    configurable["scheduler"]["warmup_updates"] = 250
    configurable["scheduler"]["max_lr"] = 0.00001
    configurable["optimizer"]["lr"] = 0.00001
    changed = RuntimeConfig.model_validate(configurable)
    assert changed.scheduler.warmup_updates == 250
    assert changed.scheduler.max_lr == changed.optimizer.lr == 0.00001

    for field, value in (("name", "cosine"), ("after_warmup", "cosine")):
        candidate = copy.deepcopy(valid_payload)
        candidate["scheduler"][field] = value
        with pytest.raises(ValidationError, match=field):
            RuntimeConfig.model_validate(candidate)

    for field, value in (("warmup_updates", 0), ("max_lr", 0.0)):
        candidate = copy.deepcopy(valid_payload)
        candidate["scheduler"][field] = value
        with pytest.raises(ValidationError, match=field):
            RuntimeConfig.model_validate(candidate)

    mismatch = copy.deepcopy(valid_payload)
    mismatch["scheduler"]["max_lr"] = 0.00001
    with pytest.raises(ValidationError, match="scheduler.max_lr and optimizer.lr"):
        RuntimeConfig.model_validate(mismatch)


def test_checkpoint_interval_and_retention_are_explicit_positive_integers(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["checkpoint"]["full_every_updates"] = 37
    valid_payload["checkpoint"]["slots"] = 5
    valid_payload["storage"]["checkpoint_copies"] = 6
    config = RuntimeConfig.model_validate(valid_payload)

    assert config.checkpoint.full_every_updates == 37
    assert config.checkpoint.slots == 5
    assert config.storage.checkpoint_copies == 6

    for field, value in (
        ("full_every_updates", 0),
        ("full_every_updates", True),
        ("slots", 0),
        ("slots", True),
    ):
        candidate = copy.deepcopy(valid_payload)
        candidate["checkpoint"][field] = value
        with pytest.raises(ValidationError, match=field):
            RuntimeConfig.model_validate(candidate)


def test_disabled_profiling_and_benchmark_reject_stale_plan_fields(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["profiling"] = {"enabled": False}
    valid_payload["benchmark"] = {"enabled": False}
    config = RuntimeConfig.model_validate(valid_payload)
    assert config.profiling.model_dump() == {"enabled": False}
    assert config.benchmark.model_dump() == {"enabled": False}

    stale = copy.deepcopy(valid_payload)
    stale["profiling"]["schedule_updates"] = 10
    stale["benchmark"]["warmup_updates"] = 100
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig.model_validate(stale)


def test_profiling_and_benchmark_enablement_must_match(
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["profiling"] = {"enabled": False}
    with pytest.raises(ValidationError, match="enabled together"):
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
