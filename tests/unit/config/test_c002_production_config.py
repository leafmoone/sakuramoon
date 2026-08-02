# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import importlib
import json
import multiprocessing as mp
import stat
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
import tomli_w
import torch
from pydantic import ValidationError
from wandb.errors import CommError

import sakuramoon.config.assembly as assembly_module
from sakuramoon.checkpoint.artifact import export_trainable_composite
from sakuramoon.config.assembly import (
    ManagedRemoteRun,
    RetryOnlyRemoteRun,
    TrainingTelemetryAssembly,
    build_trainable_composite_from_config,
    build_training_telemetry_from_config,
    initialize_wandb_run,
    trainable_composite_spec,
)
from sakuramoon.config.load import ConfigurationError, _Loader, load_config
from sakuramoon.config.schema import RuntimeConfig, looks_like_unresolved_sentinel
from sakuramoon.telemetry.metrics import CORE_TIMING_PHASES, DETAILED_TIMING_PHASES
from sakuramoon.telemetry.observer import UpdateMetricContext
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.runtime import require_single_gpu_config

REPOSITORY_ROOT = Path(__file__).parents[3]
CONFIG_ROOT = REPOSITORY_ROOT / "config"
TRAIN_CONFIGS = (
    "train_s0.toml",
    "train_s1.toml",
    "train_g1.toml",
    "train_s2.toml",
    "train_g2.toml",
    "train_s3.toml",
    "train_h1.toml",
    "train_h2.toml",
)
ENTRY_CONFIGS = (*TRAIN_CONFIGS, "eval.toml", "sample.toml")
RESOLVED_ENTRY_CONFIGS = ("train_s0.toml", "sample.toml")
UNRESOLVED_ENTRY_CONFIGS = tuple(
    name for name in ENTRY_CONFIGS if name not in RESOLVED_ENTRY_CONFIGS
)
PRODUCTION_CONFIGS = ("base.toml", *ENTRY_CONFIGS)
S000_ENGINEERING_CONFIGS = (
    "engineering_capacity_s000.toml",
    "engineering_capacity_s000_w1_b1.toml",
    "engineering_capacity_s000_w1_b2.toml",
    "engineering_capacity_s000_w2_b2.toml",
    "engineering_capacity_s000_w3_b1.toml",
    "engineering_capacity_s000_w3_b2.toml",
    "engineering_eval_s000.toml",
    "engineering_resume_s000.toml",
)


def _leaves(
    value: Mapping[str, Any], prefix: str = ""
) -> Iterator[tuple[str, object]]:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            yield from _leaves(cast(Mapping[str, Any], child), path)
        else:
            yield path, child


def _lookup(payload: Mapping[str, Any], path: str) -> object:
    current: object = payload
    for part in path.split("."):
        assert isinstance(current, Mapping)
        current = cast(Mapping[str, object], current)[part]
    return current


def _set_path(payload: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        child = current[part]
        assert isinstance(child, dict)
        current = cast(dict[str, Any], child)
    current[parts[-1]] = value


def _merged(name: str) -> dict[str, Any]:
    return _Loader(CONFIG_ROOT).load(Path(name))


def _synthetic_payload(
    name: str, valid_payload: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _merged(name)
    for path, value in tuple(_leaves(payload)):
        if looks_like_unresolved_sentinel(value):
            _set_path(payload, path, copy.deepcopy(_lookup(valid_payload, path)))
    global_batch = (
        cast(int, payload["stage"]["local_batch"])
        * cast(int, payload["stage"]["accumulation"])
        * cast(int, payload["stage"]["world_size"])
    )
    payload["stage"]["global_batch"] = global_batch
    return payload


def _load_flattened(
    tmp_path: Path,
    name: str,
    payload: dict[str, Any],
    environment: Mapping[str, str],
):
    path = tmp_path / name
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    return load_config(Path(path.name), config_root=tmp_path, environment=environment)


def _fresh_spawn_import(result: Any) -> None:
    try:
        importlib.import_module("sakuramoon.config.assembly")
        importlib.import_module("sakuramoon.data.production")
    except BaseException as error:  # noqa: BLE001
        result.put(f"{type(error).__name__}:{error}")
    else:
        result.put("ok")


def test_all_c002_toml_files_parse_and_entry_keys_are_complete(
    valid_payload: dict[str, Any],
) -> None:
    for name in PRODUCTION_CONFIGS:
        with (CONFIG_ROOT / name).open("rb") as stream:
            assert isinstance(tomllib.load(stream), dict)

    for name in ENTRY_CONFIGS:
        payload = _synthetic_payload(name, valid_payload)
        config = RuntimeConfig.model_validate(payload)
        assert set(payload) == set(RuntimeConfig.model_fields), name
        if name == "eval.toml":
            assert config.evaluation.enabled is True
        else:
            assert config.evaluation.enabled is False


def test_public_loader_accepts_resolved_and_rejects_unresolved_entries(
    secret_environment: dict[str, str],
) -> None:
    for name in RESOLVED_ENTRY_CONFIGS:
        load_config(
            Path(name),
            config_root=CONFIG_ROOT,
            environment=secret_environment,
        )

    for name in UNRESOLVED_ENTRY_CONFIGS:
        with pytest.raises(
            ConfigurationError, match="unresolved decision/benchmark placeholders"
        ):
            load_config(
                Path(name),
                config_root=CONFIG_ROOT,
                environment=secret_environment,
            )


def test_s000_engineering_configs_load_strictly(
    secret_environment: dict[str, str],
) -> None:
    for name in S000_ENGINEERING_CONFIGS:
        loaded = load_config(
            Path(name),
            config_root=CONFIG_ROOT,
            environment=secret_environment,
        )
        config = loaded.config
        assert config.run.intent == (
            "eval" if name == "engineering_eval_s000.toml" else "train"
        )
        assert config.run.stage == "S0"
        assert config.distributed.world_size == 1
        assert config.stage.world_size == 1
        assert config.stage.accumulation == 4
        assert config.stage.global_batch == (
            config.stage.local_batch
            * config.stage.accumulation
            * config.stage.world_size
        )
        assert config.evaluation.enabled is (
            name == "engineering_eval_s000.toml"
        )


def test_train_s0_has_no_unresolved_placeholders(
    secret_environment: dict[str, str],
) -> None:
    unresolved = {
        path: value
        for path, value in _leaves(_merged("train_s0.toml"))
        if looks_like_unresolved_sentinel(value)
    }
    assert unresolved == {}

    loaded = load_config(
        Path("train_s0.toml"),
        config_root=CONFIG_ROOT,
        environment=secret_environment,
    )
    assert loaded.config.stage.accumulation == 4
    assert loaded.config.stage.global_batch == (
        loaded.config.stage.local_batch
        * loaded.config.stage.accumulation
        * loaded.config.stage.world_size
    )


def test_current_templates_and_historical_c002_hashes_are_each_distinct(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    current_hashes: dict[str, str] = {}
    for index, name in enumerate(TRAIN_CONFIGS):
        loaded = _load_flattened(
            tmp_path,
            f"stage-{index}.toml",
            _synthetic_payload(name, valid_payload),
            secret_environment,
        )
        current_hashes[name] = loaded.resolved_sha256
    assert len(current_hashes) == 8
    assert len(set(current_hashes.values())) == 8

    report = json.loads(
        (
            REPOSITORY_ROOT
            / "docs/model-architecture/reviews/C002/stage_config_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["task_id"] == "C002"
    assert report["identity_scope"] == "synthetic_validation_only"
    assert report["production_hashes_published"] is False
    assert report["formal_stage_executed"] is False
    assert report["synthetic_substitution_source"] == (
        "tests/unit/config/conftest.py::SYNTHETIC_VALUES"
    )

    raw_historical_hashes = report["resolved_hashes"]
    assert isinstance(raw_historical_hashes, dict)
    historical_hashes = cast(dict[str, object], raw_historical_hashes)
    assert set(historical_hashes) == set(TRAIN_CONFIGS)
    raw_hash_values = tuple(historical_hashes.values())
    assert all(isinstance(digest, str) for digest in raw_hash_values)
    hash_values = cast(tuple[str, ...], raw_hash_values)
    assert report["all_hashes_distinct"] is True
    assert len(set(hash_values)) == len(TRAIN_CONFIGS)
    assert all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in hash_values
    )


def test_h1_h2_are_hashable_templates_but_never_trainable(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    for index, name in enumerate(("train_h1.toml", "train_h2.toml")):
        payload = _synthetic_payload(name, valid_payload)
        loaded = _load_flattened(
            tmp_path,
            f"template-{index}.toml",
            payload,
            secret_environment,
        )
        assert loaded.config.run.intent == "template"
        assert loaded.config.stage.enabled is False
        with pytest.raises(ValueError, match="train-intent"):
            require_single_gpu_config(loaded.config)

        payload["run"]["intent"] = "train"
        with pytest.raises(ValidationError, match="selected stage must be enabled"):
            RuntimeConfig.model_validate(payload)


def test_eval_and_sample_intents_cannot_cross_train_boundary(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    for index, (name, intent) in enumerate(
        (("eval.toml", "eval"), ("sample.toml", "sample"))
    ):
        loaded = _load_flattened(
            tmp_path,
            f"intent-{index}.toml",
            _synthetic_payload(name, valid_payload),
            secret_environment,
        )
        assert loaded.config.run.intent == intent
        with pytest.raises(ValueError, match="train-intent"):
            require_single_gpu_config(loaded.config)


def test_unimplemented_activation_checkpoint_modes_fail_at_runtime_boundary(
    valid_payload: dict[str, Any],
) -> None:
    payload = _synthetic_payload("train_s0.toml", valid_payload)
    payload["stage"]["activation_checkpoint_mode"] = "alternating"
    config = RuntimeConfig.model_validate(payload)
    with pytest.raises(ValueError, match="does not implement activation checkpointing"):
        require_single_gpu_config(config)


def test_adjacent_stage_diffs_are_metadata_budgets_and_one_main_axis() -> None:
    metadata_and_budgets = {
        "checkpoint.slots",
        "data.cache.download_concurrency",
        "data.cache.high_watermark_gib",
        "data.cache.low_watermark_gib",
        "data.cache.persistent_workers_per_rank",
        "data.cache.ready_batches_per_rank",
        "data.cache.verified_shard_lookahead",
        "data.service.ack_channel_capacity",
        "data.service.lease_channel_capacity",
        "data.service.request_timeout_seconds",
        "data.transport.connect_timeout_seconds",
        "data.transport.read_timeout_seconds",
        "data.transport.stream_chunk_bytes",
        "logging.flush_every_updates",
        "logging.local_jsonl_path",
        "logging.observer_event_timeout_seconds",
        "logging.observer_queue_capacity",
        "paths.artifact_dir",
        "paths.checkpoint_dir",
        "paths.run_dir",
        "run.intent",
        "run.run_id",
        "run.seed",
        "run.stage",
        "stage.accumulation",
        "stage.activation_checkpoint_mode",
        "stage.enabled",
        "stage.global_batch",
        "stage.local_batch",
        "stage.name",
        "stage.planned_updates",
        "stage.predecessor",
        "wandb.queue_capacity",
        "wandb.retry_jsonl_path",
    }
    permitted_axes = (
        {"distributed.backend", "distributed.world_size", "stage.world_size"},
        {"growth.enabled", "stage.depth"},
        {"growth.enabled", "stage.resolution"},
        {"growth.enabled", "stage.depth"},
        {"growth.enabled"},
        {"stage.resolution"},
        {"stage.resolution"},
    )
    for left_name, right_name, axes in zip(
        TRAIN_CONFIGS[:-1], TRAIN_CONFIGS[1:], permitted_axes, strict=True
    ):
        left = dict(_leaves(_merged(left_name)))
        right = dict(_leaves(_merged(right_name)))
        changed = {key for key in left if left[key] != right[key]}
        assert changed <= metadata_and_budgets | axes
        assert axes <= changed


def test_global_batch_and_valid_sample_budget_are_exact_for_all_stages(
    valid_payload: dict[str, Any],
) -> None:
    for name in TRAIN_CONFIGS:
        config = RuntimeConfig.model_validate(_synthetic_payload(name, valid_payload))
        assert config.stage.global_batch == (
            config.stage.local_batch
            * config.stage.accumulation
            * config.stage.world_size
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model.text.attention_heads", 8),
        ("model.text.mix_gate_init", 0.1),
        ("model.text.projection_bias", True),
        ("model.style.attention_heads", 8),
        ("model.style.init_std", 0.01),
        ("model.style.projection_bias", True),
    ],
)
def test_confirmed_text_style_values_reject_drift(
    valid_payload: dict[str, Any], path: str, value: object
) -> None:
    payload = _synthetic_payload("train_s0.toml", valid_payload)
    _set_path(payload, path, value)
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(payload)


def test_text_style_missing_and_unknown_constructor_fields_fail(
    valid_payload: dict[str, Any],
) -> None:
    missing = _synthetic_payload("train_s0.toml", valid_payload)
    missing["model"]["text"].pop("attention_heads")
    with pytest.raises(ValidationError, match="attention_heads"):
        RuntimeConfig.model_validate(missing)

    unknown = _synthetic_payload("train_s0.toml", valid_payload)
    unknown["model"]["style"]["constructor_default"] = 1
    with pytest.raises(ValidationError, match="constructor_default"):
        RuntimeConfig.model_validate(unknown)


def test_config_bound_composite_round_trips_exactly_on_meta(
    valid_payload: dict[str, Any],
) -> None:
    config = RuntimeConfig.model_validate(
        _synthetic_payload("train_s0.toml", valid_payload)
    )
    expected = trainable_composite_spec(config)
    module = build_trainable_composite_from_config(config, device=torch.device("meta"))
    assert export_trainable_composite(module) == expected


def test_attention_backend_mapping_is_explicit(valid_payload: dict[str, Any]) -> None:
    payload = _synthetic_payload("train_s0.toml", valid_payload)
    fa4 = RuntimeConfig.model_validate(payload)
    assert trainable_composite_spec(fa4)["dit"]["attention_backend"] == "fa4_varlen"  # type: ignore[index]

    payload["kernels"]["attention_backend"] = "dense_sdpa_reference"
    dense = RuntimeConfig.model_validate(payload)
    assert trainable_composite_spec(dense)["dit"]["attention_backend"] == "dense_sdpa"  # type: ignore[index]


def test_config_and_production_data_import_in_fresh_spawn() -> None:
    context = mp.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_fresh_spawn_import, args=(result,))
    process.start()
    process.join(timeout=20.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        pytest.fail("fresh spawned config/data import timed out")
    assert process.exitcode == 0
    assert result.get(timeout=2.0) == "ok"


class _ManagedRun:
    def __init__(
        self,
        *,
        log_error: BaseException | None = None,
        finish_error: BaseException | None = None,
    ) -> None:
        self.log_error = log_error
        self.finish_error = finish_error
        self.records: list[tuple[int, dict[str, int | float]]] = []
        self.finish_codes: list[int | None] = []

    def log(self, data: Mapping[str, int | float], *, step: int) -> None:
        if self.log_error is not None:
            raise self.log_error
        self.records.append((step, dict(data)))

    def finish(self, exit_code: int | None = None) -> None:
        self.finish_codes.append(exit_code)
        if self.finish_error is not None:
            raise self.finish_error


class _RunFactory:
    def __init__(self, run: ManagedRemoteRun) -> None:
        self.run = run
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        project: str,
        entity: str,
        run_id: str,
        run_directory: Path,
        resolved_sha256: str,
        resume_policy: str,
    ) -> ManagedRemoteRun:
        self.calls.append(
            {
                "project": project,
                "entity": entity,
                "run_id": run_id,
                "run_directory": run_directory,
                "resolved_sha256": resolved_sha256,
                "resume_policy": resume_policy,
            }
        )
        return self.run


def _metric_context(_observation: Any) -> UpdateMetricContext:
    return UpdateMetricContext(
        dit_flops=1,
        samples_per_second=1.0,
        ready_queue_depth=0,
        supplemental_phase_seconds={},
    )


def _telemetry_config(valid_payload: Mapping[str, Any]) -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        _synthetic_payload("train_s0.toml", valid_payload)
    )


def test_training_telemetry_assembly_binds_exact_config_and_close_order(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)
    run = _ManagedRun()
    factory = _RunFactory(run)
    resolved_sha256 = "a" * 64

    assembly = build_training_telemetry_from_config(
        config,
        repository_root=tmp_path,
        device=torch.device("cpu"),
        resolved_sha256=resolved_sha256,
        context_provider=_metric_context,
        remote_run_factory=factory,
    )

    assert factory.calls == [
        {
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "run_id": config.run.run_id,
            "run_directory": tmp_path / config.paths.run_dir,
            "resolved_sha256": resolved_sha256,
            "resume_policy": "allow",
        }
    ]
    assert assembly.observer.context_provider is _metric_context
    assert assembly.observer.event_timeout_seconds == (
        config.logging.observer_event_timeout_seconds
    )
    assert assembly.observer._queue.maxsize == config.logging.observer_queue_capacity
    assert assembly.remote._queue.maxsize == config.wandb.queue_capacity
    assert assembly.local.fsync_every_records == config.logging.flush_every_updates
    assert assembly.phase_timer.device == torch.device("cpu")
    assert config.timing.phases == (*CORE_TIMING_PHASES, *DETAILED_TIMING_PHASES)
    local_path = tmp_path / config.logging.local_jsonl_path
    retry_path = tmp_path / config.wandb.retry_jsonl_path
    assert stat.S_IMODE(local_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(retry_path.stat().st_mode) == 0o600

    assembly.close()

    assert run.finish_codes == [0]


@pytest.mark.parametrize(
    "phases",
    [
        pytest.param(
            (*CORE_TIMING_PHASES, *DETAILED_TIMING_PHASES[:-1]), id="missing"
        ),
        pytest.param(
            (*CORE_TIMING_PHASES, *DETAILED_TIMING_PHASES[:-1], "unknown"),
            id="unknown",
        ),
        pytest.param(
            (
                CORE_TIMING_PHASES[1],
                CORE_TIMING_PHASES[0],
                *CORE_TIMING_PHASES[2:],
                *DETAILED_TIMING_PHASES,
            ),
            id="order",
        ),
        pytest.param(
            (*CORE_TIMING_PHASES[:-1], "samples", *DETAILED_TIMING_PHASES),
            id="drift",
        ),
    ],
)
def test_timing_phase_vocabulary_rejects_every_contract_drift(
    valid_payload: dict[str, Any], phases: tuple[str, ...]
) -> None:
    payload = _synthetic_payload("train_s0.toml", valid_payload)
    payload["timing"]["phases"] = list(phases)

    with pytest.raises(ValidationError, match="fixed ordered vocabulary"):
        RuntimeConfig.model_validate(payload)


def test_training_telemetry_rejects_timing_drift_before_remote_setup(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _telemetry_config(valid_payload)
    drifted = (
        CORE_TIMING_PHASES[1],
        CORE_TIMING_PHASES[0],
        *CORE_TIMING_PHASES[2:],
    )
    monkeypatch.setattr(assembly_module, "CORE_TIMING_PHASES", drifted)
    factory = _RunFactory(_ManagedRun())

    with pytest.raises(ValueError, match="resolved timing phases"):
        build_training_telemetry_from_config(
            config,
            repository_root=tmp_path,
            device=torch.device("cpu"),
            resolved_sha256="f" * 64,
            context_provider=_metric_context,
            remote_run_factory=factory,
        )

    assert factory.calls == []


def test_training_telemetry_replays_retry_before_opening_new_queue(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)
    retry_path = tmp_path / config.wandb.retry_jsonl_path
    retry_path.parent.mkdir(parents=True)
    retry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "error_type": "ConnectionError",
                "successful_update": 3,
                "metrics": {"successful_update": 3, "total_loss": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    retry_path.chmod(0o600)
    run = _ManagedRun()

    assembly = build_training_telemetry_from_config(
        config,
        repository_root=tmp_path,
        device=torch.device("cpu"),
        resolved_sha256="b" * 64,
        context_provider=_metric_context,
        remote_run_factory=_RunFactory(run),
    )

    assert run.records == [(3, {"successful_update": 3, "total_loss": 1.0})]
    assert retry_path.exists()
    assert retry_path.read_bytes() == b""
    assembly.close()


def test_training_telemetry_replay_communication_failure_retains_queue(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)
    retry_path = tmp_path / config.wandb.retry_jsonl_path
    retry_path.parent.mkdir(parents=True)
    retry_payload = (
        json.dumps(
            {
                "schema_version": 1,
                "error_type": "ConnectionError",
                "successful_update": 3,
                "metrics": {"successful_update": 3},
            }
        )
        + "\n"
    )
    retry_path.write_text(retry_payload, encoding="utf-8")
    retry_path.chmod(0o600)
    run = _ManagedRun(log_error=ConnectionError("synthetic replay failure"))

    assembly = build_training_telemetry_from_config(
        config,
        repository_root=tmp_path,
        device=torch.device("cpu"),
        resolved_sha256="c" * 64,
        context_provider=_metric_context,
        remote_run_factory=_RunFactory(run),
    )

    assert retry_path.read_text(encoding="utf-8") == retry_payload
    assert (tmp_path / config.logging.local_jsonl_path).exists()
    assembly.close()
    assert run.finish_codes == [0]


def test_retry_only_run_retains_existing_queue_and_allows_local_startup(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)
    retry_path = tmp_path / config.wandb.retry_jsonl_path
    retry_path.parent.mkdir(parents=True)
    retry_payload = (
        json.dumps(
            {
                "schema_version": 1,
                "error_type": "ConnectionError",
                "successful_update": 4,
                "metrics": {"successful_update": 4},
            }
        )
        + "\n"
    )
    retry_path.write_text(retry_payload, encoding="utf-8")
    retry_path.chmod(0o600)

    assembly = build_training_telemetry_from_config(
        config,
        repository_root=tmp_path,
        device=torch.device("cpu"),
        resolved_sha256="1" * 64,
        context_provider=_metric_context,
        remote_run_factory=_RunFactory(RetryOnlyRemoteRun()),
    )

    assert retry_path.read_text(encoding="utf-8") == retry_payload
    assert (tmp_path / config.logging.local_jsonl_path).exists()
    assembly.close()


def test_malformed_retry_queue_remains_fatal_before_local_startup(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)
    retry_path = tmp_path / config.wandb.retry_jsonl_path
    retry_path.parent.mkdir(parents=True)
    retry_path.write_text("not-json\n", encoding="utf-8")
    retry_path.chmod(0o600)
    run = _ManagedRun()

    with pytest.raises(json.JSONDecodeError):
        build_training_telemetry_from_config(
            config,
            repository_root=tmp_path,
            device=torch.device("cpu"),
            resolved_sha256="2" * 64,
            context_provider=_metric_context,
            remote_run_factory=_RunFactory(run),
        )

    assert retry_path.read_text(encoding="utf-8") == "not-json\n"
    assert not (tmp_path / config.logging.local_jsonl_path).exists()
    assert run.finish_codes == [1]


def test_remote_finish_communication_failure_does_not_change_close_result(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)
    run = _ManagedRun(finish_error=ConnectionError("synthetic finish outage"))
    assembly = build_training_telemetry_from_config(
        config,
        repository_root=tmp_path,
        device=torch.device("cpu"),
        resolved_sha256="3" * 64,
        context_provider=_metric_context,
        remote_run_factory=_RunFactory(run),
    )

    assembly.close()

    assert run.finish_codes == [0]


def test_initialize_wandb_run_binds_stable_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandb

    captured: list[dict[str, object]] = []
    expected = _ManagedRun()

    def fake_init(**kwargs: object) -> _ManagedRun:
        captured.append(dict(kwargs))
        return expected

    monkeypatch.setattr(wandb, "init", fake_init)
    observed = initialize_wandb_run(
        project="sakuramoon",
        entity="synthetic-entity",
        run_id="stable-run-id",
        run_directory=tmp_path,
        resolved_sha256="d" * 64,
        resume_policy="allow",
    )

    assert observed is expected
    assert captured == [
        {
            "project": "sakuramoon",
            "entity": "synthetic-entity",
            "id": "stable-run-id",
            "name": "stable-run-id",
            "dir": str(tmp_path),
            "config": {"resolved_config_sha256": "d" * 64},
            "mode": "online",
            "resume": "allow",
            "reinit": "create_new",
            "save_code": False,
        }
    ]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ConnectionError("generic connection failure"), id="connection"),
        pytest.param(CommError("W&B communication failure"), id="wandb-comm"),
    ],
)
def test_initialize_wandb_network_failure_uses_explicit_retry_only_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    import wandb

    def fail_init(**_kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(wandb, "init", fail_init)
    run = initialize_wandb_run(
        project="sakuramoon",
        entity="synthetic-entity",
        run_id="stable-run-id",
        run_directory=tmp_path,
        resolved_sha256="e" * 64,
        resume_policy="allow",
    )

    assert type(run) is RetryOnlyRemoteRun
    with pytest.raises(ConnectionError):
        run.log({"successful_update": 1}, step=1)
    run.finish(exit_code=0)


def test_training_telemetry_rejects_invalid_remote_factory_result(
    tmp_path: Path,
    valid_payload: dict[str, Any],
) -> None:
    config = _telemetry_config(valid_payload)

    def invalid_factory(**_kwargs: object) -> object:
        return object()

    with pytest.raises(TypeError, match="callable log/finish"):
        build_training_telemetry_from_config(
            config,
            repository_root=tmp_path,
            device=torch.device("cpu"),
            resolved_sha256="f" * 64,
            context_provider=_metric_context,
            remote_run_factory=invalid_factory,  # type: ignore[arg-type]
        )

    assert not (tmp_path / config.logging.local_jsonl_path).exists()


class _OrderedClose:
    def __init__(
        self,
        name: str,
        order: list[str],
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.order = order
        self.error = error

    def close(self) -> None:
        self.order.append(self.name)
        if self.error is not None:
            raise self.error


class _OrderedRun(_ManagedRun):
    def __init__(
        self,
        order: list[str],
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.error = error

    def finish(self, exit_code: int | None = None) -> None:
        self.order.append(f"run:{exit_code}")
        if self.error is not None:
            raise self.error


def test_training_telemetry_close_attempts_every_component_and_preserves_errors() -> None:
    order: list[str] = []
    assembly = TrainingTelemetryAssembly(
        phase_timer=PhaseTimer(device=torch.device("cpu")),
        observer=cast(
            Any, _OrderedClose("observer", order, OSError("observer close"))
        ),
        remote=cast(Any, _OrderedClose("remote", order)),
        run=_OrderedRun(order, RuntimeError("run finish")),
        local=cast(Any, _OrderedClose("local", order, ValueError("local close"))),
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        assembly.close(exit_code=1)

    assert order == ["observer", "remote", "run:1", "local"]
    assert [type(error) for error in captured.value.exceptions] == [
        OSError,
        RuntimeError,
        ValueError,
    ]


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        ("logging.observer_queue_capacity", 0, "greater than or equal to 1"),
        ("logging.observer_event_timeout_seconds", 301.0, "less than or equal to 300"),
        ("wandb.queue_capacity", 1025, "less than or equal to 1024"),
        ("wandb.retry_jsonl_path", "outside/retry.jsonl", "artifact_dir"),
        ("wandb.retry_jsonl_path", "artifacts/s0/metrics.jsonl", "must differ"),
    ],
)
def test_training_telemetry_config_rejects_unbounded_or_unsafe_values(
    valid_payload: dict[str, Any],
    path: str,
    value: object,
    expected: str,
) -> None:
    payload = _synthetic_payload("train_s0.toml", valid_payload)
    _set_path(payload, path, value)
    with pytest.raises(ValidationError, match=expected):
        RuntimeConfig.model_validate(payload)
