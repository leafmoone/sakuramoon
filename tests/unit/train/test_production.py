from __future__ import annotations

# pyright: reportPrivateUsage=false
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from sakuramoon.checkpoint.policy import CheckpointCadence
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import ConfigurationError
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.train import production
from sakuramoon.train.runtime import SuccessfulTrainingObservation
from sakuramoon.train.step import SingleGpuUpdateState

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _state(update: int, *, terminal: int = 10) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(update, update, update * 2),
        growth=GrowthCheckpointState(BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None),
        stage_budget=StageBudgetCheckpointState(0, terminal),
        checkpoint_cadence=CheckpointCadence(update, float(update)),
    )


def _loaded(config: object) -> LoadedConfig:
    return LoadedConfig(
        cast(RuntimeConfig, config),
        (),
        "resolved\n",
        _HASH_A,
    )


def _optimizer(*, learning_rate: float = 2e-5) -> object:
    return SimpleNamespace(
        audit=SimpleNamespace(schema_sha256=_HASH_C),
        optimizer=SimpleNamespace(param_groups=[{"lr": learning_rate}]),
    )


def test_resume_requires_exact_absolute_raw_complete_directory(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "COMPLETE").write_bytes(b"complete\n")

    assert production._require_exact_resume_path(checkpoint) == checkpoint

    with pytest.raises(ConfigurationError, match="absolute"):
        production._require_exact_resume_path(Path("ckpt"))
    (checkpoint / "COMPLETE").write_bytes(b"incomplete\n")
    with pytest.raises(ConfigurationError, match="marker is invalid"):
        production._require_exact_resume_path(checkpoint)


def test_resume_rejects_symlinked_checkpoint_or_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    checkpoint = real / "ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_bytes(b"complete\n")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="symbolic links"):
        production._require_exact_resume_path(linked / "ckpt")


def test_default_runtime_bindings_report_only_unresolved_canonical_semantics() -> None:
    with pytest.raises(production.ProductionReadinessError) as captured:
        production._resolve_governed_runtime_bindings(cast(RuntimeConfig, object()))

    assert captured.value.blockers == (
        "S0_WARMUP_FUNCTION_UNRESOLVED",
        "S0_PASS_INDEX_OWNERSHIP_UNRESOLVED",
        "S0_FORMAL_PROMPT_CONDITION_CONTRACT_UNRESOLVED",
        "S0_LIVE_READY_QUEUE_DEPTH_UNBOUND",
        "S0_DIT_FLOPS_OBSERVATION_UNBOUND",
    )


def test_initial_raw_state_binds_absolute_s0_budget_and_current_wall_clock() -> None:
    config = SimpleNamespace(
        stage=SimpleNamespace(
            depth=16,
            name="S0",
            world_size=1,
            resolution=256,
            planned_updates=17,
        )
    )

    state = production._initial_raw_state(cast(RuntimeConfig, config), wall_clock=9.0)

    assert state.trainer == SingleGpuUpdateState.initial()
    assert state.growth == GrowthCheckpointState(
        BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None
    )
    assert state.stage_budget == StageBudgetCheckpointState(0, 17)
    assert state.checkpoint_cadence == CheckpointCadence(0, 9.0)


def test_scheduler_sets_fresh_rate_and_advances_only_consecutive_updates() -> None:
    optimizer = _optimizer()
    requested: list[int] = []

    def rate(_config: RuntimeConfig, update: int) -> float:
        requested.append(update)
        return float(update) / 10_000.0

    scheduler = production._SuccessfulUpdateLrScheduler(
        cast(RuntimeConfig, object()),
        cast(IsolatedAdamW8bit, optimizer),
        rate,
        restored_successful_update=0,
        fresh=True,
    )
    assert production._optimizer_learning_rate(
        cast(IsolatedAdamW8bit, optimizer)
    ) == pytest.approx(0.0001)

    scheduler(1)
    assert production._optimizer_learning_rate(
        cast(IsolatedAdamW8bit, optimizer)
    ) == pytest.approx(0.0002)
    assert requested == [1, 2]
    with pytest.raises(ValueError, match="consecutive"):
        scheduler(3)


def test_metric_context_uses_observed_flops_live_depth_and_update_wall() -> None:
    observation = cast(
        SuccessfulTrainingObservation,
        SimpleNamespace(
            loop=SimpleNamespace(
                update=SimpleNamespace(effective_samples=4),
                update_wall_seconds=2.0,
            )
        ),
    )
    flops_observed: list[SuccessfulTrainingObservation] = []
    depth_observed: list[SuccessfulTrainingObservation] = []
    context = production._ProductionMetricContext(
        lambda value: flops_observed.append(value) or 17,
        lambda value: depth_observed.append(value) or 3,
    )

    metric = context(observation)

    assert metric.dit_flops == 17
    assert metric.samples_per_second == pytest.approx(2.0)
    assert metric.ready_queue_depth == 3
    assert flops_observed == [observation]
    assert depth_observed == [observation]


def test_resume_sets_trusted_expected_lr_before_raw_loader_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SimpleNamespace()
    loaded = _loaded(config)
    optimizer = _optimizer()
    identity = CheckpointIdentity("raw-4", 4, _HASH_A, _HASH_B, _HASH_C)
    state = _state(4)
    events: list[tuple[str, object]] = []

    def read_state(_path: Path) -> tuple[Any, RawCheckpointState]:
        return SimpleNamespace(identity=identity), state

    monkeypatch.setattr(
        production,
        "read_raw_checkpoint_state",
        read_state,
    )

    def restore(
        checkpoint: Path,
        module: object,
        observed_optimizer: object,
        expected: CheckpointIdentity,
    ) -> object:
        del checkpoint, module
        events.append(
            (
                "restore",
                production._optimizer_learning_rate(
                    cast(IsolatedAdamW8bit, observed_optimizer)
                ),
            )
        )
        assert expected == identity
        return object()

    monkeypatch.setattr(production, "restore_single_gpu_checkpoint", restore)

    result = production._restore_checkpoint(
        loaded,
        checkpoint=tmp_path,
        dependency_sha256=_HASH_B,
        module=cast(Any, object()),
        optimizer=cast(IsolatedAdamW8bit, optimizer),
        learning_rate_for_update=lambda _config, update: float(update) / 100_000.0,
    )

    assert result is not None
    assert events == [("restore", 0.00005)]


@pytest.mark.parametrize("field", ["config", "dependency", "schema"])
def test_resume_rejects_identity_drift_before_raw_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    values = {
        "config_sha256": _HASH_A,
        "dependency_sha256": _HASH_B,
        "parameter_schema_sha256": _HASH_C,
    }
    values[
        {
            "config": "config_sha256",
            "dependency": "dependency_sha256",
            "schema": "parameter_schema_sha256",
        }[field]
    ] = "d" * 64
    identity = CheckpointIdentity("raw-4", 4, **values)

    def read_state(_path: Path) -> tuple[Any, RawCheckpointState]:
        return SimpleNamespace(identity=identity), _state(4)

    def must_not_restore(*_args: object, **_kwargs: object) -> None:
        pytest.fail("restore must not run after identity drift")

    monkeypatch.setattr(
        production,
        "read_raw_checkpoint_state",
        read_state,
    )
    monkeypatch.setattr(
        production,
        "restore_single_gpu_checkpoint",
        must_not_restore,
    )

    with pytest.raises(ValueError, match="identity differs"):
        production._restore_checkpoint(
            _loaded(SimpleNamespace()),
            checkpoint=tmp_path,
            dependency_sha256=_HASH_B,
            module=cast(Any, object()),
            optimizer=cast(IsolatedAdamW8bit, _optimizer()),
            learning_rate_for_update=lambda _config, _update: 2e-5,
        )


def test_bootstrap_requires_empty_root_and_restores_published_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(run_id="s0-production"),
        stage=SimpleNamespace(
            depth=16,
            name="S0",
            world_size=1,
            resolution=256,
            planned_updates=10,
        ),
    )
    optimizer = _optimizer(learning_rate=0.0001)
    checkpoint = tmp_path / "published"
    observed: list[tuple[CheckpointIdentity, RawCheckpointState, bytes]] = []

    def save(
        root: Path,
        identity: CheckpointIdentity,
        _module: object,
        _optimizer: object,
        state: RawCheckpointState,
        *,
        resolved_config: bytes,
    ) -> object:
        assert root == tmp_path
        observed.append((identity, state, resolved_config))
        checkpoint.mkdir()
        return SimpleNamespace(path=checkpoint)

    marker = object()

    def restore(
        path: Path,
        _module: object,
        _optimizer: object,
        identity: CheckpointIdentity,
    ) -> object | None:
        return marker if path == checkpoint and identity == observed[0][0] else None

    monkeypatch.setattr(production, "save_raw_checkpoint", save)
    monkeypatch.setattr(
        production,
        "restore_single_gpu_checkpoint",
        restore,
    )

    result = production._bootstrap_checkpoint(
        _loaded(config),
        checkpoint_root=tmp_path,
        dependency_sha256=_HASH_B,
        module=cast(Any, object()),
        optimizer=cast(IsolatedAdamW8bit, optimizer),
        wall_clock=11.0,
    )

    assert result is marker
    identity, state, payload = observed[0]
    assert identity.update == 0
    assert identity.config_sha256 == _HASH_A
    assert identity.dependency_sha256 == _HASH_B
    assert identity.parameter_schema_sha256 == _HASH_C
    assert state.checkpoint_cadence == CheckpointCadence(0, 11.0)
    assert payload == b"resolved\n"


def test_public_lifecycle_runs_static_preflight_then_reports_governed_blockers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = cast(RuntimeConfig, object())
    events: list[str] = []

    def load(
        _config_path: Path,
        *,
        config_root: Path,
        environment: object | None = None,
    ) -> LoadedConfig:
        del config_root, environment
        return _loaded(config)

    def accept(_config: RuntimeConfig) -> None:
        return None

    def static_preflight(observed: RuntimeConfig, root: Path) -> None:
        assert observed is config
        assert root == tmp_path
        events.append("static_preflight")

    def must_not_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "checkpoint, GPU, bootstrap, and service resources must remain blocked"
        )

    monkeypatch.setattr(production, "load_config", load)
    monkeypatch.setattr(production, "require_single_gpu_config", accept)
    monkeypatch.setattr(
        production,
        "require_static_single_gpu_preflight",
        static_preflight,
    )
    monkeypatch.setattr(production, "_dependency_sha256", must_not_run)
    monkeypatch.setattr(production, "repository_directory", must_not_run)
    monkeypatch.setattr(production, "_publish_resolved_config", must_not_run)
    monkeypatch.setattr(production.torch, "device", must_not_run)
    monkeypatch.setattr(production.torch.cuda, "is_available", must_not_run)
    monkeypatch.setattr(production, "_bootstrap_checkpoint", must_not_run)
    monkeypatch.setattr(
        production,
        "DataServiceClient",
        must_not_run,
    )

    with pytest.raises(production.ProductionReadinessError) as captured:
        production.run_production_single_gpu(
            Path("train_s0.toml"),
            config_root=Path("config"),
            repository_root=tmp_path,
        )

    assert captured.value.blockers == production._READINESS_BLOCKERS
    assert events == ["static_preflight"]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("preflight"), production.ProductionPreflightError),
        (
            production.ProductionTrainingError("training"),
            production.ProductionTrainingError,
        ),
    ],
)
def test_public_lifecycle_preserves_reachable_preflight_and_training_classes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Exception,
    expected: type[Exception],
) -> None:
    config = cast(RuntimeConfig, object())
    loaded = _loaded(config)

    def load(*_args: object, **_kwargs: object) -> LoadedConfig:
        return loaded

    def accept(_config: RuntimeConfig) -> None:
        return None

    monkeypatch.setattr(production, "load_config", load)
    monkeypatch.setattr(production, "require_single_gpu_config", accept)

    def fail(*_args: object, **_kwargs: object) -> production.ProductionTrainingResult:
        raise failure

    monkeypatch.setattr(production, "_run_accepted_lifecycle", fail)

    with pytest.raises(expected):
        production.run_production_single_gpu(
            Path("train_s0.toml"),
            config_root=Path("config"),
            repository_root=tmp_path,
        )


def test_resume_preflight_restores_before_service_and_skips_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            paths=SimpleNamespace(
                checkpoint_dir="checkpoints",
                artifact_dir="artifacts",
            ),
            evaluation=SimpleNamespace(gpu_index=0),
            run=SimpleNamespace(seed=7),
            data=SimpleNamespace(
                manifest=SimpleNamespace(sha256=_HASH_B),
                cache=SimpleNamespace(persistent_workers_per_rank=2),
                service=SimpleNamespace(
                    socket_path="/run/sakuramoon/data-service.sock",
                    request_timeout_seconds=5.0,
                ),
            ),
        ),
    )
    loaded = _loaded(config)
    resume = tmp_path / "resume"
    report_path = tmp_path / "artifacts" / "preflight.json"
    restored = SimpleNamespace(
        path=resume,
        state=SimpleNamespace(
            trainer=SingleGpuUpdateState(4, 4, 8),
            growth=SimpleNamespace(alpha=1.0),
            stage_budget=StageBudgetCheckpointState(0, 5),
        ),
    )
    qwen = SimpleNamespace(
        encoder=object(), tokenizer=SimpleNamespace(pad_token_id=248044)
    )
    module = object()
    optimizer = _optimizer()

    class Stream:
        def close(self) -> None:
            events.append("close")

    stream = Stream()

    class Factory:
        def batches(self, _client: object) -> Stream:
            events.append("batches")
            return stream

    class PipelineFactory:
        @classmethod
        def from_config(cls, *_args: object, **_kwargs: object) -> Factory:
            events.append("factory")
            return Factory()

    class Generator:
        def manual_seed(self, _seed: int) -> None:
            events.append("cuda_seed")

    def directory(root: Path, configured: str) -> Path:
        path = root / configured
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_config(_loaded: LoadedConfig, root: Path) -> Path:
        path = root / "resolved.toml"
        path.write_text("resolved\n")
        return path

    def restore(*_args: object, **_kwargs: object) -> object:
        events.append("restore")
        return restored

    def service(*_args: object, **_kwargs: object) -> object:
        assert "restore" in events
        events.append("service")
        return object()

    def run_preflight(_plan: object, destination: Path) -> object:
        nonlocal report_path
        events.append("preflight")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}")
        report_path = destination
        return object()

    def must_not_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("preflight-only lifecycle must not initialize training")

    def dependency(_root: Path) -> str:
        return _HASH_B

    def static_preflight(_config: RuntimeConfig, _root: Path) -> None:
        events.append("static_preflight")

    def resolve_bindings(
        _config: RuntimeConfig,
    ) -> production._GovernedRuntimeBindings:
        events.append("governed_bindings")
        return production._GovernedRuntimeBindings(
            pass_index=3,
            learning_rate_for_update=lambda _config, _update: 2e-5,
            dit_flops=lambda _observation: 1,
            ready_queue_depth=lambda _observation: 1,
        )

    def set_device(_device: object) -> None:
        return None

    def manual_seed(_seed: int) -> None:
        return None

    def qwen_loader(*_args: object) -> object:
        return qwen

    def vae_loader(*_args: object) -> object:
        return object()

    def module_factory(*_args: object, **_kwargs: object) -> object:
        return module

    def optimizer_factory(*_args: object) -> object:
        return optimizer

    def object_factory(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(production, "_dependency_sha256", dependency)
    monkeypatch.setattr(
        production,
        "require_static_single_gpu_preflight",
        static_preflight,
    )
    monkeypatch.setattr(
        production,
        "_resolve_governed_runtime_bindings",
        resolve_bindings,
    )
    monkeypatch.setattr(production, "repository_directory", directory)
    monkeypatch.setattr(production, "_publish_resolved_config", resolved_config)
    monkeypatch.setattr(production.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(production.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(production.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(production.torch.cuda, "default_generators", (Generator(),))
    monkeypatch.setattr(production.torch, "manual_seed", manual_seed)
    monkeypatch.setattr(production, "load_local_qwen", qwen_loader)
    monkeypatch.setattr(production, "load_local_mage_vae", vae_loader)
    monkeypatch.setattr(
        production,
        "build_trainable_composite_from_config",
        module_factory,
    )
    monkeypatch.setattr(production, "_build_optimizer", optimizer_factory)
    monkeypatch.setattr(production, "_restore_checkpoint", restore)
    monkeypatch.setattr(
        production,
        "_SuccessfulUpdateLrScheduler",
        object_factory,
    )
    monkeypatch.setattr(production, "DataServiceClient", service)
    monkeypatch.setattr(production, "ProductionPipelineFactory", PipelineFactory)
    monkeypatch.setattr(production, "_runtime", object_factory)
    monkeypatch.setattr(
        production,
        "ProductionSingleGpuCheckpointPublisher",
        object_factory,
    )
    monkeypatch.setattr(
        production,
        "build_single_gpu_preflight_workload",
        object_factory,
    )
    monkeypatch.setattr(
        production,
        "build_single_gpu_preflight_checks",
        object_factory,
    )
    monkeypatch.setattr(production, "run_single_gpu_preflight", run_preflight)
    monkeypatch.setattr(
        production, "build_training_telemetry_from_config", must_not_run
    )
    monkeypatch.setattr(production, "run_single_gpu_training", must_not_run)

    result = production._run_accepted_lifecycle(
        loaded,
        repository_root=tmp_path,
        resume=resume,
        preflight_only=True,
        wall_clock=lambda: 1.0,
    )

    assert result == production.ProductionTrainingResult(
        _HASH_A,
        report_path,
        resume,
        4,
        4,
        True,
    )
    assert events.index("restore") < events.index("service")
    assert events.index("static_preflight") < events.index("governed_bindings")
    assert events.index("governed_bindings") < events.index("restore")
    assert events[-1] == "close"
