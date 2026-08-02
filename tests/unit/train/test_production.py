from __future__ import annotations

# pyright: reportPrivateUsage=false
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

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
        checkpoint_cadence=CheckpointCadence(update, float(update), 1000),
    )


def _loaded(config: object) -> LoadedConfig:
    return LoadedConfig(
        cast(RuntimeConfig, config),
        (),
        "resolved\n",
        _HASH_A,
    )


def _optimizer_groups(*learning_rates: object) -> object:
    return SimpleNamespace(
        audit=SimpleNamespace(schema_sha256=_HASH_C),
        optimizer=SimpleNamespace(
            param_groups=[{"lr": learning_rate} for learning_rate in learning_rates]
        ),
    )


def _optimizer(*, learning_rate: float = 2e-5) -> object:
    return _optimizer_groups(learning_rate)


def _group_learning_rate(optimizer: object, index: int = 0) -> object:
    return cast(Any, optimizer).optimizer.param_groups[index]["lr"]


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


def test_initial_raw_state_binds_absolute_s0_budget_and_current_wall_clock() -> None:
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(full_every_updates=7),
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
    assert state.checkpoint_cadence == CheckpointCadence(0, 9.0, 7)


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
    assert _group_learning_rate(optimizer) == 0.0001

    scheduler(1)
    assert _group_learning_rate(optimizer) == 0.0002
    assert requested == [1, 2]
    with pytest.raises(ValueError, match="consecutive"):
        scheduler(3)


def test_restored_scheduler_canonicalizes_expected_to_fp32_tensor_exactly() -> None:
    expected = 2e-8
    actual = torch.tensor(expected, dtype=torch.float32)
    before = actual.clone()
    assert float(actual.item()) != expected
    optimizer = _optimizer_groups(actual)

    production._SuccessfulUpdateLrScheduler(
        cast(RuntimeConfig, object()),
        cast(IsolatedAdamW8bit, optimizer),
        lambda _config, _update: expected,
        restored_successful_update=0,
        fresh=False,
    )

    assert torch.equal(actual, before)


def test_restored_scheduler_keeps_python_float_comparison_exact() -> None:
    expected = 2e-8
    production._SuccessfulUpdateLrScheduler(
        cast(RuntimeConfig, object()),
        cast(IsolatedAdamW8bit, _optimizer_groups(expected)),
        lambda _config, _update: expected,
        restored_successful_update=0,
        fresh=False,
    )

    drifted = math.nextafter(expected, math.inf)
    with pytest.raises(ValueError, match="differs from governed schedule"):
        production._SuccessfulUpdateLrScheduler(
            cast(RuntimeConfig, object()),
            cast(IsolatedAdamW8bit, _optimizer_groups(drifted)),
            lambda _config, _update: expected,
            restored_successful_update=0,
            fresh=False,
        )


def test_restored_scheduler_rejects_tensor_dtype_and_value_drift() -> None:
    expected = 2e-8
    with pytest.raises(ValueError, match="dtype must be float32"):
        production._SuccessfulUpdateLrScheduler(
            cast(RuntimeConfig, object()),
            cast(
                IsolatedAdamW8bit,
                _optimizer_groups(torch.tensor(expected, dtype=torch.float64)),
            ),
            lambda _config, _update: expected,
            restored_successful_update=0,
            fresh=False,
        )

    canonical = torch.tensor(expected, dtype=torch.float32)
    drifted = torch.nextafter(canonical, torch.tensor(math.inf, dtype=torch.float32))
    with pytest.raises(ValueError, match="differs from governed schedule"):
        production._SuccessfulUpdateLrScheduler(
            cast(RuntimeConfig, object()),
            cast(IsolatedAdamW8bit, _optimizer_groups(drifted)),
            lambda _config, _update: expected,
            restored_successful_update=0,
            fresh=False,
        )


def test_restored_scheduler_rejects_mixed_and_inconsistent_multiple_groups() -> None:
    expected = 2e-8
    canonical = torch.tensor(expected, dtype=torch.float32)
    with pytest.raises(ValueError, match="representations differ across groups"):
        production._SuccessfulUpdateLrScheduler(
            cast(RuntimeConfig, object()),
            cast(IsolatedAdamW8bit, _optimizer_groups(expected, canonical.clone())),
            lambda _config, _update: expected,
            restored_successful_update=0,
            fresh=False,
        )

    production._SuccessfulUpdateLrScheduler(
        cast(RuntimeConfig, object()),
        cast(
            IsolatedAdamW8bit,
            _optimizer_groups(canonical.clone(), canonical.clone()),
        ),
        lambda _config, _update: expected,
        restored_successful_update=0,
        fresh=False,
    )

    drifted = torch.nextafter(canonical, torch.tensor(math.inf, dtype=torch.float32))
    with pytest.raises(ValueError, match="differs from governed schedule"):
        production._SuccessfulUpdateLrScheduler(
            cast(RuntimeConfig, object()),
            cast(
                IsolatedAdamW8bit,
                _optimizer_groups(canonical.clone(), drifted),
            ),
            lambda _config, _update: expected,
            restored_successful_update=0,
            fresh=False,
        )


def test_s0_schedule_uses_1000_successful_update_linear_warmup_then_stays_fixed() -> None:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            scheduler=SimpleNamespace(
                name="linear_warmup_constant",
                warmup_updates=1000,
                max_lr=2e-5,
                after_warmup="constant",
            ),
            optimizer=SimpleNamespace(lr=2e-5),
        ),
    )

    assert production._s0_linear_warmup_learning_rate(config, 1) == pytest.approx(2e-8)
    assert production._s0_linear_warmup_learning_rate(config, 500) == pytest.approx(
        1e-5
    )
    assert production._s0_linear_warmup_learning_rate(config, 1000) == pytest.approx(
        2e-5
    )
    assert production._s0_linear_warmup_learning_rate(config, 1001) == pytest.approx(
        2e-5
    )


def test_s0_schedule_uses_explicit_toml_warmup_and_max_lr() -> None:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            scheduler=SimpleNamespace(
                name="linear_warmup_constant",
                warmup_updates=4,
                max_lr=1e-5,
                after_warmup="constant",
            ),
            optimizer=SimpleNamespace(lr=1e-5),
        ),
    )

    assert production._s0_linear_warmup_learning_rate(config, 1) == pytest.approx(2.5e-6)
    assert production._s0_linear_warmup_learning_rate(config, 3) == pytest.approx(7.5e-6)
    assert production._s0_linear_warmup_learning_rate(config, 4) == pytest.approx(1e-5)
    assert production._s0_linear_warmup_learning_rate(config, 5) == pytest.approx(1e-5)


def test_metric_context_uses_observed_flops_live_depth_and_update_wall() -> None:
    observation = cast(
        SuccessfulTrainingObservation,
        SimpleNamespace(
            loop=SimpleNamespace(
                update=SimpleNamespace(effective_samples=4),
                update_wall_seconds=2.0,
            ),
            microbatches=(
                SimpleNamespace(dit_flops=7),
                SimpleNamespace(dit_flops=10),
            ),
        ),
    )
    depth_observed: list[bool] = []
    context = production._ProductionMetricContext(
        lambda: depth_observed.append(True) or 3,
    )

    metric = context(observation)

    assert metric.dit_flops == 17
    assert metric.samples_per_second == pytest.approx(2.0)
    assert metric.ready_queue_depth == 3
    assert depth_observed == [True]


def test_resume_sets_trusted_expected_lr_before_raw_loader_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(checkpoint=SimpleNamespace(full_every_updates=1000))
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
                _group_learning_rate(observed_optimizer),
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


def test_resume_rejects_cadence_config_drift_before_optimizer_or_raw_restore(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(checkpoint=SimpleNamespace(full_every_updates=7))
    loaded = _loaded(config)
    optimizer = _optimizer()
    identity = CheckpointIdentity("raw-4", 4, _HASH_A, _HASH_B, _HASH_C)
    state = _state(4)

    def read_state(_path: Path) -> tuple[Any, RawCheckpointState]:
        return SimpleNamespace(identity=identity), state

    monkeypatch.setattr(production, "read_raw_checkpoint_state", read_state)

    def must_not_restore(*_args: object, **_kwargs: object) -> None:
        pytest.fail("cadence drift must fail before RAW restore")

    monkeypatch.setattr(
        production,
        "restore_single_gpu_checkpoint",
        must_not_restore,
    )

    with pytest.raises(ValueError, match="cadence differs from resolved config"):
        production._restore_checkpoint(
            loaded,
            checkpoint=tmp_path,
            dependency_sha256=_HASH_B,
            module=cast(Any, object()),
            optimizer=cast(IsolatedAdamW8bit, optimizer),
            learning_rate_for_update=lambda _config, _update: 0.0,
        )
    assert _group_learning_rate(optimizer) == 2e-5


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
        checkpoint=SimpleNamespace(full_every_updates=13),
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
    assert state.checkpoint_cadence == CheckpointCadence(0, 11.0, 13)
    assert payload == b"resolved\n"


def test_public_lifecycle_reaches_the_concrete_lifecycle_without_readiness_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = cast(RuntimeConfig, object())
    loaded = _loaded(config)
    checkpoint = tmp_path / "checkpoint"
    report = tmp_path / "preflight.json"
    calls: list[tuple[LoadedConfig, Path, Path | None, bool]] = []

    def load(
        _config_path: Path,
        *,
        config_root: Path,
        environment: object | None = None,
    ) -> LoadedConfig:
        del config_root, environment
        return loaded

    def accept(_config: RuntimeConfig) -> None:
        return None

    def run(
        observed: LoadedConfig,
        *,
        repository_root: Path,
        resume: Path | None,
        preflight_only: bool,
        wall_clock: object,
    ) -> production.ProductionTrainingResult:
        assert callable(wall_clock)
        calls.append((observed, repository_root, resume, preflight_only))
        return production.ProductionTrainingResult(
            _HASH_A, report, checkpoint, 0, 0, True
        )

    monkeypatch.setattr(production, "load_config", load)
    monkeypatch.setattr(production, "require_single_gpu_config", accept)
    monkeypatch.setattr(production, "_run_accepted_lifecycle", run)

    result = production.run_production_single_gpu(
        Path("train_s0.toml"),
        config_root=Path("config"),
        repository_root=tmp_path,
        preflight_only=True,
    )

    assert result.preflight_only is True
    assert calls == [(loaded, tmp_path, None, True)]


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


@pytest.mark.parametrize(
    ("update", "terminal", "expected_error"),
    [
        pytest.param(4, 5, None, id="valid"),
        pytest.param(5, 5, "already exhausted", id="exhausted"),
        pytest.param(4, 6, "differs from resolved config", id="wrong-budget"),
    ],
)
def test_resume_checkpoint_binding_precedes_service_and_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    update: int,
    terminal: int,
    expected_error: str | None,
) -> None:
    events: list[str] = []
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            paths=SimpleNamespace(
                checkpoint_dir="checkpoints",
                artifact_dir="artifacts",
            ),
            run=SimpleNamespace(seed=7, intent="train", stage="S0"),
            checkpoint=SimpleNamespace(slots=7, full_every_updates=1000),
            stage=SimpleNamespace(
                enabled=True,
                world_size=1,
                activation_checkpoint_mode="none",
                name="S0",
                resolution=256,
                depth=16,
                planned_updates=5,
            ),
            distributed=SimpleNamespace(backend="native", world_size=1),
            failure=SimpleNamespace(allow_force_bypass=False),
            growth=SimpleNamespace(enabled=False),
            data=SimpleNamespace(
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
        state=_state(update, terminal=terminal),
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
        def from_config(cls, *_args: object, **kwargs: object) -> Factory:
            assert "pass_index" not in kwargs
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
        if expected_error is not None:
            pytest.fail("invalid resume reached data-service construction")
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

    publisher_arguments: dict[str, object] = {}

    def publisher_factory(*_args: object, **kwargs: object) -> object:
        publisher_arguments.update(kwargs)
        return object()

    monkeypatch.setattr(production, "_dependency_sha256", dependency)
    monkeypatch.setattr(
        production,
        "require_static_single_gpu_preflight",
        static_preflight,
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
        publisher_factory,
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

    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            production._run_accepted_lifecycle(
                loaded,
                repository_root=tmp_path,
                resume=resume,
                preflight_only=True,
                wall_clock=lambda: 1.0,
            )
        assert "restore" in events
        assert "service" not in events
        return

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
        update,
        update,
        True,
    )
    assert events.index("restore") < events.index("service")
    assert events.index("static_preflight") < events.index("restore")
    assert events[-1] == "close"
    assert publisher_arguments["retention_slots"] == 7
