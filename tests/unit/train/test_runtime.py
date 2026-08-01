from __future__ import annotations

# pyright: reportPrivateUsage=false
import secrets
from collections.abc import Callable, Iterator
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import sakuramoon.data.production as production_module
import sakuramoon.train.preflight as preflight_module
import sakuramoon.train.runtime as runtime_module
from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.caption import CaptionDropoutCounts
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    ConfiguredDataLoader,
    ProductionBatchStreamIdentity,
)
from sakuramoon.model.growth import BASE_SLOT_IDS, active_slot_ids
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.loop import LoopResult
from sakuramoon.train.preflight import (
    AcceptedPreflight,
    PreflightCheckResult,
    PreflightError,
    PreflightReport,
    RestoredSingleGpuCheckpoint,
)
from sakuramoon.train.runtime import (
    RuntimeMeasurement,
    SingleGpuBatchRuntime,
    SuccessfulTrainingObservation,
    require_single_gpu_config,
    run_single_gpu_training,
)
from sakuramoon.train.step import SingleGpuUpdateState

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


class _SgdAdapter:
    def __init__(self, parameters: Iterator[torch.nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(tuple(parameters), lr=0.01)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


class _CloseableIterator(Iterator[torch.Tensor]):
    def __init__(self, values: tuple[torch.Tensor, ...], *, close_error: bool = False):
        self._values = iter(values)
        self.close_calls = 0
        self.close_error = close_error
        self.next_calls = 0

    def __next__(self) -> torch.Tensor:
        self.next_calls += 1
        return next(self._values)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise OSError("synthetic stream close failure")


def _config(*, planned_updates: int) -> RuntimeConfig:
    return cast(
        RuntimeConfig,
        SimpleNamespace(
            run=SimpleNamespace(intent="train", stage="S0"),
            stage=SimpleNamespace(
                name="S0",
                enabled=True,
                depth=16,
                resolution=256,
                world_size=1,
                accumulation=1,
                activation_checkpoint_mode="none",
                planned_updates=planned_updates,
            ),
            growth=SimpleNamespace(enabled=False),
            distributed=SimpleNamespace(backend="native", world_size=1),
            failure=SimpleNamespace(allow_force_bypass=False),
            checkpoint=SimpleNamespace(full_every_updates=1000),
        ),
    )


def _raw_state(
    state: SingleGpuUpdateState, *, terminal: int, cadence_time: float = 0.0
) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=state,
        growth=GrowthCheckpointState(BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None),
        stage_budget=StageBudgetCheckpointState(0, terminal),
        checkpoint_cadence=CheckpointCadence(state.successful_updates, cadence_time),
    )


def _restored(
    module: torch.nn.Module,
    optimizer: object,
    state: RawCheckpointState,
) -> RestoredSingleGpuCheckpoint:
    identity = CheckpointIdentity(
        "unit",
        state.trainer.successful_updates,
        _HASH_A,
        _HASH_B,
        _HASH_C,
    )
    restored = object.__new__(RestoredSingleGpuCheckpoint)
    restored._manifest = CheckpointManifest(
        CheckpointKind.RAW,
        identity,
        (FileRecord("payload", 1, _HASH_D),),
    )
    restored._module = module
    restored._optimizer = optimizer
    restored._owner_pid = preflight_module.os.getpid()
    restored._path = Path("/test/checkpoint")
    restored._payload_bytes = 1
    restored._state = state
    restored._token = secrets.token_hex(32)
    preflight_module._RESTORED_CHECKPOINTS[restored._token] = restored
    return restored


def _stream(
    iterator: _CloseableIterator,
) -> AcceptedProductionBatchStream:
    identity = ProductionBatchStreamIdentity(
        _HASH_A,
        ConfiguredDataLoader(1, 2, 2, False, True),
        _HASH_B,
        _HASH_C,
        _HASH_D,
    )
    return production_module._issue_batch_stream(
        cast(Iterator[Any], iterator), identity
    )


def _runtime(module: torch.nn.Module) -> SingleGpuBatchRuntime:
    qwen = object()
    vae = object()

    class _Runtime:
        composite = module
        device = torch.device("cpu")
        growth_alpha = 1.0

        def __init__(self) -> None:
            self.qwen = qwen
            self.vae = vae

        @staticmethod
        def measure(
            batch: torch.Tensor, *, phase_timer: PhaseTimer | None = None
        ) -> RuntimeMeasurement:
            del phase_timer
            loss = module(batch).square().reshape(1)
            return RuntimeMeasurement(
                per_sample_loss=loss,
                image_tokens=1,
                text_tokens=1,
                sample_ids=("1",),
                shape_keys=("unit",),
                high_noise_loss_sum=loss.sum(),
                high_noise_sample_count=torch.ones((), dtype=torch.int64),
                low_noise_loss_sum=loss.new_zeros(()),
                low_noise_sample_count=torch.zeros((), dtype=torch.int64),
                timesteps=torch.ones((1,), dtype=torch.float32),
                dropout_hits=CaptionDropoutCounts(*(0 for _ in range(12))),
            )

    return cast(SingleGpuBatchRuntime, _Runtime())


def _unused_checkpoint_callback(
    _state: SingleGpuUpdateState,
    _reason: CheckpointReason,
    _cadence: CheckpointCadence,
) -> Path:
    return Path("unused")


class _CheckpointPublisher:
    def __init__(
        self,
        update: Callable[
            [SingleGpuUpdateState, CheckpointReason, CheckpointCadence], Path
        ],
    ) -> None:
        self._update = update
        self.retained: list[tuple[Path, CheckpointManifest, RawCheckpointState]] = []

    def publish_preflight(
        self, identity: CheckpointIdentity, state: RawCheckpointState
    ) -> Path:
        del identity, state
        return Path("unused-preflight")

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        return self._update(state, reason, cadence)

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: CheckpointManifest,
        state: RawCheckpointState,
    ) -> None:
        self.retained.append((checkpoint, manifest, state))

    def discard_preflight(self, checkpoint: Path) -> None:
        del checkpoint


_UNUSED_CHECKPOINT_PUBLISHER = _CheckpointPublisher(_unused_checkpoint_callback)


def _accepted(
    config: RuntimeConfig,
    stream: AcceptedProductionBatchStream,
    runtime: SingleGpuBatchRuntime,
    module: torch.nn.Module,
    optimizer: object,
    restored: RestoredSingleGpuCheckpoint,
    checkpoint_publisher: object = _UNUSED_CHECKPOINT_PUBLISHER,
) -> AcceptedPreflight:
    bindings = preflight_module._PreflightBindings(
        config=config,
        resolved_config_sha256=_HASH_A,
        batches=stream,
        runtime=runtime,
        qwen=runtime.qwen,
        vae=runtime.vae,
        module=module,
        optimizer=optimizer,
        restored=restored,
        checkpoint_publisher=checkpoint_publisher,
    )
    report = PreflightReport(
        2,
        "1GPU",
        True,
        _HASH_A,
        _HASH_B,
        _HASH_C,
        restored.manifest.identity.checkpoint_id,
        restored.state.trainer.successful_updates,
        (PreflightCheckResult("resolved_config", True, None),),
    )
    return preflight_module._accepted_preflight(report, bindings)


def _run(
    tmp_path: Path,
    *,
    config: RuntimeConfig,
    stream: AcceptedProductionBatchStream,
    runtime: SingleGpuBatchRuntime,
    module: torch.nn.Module,
    optimizer: _SgdAdapter,
    restored: RestoredSingleGpuCheckpoint,
    accepted: AcceptedPreflight,
    checkpoint_publisher: _CheckpointPublisher | None = None,
    observer: Callable[[SuccessfulTrainingObservation], None] | None = None,
    forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    clock: Callable[[], float] | None = None,
) -> LoopResult:
    def discard_observation(_observation: SuccessfulTrainingObservation) -> None:
        return None

    publisher = checkpoint_publisher or _UNUSED_CHECKPOINT_PUBLISHER
    return runtime_module._run_single_gpu_training(
        config,
        preflight=accepted,
        runtime=runtime,
        module=module,
        optimizer=optimizer,
        batches=stream,
        scheduler_step=lambda _update: None,
        checkpoint_publisher=publisher,
        diagnostic_root=tmp_path / "diagnostics",
        failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
        restored_checkpoint=restored,
        phase_timer=PhaseTimer(device=torch.device("cpu")),
        successful_update_observer=observer or discard_observation,
        forced_checkpoint=forced_checkpoint,
        clock=clock,
    )


def test_single_gpu_config_requires_native_s0() -> None:
    require_single_gpu_config(_config(planned_updates=1))
    changed = SimpleNamespace(
        run=SimpleNamespace(intent="train", stage="S1"),
        stage=SimpleNamespace(world_size=4),
        distributed=SimpleNamespace(backend="ddp", world_size=4),
        failure=SimpleNamespace(allow_force_bypass=False),
    )
    with pytest.raises(ValueError, match="topology"):
        require_single_gpu_config(cast(RuntimeConfig, changed))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_runtime_requires_checkpointed_default_cuda_generator() -> None:
    with pytest.raises(ValueError, match="default CUDA generator"):
        SingleGpuBatchRuntime(
            qwen=cast(Any, object()),
            vae=cast(Any, object()),
            composite=cast(Any, torch.nn.Identity()),
            device=torch.device("cuda", 0),
            generator=torch.Generator(device="cuda").manual_seed(1),
            p_mean=-0.8,
            p_std=0.8,
            noise_scale=1.0,
            t_eps=0.05,
            noise_observation_boundary=0.95,
            growth_alpha=0.0,
        )


def test_training_boundary_requires_d025_and_restored_checkpoint() -> None:
    parameters = signature(run_single_gpu_training).parameters
    assert parameters["batches"].annotation == "AcceptedProductionBatchStream"
    assert parameters["checkpoint_publisher"].annotation == (
        "ProductionSingleGpuCheckpointPublisher"
    )
    assert "restored_checkpoint" in parameters
    assert "state" not in parameters
    assert "stage_budget" not in parameters
    assert "cadence" not in parameters


def test_public_training_rejects_custom_publisher_and_closes_stream(
    tmp_path: Path,
) -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored = _restored(
        module,
        optimizer,
        _raw_state(SingleGpuUpdateState.initial(), terminal=1),
    )
    custom = _CheckpointPublisher(_unused_checkpoint_callback)
    accepted = _accepted(
        _config(planned_updates=1),
        stream,
        runtime,
        module,
        optimizer,
        restored,
        custom,
    )

    with pytest.raises(TypeError, match="ProductionSingleGpuCheckpointPublisher"):
        run_single_gpu_training(
            _config(planned_updates=1),
            preflight=accepted,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            batches=stream,
            scheduler_step=lambda _update: None,
            checkpoint_publisher=cast(Any, custom),
            diagnostic_root=tmp_path / "diagnostics",
            failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
            restored_checkpoint=restored,
            phase_timer=PhaseTimer(device=torch.device("cpu")),
            successful_update_observer=lambda _observation: None,
        )
    assert iterator.close_calls == 1


def test_public_training_preserves_publisher_rejection_and_stream_close_failure(
    tmp_path: Path,
) -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),), close_error=True)
    stream = _stream(iterator)
    restored = _restored(
        module,
        optimizer,
        _raw_state(SingleGpuUpdateState.initial(), terminal=1),
    )
    custom = _CheckpointPublisher(_unused_checkpoint_callback)
    config = _config(planned_updates=1)
    accepted = _accepted(
        config, stream, runtime, module, optimizer, restored, custom
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        run_single_gpu_training(
            config,
            preflight=accepted,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            batches=stream,
            scheduler_step=lambda _update: None,
            checkpoint_publisher=cast(Any, custom),
            diagnostic_root=tmp_path / "diagnostics",
            failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
            restored_checkpoint=restored,
            phase_timer=PhaseTimer(device=torch.device("cpu")),
            successful_update_observer=lambda _observation: None,
        )
    assert [type(item) for item in captured.value.exceptions] == [TypeError, OSError]
    assert iterator.close_calls == 1


def test_optimizer_learning_rate_accepts_torchao_scalar_tensor() -> None:
    optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(
            param_groups=[
                {"lr": torch.tensor(2e-5)},
                {"lr": torch.tensor(2e-5)},
            ]
        )
    )
    assert runtime_module._optimizer_learning_rate(
        cast(Any, optimizer)
    ) == pytest.approx(2e-5)


@pytest.mark.parametrize(
    "rate",
    [torch.tensor([2e-5]), torch.tensor(float("nan")), 2, "2e-5"],
)
def test_optimizer_learning_rate_rejects_invalid_runtime_value(rate: object) -> None:
    optimizer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": rate}]))
    with pytest.raises(ValueError, match="learning rate"):
        runtime_module._optimizer_learning_rate(cast(Any, optimizer))


def test_forged_preflight_is_rejected_and_stream_is_closed(tmp_path: Path) -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored = _restored(
        module, optimizer, _raw_state(SingleGpuUpdateState.initial(), terminal=1)
    )

    with pytest.raises(PreflightError, match="process-local"):
        _run(
            tmp_path,
            config=_config(planned_updates=1),
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=object.__new__(AcceptedPreflight),
        )
    assert iterator.close_calls == 1


def test_mid_stage_state_comes_only_from_restored_raw_and_stream_closes_once(
    tmp_path: Path,
) -> None:
    config = _config(planned_updates=5)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1), torch.ones(1, 1)))
    stream = _stream(iterator)
    restored = _restored(
        module,
        optimizer,
        _raw_state(SingleGpuUpdateState(3, 3, 3), terminal=5),
    )
    observations: list[SuccessfulTrainingObservation] = []

    result = _run(
        tmp_path,
        config=config,
        stream=stream,
        runtime=runtime,
        module=module,
        optimizer=optimizer,
        restored=restored,
        accepted=_accepted(config, stream, runtime, module, optimizer, restored),
        observer=observations.append,
        clock=lambda: 1.0,
    )

    assert result.state == SingleGpuUpdateState(5, 5, 5)
    assert iterator.close_calls == 1
    assert [item.loop.update.state.successful_updates for item in observations] == [
        4,
        5,
    ]
    required = {"backward", "clip", "optimizer", "zero_grad"}
    assert required <= observations[-1].phase_timer.recorded_phases
    assert "data" not in observations[-1].phase_timer.recorded_phases
    assert "checkpoint" not in observations[-1].phase_timer.recorded_phases
    assert all(item.loop.data_wait_seconds >= 0.0 for item in observations)
    assert all(item.loop.checkpoint_seconds == 0.0 for item in observations)
    assert observations[0].phase_timer is not observations[1].phase_timer
    assert all(
        not measurement.per_sample_loss.requires_grad
        and measurement.per_sample_loss.grad_fn is None
        for observation in observations
        for measurement in observation.microbatches
    )


def test_stage_budget_drift_fails_before_consuming_batch_and_closes_stream(
    tmp_path: Path,
) -> None:
    config = _config(planned_updates=5)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored_state = _raw_state(SingleGpuUpdateState.initial(), terminal=6)
    restored = _restored(module, optimizer, restored_state)

    with pytest.raises(ValueError, match="stage budget differs"):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=_accepted(
                config, stream, runtime, module, optimizer, restored
            ),
        )
    assert iterator.next_calls == 0
    assert iterator.close_calls == 1


def test_disabled_s0_fails_before_consuming_batch_and_closes_stream(
    tmp_path: Path,
) -> None:
    config = _config(planned_updates=1)
    config.stage.enabled = False
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored = _restored(
        module,
        optimizer,
        _raw_state(SingleGpuUpdateState.initial(), terminal=1),
    )

    with pytest.raises(ValueError, match="enabled S0"):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=_accepted(
                config, stream, runtime, module, optimizer, restored
            ),
        )
    assert iterator.next_calls == 0
    assert iterator.close_calls == 1


def test_nonzero_s0_budget_origin_fails_before_consuming_batch_and_closes_stream(
    tmp_path: Path,
) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored_state = RawCheckpointState(
        trainer=SingleGpuUpdateState(5, 5, 5),
        growth=GrowthCheckpointState(
            BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None
        ),
        stage_budget=StageBudgetCheckpointState(5, 6),
        checkpoint_cadence=CheckpointCadence(5, 0.0),
    )
    restored = _restored(module, optimizer, restored_state)

    with pytest.raises(ValueError, match="must start at update zero"):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=_accepted(
                config, stream, runtime, module, optimizer, restored
            ),
        )
    assert iterator.next_calls == 0
    assert iterator.close_calls == 1


@pytest.mark.parametrize(
    ("growth", "runtime_growth_alpha", "planned_updates", "message"),
    [
        pytest.param(
            GrowthCheckpointState(
                BASE_SLOT_IDS, 1.0, "S1", 1, 256, None, None
            ),
            1.0,
            1,
            "checkpoint axes differ",
            id="stage",
        ),
        pytest.param(
            GrowthCheckpointState(
                BASE_SLOT_IDS, 1.0, "S0", 2, 256, None, None
            ),
            1.0,
            1,
            "checkpoint axes differ",
            id="world-size",
        ),
        pytest.param(
            GrowthCheckpointState(
                BASE_SLOT_IDS, 1.0, "S0", 1, 512, None, None
            ),
            1.0,
            1,
            "checkpoint axes differ",
            id="resolution",
        ),
        pytest.param(
            GrowthCheckpointState(
                active_slot_ids(20), 1.0, "S0", 1, 256, None, None
            ),
            1.0,
            1,
            "checkpoint slots differ",
            id="active-slot-ids",
        ),
        pytest.param(
            GrowthCheckpointState(BASE_SLOT_IDS, 0.0, "S0", 1, 256, 0, 1000),
            0.0,
            1000,
            "checkpoint ramp presence differs",
            id="ramp-presence",
        ),
        pytest.param(
            GrowthCheckpointState(
                BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None
            ),
            0.5,
            1,
            "runtime growth alpha differs",
            id="alpha",
        ),
    ],
)
def test_checkpoint_binding_drift_fails_before_consuming_batch_and_closes_stream(
    tmp_path: Path,
    growth: GrowthCheckpointState,
    runtime_growth_alpha: float,
    planned_updates: int,
    message: str,
) -> None:
    config = _config(planned_updates=planned_updates)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    runtime.growth_alpha = runtime_growth_alpha
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    baseline = _raw_state(
        SingleGpuUpdateState.initial(), terminal=planned_updates
    )
    restored_state = RawCheckpointState(
        baseline.trainer,
        growth,
        baseline.stage_budget,
        baseline.checkpoint_cadence,
    )
    restored = _restored(module, optimizer, restored_state)

    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=_accepted(
                config, stream, runtime, module, optimizer, restored
            ),
        )
    assert iterator.next_calls == 0
    assert iterator.close_calls == 1


@pytest.mark.parametrize(
    "binding",
    ["config", "stream", "runtime", "qwen", "vae", "module", "optimizer", "restored"],
)
def test_accepted_preflight_rejects_cross_resource_reuse(
    tmp_path: Path, binding: str
) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored = _restored(
        module, optimizer, _raw_state(SingleGpuUpdateState.initial(), terminal=1)
    )
    accepted = _accepted(config, stream, runtime, module, optimizer, restored)
    run_config, run_stream, run_runtime, run_module = config, stream, runtime, module
    run_optimizer = optimizer
    run_restored = restored
    if binding == "config":
        run_config = _config(planned_updates=1)
    elif binding == "stream":
        run_stream = _stream(_CloseableIterator((torch.ones(1, 1),)))
    elif binding == "runtime":
        run_runtime = _runtime(module)
        run_runtime.qwen = runtime.qwen
        run_runtime.vae = runtime.vae
    elif binding in {"qwen", "vae"}:
        setattr(runtime, binding, object())
    elif binding == "module":
        run_module = torch.nn.Linear(1, 1, bias=False)
    elif binding == "optimizer":
        run_optimizer = _SgdAdapter(iter(module.parameters()))
    elif binding == "restored":
        run_restored = _restored(
            module,
            optimizer,
            _raw_state(SingleGpuUpdateState.initial(), terminal=1),
        )

    with pytest.raises(PreflightError, match="process-local"):
        _run(
            tmp_path,
            config=run_config,
            stream=run_stream,
            runtime=run_runtime,
            module=run_module,
            optimizer=run_optimizer,
            restored=run_restored,
            accepted=accepted,
        )


def test_accepted_preflight_rejects_cross_publisher_reuse(tmp_path: Path) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored = _restored(
        module,
        optimizer,
        _raw_state(SingleGpuUpdateState.initial(), terminal=1),
    )
    accepted = _accepted(config, stream, runtime, module, optimizer, restored)

    def different_publisher_callback(
        _state: SingleGpuUpdateState,
        _reason: CheckpointReason,
        _cadence: CheckpointCadence,
    ) -> Path:
        return tmp_path / "different"

    different_publisher = _CheckpointPublisher(different_publisher_callback)

    with pytest.raises(PreflightError, match="process-local"):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=accepted,
            checkpoint_publisher=different_publisher,
        )
    assert iterator.close_calls == 1


def test_primary_diagnostic_and_stream_close_failures_are_all_preserved(
    tmp_path: Path,
) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.tensor([[float("nan")]]),), close_error=True)
    stream = _stream(iterator)
    restored = _restored(
        module, optimizer, _raw_state(SingleGpuUpdateState.initial(), terminal=1)
    )
    accepted = _accepted(config, stream, runtime, module, optimizer, restored)
    diagnostic = tmp_path / "diagnostics" / "update-1"
    diagnostic.mkdir(parents=True)

    with pytest.raises(BaseExceptionGroup) as captured:
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=accepted,
        )

    leaf_types: list[type[BaseException]] = []

    def collect(error: BaseException) -> None:
        if isinstance(error, BaseExceptionGroup):
            for child in cast(
                tuple[BaseException, ...],
                error.exceptions,  # pyright: ignore[reportUnknownMemberType]
            ):
                collect(child)
        else:
            leaf_types.append(type(error))

    collect(captured.value)
    assert leaf_types == [FloatingPointError, FileExistsError, OSError]
    assert iterator.close_calls == 1


def test_due_checkpoint_is_read_back_and_exact_state_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored_state = _raw_state(SingleGpuUpdateState.initial(), terminal=1)
    restored = _restored(module, optimizer, restored_state)
    published = tmp_path / "ckpt_1_published"
    calls: list[tuple[SingleGpuUpdateState, CheckpointReason, CheckpointCadence]] = []

    def publisher_callback(
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        calls.append((state, reason, cadence))
        return published

    publisher = _CheckpointPublisher(publisher_callback)

    accepted = _accepted(
        config, stream, runtime, module, optimizer, restored, publisher
    )

    def read_back(path: Path) -> tuple[CheckpointManifest, RawCheckpointState]:
        assert path == published
        state, _reason, cadence = calls[-1]
        manifest = CheckpointManifest(
            CheckpointKind.RAW,
            CheckpointIdentity("published", 1, _HASH_A, _HASH_B, _HASH_C),
            (FileRecord("payload", 1, _HASH_D),),
        )
        return manifest, RawCheckpointState(
            state,
            restored_state.growth,
            restored_state.stage_budget,
            cadence,
        )

    monkeypatch.setattr(runtime_module, "read_raw_checkpoint_state", read_back)
    result = _run(
        tmp_path,
        config=config,
        stream=stream,
        runtime=runtime,
        module=module,
        optimizer=optimizer,
        restored=restored,
        accepted=accepted,
        checkpoint_publisher=publisher,
        forced_checkpoint=lambda _update: CheckpointReason.STAGE_FINALIZE,
        clock=lambda: 1.0,
    )

    assert result.checkpoint_updates == (1,)
    assert calls == [
        (
            SingleGpuUpdateState(1, 1, 1),
            CheckpointReason.STAGE_FINALIZE,
            CheckpointCadence(1, 1.0),
        )
    ]
    assert len(publisher.retained) == 1
    retained_path, retained_manifest, retained_state = publisher.retained[0]
    assert retained_path == published
    assert retained_manifest.identity.update == 1
    assert retained_state.trainer == SingleGpuUpdateState(1, 1, 1)


@pytest.mark.parametrize("mode", ["checksum", "identity", "state"])
def test_due_checkpoint_rejects_unusable_or_mismatched_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored_state = _raw_state(SingleGpuUpdateState.initial(), terminal=1)
    restored = _restored(module, optimizer, restored_state)
    published = tmp_path / "published"

    def publisher_callback(
        _state: SingleGpuUpdateState,
        _reason: CheckpointReason,
        _cadence: CheckpointCadence,
    ) -> Path:
        return published

    publisher = _CheckpointPublisher(publisher_callback)

    accepted = _accepted(
        config, stream, runtime, module, optimizer, restored, publisher
    )

    def read_back(_path: Path) -> tuple[CheckpointManifest, RawCheckpointState]:
        if mode == "checksum":
            raise CheckpointError("checkpoint payload checksum failed")
        manifest = CheckpointManifest(
            CheckpointKind.RAW,
            CheckpointIdentity(
                "published",
                1,
                _HASH_A,
                _HASH_D if mode == "identity" else _HASH_B,
                _HASH_C,
            ),
            (FileRecord("payload", 1, _HASH_D),),
        )
        state = RawCheckpointState(
            SingleGpuUpdateState(2, 1, 1)
            if mode == "state"
            else SingleGpuUpdateState(1, 1, 1),
            restored_state.growth,
            restored_state.stage_budget,
            CheckpointCadence(1, 1.0),
        )
        return manifest, state

    monkeypatch.setattr(runtime_module, "read_raw_checkpoint_state", read_back)
    expected = CheckpointError if mode == "checksum" else ValueError
    with pytest.raises(expected):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=accepted,
            checkpoint_publisher=publisher,
            forced_checkpoint=lambda _update: CheckpointReason.STAGE_FINALIZE,
            clock=lambda: 1.0,
        )
    assert publisher.retained == []
    assert iterator.close_calls == 1
    assert (tmp_path / "diagnostics" / "post_update-1" / "COMPLETE").is_file()


def test_successful_observer_failure_is_fail_closed_and_closes_stream(
    tmp_path: Path,
) -> None:
    config = _config(planned_updates=1)
    module = torch.nn.Linear(1, 1, bias=False)
    optimizer = _SgdAdapter(iter(module.parameters()))
    runtime = _runtime(module)
    iterator = _CloseableIterator((torch.ones(1, 1),))
    stream = _stream(iterator)
    restored = _restored(
        module,
        optimizer,
        _raw_state(SingleGpuUpdateState.initial(), terminal=1),
    )
    accepted = _accepted(config, stream, runtime, module, optimizer, restored)

    def fail_observer(_observation: SuccessfulTrainingObservation) -> None:
        raise OSError("observer failed")

    with pytest.raises(OSError, match="observer failed"):
        _run(
            tmp_path,
            config=config,
            stream=stream,
            runtime=runtime,
            module=module,
            optimizer=optimizer,
            restored=restored,
            accepted=accepted,
            observer=fail_observer,
            clock=lambda: 1.0,
        )
    assert iterator.close_calls == 1
    assert (tmp_path / "diagnostics" / "post_update-1" / "COMPLETE").is_file()
