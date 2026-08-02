"""Fail-closed production lifecycle for the governed S0 single-GPU trainer."""

from __future__ import annotations

import hashlib
import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import torch

from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import (
    S0_GOVERNED_SEMANTIC_BLOCKERS,
    S0_RUNTIME_INTEGRATION_BLOCKERS,
    ConfigurationError,
    LoadedConfig,
    load_config,
)
from sakuramoon.config.assembly import (
    build_trainable_composite_from_config,
    build_training_telemetry_from_config,
)
from sakuramoon.config.resolve import write_resolved_config
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.production import ProductionPipelineFactory
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity
from sakuramoon.encoders.mage_vae import FrozenMageVAE, load_local_mage_vae
from sakuramoon.encoders.qwen import QwenRuntime, load_local_qwen
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit
from sakuramoon.storage import repository_directory
from sakuramoon.telemetry.observer import UpdateMetricContext
from sakuramoon.train.preflight import (
    ProductionSingleGpuCheckpointPublisher,
    RestoredSingleGpuCheckpoint,
    build_single_gpu_preflight_checks,
    build_single_gpu_preflight_workload,
    require_static_single_gpu_preflight,
    restore_single_gpu_checkpoint,
    run_single_gpu_preflight,
)
from sakuramoon.train.runtime import (
    SingleGpuBatchRuntime,
    SuccessfulTrainingObservation,
    require_single_gpu_config,
    run_single_gpu_training,
)
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

_DEPENDENCY_LOCK = "uv.lock"
_READINESS_BLOCKERS = tuple(
    blocker.code
    for blocker in (
        *S0_GOVERNED_SEMANTIC_BLOCKERS,
        *S0_RUNTIME_INTEGRATION_BLOCKERS,
    )
)


class ProductionReadinessError(RuntimeError):
    """Canonical runtime bindings are absent and production must not guess them."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        if (
            type(blockers) is not tuple
            or not blockers
            or any(type(item) is not str or not item for item in blockers)
            or len(set(blockers)) != len(blockers)
        ):
            raise TypeError("production readiness blockers are invalid")
        self.blockers = blockers
        super().__init__("production runtime has unresolved governed bindings")


class ProductionPreflightError(RuntimeError):
    """Resource assembly or mandatory production preflight failed."""


class ProductionTrainingError(RuntimeError):
    """The accepted production loop or its owned cleanup failed."""


@dataclass(frozen=True, slots=True)
class ProductionTrainingResult:
    resolved_config_sha256: str
    preflight_report: Path
    checkpoint_path: Path
    initial_successful_update: int
    final_successful_update: int
    preflight_only: bool

    def __post_init__(self) -> None:
        if (
            len(self.resolved_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.resolved_config_sha256
            )
            or not self.preflight_report.is_absolute()
            or not self.checkpoint_path.is_absolute()
            or type(self.initial_successful_update) is not int
            or type(self.final_successful_update) is not int
            or self.initial_successful_update < 0
            or self.final_successful_update < self.initial_successful_update
            or type(self.preflight_only) is not bool
        ):
            raise ValueError("production training result is invalid")


LearningRateForUpdate = Callable[[RuntimeConfig, int], float]
DitFlopsObserver = Callable[[SuccessfulTrainingObservation], int]
ReadyQueueDepthObserver = Callable[[SuccessfulTrainingObservation], int]


@dataclass(frozen=True, slots=True)
class _GovernedRuntimeBindings:
    """Bindings that must eventually be issued from canonical config/data APIs."""

    pass_index: int
    learning_rate_for_update: LearningRateForUpdate
    dit_flops: DitFlopsObserver
    ready_queue_depth: ReadyQueueDepthObserver

    def __post_init__(self) -> None:
        if type(self.pass_index) is not int or self.pass_index < 0:
            raise ValueError("governed data pass identity is invalid")
        if not callable(self.learning_rate_for_update):
            raise TypeError("governed WSD schedule must be callable")
        if not callable(self.dit_flops):
            raise TypeError("actual DiT FLOPs observer must be callable")
        if not callable(self.ready_queue_depth):
            raise TypeError("governed ready-queue observer must be callable")


def _resolve_governed_runtime_bindings(
    _config: RuntimeConfig,
) -> _GovernedRuntimeBindings:
    """Fail until current sources bind all three semantics without inference."""

    raise ProductionReadinessError(_READINESS_BLOCKERS)


def _raise_preserving(
    message: str, primary: BaseException, cleanup: BaseException | None
) -> NoReturn:
    if cleanup is not None:
        raise BaseExceptionGroup(message, [primary, cleanup]) from None
    raise primary


def _repository_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("repository root must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("repository root may not contain symbolic links")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("repository root does not exist") from error
    if resolved != path or not resolved.is_dir():
        raise ValueError("repository root must be a real directory")
    return resolved


def _config_root(repository_root: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else repository_root / configured


def _dependency_sha256(repository_root: Path) -> str:
    dependency = repository_root / _DEPENDENCY_LOCK
    if dependency.is_symlink() or not dependency.is_file():
        raise ValueError("tracked dependency lock identity is unavailable")
    payload = dependency.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(digest) != 64:
        raise RuntimeError("dependency lock digest is invalid")
    return digest


def _require_exact_resume_path(checkpoint: Path) -> Path:
    if not checkpoint.is_absolute():
        raise ConfigurationError(
            "resume must name an exact absolute raw COMPLETE checkpoint directory"
        )
    current = Path(checkpoint.anchor)
    for part in checkpoint.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ConfigurationError("resume path may not contain symbolic links")
    try:
        resolved = checkpoint.resolve(strict=True)
    except OSError as error:
        raise ConfigurationError("resume checkpoint does not exist") from error
    marker = resolved / "COMPLETE"
    if (
        resolved != checkpoint
        or not resolved.is_dir()
        or marker.is_symlink()
        or not marker.is_file()
    ):
        raise ConfigurationError(
            "resume must name an exact absolute raw COMPLETE checkpoint directory"
        )
    try:
        complete = marker.read_bytes()
    except OSError as error:
        raise ConfigurationError("resume COMPLETE marker is unreadable") from error
    if complete != b"complete\n":
        raise ConfigurationError("resume COMPLETE marker is invalid")
    return resolved


def _publish_resolved_config(loaded: LoadedConfig, repository_root: Path) -> Path:
    run_root = repository_directory(repository_root, loaded.config.paths.run_dir)
    destination = run_root / f"resolved-{loaded.resolved_sha256}.toml"
    expected = loaded.resolved_toml.encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("resolved config artifact path is invalid")
        if destination.read_bytes() != expected:
            raise ValueError("existing resolved config artifact differs from identity")
        return destination.resolve(strict=True)
    observed = write_resolved_config(loaded.config, destination)
    if observed != loaded.resolved_sha256 or destination.read_bytes() != expected:
        raise RuntimeError("resolved config publication identity changed")
    return destination.resolve(strict=True)


def _build_optimizer(
    config: RuntimeConfig, module: TrainableComposite
) -> IsolatedAdamW8bit:
    optimizer = config.optimizer
    return build_adamw8bit(
        module,
        lr=optimizer.lr,
        betas=optimizer.betas,
        eps=optimizer.eps,
        block_size=optimizer.block_size,
        bf16_stochastic_round=optimizer.bf16_stochastic_round,
        matrix_weight_decay=optimizer.matrix_weight_decay,
        sensitive_weight_decay=optimizer.sensitive_weight_decay,
        sr_seed=config.run.seed,
    )


def _optimizer_learning_rate(optimizer: IsolatedAdamW8bit) -> float:
    groups = optimizer.optimizer.param_groups
    rates: set[float] = set()
    for group in groups:
        raw = group.get("lr")
        if type(raw) is float:
            value = raw
        elif (
            type(raw) is torch.Tensor and raw.ndim == 0 and raw.dtype.is_floating_point
        ):
            value = float(raw.detach().item())
        else:
            raise ValueError("production optimizer learning rate is invalid")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("production optimizer learning rate is invalid")
        rates.add(value)
    if len(rates) != 1:
        raise ValueError("production optimizer learning rates differ across groups")
    return rates.pop()


def _set_optimizer_learning_rate(
    optimizer: IsolatedAdamW8bit, learning_rate: float
) -> None:
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate < 0.0
    ):
        raise ValueError("governed learning rate must be a finite nonnegative float")
    for group in optimizer.optimizer.param_groups:
        current = group.get("lr")
        if type(current) is float:
            group["lr"] = learning_rate
        elif (
            type(current) is torch.Tensor
            and current.ndim == 0
            and current.dtype.is_floating_point
        ):
            with torch.no_grad():
                current.fill_(learning_rate)
        else:
            raise ValueError("production optimizer learning rate is invalid")


class _SuccessfulUpdateLrScheduler:
    """Apply an externally governed WSD curve only at successful-update edges."""

    def __init__(
        self,
        config: RuntimeConfig,
        optimizer: IsolatedAdamW8bit,
        learning_rate_for_update: LearningRateForUpdate,
        *,
        restored_successful_update: int,
        fresh: bool,
    ) -> None:
        if (
            type(restored_successful_update) is not int
            or restored_successful_update < 0
        ):
            raise ValueError("restored scheduler update is invalid")
        self.config = config
        self.optimizer = optimizer
        self.learning_rate_for_update = learning_rate_for_update
        self.last_successful_update = restored_successful_update
        expected = self._rate(restored_successful_update + 1)
        if fresh:
            _set_optimizer_learning_rate(optimizer, expected)
        elif _optimizer_learning_rate(optimizer) != expected:
            raise ValueError(
                "restored optimizer learning rate differs from governed WSD state"
            )

    def _rate(self, update: int) -> float:
        if type(update) is not int or update <= 0:
            raise ValueError("scheduled update must be positive")
        rate = self.learning_rate_for_update(self.config, update)
        if type(rate) is not float or not math.isfinite(rate) or rate < 0.0:
            raise ValueError("governed WSD schedule returned an invalid rate")
        return rate

    def __call__(self, successful_update: int) -> None:
        if successful_update != self.last_successful_update + 1:
            raise ValueError("WSD scheduler updates must be consecutive")
        _set_optimizer_learning_rate(self.optimizer, self._rate(successful_update + 1))
        self.last_successful_update = successful_update


def _initial_raw_state(
    config: RuntimeConfig, *, wall_clock: float
) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState.initial(),
        growth=GrowthCheckpointState(
            active_slot_ids(config.stage.depth),
            1.0,
            config.stage.name,
            config.stage.world_size,
            config.stage.resolution,
            None,
            None,
        ),
        stage_budget=StageBudgetCheckpointState(0, config.stage.planned_updates),
        checkpoint_cadence=CheckpointCadence(0, wall_clock),
    )


def _bootstrap_checkpoint(
    loaded: LoadedConfig,
    *,
    checkpoint_root: Path,
    dependency_sha256: str,
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    wall_clock: float,
) -> RestoredSingleGpuCheckpoint:
    if any(checkpoint_root.iterdir()):
        raise ConfigurationError(
            "fresh start requires an empty configured checkpoint directory"
        )
    checkpoint_id = (
        "bootstrap-"
        + hashlib.sha256(loaded.config.run.run_id.encode("utf-8")).hexdigest()[:16]
    )
    identity = CheckpointIdentity(
        checkpoint_id,
        0,
        loaded.resolved_sha256,
        dependency_sha256,
        optimizer.audit.schema_sha256,
    )
    saved = save_raw_checkpoint(
        checkpoint_root,
        identity,
        module,
        optimizer,
        _initial_raw_state(loaded.config, wall_clock=wall_clock),
        resolved_config=loaded.resolved_toml.encode("utf-8"),
    )
    return restore_single_gpu_checkpoint(saved.path, module, optimizer, identity)


def _restore_checkpoint(
    loaded: LoadedConfig,
    *,
    checkpoint: Path,
    dependency_sha256: str,
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    learning_rate_for_update: LearningRateForUpdate,
) -> RestoredSingleGpuCheckpoint:
    manifest, state = read_raw_checkpoint_state(checkpoint)
    identity = manifest.identity
    expected = CheckpointIdentity(
        identity.checkpoint_id,
        identity.update,
        loaded.resolved_sha256,
        dependency_sha256,
        optimizer.audit.schema_sha256,
    )
    if identity != expected:
        raise ValueError("resume checkpoint identity differs from the current runtime")
    if state.trainer.successful_updates != identity.update:
        raise ValueError("resume checkpoint update differs from trainer state")
    expected_rate = learning_rate_for_update(
        loaded.config, state.trainer.successful_updates + 1
    )
    if (
        type(expected_rate) is not float
        or not math.isfinite(expected_rate)
        or expected_rate < 0.0
    ):
        raise ValueError("governed WSD schedule returned an invalid resume rate")
    # The RAW loader compares every optimizer group field before mutation. Setting
    # the trusted expected rate here makes checkpoint LR drift a hard load failure.
    _set_optimizer_learning_rate(optimizer, expected_rate)
    return restore_single_gpu_checkpoint(checkpoint, module, optimizer, expected)


def _runtime(
    config: RuntimeConfig,
    *,
    qwen: QwenRuntime,
    vae: FrozenMageVAE,
    module: TrainableComposite,
    restored: RestoredSingleGpuCheckpoint,
    device: torch.device,
) -> SingleGpuBatchRuntime:
    index = int(device.index or 0)
    return SingleGpuBatchRuntime(
        qwen=qwen.encoder,
        vae=vae,
        composite=module,
        device=device,
        generator=torch.cuda.default_generators[index],
        p_mean=config.timestep.p_mean,
        p_std=config.timestep.p_std,
        noise_scale=config.timestep.noise_scale,
        t_eps=config.timestep.t_eps,
        noise_observation_boundary=config.logging.noise_observation_boundary,
        growth_alpha=restored.state.growth.alpha,
    )


class _ProductionMetricContext:
    def __init__(
        self,
        dit_flops: DitFlopsObserver,
        ready_queue_depth: ReadyQueueDepthObserver,
    ) -> None:
        if not callable(dit_flops):
            raise TypeError("actual DiT FLOPs observer must be callable")
        if not callable(ready_queue_depth):
            raise TypeError("live ready-queue observer must be callable")
        self.dit_flops = dit_flops
        self.ready_queue_depth = ready_queue_depth

    def __call__(
        self, observation: SuccessfulTrainingObservation
    ) -> UpdateMetricContext:
        dit_flops = self.dit_flops(observation)
        if type(dit_flops) is not int or dit_flops <= 0:
            raise ValueError("DiT FLOPs observer returned an invalid measurement")
        depth = self.ready_queue_depth(observation)
        if type(depth) is not int or depth < 0:
            raise ValueError("ready-queue observer returned an invalid depth")
        return UpdateMetricContext(
            dit_flops=dit_flops,
            samples_per_second=(
                observation.loop.update.effective_samples
                / observation.loop.update_wall_seconds
            ),
            ready_queue_depth=depth,
            supplemental_phase_seconds={},
        )


def _reject_sample(_reason: str) -> None:
    # D025 owns rejection policy; the training process does not mutate service state.
    return None


def _run_accepted_lifecycle(
    loaded: LoadedConfig,
    *,
    repository_root: Path,
    resume: Path | None,
    preflight_only: bool,
    wall_clock: Callable[[], float],
) -> ProductionTrainingResult:
    config = loaded.config
    require_static_single_gpu_preflight(config, repository_root)
    bindings = _resolve_governed_runtime_bindings(config)
    dependency_sha256 = _dependency_sha256(repository_root)
    checkpoint_root = repository_directory(repository_root, config.paths.checkpoint_dir)
    resolved_config_path = _publish_resolved_config(loaded, repository_root)
    artifact_root = repository_directory(repository_root, config.paths.artifact_dir)
    device = torch.device("cuda", config.evaluation.gpu_index)
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or config.evaluation.gpu_index != 0
    ):
        raise ValueError(
            "production S0 requires exactly one visible CUDA device at index zero"
        )
    torch.cuda.set_device(device)
    torch.manual_seed(config.run.seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.default_generators[0].manual_seed(config.run.seed)

    qwen = load_local_qwen(repository_root, device)
    vae = load_local_mage_vae(repository_root, device)
    module = build_trainable_composite_from_config(config, device=device)
    optimizer = _build_optimizer(config, module)
    if resume is None:
        scheduler = _SuccessfulUpdateLrScheduler(
            config,
            optimizer,
            bindings.learning_rate_for_update,
            restored_successful_update=0,
            fresh=True,
        )
        restored = _bootstrap_checkpoint(
            loaded,
            checkpoint_root=checkpoint_root,
            dependency_sha256=dependency_sha256,
            module=module,
            optimizer=optimizer,
            wall_clock=wall_clock(),
        )
    else:
        restored = _restore_checkpoint(
            loaded,
            checkpoint=resume,
            dependency_sha256=dependency_sha256,
            module=module,
            optimizer=optimizer,
            learning_rate_for_update=bindings.learning_rate_for_update,
        )
        scheduler = _SuccessfulUpdateLrScheduler(
            config,
            optimizer,
            bindings.learning_rate_for_update,
            restored_successful_update=restored.state.trainer.successful_updates,
            fresh=False,
        )

    # No service connection may occur before the exact RAW restore above.
    service_identity = DataServiceSessionIdentity(
        config.data.manifest.sha256,
        config.data.cache.persistent_workers_per_rank,
    )
    client = DataServiceClient(
        Path(config.data.service.socket_path),
        service_identity,
        request_timeout_seconds=config.data.service.request_timeout_seconds,
    )
    padding_token_id = qwen.tokenizer.pad_token_id
    if type(padding_token_id) is not int:
        raise ValueError("Qwen tokenizer padding identity is unavailable")
    factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=repository_root,
        tokenizer=qwen.tokenizer,
        framing=FramingContract(
            EXPECTED_PREFIX_TOKENS,
            EXPECTED_SUFFIX_TOKENS,
            padding_token_id,
        ),
        rejection_observer=_reject_sample,
        pass_index=bindings.pass_index,
    )
    batches = factory.batches(client)
    primary: BaseException | None = None
    result: ProductionTrainingResult | None = None
    try:
        runtime = _runtime(
            config,
            qwen=qwen,
            vae=vae,
            module=module,
            restored=restored,
            device=device,
        )
        publisher = ProductionSingleGpuCheckpointPublisher(
            checkpoint_root=checkpoint_root,
            resolved_config=loaded.resolved_toml.encode("utf-8"),
            module=module,
            optimizer=optimizer,
            restored_checkpoint=restored,
            accepted_checkpoint_ids=frozenset(),
        )
        workload = build_single_gpu_preflight_workload(
            config,
            runtime=runtime,
            trainable_module=module,
            optimizer=optimizer,
        )
        plan = build_single_gpu_preflight_checks(
            loaded,
            repository_root=repository_root,
            resolved_config_path=resolved_config_path,
            data_client=client,
            batches=batches,
            runtime=runtime,
            qwen=qwen.encoder,
            vae=vae,
            trainable_module=module,
            optimizer=optimizer,
            restored_checkpoint=restored,
            workload=workload,
            checkpoint_publisher=publisher,
        )
        preflight_report = artifact_root / (
            f"preflight-{restored.state.trainer.successful_updates}-"
            f"{secrets.token_hex(8)}.json"
        )
        accepted = run_single_gpu_preflight(plan, preflight_report)
        initial_update = restored.state.trainer.successful_updates
        if preflight_only:
            result = ProductionTrainingResult(
                loaded.resolved_sha256,
                preflight_report.resolve(strict=True),
                restored.path,
                initial_update,
                initial_update,
                True,
            )
        else:
            verified_checkpoints: list[Path] = []
            try:
                context_provider = _ProductionMetricContext(
                    bindings.dit_flops, bindings.ready_queue_depth
                )
                telemetry = build_training_telemetry_from_config(
                    config,
                    repository_root=repository_root,
                    device=device,
                    resolved_sha256=loaded.resolved_sha256,
                    context_provider=context_provider,
                )
                with telemetry:
                    loop_result = run_single_gpu_training(
                        config,
                        preflight=accepted,
                        runtime=runtime,
                        module=module,
                        optimizer=optimizer,
                        batches=batches,
                        scheduler_step=scheduler,
                        checkpoint_publisher=publisher,
                        diagnostic_root=artifact_root / "diagnostics",
                        failure_id=lambda phase, state: (
                            f"{phase}-{state.attempted_updates}-{secrets.token_hex(6)}"
                        ),
                        restored_checkpoint=restored,
                        phase_timer=telemetry.phase_timer,
                        successful_update_observer=telemetry.observer,
                        forced_checkpoint=lambda update: (
                            CheckpointReason.STAGE_FINALIZE
                            if update
                            == restored.state.stage_budget.terminal_successful_update
                            else None
                        ),
                        verified_checkpoint_observer=verified_checkpoints.append,
                    )
                if not verified_checkpoints:
                    raise RuntimeError(
                        "production training completed without a durable checkpoint"
                    )
            except Exception as error:
                raise ProductionTrainingError(
                    "accepted production training failed"
                ) from error
            result = ProductionTrainingResult(
                loaded.resolved_sha256,
                preflight_report.resolve(strict=True),
                verified_checkpoints[-1],
                initial_update,
                loop_result.state.successful_updates,
                False,
            )
    except BaseException as error:  # noqa: BLE001 - preserve cleanup failures
        primary = error
    cleanup: BaseException | None = None
    try:
        batches.close()
    except BaseException as error:  # noqa: BLE001 - preserve primary failure
        cleanup = error
    if primary is not None:
        if isinstance(primary, ProductionTrainingError):
            if cleanup is not None:
                combined = BaseExceptionGroup(
                    "accepted training and stream cleanup failed",
                    [primary, cleanup],
                )
                raise ProductionTrainingError(
                    "accepted production training cleanup failed"
                ) from combined
            raise primary
        _raise_preserving(
            "production lifecycle and stream cleanup failed", primary, cleanup
        )
    if cleanup is not None:
        if result is not None and not result.preflight_only:
            raise ProductionTrainingError(
                "accepted production training stream cleanup failed"
            ) from cleanup
        raise cleanup
    assert result is not None
    return result


def run_production_single_gpu(
    config_path: Path,
    *,
    config_root: Path,
    repository_root: Path,
    resume: Path | None = None,
    preflight_only: bool = False,
) -> ProductionTrainingResult:
    """Load strict config and run the exact fresh or raw-resume S0 lifecycle."""

    if type(preflight_only) is not bool:
        raise TypeError("preflight_only must be a bool")
    root = _repository_root(repository_root)
    loaded = load_config(
        config_path,
        config_root=_config_root(root, config_root),
    )
    try:
        require_single_gpu_config(loaded.config)
    except ValueError as error:
        raise ConfigurationError(
            "resolved config is not an enabled production single-GPU S0 run"
        ) from error
    exact_resume = None if resume is None else _require_exact_resume_path(resume)
    try:
        return _run_accepted_lifecycle(
            loaded,
            repository_root=root,
            resume=exact_resume,
            preflight_only=preflight_only,
            wall_clock=time.time,
        )
    except (
        ConfigurationError,
        ProductionReadinessError,
        ProductionTrainingError,
    ):
        raise
    except Exception as error:
        raise ProductionPreflightError(
            "production single-GPU lifecycle failed"
        ) from error


__all__ = [
    "ProductionPreflightError",
    "ProductionReadinessError",
    "ProductionTrainingError",
    "ProductionTrainingResult",
    "run_production_single_gpu",
]
