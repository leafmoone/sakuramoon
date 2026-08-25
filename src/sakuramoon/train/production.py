"""Fail-closed production lifecycle for the governed S0 single-GPU trainer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import NoReturn, Self, cast

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from sakuramoon.checkpoint.load import (
    load_inference_artifact,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import ConfigurationError, LoadedConfig, load_config
from sakuramoon.config.assembly import (
    build_trainable_composite_from_config,
    build_training_telemetry_from_config,
)
from sakuramoon.config.resolve import write_resolved_config
from sakuramoon.config.schema import EvaluationEnabledConfig, RuntimeConfig
from sakuramoon.data.client import DataServiceClient, RankedDataServiceClient
from sakuramoon.data.production import ProductionPipelineFactory
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
)
from sakuramoon.distributed import DistributedProgress
from sakuramoon.encoders.mage_vae import (
    FrozenMageVAE,
    compile_vae_methods,
    load_local_mage_vae,
)
from sakuramoon.encoders.qwen import QwenRuntime, load_local_qwen
from sakuramoon.eval.runtime import EvaluationResult, TrainingEvaluator
from sakuramoon.model.growth import active_slot_ids, growth_ramp_updates
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit
from sakuramoon.optim.dtk import configure_tunableop
from sakuramoon.storage import repository_directory
from sakuramoon.telemetry.observer import UpdateMetricContext
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.benchmark import BenchmarkTrace
from sakuramoon.train.preflight import (
    ProductionSingleGpuCheckpointPublisher,
    RestoredSingleGpuCheckpoint,
    build_single_gpu_preflight_checks,
    diff_resolved_toml_paths,
    record_data_policy_transition,
    require_spatial_transition_allowlist,
    require_static_single_gpu_preflight,
    restore_single_gpu_checkpoint,
    run_single_gpu_preflight,
)
from sakuramoon.train.runtime import (
    SingleGpuBatchRuntime,
    SuccessfulTrainingObservation,
    compile_packed_dit_blocks,
    require_distributed_forward_module,
    require_single_gpu_checkpoint_binding,
    require_single_gpu_checkpoint_compatibility,
    require_single_gpu_config,
    run_single_gpu_training,
)
from sakuramoon.train.sampling import TrainingSampler
from sakuramoon.train.stage import (
    GrowthProgress,
    canonical_growth_alpha,
    checkpoint_reason,
)
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite


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
    resolved_config: Path
    preflight_report: Path
    checkpoint_path: Path
    initial_successful_update: int
    final_successful_update: int
    preflight_only: bool

    def __post_init__(self) -> None:
        if (
            not self.resolved_config.is_absolute()
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
ReadyQueueDepthObserver = Callable[[], int]


def _log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def _raise_preserving(
    message: str, primary: BaseException, cleanup: BaseException | None
) -> NoReturn:
    if cleanup is not None:
        raise BaseExceptionGroup(message, [primary, cleanup]) from None
    raise primary


def _evaluate_and_release_process_group(
    evaluator: TrainingEvaluator,
    update: int,
    accelerator: Accelerator | None,
) -> EvaluationResult | None:
    """Release distributed ranks without a post-generation collective."""

    result: EvaluationResult | None = None
    primary: Exception | None = None
    try:
        result = evaluator.evaluate(update)
    except Exception as error:  # noqa: BLE001 - preserve evaluator failure
        primary = error

    cleanup: Exception | None = None
    if accelerator is not None:
        try:
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "distributed evaluation process group is not initialized"
                )
            print(
                "[eval] releasing distributed process group without final collective",
                flush=True,
            )
            torch.distributed.destroy_process_group()
        except Exception as error:  # noqa: BLE001 - preserve cleanup failure
            cleanup = error

    if primary is not None:
        _raise_preserving(
            "evaluation and process-group cleanup failed", primary, cleanup
        )
    if cleanup is not None:
        raise cleanup
    return result


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


def _require_exact_checkpoint_path(checkpoint: Path) -> Path:
    if not checkpoint.is_absolute():
        raise ConfigurationError(
            "checkpoint must be an exact absolute raw COMPLETE directory"
        )
    current = Path(checkpoint.anchor)
    for part in checkpoint.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ConfigurationError("checkpoint path may not contain symbolic links")
    try:
        resolved = checkpoint.resolve(strict=True)
    except OSError as error:
        raise ConfigurationError("checkpoint does not exist") from error
    marker = resolved / "COMPLETE"
    if (
        resolved != checkpoint
        or not resolved.is_dir()
        or marker.is_symlink()
        or not marker.is_file()
    ):
        raise ConfigurationError(
            "checkpoint must be an exact absolute raw COMPLETE directory"
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
    destination = run_root / "resolved.toml"
    if destination.is_file():
        previous = destination.read_text(encoding="utf-8")
        if previous != loaded.resolved_toml:
            changed = diff_resolved_toml_paths(previous, loaded.resolved_toml)
            require_spatial_transition_allowlist(changed)
            record_data_policy_transition(
                repository_directory(
                    repository_root, loaded.config.paths.artifact_dir
                )
                / "data_policy_transition.json",
                {
                    "kind": "resolved-config-diff",
                    "policy_class": "data-only",
                    "changed_toml_paths": list(changed),
                    "recorded_at_unix_ns": time.time_ns(),
                },
            )
    write_resolved_config(loaded.config, destination)
    return destination.resolve(strict=True)


def _record_spatial_resume_transition(
    loaded: LoadedConfig, repository_root: Path, resume: Path
) -> None:
    """Append the checkpoint-resume record of the shifted-bucket cutover."""

    config = loaded.config
    if not config.data.spatial_crop.enabled:
        return
    artifact_path = (
        repository_directory(repository_root, config.paths.artifact_dir)
        / "data_policy_transition.json"
    )
    record_data_policy_transition(
        artifact_path,
        {
            "kind": "checkpoint-resume",
            "policy_class": "data-only",
            "resume_checkpoint": str(resume),
            "spatial_crop": config.data.spatial_crop.model_dump(),
            "recorded_at_unix_ns": time.time_ns(),
        },
        skip_if_duplicate_of_last=(
            "kind",
            "resume_checkpoint",
            "spatial_crop",
        ),
    )


def _build_optimizer(
    config: RuntimeConfig, module: TrainableComposite
) -> IsolatedAdamW8bit:
    optimizer = config.optimizer
    return build_adamw8bit(
        module,
        lr=config.scaled_learning_rate(),
        betas=optimizer.betas,
        eps=optimizer.eps,
        block_size=optimizer.block_size,
        bf16_stochastic_round=optimizer.bf16_stochastic_round,
        matrix_weight_decay=optimizer.matrix_weight_decay,
        sensitive_weight_decay=optimizer.sensitive_weight_decay,
        sr_seed=config.run.seed,
    )


def _optimizer_learning_rate_scalars(
    optimizer: IsolatedAdamW8bit,
) -> tuple[float | torch.Tensor, ...]:
    groups = optimizer.optimizer.param_groups
    if not groups:
        raise ValueError("production optimizer must expose parameter groups")
    rates: list[float | torch.Tensor] = []
    representation: tuple[str, torch.dtype | None, torch.device | None] | None = None
    for group in groups:
        raw = group.get("lr")
        if type(raw) is float:
            value = raw
            current_representation = ("float", None, None)
        elif (
            type(raw) is torch.Tensor and raw.ndim == 0 and raw.dtype.is_floating_point
        ):
            value = float(raw.detach().item())
            if raw.dtype != torch.float32:
                raise ValueError(
                    "production optimizer tensor learning rate dtype must be float32"
                )
            current_representation = ("tensor", raw.dtype, raw.device)
        else:
            raise ValueError("production optimizer learning rate is invalid")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("production optimizer learning rate is invalid")
        if representation is None:
            representation = current_representation
        elif current_representation != representation:
            raise ValueError(
                "production optimizer learning rate representations differ across groups"
            )
        rates.append(raw)
    return tuple(rates)


def _optimizer_learning_rate_matches(
    optimizer: IsolatedAdamW8bit,
    expected: float,
) -> bool:
    if type(expected) is not float or not math.isfinite(expected) or expected < 0.0:
        raise ValueError("governed learning rate must be a finite nonnegative float")
    for actual in _optimizer_learning_rate_scalars(optimizer):
        if isinstance(actual, float):
            if actual != expected:
                return False
            continue
        assert isinstance(actual, torch.Tensor)
        canonical = torch.full(
            (),
            expected,
            dtype=actual.dtype,
            device=actual.device,
        )
        if not torch.equal(actual.detach(), canonical):
            return False
    return True


def _set_optimizer_learning_rate(
    optimizer: IsolatedAdamW8bit, learning_rate: float
) -> None:
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate < 0.0
    ):
        raise ValueError("governed learning rate must be a finite nonnegative float")
    groups = optimizer.optimizer.param_groups
    current_rates = _optimizer_learning_rate_scalars(optimizer)
    for group, current in zip(groups, current_rates, strict=True):
        if isinstance(current, float):
            group["lr"] = learning_rate
        else:
            assert isinstance(current, torch.Tensor)
            with torch.no_grad():
                current.fill_(learning_rate)


def _s0_linear_warmup_learning_rate(config: RuntimeConfig, update: int) -> float:
    """Return the TOML-bound S0 LR for the next successful-update attempt."""

    scheduler = config.scheduler
    max_lr = config.scaled_learning_rate()
    warmup_updates = scheduler.warmup_updates
    if (
        scheduler.name != "linear_warmup_constant"
        or scheduler.after_warmup != "constant"
        or type(max_lr) is not float
        or not math.isfinite(max_lr)
        or max_lr <= 0.0
        or type(warmup_updates) is not int
        or warmup_updates <= 0
    ):
        raise ValueError("resolved S0 linear-warmup schedule is invalid")
    if type(update) is not int or update <= 0:
        raise ValueError("scheduled update must be a positive integer")
    if update >= warmup_updates:
        return max_lr
    return max_lr * (update / warmup_updates)


class _SuccessfulUpdateLrScheduler:
    """Apply the governed LR schedule only at successful-update edges."""

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
        elif not _optimizer_learning_rate_matches(optimizer, expected):
            raise ValueError(
                "restored optimizer learning rate differs from governed schedule state"
            )

    def _rate(self, update: int) -> float:
        if type(update) is not int or update <= 0:
            raise ValueError("scheduled update must be positive")
        rate = self.learning_rate_for_update(self.config, update)
        if type(rate) is not float or not math.isfinite(rate) or rate < 0.0:
            raise ValueError("governed learning-rate schedule returned an invalid rate")
        return rate

    def __call__(self, successful_update: int) -> None:
        if successful_update != self.last_successful_update + 1:
            raise ValueError("learning-rate scheduler updates must be consecutive")
        _set_optimizer_learning_rate(self.optimizer, self._rate(successful_update + 1))
        self.last_successful_update = successful_update


def _initial_raw_state(
    config: RuntimeConfig, *, wall_clock: float
) -> RawCheckpointState:
    growth_enabled = config.growth.enabled
    return RawCheckpointState(
        trainer=SingleGpuUpdateState.initial(),
        growth=GrowthCheckpointState(
            active_slot_ids(config.stage.depth),
            0.0 if growth_enabled else 1.0,
            config.stage.name,
            config.stage.world_size,
            config.stage.resolution,
            0 if growth_enabled else None,
            growth_ramp_updates(config.stage.planned_updates)
            if growth_enabled
            else None,
        ),
        stage_budget=StageBudgetCheckpointState(0, config.stage.planned_updates),
        checkpoint_cadence=CheckpointCadence(
            0,
            wall_clock,
            config.checkpoint.full_every_updates,
        ),
    )


def _forced_production_checkpoint_reason(
    state: RawCheckpointState,
    *,
    initial_update: int,
    update: int,
) -> CheckpointReason | None:
    """Resolve durable G1 ramp points without duplicating cadence saves."""

    if type(initial_update) is not int or type(update) is not int:
        raise TypeError("forced checkpoint updates must be integers")
    if update <= initial_update:
        return None
    if state.growth.ramp_start_successful_update is not None:
        forced = GrowthProgress.from_checkpoint(state).forced_checkpoint(update)
        if forced is not None and forced.value != "post-transition":
            return checkpoint_reason(forced)
    if update == state.stage_budget.terminal_successful_update:
        return CheckpointReason.STAGE_FINALIZE
    return None


def _bootstrap_checkpoint(
    loaded: LoadedConfig,
    *,
    checkpoint_root: Path,
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    wall_clock: float,
) -> RestoredSingleGpuCheckpoint:
    identity = CheckpointIdentity("bootstrap", 0)
    bootstrap = checkpoint_root / "ckpt_0_bootstrap"
    entries = tuple(checkpoint_root.iterdir())
    if entries == (bootstrap,) and (bootstrap / "COMPLETE").is_file():
        _log("复用尚未开始训练的初始化状态")
        return restore_single_gpu_checkpoint(bootstrap, module, optimizer, identity)
    if entries:
        raise ConfigurationError(
            "fresh start requires an empty configured checkpoint directory"
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


def _resume_state_for_config(
    config: RuntimeConfig,
    state: RawCheckpointState,
) -> RawCheckpointState:
    """Apply explicit governed resume-policy changes to a validated RAW state."""

    terminal = state.stage_budget.terminal_successful_update
    configured_terminal = (
        state.stage_budget.start_successful_update + config.stage.planned_updates
    )
    if configured_terminal < terminal:
        raise ValueError("configured planned updates cannot shrink checkpoint budget")
    resumed = state
    if state.growth.world_size != config.stage.world_size:
        if not (
            state.growth.world_size == 1
            and config.distributed.backend == "accelerate"
            and config.stage.world_size == 2
        ):
            raise ValueError("unsupported checkpoint world-size transition")
        _log("迁移检查点拓扑: world_size 1 -> 2")
        resumed = replace(
            resumed,
            growth=replace(state.growth, world_size=config.stage.world_size),
        )
    if configured_terminal != terminal:
        _log(f"扩展训练总步数: {terminal} -> {configured_terminal}")
        resumed = replace(
            resumed,
            stage_budget=StageBudgetCheckpointState(
                state.stage_budget.start_successful_update,
                configured_terminal,
            ),
        )
    persisted_interval = state.checkpoint_cadence.every_successful_updates
    configured_interval = config.checkpoint.full_every_updates
    if configured_interval != persisted_interval:
        _log(
            "重绑定检查点保存周期: "
            f"{persisted_interval} -> {configured_interval} successful updates"
        )
        resumed = replace(
            resumed,
            checkpoint_cadence=replace(
                resumed.checkpoint_cadence,
                every_successful_updates=configured_interval,
            ),
        )
    return resumed


def _restore_checkpoint(
    loaded: LoadedConfig,
    *,
    checkpoint: Path,
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    learning_rate_for_update: LearningRateForUpdate,
) -> RestoredSingleGpuCheckpoint:
    manifest, state = read_raw_checkpoint_state(checkpoint)
    identity = manifest.identity
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
        raise ValueError("governed schedule returned an invalid resume rate")
    # The RAW loader compares every optimizer group field before mutation. Setting
    # the trusted expected rate here makes checkpoint LR drift a hard load failure.
    _set_optimizer_learning_rate(optimizer, expected_rate)
    restored = restore_single_gpu_checkpoint(checkpoint, module, optimizer, identity)
    resumed_state = _resume_state_for_config(loaded.config, restored.state)
    return replace(restored, state=resumed_state)


def _runtime(
    config: RuntimeConfig,
    *,
    qwen: QwenRuntime,
    vae: FrozenMageVAE,
    module: TrainableComposite,
    forward_module: nn.Module | None,
    restored: RestoredSingleGpuCheckpoint,
    device: torch.device,
) -> SingleGpuBatchRuntime:
    index = int(device.index or 0)
    return SingleGpuBatchRuntime(
        qwen=qwen.encoder,
        vae=vae,
        composite=module,
        forward_module=forward_module,
        device=device,
        generator=torch.cuda.default_generators[index],
        p_mean=config.timestep.p_mean,
        p_std=config.timestep.p_std,
        noise_scale=config.timestep.noise_scale,
        t_eps=config.timestep.t_eps,
        noise_observation_boundary=config.logging.noise_observation_boundary,
        growth_alpha=restored.state.growth.alpha,
        torch_compile_enabled=config.kernels.torch_compile_enabled,
        torch_compile_backend=config.kernels.torch_compile_backend,
        torch_compile_mode=config.kernels.torch_compile_mode,
        torch_compile_dynamic=config.kernels.torch_compile_dynamic,
    )


class _ProductionMetricContext:
    def __init__(
        self,
        ready_queue_depth: ReadyQueueDepthObserver,
        world_size: int = 1,
        transparent_rejection_totals: Callable[[], Mapping[str, int]] | None = None,
    ) -> None:
        if not callable(ready_queue_depth):
            raise TypeError("live ready-queue observer must be callable")
        self.ready_queue_depth = ready_queue_depth
        self.world_size = world_size
        self.transparent_rejection_totals = transparent_rejection_totals

    def __call__(
        self, observation: SuccessfulTrainingObservation
    ) -> UpdateMetricContext:
        dit_flops = sum(item.dit_flops for item in observation.microbatches)
        if dit_flops <= 0:
            raise ValueError("DiT runtime produced an invalid FLOP observation")
        depth = self.ready_queue_depth()
        if type(depth) is not int or depth < 0:
            raise ValueError("ready-queue observer returned an invalid depth")
        rejection_totals = (
            self.transparent_rejection_totals()
            if self.transparent_rejection_totals is not None
            else None
        )
        return UpdateMetricContext(
            dit_flops=dit_flops,
            samples_per_second=(
                observation.loop.update.effective_samples
                * self.world_size
                / observation.loop.update_wall_seconds
            ),
            ready_queue_depth=depth,
            supplemental_phase_seconds={},
            transparent_rejection_totals=rejection_totals,
        )


class _NoopTelemetry:
    """Keep non-main ranks timed without duplicating logs or W&B runs."""

    def __init__(self, device: torch.device) -> None:
        self.phase_timer = PhaseTimer(device=device)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def observer(self, _observation: SuccessfulTrainingObservation) -> None:
        return None

    def submit_wandb_images(self, *_args: object, **_kwargs: object) -> None:
        return None

    def submit_wandb_metrics(self, *_args: object, **_kwargs: object) -> None:
        return None


class _AccelerateCheckpointPublisher(ProductionSingleGpuCheckpointPublisher):
    """Let rank zero publish once while every rank verifies the same checkpoint."""

    def __init__(
        self,
        delegate: ProductionSingleGpuCheckpointPublisher,
        accelerator: Accelerator,
        progress: DistributedProgress,
    ) -> None:
        self._delegate = delegate
        self._accelerator = accelerator
        self._progress = progress

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        stage = (
            f"checkpoint/update-{state.successful_updates}/"
            f"publish-{reason.value}"
        )
        published = self._progress.run_on_rank(
            stage,
            0,
            lambda: self._delegate.publish_update(state, reason, cadence),
        )
        value = [
            str(published) if self._accelerator.is_main_process else ""
        ]
        torch.distributed.broadcast_object_list(value, src=0)
        if not value[0]:
            raise RuntimeError("rank zero did not publish a checkpoint path")
        return Path(value[0]).resolve(strict=True)

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: object,
        state: RawCheckpointState,
    ) -> None:
        self._progress.run_on_rank(
            f"checkpoint/{checkpoint.name}/retention",
            0,
            lambda: self._delegate.apply_verified_retention(
                checkpoint,
                manifest,  # type: ignore[arg-type]
                state,
            ),
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
    accelerator = (
        Accelerator(
            mixed_precision="no",
            kwargs_handlers=[
                # Condition-token encoding is data-dependent: a rank whose final
                # accumulation microbatch contains only null-condition samples does
                # not traverse the trainable ConditionTokenEncoder projection path.
                # DDP must discover those dynamic unused parameters or that rank
                # can enter the post-update barrier while a peer still reduces
                # the condition gradient buckets.
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(minutes=5)),
            ],
        )
        if config.distributed.backend == "accelerate"
        else None
    )
    rank = 0 if accelerator is None else accelerator.process_index
    is_main_process = accelerator is None or accelerator.is_main_process
    world_size = 1 if accelerator is None else accelerator.num_processes
    progress = DistributedProgress.from_default_store(
        namespace=f"training/{config.run.run_id}",
        rank=rank,
        world_size=world_size,
    )
    benchmark_trace = BenchmarkTrace.from_environment(rank)
    if accelerator is not None and (
        accelerator.num_processes != config.distributed.world_size
        or accelerator.device.type != "cuda"
    ):
        raise ValueError("Accelerate launch topology differs from resolved config")
    _log("检查本地模型与运行目录")
    require_static_single_gpu_preflight(config, repository_root)
    checkpoint_root = repository_directory(repository_root, config.paths.checkpoint_dir)
    if is_main_process:
        resolved_config_path = _publish_resolved_config(loaded, repository_root)
    else:
        resolved_config_path = (
            repository_directory(repository_root, config.paths.run_dir)
            / "resolved.toml"
        )
    progress.synchronize("startup/resolved-config-published")
    resolved_config_path = resolved_config_path.resolve(strict=True)
    artifact_root = repository_directory(repository_root, config.paths.artifact_dir)
    device = torch.device("cuda", 0) if accelerator is None else accelerator.device
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != config.distributed.world_size
        or int(device.index or 0) != rank
    ):
        raise ValueError(
            "production S0 accelerator visibility differs from rank topology"
        )
    torch.cuda.set_device(device)
    process_seed = config.run.seed + rank * 1_000_003
    torch.manual_seed(process_seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.default_generators[int(device.index or 0)].manual_seed(process_seed)

    if is_main_process:
        _log(
            "learning-rate scaling: "
            f"base_lr={config.optimizer.base_lr:.8g}, "
            f"reference_batch={config.optimizer.reference_batch}, "
            f"effective_global_batch={config.stage.global_batch}, "
            f"actual_lr={config.scaled_learning_rate():.8g}"
        )

    tunable_state = configure_tunableop(
        repository_root,
        config.paths.run_dir,
        enabled=config.kernels.tunableop_enabled,
        tuning=config.kernels.tunableop_tuning,
        record_untuned=config.kernels.tunableop_record_untuned,
        max_tuning_duration_ms=config.kernels.tunableop_max_tuning_duration_ms,
    )
    if tunable_state.enabled:
        _log(
            "TunableOp enabled "
            f"(tuning={tunable_state.tuning}, loaded={tunable_state.loaded_results})"
        )

    _log("加载 Qwen 文本编码器")
    qwen = load_local_qwen(
        repository_root,
        device,
        attention_backend=config.kernels.qwen_attention_backend,
    )
    _log("加载 Mage VAE")
    vae = load_local_mage_vae(repository_root, device)
    if config.kernels.vae_torch_compile:
        vae = compile_vae_methods(vae)
        _log("Mage VAE encode/decode 已启用 torch.compile (opt-in)")
    _log(f"构建 {config.stage.depth} 层 DiT")
    module = build_trainable_composite_from_config(config, device=device)
    _log("构建优化器")
    optimizer = _build_optimizer(config, module)
    if resume is None:
        _log("创建全新训练状态")
        scheduler = _SuccessfulUpdateLrScheduler(
            config,
            optimizer,
            _s0_linear_warmup_learning_rate,
            restored_successful_update=0,
            fresh=True,
        )
        restored = _bootstrap_checkpoint(
            loaded,
            checkpoint_root=checkpoint_root,
            module=module,
            optimizer=optimizer,
            wall_clock=wall_clock(),
        )
    else:
        _log(f"恢复训练状态: {resume}")
        restored = _restore_checkpoint(
            loaded,
            checkpoint=resume,
            module=module,
            optimizer=optimizer,
            learning_rate_for_update=_s0_linear_warmup_learning_rate,
        )
        scheduler = _SuccessfulUpdateLrScheduler(
            config,
            optimizer,
            _s0_linear_warmup_learning_rate,
            restored_successful_update=restored.state.trainer.successful_updates,
            fresh=False,
        )

    require_single_gpu_checkpoint_binding(
        config,
        restored.state,
        runtime_growth_alpha=restored.state.growth.alpha,
    )
    if is_main_process and resume is not None:
        _record_spatial_resume_transition(loaded, repository_root, resume)
    if accelerator is not None and rank > 0:
        resumed_seed = (
            config.run.seed
            + rank * 1_000_003
            + restored.state.trainer.successful_updates
        )
        torch.manual_seed(resumed_seed)  # pyright: ignore[reportUnknownMemberType]
        torch.cuda.default_generators[int(device.index or 0)].manual_seed(resumed_seed)
    distributed_sync_module: DistributedDataParallel | None
    if accelerator is None:
        forward_module: nn.Module = module
        distributed_sync_module = None
    else:
        prepared_module, prepared_optimizer = accelerator.prepare(
            module,
            optimizer.optimizer,
        )
        forward_module = cast(nn.Module, prepared_module)
        distributed_sync_module = require_distributed_forward_module(
            module,
            forward_module,
        )
        optimizer.optimizer = prepared_optimizer  # type: ignore[assignment]
    if config.kernels.torch_compile_enabled:
        compiled_blocks = compile_packed_dit_blocks(
            module,
            backend=config.kernels.torch_compile_backend,
            mode=config.kernels.torch_compile_mode,
            dynamic=config.kernels.torch_compile_dynamic,
        )
        if is_main_process:
            _log(
                "regional torch.compile 已启用: DDP 保持 eager，"
                f"编译 {len(compiled_blocks)} 个 PackedDiTBlock，FA2 为显式 eager 边界"
            )
    _log(f"连接数据服务: {config.data.service.socket_path}")
    # No service connection may occur before exact RAW restore and full binding.
    service_client = DataServiceClient(
        Path(config.data.service.socket_path),
        worker_count=(
            config.data.cache.persistent_workers_per_rank
            * config.distributed.world_size
        ),
        request_timeout_seconds=config.data.service.request_timeout_seconds,
    )
    client = (
        service_client
        if accelerator is None
        else RankedDataServiceClient(
            service_client,
            rank=rank,
            workers_per_rank=config.data.cache.persistent_workers_per_rank,
            world_size=config.distributed.world_size,
        )
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
    )
    _log(
        f"数据分桶已就绪: {config.data.buckets.shape_count} 个形状，"
        f"batch={config.stage.local_batch}，accumulation={config.stage.accumulation}"
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
            forward_module=forward_module,
            restored=restored,
            device=device,
        )
        base_publisher = ProductionSingleGpuCheckpointPublisher(
            checkpoint_root=checkpoint_root,
            resolved_config=loaded.resolved_toml.encode("utf-8"),
            module=module,
            optimizer=optimizer,
            restored_checkpoint=restored,
            accepted_checkpoint_ids=frozenset(),
            retention_slots=config.checkpoint.slots,
        )
        publisher = (
            base_publisher
            if accelerator is None
            else _AccelerateCheckpointPublisher(
                base_publisher,
                accelerator,
                progress,
            )
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
            checkpoint_publisher=publisher,
            transition_artifact=artifact_root / "data_policy_transition.json",
        )
        preflight_report = artifact_root / (
            f"preflight-{restored.state.trainer.successful_updates}"
            f"{'' if is_main_process else f'-rank{rank}'}.json"
        )
        _log("运行训练前检查")
        accepted = run_single_gpu_preflight(plan, preflight_report)
        _log("训练前检查通过")
        initial_update = restored.state.trainer.successful_updates
        if preflight_only:
            result = ProductionTrainingResult(
                resolved_config_path,
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
                    batches.ready_batch_depth_snapshot,
                    world_size=config.distributed.world_size,
                    transparent_rejection_totals=batches.transparent_rejection_totals,
                )
                telemetry = (
                    build_training_telemetry_from_config(
                        config,
                        repository_root=repository_root,
                        device=device,
                        context_provider=context_provider,
                        resume_from_update=(
                            initial_update if resume is not None else None
                        ),
                    )
                    if is_main_process
                    else _NoopTelemetry(device)
                )
                evaluation_config = config.evaluation
                if isinstance(evaluation_config, EvaluationEnabledConfig):
                    evaluator: TrainingEvaluator | None = TrainingEvaluator(
                        config,
                        repository_root=repository_root,
                        composite=module,
                        qwen=qwen,
                        vae=vae,
                        device=device,
                        growth_alpha=restored.state.growth.alpha,
                        accelerator=accelerator,
                    )
                    evaluation_is_splits: int | None = evaluation_config.is_splits
                else:
                    evaluator = None
                    evaluation_is_splits = None
                training_sampler = (
                    TrainingSampler(
                        config,
                        repository_root=repository_root,
                        composite=module,
                        qwen=qwen,
                        vae=vae,
                        device=device,
                        growth_alpha=restored.state.growth.alpha,
                    )
                    if is_main_process and config.sampling.training.enabled
                    else None
                )

                def observe_successful_update(
                    observation: SuccessfulTrainingObservation,
                ) -> None:
                    update = observation.loop.update.state.successful_updates
                    if benchmark_trace is not None:
                        benchmark_trace.append(observation)
                    sampling_due = (
                        config.sampling.training.enabled
                        and update > 0
                        and update % config.sampling.training.every_updates == 0
                    )

                    def sample_and_publish() -> None:
                        if training_sampler is None:
                            raise RuntimeError(
                                "rank zero training sampler is unavailable"
                            )
                        with observation.phase_timer.record("sample"):
                            training_sampler.set_growth_alpha(
                                canonical_growth_alpha(restored.state.growth, update)
                            )
                            samples = training_sampler.sample(
                                update, observation.microbatches
                            )
                        telemetry.submit_wandb_images(
                            samples.paths,
                            samples.captions,
                            successful_update=update,
                        )
                        telemetry.submit_wandb_metrics(
                            {
                                "training_samples/count": len(samples.paths),
                                **{
                                    f"condition_diagnostics/{key}": value
                                    for key, value in samples.diagnostics.items()
                                },
                            },
                            successful_update=update,
                        )

                    if sampling_due:
                        progress.run_on_rank(
                            f"observer/update-{update}/training-samples",
                            0,
                            sample_and_publish,
                        )
                    progress.run_on_rank(
                        f"observer/update-{update}/telemetry",
                        0,
                        lambda: telemetry.observer(observation),
                    )
                    if evaluator is not None and evaluator.due(update):
                        if evaluation_is_splits is None:
                            raise RuntimeError("evaluation config is unavailable")
                        evaluator.set_growth_alpha(
                            canonical_growth_alpha(restored.state.growth, update)
                        )
                        evaluation = evaluator.evaluate(update)
                        if is_main_process:
                            if evaluation is None:
                                raise RuntimeError(
                                    "rank zero evaluation is unavailable"
                                )
                            telemetry.submit_wandb_metrics(
                                {
                                    "evaluation/fid": evaluation.fid,
                                    "evaluation/inception_score_mean": (
                                        evaluation.inception_score_mean
                                    ),
                                    "evaluation/inception_score_std": (
                                        evaluation.inception_score_std
                                    ),
                                    "evaluation/kid_mean": evaluation.kid_mean,
                                    "evaluation/kid_std": evaluation.kid_std,
                                    "evaluation/cmmd": evaluation.cmmd,
                                    "evaluation/sample_count": evaluation.sample_count,
                                    "evaluation/real_sample_count": (
                                        evaluation.real_sample_count
                                    ),
                                    "evaluation/is_splits": evaluation_is_splits,
                                },
                                successful_update=update,
                            )

                            suite_metrics = evaluator.run_concept_suite(update)
                            if suite_metrics:
                                telemetry.submit_wandb_metrics(
                                    {
                                        f"evaluation/concept/{key}": value
                                        for key, value in suite_metrics.items()
                                    },
                                    successful_update=update,
                                )
                    progress.synchronize(
                        f"observer/update-{update}/complete"
                    )

                with telemetry:
                    _log(
                        f"开始训练: update {initial_update + 1} -> "
                        f"{config.stage.planned_updates}"
                    )
                    loop_result = run_single_gpu_training(
                        config,
                        preflight=accepted,
                        runtime=runtime,
                        module=module,
                        optimizer=optimizer,
                        batches=batches,
                        scheduler_step=scheduler,
                        checkpoint_publisher=publisher,
                        diagnostic_root=(
                            artifact_root / "diagnostics" / f"rank-{rank}"
                        ),
                        failure_id=lambda phase, state: (
                            f"{phase}-{state.attempted_updates}"
                        ),
                        restored_checkpoint=restored,
                        phase_timer=telemetry.phase_timer,
                        successful_update_observer=observe_successful_update,
                        forced_checkpoint=(
                            lambda update: _forced_production_checkpoint_reason(
                                restored.state,
                                initial_update=initial_update,
                                update=update,
                            )
                        ),
                        verified_checkpoint_observer=verified_checkpoints.append,
                        backward=(
                            None if accelerator is None else accelerator.backward
                        ),
                        no_sync=(
                            None
                            if distributed_sync_module is None
                            else distributed_sync_module.no_sync
                        ),
                        log_updates=is_main_process,
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
                resolved_config_path,
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
    _log(f"加载 TOML: {config_path}")
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
    exact_resume = None if resume is None else _require_exact_checkpoint_path(resume)
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


def run_production_evaluation(
    config_path: Path,
    *,
    config_root: Path,
    repository_root: Path,
    checkpoint: Path,
) -> EvaluationResult | None:
    """Evaluate one complete RAW checkpoint without restarting training."""

    root = _repository_root(repository_root)
    print(f"[eval] 加载 TOML: {config_path}", flush=True)
    loaded = load_config(
        config_path,
        config_root=_config_root(root, config_root),
    )
    config = loaded.config
    try:
        require_single_gpu_config(config)
    except ValueError as error:
        raise ConfigurationError(
            "evaluation requires the enabled single-GPU S0 config"
        ) from error
    if config.evaluation.enabled is not True:
        raise ConfigurationError("evaluation is disabled in the resolved config")

    exact_checkpoint = _require_exact_checkpoint_path(checkpoint)
    require_static_single_gpu_preflight(config, root)
    accelerator = (
        Accelerator(
            mixed_precision="no",
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=5))],
        )
        if config.distributed.backend == "accelerate"
        else None
    )
    rank = 0 if accelerator is None else accelerator.process_index
    if accelerator is not None and (
        accelerator.num_processes != config.distributed.world_size
        or accelerator.device.type != "cuda"
    ):
        raise ValueError("evaluation launch topology differs from resolved config")
    device = torch.device("cuda", 0) if accelerator is None else accelerator.device
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != config.distributed.world_size
        or int(device.index or 0) != rank
    ):
        raise ValueError("evaluation accelerator visibility differs from rank topology")
    torch.cuda.set_device(device)
    process_seed = config.run.seed + rank * 1_000_003
    torch.manual_seed(process_seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.default_generators[int(device.index or 0)].manual_seed(process_seed)

    configure_tunableop(
        root,
        config.paths.run_dir,
        enabled=config.kernels.tunableop_enabled,
        tuning=False,
        record_untuned=False,
        max_tuning_duration_ms=config.kernels.tunableop_max_tuning_duration_ms,
    )

    print(f"[eval] 检查最终模型: {exact_checkpoint}", flush=True)
    manifest, state = read_raw_checkpoint_state(exact_checkpoint)
    require_single_gpu_checkpoint_compatibility(
        config,
        state,
        runtime_growth_alpha=state.growth.alpha,
    )
    update = state.trainer.successful_updates
    if update <= 0 or update % config.evaluation.every_updates != 0:
        raise ConfigurationError(
            "checkpoint update is not scheduled for FID/IS evaluation"
        )

    print("[eval] 加载 Qwen 文本编码器", flush=True)
    qwen = load_local_qwen(
        root,
        device,
        attention_backend=config.kernels.qwen_attention_backend,
    )
    print("[eval] 加载 Mage VAE", flush=True)
    vae = load_local_mage_vae(root, device)
    if config.kernels.vae_torch_compile:
        vae = compile_vae_methods(vae)
        print("[eval] Mage VAE encode/decode 已启用 torch.compile (opt-in)", flush=True)
    print(f"[eval] 加载 update {update} DiT", flush=True)
    module = load_inference_artifact(
        exact_checkpoint,
        manifest.identity,
        device=device,
    )
    if not isinstance(module, TrainableComposite):
        raise TypeError("checkpoint does not contain the trainable composite")
    evaluator = TrainingEvaluator(
        config,
        repository_root=root,
        composite=module,
        qwen=qwen,
        vae=vae,
        device=device,
        growth_alpha=state.growth.alpha,
        accelerator=accelerator,
    )
    result = _evaluate_and_release_process_group(
        evaluator, update, accelerator
    )
    if accelerator is None and result is None:
        raise RuntimeError("native evaluation returned no result")
    return result


__all__ = [
    "ProductionPreflightError",
    "ProductionReadinessError",
    "ProductionTrainingError",
    "ProductionTrainingResult",
    "run_production_evaluation",
    "run_production_single_gpu",
]
