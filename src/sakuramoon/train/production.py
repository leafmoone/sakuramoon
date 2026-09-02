"""Fail-closed production lifecycle for the governed S0 single-GPU trainer."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Self, cast

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
from sakuramoon.optim.guard_calibration import (
    GuardCalibration,
    GuardCalibrationComplete,
    install_guard_calibration,
)
from sakuramoon.optim.structural_calibration import (
    StructuralCalibrationComplete,
    install_structural_calibration,
)

if TYPE_CHECKING:
    from sakuramoon.optim.cmuon import HybridCMuon
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


def require_production_irepa_readiness(config: RuntimeConfig) -> None:
    """Phase 2 safety gate for the iREPA auxiliary.

    ``[irepa] enabled = true`` parses and its schema v4 architecture artifact
    constructs, but the runtime teacher/tap/loss integration is not installed
    yet.  Production must fail closed BEFORE optimizer build, data-service
    connection, and the training loop — never start a run where the projector
    exists but no loss consumes it.  Unlocked in the runtime-integration
    phase.
    """

    irepa = config.irepa
    if irepa is not None and irepa.enabled:
        raise ProductionReadinessError(
            ("iREPA enabled but runtime teacher/tap/loss integration is not installed",)
        )


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
            record_data_policy_transition(
                repository_directory(repository_root, loaded.config.paths.artifact_dir)
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


def _record_optimizer_transition(
    loaded: LoadedConfig,
    repository_root: Path,
    resume: Path,
    optimizer: IsolatedAdamW8bit | HybridCMuon,
    source_update: int,
) -> None:
    """Record the AdamW8bit -> Hybrid CMuon transition manifest (main rank).

    Written when the restored optimizer actually performed the AdamW8bit ->
    Hybrid transition (a pure-AdamW source checkpoint loaded into a hybrid
    optimizer). It records exactly what was preserved, what was dropped, and
    the canonical per-role NS map, so a fork point is auditable from the
    artifact alone. A native hybrid -> hybrid resume records nothing (there
    is no optimizer-semantics change).
    """
    if not isinstance(optimizer, _hybrid_cmuon_class()):
        return
    if not optimizer.transition_from_adamw8bit:
        return
    record_data_policy_transition(
        repository_directory(repository_root, loaded.config.paths.artifact_dir)
        / "optimizer_transition.json",
        {
            "kind": "optimizer-adamw8bit-to-hybrid-cmuon",
            "policy_class": "optimizer",
            "source_optimizer": "torchao_adamw8bit",
            "source_checkpoint": str(resume),
            "source_update": source_update,
            "fresh_cmuon_momentum": True,
            "preserved_adamw_fallback_params": (
                optimizer.transition_preserved_adamw_params
            ),
            "dropped_adamw_state_cmuon_params": (
                optimizer.transition_dropped_cmuon_params
            ),
            "canonical_ns_map": optimizer.cfg.canonical_ns_map(),
            "recorded_at_unix_ns": time.time_ns(),
        },
        skip_if_duplicate_of_last=("kind", "source_checkpoint", "source_update"),
    )


def _record_data_policy_resume_transition(
    loaded: LoadedConfig, repository_root: Path, resume: Path
) -> None:
    """Append the checkpoint-resume record of a data-policy cutover.

    One record per distinct (checkpoint, policy configuration) pair;
    restarts at the same checkpoint with the same configuration are a
    no-op. When the source checkpoint carries a different resolved config
    the changed leaf paths are diffed and recorded so the transition
    artifact shows exactly what moved.
    """

    config = loaded.config
    spatial_enabled = config.data.spatial_crop.enabled
    transparent_enabled = config.data.transparent_background.enabled
    if not spatial_enabled and not transparent_enabled:
        return
    changed: tuple[str, ...] = ()
    sidecar = resume / "resolved_config.toml"
    if sidecar.is_file():
        try:
            source_toml = sidecar.read_text(encoding="utf-8")
            if source_toml != loaded.resolved_toml:
                changed = diff_resolved_toml_paths(source_toml, loaded.resolved_toml)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ConfigurationError(
                f"cannot audit the resume config transition: {exc}"
            ) from None
    record: dict[str, object] = {
        "kind": "checkpoint-resume",
        "policy_class": "data-only",
        "resume_checkpoint": str(resume),
        "recorded_at_unix_ns": time.time_ns(),
    }
    skip_keys = ("kind", "resume_checkpoint")
    if spatial_enabled:
        # JSON mode so the stored values are exactly what the JSON artifact
        # round-trips back (sequences as lists): restart dedupe compares
        # these fields against the previously written record.
        record["spatial_crop"] = config.data.spatial_crop.model_dump(mode="json")
        skip_keys = (*skip_keys, "spatial_crop")
    if transparent_enabled:
        record["transparent_background"] = (
            config.data.transparent_background.model_dump(mode="json")
        )
        skip_keys = (*skip_keys, "transparent_background")
    if changed:
        record["resolved_config_changed_toml_paths"] = list(changed)
        skip_keys = (*skip_keys, "resolved_config_changed_toml_paths")
    record_data_policy_transition(
        repository_directory(repository_root, config.paths.artifact_dir)
        / "data_policy_transition.json",
        record,
        skip_if_duplicate_of_last=skip_keys,
    )


def _build_optimizer(
    config: RuntimeConfig,
    module: TrainableComposite,
    *,
    telemetry_logger: Callable[[str], None] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> IsolatedAdamW8bit | HybridCMuon:
    optimizer = config.optimizer
    common = {
        "lr": config.scaled_learning_rate(),
        "betas": optimizer.betas,
        "eps": optimizer.eps,
        "block_size": optimizer.block_size,
        "bf16_stochastic_round": optimizer.bf16_stochastic_round,
        "matrix_weight_decay": optimizer.matrix_weight_decay,
        "sensitive_weight_decay": optimizer.sensitive_weight_decay,
        "sr_seed": config.run.seed,
    }
    if optimizer.name == "hybrid_cmuon":
        from sakuramoon.optim.cmuon import build_hybrid_cmuon

        telemetry = optimizer.cmuon_ns_telemetry
        telemetry_kwargs = {
            "ns_telemetry_enabled": bool(telemetry.enabled) if telemetry else False,
            "ns_telemetry_log_every_n": telemetry.log_every_n if telemetry else 100,
            "ns_telemetry_logger": telemetry_logger,
            "ns_telemetry_update_offset": 0,
        }
        if telemetry is not None and telemetry.roles:
            telemetry_kwargs["ns_telemetry_roles"] = tuple(telemetry.roles)
        # Forensic routing ablation (empty for the candidate under test): the
        # excluded roles route to the AdamW8bit fallback, split stays complete.
        telemetry_kwargs["exclude_roles"] = optimizer.cmuon_routing_exclude
        if optimizer.cmuon_forensic is not None and optimizer.cmuon_forensic.enabled:
            from sakuramoon.optim.cmuon_forensic import ForensicConfig

            # Built on EVERY rank (the rank comparison needs all ranks); the
            # logger is main-rank only so the log lines are not duplicated.
            telemetry_kwargs["forensic"] = ForensicConfig(
                enabled=True,
                ring_size=optimizer.cmuon_forensic.ring_size,
                ceiling_multiplier=optimizer.cmuon_forensic.ceiling_multiplier,
                max_abs_learn_steps=optimizer.cmuon_forensic.max_abs_learn_steps,
                max_abs_alarm_mult=optimizer.cmuon_forensic.max_abs_alarm_mult,
                dump_dir=optimizer.cmuon_forensic.dump_dir,
                divergence_rel_tol=optimizer.cmuon_forensic.divergence_rel_tol,
            )
        if optimizer.cmuon_ns is not None:
            # Per-role (per-spec) NS depth: canonical role->depth map.
            return build_hybrid_cmuon(
                module,
                ns_steps_by_role=optimizer.cmuon_ns.canonical_map(),
                momentum_dtype=optimizer.cmuon_momentum_dtype,
                chunk_rescale_sqrt_n=optimizer.cmuon_chunk_rescale_sqrt_n,
                **telemetry_kwargs,
                **common,
            )
        # Legacy scalar NS depth (every role identical).
        return build_hybrid_cmuon(
            module,
            ns_steps=optimizer.cmuon_ns_steps,
            momentum_dtype=optimizer.cmuon_momentum_dtype,
            chunk_rescale_sqrt_n=optimizer.cmuon_chunk_rescale_sqrt_n,
            **telemetry_kwargs,
            **common,
        )
    if optimizer.name in (
        "hybrid_cmuon_guarded_canonical",
        "hybrid_cmuon_canonical_ns4_fp32_rescue",
    ):
        from sakuramoon.optim.guarded_canonical import (
            GuardedCanonicalGuardConfig,
            build_guarded_canonical,
        )

        guard = optimizer.cmuon_guard
        assert guard is not None and guard.enabled  # enforced by the schema
        if optimizer.cmuon_forensic is not None and optimizer.cmuon_forensic.enabled:
            raise ValueError(
                "the guarded canonical candidate is two-phase fail-closed by "
                "construction; [optimizer.cmuon_forensic] is for the retired "
                "original candidate and must stay disabled"
            )
        if optimizer.cmuon_ns_telemetry is not None and optimizer.cmuon_ns_telemetry.enabled:
            raise ValueError(
                "the guarded canonical candidate has its own per-step safety "
                "checks; [optimizer.cmuon_ns_telemetry] must stay disabled"
            )
        if optimizer.cmuon_ns is None:
            raise ValueError(
                f"{optimizer.name} requires an explicit [optimizer.cmuon_ns] "
                "per-role NS map (no legacy scalar fallback)"
            )
        stats_logger = telemetry_logger if rank == 0 else None
        if optimizer.name == "hybrid_cmuon_canonical_ns4_fp32_rescue":
            from sakuramoon.optim.fp32_rescue import build_fp32_rescue

            return build_fp32_rescue(
                module,
                ns_steps_by_role=optimizer.cmuon_ns.canonical_map(),
                guard_cfg=GuardedCanonicalGuardConfig(
                    guard_ratio=guard.guard_ratio,
                    reference_decay=guard.reference_decay,
                    min_reference=guard.min_reference,
                    numerical_floor=guard.numerical_floor,
                    warmup_observations=guard.warmup_observations,
                    invariant_check=guard.invariant_check,
                ),
                guard_bootstrap_refs=dict(guard.references),
                rank=rank,
                world_size=world_size,
                momentum_dtype=optimizer.cmuon_momentum_dtype,
                chunk_rescale_sqrt_n=optimizer.cmuon_chunk_rescale_sqrt_n,
                stats_logger=stats_logger,
                **common,
            )
        return build_guarded_canonical(
            module,
            ns_steps_by_role=optimizer.cmuon_ns.canonical_map(),
            guard_cfg=GuardedCanonicalGuardConfig(
                guard_ratio=guard.guard_ratio,
                reference_decay=guard.reference_decay,
                min_reference=guard.min_reference,
                numerical_floor=guard.numerical_floor,
                warmup_observations=guard.warmup_observations,
                invariant_check=guard.invariant_check,
            ),
            guard_bootstrap_refs=dict(guard.references),
            rank=rank,
            world_size=world_size,
            momentum_dtype=optimizer.cmuon_momentum_dtype,
            chunk_rescale_sqrt_n=optimizer.cmuon_chunk_rescale_sqrt_n,
            stats_logger=stats_logger,
            **common,
        )
    return build_adamw8bit(module, **common)


def _optimizer_learning_rate_scalars(
    optimizer: IsolatedAdamW8bit | HybridCMuon,
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
    optimizer: IsolatedAdamW8bit | HybridCMuon,
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
    optimizer: IsolatedAdamW8bit | HybridCMuon, learning_rate: float
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
        optimizer: IsolatedAdamW8bit | HybridCMuon,
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
    optimizer: IsolatedAdamW8bit | HybridCMuon,
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
    # Set the configured resume rate before the restore so the RAW loader
    # overwrites the source checkpoint's saved per-group learning rates with
    # the current configuration values; batch size and LR may change freely
    # across a resume because the loader replaces the saved group rates.
    _set_optimizer_learning_rate(optimizer, expected_rate)
    restored = restore_single_gpu_checkpoint(checkpoint, module, optimizer, identity)
    resumed_state = _resume_state_for_config(loaded.config, restored.state)
    # Keep the opt-in NS safety telemetry reporting absolute update numbers
    # across a resume (it was constructed with offset 0).
    if (
        isinstance(optimizer, _hybrid_cmuon_class())
        and optimizer.ns_telemetry is not None
    ):
        optimizer.ns_telemetry.update_offset = restored.state.trainer.successful_updates
    return replace(restored, state=resumed_state)


def _hybrid_cmuon_class() -> type:
    """Runtime access to HybridCMuon (imported lazily; the default path,
    torchao_adamw8bit, never pays for the experimental module)."""
    from sakuramoon.optim.cmuon import HybridCMuon

    return HybridCMuon


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
        stage = f"checkpoint/update-{state.successful_updates}/publish-{reason.value}"
        published = self._progress.run_on_rank(
            stage,
            0,
            lambda: self._delegate.publish_update(state, reason, cadence),
        )
        value = [str(published) if self._accelerator.is_main_process else ""]
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


class _GuardCalibrationCheckpointPublisher(ProductionSingleGpuCheckpointPublisher):
    """Fail-closed publisher for guard shadow calibration runs.

    Calibration must never write a checkpoint (no optimizer transition
    state, no guard reference state exists yet). If the cadence ever fires
    inside the calibration window, fail loudly instead of saving.
    """

    def __init__(self) -> None:
        # Intentionally skips the base binding: nothing may be published.
        self._published = False

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        raise RuntimeError(
            "guard shadow calibration must not publish checkpoints "
            f"(cadence fired at update {state.successful_updates}, "
            f"reason {reason.value}); shrink the calibration window or move "
            "the cadence point"
        )

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: object,
        state: RawCheckpointState,
    ) -> None:
        raise RuntimeError("guard shadow calibration must not apply retention")


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
    # iREPA Phase 2 gate: fail before accelerator setup, static preflight,
    # resolved-config publish, encoder loading, module/optimizer build,
    # data-service connection, and the training loop.
    require_production_irepa_readiness(config)
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
    optimizer = _build_optimizer(
        config,
        module,
        # Telemetry log lines only on the main rank (every rank sees the same
        # all-reduced gradients, so the samples would be duplicates).
        telemetry_logger=_log if is_main_process else None,
        rank=rank,
        world_size=world_size,
    )
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

    # Guard shadow calibration (Guarded Canonical candidate development):
    # SAKURAMOON_GUARD_CALIBRATION_STEPS>0 swaps the optimizer step for a
    # gradient/momentum shadow observation (no parameter update, no NS, no
    # AdamW step) and neutralizes W&B / checkpoint / sampling side effects.
    # Fail-closed: calibration is only legal on the hybrid CMuon optimizer.
    calibration: GuardCalibration | None = None
    calibration_steps = int(os.environ.get("SAKURAMOON_GUARD_CALIBRATION_STEPS", "0") or 0)
    if calibration_steps > 0:
        if not isinstance(optimizer, _hybrid_cmuon_class()):
            raise ValueError(
                "SAKURAMOON_GUARD_CALIBRATION_STEPS requires the hybrid CMuon "
                f"optimizer, got {type(optimizer)!r}"
            )
        calibration = install_guard_calibration(
            optimizer,
            steps=calibration_steps,
            output_path=artifact_root / "guard-calibration-rank0.jsonl",
            rank=rank,
            world_size=world_size,
            update_offset=restored.state.trainer.successful_updates,
        )
        if is_main_process:
            _log(
                f"[guard-calibration] shadow step 已安装: "
                f"steps={calibration_steps} out={calibration.output_path}"
            )

    # Structural/SNR pre-NS classifier calibration (D1 round, 08-31):
    # SAKURAMOON_STRUCTURAL_CALIBRATION_STEPS>0 swaps the optimizer step for
    # the structural shadow observation (pre-NS features + K production NS4
    # label runs + HCU cost accounting; NO parameter update, NO AdamW step)
    # and neutralizes the same side effects as the guard calibration.
    # Fail-closed: only legal on the hybrid CMuon optimizer.
    structural: object | None = None
    structural_steps = int(
        os.environ.get("SAKURAMOON_STRUCTURAL_CALIBRATION_STEPS", "0") or 0
    )
    if structural_steps > 0:
        if not isinstance(optimizer, _hybrid_cmuon_class()):
            raise ValueError(
                "SAKURAMOON_STRUCTURAL_CALIBRATION_STEPS requires the hybrid "
                f"CMuon optimizer, got {type(optimizer)!r}"
            )
        refs_path = os.environ.get("SAKURAMOON_STRUCTURAL_REFS_JSON", "")
        refs = None
        if refs_path:
            import json as _json

            _raw = _json.loads(Path(refs_path).read_text(encoding="utf-8"))
            refs = {
                (k.rpartition("#chunk")[0], int(k.rpartition("#chunk")[2])): float(v)
                for k, v in _raw.items()
            }
        structural = install_structural_calibration(
            optimizer,
            observations=structural_steps,
            ns_repeat=int(
                os.environ.get("SAKURAMOON_STRUCTURAL_NS_REPEAT", "5") or 5
            ),
            pi_iters=int(
                os.environ.get("SAKURAMOON_STRUCTURAL_PI_ITERS", "20") or 20
            ),
            sigma_method=os.environ.get("SAKURAMOON_STRUCTURAL_SIGMA_METHOD", "pi")
            or "pi",
            output_path=artifact_root / f"structural-calibration-rank{rank}.jsonl",
            artifact_dir=artifact_root / "structural-calibration",
            rank=rank,
            world_size=world_size,
            update_offset=restored.state.trainer.successful_updates,
            refs=refs,
            full_sample_obs=int(
                os.environ.get(
                    "SAKURAMOON_STRUCTURAL_FULL_SAMPLE_OBS", "5"
                )
                or 5
            ),
        )
        if is_main_process:
            _log(
                f"[structural-calibration] shadow step 已安装: "
                f"steps={structural_steps} refs={refs_path or '-'} "
                f"out={structural.output_path}"  # type: ignore[union-attr]
            )
    in_calibration = calibration is not None or structural is not None

    if isinstance(optimizer, _hybrid_cmuon_class()) and optimizer.forensic is not None:
        # The forensic run is forked from a live checkpoint: anchor the
        # monitor's update numbering to the restored trainer count so the
        # ring buffer / crash dumps carry the real update numbers.
        optimizer.forensic.update_offset = restored.state.trainer.successful_updates
        if is_main_process:
            _log(
                f"[cmuon-forensic] 监控就绪: update_offset="
                f"{optimizer.forensic.update_offset} dump_dir="
                f"{optimizer.forensic.fcfg.dump_dir}"
            )
    require_single_gpu_checkpoint_binding(
        config,
        restored.state,
        runtime_growth_alpha=restored.state.growth.alpha,
    )
    if is_main_process and resume is not None:
        _record_data_policy_resume_transition(loaded, repository_root, resume)
        _record_optimizer_transition(
            loaded,
            repository_root,
            resume,
            optimizer,
            restored.state.trainer.successful_updates,
        )
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
        if in_calibration:
            publisher: (
                ProductionSingleGpuCheckpointPublisher
            ) = _GuardCalibrationCheckpointPublisher()
        else:
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
                    # Calibration: no W&B run at all (design constraint).
                    _NoopTelemetry(device)
                    if in_calibration
                    else build_training_telemetry_from_config(
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
                    progress.synchronize(f"observer/update-{update}/complete")

                # Calibration: the per-update observer (sampling / W&B /
                # evaluation / named barrier) is fully bypassed; both ranks
                # skip it, so no barrier is lost.
                calibration_observer = (
                    (lambda observation: None)
                    if in_calibration
                    else observe_successful_update
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
                        successful_update_observer=calibration_observer,
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
                if not verified_checkpoints and not in_calibration:
                    raise RuntimeError(
                        "production training completed without a durable checkpoint"
                    )
                result = ProductionTrainingResult(
                    resolved_config_path,
                    preflight_report.resolve(strict=True),
                    verified_checkpoints[-1],
                    initial_update,
                    loop_result.state.successful_updates,
                    False,
                )
            except (GuardCalibrationComplete, StructuralCalibrationComplete) as done:
                # Clean calibration stop (no failure, no checkpoint).
                result = ProductionTrainingResult(
                    resolved_config_path,
                    preflight_report.resolve(strict=True),
                    resume if resume is not None else checkpoint_root,
                    initial_update,
                    initial_update + done.observations,
                    False,
                )
                if is_main_process:
                    _cal_out = (
                        calibration.output_path
                        if calibration is not None
                        else getattr(structural, "output_path", "-")
                    )
                    _log(
                        f"[calibration] 完成: {done.observations} 次观察 "
                        f"({type(done).__name__}), 记录={_cal_out}"
                    )
            except Exception as error:
                raise ProductionTrainingError(
                    "accepted production training failed"
                ) from error
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
    result = _evaluate_and_release_process_group(evaluator, update, accelerator)
    if accelerator is None and result is None:
        raise RuntimeError("native evaluation returned no result")
    return result


__all__ = [
    "ProductionPreflightError",
    "ProductionReadinessError",
    "ProductionTrainingError",
    "ProductionTrainingResult",
    "require_production_irepa_readiness",
    "run_production_evaluation",
    "run_production_single_gpu",
]
