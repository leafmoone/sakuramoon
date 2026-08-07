"""Small, explicit checks performed before the single-GPU training loop."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from sakuramoon.assets import require_local_qwen, require_local_vae
from sakuramoon.checkpoint.policy import CheckpointReason
from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    RawCheckpointState,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.data.collate import DataLeaseClient
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    require_accepted_production_batch_stream,
)
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.storage import repository_directory, repository_file_parent
from sakuramoon.train.runtime import (
    require_single_gpu_checkpoint_binding,
    require_single_gpu_config,
)

if TYPE_CHECKING:
    from sakuramoon.config.schema import RuntimeConfig
    from sakuramoon.train.runtime import SingleGpuBatchRuntime
    from sakuramoon.train.step import SingleGpuUpdateState


PREFLIGHT_CHECKS = (
    "resolved_config",
    "output_paths",
    "local_assets",
    "data_service",
    "single_gpu_runtime",
    "frozen_encoders",
    "optimizer_parameters",
)


class PreflightError(RuntimeError):
    """A required preflight check failed."""


def require_logging_checkpoint_contracts(
    config: RuntimeConfig,
    repository_root: Path,
) -> tuple[Path, Path, Path]:
    """Create the configured model and log directories."""

    require_single_gpu_config(config)
    checkpoint_root = repository_directory(
        repository_root, config.paths.checkpoint_dir
    )
    local_parent = repository_file_parent(
        repository_root, config.logging.local_jsonl_path
    )
    retry_parent = repository_file_parent(
        repository_root, config.wandb.retry_jsonl_path
    )
    local_path = local_parent / Path(config.logging.local_jsonl_path).name
    retry_path = retry_parent / Path(config.wandb.retry_jsonl_path).name
    if local_path == retry_path:
        raise PreflightError("local and W&B retry logs must use different paths")
    return checkpoint_root, local_path, retry_path


def require_static_single_gpu_preflight(
    config: RuntimeConfig,
    repository_root: Path,
) -> None:
    """Check paths and required local model directories before model loading."""

    require_logging_checkpoint_contracts(config, repository_root)
    require_local_qwen(repository_root)
    require_local_vae(repository_root)


class _SingleGpuCheckpointPublisher(Protocol):
    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path: ...

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: CheckpointManifest,
        state: RawCheckpointState,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PreflightCheckResult:
    name: str
    passed: bool
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: int
    hardware: str
    passed: bool
    dataset_id: str
    checkpoint_id: str
    checkpoint_update: int
    checks: tuple[PreflightCheckResult, ...]


@dataclass(frozen=True, slots=True, eq=False)
class RestoredSingleGpuCheckpoint:
    """A loaded raw checkpoint and the objects to which it was restored."""

    path: Path
    manifest: CheckpointManifest
    state: RawCheckpointState
    payload_bytes: int
    module: nn.Module = field(repr=False)
    optimizer: IsolatedAdamW8bit = field(repr=False)


def _require_restored_checkpoint(
    restored: object,
    *,
    module: nn.Module | None = None,
    optimizer: object | None = None,
) -> None:
    if not isinstance(restored, RestoredSingleGpuCheckpoint):
        raise PreflightError("a restored raw model is required")
    if module is not None and restored.module is not module:
        raise PreflightError("restored model differs from the training model")
    if optimizer is not None and restored.optimizer is not optimizer:
        raise PreflightError("restored optimizer differs from the training optimizer")


def restore_single_gpu_checkpoint(
    checkpoint: Path,
    module: nn.Module,
    optimizer: IsolatedAdamW8bit,
    expected: CheckpointIdentity,
) -> RestoredSingleGpuCheckpoint:
    """Load a raw checkpoint into the supplied model and optimizer."""

    from sakuramoon.checkpoint.load import (
        load_raw_checkpoint,
        read_checkpoint_manifest,
    )

    state = load_raw_checkpoint(checkpoint, module, optimizer, expected)
    manifest = read_checkpoint_manifest(checkpoint)
    if manifest.kind is not CheckpointKind.RAW or manifest.identity != expected:
        raise PreflightError("restored raw model identity changed while loading")
    payload_bytes = sum(record.size for record in manifest.files)
    if payload_bytes <= 0:
        raise PreflightError("restored raw model is empty")
    return RestoredSingleGpuCheckpoint(
        path=checkpoint.resolve(strict=True),
        manifest=manifest,
        state=state,
        payload_bytes=payload_bytes,
        module=module,
        optimizer=optimizer,
    )


class ProductionSingleGpuCheckpointPublisher:
    """Save successful updates and retain the configured rolling model set."""

    def __init__(
        self,
        *,
        checkpoint_root: Path,
        resolved_config: bytes,
        module: nn.Module,
        optimizer: IsolatedAdamW8bit,
        restored_checkpoint: RestoredSingleGpuCheckpoint,
        accepted_checkpoint_ids: frozenset[str],
        retention_slots: int,
    ) -> None:
        _require_restored_checkpoint(
            restored_checkpoint, module=module, optimizer=optimizer
        )
        if type(resolved_config) is not bytes or not resolved_config:
            raise TypeError("resolved config must be nonempty bytes")
        if type(retention_slots) is not int or retention_slots <= 0:
            raise ValueError("model retention slots must be positive")
        self._checkpoint_root = checkpoint_root.resolve(strict=True)
        self._resolved_config = resolved_config
        self._module = module
        self._optimizer = optimizer
        self._base_identity = restored_checkpoint.manifest.identity
        self._restored_state = restored_checkpoint.state
        self._accepted_checkpoint_ids = accepted_checkpoint_ids
        self._retention_slots = retention_slots
        self._pending: dict[Path, tuple[CheckpointIdentity, RawCheckpointState]] = {}

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        from sakuramoon.checkpoint.save import save_raw_checkpoint

        restored = self._restored_state
        if (
            state.successful_updates <= restored.trainer.successful_updates
            or state.successful_updates
            > restored.stage_budget.terminal_successful_update
            or cadence.last_successful_update != state.successful_updates
        ):
            raise ValueError("model save update is outside the active training run")
        identity = replace(
            self._base_identity,
            checkpoint_id=f"raw-{state.successful_updates}-{reason.value}",
            update=state.successful_updates,
        )
        raw_state = RawCheckpointState(
            trainer=state,
            growth=restored.growth,
            stage_budget=restored.stage_budget,
            checkpoint_cadence=cadence,
        )
        result = save_raw_checkpoint(
            self._checkpoint_root,
            identity,
            self._module,
            self._optimizer,
            raw_state,
            resolved_config=self._resolved_config,
        )
        path = result.path.resolve(strict=True)
        self._pending[path] = (identity, raw_state)
        return path

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: CheckpointManifest,
        state: RawCheckpointState,
    ) -> None:
        from sakuramoon.checkpoint.policy import (
            apply_raw_retention,
            plan_raw_retention,
        )

        expected = self._pending.pop(checkpoint, None)
        if expected is None or expected != (manifest.identity, state):
            raise PreflightError("model retention requires the model just saved")
        plan = plan_raw_retention(
            self._checkpoint_root,
            accepted_checkpoint_ids=self._accepted_checkpoint_ids,
            rolling_slots=self._retention_slots,
        )
        apply_raw_retention(
            self._checkpoint_root,
            plan,
            accepted_checkpoint_ids=self._accepted_checkpoint_ids,
            rolling_slots=self._retention_slots,
        )


@dataclass(frozen=True, slots=True)
class _PreflightBindings:
    config: object
    batches: AcceptedProductionBatchStream
    runtime: object
    qwen: object
    vae: object
    module: nn.Module
    optimizer: object
    restored: RestoredSingleGpuCheckpoint
    checkpoint_publisher: object


@dataclass(frozen=True, slots=True, eq=False)
class AcceptedPreflight:
    report: PreflightReport
    bindings: _PreflightBindings


def require_accepted_preflight(
    value: object,
    *,
    config: object | None = None,
    batches: AcceptedProductionBatchStream | None = None,
    runtime: object | None = None,
    qwen: object | None = None,
    vae: object | None = None,
    module: nn.Module | None = None,
    optimizer: object | None = None,
    restored: RestoredSingleGpuCheckpoint | None = None,
    checkpoint_publisher: object | None = None,
) -> None:
    if not isinstance(value, AcceptedPreflight) or not value.report.passed:
        raise PreflightError("training requires a passed preflight")
    bindings = value.bindings
    expected = (
        (config, bindings.config),
        (batches, bindings.batches),
        (runtime, bindings.runtime),
        (qwen, bindings.qwen),
        (vae, bindings.vae),
        (module, bindings.module),
        (optimizer, bindings.optimizer),
        (restored, bindings.restored),
        (checkpoint_publisher, bindings.checkpoint_publisher),
    )
    if any(requested is not None and requested is not bound for requested, bound in expected):
        raise PreflightError("training resources differ from the passed preflight")


@dataclass(frozen=True, slots=True)
class SingleGpuPreflightPlan:
    checks: tuple[tuple[str, Callable[[], None]], ...]
    dataset_id: str
    bindings: _PreflightBindings


def build_single_gpu_preflight_checks(
    loaded: LoadedConfig,
    *,
    repository_root: Path,
    resolved_config_path: Path,
    data_client: DataLeaseClient,
    batches: AcceptedProductionBatchStream,
    runtime: SingleGpuBatchRuntime,
    qwen: object,
    vae: object,
    trainable_module: torch.nn.Module,
    optimizer: object,
    restored_checkpoint: RestoredSingleGpuCheckpoint,
    checkpoint_publisher: _SingleGpuCheckpointPublisher,
) -> SingleGpuPreflightPlan:
    """Build cheap checks for resources that are about to enter training."""

    accepted_batches = require_accepted_production_batch_stream(batches)
    _require_restored_checkpoint(
        restored_checkpoint, module=trainable_module, optimizer=optimizer
    )
    stream_identity = accepted_batches.identity
    if (
        stream_identity.dataset_id != data_client.identity.dataset_id
        or stream_identity.session_id != data_client.identity.session_id
    ):
        raise PreflightError("training data stream differs from the data service")
    if (
        runtime.composite is not trainable_module
        or runtime.qwen is not qwen
        or runtime.vae is not vae
    ):
        raise PreflightError("training runtime resources are inconsistent")

    def resolved_config() -> None:
        if resolved_config_path.read_text(encoding="utf-8") != loaded.resolved_toml:
            raise ValueError("resolved config file differs from the loaded TOML")

    def output_paths() -> None:
        require_logging_checkpoint_contracts(loaded.config, repository_root)

    def local_assets() -> None:
        require_local_qwen(repository_root)
        require_local_vae(repository_root)

    def data_service() -> None:
        if data_client.health():
            raise RuntimeError("data service has no remaining training shards")
        depth = accepted_batches.ready_batch_depth_snapshot()
        print(f"[preflight] ready_batches={depth}", flush=True)

    def single_gpu_runtime() -> None:
        require_single_gpu_config(loaded.config)
        require_single_gpu_checkpoint_binding(
            loaded.config,
            restored_checkpoint.state,
            runtime_growth_alpha=runtime.growth_alpha,
        )
        expected_devices = loaded.config.distributed.world_size
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != expected_devices
            or int(runtime.device.index or 0) >= expected_devices
        ):
            raise RuntimeError("training accelerator topology differs from config")

    def frozen_encoders() -> None:
        for name, encoder in (("Qwen", qwen), ("VAE", vae)):
            if not isinstance(encoder, nn.Module) or encoder.training:
                raise RuntimeError(f"{name} encoder is not frozen")
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError(f"{name} encoder exposes trainable parameters")

    def optimizer_parameters() -> None:
        from sakuramoon.checkpoint.artifact import validate_optimizer_coverage

        if not isinstance(optimizer, IsolatedAdamW8bit):
            raise TypeError("training requires the configured AdamW8 optimizer")
        validate_optimizer_coverage(
            trainable_module,
            tuple((spec.name, spec.parameter) for spec in optimizer.audit.specs),
        )

    checks = (
        ("resolved_config", resolved_config),
        ("output_paths", output_paths),
        ("local_assets", local_assets),
        ("data_service", data_service),
        ("single_gpu_runtime", single_gpu_runtime),
        ("frozen_encoders", frozen_encoders),
        ("optimizer_parameters", optimizer_parameters),
    )
    return SingleGpuPreflightPlan(
        checks=checks,
        dataset_id=data_client.identity.dataset_id,
        bindings=_PreflightBindings(
            config=loaded.config,
            batches=accepted_batches,
            runtime=runtime,
            qwen=qwen,
            vae=vae,
            module=trainable_module,
            optimizer=optimizer,
            restored=restored_checkpoint,
            checkpoint_publisher=checkpoint_publisher,
        ),
    )


def _write_report(report: PreflightReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(6)}.tmp"
    )
    payload = json.dumps(asdict(report), ensure_ascii=False, separators=(",", ":"))
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_single_gpu_preflight(
    plan: SingleGpuPreflightPlan,
    destination: Path,
) -> AcceptedPreflight:
    """Run the small preflight set and write a readable report."""

    results: list[PreflightCheckResult] = []
    for name, check in plan.checks:
        print(f"[preflight] {name}", flush=True)
        try:
            check()
        except Exception as error:
            results.append(PreflightCheckResult(name, False, type(error).__name__))
            restored = plan.bindings.restored
            report = PreflightReport(
                schema_version=1,
                hardware=f"{plan.bindings.config.distributed.world_size}GPU",
                passed=False,
                dataset_id=plan.dataset_id,
                checkpoint_id=restored.manifest.identity.checkpoint_id,
                checkpoint_update=restored.state.trainer.successful_updates,
                checks=tuple(results),
            )
            _write_report(report, destination)
            print(f"[preflight] {name} FAILED: {error}", flush=True)
            raise PreflightError(f"preflight failed at {name}: {error}") from error
        results.append(PreflightCheckResult(name, True))
        print(f"[preflight] {name} OK", flush=True)
    restored = plan.bindings.restored
    report = PreflightReport(
        schema_version=1,
        hardware=f"{plan.bindings.config.distributed.world_size}GPU",
        passed=True,
        dataset_id=plan.dataset_id,
        checkpoint_id=restored.manifest.identity.checkpoint_id,
        checkpoint_update=restored.state.trainer.successful_updates,
        checks=tuple(results),
    )
    _write_report(report, destination)
    return AcceptedPreflight(report, plan.bindings)


__all__ = [
    "AcceptedPreflight",
    "PreflightError",
    "ProductionSingleGpuCheckpointPublisher",
    "RestoredSingleGpuCheckpoint",
    "build_single_gpu_preflight_checks",
    "require_accepted_preflight",
    "require_static_single_gpu_preflight",
    "restore_single_gpu_checkpoint",
    "run_single_gpu_preflight",
]
