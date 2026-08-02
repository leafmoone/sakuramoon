"""Ordered, non-bypassable single-GPU preflight orchestration."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import weakref
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Never, Protocol, SupportsIndex, cast

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
    manifest_from_dict,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.data.collate import DataLeaseClient
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    require_accepted_production_batch_stream,
)
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.storage import (
    repository_directory,
    repository_file_parent,
    require_training_storage,
)
from sakuramoon.train.runtime import (
    require_single_gpu_checkpoint_binding,
    require_single_gpu_config,
)

if TYPE_CHECKING:
    from sakuramoon.config.schema import RuntimeConfig
    from sakuramoon.data.collate import TrainingBatch
    from sakuramoon.train.runtime import SingleGpuBatchRuntime
    from sakuramoon.train.step import SingleGpuUpdateState

PREFLIGHT_CHECKS = (
    "resolved_config",
    "production_contracts",
    "local_assets",
    "dataset_revision",
    "ready_batch_depth",
    "single_gpu_runtime",
    "storage_capacity",
    "frozen_encoders",
    "parameter_schema",
    "image_shapes",
    "text_shapes",
    "zero_update_loss",
    "optimizer_step",
    "sample",
    "checkpoint_round_trip",
)


class PreflightError(RuntimeError):
    """A mandatory preflight check failed."""


def require_logging_checkpoint_contracts(
    config: RuntimeConfig,
    repository_root: Path,
) -> tuple[Path, Path, Path]:
    """Bind production logs and raw checkpoints to durable repository paths."""

    require_single_gpu_config(config)
    if not config.wandb.enabled or not config.timing.enabled:
        raise PreflightError("production logging and W&B telemetry must be enabled")
    checkpoint = config.checkpoint
    if (
        checkpoint.kind != "raw"
        or type(checkpoint.full_every_updates) is not int
        or checkpoint.full_every_updates <= 0
        or type(checkpoint.slots) is not int
        or checkpoint.slots <= 0
        or not checkpoint.atomic_complete_marker
        or not checkpoint.checksum_required
        or not checkpoint.canonical_fqn
        or config.storage.checkpoint_copies != checkpoint.slots + 1
    ):
        raise PreflightError("production raw checkpoint contract is invalid")
    checkpoint_root = repository_directory(repository_root, config.paths.checkpoint_dir)
    local_parent = repository_file_parent(
        repository_root, config.logging.local_jsonl_path
    )
    retry_parent = repository_file_parent(
        repository_root, config.wandb.retry_jsonl_path
    )
    local_path = local_parent / Path(config.logging.local_jsonl_path).name
    retry_path = retry_parent / Path(config.wandb.retry_jsonl_path).name
    if local_path == retry_path:
        raise PreflightError("local metric and W&B retry paths must differ")
    for role, path in (("local metric", local_path), ("W&B retry", retry_path)):
        if path.is_symlink():
            raise PreflightError(f"{role} path may not be a symlink")
        if path.exists():
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise PreflightError(f"{role} path must be a regular file")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise PreflightError(f"{role} path must use mode 0600")
    return checkpoint_root, local_path, retry_path


def require_static_single_gpu_preflight(
    config: RuntimeConfig,
    repository_root: Path,
) -> None:
    """Reject static production blockers before a fresh RAW is published."""

    require_single_gpu_config(config)
    require_logging_checkpoint_contracts(config, repository_root)
    require_training_storage(
        config,
        repository_root,
        checkpoint_payload_bytes=config.storage.measured_raw_checkpoint_bytes,
    )
    require_local_qwen(repository_root)
    require_local_vae(repository_root)


class _SingleGpuCheckpointPublisher(Protocol):
    """One bound publisher with distinct preflight and update operations."""

    def publish_preflight(
        self, identity: CheckpointIdentity, state: RawCheckpointState
    ) -> Path: ...

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

    def discard_preflight(self, checkpoint: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class PreflightCheckResult:
    name: str
    passed: bool
    error_type: str | None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: int
    hardware: str
    passed: bool
    resolved_config_sha256: str
    manifest_id: str
    service_session_sha256: str
    checkpoint_id: str
    checkpoint_update: int
    checks: tuple[PreflightCheckResult, ...]


class RestoredSingleGpuCheckpoint:
    """Process-local proof that a concrete T044 RAW checkpoint was restored."""

    __slots__ = (
        "__weakref__",
        "_manifest",
        "_module",
        "_optimizer",
        "_owner_pid",
        "_path",
        "_payload_bytes",
        "_state",
        "_token",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        del _args, _kwargs
        raise TypeError("restored checkpoint handles are created only by restore")

    @property
    def manifest(self) -> CheckpointManifest:
        _require_restored_checkpoint(self)
        return self._manifest

    @property
    def path(self) -> Path:
        _require_restored_checkpoint(self)
        return self._path

    @property
    def payload_bytes(self) -> int:
        _require_restored_checkpoint(self)
        return self._payload_bytes

    @property
    def state(self) -> RawCheckpointState:
        _require_restored_checkpoint(self)
        return self._state

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise PreflightError("restored checkpoint handles cannot be serialized")


_RESTORED_CHECKPOINTS: weakref.WeakValueDictionary[str, RestoredSingleGpuCheckpoint] = (
    weakref.WeakValueDictionary()
)


def _require_restored_checkpoint(
    value: RestoredSingleGpuCheckpoint,
    *,
    module: nn.Module | None = None,
    optimizer: object | None = None,
) -> None:
    try:
        token = value._token
        owner_pid = value._owner_pid
    except AttributeError:
        raise PreflightError(
            "a restored process-local RAW checkpoint is required"
        ) from None
    if (
        type(value) is not RestoredSingleGpuCheckpoint
        or owner_pid != os.getpid()
        or _RESTORED_CHECKPOINTS.get(token) is not value
        or (module is not None and value._module is not module)
        or (optimizer is not None and value._optimizer is not optimizer)
    ):
        raise PreflightError("a restored process-local RAW checkpoint is required")


def restore_single_gpu_checkpoint(
    checkpoint: Path,
    module: nn.Module,
    optimizer: IsolatedAdamW8bit,
    expected: CheckpointIdentity,
) -> RestoredSingleGpuCheckpoint:
    """Fully validate and restore one T044 RAW artifact before data connection."""

    from sakuramoon.checkpoint.load import load_raw_checkpoint

    if type(optimizer) is not IsolatedAdamW8bit:
        raise TypeError("production RAW restore requires IsolatedAdamW8bit")
    state = load_raw_checkpoint(checkpoint, module, optimizer, expected)
    try:
        document = json.loads((checkpoint / "manifest.json").read_bytes())
        manifest = manifest_from_dict(document)
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(
            "restored checkpoint manifest became unreadable"
        ) from error
    if manifest.kind is not CheckpointKind.RAW or manifest.identity != expected:
        raise PreflightError("restored checkpoint identity changed after validation")
    payload_bytes = sum(record.size for record in manifest.files)
    if payload_bytes <= 0:
        raise PreflightError("restored RAW checkpoint payload is empty")
    restored = object.__new__(RestoredSingleGpuCheckpoint)
    restored._manifest = manifest
    restored._module = module
    restored._optimizer = optimizer
    restored._owner_pid = os.getpid()
    restored._path = checkpoint
    restored._payload_bytes = payload_bytes
    restored._state = state
    restored._token = secrets.token_hex(32)
    _RESTORED_CHECKPOINTS[restored._token] = restored
    return restored


class ProductionSingleGpuCheckpointPublisher:
    """Publish T044 RAW artifacts bound to one restored training state."""

    __slots__ = (
        "_accepted_checkpoint_ids",
        "_base_identity",
        "_checkpoint_root",
        "_module",
        "_optimizer",
        "_owner_pid",
        "_pending_update_paths",
        "_preflight_paths",
        "_resolved_config",
        "_restored_state",
        "_retention_slots",
    )

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
            restored_checkpoint,
            module=module,
            optimizer=optimizer,
        )
        if (
            not checkpoint_root.is_absolute()
            or checkpoint_root.is_symlink()
            or not checkpoint_root.is_dir()
        ):
            raise ValueError("checkpoint publication root must be a real directory")
        if type(resolved_config) is not bytes or not resolved_config:
            raise TypeError("resolved checkpoint config must be nonempty bytes")
        if type(optimizer) is not IsolatedAdamW8bit:
            raise TypeError(
                "production checkpoint publication requires IsolatedAdamW8bit"
            )
        if type(accepted_checkpoint_ids) is not frozenset or any(
            type(value) is not str or not value for value in accepted_checkpoint_ids
        ):
            raise TypeError(
                "accepted checkpoint IDs must be an explicit string frozenset"
            )
        if type(retention_slots) is not int or retention_slots <= 0:
            raise TypeError(
                "checkpoint retention slots must be an explicit positive integer"
            )
        identity = restored_checkpoint.manifest.identity
        if hashlib.sha256(resolved_config).hexdigest() != identity.config_sha256:
            raise ValueError(
                "checkpoint publisher resolved identity differs from restore"
            )
        self._accepted_checkpoint_ids = accepted_checkpoint_ids
        self._base_identity = identity
        self._checkpoint_root = checkpoint_root.resolve(strict=True)
        self._module = module
        self._optimizer = optimizer
        self._owner_pid = os.getpid()
        self._pending_update_paths: dict[
            Path, tuple[CheckpointIdentity, RawCheckpointState]
        ] = {}
        self._preflight_paths: dict[Path, CheckpointIdentity] = {}
        self._resolved_config = resolved_config
        self._retention_slots = retention_slots
        self._restored_state = restored_checkpoint.state

    def _require_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise PreflightError("checkpoint publisher cannot cross a process boundary")

    def _save(
        self,
        identity: CheckpointIdentity,
        state: RawCheckpointState,
    ) -> Path:
        from sakuramoon.checkpoint.save import save_raw_checkpoint

        result = save_raw_checkpoint(
            self._checkpoint_root,
            identity,
            self._module,
            self._optimizer,
            state,
            resolved_config=self._resolved_config,
        )
        return result.path

    def publish_preflight(
        self,
        identity: CheckpointIdentity,
        state: RawCheckpointState,
    ) -> Path:
        self._require_owner()
        base = self._base_identity
        if (
            identity.update != base.update
            or identity.config_sha256 != base.config_sha256
            or identity.dependency_sha256 != base.dependency_sha256
            or identity.parameter_schema_sha256 != base.parameter_schema_sha256
            or identity.checkpoint_id == base.checkpoint_id
            or state != self._restored_state
        ):
            raise ValueError("preflight RAW publication differs from restored state")
        path = self._save(identity, state)
        self._preflight_paths[path] = identity
        return path

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        self._require_owner()
        if type(reason) is not CheckpointReason:
            raise TypeError("checkpoint publication reason is invalid")
        restored = self._restored_state
        if (
            state.successful_updates <= restored.trainer.successful_updates
            or state.successful_updates
            > restored.stage_budget.terminal_successful_update
            or cadence.last_successful_update != state.successful_updates
        ):
            raise ValueError(
                "checkpoint publication state is outside the restored budget"
            )
        identity = replace(
            self._base_identity,
            checkpoint_id=(
                f"raw-{state.successful_updates}-{reason.value}-{secrets.token_hex(6)}"
            ),
            update=state.successful_updates,
        )
        raw_state = RawCheckpointState(
            trainer=state,
            growth=restored.growth,
            stage_budget=restored.stage_budget,
            checkpoint_cadence=cadence,
        )
        path = self._save(identity, raw_state).resolve(strict=True)
        self._pending_update_paths[path] = (identity, raw_state)
        return path

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: CheckpointManifest,
        state: RawCheckpointState,
    ) -> None:
        """Apply retention only after runtime readback of this publication."""

        self._require_owner()
        expected = self._pending_update_paths.get(checkpoint)
        if expected is None:
            raise PreflightError("retention requires a pending update publication")
        expected_identity, expected_state = expected
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_dir()
            or checkpoint.parent.resolve(strict=True) != self._checkpoint_root
            or checkpoint.resolve(strict=True) != checkpoint
            or manifest.kind is not CheckpointKind.RAW
            or manifest.identity != expected_identity
            or state != expected_state
        ):
            raise PreflightError("verified RAW publication changed before retention")
        from sakuramoon.checkpoint.policy import (
            apply_raw_retention,
            plan_raw_retention,
        )

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
        self._pending_update_paths.pop(checkpoint)

    def discard_preflight(self, checkpoint: Path) -> None:
        self._require_owner()
        expected = self._preflight_paths.get(checkpoint)
        if expected is None:
            raise ValueError("checkpoint is not an outstanding preflight RAW")
        if (
            checkpoint.parent.resolve(strict=True) != self._checkpoint_root
            or checkpoint.is_symlink()
            or not checkpoint.is_dir()
        ):
            raise ValueError("preflight RAW deletion target changed after publication")
        manifest = checkpoint / "manifest.json"
        try:
            observed = manifest_from_dict(json.loads(manifest.read_bytes()))
        except (OSError, json.JSONDecodeError) as error:
            raise PreflightError(
                "preflight RAW manifest is unreadable at cleanup"
            ) from error
        if observed.kind is not CheckpointKind.RAW or observed.identity != expected:
            raise ValueError("preflight RAW identity changed before cleanup")
        shutil.rmtree(checkpoint)
        directory = os.open(self._checkpoint_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._preflight_paths.pop(checkpoint)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise PreflightError("checkpoint publishers cannot be serialized")


@dataclass(frozen=True, slots=True)
class _PreflightBindings:
    config: object
    resolved_config_sha256: str
    batches: AcceptedProductionBatchStream
    runtime: object
    qwen: object
    vae: object
    module: nn.Module
    optimizer: object
    restored: RestoredSingleGpuCheckpoint
    checkpoint_publisher: object


class AcceptedPreflight:
    """Process-local proof that every mandatory preflight check passed."""

    __slots__ = ("__weakref__", "_bindings", "_owner_pid", "_token", "report")

    report: PreflightReport

    def __init__(self, report: PreflightReport) -> None:
        del report
        raise TypeError("accepted preflight handles are created only by preflight")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise PreflightError("accepted preflight handles cannot be serialized")


_ACCEPTED_PREFLIGHTS: weakref.WeakValueDictionary[str, AcceptedPreflight] = (
    weakref.WeakValueDictionary()
)


def _accepted_preflight(
    report: PreflightReport, bindings: _PreflightBindings
) -> AcceptedPreflight:
    accepted = object.__new__(AcceptedPreflight)
    accepted._bindings = bindings
    accepted._owner_pid = os.getpid()
    accepted._token = secrets.token_hex(32)
    accepted.report = report
    _ACCEPTED_PREFLIGHTS[accepted._token] = accepted
    return accepted


def require_accepted_preflight(
    value: AcceptedPreflight,
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
    try:
        token = value._token
        owner_pid = value._owner_pid
        bindings = value._bindings
    except AttributeError:
        raise PreflightError(
            "training requires an accepted process-local preflight"
        ) from None
    if (
        type(value) is not AcceptedPreflight
        or owner_pid != os.getpid()
        or _ACCEPTED_PREFLIGHTS.get(token) is not value
        or value.report.passed is not True
        or (config is not None and bindings.config is not config)
        or (batches is not None and bindings.batches is not batches)
        or (runtime is not None and bindings.runtime is not runtime)
        or (qwen is not None and bindings.qwen is not qwen)
        or (vae is not None and bindings.vae is not vae)
        or (module is not None and bindings.module is not module)
        or (optimizer is not None and bindings.optimizer is not optimizer)
        or (restored is not None and bindings.restored is not restored)
        or (
            checkpoint_publisher is not None
            and bindings.checkpoint_publisher is not checkpoint_publisher
        )
    ):
        raise PreflightError("training requires an accepted process-local preflight")


class SingleGpuPreflightPlan:
    """One-shot concrete plan issued by the production preflight builder."""

    __slots__ = (
        "__weakref__",
        "_bindings",
        "_checks",
        "_manifest_id",
        "_owner_pid",
        "_service_session_sha256",
        "_token",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        del _args, _kwargs
        raise TypeError("preflight plans are created only by the concrete builder")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise PreflightError("preflight plans cannot be serialized")


_PREFLIGHT_PLANS: weakref.WeakValueDictionary[str, SingleGpuPreflightPlan] = (
    weakref.WeakValueDictionary()
)


_WORKLOAD_CHECKS = (
    "image_shapes",
    "text_shapes",
    "zero_update_loss",
    "optimizer_step",
    "sample",
)

_ATTENTION_BACKEND_METADATA = {
    "fa4_varlen": "fa4_varlen",
    "dense_sdpa_reference": "dense_sdpa",
}


def _require_attention_backend_binding(
    config: RuntimeConfig,
    trainable_module: nn.Module,
) -> None:
    """Bind the resolved attention selection to the assembled DiT implementation."""

    try:
        configured = config.kernels.attention_backend
        dit = trainable_module.dit
    except AttributeError as error:
        raise TypeError("attention backend binding resources are incomplete") from error
    if type(configured) is not str or configured not in _ATTENTION_BACKEND_METADATA:
        raise ValueError("configured attention backend is unsupported")
    artifact_method = getattr(dit, "artifact_config", None)
    if not callable(artifact_method):
        raise TypeError("DiT artifact config is unavailable")
    artifact = artifact_method()
    if type(artifact) is not dict:
        raise TypeError("DiT artifact config is malformed")
    observed = cast(dict[object, object], artifact).get("attention_backend")
    if type(observed) is not str:
        raise TypeError("DiT artifact attention backend is missing or malformed")
    if observed != _ATTENTION_BACKEND_METADATA[configured]:
        raise ValueError("configured attention backend differs from DiT artifact")


class SingleGpuPreflightWorkload:
    """Process-local capability carrying the fixed concrete GPU probes."""

    __slots__ = (
        "__weakref__",
        "_checks",
        "_config",
        "_module",
        "_optimizer",
        "_owner_pid",
        "_qwen",
        "_runtime",
        "_token",
        "_vae",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        del _args, _kwargs
        raise TypeError("verified preflight workloads are issued internally")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise PreflightError("verified preflight workloads cannot be serialized")


_VERIFIED_WORKLOADS: weakref.WeakValueDictionary[str, SingleGpuPreflightWorkload] = (
    weakref.WeakValueDictionary()
)


def _issue_verified_preflight_workload(  # pyright: ignore[reportUnusedFunction]
    checks: Mapping[str, Callable[[], None]],
    *,
    config: object,
    runtime: object,
    trainable_module: nn.Module,
    optimizer: object,
) -> SingleGpuPreflightWorkload:
    """Test/internal hook; production assembly must run the named probes first."""

    if tuple(checks) != _WORKLOAD_CHECKS or set(checks) != set(_WORKLOAD_CHECKS):
        raise ValueError("verified workload requires every fixed probe in order")
    workload = object.__new__(SingleGpuPreflightWorkload)
    workload._checks = tuple((name, checks[name]) for name in _WORKLOAD_CHECKS)
    workload._config = config
    workload._runtime = runtime
    workload._qwen = getattr(runtime, "qwen", None)
    workload._vae = getattr(runtime, "vae", None)
    workload._module = trainable_module
    workload._optimizer = optimizer
    workload._owner_pid = os.getpid()
    workload._token = secrets.token_hex(32)
    _VERIFIED_WORKLOADS[workload._token] = workload
    return workload


def _synthetic_preflight_batch(
    config: RuntimeConfig,
    *,
    height: int,
    width: int,
    dense_length: int,
    main_length: int,
    empty_condition: bool = False,
) -> TrainingBatch:
    from sakuramoon.data.caption import CaptionDropoutCounts
    from sakuramoon.data.collate import TrainingBatch
    from sakuramoon.data.pipeline import ImageAudit, RngIdentity

    if not 0 < main_length <= dense_length:
        raise ValueError("synthetic main-token length is outside the dense shape")
    batch_size = config.stage.local_batch
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("production preflight local batch is invalid")
    attention_mask = torch.zeros((batch_size, dense_length), dtype=torch.bool)
    attention_mask[:, :main_length] = True
    main_token_indices = torch.arange(main_length, dtype=torch.long).repeat(
        batch_size, 1
    )
    audit = ImageAudit(
        width,
        height,
        width,
        height,
        (0, 0, width, height),
        1.0,
    )
    return TrainingBatch(
        images=torch.zeros((batch_size, 3, height, width), dtype=torch.uint8),
        input_ids=torch.zeros((batch_size, dense_length), dtype=torch.long),
        attention_mask=attention_mask,
        main_token_indices=main_token_indices,
        main_mask=torch.ones((batch_size, main_length), dtype=torch.bool),
        main_token_lengths=(main_length,) * batch_size,
        artist_token_indices=torch.empty((batch_size, 0), dtype=torch.long),
        artist_mask=torch.empty((batch_size, 0), dtype=torch.bool),
        active_style_sample_indices=torch.empty((0,), dtype=torch.long),
        sample_ids=torch.arange(1, batch_size + 1, dtype=torch.long),
        target_height=height,
        target_width=width,
        dense_length=dense_length,
        use_null_style=torch.ones((batch_size,), dtype=torch.bool),
        all_condition_dropped=torch.full(
            (batch_size,), empty_condition, dtype=torch.bool
        ),
        dropout_hits=CaptionDropoutCounts(
            batch_size if empty_condition else 0,
            *(0 for _ in range(11)),
        ),
        source_shards=("synthetic/preflight.tar",) * batch_size,
        audits=(audit,) * batch_size,
        rng_identities=tuple(
            RngIdentity(config.run.seed, "S0", 0, sample_id, 1, 1)
            for sample_id in range(1, batch_size + 1)
        ),
    )


def build_single_gpu_preflight_workload(
    config: RuntimeConfig,
    *,
    runtime: SingleGpuBatchRuntime,
    trainable_module: nn.Module,
    optimizer: IsolatedAdamW8bit,
) -> SingleGpuPreflightWorkload:
    """Build the non-injectable image/text/update/sample GPU workload."""

    from sakuramoon.config.schema import RuntimeConfig
    from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
    from sakuramoon.data.serialize import EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS
    from sakuramoon.encoders.mage_vae import FrozenMageVAE
    from sakuramoon.encoders.qwen import FrozenQwenEncoder
    from sakuramoon.objective.flow import sample_noise, x_prediction_to_velocity
    from sakuramoon.sampling.sampler import sample_profile
    from sakuramoon.train.runtime import SingleGpuBatchRuntime
    from sakuramoon.train.step import (
        SingleGpuStep,
        SingleGpuUpdateState,
        TrainableComposite,
    )

    if (
        not isinstance(config, RuntimeConfig)  # pyright: ignore[reportUnnecessaryIsInstance]
        or type(runtime) is not SingleGpuBatchRuntime
        or type(runtime.qwen) is not FrozenQwenEncoder
        or type(runtime.vae) is not FrozenMageVAE
        or type(trainable_module) is not TrainableComposite
        or trainable_module is not runtime.composite
        or type(optimizer) is not IsolatedAdamW8bit
    ):
        raise TypeError("production preflight workload resources are invalid")
    require_single_gpu_config(config)
    _require_attention_backend_binding(config, trainable_module)
    image_shapes = scale_buckets(
        generate_base_buckets(config.data.buckets), config.stage.resolution
    )
    text_shapes = tuple(
        zip(
            config.caption.qwen_dense_lengths,
            config.caption.condition_buckets,
            strict=True,
        )
    )
    square = next(
        (shape for shape in image_shapes if shape.height == shape.width),
        None,
    )
    if square is None:
        raise ValueError("production bucket registry has no square probe shape")

    def require_measurement(batch: TrainingBatch, *, backward: bool) -> None:
        measurement = runtime.measure(batch)
        if (
            measurement.per_sample_loss.shape != (config.stage.local_batch,)
            or measurement.per_sample_loss.dtype is not torch.float32
            or not bool(torch.isfinite(measurement.per_sample_loss).all().item())
        ):
            raise RuntimeError("production preflight measurement is invalid")
        if not backward:
            return
        if not measurement.per_sample_loss.requires_grad:
            raise RuntimeError("production shape probe lacks a trainable graph")
        try:
            measurement.per_sample_loss.sum().backward()  # pyright: ignore[reportUnknownMemberType]
        finally:
            optimizer.zero_grad(set_to_none=True)
        if any(
            parameter.grad is not None for parameter in trainable_module.parameters()
        ):
            raise RuntimeError("production shape probe left durable gradients")

    def check_image_shapes() -> None:
        dense_length, _condition_bucket = text_shapes[0]
        for shape in image_shapes:
            require_measurement(
                _synthetic_preflight_batch(
                    config,
                    height=shape.height,
                    width=shape.width,
                    dense_length=dense_length,
                    main_length=dense_length,
                ),
                backward=True,
            )

    def check_text_shapes() -> None:
        for dense_length, _condition_bucket in text_shapes:
            require_measurement(
                _synthetic_preflight_batch(
                    config,
                    height=square.height,
                    width=square.width,
                    dense_length=dense_length,
                    main_length=dense_length,
                ),
                backward=True,
            )
        shortest_dense, _shortest_bucket = text_shapes[0]
        require_measurement(
            _synthetic_preflight_batch(
                config,
                height=square.height,
                width=square.width,
                dense_length=shortest_dense,
                main_length=EXPECTED_PREFIX_TOKENS + EXPECTED_SUFFIX_TOKENS,
                empty_condition=True,
            ),
            backward=True,
        )

    def check_zero_update_loss() -> None:
        dense_length, _condition_bucket = text_shapes[0]
        measurement = runtime.measure(
            _synthetic_preflight_batch(
                config,
                height=square.height,
                width=square.width,
                dense_length=dense_length,
                main_length=dense_length,
            )
        )
        if not measurement.per_sample_loss.requires_grad:
            raise RuntimeError("zero-update loss lacks a trainable graph")

    def check_optimizer_step() -> None:
        dense_length, _condition_bucket = text_shapes[0]
        step = SingleGpuStep(
            trainable_module,
            optimizer,
            accumulation_steps=config.stage.accumulation,
            state=SingleGpuUpdateState.initial(),
        )
        for _ in range(config.stage.accumulation):
            measurement = runtime.measure(
                _synthetic_preflight_batch(
                    config,
                    height=square.height,
                    width=square.width,
                    dense_length=dense_length,
                    main_length=dense_length,
                )
            )
            step.backward(measurement.per_sample_loss)
        result = step.finish_update()
        if result.state.successful_updates != 1:
            raise RuntimeError("preflight optimizer step did not complete")

    def check_sample() -> None:
        dense_length, _condition_bucket = text_shapes[0]
        batch = _synthetic_preflight_batch(
            config,
            height=square.height,
            width=square.width,
            dense_length=dense_length,
            main_length=dense_length,
        )
        with torch.inference_mode():
            prepared = runtime.prepare(batch)
            clean = torch.stack(prepared.clean_latents)
            initial_noise = sample_noise(
                clean,
                noise_scale=runtime.noise_scale,
                generator=runtime.generator,
            )

            def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                inputs = replace(
                    prepared.inputs,
                    latents=tuple(state.to(torch.bfloat16).unbind(0)),
                    timestep=timestep,
                )
                prediction = torch.stack(trainable_module(inputs))
                return x_prediction_to_velocity(
                    prediction,
                    state,
                    timestep,
                    t_eps=runtime.t_eps,
                )

            sampled = sample_profile(
                velocity,
                initial_noise,
                profile=config.sampling.profile,
            )
            decode = getattr(runtime.vae, "decode", None)
            if not callable(decode):
                raise TypeError("production Mage-VAE must expose decode")
            image = decode(sampled.state.to(torch.bfloat16))
            if (
                not isinstance(image, torch.Tensor)
                or image.shape != batch.images.shape
                or not bool(torch.isfinite(image).all().item())
            ):
                raise RuntimeError("production preflight sample is invalid")

    return _issue_verified_preflight_workload(
        {
            "image_shapes": check_image_shapes,
            "text_shapes": check_text_shapes,
            "zero_update_loss": check_zero_update_loss,
            "optimizer_step": check_optimizer_step,
            "sample": check_sample,
        },
        config=config,
        runtime=runtime,
        trainable_module=trainable_module,
        optimizer=optimizer,
    )


def _write_report(report: PreflightReport, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("preflight report already exists")
    absolute = destination.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("preflight report parent may not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("preflight temporary report already exists")
    payload = (
        json.dumps(asdict(report), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def run_single_gpu_preflight(
    plan: SingleGpuPreflightPlan, destination: Path
) -> AcceptedPreflight:
    """Run every mandatory check in fixed order and stop at the first failure."""

    try:
        token = plan._token
    except AttributeError:
        raise PreflightError(
            "preflight requires a concrete builder-issued plan"
        ) from None
    if (
        type(plan) is not SingleGpuPreflightPlan
        or plan._owner_pid != os.getpid()
        or _PREFLIGHT_PLANS.pop(token, None) is not plan
        or tuple(name for name, _check in plan._checks) != PREFLIGHT_CHECKS
    ):
        raise PreflightError("preflight requires a concrete builder-issued plan")
    results: list[PreflightCheckResult] = []
    try:
        for name, check in plan._checks:
            try:
                check()
            except Exception as exc:  # noqa: BLE001
                results.append(PreflightCheckResult(name, False, type(exc).__name__))
                restored = plan._bindings.restored
                report = PreflightReport(
                    2,
                    "1GPU",
                    False,
                    plan._bindings.resolved_config_sha256,
                    plan._manifest_id,
                    plan._service_session_sha256,
                    restored.manifest.identity.checkpoint_id,
                    restored.state.trainer.successful_updates,
                    tuple(results),
                )
                failure = PreflightError(f"mandatory preflight check failed: {name}")
                failure.__cause__ = exc
                try:
                    _write_report(report, destination)
                except BaseException as report_error:  # noqa: BLE001
                    raise BaseExceptionGroup(
                        "preflight check and report publication both failed",
                        [failure, report_error],
                    ) from None
                raise failure
            results.append(PreflightCheckResult(name, True, None))
        restored = plan._bindings.restored
        report = PreflightReport(
            2,
            "1GPU",
            True,
            plan._bindings.resolved_config_sha256,
            plan._manifest_id,
            plan._service_session_sha256,
            restored.manifest.identity.checkpoint_id,
            restored.state.trainer.successful_updates,
            tuple(results),
        )
        _write_report(report, destination)
        return _accepted_preflight(report, plan._bindings)
    except BaseException as primary:
        try:
            plan._bindings.batches.close()
        except BaseException as close_error:  # noqa: BLE001
            raise BaseExceptionGroup(
                "preflight failure and batch-stream close both failed",
                [primary, close_error],
            ) from None
        raise


_GPU_NAME = "NVIDIA GeForce RTX 5090"
_DRIVER_VERSION = "580.105.08"
_CUDA_VERSION = "12.8"
_COMPUTE_CAPABILITY = (12, 0)
_MIN_GPU_MEMORY_MIB = 32_000
_MIN_LOGICAL_CPUS = 14
_MIN_RAM_BYTES = 120 * 1024**3


class _CudaDeviceProperties(Protocol):
    name: str
    major: int
    minor: int


def _nvidia_smi_identity() -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi identity check failed") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if result.returncode != 0 or len(rows) != 1:
        raise RuntimeError("nvidia-smi did not return one healthy GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError("nvidia-smi identity output is malformed")
    try:
        memory_mib = int(fields[2])
    except ValueError:
        raise RuntimeError("nvidia-smi memory output is malformed") from None
    return fields[0], fields[1], memory_mib


def _memory_identity() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, separator, raw = line.partition(":")
            if separator and name in {"MemTotal", "SwapTotal"}:
                amount, unit = raw.split()
                if unit != "kB":
                    raise RuntimeError("host memory unit is unsupported")
                values[name] = int(amount) * 1024
    except (OSError, ValueError) as exc:
        raise RuntimeError("host memory identity is unreadable") from exc
    if set(values) != {"MemTotal", "SwapTotal"}:
        raise RuntimeError("host memory identity is incomplete")
    return values["MemTotal"], values["SwapTotal"]


def _build_single_gpu_preflight_checks(
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
    workload: SingleGpuPreflightWorkload,
    checkpoint_publisher: _SingleGpuCheckpointPublisher,
) -> SingleGpuPreflightPlan:
    """Internal builder shared by production and targeted failure tests."""

    if not resolved_config_path.is_file() or resolved_config_path.is_symlink():
        raise ValueError("resolved config must be an existing regular file")
    accepted_batches = require_accepted_production_batch_stream(batches)
    _require_restored_checkpoint(
        restored_checkpoint,
        module=trainable_module,
        optimizer=optimizer,
    )
    stream_identity = accepted_batches.identity
    if stream_identity.resolved_config_sha256 != loaded.resolved_sha256:
        raise ValueError(
            "production batch stream resolved identity differs from config"
        )
    if stream_identity.manifest_id != data_client.identity.manifest_id:
        raise ValueError("production batch stream manifest differs from data service")
    if stream_identity.service_session_sha256 != data_client.identity.sha256:
        raise ValueError("production batch stream service session differs from client")
    checkpoint_identity = restored_checkpoint.manifest.identity
    if checkpoint_identity.config_sha256 != loaded.resolved_sha256:
        raise ValueError("restored checkpoint resolved identity differs from config")
    try:
        workload_token = workload._token
    except AttributeError:
        raise PreflightError(
            "a verified concrete preflight workload is required"
        ) from None
    if (
        type(workload) is not SingleGpuPreflightWorkload
        or workload._owner_pid != os.getpid()
        or _VERIFIED_WORKLOADS.pop(workload_token, None) is not workload
        or tuple(name for name, _check in workload._checks) != _WORKLOAD_CHECKS
        or workload._config is not loaded.config
        or workload._runtime is not runtime
        or workload._qwen is not qwen
        or workload._vae is not vae
        or workload._module is not trainable_module
        or workload._optimizer is not optimizer
        or getattr(runtime, "qwen", None) is not qwen
        or getattr(runtime, "vae", None) is not vae
        or getattr(runtime, "composite", None) is not trainable_module
    ):
        raise PreflightError(
            "a verified concrete preflight workload for the exact resources is required"
        )
    workload_checks = dict(workload._checks)

    def resolved_config() -> None:
        payload = resolved_config_path.read_bytes()
        if payload != loaded.resolved_toml.encode("utf-8"):
            raise ValueError("resolved config bytes differ from loaded identity")

    def production_contracts() -> None:
        require_logging_checkpoint_contracts(loaded.config, repository_root)

    def local_assets() -> None:
        require_local_qwen(repository_root)
        require_local_vae(repository_root)

    def dataset_revision() -> None:
        if accepted_batches.identity.manifest_id != data_client.identity.manifest_id:
            raise ValueError("production batch stream manifest changed")
        if (
            data_client.identity.worker_count
            != loaded.config.data.cache.persistent_workers_per_rank
        ):
            raise ValueError("data service worker topology differs from config")
        if data_client.health():
            raise RuntimeError("data service has no training lease available")

    def ready_batch_depth() -> None:
        accepted_batches.ready_batch_depth_snapshot()

    def single_gpu_runtime() -> None:
        require_single_gpu_config(loaded.config)
        require_single_gpu_checkpoint_binding(
            loaded.config,
            restored_checkpoint.state,
            runtime_growth_alpha=runtime.growth_alpha,
        )
        _require_attention_backend_binding(loaded.config, trainable_module)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("single-GPU preflight requires exactly one visible GPU")
        if torch.cuda.current_device() != 0:
            raise RuntimeError("single-GPU preflight requires CUDA device index zero")
        properties = cast(
            _CudaDeviceProperties,
            torch.cuda.get_device_properties(  # pyright: ignore[reportUnknownMemberType]
                torch.cuda.current_device()
            ),
        )
        if (
            properties.name != _GPU_NAME
            or (properties.major, properties.minor) != _COMPUTE_CAPABILITY
            or torch.version.cuda != _CUDA_VERSION
        ):
            raise RuntimeError(
                "CUDA runtime GPU identity differs from the environment lock"
            )
        name, driver, memory_mib = _nvidia_smi_identity()
        if (
            name != _GPU_NAME
            or driver != _DRIVER_VERSION
            or memory_mib < _MIN_GPU_MEMORY_MIB
        ):
            raise RuntimeError("GPU driver or memory differs from the environment lock")
        logical_cpus = os.cpu_count()
        ram_bytes, swap_bytes = _memory_identity()
        if (
            logical_cpus is None
            or logical_cpus < _MIN_LOGICAL_CPUS
            or ram_bytes < _MIN_RAM_BYTES
            or swap_bytes != 0
        ):
            raise RuntimeError("host CPU, RAM, or swap differs from the training floor")

    def storage_capacity() -> None:
        require_training_storage(
            loaded.config,
            repository_root,
            checkpoint_payload_bytes=restored_checkpoint.payload_bytes,
        )

    def frozen_encoders() -> None:
        from sakuramoon.encoders.mage_vae import FrozenMageVAE
        from sakuramoon.encoders.qwen import FrozenQwenEncoder
        from sakuramoon.train.step import TrainableComposite

        if (
            type(qwen) is not FrozenQwenEncoder
            or type(vae) is not FrozenMageVAE
            or type(trainable_module) is not TrainableComposite
        ):
            raise TypeError(
                "preflight requires the production encoder and composite types"
            )
        for encoder in (qwen, vae):
            if getattr(encoder, "training", True):
                raise RuntimeError("frozen encoder is in training mode")
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError("frozen encoder exposes trainable parameters")

    def trainable_schema() -> None:
        from sakuramoon.checkpoint.artifact import validate_optimizer_coverage

        if type(optimizer) is not IsolatedAdamW8bit:
            raise TypeError("production preflight requires IsolatedAdamW8bit")
        validate_optimizer_coverage(
            trainable_module,
            tuple((spec.name, spec.parameter) for spec in optimizer.audit.specs),
        )
        if optimizer.audit.schema_sha256 != checkpoint_identity.parameter_schema_sha256:
            raise ValueError(
                "optimizer schema differs from restored checkpoint identity"
            )

    def raw_checkpoint_round_trip() -> None:
        from sakuramoon.checkpoint.load import (
            load_raw_checkpoint,
            read_raw_checkpoint_state,
        )

        _require_restored_checkpoint(
            restored_checkpoint,
            module=trainable_module,
            optimizer=optimizer,
        )
        if type(optimizer) is not IsolatedAdamW8bit:
            raise TypeError("production preflight requires IsolatedAdamW8bit")
        production_optimizer = optimizer
        if restored_checkpoint.state.trainer.successful_updates != (
            restored_checkpoint.manifest.identity.update
        ):
            raise ValueError("restored checkpoint state update differs from identity")
        expected_identity = replace(
            checkpoint_identity,
            checkpoint_id=f"preflight-{secrets.token_hex(16)}",
        )
        expected_state = restored_checkpoint.state
        reloaded_state = load_raw_checkpoint(
            restored_checkpoint.path,
            trainable_module,
            production_optimizer,
            checkpoint_identity,
        )
        if reloaded_state != expected_state:
            raise ValueError("restored RAW state changed before round-trip publication")
        published = checkpoint_publisher.publish_preflight(
            expected_identity, expected_state
        )
        if (
            not isinstance(published, Path)  # pyright: ignore[reportUnnecessaryIsInstance]
            or published == restored_checkpoint.path
        ):
            raise TypeError("checkpoint round-trip must publish a new RAW path")
        primary: BaseException | None = None
        try:
            manifest, observed_state = read_raw_checkpoint_state(published)
            if (
                manifest.identity != expected_identity
                or observed_state != expected_state
            ):
                raise ValueError("preflight RAW checkpoint round-trip identity changed")
            loaded_state = load_raw_checkpoint(
                published,
                trainable_module,
                production_optimizer,
                expected_identity,
            )
            if loaded_state != expected_state:
                raise ValueError("preflight RAW checkpoint round-trip state changed")
        except BaseException as error:  # noqa: BLE001
            primary = error
        cleanup: BaseException | None = None
        try:
            checkpoint_publisher.discard_preflight(published)
            if published.exists() or published.is_symlink():
                raise RuntimeError("preflight RAW cleanup left a published artifact")
        except BaseException as error:  # noqa: BLE001
            cleanup = error
        if primary is not None and cleanup is not None:
            raise BaseExceptionGroup(
                "preflight RAW validation and cleanup both failed",
                [primary, cleanup],
            ) from None
        if primary is not None:
            raise primary
        if cleanup is not None:
            raise cleanup

    checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("resolved_config", resolved_config),
        ("production_contracts", production_contracts),
        ("local_assets", local_assets),
        ("dataset_revision", dataset_revision),
        ("ready_batch_depth", ready_batch_depth),
        ("single_gpu_runtime", single_gpu_runtime),
        ("storage_capacity", storage_capacity),
        ("frozen_encoders", frozen_encoders),
        ("parameter_schema", trainable_schema),
        *(tuple((name, workload_checks[name]) for name in _WORKLOAD_CHECKS)),
        ("checkpoint_round_trip", raw_checkpoint_round_trip),
    )
    bindings = _PreflightBindings(
        config=loaded.config,
        resolved_config_sha256=loaded.resolved_sha256,
        batches=accepted_batches,
        runtime=runtime,
        qwen=qwen,
        vae=vae,
        module=trainable_module,
        optimizer=optimizer,
        restored=restored_checkpoint,
        checkpoint_publisher=checkpoint_publisher,
    )
    plan = object.__new__(SingleGpuPreflightPlan)
    plan._bindings = bindings
    plan._checks = checks
    plan._manifest_id = data_client.identity.manifest_id
    plan._owner_pid = os.getpid()
    plan._service_session_sha256 = data_client.identity.sha256
    plan._token = secrets.token_hex(32)
    _PREFLIGHT_PLANS[plan._token] = plan
    return plan


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
    workload: SingleGpuPreflightWorkload,
    checkpoint_publisher: ProductionSingleGpuCheckpointPublisher,
) -> SingleGpuPreflightPlan:
    """Construct the fixed production preflight with the exact bound publisher."""

    if type(checkpoint_publisher) is not ProductionSingleGpuCheckpointPublisher:
        raise TypeError(
            "production preflight requires ProductionSingleGpuCheckpointPublisher"
        )
    return _build_single_gpu_preflight_checks(
        loaded,
        repository_root=repository_root,
        resolved_config_path=resolved_config_path,
        data_client=data_client,
        batches=batches,
        runtime=runtime,
        qwen=qwen,
        vae=vae,
        trainable_module=trainable_module,
        optimizer=optimizer,
        restored_checkpoint=restored_checkpoint,
        workload=workload,
        checkpoint_publisher=checkpoint_publisher,
    )


__all__ = [
    "PREFLIGHT_CHECKS",
    "AcceptedPreflight",
    "PreflightCheckResult",
    "PreflightError",
    "PreflightReport",
    "ProductionSingleGpuCheckpointPublisher",
    "RestoredSingleGpuCheckpoint",
    "SingleGpuPreflightPlan",
    "SingleGpuPreflightWorkload",
    "build_single_gpu_preflight_checks",
    "build_single_gpu_preflight_workload",
    "require_accepted_preflight",
    "require_logging_checkpoint_contracts",
    "require_static_single_gpu_preflight",
    "restore_single_gpu_checkpoint",
    "run_single_gpu_preflight",
]
