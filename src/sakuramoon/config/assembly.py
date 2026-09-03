"""Canonical model and telemetry assembly from resolved runtime configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, Self, cast

import torch

from sakuramoon.checkpoint.artifact import (
    build_trainable_composite,
    export_trainable_composite,
)
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.model.irepa import irepa_alignment_metadata
from sakuramoon.storage import repository_directory, repository_file_parent
from sakuramoon.telemetry.metrics import (
    CORE_TIMING_PHASES,
    DETAILED_TIMING_PHASES,
    TIMING_PHASES,
    DurableJsonlSink,
    MetricsPublisher,
)
from sakuramoon.telemetry.observer import (
    AsyncTrainingMetricObserver,
    UpdateMetricContext,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.telemetry.wandb_sink import (
    AsyncWandbSink,
    RemoteRun,
    is_retryable_remote_communication_error,
    replay_retry_queue,
)
from sakuramoon.train.step import TrainableComposite

if TYPE_CHECKING:
    from sakuramoon.train.runtime import SuccessfulTrainingObservation


class ManagedRemoteRun(RemoteRun, Protocol):
    """Remote run whose lifecycle is owned by the training telemetry assembly."""

    def finish(self, exit_code: int | None = None) -> None: ...


class RemoteRunFactory(Protocol):
    def __call__(
        self,
        *,
        project: str,
        entity: str,
        run_id: str,
        run_directory: Path,
        resume_policy: str,
        resume_from_update: int | None,
    ) -> ManagedRemoteRun: ...


MetricContextProvider = Callable[["SuccessfulTrainingObservation"], UpdateMetricContext]


class RemoteInitializationUnavailable(ConnectionError):
    """W&B communication failed before the governed retry sink was available."""


class RetryOnlyRemoteRun:
    """Explicit network-outage target that routes every metric to durable retry."""

    def log(self, data: object, *, step: int) -> None:
        del data, step
        raise RemoteInitializationUnavailable

    def finish(self, exit_code: int | None = None) -> None:
        del exit_code


class _NoopMetricSink:
    def submit(self, metric: object) -> None:
        del metric


class _ResilientManagedRemoteRun:
    def __init__(self, run: ManagedRemoteRun) -> None:
        self.run = run

    def log(self, data: Mapping[str, object], *, step: int) -> None:
        self.run.log(data, step=step)

    def finish(self, exit_code: int | None = None) -> None:
        try:
            self.run.finish(exit_code=exit_code)
        except BaseException as error:
            if not is_retryable_remote_communication_error(error):
                raise


def _require_managed_run(value: object) -> ManagedRemoteRun:
    if (
        value is None
        or not callable(getattr(value, "log", None))
        or not callable(getattr(value, "finish", None))
    ):
        raise TypeError("remote run factory must return callable log/finish methods")
    return cast(ManagedRemoteRun, value)


def initialize_wandb_run(
    *,
    project: str,
    entity: str,
    run_id: str,
    run_directory: Path,
    resume_policy: str,
    resume_from_update: int | None,
) -> ManagedRemoteRun:
    """Start W&B under the fixed run id, re-attaching on checkpoint resumes.

    Every process attaches to the same remote run (``id=run_id``) instead of
    minting a suffixed continuation id. With ``finish_on_close = false`` the
    run stays open between restarts, so ``resume="allow"`` continues the same
    remote run. If the fixed id is already owned by a remote run that cannot be
    re-attached from this machine, fall back to a new run under the same
    name and group so training never fails on telemetry.
    """

    import wandb
    from wandb.errors import AuthenticationError, CommError, UsageError

    if resume_policy != "allow":
        raise ValueError("W&B resume policy must be allow")
    if resume_from_update is not None and (
        type(resume_from_update) is not int or resume_from_update < 0
    ):
        raise ValueError("W&B resume update must be a non-negative integer")
    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "id": run_id,
        "name": run_id,
        "group": run_id,
        "job_type": "train-continuation" if resume_from_update is not None else "train",
        "dir": str(run_directory),
        "mode": "online",
        "resume": resume_policy,
        "reinit": "create_new",
        "save_code": False,
    }
    if resume_from_update is not None:
        init_kwargs["config"] = {"resume_from_update": resume_from_update}
    try:
        try:
            run = wandb.init(**init_kwargs)
        except UsageError:
            # The fixed id is owned by a remote run that cannot be
            # re-attached here; keep the stable display name and group.
            init_kwargs.pop("id")
            run = wandb.init(**init_kwargs)
    except AuthenticationError:
        raise
    except (ConnectionError, CommError):
        return RetryOnlyRemoteRun()
    return _require_managed_run(run)


def _raise_preserving(primary: BaseException, cleanup: list[BaseException]) -> NoReturn:
    if cleanup:
        raise BaseExceptionGroup(
            "telemetry assembly and cleanup both failed", [primary, *cleanup]
        ) from None
    raise primary


def _close_components(
    components: tuple[tuple[str, Callable[[], None]], ...],
) -> None:
    errors: list[BaseException] = []
    for _name, close in components:
        try:
            close()
        except BaseException as error:  # noqa: BLE001 - lifecycle boundary
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("telemetry components failed to close", errors)


class TrainingTelemetryAssembly:
    """Own observer, remote queue/run, and local sink in deterministic order."""

    def __init__(
        self,
        *,
        phase_timer: PhaseTimer,
        observer: AsyncTrainingMetricObserver,
        remote: AsyncWandbSink | None,
        local: DurableJsonlSink,
        run: ManagedRemoteRun | None,
        finish_remote_run: bool = True,
    ) -> None:
        self.phase_timer = phase_timer
        self.observer = observer
        self.remote = remote
        self.local = local
        self.run = run
        self.finish_remote_run = finish_remote_run
        self._closed = False

    def submit_wandb_metrics(
        self,
        metrics: Mapping[str, int | float],
        *,
        successful_update: int,
    ) -> None:
        """Upload non-training metrics when W&B is enabled."""

        if self._closed:
            raise RuntimeError("training telemetry is closed")
        if self.remote is not None:
            self.remote.submit_metrics(
                metrics,
                successful_update=successful_update,
            )

    def submit_wandb_images(
        self,
        image_paths: tuple[Path, ...],
        captions: tuple[str, ...],
        *,
        successful_update: int,
    ) -> None:
        """Upload generated training samples when W&B is enabled."""

        if self._closed:
            raise RuntimeError("training telemetry is closed")
        if self.remote is not None:
            self.remote.submit_images(
                image_paths,
                captions,
                successful_update=successful_update,
            )

    def close(self, *, exit_code: int = 0) -> None:
        if type(exit_code) is not int or exit_code not in {0, 1}:
            raise ValueError("telemetry exit_code must be zero or one")
        if self._closed:
            return
        self._closed = True
        components: list[tuple[str, Callable[[], None]]] = [
            ("observer", self.observer.close)
        ]
        if self.remote is not None:
            components.append(("remote", self.remote.close))
        run = self.run
        if run is not None and self.finish_remote_run:
            components.append(("remote_run", lambda: run.finish(exit_code=exit_code)))
        components.append(("local", self.local.close))
        _close_components(tuple(components))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close(exit_code=1 if error is not None else 0)
        except BaseException as close_error:
            if error is not None:
                raise BaseExceptionGroup(
                    "training and telemetry close both failed", [error, close_error]
                ) from None
            raise


def _artifact_file(repository_root: Path, configured: str) -> Path:
    parent = repository_file_parent(repository_root, configured)
    return parent / Path(configured).name


def _require_timing_phase_binding(config: RuntimeConfig) -> None:
    expected = (*CORE_TIMING_PHASES, *DETAILED_TIMING_PHASES)
    if len(expected) != len(TIMING_PHASES) or frozenset(expected) != TIMING_PHASES:
        raise RuntimeError("telemetry timing vocabulary is internally inconsistent")
    if config.timing.phases != expected:
        raise ValueError(
            "resolved timing phases do not match the telemetry timing vocabulary"
        )


def build_training_telemetry_from_config(
    config: RuntimeConfig,
    *,
    repository_root: Path,
    device: torch.device,
    context_provider: MetricContextProvider,
    resume_from_update: int | None = None,
    remote_run_factory: RemoteRunFactory = initialize_wandb_run,
) -> TrainingTelemetryAssembly:
    """Build the complete local-first telemetry lifecycle from strict config."""

    if not isinstance(config, RuntimeConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("resolved RuntimeConfig is required for telemetry assembly")
    if config.run.intent != "train":
        raise ValueError("training telemetry requires train intent")
    if not config.timing.enabled:
        raise ValueError("training timing is disabled")
    _require_timing_phase_binding(config)
    if not callable(context_provider):
        raise TypeError("metric context provider must be callable")

    local_path = _artifact_file(repository_root, config.logging.local_jsonl_path)

    run: ManagedRemoteRun | None = None
    local: DurableJsonlSink | None = None
    remote: AsyncWandbSink | None = None
    observer: AsyncTrainingMetricObserver | None = None
    retry_path: Path | None = None
    try:
        if config.wandb.enabled:
            run_directory = repository_directory(repository_root, config.paths.run_dir)
            retry_path = _artifact_file(repository_root, config.wandb.retry_jsonl_path)
            if local_path == retry_path:
                raise ValueError("local metric and W&B retry paths must differ")
            run = _ResilientManagedRemoteRun(
                _require_managed_run(
                    remote_run_factory(
                        project=config.wandb.project,
                        entity=config.wandb.entity,
                        run_id=config.run.run_id,
                        run_directory=run_directory,
                        resume_policy=config.wandb.resume_policy,
                        resume_from_update=resume_from_update,
                    )
                )
            )
            try:
                replay_retry_queue(run, retry_path)
            except BaseException as replay_error:
                if not is_retryable_remote_communication_error(replay_error):
                    raise
        local = DurableJsonlSink(
            local_path,
            fsync_every_records=config.logging.flush_every_updates,
        )
        if run is not None:
            assert retry_path is not None
            remote = AsyncWandbSink(
                run,
                retry_path=retry_path,
                queue_capacity=config.wandb.queue_capacity,
            )
        observer = AsyncTrainingMetricObserver(
            MetricsPublisher(
                local, remote if remote is not None else _NoopMetricSink()
            ),
            context_provider=context_provider,
            queue_capacity=config.logging.observer_queue_capacity,
            event_timeout_seconds=config.logging.observer_event_timeout_seconds,
        )
        phase_timer = PhaseTimer(device=device)
        return TrainingTelemetryAssembly(
            phase_timer=phase_timer,
            observer=observer,
            remote=remote,
            local=local,
            run=run,
            finish_remote_run=config.wandb.finish_on_close,
        )
    except BaseException as error:  # noqa: BLE001 - construction cleanup boundary
        cleanup: list[BaseException] = []
        components: list[Callable[[], None]] = []
        if observer is not None:
            components.append(observer.close)
        if remote is not None:
            components.append(remote.close)
        if run is not None and config.wandb.finish_on_close:
            components.append(lambda: run.finish(exit_code=1))
        if local is not None:
            components.append(local.close)
        for close in components:
            try:
                close()
            except BaseException as close_error:  # noqa: BLE001 - construction boundary
                cleanup.append(close_error)
        _raise_preserving(error, cleanup)


def trainable_composite_spec(config: RuntimeConfig) -> dict[str, object]:
    """Bind every trainable constructor argument to one strict config field."""

    if not isinstance(config, RuntimeConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("resolved RuntimeConfig is required for production assembly")
    model = config.model
    dit = model.dit
    rope = model.rope
    condition = model.condition
    head = model.head
    text = model.text
    condition_tokens = model.condition_tokens
    backend = (
        "das_fa2_varlen"
        if config.kernels.attention_backend == "das_fa2_varlen"
        else "dense_sdpa"
    )
    irepa = config.irepa
    irepa_enabled = irepa is not None and irepa.enabled
    document: dict[str, object] = {
        "schema_version": 4 if irepa_enabled else 3,
        "class": "TrainableComposite",
        "dit": {
            "active_slot_ids": list(active_slot_ids(config.stage.depth)),
            "aspect_dim": condition.aspect_dim,
            "attention_backend": backend,
            "attention_dropout": dit.attention_dropout,
            "condition_hidden_size": condition.hidden_dim,
            "condition_token_count": condition_tokens.token_count,
            "depth": config.stage.depth,
            "final_modulation_size": head.final_modulation_size,
            "head_dim": dit.head_dim,
            "hidden_size": dit.hidden_size,
            "input_channels": config.assets.vae.latent_channels,
            "intermediate_size": dit.intermediate_size,
            "kv_heads": dit.kv_heads,
            "linear_dtype": dit.activation_dtype,
            "mlp_dropout": dit.mlp_dropout,
            "modality_init_std": model.packing.modality_init_std,
            "modulation_chunks": condition.block_modulation_chunks,
            "norm_eps": dit.norm_eps,
            "out_channels": head.out_channels,
            "output_bias_zero_init": head.bias_zero_init,
            "output_weight_zero_init": head.weight_zero_init,
            "projection_bias": dit.projection_bias,
            "q_heads": dit.q_heads,
            "rope_nope_dim": rope.nope_dim,
            "rope_position_scale": rope.position_scale,
            "rope_theta": rope.theta,
            "rope_x_dim": rope.x_dim,
            "rope_y_dim": rope.y_dim,
            "sensitive_dtype": dit.norm_accumulation,
            "size_dim": condition.size_dim,
            "stable_slot_count": dit.stable_slot_count,
            "timestep_dim": condition.timestep_dim,
        },
        "text": {
            "adapter_size": text.adapter_size,
            "attention_heads": text.attention_heads,
            "groups": text.groups,
            "input_size": text.input_size,
            "layer_scale_init": text.layer_scale_init,
            "linear_dtype": text.linear_dtype,
            "mix_gate_init": text.mix_gate_init,
            "norm_eps": text.norm_eps,
            "output_size": text.output_size,
            "projection_bias": text.projection_bias,
            "sensitive_dtype": text.sensitive_dtype,
        },
        "condition_tokens": {
            "attention_heads": condition_tokens.attention_heads,
            "hidden_size": condition_tokens.hidden_size,
            "init_std": condition_tokens.init_std,
            "input_size": condition_tokens.input_size,
            "intermediate_size": condition_tokens.mlp_intermediate_size,
            "linear_dtype": condition_tokens.linear_dtype,
            "norm_eps": condition_tokens.norm_eps,
            "output_size": condition_tokens.output_size,
            "projection_bias": condition_tokens.projection_bias,
            "sensitive_dtype": condition_tokens.sensitive_dtype,
            "token_count": condition_tokens.token_count,
        },
    }
    if irepa_enabled:
        # v1 projector input width is the configured DiT hidden width
        # (resolves to 2560 for the current SakuraMoon model).
        document["training_auxiliaries"] = {
            "irepa": irepa_alignment_metadata(dit.hidden_size),
        }
    return document


def build_trainable_composite_from_config(
    config: RuntimeConfig,
    *,
    device: torch.device | str,
) -> TrainableComposite:
    """Construct and round-trip-check the exact config-bound trainable module."""

    document = trainable_composite_spec(config)
    module = build_trainable_composite(document, device=device)
    observed: dict[str, Any] = export_trainable_composite(module)
    if observed != document:
        raise ValueError("assembled trainable composite differs from resolved config")
    irepa = config.irepa
    if irepa is not None and irepa.enabled:
        # The artifact document is tap-free; the config locks the capture
        # slot (Literal[8]) as a runtime binding after the round-trip check.
        if module.irepa_tap_slot_id is None:
            module.bind_irepa_tap_slot(irepa.tap_slot)
        elif module.irepa_tap_slot_id != irepa.tap_slot:
            raise ValueError(
                "assembled iREPA tap slot differs from the resolved config"
            )
    elif module.irepa_tap_slot_id is not None:
        raise ValueError("tap is bound while iREPA is absent or disabled")
    module.dit.set_activation_checkpoint_mode(config.stage.activation_checkpoint_mode)
    return module


__all__ = [
    "ManagedRemoteRun",
    "MetricContextProvider",
    "RemoteInitializationUnavailable",
    "RemoteRunFactory",
    "RetryOnlyRemoteRun",
    "TrainingTelemetryAssembly",
    "build_trainable_composite_from_config",
    "build_training_telemetry_from_config",
    "initialize_wandb_run",
    "trainable_composite_spec",
]
