"""Single-GPU data-to-update assembly at the typed service boundary.

This module owns only the trainer-side boundary.  The data service, its cache,
state store, and metadata policy remain outside the process and are supplied as
validated objects by the caller.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn

from sakuramoon.checkpoint.policy import (
    CheckpointCadence,
    CheckpointReason,
)
from sakuramoon.checkpoint.schema import CheckpointManifest, RawCheckpointState
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.caption import CaptionDropoutCounts
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    require_accepted_production_batch_stream,
)
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.qwen import FrozenQwenEncoder
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.objective.flow import (
    flow_matching_loss,
    interpolate_state,
    sample_jlt_timesteps,
    sample_noise,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.loop import (
    LoopResult,
    SingleGpuTrainingLoop,
    SuccessfulLoopObservation,
)
from sakuramoon.train.step import (
    SingleGpuUpdateState,
    StepOptimizer,
    TrainableComposite,
    TrainableCompositeInputs,
)

if TYPE_CHECKING:
    from sakuramoon.train.preflight import (
        AcceptedPreflight,
        ProductionSingleGpuCheckpointPublisher,
        RestoredSingleGpuCheckpoint,
        _SingleGpuCheckpointPublisher,
    )


class _Encoder(Protocol):
    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> object: ...


class _Vae(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...


def read_raw_checkpoint_state(
    checkpoint: Path,
) -> tuple[CheckpointManifest, RawCheckpointState]:
    """Import the T044 reader lazily to keep checkpoint/train packages acyclic."""

    from sakuramoon.checkpoint.load import read_raw_checkpoint_state as read_state

    return read_state(checkpoint)


class DenseDiTAdapter(nn.Module):
    """Expose the dense SDPA reference through the packed composite boundary."""

    def __init__(self, model: DenseDiT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        latents: tuple[torch.Tensor, ...],
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        text_lengths: tuple[int, ...],
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> tuple[torch.Tensor, ...]:
        del text_lengths
        latent_batch = torch.stack(latents)
        predictions = self.model(
            latent_batch,
            text_tokens,
            text_mask,
            style_tokens,
            timestep,
            size_scale,
            aspect,
            growth_alpha=growth_alpha,
        )
        return tuple(predictions.unbind(0))

    def model_metadata(self) -> dict[str, int | str]:
        return self.model.model_metadata()

    def artifact_config(self) -> dict[str, object]:
        return self.model.artifact_config()


@dataclass(frozen=True, slots=True)
class PreparedTrainingBatch:
    """The immutable tensors needed by one objective evaluation."""

    inputs: TrainableCompositeInputs
    clean_latents: tuple[torch.Tensor, ...]
    states: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    """A loss vector plus counters consumed by the loop/benchmark adapters."""

    per_sample_loss: torch.Tensor
    image_tokens: int
    text_tokens: int
    sample_ids: tuple[str, ...]
    shape_keys: tuple[str, ...]
    high_noise_loss_sum: torch.Tensor
    high_noise_sample_count: torch.Tensor
    low_noise_loss_sum: torch.Tensor
    low_noise_sample_count: torch.Tensor
    timesteps: torch.Tensor
    dropout_hits: CaptionDropoutCounts

    def detached(self) -> RuntimeMeasurement:
        """Drop the autograd graph before handing facts to an async observer."""

        return RuntimeMeasurement(
            per_sample_loss=self.per_sample_loss.detach(),
            image_tokens=self.image_tokens,
            text_tokens=self.text_tokens,
            sample_ids=self.sample_ids,
            shape_keys=self.shape_keys,
            high_noise_loss_sum=self.high_noise_loss_sum.detach(),
            high_noise_sample_count=self.high_noise_sample_count.detach(),
            low_noise_loss_sum=self.low_noise_loss_sum.detach(),
            low_noise_sample_count=self.low_noise_sample_count.detach(),
            timesteps=self.timesteps.detach(),
            dropout_hits=self.dropout_hits,
        )


@dataclass(frozen=True, slots=True)
class SuccessfulTrainingObservation:
    """Exact T050 facts emitted once after a successful update completes."""

    loop: SuccessfulLoopObservation
    microbatches: tuple[RuntimeMeasurement, ...]
    phase_timer: PhaseTimer
    learning_rate: float
    gpu_memory_allocated_bytes: int
    gpu_memory_reserved_bytes: int

    def __post_init__(self) -> None:
        if len(self.microbatches) != self.loop.update.microbatches:
            raise ValueError("observation microbatch count differs from update")
        if (
            sum(item.per_sample_loss.numel() for item in self.microbatches)
            != self.loop.update.effective_samples
        ):
            raise ValueError("observation sample count differs from update")
        if any(
            tensor.requires_grad or tensor.grad_fn is not None
            for item in self.microbatches
            for tensor in (
                item.per_sample_loss,
                item.high_noise_loss_sum,
                item.high_noise_sample_count,
                item.low_noise_loss_sum,
                item.low_noise_sample_count,
                item.timesteps,
            )
        ):
            raise ValueError("observation tensors must be detached from autograd")
        if type(self.learning_rate) is not float or not math.isfinite(
            self.learning_rate
        ):
            raise ValueError("observation learning rate must be finite")
        if (
            type(self.gpu_memory_allocated_bytes) is not int
            or self.gpu_memory_allocated_bytes < 0
            or type(self.gpu_memory_reserved_bytes) is not int
            or self.gpu_memory_reserved_bytes < self.gpu_memory_allocated_bytes
        ):
            raise ValueError("observation GPU memory facts are inconsistent")


@dataclass(frozen=True, slots=True)
class _RuntimeLoss:
    per_sample: torch.Tensor
    high_noise_loss_sum: torch.Tensor
    high_noise_sample_count: torch.Tensor
    low_noise_loss_sum: torch.Tensor
    low_noise_sample_count: torch.Tensor


def _require_batch(batch: TrainingBatch) -> None:
    if type(batch) is not TrainingBatch:
        raise TypeError("single-GPU runtime requires a typed TrainingBatch")
    if batch.images.ndim != 4 or batch.images.shape[1] != 3:
        raise ValueError("training images must have shape [B,3,H,W]")
    if batch.input_ids.ndim != 2 or batch.attention_mask.shape != batch.input_ids.shape:
        raise ValueError("training token tensors have inconsistent shapes")
    if batch.input_ids.shape[0] != batch.images.shape[0]:
        raise ValueError("training image and token batch sizes differ")
    if batch.main_token_indices.shape[0] != batch.images.shape[0]:
        raise ValueError("main token routing batch size differs from images")
    if batch.artist_token_indices.shape[0] != batch.images.shape[0]:
        raise ValueError("Artist token routing batch size differs from images")
    if len(batch.main_token_lengths) != batch.images.shape[0]:
        raise ValueError("main token length count differs from images")
    if len(batch.releases) != batch.images.shape[0]:
        raise ValueError("release count differs from images")
    if batch.target_height <= 0 or batch.target_width <= 0:
        raise ValueError("training target dimensions must be positive")


def _size_conditions(
    batch: TrainingBatch, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    height = float(batch.target_height)
    width = float(batch.target_width)
    size_scale = 0.5 * math.log2((height * width) / float(512 * 512))
    aspect = math.log2(width / height)
    count = batch.images.shape[0]
    return (
        torch.full((count,), size_scale, device=device, dtype=torch.float32),
        torch.full((count,), aspect, device=device, dtype=torch.float32),
    )


class SingleGpuBatchRuntime:
    """Encode one service batch and produce its differentiable loss vector."""

    def __init__(
        self,
        *,
        qwen: FrozenQwenEncoder | _Encoder,
        vae: FrozenMageVAE | _Vae,
        composite: TrainableComposite,
        device: torch.device,
        generator: torch.Generator,
        p_mean: float,
        p_std: float,
        noise_scale: float,
        t_eps: float,
        noise_observation_boundary: float,
        growth_alpha: float,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("single-GPU production runtime requires CUDA")
        if generator.device.type != "cuda":
            raise ValueError("training generator must be a CUDA generator")
        generator_index = int(device.index or 0)
        if generator is not torch.cuda.default_generators[generator_index]:
            raise ValueError(
                "training generator must be the checkpointed default CUDA generator"
            )
        for name, value in (
            ("p_mean", p_mean),
            ("p_std", p_std),
            ("noise_scale", noise_scale),
            ("t_eps", t_eps),
            ("noise_observation_boundary", noise_observation_boundary),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite TOML float")
        if type(growth_alpha) is not float or not 0.0 <= growth_alpha <= 1.0:
            raise ValueError("growth_alpha must be a float in [0,1]")
        self.qwen = qwen
        self.vae = vae
        self.composite = composite
        self.device = device
        self.generator = generator
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.noise_observation_boundary = noise_observation_boundary
        self.growth_alpha = growth_alpha

    def prepare(
        self, batch: TrainingBatch, *, phase_timer: PhaseTimer | None = None
    ) -> PreparedTrainingBatch:
        _require_batch(batch)
        if phase_timer is None:
            input_ids = batch.input_ids.to(
                self.device, dtype=torch.long, non_blocking=True
            )
            attention_mask = batch.attention_mask.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            images = (
                batch.images.to(self.device, dtype=torch.bfloat16, non_blocking=True)
                .div(127.5)
                .sub(1.0)
            )
            main_token_indices = batch.main_token_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            )
            main_mask = batch.main_mask.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            artist_token_indices = batch.artist_token_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            )
            artist_mask = batch.artist_mask.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            use_null_style = batch.use_null_style.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            active_style_sample_indices = batch.active_style_sample_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            )
        else:
            with phase_timer.record("h2d"):
                input_ids = batch.input_ids.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                attention_mask = batch.attention_mask.to(
                    self.device, dtype=torch.bool, non_blocking=True
                )
                images = (
                    batch.images.to(
                        self.device, dtype=torch.bfloat16, non_blocking=True
                    )
                    .div(127.5)
                    .sub(1.0)
                )
                main_token_indices = batch.main_token_indices.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                main_mask = batch.main_mask.to(
                    self.device, dtype=torch.bool, non_blocking=True
                )
                artist_token_indices = batch.artist_token_indices.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                artist_mask = batch.artist_mask.to(
                    self.device, dtype=torch.bool, non_blocking=True
                )
                use_null_style = batch.use_null_style.to(
                    self.device, dtype=torch.bool, non_blocking=True
                )
                active_style_sample_indices = batch.active_style_sample_indices.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
        if phase_timer is None:
            qwen_output = self.qwen(input_ids, attention_mask)
        else:
            with phase_timer.record("qwen"):
                qwen_output = self.qwen(input_ids, attention_mask)
        hidden_states = getattr(qwen_output, "hidden_states", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Qwen output must expose hidden_states")
        if hidden_states.ndim != 4 or hidden_states.shape[0] != input_ids.shape[0]:
            raise ValueError("Qwen hidden states have an invalid batch shape")

        if phase_timer is None:
            clean = self.vae.encode(images)
        else:
            with phase_timer.record("vae"):
                clean = self.vae.encode(images)
        expected = (
            batch.images.shape[0],
            128,
            batch.target_height // 16,
            batch.target_width // 16,
        )
        if tuple(clean.shape) != expected or clean.dtype != torch.bfloat16:
            raise ValueError("Mage-VAE returned an unexpected latent contract")

        timestep = sample_jlt_timesteps(
            clean.shape[0],
            p_mean=self.p_mean,
            p_std=self.p_std,
            device=self.device,
            generator=self.generator,
        )
        noise = sample_noise(
            clean,
            noise_scale=self.noise_scale,
            generator=self.generator,
        )
        state = interpolate_state(clean, noise, timestep)
        size_scale, aspect = _size_conditions(batch, device=self.device)
        inputs = TrainableCompositeInputs(
            qwen_states=hidden_states,
            main_token_indices=main_token_indices,
            main_mask=main_mask,
            main_token_lengths=batch.main_token_lengths,
            artist_token_indices=artist_token_indices,
            artist_mask=artist_mask,
            use_null_style=use_null_style,
            active_style_sample_indices=active_style_sample_indices,
            latents=tuple(item for item in state.unbind(0)),
            timestep=timestep,
            size_scale=size_scale,
            aspect=aspect,
            growth_alpha=self.growth_alpha,
        )
        return PreparedTrainingBatch(
            inputs=inputs,
            clean_latents=tuple(item for item in clean.unbind(0)),
            states=tuple(item for item in state.unbind(0)),
        )

    def measure(
        self, batch: TrainingBatch, *, phase_timer: PhaseTimer | None = None
    ) -> RuntimeMeasurement:
        prepared = self.prepare(batch, phase_timer=phase_timer)
        if phase_timer is None:
            predictions = self.composite(prepared.inputs)
        else:
            with phase_timer.record("conditioning"):
                predictions = self.composite.forward_conditioning(prepared.inputs)
            with phase_timer.record("dit_forward"):
                predictions = self.composite.forward_dit(prepared.inputs, predictions)
        if len(predictions) != len(prepared.clean_latents):
            raise ValueError("DiT prediction count differs from latent batch")
        if phase_timer is None:
            loss = self._loss(predictions, prepared)
        else:
            with phase_timer.record("loss"):
                loss = self._loss(predictions, prepared)
        image_tokens = (
            batch.images.shape[0]
            * (batch.target_height // 16)
            * (batch.target_width // 16)
        )
        text_tokens = sum(batch.main_token_lengths)
        sample_ids = tuple(
            str(cast(int, value.item()))
            for value in batch.sample_ids.detach().cpu().unbind()
        )
        shape_key = f"{batch.target_height}x{batch.target_width}x{batch.dense_length}"
        return RuntimeMeasurement(
            per_sample_loss=loss.per_sample,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            sample_ids=sample_ids,
            shape_keys=(shape_key,) * len(sample_ids),
            high_noise_loss_sum=loss.high_noise_loss_sum,
            high_noise_sample_count=loss.high_noise_sample_count,
            low_noise_loss_sum=loss.low_noise_loss_sum,
            low_noise_sample_count=loss.low_noise_sample_count,
            timesteps=prepared.inputs.timestep,
            dropout_hits=batch.dropout_hits,
        )

    def _loss(
        self,
        predictions: tuple[torch.Tensor, ...],
        prepared: PreparedTrainingBatch,
    ) -> _RuntimeLoss:
        if len({tuple(value.shape) for value in predictions}) == 1:
            prediction_batch = torch.stack(predictions)
            clean_batch = torch.stack(prepared.clean_latents)
            state_batch = torch.stack(prepared.states)
            result = flow_matching_loss(
                prediction_batch,
                state_batch,
                clean_batch,
                prepared.inputs.timestep,
                t_eps=self.t_eps,
                noise_observation_boundary=self.noise_observation_boundary,
            )
            return _RuntimeLoss(
                result.per_sample,
                result.high_noise_loss_sum,
                result.high_noise_sample_count,
                result.low_noise_loss_sum,
                result.low_noise_sample_count,
            )
        values: list[torch.Tensor] = []
        high_loss: list[torch.Tensor] = []
        high_count: list[torch.Tensor] = []
        low_loss: list[torch.Tensor] = []
        low_count: list[torch.Tensor] = []
        for index, prediction in enumerate(predictions):
            result = flow_matching_loss(
                prediction.unsqueeze(0),
                prepared.states[index].unsqueeze(0),
                prepared.clean_latents[index].unsqueeze(0),
                prepared.inputs.timestep[index : index + 1],
                t_eps=self.t_eps,
                noise_observation_boundary=self.noise_observation_boundary,
            )
            values.append(result.per_sample[0])
            high_loss.append(result.high_noise_loss_sum)
            high_count.append(result.high_noise_sample_count)
            low_loss.append(result.low_noise_loss_sum)
            low_count.append(result.low_noise_sample_count)
        return _RuntimeLoss(
            torch.stack(values),
            torch.stack(high_loss).sum(),
            torch.stack(high_count).sum(),
            torch.stack(low_loss).sum(),
            torch.stack(low_count).sum(),
        )


def require_single_gpu_config(config: RuntimeConfig) -> None:
    """Reject every topology except the explicitly approved S0 single card."""

    if (
        config.run.intent != "train"
        or config.run.stage != "S0"
        or not config.stage.enabled
        or config.stage.world_size != 1
    ):
        raise ValueError(
            "the single-GPU runtime accepts only train-intent enabled S0 topology"
        )
    if config.distributed.backend != "native" or config.distributed.world_size != 1:
        raise ValueError("single-GPU runtime requires native world_size=1")
    if config.failure.allow_force_bypass:
        raise ValueError("single-GPU runtime cannot enable preflight bypass")
    if config.stage.activation_checkpoint_mode != "none":
        raise ValueError(
            "single-GPU runtime does not implement activation checkpointing"
        )


def require_single_gpu_checkpoint_binding(
    config: RuntimeConfig,
    state: RawCheckpointState,
    *,
    runtime_growth_alpha: float,
) -> None:
    """Bind every restored stage/growth axis to the resolved S0 runtime."""

    require_single_gpu_config(config)
    growth = state.growth
    if (
        growth.stage != config.stage.name
        or growth.world_size != config.stage.world_size
        or growth.resolution != config.stage.resolution
    ):
        raise ValueError("restored checkpoint axes differ from resolved stage")
    if growth.active_slot_ids != active_slot_ids(config.stage.depth):
        raise ValueError("restored checkpoint slots differ from resolved stage depth")
    has_ramp = growth.ramp_start_successful_update is not None
    if has_ramp != config.growth.enabled:
        raise ValueError("restored checkpoint ramp presence differs from resolved growth")
    if growth.alpha != runtime_growth_alpha:
        raise ValueError("runtime growth alpha differs from restored checkpoint")
    stage_budget = state.stage_budget
    trainer = state.trainer
    if not (
        stage_budget.start_successful_update
        <= trainer.successful_updates
        <= stage_budget.terminal_successful_update
    ):
        raise ValueError("restored stage budget is inconsistent with trainer state")
    if stage_budget.start_successful_update != 0:
        raise ValueError("restored S0 stage budget must start at update zero")
    if (
        stage_budget.terminal_successful_update
        - stage_budget.start_successful_update
        != config.stage.planned_updates
    ):
        raise ValueError("restored stage budget differs from resolved config")
    if trainer.successful_updates >= stage_budget.terminal_successful_update:
        raise ValueError("stage successful-update budget is already exhausted")
    if state.checkpoint_cadence.last_successful_update != trainer.successful_updates:
        raise ValueError("checkpoint cadence update does not match trainer state")


def _optimizer_learning_rate(optimizer: StepOptimizer) -> float:
    wrapped = getattr(optimizer, "optimizer", None)
    groups = getattr(wrapped, "param_groups", None)
    if (
        type(groups) is not list
        or not groups
        or any(type(group) is not dict for group in cast(list[object], groups))
    ):
        raise TypeError("production optimizer must expose parameter groups")
    rates: set[float] = set()
    for group in cast(list[dict[str, object]], groups):
        raw_rate = group.get("lr")
        if type(raw_rate) is float:
            rate = raw_rate
        elif (
            type(raw_rate) is torch.Tensor
            and raw_rate.ndim == 0
            and raw_rate.dtype.is_floating_point
        ):
            rate = float(raw_rate.detach().item())
        else:
            raise ValueError("production optimizer learning rate is invalid")
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("production optimizer learning rate is invalid")
        rates.add(rate)
    if len(rates) != 1:
        raise ValueError("production optimizer learning rates differ across groups")
    rate = rates.pop()
    return rate


def _run_single_gpu_training(
    config: RuntimeConfig,
    *,
    preflight: AcceptedPreflight,
    runtime: SingleGpuBatchRuntime,
    module: nn.Module,
    optimizer: StepOptimizer,
    batches: AcceptedProductionBatchStream,
    scheduler_step: Callable[[int], None],
    checkpoint_publisher: _SingleGpuCheckpointPublisher,
    diagnostic_root: Path,
    failure_id: Callable[[str, SingleGpuUpdateState], str],
    restored_checkpoint: RestoredSingleGpuCheckpoint,
    phase_timer: PhaseTimer,
    successful_update_observer: Callable[[SuccessfulTrainingObservation], None],
    forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    clock: Callable[[], float] | None = None,
) -> LoopResult:
    """Run the locked loop after all ownership and assembly objects are ready."""

    from sakuramoon.train.preflight import require_accepted_preflight

    stream = require_accepted_production_batch_stream(batches)
    result: LoopResult | None = None
    primary: BaseException | None = None
    try:
        require_accepted_preflight(
            preflight,
            config=config,
            batches=stream,
            runtime=runtime,
            qwen=runtime.qwen,
            vae=runtime.vae,
            module=module,
            optimizer=optimizer,
            restored=restored_checkpoint,
            checkpoint_publisher=checkpoint_publisher,
        )
        require_single_gpu_checkpoint_binding(
            config,
            restored_checkpoint.state,
            runtime_growth_alpha=runtime.growth_alpha,
        )
        if module is not runtime.composite:
            raise ValueError("training module must be the runtime trainable composite")
        if phase_timer.device != runtime.device:
            raise ValueError("phase timer device differs from the training runtime")
        raw_state = restored_checkpoint.state
        stage_budget = raw_state.stage_budget
        state = raw_state.trainer
        cadence = raw_state.checkpoint_cadence
        target_successful_updates = stage_budget.terminal_successful_update
        pending_measurements: list[RuntimeMeasurement] = []
        active_phase_timer: PhaseTimer | None = None

        def update_started(timer: PhaseTimer | None) -> None:
            nonlocal active_phase_timer
            active_phase_timer = timer

        def measure_batch(batch: TrainingBatch) -> torch.Tensor:
            if active_phase_timer is None:
                raise RuntimeError("training update phase timer was not initialized")
            measurement = runtime.measure(batch, phase_timer=active_phase_timer)
            pending_measurements.append(measurement.detached())
            return measurement.per_sample_loss

        def observe_update(observation: SuccessfulLoopObservation) -> None:
            nonlocal active_phase_timer
            microbatches = tuple(pending_measurements)
            update_timer = observation.phase_timer
            if update_timer is None or update_timer is not active_phase_timer:
                raise RuntimeError("training observation phase timer identity changed")
            if runtime.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(runtime.device)
                reserved = torch.cuda.memory_reserved(runtime.device)
            else:
                allocated = 0
                reserved = 0
            emitted = SuccessfulTrainingObservation(
                loop=observation,
                microbatches=microbatches,
                phase_timer=update_timer,
                learning_rate=_optimizer_learning_rate(optimizer),
                gpu_memory_allocated_bytes=allocated,
                gpu_memory_reserved_bytes=reserved,
            )
            successful_update_observer(emitted)
            pending_measurements.clear()
            active_phase_timer = None

        def publish_and_verify(
            update_state: SingleGpuUpdateState,
            reason: CheckpointReason,
            proposed_cadence: CheckpointCadence,
        ) -> None:
            checkpoint_path = checkpoint_publisher.publish_update(
                update_state, reason, proposed_cadence
            )
            manifest, published_state = read_raw_checkpoint_state(checkpoint_path)
            restored_identity = restored_checkpoint.manifest.identity
            published_identity = manifest.identity
            if (
                published_identity.update != update_state.successful_updates
                or published_identity.config_sha256 != restored_identity.config_sha256
                or published_identity.dependency_sha256
                != restored_identity.dependency_sha256
                or published_identity.parameter_schema_sha256
                != restored_identity.parameter_schema_sha256
            ):
                raise ValueError("published RAW checkpoint identity is inconsistent")
            expected_state = RawCheckpointState(
                trainer=update_state,
                growth=raw_state.growth,
                stage_budget=stage_budget,
                checkpoint_cadence=proposed_cadence,
            )
            if published_state != expected_state:
                raise ValueError("published RAW checkpoint state is inconsistent")
            checkpoint_publisher.apply_verified_retention(
                checkpoint_path,
                manifest,
                published_state,
            )

        loop: SingleGpuTrainingLoop[TrainingBatch] = SingleGpuTrainingLoop(
            module=module,
            optimizer=optimizer,
            loss_fn=measure_batch,
            accumulation_steps=config.stage.accumulation,
            target_successful_updates=target_successful_updates,
            checkpoint_every_successful_updates=config.checkpoint.full_every_updates,
            scheduler_step=scheduler_step,
            checkpoint=lambda _update: None,
            diagnostic_root=diagnostic_root,
            failure_id=failure_id,
            state=state,
            cadence=cadence,
            forced_checkpoint=forced_checkpoint,
            checkpoint_cadence_event=publish_and_verify,
            clock=clock,
            phase_timer=phase_timer,
            update_started=update_started,
            successful_update_observer=observe_update,
        )
        result = loop.run(stream)
    except BaseException as error:  # noqa: BLE001
        primary = error
    close_error: BaseException | None = None
    try:
        stream.close()
    except BaseException as error:  # noqa: BLE001
        close_error = error
    if primary is not None and close_error is not None:
        raise BaseExceptionGroup(
            "training failure and production stream close failure",
            [primary, close_error],
        ) from None
    if primary is not None:
        raise primary
    if close_error is not None:
        raise close_error
    assert result is not None
    return result


def run_single_gpu_training(
    config: RuntimeConfig,
    *,
    preflight: AcceptedPreflight,
    runtime: SingleGpuBatchRuntime,
    module: nn.Module,
    optimizer: StepOptimizer,
    batches: AcceptedProductionBatchStream,
    scheduler_step: Callable[[int], None],
    checkpoint_publisher: ProductionSingleGpuCheckpointPublisher,
    diagnostic_root: Path,
    failure_id: Callable[[str, SingleGpuUpdateState], str],
    restored_checkpoint: RestoredSingleGpuCheckpoint,
    phase_timer: PhaseTimer,
    successful_update_observer: Callable[[SuccessfulTrainingObservation], None],
    forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    clock: Callable[[], float] | None = None,
) -> LoopResult:
    """Run production training with the exact publisher accepted by preflight."""

    from sakuramoon.train.preflight import ProductionSingleGpuCheckpointPublisher

    if type(checkpoint_publisher) is not ProductionSingleGpuCheckpointPublisher:
        stream = require_accepted_production_batch_stream(batches)
        publisher_error = TypeError(
            "production training requires ProductionSingleGpuCheckpointPublisher"
        )
        try:
            stream.close()
        except BaseException as close_error:  # noqa: BLE001
            raise BaseExceptionGroup(
                "publisher rejection and production stream close failure",
                [publisher_error, close_error],
            ) from None
        raise publisher_error
    return _run_single_gpu_training(
        config,
        preflight=preflight,
        runtime=runtime,
        module=module,
        optimizer=optimizer,
        batches=batches,
        scheduler_step=scheduler_step,
        checkpoint_publisher=checkpoint_publisher,
        diagnostic_root=diagnostic_root,
        failure_id=failure_id,
        restored_checkpoint=restored_checkpoint,
        phase_timer=phase_timer,
        successful_update_observer=successful_update_observer,
        forced_checkpoint=forced_checkpoint,
        clock=clock,
    )


__all__ = [
    "DenseDiTAdapter",
    "PreparedTrainingBatch",
    "RuntimeMeasurement",
    "SingleGpuBatchRuntime",
    "SuccessfulTrainingObservation",
    "require_single_gpu_checkpoint_binding",
    "require_single_gpu_config",
    "run_single_gpu_training",
]
