"""Single-GPU data-to-update assembly at the typed service boundary.

This module owns only the trainer-side boundary.  The data service, its cache,
state store, and metadata policy remain outside the process and are supplied as
validated objects by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn

from sakuramoon.checkpoint.policy import (
    CheckpointCadence,
    CheckpointReason,
)
from sakuramoon.checkpoint.schema import StageBudgetCheckpointState
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.collate import DataLeaseClient, TrainingBatch, iter_service_batches
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.service_protocol import ShardLeaseDescriptor
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.qwen import FrozenQwenEncoder
from sakuramoon.model.dit import DenseDiT
from sakuramoon.objective.flow import (
    flow_matching_loss,
    interpolate_state,
    sample_jlt_timesteps,
    sample_noise,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.loop import LoopResult, SingleGpuTrainingLoop
from sakuramoon.train.step import (
    SingleGpuUpdateState,
    StepOptimizer,
    TrainableComposite,
    TrainableCompositeInputs,
)


class _Encoder(Protocol):
    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> object: ...


class _Vae(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...


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


class _PreleasedClient:
    """Make one already-issued worker-0 lease visible to ``iter_service_batches``."""

    def __init__(self, delegate: DataLeaseClient, first: ShardLeaseDescriptor) -> None:
        self._delegate = delegate
        self._first: ShardLeaseDescriptor | None = first
        self.identity = delegate.identity

    def health(self) -> bool:
        # The caller performed health before taking the first lease.  Returning
        # ``False`` here prevents the iterator from asking the service to repeat
        # that health request while the preleased descriptor is still active.
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if worker_id == 0 and self._first is not None:
            descriptor, self._first = self._first, None
            return descriptor
        return self._delegate.lease(worker_id)

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self._delegate.acknowledge(descriptor)


def service_batches(
    client: DataLeaseClient,
    *,
    pipeline_for_first_lease: Callable[[ShardLeaseDescriptor], WebDatasetPipeline],
    batch_size: int,
    worker_count: int,
    ready_batches: int,
    pin_memory: bool,
    drop_last: bool,
) -> Iterator[TrainingBatch]:
    """Consume D024 leases without moving ownership into the trainer.

    ``iter_service_batches`` needs a validated pipeline template before it can
    start its persistent workers.  The first lease supplies that template's
    trusted local path; the lease is then replayed through worker 0 exactly once.
    """

    if client.health():
        return iter(())
    first = client.lease(0)
    if first is None:
        return iter(())
    pipeline = pipeline_for_first_lease(first)
    return iter_service_batches(
        pipeline,
        _PreleasedClient(client, first),
        batch_size=batch_size,
        worker_count=worker_count,
        ready_batches=ready_batches,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


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
        input_ids = batch.input_ids.to(self.device, dtype=torch.long, non_blocking=True)
        attention_mask = batch.attention_mask.to(
            self.device, dtype=torch.bool, non_blocking=True
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

        images = batch.images.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        ).div(127.5).sub(1.0)
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
            main_token_indices=batch.main_token_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            ),
            main_mask=batch.main_mask.to(self.device, dtype=torch.bool, non_blocking=True),
            main_token_lengths=batch.main_token_lengths,
            artist_token_indices=batch.artist_token_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            ),
            artist_mask=batch.artist_mask.to(
                self.device, dtype=torch.bool, non_blocking=True
            ),
            use_null_style=batch.use_null_style.to(
                self.device, dtype=torch.bool, non_blocking=True
            ),
            active_style_sample_indices=batch.active_style_sample_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            ),
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
            loss = self._loss(predictions, prepared, None)
        else:
            with phase_timer.record("loss"):
                loss = self._loss(predictions, prepared, phase_timer)
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
            per_sample_loss=loss,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            sample_ids=sample_ids,
            shape_keys=(shape_key,) * len(sample_ids),
        )

    def _loss(
        self,
        predictions: tuple[torch.Tensor, ...],
        prepared: PreparedTrainingBatch,
        _phase_timer: PhaseTimer | None,
    ) -> torch.Tensor:
        if len({tuple(value.shape) for value in predictions}) == 1:
            prediction_batch = torch.stack(predictions)
            clean_batch = torch.stack(prepared.clean_latents)
            state_batch = torch.stack(prepared.states)
            return flow_matching_loss(
                prediction_batch,
                state_batch,
                clean_batch,
                prepared.inputs.timestep,
                t_eps=self.t_eps,
                noise_observation_boundary=self.noise_observation_boundary,
            ).per_sample
        values: list[torch.Tensor] = []
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
        return torch.stack(values)


def require_single_gpu_config(config: RuntimeConfig) -> None:
    """Reject every topology except the explicitly approved S0 single card."""

    if config.run.stage != "S0" or config.stage.world_size != 1:
        raise ValueError("the single-GPU runtime accepts only enabled S0 topology")
    if config.distributed.backend != "native" or config.distributed.world_size != 1:
        raise ValueError("single-GPU runtime requires native world_size=1")
    if config.failure.allow_force_bypass:
        raise ValueError("single-GPU runtime cannot enable preflight bypass")


def run_single_gpu_training(
    config: RuntimeConfig,
    *,
    runtime: SingleGpuBatchRuntime,
    module: nn.Module,
    optimizer: StepOptimizer,
    batches: Iterator[TrainingBatch],
    scheduler_step: Callable[[int], None],
    checkpoint: Callable[[int], None],
    diagnostic_root: Path,
    failure_id: Callable[[str, SingleGpuUpdateState], str],
    state: SingleGpuUpdateState,
    stage_budget: StageBudgetCheckpointState,
    cadence: CheckpointCadence,
    forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    checkpoint_event: Callable[[int, CheckpointReason], None] | None = None,
    checkpoint_cadence_event: Callable[
        [int, CheckpointReason, CheckpointCadence], None
    ]
    | None = None,
    clock: Callable[[], float] | None = None,
) -> LoopResult:
    """Run the locked loop after all ownership and assembly objects are ready."""

    require_single_gpu_config(config)
    if module is not runtime.composite:
        raise ValueError("training module must be the runtime trainable composite")
    if type(stage_budget) is not StageBudgetCheckpointState:
        raise TypeError("single-GPU training requires restored stage budget state")
    if not (
        stage_budget.start_successful_update
        <= state.successful_updates
        <= stage_budget.terminal_successful_update
    ):
        raise ValueError("restored stage budget is inconsistent with trainer state")
    if (
        stage_budget.terminal_successful_update
        - stage_budget.start_successful_update
        != config.stage.planned_updates
    ):
        raise ValueError("restored stage budget differs from resolved config")
    target_successful_updates = stage_budget.terminal_successful_update
    if state.successful_updates >= target_successful_updates:
        raise ValueError("stage successful-update budget is already exhausted")
    if type(cadence) is not CheckpointCadence:
        raise TypeError("single-GPU training requires restored checkpoint cadence")
    if cadence.last_successful_update != state.successful_updates:
        raise ValueError("checkpoint cadence update does not match trainer state")
    loop: SingleGpuTrainingLoop[TrainingBatch] = SingleGpuTrainingLoop(
        module=module,
        optimizer=optimizer,
        loss_fn=lambda batch: runtime.measure(batch).per_sample_loss,
        accumulation_steps=config.stage.accumulation,
        target_successful_updates=target_successful_updates,
        checkpoint_every_successful_updates=config.checkpoint.full_every_updates,
        scheduler_step=scheduler_step,
        checkpoint=checkpoint,
        diagnostic_root=diagnostic_root,
        failure_id=failure_id,
        state=state,
        cadence=cadence,
        forced_checkpoint=forced_checkpoint,
        checkpoint_event=checkpoint_event,
        checkpoint_cadence_event=checkpoint_cadence_event,
        clock=clock,
    )
    return loop.run(batches)


__all__ = [
    "DenseDiTAdapter",
    "PreparedTrainingBatch",
    "RuntimeMeasurement",
    "SingleGpuBatchRuntime",
    "require_single_gpu_config",
    "run_single_gpu_training",
    "service_batches",
]
