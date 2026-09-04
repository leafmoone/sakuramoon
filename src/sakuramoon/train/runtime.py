"""Single-GPU data-to-update assembly at the typed service boundary.

This module owns only the trainer-side boundary.  The data service, its cache,
state store, and metadata policy remain outside the process and are supplied as
validated objects by the caller.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn
from torch._dynamo import config as dynamo_config
from torch.nn.parallel import DistributedDataParallel

from sakuramoon.checkpoint.policy import (
    CheckpointCadence,
    CheckpointReason,
)
from sakuramoon.checkpoint.schema import CheckpointManifest, RawCheckpointState
from sakuramoon.conditioning.rope import full_canvas_crop_coordinates
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.caption import (
    CaptionDropoutCounts,
    CaptionPlan,
    ConditionRouteCounts,
)
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    require_accepted_production_batch_stream,
)
from sakuramoon.data.serialize import SerializedCaption
from sakuramoon.data.spatial_crop import SpatialCropCounts
from sakuramoon.data.transparent_white import TransparentWhiteCounts
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.pe_spatial import FrozenPESpatialEncoder
from sakuramoon.encoders.qwen import FrozenQwenEncoder
from sakuramoon.model.attention import (
    DenseGQAAttention,
    FA4VarlenGQAAttention,
    fa4_varlen_attention,
)
from sakuramoon.model.block import DiTBlock, PackedDiTBlock
from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.objective.flow import (
    flow_matching_loss,
    interpolate_state,
    sample_jlt_timesteps,
    sample_noise,
)
from sakuramoon.objective.irepa import (
    IRepaLambdaSchedule,
    irepa_alignment_loss,
    spatial_zscore_target,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.loop import (
    LoopResult,
    SingleGpuTrainingLoop,
    SuccessfulLoopObservation,
)
from sakuramoon.train.stage import canonical_growth_alpha
from sakuramoon.train.step import (
    SingleGpuUpdateState,
    StepOptimizer,
    TrainableComposite,
    TrainableCompositeInputs,
    TrainableCompositeIRepaOutput,
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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        dense_lengths: tuple[int, ...] | None = None,
    ) -> object: ...


class _Vae(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...


def require_distributed_forward_module(
    composite: TrainableComposite,
    forward_module: nn.Module,
) -> DistributedDataParallel:
    """Require an eager DDP wrapper around the original trainable composite."""

    if forward_module is composite:
        raise ValueError("distributed forward must wrap the trainable composite")
    if not isinstance(forward_module, DistributedDataParallel):
        raise TypeError(
            "distributed forward must remain an uncompiled DistributedDataParallel"
        )
    if forward_module.module is not composite:
        raise ValueError(
            "DistributedDataParallel must wrap the original trainable composite"
        )
    return forward_module


def compile_packed_dit_blocks(
    composite: TrainableComposite,
    *,
    backend: str,
    mode: str,
    dynamic: bool,
) -> tuple[PackedDiTBlock, ...]:
    """Install fail-closed regional compilation after DDP construction."""

    if (
        backend != "inductor"
        or mode not in {"default", "reduce-overhead", "max-autotune-no-cudagraphs"}
        or type(dynamic) is not bool
    ):
        raise ValueError("regional torch.compile configuration is invalid")
    if not dynamic:
        raise ValueError("packed regional torch.compile requires dynamic=true")
    if dynamo_config.suppress_errors:
        raise RuntimeError("torch.compile error suppression must remain disabled")
    dynamo_config.fail_on_recompile_limit_hit = True
    if not getattr(fa4_varlen_attention, "_torchdynamo_disable", False):
        raise RuntimeError("DAS FA2 must remain an explicit eager compiler boundary")

    dit = composite.dit
    if not isinstance(dit, PackedDiT):
        raise TypeError("regional torch.compile requires the production PackedDiT")
    blocks = tuple(dit.blocks.values())
    if len(blocks) != len(dit.active_slot_ids) or any(
        not isinstance(block, PackedDiTBlock) for block in blocks
    ):
        raise TypeError("PackedDiT block registry is inconsistent")
    if any(
        child._forward_pre_hooks or child._forward_hooks
        for block in blocks
        for child in block.modules()
    ):
        raise RuntimeError("regional compile blocks must not carry Python forward hooks")
    if any(getattr(block, "_compiled_call_impl", None) is not None for block in blocks):
        raise RuntimeError("PackedDiT blocks are already compiled")

    parameter_ids = tuple(id(parameter) for parameter in composite.parameters())
    state_keys = tuple(composite.state_dict())
    for block in blocks:
        block.compile(
            backend=backend,
            mode=mode,
            fullgraph=False,
            dynamic=dynamic,
        )
    if any(getattr(block, "_compiled_call_impl", None) is None for block in blocks):
        raise RuntimeError("regional torch.compile was not installed on every block")
    if parameter_ids != tuple(id(parameter) for parameter in composite.parameters()):
        raise RuntimeError("regional torch.compile changed parameter identity")
    if state_keys != tuple(composite.state_dict()):
        raise RuntimeError("regional torch.compile changed state_dict keys")
    return cast(tuple[PackedDiTBlock, ...], blocks)


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
        condition_tokens: torch.Tensor,
        condition_active_mask: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        image_coordinates: tuple[torch.Tensor, ...],
        growth_alpha: float,
    ) -> tuple[torch.Tensor, ...]:
        del text_lengths
        latent_batch = torch.stack(latents)
        predictions = self.model(
            latent_batch,
            text_tokens,
            text_mask,
            condition_tokens,
            condition_active_mask,
            timestep,
            size_scale,
            aspect,
            image_coordinates=torch.stack(image_coordinates),
            growth_alpha=growth_alpha,
        )
        return tuple(predictions.unbind(0))

    def model_metadata(self) -> dict[str, int | str]:
        return self.model.model_metadata()


    def artifact_config(self) -> dict[str, object]:
        return self.model.artifact_config()


class ActualDitFlopCounter:
    """Compute exact DiT matmul FLOPs outside eager and compiled graphs.

    One multiply-add is two FLOPs. The locked DiT linear topology is checked
    once at construction; QK and AV products are added from sequence lengths.
    """

    def __init__(self, module: object) -> None:
        model = module.model if isinstance(module, DenseDiTAdapter) else module
        if not isinstance(model, (DenseDiT, PackedDiT)):
            raise TypeError("DiT FLOP counting requires DenseDiT or PackedDiT")
        self._model = model
        self._packed = isinstance(model, PackedDiT)
        condition_token_count = model.condition_token_count
        if type(condition_token_count) is not int or condition_token_count <= 0:
            raise ValueError(
                "DiT FLOP counting requires a positive integer "
                "condition_token_count"
            )
        self._condition_token_count = condition_token_count
        block_type = PackedDiTBlock if self._packed else DiTBlock
        attention_type = (
            FA4VarlenGQAAttention if self._packed else DenseGQAAttention
        )
        blocks = tuple(model.blocks.values())
        if len(blocks) != len(model.active_slot_ids) or any(
            not isinstance(block, block_type) for block in blocks
        ):
            raise TypeError("DiT block registry is inconsistent")
        attentions = tuple(block.attention for block in blocks)
        if any(not isinstance(item, attention_type) for item in attentions):
            raise TypeError("DiT attention registry is inconsistent")

        conditioner_linears = tuple(
            child
            for child in model.conditioner.modules()
            if isinstance(child, nn.Linear)
        )
        block_linears = tuple(
            child
            for block in blocks
            for child in block.modules()
            if isinstance(child, nn.Linear)
        )
        boundary_linears = (model.input_projection, model.output_head.projection)
        accounted = (*boundary_linears, *conditioner_linears, *block_linears)
        observed = tuple(
            child for child in model.modules() if isinstance(child, nn.Linear)
        )
        if len({id(item) for item in accounted}) != len(accounted):
            raise RuntimeError("DiT FLOP registry contains duplicate linears")
        if {id(item) for item in accounted} != {id(item) for item in observed}:
            raise RuntimeError("DiT topology contains an unaccounted linear")

        self._image_linear_flops = sum(
            self._linear_flops(item) for item in boundary_linears
        )
        self._conditioner_linear_flops = sum(
            self._linear_flops(item) for item in conditioner_linears
        )
        self._block_linear_flops = sum(
            self._linear_flops(item) for item in block_linears
        )
        self._attention_flops = sum(
            4 * item.q_heads * item.head_dim for item in attentions
        )
        if min(
            self._image_linear_flops,
            self._conditioner_linear_flops,
            self._block_linear_flops,
            self._attention_flops,
        ) <= 0:
            raise RuntimeError("DiT FLOP coefficients must be positive")

    @staticmethod
    def _linear_flops(module: nn.Linear) -> int:
        return 2 * module.in_features * module.out_features

    def count(self, inputs: TrainableCompositeInputs) -> int:
        """Return exact forward matmul FLOPs for accepted composite inputs."""

        if not isinstance(inputs, TrainableCompositeInputs):
            raise TypeError("DiT FLOP counting requires TrainableCompositeInputs")
        batch = len(inputs.latents)
        if (
            batch <= 0
            or inputs.main_token_indices.ndim != 2
            or inputs.main_token_indices.shape[0] != batch
        ):
            raise ValueError("DiT FLOP inputs contain an invalid batch")
        image_lengths: list[int] = []
        for latent in inputs.latents:
            if (
                latent.ndim != 3
                or latent.shape[0] != self._model.input_projection.in_features
                or latent.shape[-2] <= 0
                or latent.shape[-1] <= 0
            ):
                raise ValueError("DiT FLOP latent shapes are invalid")
            image_lengths.append(latent.shape[-2] * latent.shape[-1])
        image_tokens = sum(image_lengths)

        if self._packed:
            text_lengths = inputs.main_token_lengths
            if (
                type(text_lengths) is not tuple
                or len(text_lengths) != batch
                or any(type(item) is not int or item <= 0 for item in text_lengths)
            ):
                raise ValueError("packed DiT FLOP text lengths are invalid")
            sequence_lengths = tuple(
                text_length + self._condition_token_count + image_length
                for text_length, image_length in zip(
                    text_lengths, image_lengths, strict=True
                )
            )
            block_vectors = sum(sequence_lengths)
            attention_squares = sum(item * item for item in sequence_lengths)
        else:
            if len(set(image_lengths)) != 1:
                raise ValueError("dense DiT FLOP latents must share one image shape")
            dense_text_length = inputs.main_token_indices.shape[1]
            if dense_text_length <= 0:
                raise ValueError("dense DiT FLOP text width must be positive")
            sequence_length = (
                dense_text_length
                + self._condition_token_count
                + image_lengths[0]
            )
            block_vectors = batch * sequence_length
            attention_squares = batch * sequence_length * sequence_length

        flops = (
            image_tokens * self._image_linear_flops
            + batch * self._conditioner_linear_flops
            + block_vectors * self._block_linear_flops
            + attention_squares * self._attention_flops
        )
        if type(flops) is not int or flops <= 0:
            raise RuntimeError("DiT FLOP result is invalid")
        return flops


@dataclass(frozen=True, slots=True)
class PreparedTrainingBatch:
    """The immutable tensors needed by one objective evaluation."""

    inputs: TrainableCompositeInputs
    clean_latents: tuple[torch.Tensor, ...]
    states: tuple[torch.Tensor, ...]
    # Frozen-teacher z-scored patch features [B, T, 768] FP32, detached.
    # None when iREPA is absent/disabled.  Deliberately NOT part of
    # TrainableCompositeInputs: the teacher target never enters the trainable
    # module, the optimizer, or the checkpoint.
    irepa_targets: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    """A loss vector plus counters consumed by the loop/benchmark adapters."""

    per_sample_loss: torch.Tensor
    image_tokens: int
    text_tokens: int
    dit_flops: int
    sample_ids: tuple[str, ...]
    shape_keys: tuple[str, ...]
    high_noise_loss_sum: torch.Tensor
    high_noise_sample_count: torch.Tensor
    low_noise_loss_sum: torch.Tensor
    low_noise_sample_count: torch.Tensor
    timesteps: torch.Tensor
    dropout_hits: CaptionDropoutCounts
    condition_routes: ConditionRouteCounts
    captions: tuple[SerializedCaption, ...]
    caption_plans: tuple[CaptionPlan, ...]
    spatial_crop: SpatialCropCounts
    transparent: TransparentWhiteCounts
    # Phase-4 iREPA split of the per-sample losses (all FP32, shape [B]).
    # ``per_sample_loss`` is the actual backward objective:
    # main + lambda * irepa when enabled, plain main otherwise.  The high/low
    # noise buckets and the t-bin telemetry stay strictly MAIN-JLT-only via
    # ``main_per_sample_loss`` / the FlowLossOutput bucket sums.
    main_per_sample_loss: torch.Tensor
    irepa_per_sample_loss: torch.Tensor
    irepa_weighted_per_sample_loss: torch.Tensor
    irepa_cosine: torch.Tensor
    irepa_lambda: float

    def detached(self) -> RuntimeMeasurement:
        """Drop the autograd graph before handing facts to an async observer."""

        return RuntimeMeasurement(
            per_sample_loss=self.per_sample_loss.detach(),
            image_tokens=self.image_tokens,
            text_tokens=self.text_tokens,
            dit_flops=self.dit_flops,
            sample_ids=self.sample_ids,
            shape_keys=self.shape_keys,
            high_noise_loss_sum=self.high_noise_loss_sum.detach(),
            high_noise_sample_count=self.high_noise_sample_count.detach(),
            low_noise_loss_sum=self.low_noise_loss_sum.detach(),
            low_noise_sample_count=self.low_noise_sample_count.detach(),
            timesteps=self.timesteps.detach(),
            dropout_hits=self.dropout_hits,
            condition_routes=self.condition_routes,
            captions=self.captions,
            caption_plans=self.caption_plans,
            spatial_crop=self.spatial_crop,
            transparent=self.transparent,
            main_per_sample_loss=self.main_per_sample_loss.detach(),
            irepa_per_sample_loss=self.irepa_per_sample_loss.detach(),
            irepa_weighted_per_sample_loss=self.irepa_weighted_per_sample_loss.detach(),
            irepa_cosine=self.irepa_cosine.detach(),
            irepa_lambda=self.irepa_lambda,
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
                item.main_per_sample_loss,
                item.irepa_per_sample_loss,
                item.irepa_weighted_per_sample_loss,
                item.irepa_cosine,
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
    main_per_sample: torch.Tensor
    irepa_per_sample: torch.Tensor
    irepa_weighted_per_sample: torch.Tensor
    irepa_cosine: torch.Tensor
    irepa_lambda: float


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
    if batch.condition_token_indices.shape[0] != batch.images.shape[0]:
        raise ValueError("condition token routing batch size differs from images")
    if len(batch.main_token_lengths) != batch.images.shape[0]:
        raise ValueError("main token length count differs from images")
    if len(batch.source_shards) != batch.images.shape[0]:
        raise ValueError("source shard count differs from images")
    if len(batch.audits) != batch.images.shape[0]:
        raise ValueError("image audit count differs from images")
    if len(batch.captions) != batch.images.shape[0]:
        raise ValueError("structured caption count differs from images")
    if batch.target_height <= 0 or batch.target_width <= 0:
        raise ValueError("training target dimensions must be positive")
    if tuple(batch.images.shape[-2:]) != (
        batch.target_height,
        batch.target_width,
    ):
        raise ValueError("training image shape differs from its target canvas")


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


def _full_canvas_coordinate_maps(
    batch: TrainingBatch,
    *,
    token_height: int,
    token_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    maps: list[torch.Tensor] = []
    for sample_index, audit in enumerate(batch.audits):
        try:
            coordinates = full_canvas_crop_coordinates(
                token_height,
                token_width,
                full_height=audit.resized_height,
                full_width=audit.resized_width,
                crop_box=audit.crop_box,
                device=device,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"image audit {sample_index} cannot define full-canvas coordinates"
            ) from error
        left, top, right, bottom = audit.crop_box
        if (bottom - top, right - left) != (
            batch.target_height,
            batch.target_width,
        ):
            raise ValueError(
                f"image audit {sample_index} crop differs from the training target"
            )
        maps.append(coordinates)
    return tuple(maps)


class SingleGpuBatchRuntime:
    """Encode one service batch and produce its differentiable loss vector."""

    def __init__(
        self,
        *,
        qwen: FrozenQwenEncoder | _Encoder,
        vae: FrozenMageVAE | _Vae,
        composite: TrainableComposite,
        forward_module: nn.Module | None = None,
        device: torch.device,
        generator: torch.Generator,
        p_mean: float,
        p_std: float,
        noise_scale: float,
        t_eps: float,
        noise_observation_boundary: float,
        growth_alpha: float,
        torch_compile_enabled: bool = False,
        torch_compile_backend: str = "inductor",
        torch_compile_mode: str = "default",
        torch_compile_dynamic: bool = False,
        irepa_teacher: FrozenPESpatialEncoder | None = None,
        irepa_gamma: float = 0.6,
        irepa_eps: float = 1e-6,
        irepa_schedule: IRepaLambdaSchedule | None = None,
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
        if (
            type(torch_compile_enabled) is not bool
            or torch_compile_backend != "inductor"
            or torch_compile_mode
            not in {"default", "reduce-overhead", "max-autotune-no-cudagraphs"}
            or type(torch_compile_dynamic) is not bool
        ):
            raise ValueError("torch.compile configuration is invalid")
        self.qwen = qwen
        self.vae = vae
        self.composite = composite
        self.forward_module = composite if forward_module is None else forward_module
        if self.forward_module is not composite:
            require_distributed_forward_module(composite, self.forward_module)
        blocks = (
            tuple(composite.dit.blocks.values())
            if isinstance(composite.dit, PackedDiT)
            else ()
        )
        compiled_blocks = tuple(
            block
            for block in blocks
            if getattr(block, "_compiled_call_impl", None) is not None
        )
        if torch_compile_enabled:
            if not isinstance(composite.dit, PackedDiT):
                raise TypeError("regional torch.compile requires PackedDiT")
            if not torch_compile_dynamic:
                raise ValueError("packed regional torch.compile requires dynamic=true")
            if len(compiled_blocks) != len(blocks) or not blocks:
                raise RuntimeError(
                    "regional torch.compile must be installed before runtime creation"
                )
        elif compiled_blocks:
            raise RuntimeError("compiled DiT blocks are present while compile is disabled")
        self.device = device
        self.generator = generator
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.noise_observation_boundary = noise_observation_boundary
        self.growth_alpha = growth_alpha
        self.dit_flop_counter = ActualDitFlopCounter(composite.dit)
        # Phase-4 iREPA wiring: the frozen teacher stays OUTSIDE the composite
        # (per-rank, eval, no_grad, never DDP-wrapped, never optimized, never
        # checkpointed).  Teacher presence must match the composite projector
        # exactly — both absent/disabled or both installed.
        has_projector = composite.irepa_alignment is not None
        if has_projector != (irepa_teacher is not None):
            raise ValueError(
                "iREPA teacher and composite projector must both be present "
                "or both be absent"
            )
        if irepa_teacher is not None:
            if type(irepa_gamma) is not float or not 0.0 <= irepa_gamma <= 1.0:
                raise ValueError("irepa_gamma must be a float in [0,1]")
            if type(irepa_eps) is not float or not irepa_eps > 0.0:
                raise ValueError("irepa_eps must be a positive float")
            if irepa_schedule is None:
                raise ValueError(
                    "an installed iREPA runtime requires a lambda schedule"
                )
        elif irepa_schedule is not None:
            raise ValueError(
                "an iREPA lambda schedule requires an installed teacher"
            )
        self.irepa_teacher = irepa_teacher
        self.irepa_gamma = irepa_gamma
        self.irepa_eps = irepa_eps
        self.irepa_schedule = irepa_schedule
        # Bound once per successful update by set_irepa_weight (mirrors
        # set_growth_alpha); 0.0 until the first binding.  A lambda of 0.0
        # still runs the FULL teacher/capture/projector/z-score/cosine path.
        self.irepa_weight = 0.0

    def set_growth_alpha(self, value: float) -> None:
        """Select the canonical alpha before starting one successful update."""

        if type(value) is not float or not 0.0 <= value <= 1.0:
            raise ValueError("growth_alpha must be a float in [0,1]")
        self.growth_alpha = value

    def set_irepa_weight(self, value: float) -> None:
        """Bind the iREPA lambda before starting one successful update.

        Mirrors ``set_growth_alpha``: the value is a pure function of the
        successful-update number the in-flight update will produce, so a
        failed update retries with the same lambda.
        """

        if self.irepa_teacher is None:
            raise ValueError("iREPA weight binding requires an installed teacher")
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError("irepa weight must be a finite nonnegative float")
        self.irepa_weight = value

    def _encode_qwen(
        self,
        batch: TrainingBatch,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> object:
        if not isinstance(self.qwen, FrozenQwenEncoder):
            return self.qwen(input_ids, attention_mask)
        dense_lengths = (
            tuple(caption.dense_length for caption in batch.captions)
            if len(batch.captions) == batch.images.shape[0]
            else (batch.dense_length,) * batch.images.shape[0]
        )
        return self.qwen(
            input_ids,
            attention_mask,
            dense_lengths=dense_lengths,
        )

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
            condition_token_indices = batch.condition_token_indices.to(
                self.device, dtype=torch.long, non_blocking=True
            )
            condition_mask = batch.condition_mask.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            use_null_condition = batch.use_null_condition.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
            active_condition_sample_indices = batch.active_condition_sample_indices.to(
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
                condition_token_indices = batch.condition_token_indices.to(
                    self.device, dtype=torch.long, non_blocking=True
                )
                condition_mask = batch.condition_mask.to(
                    self.device, dtype=torch.bool, non_blocking=True
                )
                use_null_condition = batch.use_null_condition.to(
                    self.device, dtype=torch.bool, non_blocking=True
                )
                active_condition_sample_indices = (
                    batch.active_condition_sample_indices.to(
                        self.device, dtype=torch.long, non_blocking=True
                    )
                )
        if phase_timer is None:
            qwen_output = self._encode_qwen(batch, input_ids, attention_mask)
        else:
            with phase_timer.record("qwen"):
                qwen_output = self._encode_qwen(batch, input_ids, attention_mask)
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
        # iREPA teacher target: consumes the EXACT final GPU RGB tensor
        # (same bf16 [-1,1] tensor the VAE encoded — no second H2D, no
        # resize), runs per rank outside the composite/DDP, no_grad.
        irepa_targets: torch.Tensor | None = None
        if self.irepa_teacher is not None:
            if phase_timer is None:
                teacher_output = self.irepa_teacher(images)
            else:
                with phase_timer.record("irepa_teacher"):
                    teacher_output = self.irepa_teacher(images)
            if tuple(teacher_output.patch_features.shape[:2]) != (
                batch.images.shape[0],
                (batch.target_height // 16) * (batch.target_width // 16),
            ):
                raise ValueError("iREPA teacher returned an unexpected grid")
            irepa_targets = spatial_zscore_target(
                teacher_output.patch_features,
                gamma=self.irepa_gamma,
                eps=self.irepa_eps,
            ).detach()
        image_coordinates = _full_canvas_coordinate_maps(
            batch,
            token_height=clean.shape[-2],
            token_width=clean.shape[-1],
            device=clean.device,
        )

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
            condition_token_indices=condition_token_indices,
            condition_mask=condition_mask,
            use_null_condition=use_null_condition,
            active_condition_sample_indices=active_condition_sample_indices,
            latents=tuple(item for item in state.unbind(0)),
            image_coordinates=image_coordinates,
            timestep=timestep,
            size_scale=size_scale,
            aspect=aspect,
            growth_alpha=self.growth_alpha,
        )
        return PreparedTrainingBatch(
            inputs=inputs,
            clean_latents=tuple(item for item in clean.unbind(0)),
            states=tuple(item for item in state.unbind(0)),
            irepa_targets=irepa_targets,
        )

    def measure(
        self, batch: TrainingBatch, *, phase_timer: PhaseTimer | None = None
    ) -> RuntimeMeasurement:
        prepared = self.prepare(batch, phase_timer=phase_timer)
        dit_flops = self.dit_flop_counter.count(prepared.inputs)
        # The DDP-wrapped module is the composite itself, so the projector
        # runs INSIDE this forward (never on forward_module.module from the
        # outside) and its parameters are part of the distributed graph.
        forward_output = self.forward_module(
            prepared.inputs,
            phase_timer=phase_timer,
        )
        if isinstance(forward_output, TrainableCompositeIRepaOutput):
            predictions = forward_output.predictions
            projected = forward_output.projected_student_features
            if prepared.irepa_targets is None:
                raise RuntimeError(
                    "iREPA composite output arrived without a teacher target"
                )
        else:
            predictions = forward_output
            projected = None
            if self.irepa_teacher is not None:
                raise RuntimeError(
                    "iREPA teacher was configured but the composite has no "
                    "projector"
                )
        if len(predictions) != len(prepared.clean_latents):
            raise ValueError("DiT prediction count differs from latent batch")
        if phase_timer is None:
            loss = self._loss(predictions, prepared, projected)
        else:
            with phase_timer.record("loss"):
                loss = self._loss(predictions, prepared, projected)
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
            dit_flops=dit_flops,
            sample_ids=sample_ids,
            shape_keys=(shape_key,) * len(sample_ids),
            high_noise_loss_sum=loss.high_noise_loss_sum,
            high_noise_sample_count=loss.high_noise_sample_count,
            low_noise_loss_sum=loss.low_noise_loss_sum,
            low_noise_sample_count=loss.low_noise_sample_count,
            timesteps=prepared.inputs.timestep,
            dropout_hits=batch.dropout_hits,
            condition_routes=batch.condition_routes,
            captions=batch.captions,
            caption_plans=tuple(caption.plan for caption in batch.captions),
            spatial_crop=batch.spatial_crop,
            transparent=batch.transparent,
            main_per_sample_loss=loss.main_per_sample,
            irepa_per_sample_loss=loss.irepa_per_sample,
            irepa_weighted_per_sample_loss=loss.irepa_weighted_per_sample,
            irepa_cosine=loss.irepa_cosine,
            irepa_lambda=loss.irepa_lambda,
        )

    def _loss(
        self,
        predictions: tuple[torch.Tensor, ...],
        prepared: PreparedTrainingBatch,
        projected: torch.Tensor | None,
    ) -> _RuntimeLoss:
        if projected is not None and prepared.irepa_targets is None:
            raise RuntimeError("iREPA projected features arrived without a target")
        if projected is not None and prepared.irepa_targets is not None:
            alignment = irepa_alignment_loss(projected, prepared.irepa_targets)
            irepa_per_sample = alignment.per_sample
            irepa_cosine = alignment.cosine_per_sample
            irepa_weighted = self.irepa_weight * irepa_per_sample
            irepa_lambda = self.irepa_weight
        else:
            irepa_per_sample = None
            irepa_cosine = None
            irepa_weighted = None
            irepa_lambda = 0.0

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
            main_per_sample = result.per_sample
            high_sum = result.high_noise_loss_sum
            high_count = result.high_noise_sample_count
            low_sum = result.low_noise_loss_sum
            low_count = result.low_noise_sample_count
        else:
            values: list[torch.Tensor] = []
            high_loss: list[torch.Tensor] = []
            high_count_parts: list[torch.Tensor] = []
            low_loss: list[torch.Tensor] = []
            low_count_parts: list[torch.Tensor] = []
            for index, prediction in enumerate(predictions):
                per_sample_result = flow_matching_loss(
                    prediction.unsqueeze(0),
                    prepared.states[index].unsqueeze(0),
                    prepared.clean_latents[index].unsqueeze(0),
                    prepared.inputs.timestep[index : index + 1],
                    t_eps=self.t_eps,
                    noise_observation_boundary=self.noise_observation_boundary,
                )
                values.append(per_sample_result.per_sample[0])
                high_loss.append(per_sample_result.high_noise_loss_sum)
                high_count_parts.append(per_sample_result.high_noise_sample_count)
                low_loss.append(per_sample_result.low_noise_loss_sum)
                low_count_parts.append(per_sample_result.low_noise_sample_count)
            main_per_sample = torch.stack(values)
            high_sum = torch.stack(high_loss).sum()
            high_count = torch.stack(high_count_parts).sum()
            low_sum = torch.stack(low_loss).sum()
            low_count = torch.stack(low_count_parts).sum()

        if (
            irepa_per_sample is None
            or irepa_cosine is None
            or irepa_weighted is None
        ):
            # iREPA absent/disabled: the backward objective is EXACTLY the
            # legacy main per-sample vector (byte-identical path); the split
            # fields are strict zeros (no NaN sentinels).
            zero_irepa = main_per_sample.new_zeros(main_per_sample.shape)
            return _RuntimeLoss(
                main_per_sample,
                high_sum,
                high_count,
                low_sum,
                low_count,
                main_per_sample=main_per_sample,
                irepa_per_sample=zero_irepa,
                irepa_weighted_per_sample=zero_irepa,
                irepa_cosine=zero_irepa,
                irepa_lambda=0.0,
            )
        # Enabled: the iREPA path is ALWAYS in the backward graph
        # (spec: no "if lambda == 0: skip"); at lambda=0 its contribution is
        # an exact zero, so every legacy gradient stays bit-identical.
        return _RuntimeLoss(
            main_per_sample + irepa_weighted,
            high_sum,
            high_count,
            low_sum,
            low_count,
            main_per_sample=main_per_sample,
            irepa_per_sample=irepa_per_sample,
            irepa_weighted_per_sample=irepa_weighted,
            irepa_cosine=irepa_cosine,
            irepa_lambda=irepa_lambda,
        )


def require_single_gpu_config(config: RuntimeConfig) -> None:
    """Accept the governed S0/G1 native or Accelerate topology."""

    if (
        config.run.intent != "train"
        or config.run.stage not in {"S0", "G1"}
        or not config.stage.enabled
        or config.stage.world_size not in {1, 2}
    ):
        raise ValueError(
            "the production runtime accepts only governed train-intent S0/G1 topology"
        )
    topology = (config.distributed.backend, config.distributed.world_size)
    if topology not in {("native", 1), ("accelerate", 2)}:
        raise ValueError("production runtime requires native/1 or accelerate/2")
    if config.failure.allow_force_bypass:
        raise ValueError("single-GPU runtime cannot enable preflight bypass")


def require_single_gpu_checkpoint_compatibility(
    config: RuntimeConfig,
    state: RawCheckpointState,
    *,
    runtime_growth_alpha: float,
) -> None:
    """Bind a RAW checkpoint to the resolved S0/G1 model and stage."""

    require_single_gpu_config(config)
    require_checkpoint_cadence_binding(config, state)
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
        raise ValueError(
            "restored checkpoint ramp presence differs from resolved growth"
        )
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
    if config.stage.name == "S0" and stage_budget.start_successful_update != 0:
        raise ValueError("restored S0 stage budget must start at update zero")
    if config.stage.name == "G1" and stage_budget.start_successful_update <= 0:
        raise ValueError("restored G1 stage budget must start at its transition update")
    if stage_budget.terminal_successful_update != config.stage.planned_updates:
        raise ValueError("restored stage budget differs from resolved config")
    if state.checkpoint_cadence.last_successful_update != trainer.successful_updates:
        raise ValueError("checkpoint cadence update does not match trainer state")


def require_single_gpu_checkpoint_binding(
    config: RuntimeConfig,
    state: RawCheckpointState,
    *,
    runtime_growth_alpha: float,
) -> None:
    """Require a compatible checkpoint that still has training updates left."""

    require_single_gpu_checkpoint_compatibility(
        config,
        state,
        runtime_growth_alpha=runtime_growth_alpha,
    )
    if (
        state.trainer.successful_updates
        >= state.stage_budget.terminal_successful_update
    ):
        raise ValueError("stage successful-update budget is already exhausted")


def require_checkpoint_cadence_binding(
    config: RuntimeConfig,
    state: RawCheckpointState,
) -> None:
    """Bind a persisted RAW cadence to the resolved TOML interval."""

    interval = config.checkpoint.full_every_updates
    if type(interval) is not int or interval <= 0:
        raise ValueError("resolved checkpoint update interval is invalid")
    if state.checkpoint_cadence.every_successful_updates != interval:
        raise ValueError("restored checkpoint cadence differs from resolved config")


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
    verified_checkpoint_observer: Callable[[Path], None] | None = None,
    forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    clock: Callable[[], float] | None = None,
    backward: Callable[[torch.Tensor], None] | None = None,
    no_sync: Callable[[], AbstractContextManager[None]] | None = None,
    log_updates: bool = True,
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
        active_learning_rate: float | None = None
        next_growth_update = state.successful_updates + 1

        def update_started(timer: PhaseTimer | None) -> None:
            nonlocal active_learning_rate, active_phase_timer
            active_phase_timer = timer
            active_learning_rate = _optimizer_learning_rate(optimizer)
            runtime.set_growth_alpha(
                canonical_growth_alpha(
                    raw_state.growth,
                    next_growth_update,
                )
            )
            # iREPA lambda binding mirrors growth: a pure function of the
            # successful-update number this in-flight update will produce.
            # A failed update does not advance it, so the retry rebinds the
            # same value.  Production keeps this dormant: Phase 5 provides
            # the persisted start anchor; until then the production
            # enabled path stays fail-closed at the readiness gate.
            if runtime.irepa_schedule is not None:
                runtime.set_irepa_weight(
                    runtime.irepa_schedule.weight_for_update(next_growth_update)
                )

        def measure_batch(batch: TrainingBatch) -> torch.Tensor:
            if active_phase_timer is None:
                raise RuntimeError("training update phase timer was not initialized")
            measurement = runtime.measure(batch, phase_timer=active_phase_timer)
            pending_measurements.append(measurement.detached())
            return measurement.per_sample_loss

        def observe_update(observation: SuccessfulLoopObservation) -> None:
            nonlocal active_learning_rate, active_phase_timer, next_growth_update
            microbatches = tuple(pending_measurements)
            update_timer = observation.phase_timer
            if update_timer is None or update_timer is not active_phase_timer:
                raise RuntimeError("training observation phase timer identity changed")
            if active_learning_rate is None:
                raise RuntimeError("training update learning rate was not captured")
            if observation.update.state.successful_updates != next_growth_update:
                raise RuntimeError("growth alpha update edge changed during the update")
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
                learning_rate=active_learning_rate,
                gpu_memory_allocated_bytes=allocated,
                gpu_memory_reserved_bytes=reserved,
            )
            successful_update_observer(emitted)
            if log_updates:
                print(
                    f"[train] update={observation.update.state.successful_updates} "
                    f"loss={float(observation.update.mean_loss.detach().item()):.6f} "
                    f"time={observation.update_wall_seconds:.2f}s",
                    flush=True,
                )
            pending_measurements.clear()
            active_learning_rate = None
            active_phase_timer = None
            next_growth_update += 1

        def publish_and_verify(
            update_state: SingleGpuUpdateState,
            reason: CheckpointReason,
            proposed_cadence: CheckpointCadence,
        ) -> None:
            checkpoint_path = checkpoint_publisher.publish_update(
                update_state, reason, proposed_cadence
            )
            if log_updates:
                print(f"[train] 保存模型: {checkpoint_path}", flush=True)
            manifest, published_state = read_raw_checkpoint_state(checkpoint_path)
            restored_identity = restored_checkpoint.manifest.identity
            published_identity = manifest.identity
            if (
                published_identity.update != update_state.successful_updates
                or published_identity.checkpoint_id == restored_identity.checkpoint_id
            ):
                raise ValueError("published RAW checkpoint identity is inconsistent")
            expected_state = RawCheckpointState(
                trainer=update_state,
                growth=replace(
                    raw_state.growth,
                    alpha=canonical_growth_alpha(
                        raw_state.growth, update_state.successful_updates
                    ),
                ),
                stage_budget=stage_budget,
                checkpoint_cadence=replace(
                    proposed_cadence,
                    last_wall_clock_unix_seconds=(
                        published_state.checkpoint_cadence.last_wall_clock_unix_seconds
                    ),
                ),
            )
            if published_state != expected_state:
                raise ValueError("published RAW checkpoint state is inconsistent")
            checkpoint_publisher.apply_verified_retention(
                checkpoint_path,
                manifest,
                published_state,
            )
            if verified_checkpoint_observer is not None:
                verified_checkpoint_observer(checkpoint_path)

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
            effective_sample_multiplier=config.distributed.world_size,
            growth_alpha_for_update=lambda update: canonical_growth_alpha(
                raw_state.growth, update
            ),
            backward=backward,
            no_sync=no_sync,
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
    verified_checkpoint_observer: Callable[[Path], None] | None = None,
    forced_checkpoint: Callable[[int], CheckpointReason | None] | None = None,
    clock: Callable[[], float] | None = None,
    backward: Callable[[torch.Tensor], None] | None = None,
    no_sync: Callable[[], AbstractContextManager[None]] | None = None,
    log_updates: bool = True,
) -> LoopResult:
    """Run production training with the exact publisher accepted by preflight."""

    from sakuramoon.train.preflight import ProductionSingleGpuCheckpointPublisher

    if not isinstance(
        checkpoint_publisher, ProductionSingleGpuCheckpointPublisher
    ):
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
        verified_checkpoint_observer=verified_checkpoint_observer,
        forced_checkpoint=forced_checkpoint,
        clock=clock,
        backward=backward,
        no_sync=no_sync,
        log_updates=log_updates,
    )


__all__ = [
    "ActualDitFlopCounter",
    "DenseDiTAdapter",
    "PreparedTrainingBatch",
    "RuntimeMeasurement",
    "SingleGpuBatchRuntime",
    "SuccessfulTrainingObservation",
    "compile_packed_dit_blocks",
    "require_checkpoint_cadence_binding",
    "require_distributed_forward_module",
    "require_single_gpu_checkpoint_binding",
    "require_single_gpu_checkpoint_compatibility",
    "require_single_gpu_config",
    "run_single_gpu_training",
]
