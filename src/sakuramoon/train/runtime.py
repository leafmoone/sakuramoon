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
from sakuramoon.data.camera_viewport import (
    CameraMirrorCounts,
    CameraViewportCounts,
    camera_zoom_band,
)
from sakuramoon.data.caption import (
    CaptionDropoutCounts,
    CaptionPlan,
    ConditionRouteCounts,
)
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.pipeline import ImageAudit
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    require_accepted_production_batch_stream,
)
from sakuramoon.data.serialize import SerializedCaption
from sakuramoon.data.spatial_crop import SpatialCropCounts
from sakuramoon.data.transparent_white import TransparentWhiteCounts
from sakuramoon.encoders.mage_vae import FrozenMageVAE
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
class MirrorPhysicalLayout:
    """Pure CPU index layout expanding one logical batch into physical views.

    Every logical row keeps its original view in place; a mirror pair
    inserts its mirror view in the physical row immediately after its
    original view. No logical row is ever reordered, dropped, or rescaled,
    so a batch without mirror pairs takes the exact pre-mirror path.
    """

    logical_count: int
    physical_count: int
    # phys_to_log[p]: the logical row that produced physical row p.
    phys_to_log: tuple[int, ...]
    # logical_orig_phys[i]: the physical row of logical row i's original view.
    logical_orig_phys: tuple[int, ...]
    # Per mirror pair (in logical row order): the physical row of the pair's
    # original view and of its mirror view.
    pair_orig_phys: tuple[int, ...]
    mirror_phys: tuple[int, ...]
    # physical_source_order[p]: the row of the concatenation
    # [logical views; mirror views] that fills physical row p.
    physical_source_order: tuple[int, ...]

    def __post_init__(self) -> None:
        pair_count = len(self.pair_orig_phys)
        if (
            type(self.logical_count) is not int
            or self.logical_count <= 0
            or type(self.physical_count) is not int
            or self.physical_count < self.logical_count
            or len(self.phys_to_log) != self.physical_count
            or len(self.logical_orig_phys) != self.logical_count
            or len(self.pair_orig_phys) != len(self.mirror_phys)
            or len(self.physical_source_order) != self.physical_count
        ):
            raise ValueError("mirror physical layout has inconsistent counts")
        if self.physical_count != self.logical_count + pair_count:
            raise ValueError(
                "mirror physical layout row count differs from logical + pairs"
            )
        source_bound = self.logical_count + pair_count
        for value in self.physical_source_order:
            if type(value) is not int or not 0 <= value < source_bound:
                raise ValueError("mirror physical source order is invalid")
        if sorted(self.physical_source_order) != list(range(source_bound)):
            raise ValueError(
                "mirror physical source order must cover every view exactly once"
            )
        previous = -1
        for logical in self.phys_to_log:
            if type(logical) is not int or not 0 <= logical < self.logical_count:
                raise ValueError("mirror physical layout maps to an invalid row")
            if logical < previous:
                raise ValueError(
                    "mirror physical layout must keep ascending logical order"
                )
            previous = logical
        for logical, orig in enumerate(self.logical_orig_phys):
            if type(orig) is not int or not 0 <= orig < self.physical_count:
                raise ValueError("mirror original view points to an invalid row")
            if self.phys_to_log[orig] != logical:
                raise ValueError(
                    "mirror original view points at the wrong logical row"
                )
        for orig, mirror in zip(self.pair_orig_phys, self.mirror_phys):
            if (
                type(orig) is not int
                or type(mirror) is not int
                or not 0 <= orig < self.physical_count - 1
                or mirror != orig + 1
                or self.phys_to_log[orig] != self.phys_to_log[mirror]
            ):
                raise ValueError(
                    "mirror view must immediately follow its original view"
                )
            logical = self.phys_to_log[mirror]
            if self.logical_orig_phys[logical] != orig:
                raise ValueError(
                    "mirror view original pointer differs from the layout"
                )

    @property
    def pair_logical_rows(self) -> tuple[int, ...]:
        """Logical rows that are mirror pairs, in logical row order."""

        return tuple(self.phys_to_log[mirror] for mirror in self.mirror_phys)

    @classmethod
    def build(cls, has_mirror: tuple[bool, ...]) -> MirrorPhysicalLayout:
        """Build the layout for one logical batch; mirrors follow originals."""

        phys_to_log: list[int] = []
        logical_orig_phys: list[int] = []
        pair_orig_phys: list[int] = []
        mirror_phys: list[int] = []
        physical_source_order: list[int] = []
        for logical, mirrored in enumerate(has_mirror):
            if type(mirrored) is not bool:
                raise ValueError("mirror layout flags must be boolean")
            logical_orig_phys.append(len(phys_to_log))
            phys_to_log.append(logical)
            physical_source_order.append(logical)
            if mirrored:
                pair_orig_phys.append(len(phys_to_log) - 1)
                mirror_phys.append(len(phys_to_log))
                phys_to_log.append(logical)
                physical_source_order.append(
                    len(has_mirror) + len(mirror_phys) - 1
                )
        return cls(
            logical_count=len(has_mirror),
            physical_count=len(phys_to_log),
            phys_to_log=tuple(phys_to_log),
            logical_orig_phys=tuple(logical_orig_phys),
            pair_orig_phys=tuple(pair_orig_phys),
            mirror_phys=tuple(mirror_phys),
            physical_source_order=tuple(physical_source_order),
        )


def reduce_mirror_pair_loss(
    per_physical: torch.Tensor,
    layout: MirrorPhysicalLayout,
) -> torch.Tensor:
    """Reduce one physical per-view loss to one logical per-sample loss.

    An ordinary logical row keeps ``L_original``; a mirror logical row
    becomes ``0.5 * L_original + 0.5 * L_mirror``, so one mirror pair
    represents exactly one logical source sample (total source weight 1.0)
    and the existing logical-sample reduction keeps its meaning.
    """

    if per_physical.ndim != 1:
        raise ValueError(
            "mirror pair reduction requires a one-dimensional loss"
        )
    if per_physical.numel() != layout.physical_count:
        raise ValueError(
            "mirror pair loss length differs from the physical layout"
        )
    device = per_physical.device
    logical = per_physical.index_select(
        0,
        torch.as_tensor(
            layout.logical_orig_phys, dtype=torch.long, device=device
        ),
    )
    if not layout.mirror_phys:
        return logical
    pair_logical = torch.as_tensor(
        layout.pair_logical_rows, dtype=torch.long, device=device
    )
    pair_loss = (
        0.5 * logical.index_select(0, pair_logical)
        + 0.5
        * per_physical.index_select(
            0,
            torch.as_tensor(
                layout.mirror_phys, dtype=torch.long, device=device
            ),
        )
    )
    return logical.index_copy(0, pair_logical, pair_loss)


@dataclass(frozen=True, slots=True)
class PreparedTrainingBatch:
    """The immutable tensors needed by one objective evaluation."""

    inputs: TrainableCompositeInputs
    clean_latents: tuple[torch.Tensor, ...]
    states: tuple[torch.Tensor, ...]
    # Mirror-pair physical layout for this batch; None when the batch has no
    # mirror pairs (the exact pre-mirror path).
    mirror_layout: MirrorPhysicalLayout | None = None
    # Logical (per-source-sample) timesteps; None without mirror pairs, where
    # inputs.timestep is already the logical vector.
    timestep_logical: torch.Tensor | None = None


def _camera_zoom_band_index(audit: ImageAudit) -> int:
    """Fixed camera zoom-band index for one sample; -1 when not applied."""

    if not audit.camera_applied:
        return -1
    return camera_zoom_band(audit.camera_equivalent_zoom)


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
    camera_viewport: CameraViewportCounts
    # Per-sample camera zoom-band index in fixed CAMERA_ZOOM_BAND_LABELS
    # order; -1 marks a sample without an applied camera viewport. The
    # length must equal per_sample_loss.numel().
    camera_zoom_bands: tuple[int, ...]
    transparent: TransparentWhiteCounts
    # Fixed camera-mirror counters for the batch (None = pre-mirror batch);
    # they carry the logical-vs-physical view distinction for telemetry.
    camera_mirror: CameraMirrorCounts | None = None

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
            camera_viewport=self.camera_viewport,
            camera_zoom_bands=self.camera_zoom_bands,
            transparent=self.transparent,
            camera_mirror=self.camera_mirror,
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


def _reduce_mirror_observation(
    physical: _RuntimeLoss,
    layout: MirrorPhysicalLayout,
    timestep_logical: torch.Tensor,
    noise_observation_boundary: float,
) -> _RuntimeLoss:
    """Move one physical per-view observation to the logical domain.

    The per-sample loss uses the pair weighting contract (0.5 * L_original
    + 0.5 * L_mirror per logical pair). The high/low noise buckets are
    re-bucketed on the LOGICAL timesteps — a pair shares one timestep, so
    both views of a pair fall into the same bucket — keeping
    high + low == total loss in the logical-sample domain.
    """

    if timestep_logical.numel() != layout.logical_count:
        raise ValueError(
            "logical timestep count differs from the mirror layout"
        )
    per_logical = reduce_mirror_pair_loss(physical.per_sample, layout)
    high = timestep_logical < noise_observation_boundary
    low = ~high
    return _RuntimeLoss(
        per_logical,
        (per_logical * high).sum(),
        high.sum(),
        (per_logical * low).sum(),
        low.sum(),
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
    batch: TrainingBatch,
    *,
    device: torch.device,
    count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    height = float(batch.target_height)
    width = float(batch.target_width)
    size_scale = 0.5 * math.log2((height * width) / float(512 * 512))
    aspect = math.log2(width / height)
    if count is None:
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


def _mirror_view_images(
    batch: TrainingBatch, *, device: torch.device
) -> torch.Tensor | None:
    """Stack the batch's mirror views on-device, normalized like originals.

    Returns None when the batch carries no mirror payloads; otherwise a
    [P,3,H,W] bfloat16 tensor (P = pair count) produced from the uint8
    payloads with the exact ordinary training normalization
    (uint8 -> bf16 -> /127.5 -> -1).
    """

    payloads = tuple(
        payload for payload in batch.mirror if payload is not None
    )
    if not payloads:
        return None
    stacked = torch.stack(tuple(payload.mirror_image for payload in payloads))
    if stacked.ndim != 4 or stacked.shape[1] != 3:
        raise ValueError("mirror view images must be [3,H,W] uint8 tensors")
    return (
        stacked.to(device, dtype=torch.bfloat16, non_blocking=True)
        .div(127.5)
        .sub(1.0)
    )


def _physical_view_images(
    images: torch.Tensor,
    mirror_images: torch.Tensor,
    layout: MirrorPhysicalLayout,
) -> torch.Tensor:
    """Interleave original and mirror views into one physical batch."""

    if images.ndim != 4 or images.shape[0] != layout.logical_count:
        raise ValueError("logical view count differs from the mirror layout")
    if mirror_images.shape[0] != len(layout.mirror_phys):
        raise ValueError("mirror view count differs from the mirror layout")
    if mirror_images.shape[1:] != images.shape[1:]:
        raise ValueError("mirror views must match the logical view shape")
    source = torch.cat((images, mirror_images), dim=0)
    return source.index_select(
        0,
        torch.as_tensor(
            layout.physical_source_order,
            dtype=torch.long,
            device=images.device,
        ),
    )


def _mirror_view_coordinate_maps(
    batch: TrainingBatch,
    logical_maps: tuple[torch.Tensor, ...],
    layout: MirrorPhysicalLayout,
    *,
    token_height: int,
    token_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Physical-order full-canvas coordinate maps including mirror views.

    Original views reuse the logical maps (existing production camera
    coordinates); mirror views map their mirror crop box inside the SAME
    full canvas of their logical sample through the unchanged
    ``full_canvas_crop_coordinates`` math — no new position encoding,
    camera embedding, direction token, or TOP/BOTTOM label.
    """

    if len(logical_maps) != layout.logical_count:
        raise ValueError(
            "logical coordinate map count differs from the mirror layout"
        )
    mirror_maps: dict[int, torch.Tensor] = {}
    for logical in layout.pair_logical_rows:
        payload = batch.mirror[logical]
        if payload is None:
            raise ValueError("mirror layout pair row has no mirror payload")
        audit = batch.audits[logical]
        try:
            mirror_maps[logical] = full_canvas_crop_coordinates(
                token_height,
                token_width,
                full_height=audit.resized_height,
                full_width=audit.resized_width,
                crop_box=payload.crop_box,
                device=device,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"mirror view {logical} cannot define full-canvas coordinates"
            ) from error
    mirror_at = set(layout.mirror_phys)
    return tuple(
        mirror_maps[logical] if phys in mirror_at else logical_maps[logical]
        for phys, logical in enumerate(layout.phys_to_log)
    )


def _physical_active_condition_indices(
    active: torch.Tensor, layout: MirrorPhysicalLayout
) -> torch.Tensor:
    """Remap logical active-condition row indices to physical rows.

    A mirror pair shares its caption and condition routing, so both views
    of a pair are active exactly when the logical row is active. The result
    is strictly ascending physical row order (the collate convention).
    """

    if active.ndim != 1 or active.dtype != torch.long:
        raise ValueError("active condition indices must be a 1-D long tensor")
    active_mask = torch.zeros(
        layout.logical_count, dtype=torch.bool, device=active.device
    )
    active_mask.scatter_(0, active, True)
    physical_mask = active_mask[
        torch.as_tensor(
            layout.phys_to_log, dtype=torch.long, device=active.device
        )
    ]
    return torch.nonzero(physical_mask, as_tuple=True)[0]


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

    def set_growth_alpha(self, value: float) -> None:
        """Select the canonical alpha before starting one successful update."""

        if type(value) is not float or not 0.0 <= value <= 1.0:
            raise ValueError("growth_alpha must be a float in [0,1]")
        self.growth_alpha = value

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
        logical_count = batch.images.shape[0]
        if len(batch.mirror) not in (0, logical_count):
            raise ValueError(
                "mirror payload row count differs from the logical batch"
            )
        mirror_flags = (
            tuple(payload is not None for payload in batch.mirror)
            if batch.mirror
            else ()
        )
        mirror_layout = (
            MirrorPhysicalLayout.build(mirror_flags)
            if any(mirror_flags)
            else None
        )
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

        if mirror_layout is None:
            if phase_timer is None:
                clean = self.vae.encode(images)
            else:
                with phase_timer.record("vae"):
                    clean = self.vae.encode(images)
            row_count = logical_count
            clean_logical = clean
        else:
            # One VAE call over the physical views (original + mirror); the
            # source decode, canvas resize, caption and Qwen work already
            # happened once per logical sample.
            if phase_timer is None:
                mirror_views = _mirror_view_images(
                    batch, device=self.device
                )
                physical_images = _physical_view_images(
                    images, mirror_views, mirror_layout
                )
                clean = self.vae.encode(physical_images)
            else:
                with phase_timer.record("h2d"):
                    mirror_views = _mirror_view_images(
                        batch, device=self.device
                    )
                    physical_images = _physical_view_images(
                        images, mirror_views, mirror_layout
                    )
                with phase_timer.record("vae"):
                    clean = self.vae.encode(physical_images)
            row_count = mirror_layout.physical_count
            clean_logical = clean.index_select(
                0,
                torch.as_tensor(
                    mirror_layout.logical_orig_phys,
                    dtype=torch.long,
                    device=clean.device,
                ),
            )
        expected = (
            row_count,
            128,
            batch.target_height // 16,
            batch.target_width // 16,
        )
        if tuple(clean.shape) != expected or clean.dtype != torch.bfloat16:
            raise ValueError("Mage-VAE returned an unexpected latent contract")
        token_height = clean.shape[-2]
        token_width = clean.shape[-1]
        logical_coordinates = _full_canvas_coordinate_maps(
            batch,
            token_height=token_height,
            token_width=token_width,
            device=clean.device,
        )
        if mirror_layout is None:
            image_coordinates = logical_coordinates
        else:
            image_coordinates = _mirror_view_coordinate_maps(
                batch,
                logical_coordinates,
                mirror_layout,
                token_height=token_height,
                token_width=token_width,
                device=clean.device,
            )

        # Timesteps and noise are drawn ONCE per logical sample (the exact
        # pre-mirror RNG consumption) and expanded to the physical views, so
        # both views of a pair share the same timestep and the same noise
        # tensor while logical samples stay independent.
        timestep_logical = sample_jlt_timesteps(
            logical_count,
            p_mean=self.p_mean,
            p_std=self.p_std,
            device=self.device,
            generator=self.generator,
        )
        noise_logical = sample_noise(
            clean_logical,
            noise_scale=self.noise_scale,
            generator=self.generator,
        )
        if mirror_layout is None:
            timestep = timestep_logical
            noise = noise_logical
            qwen_states = hidden_states
            main_token_lengths = batch.main_token_lengths
        else:
            expand = torch.as_tensor(
                mirror_layout.phys_to_log,
                dtype=torch.long,
                device=self.device,
            )
            timestep = timestep_logical.index_select(0, expand)
            noise = noise_logical.index_select(0, expand)
            # Conditioning is row-wise and pair-shared: repeating the logical
            # rows (one Qwen forward per logical sample) is numerically
            # identical to re-running the encoders for the mirror views.
            qwen_states = hidden_states.index_select(0, expand)
            main_token_indices = main_token_indices.index_select(0, expand)
            main_mask = main_mask.index_select(0, expand)
            main_token_lengths = tuple(
                batch.main_token_lengths[logical]
                for logical in mirror_layout.phys_to_log
            )
            condition_token_indices = condition_token_indices.index_select(
                0, expand
            )
            condition_mask = condition_mask.index_select(0, expand)
            use_null_condition = use_null_condition.index_select(0, expand)
            # The device-local tensor from the H2D block is the input: the
            # batch attribute is still CPU-side, and the condition-token
            # path below requires the remapped indices on the same device
            # as qwen_states.
            active_condition_sample_indices = (
                _physical_active_condition_indices(
                    active_condition_sample_indices, mirror_layout
                )
            )
        # Fail closed on a device/dtype/ndim mismatch instead of letting a
        # CPU index tensor reach the condition-token path, where the
        # failure would surface as an opaque downstream device error.
        if mirror_layout is not None and (
            active_condition_sample_indices.device != qwen_states.device
            or active_condition_sample_indices.dtype != torch.long
            or active_condition_sample_indices.ndim != 1
        ):
            raise ValueError(
                "mirror batch active-condition indices must be a 1-D "
                "long tensor on the conditioning device"
            )
        state = interpolate_state(clean, noise, timestep)
        size_scale, aspect = _size_conditions(
            batch, device=self.device, count=row_count
        )
        inputs = TrainableCompositeInputs(
            qwen_states=qwen_states,
            main_token_indices=main_token_indices,
            main_mask=main_mask,
            main_token_lengths=main_token_lengths,
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
            mirror_layout=mirror_layout,
            timestep_logical=(
                timestep_logical if mirror_layout is not None else None
            ),
        )

    def measure(
        self, batch: TrainingBatch, *, phase_timer: PhaseTimer | None = None
    ) -> RuntimeMeasurement:
        prepared = self.prepare(batch, phase_timer=phase_timer)
        dit_flops = self.dit_flop_counter.count(prepared.inputs)
        predictions = self.forward_module(
            prepared.inputs,
            phase_timer=phase_timer,
        )
        if len(predictions) != len(prepared.clean_latents):
            raise ValueError("DiT prediction count differs from latent batch")
        if phase_timer is None:
            loss = self._loss(predictions, prepared)
        else:
            with phase_timer.record("loss"):
                loss = self._loss(predictions, prepared)
        # Compute facts (image tokens / DiT flops) count the physical views
        # the DiT actually processed; the logical/physical distinction is
        # carried by the camera_mirror counters.
        view_count = (
            prepared.mirror_layout.physical_count
            if prepared.mirror_layout is not None
            else batch.images.shape[0]
        )
        image_tokens = (
            view_count
            * (batch.target_height // 16)
            * (batch.target_width // 16)
        )
        text_tokens = sum(batch.main_token_lengths)
        sample_ids = tuple(
            str(cast(int, value.item()))
            for value in batch.sample_ids.detach().cpu().unbind()
        )
        shape_key = f"{batch.target_height}x{batch.target_width}x{batch.dense_length}"
        camera_bands = tuple(_camera_zoom_band_index(audit) for audit in batch.audits)
        if len(camera_bands) != loss.per_sample.numel():
            raise ValueError(
                "camera zoom band count differs from per-sample loss count"
            )
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
            # The observed timesteps stay in the logical-sample domain (one
            # value per source sample); mirror pairs share one timestep.
            timesteps=(
                prepared.timestep_logical
                if prepared.timestep_logical is not None
                else prepared.inputs.timestep
            ),
            dropout_hits=batch.dropout_hits,
            condition_routes=batch.condition_routes,
            captions=batch.captions,
            caption_plans=tuple(caption.plan for caption in batch.captions),
            spatial_crop=batch.spatial_crop,
            camera_viewport=batch.camera_viewport,
            camera_zoom_bands=camera_bands,
            transparent=batch.transparent,
            camera_mirror=batch.camera_mirror,
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
            physical = _RuntimeLoss(
                result.per_sample,
                result.high_noise_loss_sum,
                result.high_noise_sample_count,
                result.low_noise_loss_sum,
                result.low_noise_sample_count,
            )
        else:
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
            physical = _RuntimeLoss(
                torch.stack(values),
                torch.stack(high_loss).sum(),
                torch.stack(high_count).sum(),
                torch.stack(low_loss).sum(),
                torch.stack(low_count).sum(),
            )
        layout = prepared.mirror_layout
        if layout is None or not layout.mirror_phys:
            return physical
        if prepared.timestep_logical is None:
            raise ValueError("mirror batch is missing its logical timesteps")
        return _reduce_mirror_observation(
            physical,
            layout,
            prepared.timestep_logical,
            self.noise_observation_boundary,
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
