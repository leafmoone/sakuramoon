"""Trainable composite boundary and strict single-GPU sample-mean updates."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch import nn

from sakuramoon.conditioning.condition_tokens import (
    ConditionTokenEncoder,
    ConditionTokenOutput,
)
from sakuramoon.conditioning.text_mixer import TextConditioner, TextConditioningOutput
from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.model.growth import new_slot_fqn_prefixes
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.optim.clip import ClipResult, clip_grad_norm_fp32
from sakuramoon.telemetry.timers import PhaseTimer


@contextmanager
def _record_phase(timer: PhaseTimer | None, phase: str) -> Generator[None]:
    if timer is None:
        yield
        return
    with timer.record(phase):
        yield


def _complete_device_work(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_grad_norm(
    module: nn.Module,
    *,
    predicate: Callable[[str, nn.Parameter], bool],
    device: torch.device,
) -> torch.Tensor:
    squared_norm = torch.zeros((), device=device, dtype=torch.float32)
    found = False
    for name, parameter in module.named_parameters():
        if not predicate(name, parameter) or parameter.grad is None:
            continue
        if parameter.grad.is_sparse:
            raise RuntimeError("sparse diagnostic gradients are unsupported")
        squared_norm.add_(parameter.grad.float().square().sum())
        found = True
    if not found:
        return squared_norm
    return squared_norm.sqrt()


@dataclass(frozen=True, slots=True)
class TrainableCompositeIRepaOutput:
    """Phase-4 composite output when the iREPA projector is installed.

    ``predictions`` is exactly the legacy per-sample tuple; ``projected_student_features``
    is the projector's ``[B, T, 768]`` output computed INSIDE the (possibly
    DDP-wrapped) composite forward, so the projector participates in the
    distributed training graph.  It is consumed by the trainer for the cosine
    alignment loss; it is not part of the checkpoint contract.
    """

    predictions: tuple[torch.Tensor, ...]
    projected_student_features: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainableCompositeInputs:
    qwen_states: torch.Tensor
    main_token_indices: torch.Tensor
    main_mask: torch.Tensor
    main_token_lengths: tuple[int, ...]
    condition_token_indices: torch.Tensor
    condition_mask: torch.Tensor
    use_null_condition: torch.Tensor
    active_condition_sample_indices: torch.Tensor
    latents: tuple[torch.Tensor, ...]
    image_coordinates: tuple[torch.Tensor, ...]
    timestep: torch.Tensor
    size_scale: torch.Tensor
    aspect: torch.Tensor
    growth_alpha: float


class TrainableComposite(nn.Module):
    """The complete trainable module boundary; frozen encoders stay outside."""

    def __init__(
        self,
        *,
        dit: DenseDiT | PackedDiT,
        text: TextConditioner,
        condition_tokens: ConditionTokenEncoder,
        irepa_alignment: IRepaAlignment | None = None,
        irepa_tap_slot_id: int | None = None,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.text = text
        self.condition_tokens = condition_tokens
        # None stays a plain attribute (invisible to named_children /
        # named_parameters / state_dict), keeping the legacy v3 contract
        # byte-for-byte.  A real IRepaAlignment becomes the canonical
        # trainable child ``irepa_alignment.*``.
        self.irepa_alignment = irepa_alignment
        # The stable slot id captured for iREPA (config-locked to 8 in
        # production).  Plain int attribute: it is a runtime binding, not a
        # module, so it never enters named_children / state_dict.  The
        # projector may be constructed artifact-first without a bound tap
        # (the v4 artifact document carries no tap); bind_irepa_tap_slot
        # supplies it, and forward fails closed until then.
        if irepa_tap_slot_id is not None:
            if irepa_alignment is None:
                raise ValueError(
                    "irepa_tap_slot_id requires an installed irepa_alignment"
                )
            if type(irepa_tap_slot_id) is not int:
                raise ValueError("irepa_tap_slot_id must be an int")
            if irepa_tap_slot_id not in dit.active_slot_ids:
                raise ValueError(
                    "irepa_tap_slot_id is not an active stable slot at the "
                    f"current depth ({irepa_tap_slot_id} not in "
                    f"{dit.active_slot_ids})"
                )
        self.irepa_tap_slot_id = irepa_tap_slot_id

    def bind_irepa_tap_slot(self, slot_id: int) -> None:
        """Bind the config-locked capture slot after artifact construction.

        The v4 artifact document is tap-free (the tap is a runtime binding,
        not a checkpointable parameter), so artifact-first construction
        leaves the slot unbound; the config-bound assembly binds it here.
        Idempotent for the same slot, explicit for any other slot.
        """

        if self.irepa_alignment is None:
            raise ValueError(
                "binding an iREPA tap slot requires an installed projector"
            )
        if type(slot_id) is not int:
            raise ValueError("irepa tap slot must be an int")
        if slot_id not in self.dit.active_slot_ids:
            raise ValueError(
                "irepa tap slot is not an active stable slot at the current "
                f"depth ({slot_id} not in {self.dit.active_slot_ids})"
            )
        self.irepa_tap_slot_id = slot_id

    def forward_conditioning(
        self, inputs: TrainableCompositeInputs
    ) -> tuple[TextConditioningOutput, ConditionTokenOutput]:
        text = self.text(
            inputs.qwen_states,
            inputs.main_token_indices,
            inputs.main_mask,
        )
        condition = self.condition_tokens(
            inputs.qwen_states,
            inputs.condition_token_indices,
            inputs.condition_mask,
            inputs.use_null_condition,
            inputs.active_condition_sample_indices,
        )
        return text, condition

    def forward_dit(
        self,
        inputs: TrainableCompositeInputs,
        conditioning: tuple[TextConditioningOutput, ConditionTokenOutput],
    ) -> tuple[torch.Tensor, ...]:
        text, condition = conditioning
        if type(self.dit) is DenseDiT:
            dense_predictions = self.dit(
                torch.stack(inputs.latents),
                text.tokens,
                text.mask,
                condition.tokens,
                condition.active_mask,
                inputs.timestep,
                inputs.size_scale,
                inputs.aspect,
                image_coordinates=torch.stack(inputs.image_coordinates),
                growth_alpha=inputs.growth_alpha,
            )
            return tuple(dense_predictions.unbind(0))
        return self.dit(
            inputs.latents,
            text.tokens,
            text.mask,
            inputs.main_token_lengths,
            condition.tokens,
            condition.active_mask,
            inputs.timestep,
            inputs.size_scale,
            inputs.aspect,
            image_coordinates=inputs.image_coordinates,
            growth_alpha=inputs.growth_alpha,
        )

    def forward_dit_tapped(
        self,
        inputs: TrainableCompositeInputs,
        conditioning: tuple[TextConditioningOutput, ConditionTokenOutput],
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        text, condition = conditioning
        tap_slot_id = self.irepa_tap_slot_id
        if tap_slot_id is None:
            raise RuntimeError("tapped forward requires a bound irepa tap slot")
        if type(self.dit) is DenseDiT:
            dense_predictions, capture = self.dit.forward_tapped(
                torch.stack(inputs.latents),
                text.tokens,
                text.mask,
                condition.tokens,
                condition.active_mask,
                inputs.timestep,
                inputs.size_scale,
                inputs.aspect,
                image_coordinates=torch.stack(inputs.image_coordinates),
                growth_alpha=inputs.growth_alpha,
                tap_slot_id=tap_slot_id,
            )
            return tuple(dense_predictions.unbind(0)), capture
        packed_dit = self.dit
        if type(packed_dit) is not PackedDiT:
            raise TypeError("tapped forward requires a DenseDiT or PackedDiT")
        packed = packed_dit.prepare_packed_sequences(
            inputs.latents,
            text.tokens,
            text.mask,
            inputs.main_token_lengths,
            condition.tokens,
        )
        return packed_dit.forward_packed_tapped(
            packed,
            condition.tokens,
            condition.active_mask,
            inputs.timestep,
            inputs.size_scale,
            inputs.aspect,
            image_coordinates=inputs.image_coordinates,
            growth_alpha=inputs.growth_alpha,
            tap_slot_id=tap_slot_id,
        )

    def forward(
        self,
        inputs: TrainableCompositeInputs,
        *,
        phase_timer: PhaseTimer | None = None,
    ) -> tuple[torch.Tensor, ...] | TrainableCompositeIRepaOutput:
        with _record_phase(phase_timer, "conditioning"):
            conditioning = self.forward_conditioning(inputs)
        if self.irepa_alignment is None:
            # Legacy contract: iREPA absent or disabled — identical return
            # type and computation to before Phase 4.
            with _record_phase(phase_timer, "dit_forward"):
                return self.forward_dit(inputs, conditioning)
        with _record_phase(phase_timer, "dit_forward"):
            predictions, capture = self.forward_dit_tapped(inputs, conditioning)
        with _record_phase(phase_timer, "irepa_projector"):
            projected = self._project_student_capture(inputs, capture)
        return TrainableCompositeIRepaOutput(
            predictions=predictions,
            projected_student_features=projected,
        )

    def _project_student_capture(
        self,
        inputs: TrainableCompositeInputs,
        capture: torch.Tensor,
    ) -> torch.Tensor:
        """Project the stable-slot capture onto the teacher grid.

        Dense capture arrives batched as ``[B, T, D]``; the packed capture is
        flat ``[total_image_tokens, D]`` in per-sample row-major order and is
        stacked only after verifying a homogeneous image grid (per-sample
        ``T_i == H_i * W_i`` and one shared grid for the whole batch).  No
        padding and no feature interpolation: a non-homogeneous batch is a
        Phase-4 contract violation and fails closed.
        """

        grids = tuple(latent.shape[-2:] for latent in inputs.latents)
        batch = len(inputs.latents)
        if batch == 0:
            raise ValueError("iREPA training batch must not be empty")
        if any(grid != grids[0] for grid in grids):
            raise ValueError(
                "iREPA requires a homogeneous image grid across the training "
                "batch, got "
                f"{grids}"
            )
        grid_height, grid_width = grids[0]
        if capture.ndim == 3:
            if capture.shape[0] != batch:
                raise ValueError("dense iREPA capture batch mismatch")
            image_hidden = capture
        else:
            if capture.ndim != 2:
                raise ValueError(
                    "packed iREPA capture must be flat [T_total, D]"
                )
            image_tokens = grid_height * grid_width
            if capture.shape[0] != image_tokens * batch:
                raise ValueError(
                    "packed iREPA capture token count does not match the "
                    "image grid: "
                    f"{capture.shape[0]} != {image_tokens * batch}"
                )
            image_hidden = capture.reshape(
                batch, image_tokens, capture.shape[1]
            )
        alignment = self.irepa_alignment
        if alignment is None:
            raise RuntimeError("iREPA capture arrived without a projector")
        return alignment(image_hidden, (grid_height, grid_width))


@dataclass(frozen=True, slots=True)
class SingleGpuUpdateState:
    attempted_updates: int
    successful_updates: int
    effective_samples: int

    @classmethod
    def initial(cls) -> SingleGpuUpdateState:
        return cls(attempted_updates=0, successful_updates=0, effective_samples=0)

    def __post_init__(self) -> None:
        if (
            self.attempted_updates < 0
            or self.successful_updates < 0
            or self.successful_updates > self.attempted_updates
            or self.effective_samples < 0
        ):
            raise ValueError("single-GPU update counters are inconsistent")


@dataclass(frozen=True, slots=True)
class SingleGpuUpdateResult:
    mean_loss: torch.Tensor
    clip: ClipResult
    condition_encoder_grad_norm: torch.Tensor
    condition_global_projection_grad_norm: torch.Tensor
    microbatches: int
    effective_samples: int
    state: SingleGpuUpdateState
    growth_alpha: float = 1.0
    growth_new_block_grad_norm: torch.Tensor | None = None
    growth_new_conditioner_grad_norm: torch.Tensor | None = None
    growth_new_slot_grad_norm: torch.Tensor | None = None
    # Scaled pre-clip grad norm over ``irepa_alignment.*`` (0.0 when the
    # projector is absent, disabled, or produced no gradient — at lambda=0
    # the projector grad is an exact zero tensor).
    irepa_projector_grad_norm: float = 0.0


class StepOptimizer(Protocol):
    def step(self) -> None: ...

    def zero_grad(self, *, set_to_none: bool) -> None: ...


def _note_forensic_update(
    optimizer: StepOptimizer, state: SingleGpuUpdateState
) -> None:
    """F2 telemetry hook: optimizers that implement ``note_forensic_update``
    learn the global update identity immediately before the optimizer
    step, so the minimal hard-fail capsule can record
    (last_successful_update, attempted_update). Duck-typed: a no-op for
    every optimizer that does not implement it (the production AdamW
    paths are untouched). Pure attribute storage — it never feeds any
    computation."""
    note = getattr(optimizer, "note_forensic_update", None)
    if callable(note):
        note(
            last_successful_update=state.successful_updates,
            attempted_update=state.attempted_updates,
        )


class SingleGpuStep:
    """Accumulate per-sample sums, then normalize once at the update boundary."""

    def __init__(
        self,
        module: nn.Module,
        optimizer: StepOptimizer,
        *,
        accumulation_steps: int,
        state: SingleGpuUpdateState,
        effective_sample_multiplier: int = 1,
        growth_alpha: float = 1.0,
        backward: Callable[[torch.Tensor], None] | None = None,
        irepa_projector: bool = False,
    ) -> None:
        if type(accumulation_steps) is not int or accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        if not any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("training module has no trainable parameters")
        if (
            type(effective_sample_multiplier) is not int
            or effective_sample_multiplier <= 0
            or (backward is not None and not callable(backward))
        ):
            raise ValueError("distributed step controls are invalid")
        if type(growth_alpha) is not float or not 0.0 <= growth_alpha <= 1.0:
            raise ValueError("growth_alpha must be a float in [0,1]")
        if type(irepa_projector) is not bool:
            raise ValueError("irepa_projector must be a bool")
        self.module = module
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.effective_sample_multiplier = effective_sample_multiplier
        self.growth_alpha = growth_alpha
        self.irepa_projector = irepa_projector
        self._backward = backward
        self._state: SingleGpuUpdateState = state
        self._microbatches = 0
        self._samples = 0
        self._loss_sum: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._nonfinite_seen: torch.Tensor | None = None
        self._failed = False
        self._detection_phase: str | None = None

    @property
    def state(self) -> SingleGpuUpdateState:
        return self._state

    @property
    def pending_samples(self) -> int:
        return self._samples

    @property
    def pending_microbatches(self) -> int:
        return self._microbatches

    @property
    def detection_phase(self) -> str | None:
        """Return the host boundary that observed failure, not a proven kernel origin."""

        return self._detection_phase

    def backward(self, per_sample_loss: torch.Tensor) -> None:
        if self._failed:
            raise RuntimeError("failed update state cannot continue")
        if self._microbatches >= self.accumulation_steps:
            raise RuntimeError("received more microbatches than configured")
        if per_sample_loss.ndim != 1 or per_sample_loss.numel() == 0:
            raise ValueError("per_sample_loss must be a nonempty one-dimensional tensor")
        if per_sample_loss.dtype != torch.float32:
            raise TypeError("per_sample_loss must use float32")
        if not per_sample_loss.requires_grad:
            raise ValueError("per_sample_loss must retain a gradient graph")
        if self._device is not None and per_sample_loss.device != self._device:
            raise ValueError("all accumulated losses must share one device")
        # Device-side finiteness flag: accumulate on-device so backward stays
        # sync-free. finish_update syncs the flag once at the update boundary
        # and aborts the whole update if any microbatch loss was nonfinite.
        flag = torch.logical_not(torch.isfinite(per_sample_loss).all())
        if self._nonfinite_seen is None:
            self._nonfinite_seen = flag
        else:
            self._nonfinite_seen = torch.logical_or(self._nonfinite_seen, flag)

        loss_sum = per_sample_loss.sum()
        try:
            if self._backward is None:
                loss_sum.backward()  # pyright: ignore[reportUnknownMemberType]
            else:
                self._backward(loss_sum)
        except BaseException as error:
            self._detection_phase = "backward"
            self._abort_preserving(error)
            raise
        self._loss_sum = (
            loss_sum.detach()
            if self._loss_sum is None
            else self._loss_sum + loss_sum.detach()
        )
        self._device = per_sample_loss.device
        self._microbatches += 1
        self._samples += per_sample_loss.numel()

    def _abort_pending_update(self) -> None:
        if self._failed:
            return
        self._state = replace(
            self._state,
            attempted_updates=self._state.attempted_updates + 1,
        )
        self._failed = True
        self.optimizer.zero_grad(set_to_none=True)

    def _abort_preserving(self, error: BaseException) -> None:
        try:
            self._abort_pending_update()
        except BaseException as cleanup_error:  # noqa: BLE001
            raise BaseExceptionGroup(
                "training failure and gradient cleanup failure",
                [error, cleanup_error],
            ) from None

    def abort(self) -> None:
        """Poison an interrupted update and discard every pending gradient."""

        self._abort_pending_update()

    def finish_update(
        self,
        *,
        phase_timer: PhaseTimer | None = None,
    ) -> SingleGpuUpdateResult:
        if self._failed:
            raise RuntimeError("failed update state cannot continue")
        if self._microbatches != self.accumulation_steps or self._loss_sum is None:
            raise RuntimeError("update requires exactly the configured microbatch count")
        attempted = replace(
            self._state,
            attempted_updates=self._state.attempted_updates + 1,
        )
        self._state = attempted
        if self._nonfinite_seen is not None and bool(self._nonfinite_seen.item()):
            # Deferred detection boundary: the flag records that a backward
            # loss was nonfinite; the single sync happens here, once per
            # update. detection_phase stays "backward" because the failure
            # originates in the per-sample losses, not in finalize.
            error = FloatingPointError("per-sample loss is nonfinite")
            self._detection_phase = "backward"
            self._failed = True
            try:
                with _record_phase(phase_timer, "zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "update failed and gradient cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise error
        parameters = tuple(
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        )
        try:
            with _record_phase(phase_timer, "clip"):
                gradient_scale = 1.0 / self._samples
                gradients = [
                    parameter.grad
                    for parameter in parameters
                    if parameter.grad is not None
                ]
                if gradients:
                    torch._foreach_mul_(gradients, gradient_scale)
                condition_encoder_grad_norm = _parameter_grad_norm(
                    self.module,
                    predicate=lambda name, _parameter: (
                        "condition_tokens" in name.split(".")
                    ),
                    device=self._device,
                )
                condition_global_projection_grad_norm = _parameter_grad_norm(
                    self.module,
                    predicate=lambda name, _parameter: name.endswith(
                        "dit.conditioner.condition_global_projection.weight"
                    ),
                    device=self._device,
                )
                try:
                    prefixes = new_slot_fqn_prefixes(self.module.dit.depth)
                except (AttributeError, ValueError):
                    prefixes = ()
                block_prefixes = tuple(
                    prefix for prefix in prefixes if prefix.startswith("dit.blocks.")
                )
                conditioner_names = tuple(
                    prefix
                    for prefix in prefixes
                    if prefix.startswith("dit.conditioner.")
                )
                growth_new_block_grad_norm = _parameter_grad_norm(
                    self.module,
                    predicate=lambda name, _parameter: name.startswith(block_prefixes),
                    device=self._device,
                )
                growth_new_conditioner_grad_norm = _parameter_grad_norm(
                    self.module,
                    predicate=lambda name, _parameter: name in conditioner_names,
                    device=self._device,
                )
                growth_new_slot_grad_norm = torch.sqrt(
                    growth_new_block_grad_norm.square()
                    + growth_new_conditioner_grad_norm.square()
                )
                # At this point at least one microbatch backward has run, so
                # the device bound during backward is non-None.  Narrow once
                # so the iREPA diagnostic does not widen the error surface.
                irepa_device = self._device
                if irepa_device is None:
                    raise RuntimeError(
                        "iREPA projector grad norm requires a bound device"
                    )
                irepa_projector_grad_norm = (
                    float(
                        _parameter_grad_norm(
                            self.module,
                            predicate=lambda name, _parameter: (
                                name.startswith("irepa_alignment.")
                            ),
                            device=irepa_device,
                        ).item()
                    )
                    if self.irepa_projector
                    else 0.0
                )
                clip = clip_grad_norm_fp32(parameters, max_norm=1.0)
        except BaseException as error:
            self._detection_phase = "clip"
            self._failed = True
            try:
                with _record_phase(phase_timer, "zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "update failed and gradient cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise

        try:
            with _record_phase(phase_timer, "optimizer"):
                _note_forensic_update(self.optimizer, attempted)
                self.optimizer.step()
        except BaseException as error:
            self._detection_phase = "optimizer"
            self._failed = True
            try:
                with _record_phase(phase_timer, "zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "update failed and gradient cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise

        assert self._device is not None
        try:
            _complete_device_work(self._device)
        except BaseException as error:
            self._detection_phase = "device_completion"
            self._failed = True
            try:
                with _record_phase(phase_timer, "zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "device completion failed and gradient cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise

        successful = replace(
            attempted,
            successful_updates=attempted.successful_updates + 1,
            effective_samples=(
                attempted.effective_samples
                + self._samples * self.effective_sample_multiplier
            ),
        )
        self._state = successful
        try:
            with _record_phase(phase_timer, "zero_grad"):
                self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self._detection_phase = "zero_grad"
            self._failed = True
            raise

        result = SingleGpuUpdateResult(
            mean_loss=self._loss_sum / self._samples,
            clip=clip,
            condition_encoder_grad_norm=condition_encoder_grad_norm,
            condition_global_projection_grad_norm=(
                condition_global_projection_grad_norm
            ),
            microbatches=self._microbatches,
            effective_samples=self._samples,
            state=successful,
            growth_alpha=self.growth_alpha,
            growth_new_block_grad_norm=growth_new_block_grad_norm,
            growth_new_conditioner_grad_norm=growth_new_conditioner_grad_norm,
            growth_new_slot_grad_norm=growth_new_slot_grad_norm,
            irepa_projector_grad_norm=irepa_projector_grad_norm,
        )
        self._state = result.state
        self._microbatches = 0
        self._samples = 0
        self._loss_sum = None
        self._device = None
        self._nonfinite_seen = None
        return result


__all__ = [
    "SingleGpuStep",
    "SingleGpuUpdateResult",
    "SingleGpuUpdateState",
    "StepOptimizer",
    "TrainableComposite",
    "TrainableCompositeIRepaOutput",
    "TrainableCompositeInputs",
]
