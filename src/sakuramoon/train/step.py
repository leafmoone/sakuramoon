"""Trainable composite boundary and strict single-GPU sample-mean updates."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch import nn

from sakuramoon.conditioning.style_resampler import (
    StyleConditioningOutput,
    StyleResampler,
)
from sakuramoon.conditioning.text_mixer import TextConditioner, TextConditioningOutput
from sakuramoon.model.dit import PackedDiT
from sakuramoon.optim.clip import ClipResult, clip_grad_norm_fp32
from sakuramoon.telemetry.timers import PhaseTimer


@contextmanager
def _record_phase(timer: PhaseTimer | None, phase: str) -> Generator[None]:
    if timer is None:
        yield
        return
    with timer.record(phase):
        yield


@dataclass(frozen=True, slots=True)
class TrainableCompositeInputs:
    qwen_states: torch.Tensor
    main_token_indices: torch.Tensor
    main_mask: torch.Tensor
    main_token_lengths: tuple[int, ...]
    artist_token_indices: torch.Tensor
    artist_mask: torch.Tensor
    use_null_style: torch.Tensor
    active_style_sample_indices: torch.Tensor
    latents: tuple[torch.Tensor, ...]
    timestep: torch.Tensor
    size_scale: torch.Tensor
    aspect: torch.Tensor
    growth_alpha: float


class TrainableComposite(nn.Module):
    """The complete trainable module boundary; frozen encoders stay outside."""

    def __init__(
        self,
        *,
        dit: PackedDiT,
        text: TextConditioner,
        style: StyleResampler,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.text = text
        self.style = style

    def forward_conditioning(
        self, inputs: TrainableCompositeInputs
    ) -> tuple[TextConditioningOutput, StyleConditioningOutput]:
        text = self.text(
            inputs.qwen_states,
            inputs.main_token_indices,
            inputs.main_mask,
        )
        style = self.style(
            inputs.qwen_states,
            inputs.artist_token_indices,
            inputs.artist_mask,
            inputs.use_null_style,
            inputs.active_style_sample_indices,
        )
        return text, style

    def forward_dit(
        self,
        inputs: TrainableCompositeInputs,
        conditioning: tuple[TextConditioningOutput, StyleConditioningOutput],
    ) -> tuple[torch.Tensor, ...]:
        text, style = conditioning
        return self.dit(
            inputs.latents,
            text.tokens,
            text.mask,
            inputs.main_token_lengths,
            style.tokens,
            inputs.timestep,
            inputs.size_scale,
            inputs.aspect,
            growth_alpha=inputs.growth_alpha,
        )

    def forward(
        self,
        inputs: TrainableCompositeInputs,
        *,
        phase_timer: PhaseTimer | None = None,
    ) -> tuple[torch.Tensor, ...]:
        with _record_phase(phase_timer, "conditioning"):
            conditioning = self.forward_conditioning(inputs)
        with _record_phase(phase_timer, "dit_forward"):
            return self.forward_dit(inputs, conditioning)


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
    microbatches: int
    effective_samples: int
    state: SingleGpuUpdateState


class StepOptimizer(Protocol):
    def step(self) -> None: ...

    def zero_grad(self, *, set_to_none: bool) -> None: ...


class SingleGpuStep:
    """Accumulate per-sample sums, then normalize once at the update boundary."""

    def __init__(
        self,
        module: nn.Module,
        optimizer: StepOptimizer,
        *,
        accumulation_steps: int,
        state: SingleGpuUpdateState,
    ) -> None:
        if type(accumulation_steps) is not int or accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        if not any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("training module has no trainable parameters")
        self.module = module
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self._state: SingleGpuUpdateState = state
        self._microbatches = 0
        self._samples = 0
        self._loss_sum: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._failed = False

    @property
    def state(self) -> SingleGpuUpdateState:
        return self._state

    @property
    def pending_samples(self) -> int:
        return self._samples

    @property
    def pending_microbatches(self) -> int:
        return self._microbatches

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
        if not bool(torch.isfinite(per_sample_loss).all().item()):
            self._abort_pending_update()
            raise FloatingPointError("per-sample loss is nonfinite")

        loss_sum = per_sample_loss.sum()
        try:
            loss_sum.backward()  # pyright: ignore[reportUnknownMemberType]
        except Exception:
            self._abort_pending_update()
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

    def abort(self) -> None:
        """Poison an interrupted update and discard every pending gradient."""

        self._abort_pending_update()

    def finish_update(
        self, *, phase_timer: PhaseTimer | None = None
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
        parameters = tuple(
            parameter for parameter in self.module.parameters() if parameter.requires_grad
        )
        try:
            with _record_phase(phase_timer, "clip"):
                gradient_scale = 1.0 / self._samples
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale)
                clip = clip_grad_norm_fp32(parameters, max_norm=1.0)
            with _record_phase(phase_timer, "optimizer"):
                self.optimizer.step()
        except Exception as error:
            self._failed = True
            try:
                with _record_phase(phase_timer, "zero_grad"):
                    self.optimizer.zero_grad(set_to_none=True)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve both failures
                raise ExceptionGroup(
                    "update failed and gradient cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise

        successful = replace(
            attempted,
            successful_updates=attempted.successful_updates + 1,
            effective_samples=attempted.effective_samples + self._samples,
        )
        self._state = successful
        try:
            with _record_phase(phase_timer, "zero_grad"):
                self.optimizer.zero_grad(set_to_none=True)
        except Exception:
            self._failed = True
            raise

        result = SingleGpuUpdateResult(
            mean_loss=self._loss_sum / self._samples,
            clip=clip,
            microbatches=self._microbatches,
            effective_samples=self._samples,
            state=successful,
        )
        self._state = result.state
        self._microbatches = 0
        self._samples = 0
        self._loss_sum = None
        self._device = None
        return result


__all__ = [
    "SingleGpuStep",
    "SingleGpuUpdateResult",
    "SingleGpuUpdateState",
    "StepOptimizer",
    "TrainableComposite",
    "TrainableCompositeInputs",
]
