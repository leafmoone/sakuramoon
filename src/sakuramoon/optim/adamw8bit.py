"""TorchAO AdamW8bit with isolated stochastic-rounding RNG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torchao.optim import AdamW8bit  # pyright: ignore[reportMissingTypeStubs]

from sakuramoon.optim.groups import ParameterAudit, audit_trainable_parameters
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG


class _QuantizedState(Protocol):
    block_size: int
    codes: torch.Tensor
    qmap: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True, slots=True)
class OptimizerStateSpec:
    name: str
    initialized: bool
    state_class: str
    state_bytes: int
    step: int
    block_size: int | None


def _moment_storage_bytes(moment: torch.Tensor) -> int:
    if type(moment).__name__ != "OptimState8bit":
        return moment.numel() * moment.element_size()
    quantized = cast(_QuantizedState, moment)
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (quantized.codes, quantized.qmap, quantized.scale)
    )


class IsolatedAdamW8bit:
    def __init__(
        self,
        optimizer: Optimizer,
        audit: ParameterAudit,
        sr_rng: StochasticRoundingRNG,
    ) -> None:
        self.optimizer = optimizer
        self.audit = audit
        self.sr_rng = sr_rng

    def _validate_finite_gradients(self) -> None:
        gradients = [
            spec.parameter.grad
            for spec in self.audit.specs
            if spec.parameter.grad is not None
        ]
        if not gradients:
            return
        device = gradients[0].device
        if any(gradient.device != device for gradient in gradients):
            raise ValueError("all gradients must share one device")
        finite = torch.ones((), device=device, dtype=torch.bool)
        for gradient in gradients:
            finite.logical_and_(torch.isfinite(gradient).all())
        if not bool(finite.item()):
            raise FloatingPointError("optimizer received a nonfinite gradient")

    def step(self) -> None:
        self._validate_finite_gradients()
        self.sr_rng.run_step(self.optimizer.step)

    def zero_grad(self, *, set_to_none: bool) -> None:
        if not set_to_none:
            raise ValueError("optimizer zero_grad requires set_to_none=True")
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "sr_rng": self.sr_rng.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        optimizer_state = state_dict.get("optimizer")
        sr_state = state_dict.get("sr_rng")
        if not isinstance(optimizer_state, dict) or not isinstance(sr_state, dict):
            raise TypeError("optimizer and SR RNG state must be mappings")
        self.optimizer.load_state_dict(cast(dict[str, object], optimizer_state))
        self.sr_rng.load_state_dict(cast(dict[str, object], sr_state))

    def audit_state(self) -> tuple[OptimizerStateSpec, ...]:
        """Validate lazy/quantized state and report physical bytes by canonical FQN."""

        results: list[OptimizerStateSpec] = []
        for spec in self.audit.specs:
            state = self.optimizer.state.get(spec.parameter)
            if not state:
                results.append(
                    OptimizerStateSpec(
                        name=spec.name,
                        initialized=False,
                        state_class="uninitialized",
                        state_bytes=0,
                        step=0,
                        block_size=None,
                    )
                )
                continue
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError(f"unexpected optimizer state fields: {spec.name}")
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (step, exp_avg, exp_avg_sq)
            ):
                raise TypeError(f"optimizer state values must be tensors: {spec.name}")
            step_tensor = cast(torch.Tensor, step)
            first_moment = cast(torch.Tensor, exp_avg)
            second_moment = cast(torch.Tensor, exp_avg_sq)
            state_classes = {type(first_moment).__name__, type(second_moment).__name__}
            if len(state_classes) != 1:
                raise TypeError(f"optimizer moment classes differ: {spec.name}")
            state_class = next(iter(state_classes))
            block_size: int | None = None
            if state_class == "OptimState8bit":
                first_quantized = cast(_QuantizedState, first_moment)
                second_quantized = cast(_QuantizedState, second_moment)
                if first_quantized.block_size != second_quantized.block_size:
                    raise ValueError(f"optimizer moment block sizes differ: {spec.name}")
                block_size = first_quantized.block_size
            if spec.group == "matrix_decay" and (
                state_class != "OptimState8bit" or block_size != 256
            ):
                raise RuntimeError(
                    f"BF16 decay parameter did not receive 256-block 8-bit state: {spec.name}"
                )
            if step_tensor.numel() != 1:
                raise ValueError(f"optimizer step must be scalar: {spec.name}")
            results.append(
                OptimizerStateSpec(
                    name=spec.name,
                    initialized=True,
                    state_class=state_class,
                    state_bytes=(
                        step_tensor.numel() * step_tensor.element_size()
                        + _moment_storage_bytes(first_moment)
                        + _moment_storage_bytes(second_moment)
                    ),
                    step=int(step_tensor.item()),
                    block_size=block_size,
                )
            )
        return tuple(results)


def build_adamw8bit(
    module: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    block_size: int,
    bf16_stochastic_round: bool,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
    sr_seed: int,
) -> IsolatedAdamW8bit:
    if (
        type(lr) is not float
        or not math.isfinite(lr)
        or lr <= 0.0
        or betas != (0.9, 0.95)
        or eps != 1e-8
        or block_size != 256
        or not bf16_stochastic_round
    ):
        raise ValueError("optimizer settings differ from the locked policy")
    audit = audit_trainable_parameters(
        module,
        matrix_weight_decay=matrix_weight_decay,
        sensitive_weight_decay=sensitive_weight_decay,
    )
    fallback = [
        spec.name
        for spec in audit.decay
        if spec.parameter.numel() < 4096
        or spec.parameter.numel() % block_size != 0
    ]
    if fallback:
        raise ValueError(
            "BF16 decay parameters would use non-quantized optimizer state: "
            + ", ".join(fallback)
        )
    devices = {spec.parameter.device for spec in audit.specs}
    if len(devices) != 1:
        raise ValueError("all trainable parameters must share one device")
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError("TorchAO AdamW8bit production optimizer requires CUDA")
    parameter_groups = [
        {
            "params": [spec.parameter for spec in audit.decay],
            "param_names": [spec.name for spec in audit.decay],
            "weight_decay": matrix_weight_decay,
            "group_name": "matrix_decay",
        },
        {
            "params": [spec.parameter for spec in audit.sensitive],
            "param_names": [spec.name for spec in audit.sensitive],
            "weight_decay": sensitive_weight_decay,
            "group_name": "sensitive_no_decay",
        },
    ]
    optimizer = cast(
        Optimizer,
        AdamW8bit(
            parameter_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            amsgrad=False,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        ),
    )
    return IsolatedAdamW8bit(
        optimizer=optimizer,
        audit=audit,
        sr_rng=StochasticRoundingRNG.seeded(device, sr_seed),
    )


__all__ = ["IsolatedAdamW8bit", "OptimizerStateSpec", "build_adamw8bit"]
