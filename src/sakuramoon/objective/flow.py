"""JLT interpolation, velocity loss, and classifier-free guidance.

Every numeric constant is a real runtime parameter supplied by the resolved
config (see config/schema.py TimestepConfig / CfgConfig).  The formulas are
unchanged: sigmoid-normal JLT sampling, the clamped x-to-v conversion, the
inverse-square endpoint weighting, and the x-to-v-then-CFG ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowLossOutput:
    loss: torch.Tensor
    per_sample: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor
    high_noise_loss_sum: torch.Tensor
    high_noise_sample_count: torch.Tensor
    low_noise_loss_sum: torch.Tensor
    low_noise_sample_count: torch.Tensor


def _require_finite_float(name: str, value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _require_t_eps(t_eps: object) -> float:
    value = _require_finite_float("t_eps", t_eps)
    if not 0.0 < value < 1.0:
        raise ValueError("t_eps must lie in (0,1) (denominator protection)")
    return value


def _validate_flow_tensors(
    reference: torch.Tensor,
    *others: torch.Tensor,
) -> None:
    if reference.ndim < 2 or reference.shape[0] <= 0:
        raise ValueError("flow tensors must have shape [batch,...] with nonempty batch")
    if not reference.is_floating_point():
        raise ValueError("flow tensors must be floating point")
    for tensor in others:
        if tensor.shape != reference.shape:
            raise ValueError("all flow tensors must have matching shapes")
        if tensor.device != reference.device:
            raise ValueError("all flow tensors must share a device")
        if not tensor.is_floating_point():
            raise ValueError("flow tensors must be floating point")


def _validate_batch_timestep(
    timestep: torch.Tensor,
    reference: torch.Tensor,
    *,
    validate_range: bool,
) -> None:
    if timestep.shape != (reference.shape[0],) or timestep.dtype != torch.float32:
        raise ValueError("timestep must be FP32 with shape [batch]")
    if timestep.device != reference.device:
        raise ValueError("timestep and data tensors must share a device")
    if validate_range and bool(((timestep < 0.0) | (timestep > 1.0)).any().item()):
        raise ValueError("timestep must be in [0,1]")


def _broadcast_timestep(
    timestep: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return timestep.reshape(timestep.shape[0], *((1,) * (reference.ndim - 1)))


def _x_prediction_to_velocity(
    x_prediction: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    t_eps: float,
) -> torch.Tensor:
    _require_t_eps(t_eps)
    denominator = (1.0 - timestep).clamp_min(t_eps)
    denominator = _broadcast_timestep(denominator, state)
    return (x_prediction.float() - state.float()) / denominator


def sample_jlt_timesteps(
    batch_size: int,
    *,
    p_mean: float,
    p_std: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sigmoid-normal JLT timestep sampling with the given parameters."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    _require_finite_float("p_mean", p_mean)
    if type(p_std) is not float or not math.isfinite(p_std) or p_std <= 0.0:
        raise ValueError("p_std must be a finite positive float")
    normal = torch.randn(
        batch_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return torch.sigmoid(normal * p_std + p_mean)


def sample_noise(
    clean: torch.Tensor,
    *,
    noise_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if clean.ndim < 2 or clean.shape[0] <= 0 or not clean.is_floating_point():
        raise ValueError("clean must be floating point with shape [batch,...]")
    scale = _require_finite_float("noise_scale", noise_scale)
    if scale <= 0.0:
        raise ValueError("noise_scale must be positive")
    return torch.randn(
        clean.shape,
        device=clean.device,
        dtype=torch.float32,
        generator=generator,
    ) * scale


def interpolate_state(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """z_t = t * x + (1 - t) * epsilon, broadcast over the batch axis."""

    _validate_flow_tensors(clean, noise)
    _validate_batch_timestep(timestep, clean, validate_range=True)
    t = _broadcast_timestep(timestep, clean)
    return t * clean.float() + (1.0 - t) * noise.float()


def flow_matching_loss(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    x_prediction: torch.Tensor,
    *,
    t_eps: float,
    noise_observation_boundary: float,
) -> FlowLossOutput:
    """Per-sample velocity MSE with the inverse-square endpoint weighting.

    The state is rebuilt from the same (clean, noise, timestep) triple so the
    loss target is exactly the x-to-v conversion of the prediction.
    ``noise_observation_boundary`` splits the high/low-noise telemetry
    buckets; it only needs to lie in (0,1).
    """

    _validate_flow_tensors(clean, noise, x_prediction)
    _validate_batch_timestep(timestep, clean, validate_range=True)
    t_eps_value = _require_t_eps(t_eps)
    if (
        type(noise_observation_boundary) is not float
        or not math.isfinite(noise_observation_boundary)
        or not 0.0 < noise_observation_boundary < 1.0
    ):
        raise ValueError("noise_observation_boundary must lie in (0,1)")
    state = interpolate_state(clean, noise, timestep)
    target_velocity = (
        clean.float() - state.float()
    ) / (1.0 - _broadcast_timestep(timestep, clean)).clamp_min(t_eps_value)
    predicted_velocity = _x_prediction_to_velocity(
        x_prediction, state, timestep, t_eps_value
    )
    per_sample = torch.mean(
        (predicted_velocity - target_velocity) ** 2,
        dim=tuple(range(1, x_prediction.ndim)),
    )
    weight = 1.0 / (timestep - t_eps_value).square()
    loss = torch.sum(per_sample * weight) / torch.sum(weight)
    high_mask = timestep >= noise_observation_boundary
    return FlowLossOutput(
        loss=loss,
        per_sample=per_sample,
        predicted_velocity=predicted_velocity,
        target_velocity=target_velocity,
        high_noise_loss_sum=per_sample[high_mask].sum(),
        high_noise_sample_count=high_mask.sum(),
        low_noise_loss_sum=per_sample[~high_mask].sum(),
        low_noise_sample_count=(~high_mask).sum(),
    )


def x_prediction_to_velocity(
    x_prediction: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    t_eps: float,
) -> torch.Tensor:
    _validate_flow_tensors(state, x_prediction)
    _validate_batch_timestep(timestep, state, validate_range=False)
    return _x_prediction_to_velocity(x_prediction, state, timestep, t_eps)


def guided_velocity(
    conditional_x_prediction: torch.Tensor,
    unconditional_x_prediction: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    *,
    t_eps: float,
    guidance_scale: float,
) -> torch.Tensor:
    """CFG applied after both x-predictions are converted to velocity.

    ``guidance_scale`` comes from the resolved config ([cfg].scale); the
    x-to-v -> CFG ordering is preserved exactly.
    """

    _validate_flow_tensors(state, conditional_x_prediction, unconditional_x_prediction)
    _require_t_eps(t_eps)
    scale = _require_finite_float("guidance_scale", guidance_scale)
    if scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    _validate_batch_timestep(timestep, state, validate_range=False)
    conditional_velocity = _x_prediction_to_velocity(
        conditional_x_prediction,
        state,
        timestep,
        t_eps,
    )
    unconditional_velocity = _x_prediction_to_velocity(
        unconditional_x_prediction,
        state,
        timestep,
        t_eps,
    )
    return unconditional_velocity + guidance_scale * (
        conditional_velocity - unconditional_velocity
    )


__all__ = [
    "FlowLossOutput",
    "flow_matching_loss",
    "guided_velocity",
    "interpolate_state",
    "sample_jlt_timesteps",
    "sample_noise",
    "x_prediction_to_velocity",
]
