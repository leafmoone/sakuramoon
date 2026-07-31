"""JLT interpolation, velocity loss, and classifier-free guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_JLT_P_MEAN = -0.8
_JLT_P_STD = 0.8
_NOISE_SCALE = 1.0
_T_EPS = 0.05
_GUIDANCE_SCALE = 2.9


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


def _require_locked_float(name: str, value: object, expected: float) -> None:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        raise ValueError(f"{name} must be the locked TOML float {expected}")


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
    _require_locked_float("t_eps", t_eps, _T_EPS)
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
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    _require_locked_float("p_mean", p_mean, _JLT_P_MEAN)
    _require_locked_float("p_std", p_std, _JLT_P_STD)
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
    _require_locked_float("noise_scale", noise_scale, _NOISE_SCALE)
    return (
        torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        * noise_scale
    )


def interpolate_state(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    _validate_flow_tensors(clean, noise)
    if clean.dtype != noise.dtype:
        raise ValueError("clean and noise must have matching dtypes")
    _validate_batch_timestep(timestep, clean, validate_range=True)
    weight = _broadcast_timestep(timestep, clean).to(clean.dtype)
    return weight * clean + (1.0 - weight) * noise


def x_prediction_to_velocity(
    x_prediction: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    *,
    t_eps: float,
) -> torch.Tensor:
    _validate_flow_tensors(state, x_prediction)
    _require_locked_float("t_eps", t_eps, _T_EPS)
    _validate_batch_timestep(timestep, state, validate_range=True)
    return _x_prediction_to_velocity(x_prediction, state, timestep, t_eps)


def flow_matching_loss(
    x_prediction: torch.Tensor,
    state: torch.Tensor,
    clean: torch.Tensor,
    timestep: torch.Tensor,
    *,
    t_eps: float,
) -> FlowLossOutput:
    _validate_flow_tensors(state, x_prediction, clean)
    _require_locked_float("t_eps", t_eps, _T_EPS)
    _validate_batch_timestep(timestep, state, validate_range=True)
    predicted_velocity = _x_prediction_to_velocity(
        x_prediction, state, timestep, t_eps
    )
    target_velocity = _x_prediction_to_velocity(clean, state, timestep, t_eps)
    squared_error = (predicted_velocity - target_velocity).square()
    per_sample = squared_error.flatten(1).mean(dim=1)
    high_noise = timestep < 0.5
    low_noise = ~high_noise
    return FlowLossOutput(
        loss=per_sample.mean(),
        per_sample=per_sample,
        predicted_velocity=predicted_velocity,
        target_velocity=target_velocity,
        high_noise_loss_sum=(per_sample * high_noise).sum(),
        high_noise_sample_count=high_noise.sum(),
        low_noise_loss_sum=(per_sample * low_noise).sum(),
        low_noise_sample_count=low_noise.sum(),
    )


def guided_velocity(
    conditional_x_prediction: torch.Tensor,
    unconditional_x_prediction: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    *,
    t_eps: float,
    guidance_scale: float,
) -> torch.Tensor:
    _validate_flow_tensors(
        state,
        conditional_x_prediction,
        unconditional_x_prediction,
    )
    _require_locked_float("t_eps", t_eps, _T_EPS)
    _require_locked_float("guidance_scale", guidance_scale, _GUIDANCE_SCALE)
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
