"""JLT interpolation, velocity loss, and classifier-free guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowLossOutput:
    loss: torch.Tensor
    per_sample: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor


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
    if batch_size <= 0 or p_std <= 0.0:
        raise ValueError("batch_size and p_std must be positive")
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
    if not clean.is_floating_point() or noise_scale <= 0.0:
        raise ValueError(
            "clean must be floating point and noise_scale must be positive"
        )
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
    if clean.shape != noise.shape or clean.dtype != noise.dtype:
        raise ValueError("clean and noise must have matching shapes and dtypes")
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
    if x_prediction.shape != state.shape:
        raise ValueError("x_prediction and state must have matching shapes")
    if t_eps <= 0.0 or t_eps >= 1.0:
        raise ValueError("t_eps must be in (0,1)")
    _validate_batch_timestep(timestep, state, validate_range=True)
    return _x_prediction_to_velocity(x_prediction, state, timestep, t_eps)


def flow_matching_loss(
    x_prediction: torch.Tensor,
    state: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    *,
    t_eps: float,
) -> FlowLossOutput:
    if not (x_prediction.shape == state.shape == clean.shape == noise.shape):
        raise ValueError("all flow tensors must have matching shapes")
    predicted_velocity = x_prediction_to_velocity(
        x_prediction,
        state,
        timestep,
        t_eps=t_eps,
    )
    target_velocity = clean.float() - noise.float()
    squared_error = (predicted_velocity - target_velocity).square()
    per_sample = squared_error.flatten(1).mean(dim=1)
    return FlowLossOutput(
        loss=per_sample.mean(),
        per_sample=per_sample,
        predicted_velocity=predicted_velocity,
        target_velocity=target_velocity,
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
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be nonnegative")
    if conditional_x_prediction.shape != state.shape:
        raise ValueError("conditional prediction and state must have matching shapes")
    if unconditional_x_prediction.shape != state.shape:
        raise ValueError("unconditional prediction and state must have matching shapes")
    if t_eps <= 0.0 or t_eps >= 1.0:
        raise ValueError("t_eps must be in (0,1)")
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
