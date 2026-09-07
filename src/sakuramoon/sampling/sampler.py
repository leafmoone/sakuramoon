"""Config-profile sampling dispatch and immutable generation metadata."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import torch

from sakuramoon.sampling.heun import VelocityFunction, euler, heun_final_euler
from sakuramoon.sampling.profiles import SamplingProfile

CheckpointKind = Literal["raw", "model-only", "pma", "release"]
ObjectiveProvenance = Literal["strict_jlt", "pre_fix"]
_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ProfileSamplingResult:
    state: torch.Tensor
    profile: SamplingProfile
    nfe: int

    def __post_init__(self) -> None:
        if self.nfe != self.profile.nfe:
            raise ValueError("solver NFE differs from the selected profile")


@dataclass(frozen=True, slots=True)
class GenerationMetadata:
    """Identity of one generation batch; every value is config-bound."""

    checkpoint_id: str
    checkpoint_kind: CheckpointKind
    objective_provenance: ObjectiveProvenance
    profile: SamplingProfile
    cfg_scale: float
    noise_scale: float
    t_eps: float

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint_id) is not str
            or _CHECKPOINT_ID.fullmatch(self.checkpoint_id) is None
        ):
            raise ValueError("checkpoint_id is invalid")
        if self.checkpoint_kind not in ("raw", "model-only", "pma", "release"):
            raise ValueError("checkpoint kind is invalid")
        if self.objective_provenance not in ("strict_jlt", "pre_fix"):
            raise ValueError("objective provenance is invalid")
        if (
            self.objective_provenance == "pre_fix"
            and self.checkpoint_kind != "model-only"
        ):
            raise ValueError(
                "pre-fix weights are only valid as model-only inference input"
            )
        if (
            type(self.cfg_scale) is not float
            or not math.isfinite(self.cfg_scale)
            or self.cfg_scale < 0.0
        ):
            raise ValueError("generation CFG scale must be a finite nonnegative float")
        if (
            type(self.noise_scale) is not float
            or not math.isfinite(self.noise_scale)
            or self.noise_scale <= 0.0
        ):
            raise ValueError("generation noise scale must be a positive finite float")
        if (
            type(self.t_eps) is not float
            or not math.isfinite(self.t_eps)
            or not 0.0 < self.t_eps < 1.0
        ):
            raise ValueError("generation t_eps must lie in (0, 1)")

    def as_mapping(self) -> dict[str, object]:
        profile = self.profile
        return {
            "cfg_scale": self.cfg_scale,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_kind": self.checkpoint_kind,
            "nfe": profile.nfe,
            "noise_scale": self.noise_scale,
            "objective_provenance": self.objective_provenance,
            "prediction_type": "x",
            "profile": profile.name,
            "schema_version": 1,
            "solver": profile.solver,
            "state_dtype": "float32",
            "steps": profile.steps,
            "t_eps": self.t_eps,
            "time_schedule": profile.time_schedule,
        }


def build_generation_metadata(
    sampled: ProfileSamplingResult,
    *,
    checkpoint_id: str,
    checkpoint_kind: CheckpointKind,
    objective_provenance: ObjectiveProvenance,
    cfg_scale: float,
    noise_scale: float,
    t_eps: float,
) -> GenerationMetadata:
    """Bind metadata to the profile that produced the sampled state."""

    return GenerationMetadata(
        checkpoint_id=checkpoint_id,
        checkpoint_kind=checkpoint_kind,
        objective_provenance=objective_provenance,
        profile=sampled.profile,
        cfg_scale=cfg_scale,
        noise_scale=noise_scale,
        t_eps=t_eps,
    )


def sample_profile(
    velocity_function: VelocityFunction,
    initial_noise: torch.Tensor,
    *,
    profile: SamplingProfile,
) -> ProfileSamplingResult:
    """Integrate one resolved sampling profile and verify its NFE."""

    if profile.solver == "euler":
        result = euler(velocity_function, initial_noise, steps=profile.steps)
    else:
        result = heun_final_euler(
            velocity_function,
            initial_noise,
            steps=profile.steps,
        )
    return ProfileSamplingResult(result.state, profile, result.nfe)


__all__ = [
    "CheckpointKind",
    "GenerationMetadata",
    "ObjectiveProvenance",
    "ProfileSamplingResult",
    "build_generation_metadata",
    "sample_profile",
]
