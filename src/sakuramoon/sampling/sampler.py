"""Profile-only sampling dispatch and immutable generation metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch

from sakuramoon.sampling.heun import VelocityFunction, euler, heun_final_euler
from sakuramoon.sampling.profiles import (
    SamplingProfile,
    SamplingProfileName,
    resolve_sampling_profile,
)

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
    checkpoint_id: str
    checkpoint_kind: CheckpointKind
    objective_provenance: ObjectiveProvenance
    profile: SamplingProfileName
    cfg_scale: float

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
        if type(self.cfg_scale) is not float or self.cfg_scale != 2.9:
            raise ValueError("generation CFG must equal 2.9")
        resolve_sampling_profile(self.profile)

    def as_mapping(self) -> dict[str, object]:
        selected = resolve_sampling_profile(self.profile)
        return {
            "cfg_scale": self.cfg_scale,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_kind": self.checkpoint_kind,
            "nfe": selected.nfe,
            "noise_scale": 1.0,
            "objective_provenance": self.objective_provenance,
            "prediction_type": "x",
            "profile": selected.name,
            "schema_version": 1,
            "solver": selected.solver,
            "state_dtype": "float32",
            "steps": selected.steps,
            "t_eps": 0.05,
            "time_schedule": selected.time_schedule,
        }


def build_generation_metadata(
    sampled: ProfileSamplingResult,
    *,
    checkpoint_id: str,
    checkpoint_kind: CheckpointKind,
    objective_provenance: ObjectiveProvenance,
    cfg_scale: float,
) -> GenerationMetadata:
    """Bind metadata to the profile that produced the sampled state."""

    return GenerationMetadata(
        checkpoint_id=checkpoint_id,
        checkpoint_kind=checkpoint_kind,
        objective_provenance=objective_provenance,
        profile=sampled.profile.name,
        cfg_scale=cfg_scale,
    )


def sample_profile(
    velocity_function: VelocityFunction,
    initial_noise: torch.Tensor,
    *,
    profile: SamplingProfileName,
) -> ProfileSamplingResult:
    """Integrate one of the three canonical profiles and verify its NFE."""

    selected = resolve_sampling_profile(profile)
    if selected.solver == "euler":
        result = euler(velocity_function, initial_noise, steps=selected.steps)
    else:
        result = heun_final_euler(
            velocity_function,
            initial_noise,
            steps=selected.steps,
        )
    return ProfileSamplingResult(result.state, selected, result.nfe)


__all__ = [
    "CheckpointKind",
    "GenerationMetadata",
    "ObjectiveProvenance",
    "ProfileSamplingResult",
    "build_generation_metadata",
    "sample_profile",
]
