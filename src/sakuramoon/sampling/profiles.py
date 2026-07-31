"""Canonical sampling profile identities shared by config and runtime code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

SamplingProfileName = Literal["preview", "balanced", "reference"]
SamplingSolver = Literal["euler", "heun_final_euler"]
TimeSchedule = Literal["linear"]

_PROFILE_SPECS: Final[
    dict[SamplingProfileName, tuple[SamplingSolver, int, TimeSchedule]]
] = {
    "preview": ("euler", 28, "linear"),
    "balanced": ("heun_final_euler", 25, "linear"),
    "reference": ("heun_final_euler", 50, "linear"),
}


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    name: SamplingProfileName
    solver: SamplingSolver
    steps: int
    time_schedule: TimeSchedule

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in _PROFILE_SPECS:
            raise ValueError("sampling profile name is not canonical")
        expected = _PROFILE_SPECS[self.name]
        if (self.solver, self.steps, self.time_schedule) != expected:
            raise ValueError("sampling profile differs from the canonical registry")

    @property
    def nfe(self) -> int:
        if self.solver == "euler":
            return self.steps
        return 2 * self.steps - 1


_SAMPLING_PROFILES: Final[dict[SamplingProfileName, SamplingProfile]] = {
    name: SamplingProfile(name, solver, steps, schedule)
    for name, (solver, steps, schedule) in _PROFILE_SPECS.items()
}
SAMPLING_PROFILES: Final[Mapping[SamplingProfileName, SamplingProfile]] = (
    MappingProxyType(_SAMPLING_PROFILES)
)


def resolve_sampling_profile(name: SamplingProfileName) -> SamplingProfile:
    """Return one canonical profile without constructing ad-hoc combinations."""

    if type(name) is not str:
        raise ValueError("sampling profile name must be a string")
    try:
        return SAMPLING_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown sampling profile: {name}") from error


__all__ = [
    "SAMPLING_PROFILES",
    "SamplingProfile",
    "SamplingProfileName",
    "SamplingSolver",
    "TimeSchedule",
    "resolve_sampling_profile",
]
