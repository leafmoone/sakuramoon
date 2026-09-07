"""Sampling profile identity shared by config and runtime code.

Profiles are user-defined presets in the resolved config
(``[sampling.profiles.<name>]``); the historical preview/balanced/reference
entries remain the default template presets, not a code-level registry.
NFE is computed from the selected solver and step count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SamplingSolver = Literal["euler", "heun_final_euler"]
TimeSchedule = Literal["linear"]


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    name: str
    solver: SamplingSolver
    steps: int
    time_schedule: TimeSchedule

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or self.name != self.name.strip():
            raise ValueError("sampling profile name must be a non-empty string")
        if len(self.name) > 128:
            raise ValueError("sampling profile name is too long")
        if self.solver not in ("euler", "heun_final_euler"):
            raise ValueError(
                "sampling solver must be one of the implemented solvers: "
                "euler, heun_final_euler"
            )
        if self.time_schedule != "linear":
            raise ValueError(
                "sampling time schedule must be the implemented 'linear' schedule"
            )
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("sampling steps must be a positive integer")

    @property
    def nfe(self) -> int:
        """Network function evaluations for the solver at this step count."""

        if self.solver == "euler":
            return self.steps
        return 2 * self.steps - 1


__all__ = [
    "SamplingProfile",
    "SamplingSolver",
    "TimeSchedule",
]
