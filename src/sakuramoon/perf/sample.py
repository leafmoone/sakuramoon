"""One measured logical-update sample (P0 observatory).

A sample is one successful logical update (including all configured
accumulation microbatches) on one rank.  Only phases that were actually
measured appear in ``phases``; a phase that was not measured is ABSENT,
never recorded as a fake 0.0 second value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from sakuramoon.telemetry.metrics import TIMING_PHASES

PERFORMANCE_SCHEMA_VERSION = 1
PHASE_NAMES: frozenset[str] = frozenset(TIMING_PHASES)


def _float_field(payload: Mapping[str, object], key: str) -> float:
    value = payload[key]
    if type(value) is not float:
        raise ValueError(f"{key} must be a float")
    return value


def _int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an int")
    return value


def _require_finite_nonnegative_seconds(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")
    return value


def _require_nonnegative_int(value: object, name: str, *, positive: bool) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an int")
    if positive and value <= 0:
        raise ValueError(f"{name} must be a positive int")
    if not positive and value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


@dataclass(frozen=True, slots=True)
class PerformanceSample:
    """Strict, JSON-round-trippable record of one measured update."""

    update: int
    wall_seconds: float
    samples: int
    image_tokens: int
    text_tokens: int
    dit_forward_matmul_flops: int
    phases: dict[str, float]
    memory_allocated_bytes: int
    memory_reserved_bytes: int
    schema_version: int = PERFORMANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_SCHEMA_VERSION:
            raise ValueError("unsupported performance sample schema version")
        _require_nonnegative_int(self.update, "update", positive=True)
        if (
            type(self.wall_seconds) is not float
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0.0
        ):
            raise ValueError("wall_seconds must be a finite positive float")
        _require_nonnegative_int(self.samples, "samples", positive=True)
        _require_nonnegative_int(self.image_tokens, "image_tokens", positive=False)
        _require_nonnegative_int(self.text_tokens, "text_tokens", positive=False)
        _require_nonnegative_int(
            self.dit_forward_matmul_flops,
            "dit_forward_matmul_flops",
            positive=False,
        )
        for name, seconds in self.phases.items():
            if name not in PHASE_NAMES:
                raise ValueError(f"unknown performance phase: {name}")
            _require_finite_nonnegative_seconds(seconds, f"phases[{name}]")
        _require_nonnegative_int(
            self.memory_allocated_bytes, "memory_allocated_bytes", positive=False
        )
        _require_nonnegative_int(
            self.memory_reserved_bytes, "memory_reserved_bytes", positive=False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "update": self.update,
            "wall_seconds": self.wall_seconds,
            "samples": self.samples,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
            "dit_forward_matmul_flops": self.dit_forward_matmul_flops,
            "phases": {key: self.phases[key] for key in sorted(self.phases)},
            "memory_allocated_bytes": self.memory_allocated_bytes,
            "memory_reserved_bytes": self.memory_reserved_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PerformanceSample:
        expected = {
            "schema_version",
            "update",
            "wall_seconds",
            "samples",
            "image_tokens",
            "text_tokens",
            "dit_forward_matmul_flops",
            "phases",
            "memory_allocated_bytes",
            "memory_reserved_bytes",
        }
        if set(payload) != expected:
            raise ValueError(
                f"performance sample payload keys differ: {sorted(payload)}"
            )
        raw_phases_value = payload["phases"]
        if type(raw_phases_value) is not dict:
            raise ValueError("phases must map strings to seconds")
        raw_phases = cast(dict[str, object], raw_phases_value)
        if not all(type(name) is str for name in raw_phases):
            raise ValueError("phases must map strings to seconds")
        return cls(
            update=_int_field(payload, "update"),
            wall_seconds=_float_field(payload, "wall_seconds"),
            samples=_int_field(payload, "samples"),
            image_tokens=_int_field(payload, "image_tokens"),
            text_tokens=_int_field(payload, "text_tokens"),
            dit_forward_matmul_flops=_int_field(payload, "dit_forward_matmul_flops"),
            phases={str(name): _float_field(raw_phases, name) for name in raw_phases},
            memory_allocated_bytes=_int_field(payload, "memory_allocated_bytes"),
            memory_reserved_bytes=_int_field(payload, "memory_reserved_bytes"),
            schema_version=_int_field(payload, "schema_version"),
        )


__all__ = [
    "PERFORMANCE_SCHEMA_VERSION",
    "PHASE_NAMES",
    "PerformanceSample",
]
