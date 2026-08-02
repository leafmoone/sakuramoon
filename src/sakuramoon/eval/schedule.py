"""TOML-derived successful-update and stage-end evaluation scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sakuramoon.config.schema import (
    EvaluationConfig,
    EvaluationEnabledConfig,
    FidEnabledConfig,
    IsEnabledConfig,
)

EvaluationRunKind = Literal["trend", "formal"]


@dataclass(frozen=True, slots=True)
class ScheduledEvaluation:
    metric: Literal["fid", "is"]
    run_kind: EvaluationRunKind
    sample_count: int
    successful_update: int


def scheduled_evaluations(
    config: EvaluationConfig,
    *,
    successful_update: int,
    stage_end: bool,
) -> tuple[ScheduledEvaluation, ...]:
    if type(successful_update) is not int or successful_update <= 0:
        raise ValueError("successful update must be positive")
    if type(stage_end) is not bool:
        raise TypeError("stage_end must be explicit")
    if not isinstance(config, EvaluationEnabledConfig):
        return ()
    requests: list[ScheduledEvaluation] = []
    metrics = (
        ("fid", config.fid) if isinstance(config.fid, FidEnabledConfig) else None,
        ("is", config.is_) if isinstance(config.is_, IsEnabledConfig) else None,
    )
    for selected in metrics:
        if selected is None:
            continue
        name, metric = selected
        if stage_end:
            requests.append(
                ScheduledEvaluation(
                    name,  # pyright: ignore[reportArgumentType]
                    "formal",
                    metric.acceptance_samples,
                    successful_update,
                )
            )
        elif successful_update % metric.every_successful_updates == 0:
            requests.append(
                ScheduledEvaluation(
                    name,  # pyright: ignore[reportArgumentType]
                    "trend",
                    metric.trend_samples,
                    successful_update,
                )
            )
    return tuple(requests)


__all__ = ["EvaluationRunKind", "ScheduledEvaluation", "scheduled_evaluations"]
