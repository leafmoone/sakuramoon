"""Manual quality indexing without converting FID/IS into a release gate."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

QualityField = Literal[
    "tag_control", "aesthetic", "nl_following", "composition", "detail"
]


@dataclass(frozen=True, slots=True)
class ManualQualityObservation:
    prompt_id: str
    tag_control: float
    aesthetic: float
    nl_following: float
    composition: float
    detail: float
    severe_artifact: bool

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("manual quality prompt ID must not be empty")
        for name in (
            "tag_control",
            "aesthetic",
            "nl_following",
            "composition",
            "detail",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 5.0:
                raise ValueError("manual quality scores must be finite floats in [0,5]")
        if type(self.severe_artifact) is not bool:
            raise TypeError("severe artifact label must be boolean")


@dataclass(frozen=True, slots=True)
class ManualQualityReport:
    sample_count: int
    tag_control_mean: float
    aesthetic_mean: float
    nl_following_mean: float
    composition_mean: float
    detail_mean: float
    severe_artifact_rate: float
    automatic_release: bool = False


def summarize_manual_quality(
    observations: tuple[ManualQualityObservation, ...],
) -> ManualQualityReport:
    if not observations:
        raise ValueError("manual quality observations must not be empty")
    identifiers = tuple(item.prompt_id for item in observations)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("manual quality prompt IDs must be unique")
    def mean(name: QualityField) -> float:
        return float(statistics.fmean(getattr(item, name) for item in observations))
    return ManualQualityReport(
        sample_count=len(observations),
        tag_control_mean=mean("tag_control"),
        aesthetic_mean=mean("aesthetic"),
        nl_following_mean=mean("nl_following"),
        composition_mean=mean("composition"),
        detail_mean=mean("detail"),
        severe_artifact_rate=sum(item.severe_artifact for item in observations)
        / len(observations),
    )


__all__ = [
    "ManualQualityObservation",
    "ManualQualityReport",
    "summarize_manual_quality",
]
