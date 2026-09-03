"""iREPA spatial normalization, cosine alignment loss, and lambda schedule.

Phase 4 installs the pure objective-side building blocks of the iREPA
representation alignment: the FP32 spatial z-score of the frozen teacher's
raw patch features, the FP32 per-sample cosine alignment loss between the
projected student features and the z-scored teacher target, and the
successful-update-based lambda schedule (half-cosine ramp in, optional
half-cosine ramp out).

All three functions are deterministic and stateless: no RNG, no mutable
scheduler state, no counters.  A resume to the same successful-update
number always reproduces the same lambda.

This module deliberately does not touch ``flow.py``: the JLT main loss is
frozen and the iREPA term is combined per sample by the runtime as
``main_per_sample + lambda * irepa_per_sample``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sakuramoon.model.irepa import IREPA_TEACHER_FEATURE_WIDTH


def _require_nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite_nonnegative_float(name: str, value: object) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")


def _require_finite_float(name: str, value: object) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")


@dataclass(frozen=True, slots=True)
class IRepaLambdaSchedule:
    """Pure data carrier for one iREPA lambda schedule.

    ``start_successful_update`` is the successful-update anchor at which the
    ramp-in begins.  Phase 5 persists the production start anchor; until
    then isolated runtime tests/canaries pass it explicitly.
    """

    start_successful_update: int
    target_weight: float
    ramp_in_updates: int
    ramp_out_after_updates: int | None
    ramp_out_updates: int

    def __post_init__(self) -> None:
        _require_nonnegative_int("start_successful_update", self.start_successful_update)
        _require_finite_nonnegative_float("target_weight", self.target_weight)
        _require_positive_int("ramp_in_updates", self.ramp_in_updates)
        if self.ramp_out_after_updates is not None:
            _require_positive_int("ramp_out_after_updates", self.ramp_out_after_updates)
            if self.ramp_out_after_updates <= self.ramp_in_updates:
                raise ValueError(
                    "ramp_out_after_updates must be greater than ramp_in_updates"
                )
        _require_positive_int("ramp_out_updates", self.ramp_out_updates)

    def weight_for_update(self, successful_update: int) -> float:
        return irepa_weight_for_update(
            successful_update=successful_update,
            start_successful_update=self.start_successful_update,
            target_weight=self.target_weight,
            ramp_in_updates=self.ramp_in_updates,
            ramp_out_after_updates=self.ramp_out_after_updates,
            ramp_out_updates=self.ramp_out_updates,
        )


def irepa_weight_for_update(
    *,
    successful_update: int,
    start_successful_update: int,
    target_weight: float,
    ramp_in_updates: int,
    ramp_out_after_updates: int | None,
    ramp_out_updates: int,
) -> float:
    """Return the iREPA weight for one attempted successful update.

    Schedule (all boundaries inclusive, pure function of the arguments):

    - ``successful_update < start_successful_update``: ``0.0``.
    - Ramp-in, ``start <= u < start + ramp_in_updates``:
      ``target * 0.5 * (1 - cos(pi * (u - start) / ramp_in_updates))``.
      Off-by-one contract (locked by tests): the first enabled update
      (``u == start``) gets exactly ``0.0`` and the ramp reaches exactly
      ``target_weight`` at ``u == start + ramp_in_updates``.
    - Hold, ``start + ramp_in_updates <= u`` (and no ramp-out, or
      ``u < ramp_out_after_updates``): ``target_weight``.
    - Optional ramp-out, ``ramp_out_after_updates <= u <
      ramp_out_after_updates + ramp_out_updates``:
      ``target * 0.5 * (1 + cos(pi * (u - ramp_out_after_updates) /
      ramp_out_updates))``, i.e. half-cosine from ``target_weight`` down to
      ``0.0``.
    - ``u >= ramp_out_after_updates + ramp_out_updates``: ``0.0``.

    No mutable state, no RNG, no hidden counter: a resume that rebinds the
    same successful-update number reproduces the same weight.
    """

    _require_nonnegative_int("successful_update", successful_update)
    _require_nonnegative_int("start_successful_update", start_successful_update)
    _require_finite_nonnegative_float("target_weight", target_weight)
    _require_positive_int("ramp_in_updates", ramp_in_updates)
    if ramp_out_after_updates is not None:
        _require_positive_int("ramp_out_after_updates", ramp_out_after_updates)
        if ramp_out_after_updates <= ramp_in_updates:
            raise ValueError(
                "ramp_out_after_updates must be greater than ramp_in_updates"
            )
    _require_positive_int("ramp_out_updates", ramp_out_updates)
    if target_weight == 0.0:
        return 0.0

    if successful_update < start_successful_update:
        return 0.0

    if successful_update >= start_successful_update + ramp_in_updates:
        weight = target_weight
        if ramp_out_after_updates is not None:
            ramp_out_offset = successful_update - ramp_out_after_updates
            if ramp_out_offset <= 0:
                return weight
            if ramp_out_offset >= ramp_out_updates:
                return 0.0
            progress = ramp_out_offset / ramp_out_updates
            return target_weight * 0.5 * (1.0 + math.cos(math.pi * progress))
        return weight

    progress = (successful_update - start_successful_update) / ramp_in_updates
    return target_weight * 0.5 * (1.0 - math.cos(math.pi * progress))


def spatial_zscore_target(
    teacher_features: torch.Tensor,
    *,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    """FP32 spatial z-score of the frozen teacher's raw patch features.

    ``teacher_features`` is ``[B, T, 768]`` (BF16 or FP32 acceptable) with
    ``T = (H // 16) * (W // 16)`` row-major patch tokens.  Normalization is
    over the token/spatial axis (``dim=1``), never the feature axis, with
    the unbiased (``correction=1``) sample variance:

    ``target = (x - gamma * mean) / (sqrt(var) + eps)``

    The result is float32 and requires no gradient.  This normalizes the
    teacher target only; the student features are never spatially
    z-scored.
    """

    if teacher_features.ndim != 3:
        raise ValueError(
            "teacher_features must be [B, T, D], "
            f"got {teacher_features.ndim} dimensions"
    )
    batch, tokens, width = teacher_features.shape
    if batch <= 0 or tokens <= 1 or width != IREPA_TEACHER_FEATURE_WIDTH:
        raise ValueError(
            "teacher_features require B > 0, T > 1 (unbiased token-axis "
            f"z-score), and D = {IREPA_TEACHER_FEATURE_WIDTH} "
            f"(got B={batch}, T={tokens}, D={width})"
        )
    if not teacher_features.is_floating_point():
        raise ValueError("teacher_features must be floating point")
    _require_finite_float("gamma", gamma)
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    _require_finite_float("eps", eps)
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    x = teacher_features.float()
    if not torch.isfinite(x).all():
        raise ValueError("teacher_features contain nonfinite values")
    var, mean = torch.var_mean(x, dim=1, correction=1, keepdim=True)
    target = (x - gamma * mean) / (var.sqrt() + eps)
    if not torch.isfinite(target).all():
        raise ValueError("spatial z-score produced nonfinite values")
    return target


@dataclass(frozen=True, slots=True)
class IRepaAlignmentLossOutput:
    """Per-sample cosine alignment loss and its per-sample mean cosine.

    Both tensors are float32 with shape ``[B]``.  ``per_sample`` is the
    token-mean of ``1 - cosine`` for one sample (one sample = one weight,
    independent of token count); ``cosine_per_sample`` is the token-mean
    cosine used by telemetry.
    """

    per_sample: torch.Tensor
    cosine_per_sample: torch.Tensor


def irepa_alignment_loss(
    student_features: torch.Tensor,
    target: torch.Tensor,
) -> IRepaAlignmentLossOutput:
    """FP32 per-sample cosine alignment loss between student and teacher.

    Both inputs are ``[B, T, 768]`` with exactly equal shapes (no padding,
    no interpolation, no pad-to-max).  The loss is computed explicitly in
    float32:

    ``cosine = F.cosine_similarity(student.float(), target.float(), dim=-1)``
    ``per_sample = (1 - cosine).mean(dim=1)``

    ``1 - cosine`` is gradient-equivalent to ``-cosine`` and satisfies the
    SakuraMoon nonnegative loss telemetry contract.  Aggregation is strictly
    token-mean first (per sample), sample-mean second (by the trainer): a
    batch token flatten followed by a global mean is forbidden because it
    would let token count change the sample weight.
    """

    if student_features.ndim != 3 or target.ndim != 3:
        raise ValueError(
            "student_features and target must both be [B, T, D], got "
            f"{student_features.ndim} and {target.ndim} dimensions"
        )
    if student_features.shape != target.shape:
        raise ValueError(
            "student_features and target must have exactly equal shapes, "
            f"got {tuple(student_features.shape)} vs {tuple(target.shape)}"
        )
    batch, tokens, width = student_features.shape
    if batch <= 0 or tokens <= 0 or width != IREPA_TEACHER_FEATURE_WIDTH:
        raise ValueError(
            f"iREPA features require B > 0, T > 0, and D = "
            f"{IREPA_TEACHER_FEATURE_WIDTH} (got B={batch}, T={tokens}, "
            f"D={width})"
        )
    if not student_features.is_floating_point() or not target.is_floating_point():
        raise ValueError("iREPA loss inputs must be floating point")
    if not torch.isfinite(student_features.float()).all():
        raise ValueError("student_features contain nonfinite values")
    if not torch.isfinite(target.float()).all():
        raise ValueError("target contains nonfinite values")

    student_fp32 = student_features.float()
    target_fp32 = target.float()
    cosine = F.cosine_similarity(student_fp32, target_fp32, dim=-1)
    token_loss = 1.0 - cosine
    return IRepaAlignmentLossOutput(
        per_sample=token_loss.mean(dim=1),
        cosine_per_sample=cosine.mean(dim=1),
    )


__all__ = [
    "IRepaAlignmentLossOutput",
    "IRepaLambdaSchedule",
    "irepa_alignment_loss",
    "irepa_weight_for_update",
    "spatial_zscore_target",
]
