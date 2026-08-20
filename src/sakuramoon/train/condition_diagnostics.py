"""Numerical diagnostics for condition representations and the global path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from sakuramoon.conditioning.global_condition import GlobalConditionOutput
from sakuramoon.eval.spec import PromptCase, PromptManifest

_FIXED_CONDITION_PAIR_IDS = (
    (
        "style-1",
        "validation-fac64aa1d4574f386dec1fefcb2eaca0",
        "validation-fe7761d38db73e6709e5afac5a476aea",
    ),
    (
        "style-2",
        "validation-3d974be80ac708305f26c9ebffe05fde",
        "validation-66c28a00f48348cb118c459eb22bdea1",
    ),
    (
        "identity-1",
        "validation-c17a74b546e841cbac26930ea4119aef",
        "validation-d985f354c95ef3d2137bd3b5468adb43",
    ),
    (
        "identity-2",
        "validation-3cd1cf620669ec4abd38027db060b18b",
        "validation-5821ee0f6504d5eb4cb4135a17aee3fc",
    ),
)


class TrainingSamplingError(RuntimeError):
    """A periodic training sample or condition diagnostic is invalid."""


@dataclass(frozen=True, slots=True)
class FixedConditionPair:
    label: str
    a: PromptCase
    b: PromptCase

    def __post_init__(self) -> None:
        plans = (self.a.caption_plan, self.b.caption_plan)
        if any(plan is None or plan.condition is None for plan in plans):
            raise TrainingSamplingError(
                f"fixed condition pair {self.label} lacks structured conditions"
            )
        a_plan = self.a.caption_plan
        b_plan = self.b.caption_plan
        assert a_plan is not None and a_plan.condition is not None
        assert b_plan is not None and b_plan.condition is not None
        a_condition = a_plan.condition
        b_condition = b_plan.condition
        if (
            a_condition.source != b_condition.source
            or a_condition.role != b_condition.role
            or not frozenset(tag.canonical for tag in a_condition.tags).isdisjoint(
                tag.canonical for tag in b_condition.tags
            )
        ):
            raise TrainingSamplingError(
                f"fixed condition pair {self.label} has incompatible protocols"
            )


def load_fixed_condition_pairs(path: Path) -> tuple[FixedConditionPair, ...]:
    if path.is_symlink() or not path.is_file():
        raise TrainingSamplingError("fixed condition prompt manifest is unavailable")
    try:
        manifest = PromptManifest.from_canonical_bytes(path.read_bytes())
    except (OSError, ValueError):
        raise TrainingSamplingError(
            "fixed condition prompt manifest is invalid"
        ) from None
    by_id = {case.prompt_id: case for case in manifest.cases}
    expected_ids = {
        prompt_id
        for _label, first_id, second_id in _FIXED_CONDITION_PAIR_IDS
        for prompt_id in (first_id, second_id)
    }
    if not expected_ids <= set(by_id):
        raise TrainingSamplingError(
            "fixed condition prompt manifest lacks the locked cohort"
        )
    return tuple(
        FixedConditionPair(label, by_id[first_id], by_id[second_id])
        for label, first_id, second_id in _FIXED_CONDITION_PAIR_IDS
    )


def tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise TrainingSamplingError(
            "diagnostic tensor must be nonempty floating point"
        )
    return tensor.float().square().mean().sqrt()


def tensor_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.numel() == 0:
        raise TrainingSamplingError("diagnostic cosine tensors must match")
    left = first.float().flatten()
    right = second.float().flatten()
    denominator = left.norm() * right.norm()
    if denominator <= 1e-12:
        raise TrainingSamplingError("diagnostic cosine is undefined at zero norm")
    return torch.dot(left, right) / denominator


def metric_float(name: str, value: torch.Tensor) -> float:
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise TrainingSamplingError(f"condition diagnostic is nonfinite: {name}")
    return result


def condition_representation_diagnostics(
    tokens: torch.Tensor,
    *,
    expected_batch: int,
    a_index: int,
    b_index: int,
    null_index: int,
) -> dict[str, float]:
    if tokens.ndim != 3 or tokens.shape[0] != expected_batch:
        raise TrainingSamplingError(
            f"condition diagnostic tokens must match the {expected_batch} CFG branches"
        )
    condition_a = tokens[a_index]
    condition_b = tokens[b_index]
    condition_null = tokens[null_index]
    rms_a = tensor_rms(condition_a)
    rms_b = tensor_rms(condition_b)
    values = {
        "condition_A_rms": rms_a,
        "condition_B_rms": rms_b,
        "condition_active_rms": 0.5 * (rms_a + rms_b),
        "condition_null_rms": tensor_rms(condition_null),
        "condition_A_B_cosine": tensor_cosine(condition_a, condition_b),
        "condition_A_null_cosine": tensor_cosine(condition_a, condition_null),
        "condition_B_null_cosine": tensor_cosine(condition_b, condition_null),
        "condition_A_B_delta_rms": tensor_rms(condition_a - condition_b),
        "condition_A_null_delta_rms": tensor_rms(condition_a - condition_null),
    }
    return {name: metric_float(name, value) for name, value in values.items()}


def global_path_diagnostics(
    output: GlobalConditionOutput,
    projection_weight: torch.Tensor,
    *,
    a_index: int,
    b_index: int,
    null_index: int,
) -> dict[str, float]:
    global_a = output.condition_residual[a_index]
    global_b = output.condition_residual[b_index]
    global_null = output.condition_residual[null_index]
    if torch.count_nonzero(global_null).item() != 0:
        raise TrainingSamplingError(
            "null condition produced a nonzero global residual"
        )
    base_rms = 0.5 * (
        tensor_rms(output.base_hidden[a_index])
        + tensor_rms(output.base_hidden[b_index])
    )
    active_rms = 0.5 * (tensor_rms(global_a) + tensor_rms(global_b))
    values = {
        "global_base_rms": base_rms,
        "global_condition_active_rms": active_rms,
        "global_condition_to_base_ratio": active_rms / base_rms.clamp_min(1e-12),
        "global_total_rms": 0.5
        * (
            tensor_rms(output.total_hidden[a_index])
            + tensor_rms(output.total_hidden[b_index])
        ),
        "global_A_B_delta_rms": tensor_rms(global_a - global_b),
        "condition_global_projection_weight_rms": tensor_rms(projection_weight),
    }
    diagnostics = {
        name: metric_float(name, value) for name, value in values.items()
    }
    if global_a.float().norm() > 1e-12 and global_b.float().norm() > 1e-12:
        diagnostics["global_A_B_cosine"] = metric_float(
            "global_A_B_cosine", tensor_cosine(global_a, global_b)
        )
    return diagnostics


__all__ = [
    "FixedConditionPair",
    "TrainingSamplingError",
    "condition_representation_diagnostics",
    "global_path_diagnostics",
    "load_fixed_condition_pairs",
    "metric_float",
    "tensor_cosine",
    "tensor_rms",
]
