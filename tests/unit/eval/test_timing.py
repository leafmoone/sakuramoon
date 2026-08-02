from __future__ import annotations

# pyright: reportPrivateUsage=false
import math
from typing import Literal

import pytest

from sakuramoon.eval import runner as runner_module
from sakuramoon.eval.spec import EvaluationCost
from sakuramoon.eval.timing import allowed_gpu_clock_overshoot_seconds


@pytest.mark.parametrize("scope", ["checkpoint", "overall"])
def test_runtime_cost_accepts_clock_quantization_overshoot_without_clamping(
    scope: Literal["checkpoint", "overall"],
) -> None:
    wall_seconds = 1200.0
    allowed_overshoot = allowed_gpu_clock_overshoot_seconds(wall_seconds)
    gpu_seconds = wall_seconds + allowed_overshoot / 2.0

    cost = runner_module._runtime_evaluation_cost(
        wall_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
        training_paused=True,
        scope=scope,
    )

    assert cost.wall_seconds == wall_seconds
    assert cost.gpu_seconds == gpu_seconds
    assert cost.training_pause_seconds == wall_seconds


@pytest.mark.parametrize("scope", ["checkpoint", "overall"])
def test_runtime_cost_rejects_impossible_overshoot_with_measured_values(
    scope: Literal["checkpoint", "overall"],
) -> None:
    wall_seconds = 1200.0
    allowed_overshoot = allowed_gpu_clock_overshoot_seconds(wall_seconds)
    gpu_seconds = wall_seconds + allowed_overshoot * 4.0

    with pytest.raises(RuntimeError) as captured:
        runner_module._runtime_evaluation_cost(
            wall_seconds=wall_seconds,
            gpu_seconds=gpu_seconds,
            training_paused=False,
            scope=scope,
        )

    message = str(captured.value)
    assert message.startswith(f"{scope} evaluator cost is invalid:")
    assert f"gpu_seconds={gpu_seconds!r}" in message
    assert f"wall_seconds={wall_seconds!r}" in message
    assert f"overshoot_seconds={gpu_seconds - wall_seconds!r}" in message
    assert "allowed_clock_overshoot_seconds=" in message


@pytest.mark.parametrize("wall_seconds", [1.0, 60.0, 1200.0])
def test_evaluation_cost_accepts_exact_boundary_and_rejects_next_float(
    wall_seconds: float,
) -> None:
    allowed_overshoot = allowed_gpu_clock_overshoot_seconds(wall_seconds)
    boundary_gpu_seconds = wall_seconds + allowed_overshoot
    assert (
        allowed_gpu_clock_overshoot_seconds(boundary_gpu_seconds)
        == allowed_overshoot
    )

    accepted = EvaluationCost(wall_seconds, boundary_gpu_seconds, 0.0)
    assert accepted.wall_seconds == wall_seconds
    assert accepted.gpu_seconds == boundary_gpu_seconds

    rejected_gpu_seconds = math.nextafter(boundary_gpu_seconds, math.inf)
    with pytest.raises(ValueError) as captured:
        EvaluationCost(wall_seconds, rejected_gpu_seconds, 0.0)
    message = str(captured.value)
    assert f"gpu_seconds={rejected_gpu_seconds!r}" in message
    assert f"wall_seconds={wall_seconds!r}" in message
    assert f"overshoot_seconds={rejected_gpu_seconds - wall_seconds!r}" in message
    assert f"allowed_clock_overshoot_seconds={allowed_overshoot!r}" in message
