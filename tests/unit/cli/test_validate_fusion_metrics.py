"""Error diagnostics add context without replacing numerical acceptance."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import math

import pytest
import torch

from sakuramoon.cli.validate_fusions import _gradient_names, _tensor_error_metrics
from sakuramoon.model.fusion import norm_modulation_reference


def test_relative_l2_and_reference_scales() -> None:
    values = _tensor_error_metrics(
        torch.tensor([3.0, 4.0]), torch.tensor([3.0, 5.0]), atol=0.01, rtol=0.01
    )
    assert values["reference_l2"] == 5
    assert values["reference_max_abs"] == 4
    assert values["error_l2"] == 1
    assert values["relative_l2"] == 0.2
    assert values["floored_max_relative"] == 0.25


def test_near_zero_errors_are_reported_not_hidden() -> None:
    values = _tensor_error_metrics(
        torch.tensor([0.0, 0.01, 100.0]),
        torch.tensor([0.1, 0.21, 101.0]),
        atol=0.06,
        rtol=0.06,
    )
    assert values["relative_floor"] == 1
    assert values["near_zero_count"] == 2
    assert values["near_zero_max_abs_error"] == pytest.approx(0.2)
    assert values["floored_max_relative"] == pytest.approx(0.2)
    assert values["max_abs_error"] == 1


def test_zero_reference_and_absolute_only_checks() -> None:
    zero = torch.zeros(2)
    assert _tensor_error_metrics(zero, zero, atol=0, rtol=0)["relative_l2"] == 0
    values = _tensor_error_metrics(zero, torch.ones(2), atol=0.1, rtol=0)
    assert math.isinf(values["relative_l2"])
    assert values["relative_floor"] == 0.1
    assert values["floored_max_relative"] == 10


def test_nonfinite_is_visible_and_shape_mismatch_fails() -> None:
    values = _tensor_error_metrics(
        torch.ones(2), torch.tensor([1.0, float("nan")]), atol=0.06, rtol=0.06
    )
    assert math.isnan(values["relative_l2"])
    with pytest.raises(ValueError, match="matching nonempty"):
        _tensor_error_metrics(torch.ones(2), torch.ones(3), atol=0.06, rtol=0.06)


def test_gradient_labels_follow_requires_grad_inputs() -> None:
    inputs = [
        torch.ones(2, 4, requires_grad=True),
        torch.ones(4, requires_grad=True),
        torch.ones(1, 4, requires_grad=True),
        torch.ones(1, 4, requires_grad=True),
        1e-6,
        torch.zeros(2, dtype=torch.int64),
    ]
    assert _gradient_names(norm_modulation_reference, inputs) == [
        "x",
        "weight",
        "scale",
        "shift",
    ]


def test_comparison_prints_diagnostics_then_still_rejects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse
    from typing import Any

    import sakuramoon.cli.validate_fusions as validation

    def reference(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    def bad_candidate(x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1

    def fake_compile(function: Any) -> Any:
        return bad_candidate

    monkeypatch.setattr(validation, "compiled", fake_compile)
    args = argparse.Namespace(explain=False, atol=0.01, rtol=0.01)
    with pytest.raises(AssertionError):
        validation._compare(
            "intentional_failure",
            reference,
            [torch.ones(2, requires_grad=True)],
            args,
            torch.device("cpu"),
        )
    output = capsys.readouterr().out
    assert "forward[output_0]" in output
    assert "reference_l2=" in output
    assert "relative_l2=0.5" in output
    assert "floored_max_relative=" in output
