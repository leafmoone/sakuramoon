"""B1 tests for the raw-loader saved-learning-rate sink.

The restore path replaces every saved optimizer group learning rate with
the current configuration value; the sink captures the saved rates (with
fail-closed type validation) so the preflight resume contract can compare
them against the configured rate.
"""

from __future__ import annotations

import inspect

import pytest

from sakuramoon.checkpoint.load import (
    _capture_saved_group_lrs,  # pyright: ignore[reportPrivateUsage]
    load_raw_checkpoint,
)
from sakuramoon.checkpoint.schema import CheckpointError

TWO_GROUPS = [
    {
        "params": [1],
        "param_names": ["old.weight"],
        "weight_decay": 0.0,
        "group_name": "matrix_decay",
        "lr": 0.00007421875,
    },
    {
        "params": [2],
        "param_names": ["old.norm"],
        "weight_decay": 0.1,
        "group_name": "sensitive_no_decay",
        "lr": 0.00007421875,
    },
]


def test_valid_groups_capture_in_order() -> None:
    sink: list[float] = []
    _capture_saved_group_lrs(TWO_GROUPS, sink)  # pyright: ignore[reportPrivateUsage]
    assert sink == [0.00007421875, 0.00007421875]


@pytest.mark.parametrize(
    "bad_lr",
    [
        1,  # int is not a float
        float("nan"),
        float("inf"),
        0.0,
        -1.0,
    ],
)
def test_invalid_saved_lr_fails_closed(bad_lr: object) -> None:
    groups = [dict(TWO_GROUPS[0], lr=bad_lr)]
    sink: list[float] = []
    with pytest.raises(CheckpointError, match="learning rate is invalid"):
        _capture_saved_group_lrs(groups, sink)  # pyright: ignore[reportPrivateUsage]
    assert sink == []


def test_missing_lr_fails_closed() -> None:
    group = {key: value for key, value in TWO_GROUPS[0].items() if key != "lr"}
    sink: list[float] = []
    with pytest.raises(CheckpointError, match="learning rate is invalid"):
        _capture_saved_group_lrs([group], sink)  # pyright: ignore[reportPrivateUsage]


def test_non_dict_group_fails_closed() -> None:
    sink: list[float] = []
    with pytest.raises(CheckpointError, match="group is invalid"):
        _capture_saved_group_lrs(["not a group"], sink)  # pyright: ignore[reportPrivateUsage]


def test_load_raw_checkpoint_accepts_saved_lr_sink() -> None:
    """The public loader exposes the keyword-only, default-None capture sink."""
    parameter = inspect.signature(load_raw_checkpoint).parameters["saved_lr_sink"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
