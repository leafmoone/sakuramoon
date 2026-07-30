from __future__ import annotations

import math
from itertools import pairwise

import pytest

from sakuramoon.model.growth import (
    growth_ramp_updates,
    half_cosine_growth_alpha,
    is_new_slot_fqn,
    new_slot_fqn_prefixes,
)


def test_growth_ramp_is_two_percent_with_locked_bounds() -> None:
    assert growth_ramp_updates(10) == 1000
    assert growth_ramp_updates(75_001) == 1501
    assert growth_ramp_updates(1_000_000) == 5000
    with pytest.raises(ValueError, match="positive"):
        growth_ramp_updates(0)


def test_half_cosine_has_exact_endpoints_and_midpoint() -> None:
    assert half_cosine_growth_alpha(0, 1000) == 0.0
    assert half_cosine_growth_alpha(500, 1000) == pytest.approx(0.5)
    assert half_cosine_growth_alpha(1000, 1000) == 1.0
    assert half_cosine_growth_alpha(5000, 1000) == 1.0
    values = [half_cosine_growth_alpha(index, 1000) for index in range(1001)]
    assert all(left <= right for left, right in pairwise(values))
    assert values[250] == pytest.approx(0.5 - 0.5 * math.cos(math.pi / 4))


def test_new_slot_prefixes_are_exact_and_base_has_none() -> None:
    assert new_slot_fqn_prefixes(20) == (
        "dit.blocks.slot_02.",
        "dit.conditioner.block_biases.slot_02",
        "dit.blocks.slot_08.",
        "dit.conditioner.block_biases.slot_08",
        "dit.blocks.slot_14.",
        "dit.conditioner.block_biases.slot_14",
        "dit.blocks.slot_20.",
        "dit.conditioner.block_biases.slot_20",
    )
    with pytest.raises(ValueError, match="base"):
        new_slot_fqn_prefixes(16)
    assert is_new_slot_fqn(20, "dit.blocks.slot_02.attention.q_proj.weight")
    assert is_new_slot_fqn(20, "dit.conditioner.block_biases.slot_02")
    assert not is_new_slot_fqn(20, "dit.conditioner.block_biases.slot_02evil")
