from __future__ import annotations

import math
from itertools import pairwise

import pytest

from sakuramoon.model.growth import (
    half_cosine_growth_alpha,
    is_new_slot_fqn,
    new_slot_fqn_prefixes,
    new_slot_ids,
    slot_growth,
)


def test_new_slot_ids_is_sorted_set_difference() -> None:
    source = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22)
    target = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
    assert new_slot_ids(target, source) == (2, 8, 14, 20)
    # order of the inputs never matters
    assert new_slot_ids(source, source) == ()
    assert new_slot_ids([2, 8, 14, 20], []) == (2, 8, 14, 20)


def test_half_cosine_has_exact_endpoints_and_midpoint() -> None:
    assert half_cosine_growth_alpha(0, 1000) == 0.0
    assert half_cosine_growth_alpha(500, 1000) == pytest.approx(0.5)
    assert half_cosine_growth_alpha(1000, 1000) == 1.0
    assert half_cosine_growth_alpha(5000, 1000) == 1.0
    values = [half_cosine_growth_alpha(index, 1000) for index in range(1001)]
    assert all(left <= right for left, right in pairwise(values))
    assert values[250] == pytest.approx(0.5 - 0.5 * math.cos(math.pi / 4))


def test_half_cosine_accepts_any_positive_ramp() -> None:
    for ramp in (1, 2, 3360, 123457):
        assert half_cosine_growth_alpha(0, ramp) == 0.0
        assert half_cosine_growth_alpha(ramp, ramp) == 1.0
        assert half_cosine_growth_alpha(ramp + 1, ramp) == 1.0
    with pytest.raises(ValueError, match="positive"):
        half_cosine_growth_alpha(0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        half_cosine_growth_alpha(-1, 10)


def test_slot_growth_gates_alpha_to_new_slots_only() -> None:
    active = (0, 1, 2, 8)
    assert slot_growth(active, (2, 8), 0, 0.25) == 1.0
    assert slot_growth(active, (2, 8), 2, 0.25) == 0.25
    assert slot_growth(active, (2, 8), 8, 0.25) == 0.25
    with pytest.raises(ValueError, match="not active"):
        slot_growth(active, (2, 8), 5, 0.25)
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        slot_growth(active, (2, 8), 0, 1.5)


def test_new_slot_prefixes_are_exact() -> None:
    assert new_slot_fqn_prefixes((2, 8, 14, 20)) == (
        "dit.blocks.slot_02.",
        "dit.conditioner.block_biases.slot_02",
        "dit.blocks.slot_08.",
        "dit.conditioner.block_biases.slot_08",
        "dit.blocks.slot_14.",
        "dit.conditioner.block_biases.slot_14",
        "dit.blocks.slot_20.",
        "dit.conditioner.block_biases.slot_20",
    )
    assert new_slot_fqn_prefixes(()) == ()
    assert is_new_slot_fqn((2,), "dit.blocks.slot_02.attention.q_proj.weight")
    assert is_new_slot_fqn((2,), "dit.conditioner.block_biases.slot_02")
    assert not is_new_slot_fqn((2,), "dit.conditioner.block_biases.slot_02evil")
    assert not is_new_slot_fqn((2,), "dit.blocks.slot_00.mlp.down_proj.weight")
