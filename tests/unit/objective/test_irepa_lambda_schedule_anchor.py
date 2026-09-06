"""Ramp-out anchor boundary regression for the iREPA lambda schedule.

``ramp_out_after_updates`` is an ABSOLUTE successful-update number.  The
anchored ramp-in ends at ``start_successful_update + ramp_in_updates``; a
ramp-out start earlier than that anchored end must be rejected by both
entry points (``IRepaLambdaSchedule.__post_init__`` and
``irepa_weight_for_update``) before any weight computation, including the
``target_weight == 0.0`` and pre-anchor early returns.  Equality with the
anchored ramp-in end is the continuous no-hold boundary and stays legal.

Reference schedule (production contract, see reports/):

    start_successful_update = 113401
    target_weight           = 0.5
    ramp_in_updates         = 1000
    ramp_out_updates        = 1000
    ramp_in_end             = 114401
"""

from __future__ import annotations

import itertools
import math

import pytest  # pyright: ignore[reportMissingImports]

from sakuramoon.objective.irepa import (
    IRepaLambdaSchedule,
    irepa_weight_for_update,
)

START = 113401
TARGET = 0.5
RAMP_IN = 1000
RAMP_OUT = 1000
RAMP_IN_END = START + RAMP_IN  # 114401


def _schedule(
    ramp_out_after_updates: int | None, *, target_weight: float = TARGET
) -> IRepaLambdaSchedule:
    return IRepaLambdaSchedule(
        start_successful_update=START,
        target_weight=target_weight,
        ramp_in_updates=RAMP_IN,
        ramp_out_after_updates=ramp_out_after_updates,
        ramp_out_updates=RAMP_OUT,
    )


def _weight(
    successful_update: int,
    ramp_out_after_updates: int | None,
    *,
    target_weight: float = TARGET,
) -> float:
    return irepa_weight_for_update(
        successful_update=successful_update,
        start_successful_update=START,
        target_weight=target_weight,
        ramp_in_updates=RAMP_IN,
        ramp_out_after_updates=ramp_out_after_updates,
        ramp_out_updates=RAMP_OUT,
    )


def _base_reference(
    successful_update: int,
    *,
    start: int = START,
    target_weight: float = TARGET,
    ramp_in: int = RAMP_IN,
    ramp_out_after: int | None = None,
    ramp_out: int = RAMP_OUT,
) -> float:
    """The pre-fix (fa7096d) schedule formula, verbatim, for value pinning.

    The fix only adds validation; this reference pins that the computed
    value for every legal schedule remains bit-identical to fa7096d
    (exact float equality, no tolerance).
    """

    if target_weight == 0.0:
        return 0.0
    if successful_update < start:
        return 0.0
    if successful_update >= start + ramp_in:
        weight = target_weight
        if ramp_out_after is not None:
            offset = successful_update - ramp_out_after
            if offset <= 0:
                return weight
            if offset >= ramp_out:
                return 0.0
            return target_weight * 0.5 * (1.0 + math.cos(math.pi * offset / ramp_out))
        return weight
    progress = (successful_update - start) / ramp_in
    return target_weight * 0.5 * (1.0 - math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# A-C: ramp-out start before the anchored ramp-in end is rejected (RED on
# fa7096d: the old check compares an absolute update number against a
# duration, so these were silently accepted)
# ---------------------------------------------------------------------------


def test_ramp_out_far_before_anchor_is_rejected_at_both_entries() -> None:
    # A: ramp_out_after_updates = 5000 (absurdly early absolute number; the
    # old check 5000 > 1000 wrongly accepted it)
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _schedule(5000)
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _weight(120000, 5000)


def test_ramp_out_inside_ramp_in_is_rejected_at_both_entries() -> None:
    # B: 114000 lands inside ramp-in (113401..114400]
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _schedule(114000)
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _weight(114200, 114000)


def test_ramp_out_one_update_before_ramp_in_end_is_rejected_at_both_entries() -> None:
    # C: 114400 is the last ramp-in update, one before the anchored end
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _schedule(RAMP_IN_END - 1)
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _weight(RAMP_IN_END, RAMP_IN_END - 1)


# ---------------------------------------------------------------------------
# D-E: legal anchored schedules stay legal and continuous
# ---------------------------------------------------------------------------


def test_ramp_out_at_ramp_in_end_is_continuous_no_hold() -> None:
    # D: equality boundary (114401): ramp-in ends, ramp-out starts at the
    # same update; no boundary jump, exactly 0.0 after the full ramp-out
    schedule = _schedule(RAMP_IN_END)
    assert schedule.weight_for_update(RAMP_IN_END) == TARGET
    assert _weight(RAMP_IN_END, RAMP_IN_END) == TARGET
    values = [
        schedule.weight_for_update(RAMP_IN_END + i) for i in range(RAMP_OUT + 1)
    ]
    assert values[0] == TARGET
    assert all(a > b for a, b in itertools.pairwise(values))
    assert values[RAMP_OUT] == 0.0
    assert schedule.weight_for_update(RAMP_IN_END + RAMP_OUT + 1) == 0.0


def test_ramp_out_after_hold_window() -> None:
    # E: 115401: 1000-update hold then ramp-out; monotone within each phase
    after = RAMP_IN_END + 1000  # 115401
    schedule = _schedule(after)
    assert schedule.weight_for_update(START) == 0.0
    assert schedule.weight_for_update(RAMP_IN_END) == TARGET
    assert schedule.weight_for_update(after) == TARGET
    assert schedule.weight_for_update(after + RAMP_OUT) == 0.0
    # ramp-in phase: strictly increasing
    ramp_in = [
        schedule.weight_for_update(u) for u in range(START, RAMP_IN_END, 37)
    ]
    assert ramp_in[0] == 0.0
    assert all(a < b for a, b in itertools.pairwise(ramp_in))
    # hold phase: exactly the target
    hold = [
        schedule.weight_for_update(u) for u in range(RAMP_IN_END, after, 97)
    ]
    assert all(v == TARGET for v in hold)
    # ramp-out phase: strictly decreasing toward exactly 0.0
    ramp_out = [
        schedule.weight_for_update(u) for u in range(after, after + RAMP_OUT, 37)
    ]
    assert ramp_out[0] == TARGET
    assert all(a > b for a, b in itertools.pairwise(ramp_out))


# ---------------------------------------------------------------------------
# F-G: legal-path values are bit-identical to the pre-fix formula
# ---------------------------------------------------------------------------


def test_ramp_out_none_values_bit_identical_to_base() -> None:
    # F: ramp_out_after_updates = None; before anchor, at anchor, mid-ramp,
    # ramp end, long hold
    for u in (
        0,
        START - 1,
        START,
        START + 1,
        START + 499,
        START + 500,
        RAMP_IN_END - 1,
        RAMP_IN_END,
        RAMP_IN_END + 7,
        10**6,
    ):
        assert _weight(u, None) == _base_reference(u, ramp_out_after=None)


def test_legal_ramp_out_values_bit_identical_to_base() -> None:
    # G: boundary, interior, and post-end points for every legal ramp-out
    for after in (RAMP_IN_END, RAMP_IN_END + 500, 120000):
        points = (
            after - 5,
            after - 1,
            after,
            after + 1,
            after + 250,
            after + 500,
            after + 749,
            after + 999,
            after + 1000,
            after + 1001,
        )
        for u in points:
            assert _weight(u, after) == _base_reference(u, ramp_out_after=after), (
                f"value changed for u={u}, after={after}"
            )


# ---------------------------------------------------------------------------
# H: invalid configs cannot be masked by early returns
# ---------------------------------------------------------------------------


def test_invalid_anchor_rejected_despite_zero_target_and_pre_anchor_update() -> None:
    # H: target_weight == 0.0 or successful_update < anchor must not let an
    # invalid time boundary through the early returns
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _schedule(5000, target_weight=0.0)
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _weight(110000, 5000, target_weight=0.0)  # pre-anchor update
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        _weight(0, 5000)  # successful_update far before the anchor


# ---------------------------------------------------------------------------
# I: the anchored boundary is computed, not hardcoded to 113401
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", [113401, 1, 500000, 999999937])  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
def test_anchored_boundary_is_not_hardcoded(anchor: int) -> None:
    ramp_in = 1000
    ramp_in_end = anchor + ramp_in

    def sched(after: int) -> IRepaLambdaSchedule:
        return IRepaLambdaSchedule(anchor, 0.5, ramp_in, after, 1000)

    def w(u: int, after: int) -> float:
        return irepa_weight_for_update(
            successful_update=u,
            start_successful_update=anchor,
            target_weight=0.5,
            ramp_in_updates=ramp_in,
            ramp_out_after_updates=after,
            ramp_out_updates=1000,
        )

    # one update before the anchored end is rejected at both entries
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        sched(ramp_in_end - 1)
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        w(anchor + 50000, ramp_in_end - 1)
    # the anchored end itself is accepted (equality boundary)
    assert sched(ramp_in_end).weight_for_update(ramp_in_end) == 0.5


def test_zero_anchor_keeps_legacy_stricter_relation_check() -> None:
    # anchor 0: ramp_in_end == ramp_in == 1000.  The legacy relation check
    # (after > ramp_in) stays in force, so after == ramp_in (== ramp_in_end
    # at anchor 0) is still rejected even though the anchored check alone
    # would allow equality; one past the anchored end is accepted.
    with pytest.raises(ValueError, match="ramp_out_after_updates"):  # pyright: ignore[reportUnknownMemberType]
        IRepaLambdaSchedule(0, 0.5, 1000, 1000, 1000)
    schedule = IRepaLambdaSchedule(0, 0.5, 1000, 1001, 1000)
    assert schedule.weight_for_update(1001) == 0.5
