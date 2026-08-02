from __future__ import annotations

import pytest

from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.train.scheduler import CheckpointScheduler
from sakuramoon.train.stage import ForcedCheckpoint, checkpoint_reason


def test_elapsed_wall_time_never_triggers_and_update_commit_records_audit_time() -> None:
    wall = iter((6.0 * 3600.0, 365.0 * 24.0 * 3600.0, 32_000_001.0))
    scheduler = CheckpointScheduler(
        CheckpointCadence(0, 0.0, 1000),
        clock=lambda: next(wall),
    )

    assert scheduler.due(1) is None
    assert scheduler.due(999) is None
    decision = scheduler.due(1000)
    assert decision is not None
    assert decision.reason is CheckpointReason.UPDATE_CADENCE
    assert scheduler.cadence.last_successful_update == 0
    proposed = scheduler.proposed_cadence(decision)
    assert proposed.last_successful_update == 1000
    assert proposed.last_wall_clock_unix_seconds == 32_000_001.0
    assert scheduler.cadence.last_successful_update == 0
    scheduler.committed(decision)
    assert scheduler.cadence == proposed


def test_forced_reason_takes_precedence_over_periodic_cadence() -> None:
    events: list[tuple[int, CheckpointReason | None]] = []
    scheduler = CheckpointScheduler(
        CheckpointCadence(0, 0.0, 1000),
        clock=lambda: 0.0,
        forced_checkpoint=lambda update: (
            CheckpointReason.PRE_DECAY if update == 1 else None
        ),
    )

    decision = scheduler.due(1)
    assert decision is not None
    assert decision.reason is CheckpointReason.PRE_DECAY
    events.append((decision.successful_update, decision.reason))
    scheduler.committed(decision)
    assert events == [(1, CheckpointReason.PRE_DECAY)]


def test_failed_update_cadence_save_does_not_advance_audit_anchor() -> None:
    scheduler = CheckpointScheduler(
        CheckpointCadence(0, 0.0, 1000),
        clock=lambda: 6.0 * 3600.0,
    )

    decision = scheduler.due(1000)
    assert decision is not None
    assert decision.reason is CheckpointReason.UPDATE_CADENCE
    # The caller deliberately does not commit after a failed checkpoint.
    assert scheduler.cadence == CheckpointCadence(0, 0.0, 1000)
    retry = scheduler.due(1000)
    assert retry == decision


@pytest.mark.parametrize(
    "reason",
    [
        CheckpointReason.STAGE_FINALIZE,
        CheckpointReason.PRE_DECAY,
        *(checkpoint_reason(value) for value in ForcedCheckpoint),
    ],
)
def test_all_transition_and_stage_reasons_reach_scheduler_as_exact_type(
    reason: CheckpointReason,
) -> None:
    scheduler = CheckpointScheduler(
        CheckpointCadence(0, 0.0, 1000),
        clock=lambda: 1.0,
        forced_checkpoint=lambda _update: reason,
    )

    decision = scheduler.due(1)

    assert decision is not None
    assert decision.reason is reason
