from __future__ import annotations

from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.schema import (
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.production import _forced_production_checkpoint_reason
from sakuramoon.train.step import SingleGpuUpdateState


def _g1_state() -> RawCheckpointState:
    trainer = SingleGpuUpdateState(
        attempted_updates=47_900,
        successful_updates=47_900,
        effective_samples=1,
    )
    return RawCheckpointState(
        trainer=trainer,
        growth=GrowthCheckpointState(
            active_slot_ids(20),
            0.0,
            "G1",
            2,
            256,
            47_900,
            1_000,
        ),
        stage_budget=StageBudgetCheckpointState(47_900, 50_000),
        checkpoint_cadence=CheckpointCadence(47_900, 1.0, 100),
    )


def test_forced_checkpoint_reasons_are_durable_and_ordered() -> None:
    state = _g1_state()
    assert _forced_production_checkpoint_reason(state, initial_update=47_900, update=47_900) is None
    assert _forced_production_checkpoint_reason(state, initial_update=47_900, update=48_400) is CheckpointReason.RAMP_MIDPOINT
    assert _forced_production_checkpoint_reason(state, initial_update=47_900, update=48_900) is CheckpointReason.RAMP_END
    assert _forced_production_checkpoint_reason(state, initial_update=47_900, update=50_000) is CheckpointReason.STAGE_FINALIZE
