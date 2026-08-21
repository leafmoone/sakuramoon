from __future__ import annotations

import pytest

from sakuramoon.checkpoint.schema import GrowthCheckpointState
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.stage import GrowthProgress, canonical_growth_alpha


def test_mid_ramp_resume_continues_from_checkpoint_edge() -> None:
    growth = GrowthCheckpointState(
        active_slot_ids=active_slot_ids(20),
        alpha=0.5,
        stage="G1",
        world_size=2,
        resolution=256,
        ramp_start_successful_update=47_900,
        ramp_updates=1_000,
    )
    midpoint = canonical_growth_alpha(growth, 48_400)
    assert midpoint == pytest.approx(0.5)
    restored = GrowthProgress.from_checkpoint(
        type("State", (), {"growth": growth})()
    )
    assert restored.alpha(48_401) > midpoint
    assert restored.alpha(48_899) < 1.0
    assert restored.alpha(48_900) > restored.alpha(48_899)
    assert restored.alpha(48_900) == 1.0
