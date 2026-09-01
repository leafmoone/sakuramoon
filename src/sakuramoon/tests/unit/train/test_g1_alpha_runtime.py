from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.checkpoint.schema import GrowthCheckpointState
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.loop import SingleGpuTrainingLoop
from sakuramoon.train.stage import canonical_growth_alpha
from sakuramoon.train.step import SingleGpuUpdateState


def test_canonical_alpha_is_resume_stable() -> None:
    growth = GrowthCheckpointState(
        active_slot_ids=active_slot_ids(20),
        alpha=0.5,
        stage="G1",
        world_size=2,
        resolution=256,
        ramp_start_successful_update=41_500,
        ramp_updates=1_000,
    )
    assert canonical_growth_alpha(growth, 41_500) == 0.0
    assert canonical_growth_alpha(growth, 42_000) == pytest.approx(0.5)
    assert canonical_growth_alpha(growth, 42_500) == 1.0
    assert canonical_growth_alpha(growth, 47_500) == 1.0


def test_training_loop_passes_one_canonical_alpha_to_each_update(tmp_path) -> None:
    torch.manual_seed(12)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    observed: list[float] = []
    state = SingleGpuUpdateState.initial()

    def record(observation) -> None:
        observed.append(observation.update.growth_alpha)

    loop = SingleGpuTrainingLoop(
        module=model,
        optimizer=optimizer,
        loss_fn=lambda batch: (model(batch[0]) - batch[1]).square().flatten(),
        accumulation_steps=1,
        target_successful_updates=3,
        checkpoint_every_successful_updates=100,
        scheduler_step=lambda _update: None,
        checkpoint=lambda _update: None,
        diagnostic_root=tmp_path,
        failure_id=lambda _phase, _state: "g1-alpha-test",
        state=state,
        growth_alpha_for_update=lambda update: (0.1, 0.5, 1.0)[update - 1],
        successful_update_observer=record,
    )
    batch = (torch.ones(2, 2), torch.zeros(2, 1))
    result = loop.run((batch, batch, batch))

    assert result.state.successful_updates == 3
    assert observed == [0.1, 0.5, 1.0]
