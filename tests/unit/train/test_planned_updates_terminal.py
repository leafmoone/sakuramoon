# pyright: reportPrivateUsage=false
"""Regression lock: ``config.train.max_updates`` is the ABSOLUTE
successful-update terminal (production cutover semantics), read live from
the config on every resume.

Locks ``_resume_state_for_config`` in ``sakuramoon.train.production`` after
the R2A baseline reconciliation (F2 production.py was rebased onto the
production cutover commit so this semantics survives F2 deployment):

- a persisted budget terminal below the configured terminal is live-
  extended to EXACTLY ``config.train.max_updates`` — NOT
  ``start_successful_update + planned_updates`` (the legacy offset
  expression, which would yield a different value when start > 0)
- a configured terminal below the persisted terminal stays fail-closed
  (``ValueError``), unchanged by F2

The equality/no-op path (persisted terminal == configured terminal) is
already covered by ``test_resume_rebinds_only_checkpoint_cadence_policy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sakuramoon.checkpoint.policy import CheckpointCadence
from sakuramoon.checkpoint.schema import (
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import load_config
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.production import _resume_state_for_config
from sakuramoon.train.step import SingleGpuUpdateState

REPOSITORY_ROOT = Path(__file__).parents[3]


def _config(max_updates: int) -> Any:
    config = load_config(
        Path("train_s0.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
            "WANDB_API_KEY": "synthetic-wandb-secret",
        },
    ).config
    train = config.train.model_copy(update={"max_updates": max_updates})
    return config.model_copy(update={"train": train})


def _state(
    *,
    successful_updates: int,
    start_successful_update: int,
    terminal_successful_update: int,
    config: Any,
) -> RawCheckpointState:
    trainer = SingleGpuUpdateState(
        attempted_updates=successful_updates,
        successful_updates=successful_updates,
        effective_samples=successful_updates * 468,
    )
    growth = GrowthCheckpointState(
        active_slot_ids=active_slot_ids(config.model.dit.depth),
        alpha=1.0,
        stage=config.run.label or "",
        world_size=config.distributed.world_size,
        resolution=config.train.resolution,
        ramp_start_successful_update=None,
        ramp_updates=None,
    )
    return RawCheckpointState(
        trainer=trainer,
        growth=growth,
        stage_budget=StageBudgetCheckpointState(
            start_successful_update=start_successful_update,
            terminal_successful_update=terminal_successful_update,
        ),
        checkpoint_cadence=CheckpointCadence(
            last_successful_update=successful_updates,
            last_wall_clock_unix_seconds=123.5,
            every_successful_updates=config.checkpoint.full_every_updates,
        ),
    )


def test_resume_live_extends_terminal_to_absolute_planned_updates() -> None:
    config = _config(max_updates=110_000)
    state = _state(
        successful_updates=42_000,
        start_successful_update=10_000,
        terminal_successful_update=50_000,
        config=config,
    )

    resumed = _resume_state_for_config(config, state)

    # The live read of config.train.max_updates is the absolute
    # terminal: exactly the configured value, NOT start + planned
    # (the legacy offset expression would give 120_000 here).
    assert resumed.stage_budget.terminal_successful_update == 110_000
    assert resumed.stage_budget.start_successful_update == 10_000
    assert resumed.trainer is state.trainer


def test_resume_configured_terminal_shrink_stays_fail_closed() -> None:
    config = _config(max_updates=40_000)
    state = _state(
        successful_updates=42_000,
        start_successful_update=0,
        terminal_successful_update=50_000,
        config=config,
    )

    with pytest.raises(ValueError, match="cannot shrink the checkpoint terminal"):
        _resume_state_for_config(config, state)
