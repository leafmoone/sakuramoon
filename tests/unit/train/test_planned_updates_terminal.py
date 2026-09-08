# pyright: reportPrivateUsage=false
"""Regression lock: ``config.train.max_updates`` is the ABSOLUTE
successful-update terminal of the CURRENT invocation, read live from the
config on every resume, in BOTH directions.

Locks ``_resume_state_for_config`` and ``_terminal_completed`` in
``sakuramoon.train.production``:

- a persisted budget terminal below the configured terminal is live-
  extended to EXACTLY ``config.train.max_updates`` (NOT
  ``start_successful_update + planned_updates``, the legacy offset
  expression), because later saved updates must stay representable inside
  the persisted envelope;
- a configured terminal at or below the historical one is ACCEPTED: the
  historical terminal is compatibility metadata, never a resume permission,
  and stays persisted unchanged (the live loop / forced final checkpoint
  stop at the configured terminal);
- a configured terminal at or below the restored update completes with
  zero updates (a successful no-op, not a failure and not a rollback):
  counters are never decremented and the source state is not rewritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sakuramoon.checkpoint.policy import CheckpointCadence
from sakuramoon.checkpoint.schema import (
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import load_config
from sakuramoon.model.growth import active_slot_ids, half_cosine_growth_alpha
from sakuramoon.train.production import (
    _resume_state_for_config,
    _terminal_completed,
)
from sakuramoon.train.step import SingleGpuUpdateState

REPOSITORY_ROOT = Path(__file__).parents[3]

# Canonical worked example: a checkpoint 130k updates into a stage whose
# historically recorded terminal was 168k.
CURRENT = 130_000
HISTORICAL = 168_000


def _config(max_updates: int, world_size: int | None = None) -> Any:
    config = load_config(
        Path("train_s0.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
            "WANDB_API_KEY": "synthetic-wandb-secret",
        },
    ).config
    train = config.train.model_copy(update={"max_updates": max_updates})
    if world_size is not None:
        config = config.model_copy(
            update={
                "distributed": config.distributed.model_copy(
                    update={"world_size": world_size}
                )
            }
        )
    return config.model_copy(update={"train": train})


def _state(
    *,
    successful_updates: int,
    start_successful_update: int,
    terminal_successful_update: int,
    config: Any,
    ramp_updates: int | None = None,
) -> RawCheckpointState:
    growth_kwargs: dict[str, Any] = {
        "ramp_start_successful_update": (
            start_successful_update if ramp_updates is not None else None
        ),
        "ramp_updates": ramp_updates,
        "alpha": (
            half_cosine_growth_alpha(
                successful_updates - start_successful_update, ramp_updates
            )
            if ramp_updates is not None
            else 1.0
        ),
    }
    trainer = SingleGpuUpdateState(
        attempted_updates=successful_updates,
        successful_updates=successful_updates,
        effective_samples=successful_updates * 468,
    )
    growth = GrowthCheckpointState(
        active_slot_ids=active_slot_ids(config.model.dit.depth),
        alpha=growth_kwargs["alpha"],
        stage=config.run.label or "",
        world_size=config.distributed.world_size,
        resolution=config.train.resolution,
        ramp_start_successful_update=growth_kwargs["ramp_start_successful_update"],
        ramp_updates=growth_kwargs["ramp_updates"],
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


def test_resume_shortened_terminal_above_current_is_accepted() -> None:
    # max_updates=150000 below the historical 168000 but above the
    # restored 130000: train 20000 updates; the historical envelope is
    # kept verbatim (it still bounds every update this run will save).
    config = _config(max_updates=150_000)
    state = _state(
        successful_updates=CURRENT,
        start_successful_update=0,
        terminal_successful_update=HISTORICAL,
        config=config,
    )

    resumed = _resume_state_for_config(config, state)

    assert resumed.stage_budget.terminal_successful_update == HISTORICAL
    assert resumed.trainer.successful_updates == CURRENT


def test_resume_terminal_below_historical_but_above_current_is_accepted() -> None:
    # max_updates=135000: train 5000 updates to 135000.
    config = _config(max_updates=135_000)
    state = _state(
        successful_updates=CURRENT,
        start_successful_update=0,
        terminal_successful_update=HISTORICAL,
        config=config,
    )

    resumed = _resume_state_for_config(config, state)

    assert resumed.stage_budget.terminal_successful_update == HISTORICAL
    assert resumed.trainer.successful_updates == CURRENT


def test_resume_terminal_equal_to_current_completes_accepted() -> None:
    # max_updates == restored update: accepted here; the production
    # lifecycle completes it with zero updates (tested below).
    config = _config(max_updates=CURRENT)
    state = _state(
        successful_updates=CURRENT,
        start_successful_update=0,
        terminal_successful_update=HISTORICAL,
        config=config,
    )

    resumed = _resume_state_for_config(config, state)

    assert resumed.stage_budget.terminal_successful_update == HISTORICAL
    assert resumed.trainer.successful_updates == CURRENT


def test_resume_terminal_below_current_never_rewinds_counters() -> None:
    # max_updates=120000 < restored 130000: a clean zero-update completion,
    # NOT a rollback.  The source state's counters are untouched.
    config = _config(max_updates=120_000)
    state = _state(
        successful_updates=CURRENT,
        start_successful_update=0,
        terminal_successful_update=HISTORICAL,
        config=config,
    )

    resumed = _resume_state_for_config(config, state)

    assert resumed.trainer.successful_updates == CURRENT
    assert resumed.trainer.attempted_updates == CURRENT
    # Frozen source state object is byte-identical (never mutated in place).
    assert state.trainer.successful_updates == CURRENT
    assert state.stage_budget.terminal_successful_update == HISTORICAL


def test_resume_source_state_object_is_immutable_across_rebinds() -> None:
    for max_updates in (200_000, 150_000, 135_000, CURRENT, 120_000):
        config = _config(max_updates=max_updates)
        state = _state(
            successful_updates=CURRENT,
            start_successful_update=0,
            terminal_successful_update=HISTORICAL,
            config=config,
        )
        snapshot = (
            state.trainer.successful_updates,
            state.stage_budget.terminal_successful_update,
            state.growth.world_size,
            state.checkpoint_cadence.every_successful_updates,
        )
        _resume_state_for_config(config, state)
        assert (
            state.trainer.successful_updates,
            state.stage_budget.terminal_successful_update,
            state.growth.world_size,
            state.checkpoint_cadence.every_successful_updates,
        ) == snapshot


def test_shortened_resume_still_rebinds_checkpoint_cadence() -> None:
    config = _config(max_updates=135_000)
    state = _state(
        successful_updates=CURRENT,
        start_successful_update=0,
        terminal_successful_update=HISTORICAL,
        config=config,
    )
    rebound_config = config.model_copy(
        update={
            "checkpoint": config.checkpoint.model_copy(
                update={"full_every_updates": 5_000}
            )
        }
    )

    resumed = _resume_state_for_config(rebound_config, state)

    assert resumed.checkpoint_cadence.every_successful_updates == 5_000
    assert resumed.stage_budget.terminal_successful_update == HISTORICAL


def test_shortened_resume_still_rebinds_world_size() -> None:
    config = _config(max_updates=135_000, world_size=2)
    state = _state(
        successful_updates=CURRENT,
        start_successful_update=0,
        terminal_successful_update=HISTORICAL,
        config=_config(max_updates=135_000),  # state carries world_size 1
    )

    resumed = _resume_state_for_config(config, state)

    assert resumed.growth.world_size == 2
    assert resumed.stage_budget.terminal_successful_update == HISTORICAL


def test_shortened_terminal_stops_before_in_flight_ramp_end() -> None:
    # In-flight growth: 100 updates in, ramp ends at 300, historical
    # terminal 1000, current max_updates 150.  The invocation may stop at
    # 150 with the ramp still in flight; the historical envelope (1000)
    # keeps the persisted ramp state valid — no schema corruption, no
    # silent ramp completion.
    config = _config(max_updates=150)
    state = _state(
        successful_updates=100,
        start_successful_update=0,
        terminal_successful_update=1_000,
        config=config,
        ramp_updates=300,
    )

    resumed = _resume_state_for_config(config, state)

    growth = resumed.growth
    assert growth.ramp_start_successful_update == 0
    assert growth.ramp_updates == 300
    assert growth.alpha == half_cosine_growth_alpha(100, 300)
    assert resumed.stage_budget.terminal_successful_update == 1_000
    # A checkpoint saved at the live terminal (150) must still validate
    # against the persisted envelope.
    final_state = RawCheckpointState(
        trainer=SingleGpuUpdateState(150, 150, 150 * 468),
        growth=GrowthCheckpointState(
            active_slot_ids=growth.active_slot_ids,
            alpha=half_cosine_growth_alpha(150, 300),
            stage=growth.stage,
            world_size=growth.world_size,
            resolution=growth.resolution,
            ramp_start_successful_update=0,
            ramp_updates=300,
        ),
        stage_budget=StageBudgetCheckpointState(0, 1_000),
        checkpoint_cadence=CheckpointCadence(150, 150.5, 5_000),
    )
    assert (
        final_state.stage_budget.terminal_successful_update
        >= final_state.trainer.successful_updates
    )


def test_terminal_completed_uses_only_the_live_config() -> None:
    # N == C: zero-update completion.
    assert _terminal_completed(_config(max_updates=CURRENT), CURRENT) is True
    # N < C: also a clean zero-update completion (never a rollback).
    assert _terminal_completed(_config(max_updates=120_000), CURRENT) is True
    # N > C: the invocation still trains.
    assert _terminal_completed(_config(max_updates=135_000), CURRENT) is False
    # The historical terminal (168000 in every state of this file) never
    # participates: a live target below it cannot be extended by it.
    assert _terminal_completed(_config(max_updates=150_000), CURRENT) is False
    assert _terminal_completed(_config(max_updates=200_000), CURRENT) is False
