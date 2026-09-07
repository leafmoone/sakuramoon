"""Governed MBS canary stop cap (100U invocation stop) tests.

Pins the runtime-only ``stage.canary_stop_successful_update`` semantics:

- absent cap: the invocation target is the restored stage-budget terminal
  (unchanged production behavior);
- present cap: fail-closed validation (positive int, strictly inside
  (restored successful update, stage terminal], automatic_transition false)
  and the target becomes ``min(terminal, cap)``;
- the checkpoint stage budget is NEVER mutated (the cap is an invocation
  stop, not a new stage terminal);
- resuming the same canary config at the stop is rejected before training;
- the single-GPU loop performs exactly the remaining updates and exits,
  with the terminal update-cadence checkpoint (UPDATE_CADENCE, never
  STAGE_FINALIZE) firing before the loop returns;
- the real config files: production and p25 stay cap-free, the MBS canary
  resolves planned_updates 168000 with a stop cap of exactly 118200
  (aligned to the cadence, 118200 % 100 == 0).

Pure CPU; no GPU, no training beyond the mock-loop updates.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config.load import load_config
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.loop import SingleGpuTrainingLoop
from sakuramoon.train.runtime import resolve_single_gpu_stop_cap_target
from sakuramoon.train.step import SingleGpuUpdateState

CANARY_STOP = 118200
SOURCE_UPDATE = 118100
STAGE_START = 58000
STAGE_TERMINAL = 168000
CADENCE = 100


def _config(cap: int | None, *, automatic_transition: bool = False):
    return SimpleNamespace(
        stage=SimpleNamespace(
            canary_stop_successful_update=cap,
            automatic_transition=automatic_transition,
        )
    )


def _growth() -> GrowthCheckpointState:
    return GrowthCheckpointState(
        active_slot_ids=active_slot_ids(20),
        alpha=1.0,
        stage="G1",
        world_size=2,
        resolution=256,
        ramp_start_successful_update=None,
        ramp_updates=None,
    )


def _state(successful_updates: int, *, start: int | None = None) -> RawCheckpointState:
    # RawCheckpointState invariants: the update lies inside the persisted
    # stage budget and the cadence is committed at the update.
    if start is None:
        start = min(STAGE_START, successful_updates)
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(
            attempted_updates=successful_updates,
            successful_updates=successful_updates,
            effective_samples=successful_updates * 800,
        ),
        growth=_growth(),
        stage_budget=StageBudgetCheckpointState(start, STAGE_TERMINAL),
        checkpoint_cadence=CheckpointCadence(
            successful_updates, 1_000.0, CADENCE
        ),
    )


# --- §15: resolver semantics -------------------------------------------------


def test_no_cap_target_is_stage_terminal() -> None:
    state = _state(SOURCE_UPDATE)
    assert (
        resolve_single_gpu_stop_cap_target(_config(None), state)
        == STAGE_TERMINAL
    )


def test_cap_shortens_target_to_stop() -> None:
    state = _state(SOURCE_UPDATE)
    assert (
        resolve_single_gpu_stop_cap_target(_config(CANARY_STOP), state)
        == CANARY_STOP
    )


def test_cap_does_not_mutate_stage_budget() -> None:
    state = _state(SOURCE_UPDATE)
    resolve_single_gpu_stop_cap_target(_config(CANARY_STOP), state)
    assert state.stage_budget.start_successful_update == STAGE_START
    assert state.stage_budget.terminal_successful_update == STAGE_TERMINAL
    assert state.trainer.successful_updates == SOURCE_UPDATE


def test_cap_equal_to_source_rejected() -> None:
    state = _state(SOURCE_UPDATE)
    with pytest.raises(ValueError, match="exceed the restored"):
        resolve_single_gpu_stop_cap_target(_config(SOURCE_UPDATE), state)


def test_cap_below_source_rejected() -> None:
    state = _state(SOURCE_UPDATE)
    with pytest.raises(ValueError, match="exceed the restored"):
        resolve_single_gpu_stop_cap_target(_config(SOURCE_UPDATE - 100), state)


def test_cap_above_terminal_rejected() -> None:
    state = _state(SOURCE_UPDATE)
    with pytest.raises(ValueError, match="stage budget terminal"):
        resolve_single_gpu_stop_cap_target(_config(STAGE_TERMINAL + 1), state)


def test_cap_zero_or_negative_rejected() -> None:
    state = _state(SOURCE_UPDATE)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_single_gpu_stop_cap_target(_config(0), state)


def test_cap_non_int_rejected_at_runtime() -> None:
    state = _state(SOURCE_UPDATE)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_single_gpu_stop_cap_target(_config(118200.0), state)  # type: ignore[arg-type]


def test_cap_with_automatic_transition_rejected() -> None:
    state = _state(SOURCE_UPDATE)
    with pytest.raises(ValueError, match="automatic_transition"):
        resolve_single_gpu_stop_cap_target(
            _config(CANARY_STOP, automatic_transition=True), state  # type: ignore[arg-type]
        )


def test_cap_aligned_to_cadence_accepted() -> None:
    assert CANARY_STOP % CADENCE == 0
    state = _state(SOURCE_UPDATE)
    assert (
        resolve_single_gpu_stop_cap_target(_config(CANARY_STOP), state)
        == CANARY_STOP
    )


def test_resume_at_stop_blocked_before_training() -> None:
    # Second invocation of the same canary config from U118200: the cap no
    # longer exceeds the restored update -> fail closed before any update.
    state = _state(CANARY_STOP)
    with pytest.raises(ValueError, match="exceed the restored"):
        resolve_single_gpu_stop_cap_target(_config(CANARY_STOP), state)


# --- §15 #9: loop mock source 3 / cap 5 --------------------------------------


def _loop(
    *,
    state: SingleGpuUpdateState,
    target: int,
    checkpoint_every: int,
    cadence: CheckpointCadence | None,
    events: list[tuple],
    batches: int,
    tmp_path: Path,
):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def record_observation(observation) -> None:
        reason = observation.checkpoint_reason
        events.append(
            (
                "observer",
                observation.update.state.successful_updates,
                reason.value if reason is not None else None,
            )
        )

    loop = SingleGpuTrainingLoop(
        module=model,
        optimizer=optimizer,
        loss_fn=lambda batch: (model(batch[0]) - batch[1]).square().flatten(),
        accumulation_steps=1,
        target_successful_updates=target,
        checkpoint_every_successful_updates=checkpoint_every,
        scheduler_step=lambda update: events.append(("scheduler", update)),
        checkpoint=lambda update: events.append(("checkpoint", update)),
        diagnostic_root=tmp_path,
        failure_id=lambda _phase, _s: "mbs-canary-stop-cap-test",
        state=state,
        cadence=cadence,
        clock=lambda: 2_000.0,
        checkpoint_event=lambda update, reason: events.append(
            ("checkpoint", update, reason.value)
        ),
        successful_update_observer=record_observation,
    )
    batch = (torch.ones(2, 2), torch.zeros(2, 1))
    return loop.run((batch,) * batches)


def test_loop_source3_cap5_runs_exactly_updates_4_and_5(tmp_path: Path) -> None:
    events: list[tuple] = []
    state = SingleGpuUpdateState(
        attempted_updates=3, successful_updates=3, effective_samples=3 * 800
    )
    result = _loop(
        state=state,
        target=resolve_single_gpu_stop_cap_target(
            _config(5),
            RawCheckpointState(
                trainer=state,
                growth=_growth(),
                stage_budget=StageBudgetCheckpointState(0, 10),
                checkpoint_cadence=CheckpointCadence(3, 1_000.0, 10),
            ),
        ),
        checkpoint_every=10,
        cadence=CheckpointCadence(3, 1_000.0, 10),
        events=events,
        batches=2,
        tmp_path=tmp_path,
    )
    assert result.state.successful_updates == 5
    schedulers = [e for e in events if e[0] == "scheduler"]
    assert [e[1] for e in schedulers] == [4, 5]
    observers = [e for e in events if e[0] == "observer"]
    assert [e[1] for e in observers] == [4, 5]
    assert all(e[1] != 6 for e in events)
    assert not any(
        "stage-finalize" in e for e in events if len(e) > 2
    )


def test_terminal_checkpoint_reason_is_update_cadence_not_stage_finalize(
    tmp_path: Path,
) -> None:
    events: list[tuple] = []
    state = SingleGpuUpdateState(
        attempted_updates=118_199,
        successful_updates=118_199,
        effective_samples=118_199 * 800,
    )
    target = resolve_single_gpu_stop_cap_target(
        _config(CANARY_STOP),
        RawCheckpointState(
            trainer=state,
            growth=_growth(),
            stage_budget=StageBudgetCheckpointState(
                STAGE_START, STAGE_TERMINAL
            ),
            checkpoint_cadence=CheckpointCadence(118_199, 1_000.0, CADENCE),
        ),
    )
    assert target == CANARY_STOP
    result = _loop(
        state=state,
        target=target,
        checkpoint_every=CADENCE,
        cadence=CheckpointCadence(118_199, 1_000.0, CADENCE),
        events=events,
        batches=1,
        tmp_path=tmp_path,
    )
    assert result.state.successful_updates == CANARY_STOP
    checkpoints = [e for e in events if e[0] == "checkpoint"]
    assert checkpoints == [("checkpoint", CANARY_STOP, "update-cadence")]
    observers = [e for e in events if e[0] == "observer"]
    assert observers == [("observer", CANARY_STOP, "update-cadence")]
    assert all(
        "stage-finalize" not in str(e) for e in events
    ), "a stop cap must never produce a stage finalize"


def test_terminal_order_update_scheduler_checkpoint_observer_exit(
    tmp_path: Path,
) -> None:
    # §16: at the canary terminal, the exact observable order is
    # update 118200 success -> scheduler step -> cadence checkpoint ->
    # successful observer -> loop exit (118201 never starts).
    events: list[tuple] = []
    state = SingleGpuUpdateState(
        attempted_updates=118_199,
        successful_updates=118_199,
        effective_samples=118_199 * 800,
    )
    result = _loop(
        state=state,
        target=CANARY_STOP,
        checkpoint_every=CADENCE,
        cadence=CheckpointCadence(118_199, 1_000.0, CADENCE),
        events=events,
        batches=1,
        tmp_path=tmp_path,
    )
    assert result.state.successful_updates == CANARY_STOP
    assert (
        events
        == [
            ("scheduler", CANARY_STOP),
            ("checkpoint", CANARY_STOP, "update-cadence"),
            ("observer", CANARY_STOP, "update-cadence"),
        ]
    )
    assert not any(
        isinstance(e[1], int) and e[1] > CANARY_STOP for e in events
    ), "U118201 must not begin"


# --- §15 #12-15: real config files -------------------------------------------


def test_default_production_config_has_no_cap_and_target_is_terminal() -> None:
    loaded = load_config(
        Path("train_g1_cmuon_production.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    assert loaded.config.stage.canary_stop_successful_update is None
    assert "canary_stop_successful_update" not in loaded.resolved_toml
    state = RawCheckpointState(
        trainer=SingleGpuUpdateState(
            attempted_updates=1, successful_updates=1, effective_samples=800
        ),
        growth=_growth(),
        stage_budget=StageBudgetCheckpointState(
            0, loaded.config.stage.planned_updates
        ),
        checkpoint_cadence=CheckpointCadence(1, 0.0, 100),
    )
    assert (
        resolve_single_gpu_stop_cap_target(loaded.config, state)
        == loaded.config.stage.planned_updates
    )


def test_p25_config_unchanged_no_cap() -> None:
    loaded = load_config(
        Path("train_g1_camera_v2_p25.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    assert loaded.config.stage.canary_stop_successful_update is None
    assert loaded.config.stage.planned_updates == STAGE_TERMINAL


def test_mbs_canary_resolved_planned_updates_stays_168000() -> None:
    loaded = load_config(
        Path("train_g1_camera_v2_p25_mirror_canary.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    assert loaded.config.stage.planned_updates == STAGE_TERMINAL
    assert loaded.config.stage.automatic_transition is False


def test_mbs_canary_stop_cap_resolves_118200_and_cadence_aligned() -> None:
    loaded = load_config(
        Path("train_g1_camera_v2_p25_mirror_canary.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    cap = loaded.config.stage.canary_stop_successful_update
    assert cap == CANARY_STOP
    # THIS canary experiment requires the stop to align to the checkpoint
    # cadence so U118200 is a normal update-cadence checkpoint.
    assert cap % loaded.config.checkpoint.full_every_updates == 0
    state = _state(SOURCE_UPDATE)
    state = RawCheckpointState(
        trainer=state.trainer,
        growth=state.growth,
        stage_budget=state.stage_budget,
        checkpoint_cadence=CheckpointCadence(
            SOURCE_UPDATE, 1_000.0, loaded.config.checkpoint.full_every_updates
        ),
    )
    assert resolve_single_gpu_stop_cap_target(loaded.config, state) == CANARY_STOP
    # and the persisted budget stays the inherited one (no shrink).
    assert (
        state.stage_budget.start_successful_update,
        state.stage_budget.terminal_successful_update,
    ) == (STAGE_START, STAGE_TERMINAL)
