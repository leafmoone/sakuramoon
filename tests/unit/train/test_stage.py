from __future__ import annotations

from pathlib import Path

import pytest

from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointReason,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train.stage import (
    ForcedCheckpoint,
    GrowthProgress,
    StageReadiness,
    StageTransitionRequest,
    checkpoint_reason,
    transition_checkpoint_state,
)
from sakuramoon.train.step import SingleGpuUpdateState


def _growth(
    depth: int,
    stage: str,
    world_size: int,
    resolution: int,
    *,
    alpha: float = 1.0,
    ramp_start: int | None = None,
    ramp_updates: int | None = None,
) -> GrowthCheckpointState:
    return GrowthCheckpointState(
        active_slot_ids(depth),
        alpha,
        stage,
        world_size,
        resolution,
        ramp_start,
        ramp_updates,
    )


def _request(root: Path, **overrides: object) -> StageTransitionRequest:
    checkpoint = root / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    values: dict[str, object] = {
        "source_stage": "S1",
        "target_stage": "G1",
        "source_checkpoint": checkpoint,
        "source_checkpoint_id": "source",
        "planned_updates": 75_000,
        "manual_approval": True,
    }
    values.update(overrides)
    return StageTransitionRequest(**values)  # pyright: ignore[reportArgumentType]


def _state(
    trainer: SingleGpuUpdateState,
    growth: GrowthCheckpointState,
) -> RawCheckpointState:
    start = growth.ramp_start_successful_update
    if start is None:
        start = 0
    terminal = max(trainer.successful_updates + 1, start + (growth.ramp_updates or 1))
    return RawCheckpointState(
        trainer=trainer,
        growth=growth,
        stage_budget=StageBudgetCheckpointState(start, terminal),
        checkpoint_cadence=CheckpointCadence(
            trainer.successful_updates,
            1_800_000_000.0 + trainer.successful_updates,
        ),
    )


def _transition_cadence(source: RawCheckpointState) -> CheckpointCadence:
    return CheckpointCadence(
        source.trainer.successful_updates,
        source.checkpoint_cadence.last_wall_clock_unix_seconds + 1.0,
    )


def test_transition_requires_unique_predecessor_and_manual_approval(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert request.is_growth
    assert request.ramp_updates == 1500
    assert request.forced_checkpoints() == (
        ForcedCheckpoint.PRE_TRANSITION,
        ForcedCheckpoint.POST_TRANSITION,
        ForcedCheckpoint.RAMP_MIDPOINT,
        ForcedCheckpoint.RAMP_END,
    )
    with pytest.raises(ValueError, match="unique predecessor"):
        _request(tmp_path, source_stage="S0")
    with pytest.raises(ValueError, match="manual approval"):
        _request(tmp_path, manual_approval=False)
    with pytest.raises(ValueError, match="H1/H2"):
        _request(tmp_path, source_stage="S3", target_stage="H1")


@pytest.mark.parametrize(
    ("forced", "expected"),
    [
        (ForcedCheckpoint.PRE_TRANSITION, CheckpointReason.PRE_TRANSITION),
        (ForcedCheckpoint.POST_TRANSITION, CheckpointReason.POST_TRANSITION),
        (ForcedCheckpoint.RAMP_MIDPOINT, CheckpointReason.RAMP_MIDPOINT),
        (ForcedCheckpoint.RAMP_END, CheckpointReason.RAMP_END),
    ],
)
def test_transition_checkpoint_reason_mapping_is_exhaustive(
    forced: ForcedCheckpoint,
    expected: CheckpointReason,
) -> None:
    assert checkpoint_reason(forced) is expected


def test_transition_checkpoint_reason_rejects_equal_string_foreign_enum() -> None:
    with pytest.raises(TypeError, match="T043 enum"):
        checkpoint_reason(
            CheckpointReason.RAMP_END  # pyright: ignore[reportArgumentType]
        )


def test_transition_preserves_global_counters_without_data_state(tmp_path: Path) -> None:
    source = _state(
        SingleGpuUpdateState(2001, 2000, 8000),
        _growth(16, "S1", 4, 256),
    )
    cadence = _transition_cadence(source)
    target = transition_checkpoint_state(
        source, _request(tmp_path), checkpoint_cadence=cadence
    )
    assert target.trainer == source.trainer
    assert target.growth == _growth(
        20, "G1", 4, 256, alpha=0.0, ramp_start=2000, ramp_updates=1500
    )
    assert target.stage_budget == StageBudgetCheckpointState(2000, 77_000)
    assert target.checkpoint_cadence == cadence


def test_transition_rejects_incomplete_growth_source(tmp_path: Path) -> None:
    source = _state(
        SingleGpuUpdateState.initial(),
        _growth(
            16, "S1", 4, 256, alpha=0.0, ramp_start=0, ramp_updates=1000
        ),
    )
    with pytest.raises(ValueError, match="completed growth ramp"):
        transition_checkpoint_state(
            source, _request(tmp_path), checkpoint_cadence=_transition_cadence(source)
        )


@pytest.mark.parametrize(
    ("request_overrides", "growth"),
    [
        ({}, _growth(16, "S0", 1, 256)),
        (
            {"source_stage": "S2", "target_stage": "G2"},
            _growth(
                20,
                "G1",
                4,
                256,
                alpha=1.0,
                ramp_start=1,
                ramp_updates=1000,
            ),
        ),
    ],
)
def test_transition_rejects_caller_relabelled_predecessor_stage(
    tmp_path: Path,
    request_overrides: dict[str, object],
    growth: GrowthCheckpointState,
) -> None:
    successful = 1001 if growth.ramp_updates is not None else 1
    source = _state(SingleGpuUpdateState(successful, successful, 4), growth)
    with pytest.raises(ValueError, match="axes do not match"):
        transition_checkpoint_state(
            source,
            _request(tmp_path, **request_overrides),
            checkpoint_cadence=_transition_cadence(source),
        )


def test_growth_progress_uses_successful_updates_and_forced_points() -> None:
    progress = GrowthProgress(start_successful_update=10_000, ramp_updates=1000)
    assert progress.alpha(10_000) == 0.0
    assert progress.alpha(10_500) == pytest.approx(0.5)
    assert progress.alpha(11_000) == 1.0
    assert progress.forced_checkpoint(10_000) is ForcedCheckpoint.POST_TRANSITION
    assert progress.forced_checkpoint(10_500) is ForcedCheckpoint.RAMP_MIDPOINT
    assert progress.forced_checkpoint(11_000) is ForcedCheckpoint.RAMP_END
    assert progress.forced_checkpoint(10_499) is None
    with pytest.raises(ValueError, match="precedes"):
        progress.alpha(9999)
    state = _state(
        SingleGpuUpdateState(10_500, 10_500, 1),
        _growth(
            20,
            "G1",
            4,
            256,
            alpha=progress.alpha(10_500),
            ramp_start=10_000,
            ramp_updates=1000,
        ),
    )
    assert GrowthProgress.from_checkpoint(state) == progress


def test_stage_readiness_reports_without_transition_side_effects() -> None:
    ready = StageReadiness(True, True, True, True, True)
    blocked = StageReadiness(True, True, True, False, True)
    assert ready.stage_ready is True
    assert blocked.stage_ready is False
    with pytest.raises(TypeError, match="explicit booleans"):
        StageReadiness(True, True, True, 1, True)  # pyright: ignore[reportArgumentType]
