# pyright: reportPrivateUsage=false
"""Recovery settings that must be applied before resumed GPU training."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any, cast

import pytest
import wandb

from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.schema import (
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.cli.train import (
    _configure_cuda_allocator,
    _configure_torchinductor_cache,
    _install_shutdown_signal_handlers,
)
from sakuramoon.config import load_config
from sakuramoon.config.assembly import (
    ManagedRemoteRun,
    TrainingTelemetryAssembly,
    initialize_wandb_run,
)
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.telemetry.metrics import DurableJsonlSink
from sakuramoon.telemetry.observer import AsyncTrainingMetricObserver
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.production import _resume_state_for_config
from sakuramoon.train.step import SingleGpuUpdateState

REPOSITORY_ROOT = Path(__file__).parents[3]


class _FakeRun:
    def log(self, data: object, *, step: int) -> None:
        del data, step

    def finish(self, exit_code: int | None = None) -> None:
        del exit_code


def test_cuda_allocator_enables_expandable_segments(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")

    configured = _configure_cuda_allocator()

    assert configured == "max_split_size_mb:128,expandable_segments:True"
    assert os.environ["PYTORCH_ALLOC_CONF"] == configured
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == configured


def test_cuda_allocator_defaults_to_bounded_splitting(monkeypatch: Any) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    configured = _configure_cuda_allocator()

    assert configured == "max_split_size_mb:512,expandable_segments:True"
    assert os.environ["PYTORCH_ALLOC_CONF"] == configured
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == configured


def test_torchinductor_cache_is_bound_inside_project(
    monkeypatch: Any, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

    configured = _configure_torchinductor_cache(project_root)

    expected = project_root / "cache" / "torchinductor"
    assert configured == expected
    assert configured.is_dir()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(expected)
    assert not tuple(configured.glob(".sakuramoon-write-test-*"))


def test_torchinductor_cache_rejects_external_override(
    monkeypatch: Any, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    external = tmp_path / "external-cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(external))

    with pytest.raises(RuntimeError, match="conflicts with the project cache"):
        _configure_torchinductor_cache(project_root)

    assert not (project_root / "cache").exists()


def test_torchinductor_cache_rejects_non_directory(
    monkeypatch: Any, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    cache_parent = project_root / "cache"
    cache_parent.mkdir(parents=True)
    cache_path = cache_parent / "torchinductor"
    cache_path.write_text("not a directory", encoding="utf-8")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

    with pytest.raises(NotADirectoryError, match="not a real directory"):
        _configure_torchinductor_cache(project_root)


def test_background_training_resets_interrupt_and_termination_signals(
    monkeypatch: Any,
) -> None:
    observed: list[tuple[int, object]] = []
    monkeypatch.setattr(
        "sakuramoon.cli.train.signal.signal",
        lambda number, handler: observed.append((number, handler)),
    )

    _install_shutdown_signal_handlers()

    assert observed == [
        (signal.SIGINT, signal.default_int_handler),
        (signal.SIGTERM, signal.default_int_handler),
    ]


def test_resume_rebinds_only_checkpoint_cadence_policy() -> None:
    config = load_config(
        Path("train_s0.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
            "WANDB_API_KEY": "synthetic-wandb-secret",
        },
    ).config
    trainer = SingleGpuUpdateState(
        attempted_updates=28_300,
        successful_updates=28_300,
        effective_samples=13_216_000,
    )
    growth = GrowthCheckpointState(
        active_slot_ids=active_slot_ids(config.stage.depth),
        alpha=1.0,
        stage=config.stage.name,
        world_size=config.stage.world_size,
        resolution=config.stage.resolution,
        ramp_start_successful_update=None,
        ramp_updates=None,
    )
    budget = StageBudgetCheckpointState(
        start_successful_update=0,
        terminal_successful_update=config.stage.planned_updates,
    )
    persisted = RawCheckpointState(
        trainer=trainer,
        growth=growth,
        stage_budget=budget,
        checkpoint_cadence=CheckpointCadence(
            last_successful_update=trainer.successful_updates,
            last_wall_clock_unix_seconds=123.5,
            every_successful_updates=500,
        ),
    )

    resumed = _resume_state_for_config(config, persisted)

    assert resumed.trainer is trainer
    assert resumed.growth is growth
    assert resumed.stage_budget is budget
    assert resumed.checkpoint_cadence == CheckpointCadence(
        last_successful_update=28_300,
        last_wall_clock_unix_seconds=123.5,
        every_successful_updates=100,
    )
    assert (
        resumed.checkpoint_cadence.due(
            successful_update=28_499,
            wall_clock_unix_seconds=124.0,
        )
        is None
    )
    assert (
        resumed.checkpoint_cadence.due(
            successful_update=28_500,
            wall_clock_unix_seconds=124.0,
        )
        is CheckpointReason.UPDATE_CADENCE
    )
    assert config.evaluation.enabled is True
    assert config.evaluation.every_updates == 5_000


def test_wandb_resume_reattaches_fixed_run_id(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_init(**kwargs: object) -> _FakeRun:
        captured.update(kwargs)
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", fake_init)

    initialize_wandb_run(
        project="sakuramoon",
        entity="owner",
        run_id="s0-production",
        run_directory=tmp_path,
        resume_policy="allow",
        resume_from_update=1200,
    )

    assert captured["id"] == "s0-production"
    assert captured["name"] == "s0-production"
    assert captured["group"] == "s0-production"
    assert captured["job_type"] == "train-continuation"
    assert captured["resume"] == "allow"
    assert captured["reinit"] == "create_new"
    assert captured["save_code"] is False
    assert captured["mode"] == "online"
    assert captured["config"] == {"resume_from_update": 1200}


def test_wandb_fresh_start_uses_fixed_run_id(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_init(**kwargs: object) -> _FakeRun:
        captured.update(kwargs)
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", fake_init)

    initialize_wandb_run(
        project="sakuramoon",
        entity="owner",
        run_id="s0-production",
        run_directory=tmp_path,
        resume_policy="allow",
        resume_from_update=None,
    )

    assert captured["id"] == "s0-production"
    assert captured["name"] == "s0-production"
    assert captured["job_type"] == "train"
    assert captured["resume"] == "allow"
    assert "config" not in captured


def test_wandb_fixed_id_collision_falls_back_to_grouped_run(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from wandb.errors import UsageError

    calls: list[dict[str, object]] = []

    def fake_init(**kwargs: object) -> _FakeRun:
        calls.append(dict(kwargs))
        if "id" in kwargs:
            raise UsageError("run id already exists")
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", fake_init)

    initialize_wandb_run(
        project="sakuramoon",
        entity="owner",
        run_id="s0-production",
        run_directory=tmp_path,
        resume_policy="allow",
        resume_from_update=1200,
    )

    assert len(calls) == 2
    assert calls[0]["id"] == "s0-production"
    assert "id" not in calls[1]
    assert calls[1]["name"] == "s0-production"
    assert calls[1]["group"] == "s0-production"


def test_wandb_close_honors_finish_on_close() -> None:
    finished: list[int | None] = []

    class _RecordingRun(_FakeRun):
        def finish(self, exit_code: int | None = None) -> None:
            finished.append(exit_code)

    class _FakeClose:
        def close(self) -> None:
            return None

    open_assembly = TrainingTelemetryAssembly(
        phase_timer=cast(PhaseTimer, _FakeClose()),
        observer=cast(AsyncTrainingMetricObserver, _FakeClose()),
        remote=None,
        local=cast(DurableJsonlSink, _FakeClose()),
        run=cast(ManagedRemoteRun, _RecordingRun()),
        finish_remote_run=False,
    )
    open_assembly.close(exit_code=0)
    assert finished == []

    closed_assembly = TrainingTelemetryAssembly(
        phase_timer=cast(PhaseTimer, _FakeClose()),
        observer=cast(AsyncTrainingMetricObserver, _FakeClose()),
        remote=None,
        local=cast(DurableJsonlSink, _FakeClose()),
        run=cast(ManagedRemoteRun, _RecordingRun()),
        finish_remote_run=True,
    )
    closed_assembly.close(exit_code=1)
    assert finished == [1]
