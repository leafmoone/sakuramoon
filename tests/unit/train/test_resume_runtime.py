# pyright: reportPrivateUsage=false
"""Recovery settings that must be applied before resumed GPU training."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

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
from sakuramoon.config.assembly import initialize_wandb_run
from sakuramoon.model.growth import active_slot_ids
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


def test_wandb_resume_creates_grouped_continuation_run(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_init(**kwargs: object) -> _FakeRun:
        captured.update(kwargs)
        return _FakeRun()

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(
        "sakuramoon.config.assembly.uuid4",
        lambda: type("_Uuid", (), {"hex": "0123456789abcdef"})(),
    )

    initialize_wandb_run(
        project="sakuramoon",
        entity="owner",
        run_id="s0-production",
        run_directory=tmp_path,
        resume_policy="allow",
        resume_from_update=1200,
    )

    assert captured["id"] == "s0-production-u1200-01234567"
    assert captured["name"] == captured["id"]
    assert captured["group"] == "s0-production"
    assert captured["job_type"] == "train-continuation"
    assert captured["resume"] == "never"
    assert captured["config"] == {
        "source_run_id": "s0-production",
        "resume_from_update": 1200,
    }
