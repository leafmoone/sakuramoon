# pyright: reportPrivateUsage=false
"""Recovery settings that must be applied before resumed GPU training."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import pytest
import wandb

from sakuramoon.cli.train import (
    _configure_cuda_allocator,
    _configure_torchinductor_cache,
    _install_shutdown_signal_handlers,
)
from sakuramoon.config.assembly import initialize_wandb_run


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
