# pyright: reportPrivateUsage=false
"""Recovery settings that must be applied before resumed GPU training."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import wandb

from sakuramoon.cli.train import (
    _configure_cuda_allocator,
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


def test_wandb_resume_truncates_to_checkpoint(
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

    assert captured["resume_from"] == "s0-production?_step=1200"
    assert "resume" not in captured
