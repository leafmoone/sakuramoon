"""W&B telemetry facade tests (2026-08-31).

The real wandb SDK is never required here: the enabled path is exercised
through a stub module injected into ``sys.modules`` (the facade only uses
``wandb.init`` / ``run.config.update`` / ``run.log`` / ``run.finish`` /
``wandb.Image``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from anime_sr.config import load_config
from anime_sr.config.schema import Config, LoggingSpec
from anime_sr.train.wandb_log import TrainLogger, tensor_grid

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"


def test_disabled_is_a_noop() -> None:
    lg = TrainLogger(LoggingSpec(), rank=0, run_dir=None)
    assert not lg.enabled
    lg.set_config({"x": 1})
    lg.log(1, scalars={"a": 1.0}, images={"b": torch.rand(2, 3, 8, 8)})
    lg.finish()


def test_rank1_is_never_active() -> None:
    lg = TrainLogger(LoggingSpec(wandb_enabled=True), rank=1, run_dir=None)
    assert not lg.enabled
    lg.log(1, scalars={"a": 1.0})
    lg.finish()


def test_enabled_without_wandb_package_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise
    with pytest.raises(RuntimeError, match="wandb"):
        TrainLogger(LoggingSpec(wandb_enabled=True), rank=0, run_dir=None)


def test_enabled_mode_conflict_fails_loud() -> None:
    with pytest.raises(ValueError, match="contradictory"):
        TrainLogger(
            LoggingSpec(wandb_enabled=True, wandb_mode="disabled"), rank=0, run_dir=None
        )


class _FakeImage:
    def __init__(self, arr: torch.Tensor, caption: str | None = None) -> None:
        self.arr = arr
        self.caption = caption


def _stub_wandb(logged: list) -> Any:
    run: Any = types.SimpleNamespace(
        config=types.SimpleNamespace(
            update=lambda d, **kw: logged.append(("config", dict(d)))
        ),
        log=lambda d: logged.append(("log", dict(d))),
        finish=lambda: logged.append(("finish", None)),
    )
    stub: Any = types.ModuleType("wandb")
    stub.init = lambda **kw: logged.append(("init", kw)) or run
    stub.Settings = object
    stub.Image = _FakeImage
    return stub


def test_enabled_with_stub_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logged: list = []
    monkeypatch.setitem(sys.modules, "wandb", _stub_wandb(logged))
    lg = TrainLogger(
        LoggingSpec(
            wandb_enabled=True,
            wandb_entity="e",
            wandb_project="p",
            wandb_run_name="r",
        ),
        rank=0,
        run_dir=tmp_path,
        tags=["t1", "t2"],
    )
    assert lg.enabled
    lg.set_config({"a": 1})
    lg.log(
        5,
        scalars={"train/loss": 0.5},
        images={"sample/x": torch.rand(3, 3, 4, 4)},
    )
    lg.finish()
    kinds = [k for k, _ in logged]
    assert "init" in kinds and "config" in kinds and "log" in kinds and "finish" in kinds
    init_kwargs = next(v for k, v in logged if k == "init")
    assert init_kwargs["project"] == "p"
    assert init_kwargs["entity"] == "e"
    assert init_kwargs["name"] == "r"
    assert init_kwargs["mode"] == "online"
    assert init_kwargs["dir"] == str(tmp_path)
    assert init_kwargs["tags"] == ["t1", "t2"]
    payload = next(v for k, v in logged if k == "log")
    assert payload["step"] == 5
    assert payload["train/loss"] == 0.5
    assert isinstance(payload["sample/x"], _FakeImage)


def test_tensor_grid_shapes_and_range() -> None:
    fake = types.SimpleNamespace(Image=_FakeImage)
    # 2 samples -> 1 row x 4 cols (white padding fills the rest)
    img = tensor_grid(fake, torch.rand(2, 3, 4, 4) * 2 - 1, caption="c")
    assert img.arr.shape == (3, 4, 16)
    assert img.arr.min() >= 0.0 and img.arr.max() <= 1.0
    assert img.caption == "c"
    # 5 samples -> 2 rows
    img2 = tensor_grid(fake, torch.rand(5, 3, 2, 2))
    assert img2.arr.shape == (3, 4, 8)
    # [3, H, W] single-image input
    img3 = tensor_grid(fake, torch.rand(3, 2, 2))
    assert img3.arr.shape == (3, 2, 8)


def test_config_schema_logging_defaults() -> None:
    cfg = Config()
    assert cfg.logging.wandb_enabled is False
    assert cfg.logging.wandb_project == "anime-sr"
    assert cfg.logging.log_every_steps == 10
    assert cfg.logging.sample_every_steps == 0
    assert cfg.logging.sample_images == 4
    # producer_intra_op_threads default 0 = legacy OMP_NUM_THREADS behavior
    assert cfg.latent_flow.producer_intra_op_threads == 0


def test_srv2_throughput_overlay_intra_op_threads() -> None:
    cfg = load_config(
        CFG / "base.toml",
        CFG / "data.toml",
        CFG / "m4_1024.toml",
        CFG / "srv2_throughput.toml",
    )
    assert cfg.latent_flow.producer_intra_op_threads == 2


def test_m4_config_logging_overlay() -> None:
    cfg = load_config(CFG / "base.toml", CFG / "data.toml", CFG / "m4_1024.toml")
    assert cfg.logging.wandb_enabled is True
    assert cfg.logging.wandb_run_name == "m4-1024-6m"
    assert cfg.logging.log_every_steps == 10
    assert cfg.logging.sample_every_steps == 3125
