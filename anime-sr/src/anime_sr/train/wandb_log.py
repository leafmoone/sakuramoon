"""Optional W&B telemetry for the latent-flow loop (2026-08-31).

Gated by ``[logging].wandb_enabled`` (config; default OFF). The loop's
console prints are untouched — W&B is strictly additive:

* ``rank 0`` only talks to the server (DDP-safe);
* the API key comes from the ``WANDB_API_KEY`` environment variable
  (repo rule: no secrets in config files); ``wandb_entity`` /
  ``wandb_project`` are plain config values;
* ``wandb_mode``: ``"online"`` (default), ``"offline"`` (air-gapped: run
  files land under ``<out_dir>/wandb``) or ``"disabled"`` (rejected when
  enabled — a config typo guard);
* ``wandb_enabled=true`` but the package is not importable: fail loud at
  init (an operator who asked for W&B must not get a silent no-op).

Every method is a cheap no-op while disabled (no wandb import, no
allocation), so the default-off path adds nothing to the loop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from anime_sr.config.schema import LoggingSpec


def tensor_grid(wandb: Any, t: torch.Tensor, *, caption: str = "") -> Any:
    """``[B, 3, H, W]`` (or ``[3, H, W]``) in [-1, 1] or [0, 1] -> ONE
    wandb.Image grid (4 samples per row, white padding). One image per
    category keeps the W&B UI scannable across many log steps."""
    x = t.detach().float().cpu()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if float(x.min()) < -0.05:
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    b, c, h, w = x.shape
    cols = 4
    rows = math.ceil(b / cols)
    grid = torch.full((rows, cols, c, h, w), 1.0, dtype=torch.float32)
    for i in range(b):
        grid[i // cols, i % cols] = x[i]
    grid = grid.permute(2, 0, 3, 1, 4).reshape(c, rows * h, cols * w)
    return wandb.Image(grid, caption=caption or None)


class TrainLogger:
    """Thin W&B facade for the training loop (see module docstring).

    Construct once per rank (``rank != 0`` is always a no-op instance) and
    call :meth:`log` from the loop / validation probes. The loop never
    checks ``enabled`` itself — the facade absorbs it."""

    def __init__(
        self,
        spec: LoggingSpec,
        *,
        rank: int,
        run_dir: str | Path | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.spec = spec
        self.rank = rank
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.enabled = False
        self._run: Any = None
        self._wandb: Any = None
        self._warned: set[str] = set()
        if spec.wandb_enabled and rank == 0:
            self._init(run_dir, list(tags or []))

    def _init(self, run_dir: str | Path | None, tags: list[str]) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "[logging] wandb_enabled=true but the 'wandb' package is not "
                f"importable ({exc}); install the dependency or set "
                "wandb_enabled=false"
            ) from exc
        if self.spec.wandb_mode == "disabled":
            raise ValueError(
                "[logging] wandb_enabled=true but wandb_mode='disabled' — "
                "contradictory config"
            )
        kwargs: dict[str, Any] = {
            "project": self.spec.wandb_project,
            "name": self.spec.wandb_run_name or None,
            "mode": self.spec.wandb_mode,
        }
        if self.spec.wandb_entity:
            kwargs["entity"] = self.spec.wandb_entity
        if run_dir is not None:
            kwargs["dir"] = str(run_dir)
        if tags:
            kwargs["tags"] = tags
        run = wandb.init(**kwargs)
        if run is None:
            # e.g. WANDB_MODE=disabled in the environment
            print(
                "[wandb] wandb.init returned no run (WANDB_MODE disabled?) — "
                "W&B telemetry stays off",
                flush=True,
            )
            return
        self._run = run
        self._wandb = wandb
        self.enabled = True
        print(
            f"[wandb] run started (project={self.spec.wandb_project} "
            f"mode={self.spec.wandb_mode} dir={run_dir})",
            flush=True,
        )

    def _swallow(self, where: str, exc: Exception) -> None:
        # A W&B transport/serialization failure must NEVER kill training:
        # warn (at most once per exception type), then drop the point.
        key = type(exc).__name__
        if key in self._warned:
            return
        self._warned.add(key)
        print(
            f"[wandb] {where} failed ({key}: {exc}) — dropping this "
            "log point; W&B telemetry degraded, training continues",
            flush=True,
        )

    def set_config(self, cfg: dict[str, Any]) -> None:
        """Structured run config (the resolved-config doc; no secrets —
        it is already written to disk next to the checkpoints)."""
        if self.enabled:
            try:
                self._run.config.update(cfg, allow_val_change=True)
            except Exception as exc:  # noqa: BLE001 - telemetry must not kill training
                self._swallow("config.update", exc)

    def log(
        self,
        step: int,
        *,
        scalars: dict[str, float] | None = None,
        images: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """One log point: arbitrary scalars and/or sample-image grids
        (``[B, 3, H, W]`` each, collapsed to one grid image per name)."""
        if not self.enabled:
            return
        payload: dict[str, Any] = {"step": int(step)}
        for k, v in (scalars or {}).items():
            payload[k] = float(v)
        for k, t in (images or {}).items():
            payload[k] = tensor_grid(self._wandb, t, caption=f"step {step}")
        try:
            self._run.log(payload)
        except Exception as exc:  # noqa: BLE001 - telemetry must not kill training
            self._swallow(f"log(step={step})", exc)

    def finish(self) -> None:
        if self.enabled:
            try:
                self._run.finish()
            except Exception as exc:  # noqa: BLE001 - telemetry must not kill training
                self._swallow("finish", exc)
