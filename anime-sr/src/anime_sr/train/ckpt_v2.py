"""Production checkpoint schema v2 (P2-prep, for M4 launches).

v1 (``_save_ckpt`` in :mod:`anime_sr.train.pixel_baseline`) stores
``{step, model, optimizer}`` only.  v2 adds, as *optional* sections:

* ``ema``        -- ``SampleEMA.state_dict()`` (fp32 shadow + decay config)
* ``scalars``    -- windowed scalars at save time (loss, lr, data_wait, ...)
* ``rng``        -- reproducible-resume RNG states (cpu / cuda / numpy)
* ``exposure``   -- deterministic-schedule cursor (index / cycle / per_cycle)
* ``provenance`` -- git commit, config file name, source checkpoint, torch
                    version, platform, UTC timestamp

Rules honoured here:

* **No project-level hashing** (repo rule): provenance carries plain
  identifiers only -- no config/weight digests.
* **v1 forward-compat**: a v2 file loads through the v1 loader (the extra
  keys are ignored) and through :func:`load_v2`; a v1 file loads through
  :func:`load_v2` with ``legacy=True`` and the new sections as ``None``.
* ``torch.load(..., weights_only=False)`` is explicit: v2 payloads embed
  numpy RNG state (a non-tensor pickled object).  The remote DTK torch
  (>= 2.6) defaults ``weights_only=True`` and would reject it; v1 payloads
  are tensor-only and load fine either way.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from anime_sr.train.ema_sample import SampleEMA

__all__ = ["CKPT_VERSION", "load_v2", "restore_rng", "save_v2", "snapshot_rng"]

CKPT_VERSION = 2


# ----------------------------------------------------------------------
def _unwrap(model: nn.Module) -> nn.Module:
    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        return model.module
    return model


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic; overwrites an existing file (unlike Path.rename on Windows)


# ----------------------------------------------------------------------
def snapshot_rng() -> dict:
    """Capture cpu / cuda / numpy RNG states (all optional, None if absent)."""
    st: dict = {
        "cpu": torch.get_rng_state().clone(),
        "cuda": None,
        "numpy": np.random.get_state(),  # fresh (name, int-array, pos) tuple
    }
    if torch.cuda.is_available():
        st["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return st


def restore_rng(rng: dict | None) -> None:
    """Restore :func:`snapshot_rng` output (missing keys are skipped).

    Device-safe: the resume path loads checkpoints with
    ``map_location=<accelerator>`` (weights must land on-device), which moves
    the stored CPU RNG state there too — but the CPU generator setter requires
    a CPU ByteTensor (crash: "RNG state must be a torch.ByteTensor", M4
    canary Leg B on HCU, 08-30).  Re-home each state to the device its setter
    expects; ``set_rng_state_all`` wants CPU states (it moves them itself)."""
    if not rng:
        return
    if "cpu" in rng and rng["cpu"] is not None:
        torch.set_rng_state(rng["cpu"].cpu().clone())
    if "cuda" in rng and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu().clone() for s in rng["cuda"]])
    if "numpy" in rng and rng["numpy"] is not None:
        np.random.set_state(rng["numpy"])


# ----------------------------------------------------------------------
def save_v2(
    path: str | Path,
    *,
    step: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ema: SampleEMA | None = None,
    scalars: dict | None = None,
    exposure: dict | None = None,
    provenance: dict | None = None,
    capture_rng: bool = True,
) -> Path:
    """Write an atomic v2 checkpoint.  See module docstring for sections."""
    payload: dict = {
        "version": CKPT_VERSION,
        "step": int(step),
        "model": _unwrap(model).state_dict(),
        "optimizer": opt.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "scalars": dict(scalars) if scalars else None,
        "rng": snapshot_rng() if capture_rng else None,
        "exposure": dict(exposure) if exposure else None,
        "provenance": dict(provenance) if provenance else None,
    }
    out = Path(path)
    _atomic_save(payload, out)
    return out


def load_v2(
    path: str | Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ema: SampleEMA | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Load a v2 (or v1 legacy) checkpoint into ``model``/``opt``.

    Returns metadata:
    ``{"step": int, "legacy": bool, "scalars": ..., "exposure": ...,
    "provenance": ..., "rng": ...}`` -- restore the RNG with
    :func:`restore_rng` when the resume must be bit-reproducible.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    legacy = int(payload.get("version", 1)) != CKPT_VERSION
    step = int(payload["step"])
    _unwrap(model).load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    ema_sd = payload.get("ema")
    if ema is not None:
        if ema_sd is None:
            raise ValueError("checkpoint has no EMA section but an EMA instance was passed")
        ema.load_state_dict(ema_sd)
    return {
        "step": step,
        "legacy": legacy,
        "scalars": payload.get("scalars"),
        "exposure": payload.get("exposure"),
        "provenance": payload.get("provenance"),
        "rng": payload.get("rng"),
    }


# ----------------------------------------------------------------------
def make_provenance(*, git_commit: str | None = None, config: str | None = None,
                    source_ckpt: str | None = None, platform: str | None = None) -> dict:
    """Plain-identifier provenance block (no hashing, per repo rule)."""
    return {
        "git_commit": git_commit,
        "config": config,
        "source_ckpt": source_ckpt,
        "torch_version": torch.__version__,
        "platform": platform,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
