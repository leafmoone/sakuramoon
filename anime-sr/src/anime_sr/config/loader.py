"""TOML config loading + resolved-config dump (checkpoint contract §18).

Usage (CLI later, plan §15.2):

    cfg = load_config("config/base.toml", "config/stage1_flow.toml")
    dump_resolved(cfg, out_path)   # json, stored with each checkpoint
"""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

from anime_sr.config.schema import Config


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge ``overlay`` onto ``base`` (dicts merge; lists/scalars replace)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path) -> Config:
    """Load and validate one or more TOML configs (later files win on conflicts)."""
    merged: dict = {}
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("rb") as f:
            merged = _deep_merge(merged, tomllib.load(f))
    cfg = Config.model_validate(merged)
    cfg.validate_all()
    return cfg


def resolve(cfg: Config) -> dict:
    """JSON-serializable resolved config (all defaults filled in)."""
    return cfg.model_dump(mode="json")


def dump_resolved(cfg: Config, out_path: str | Path) -> Path:
    """Write the resolved config as JSON (checkpoint contract §18)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(resolve(cfg), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out_path
