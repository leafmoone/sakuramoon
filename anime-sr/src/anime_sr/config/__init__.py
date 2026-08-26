"""Frozen config schema + loader (repo rule: 训练参数只从 config/*.toml 读取)."""

from anime_sr.config.loader import dump_resolved, load_config, resolve
from anime_sr.config.schema import Config

__all__ = ["Config", "dump_resolved", "load_config", "resolve"]
