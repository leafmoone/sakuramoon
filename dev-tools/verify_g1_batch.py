"""Verify the G1 batch-800 config: global_batch validator + LR held flat."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sakuramoon.config import load_config


def main() -> int:
    cfg_path = REPO / "config" / "train_g1.toml"
    cfg_root = REPO / "config"
    loaded = load_config(cfg_path, config_root=cfg_root, validate_secrets=False)
    c = loaded.config
    lr = c.scaled_learning_rate()
    expected_global = c.stage.local_batch * c.stage.accumulation * c.stage.world_size
    print(f"local_batch       = {c.stage.local_batch}")
    print(f"accumulation      = {c.stage.accumulation}")
    print(f"world_size        = {c.stage.world_size}")
    print(f"global_batch      = {c.stage.global_batch}")
    print(f"  expected (l*a*w)= {expected_global}")
    print(f"base_lr           = {c.optimizer.base_lr}")
    print(f"reference_batch   = {c.optimizer.reference_batch}")
    print(f"scaled_learning_rate = {lr!r}")
    assert c.stage.global_batch == expected_global == 800, "global_batch mismatch"
    assert abs(lr - 0.0001484375) < 1e-12, f"LR not held flat: {lr}"
    print("OK: global_batch=800 validated, LR held flat at 0.0001484375")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
