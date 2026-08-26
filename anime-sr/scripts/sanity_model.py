"""One-shot local sanity: param counts, v-hat=0 at init, smoke spec forward.

Run (Windows, repo root):
    $env:PYTHONPATH = "anime-sr/src"; python -m anime-sr.scripts.sanity_model
    # or: python anime-sr/scripts/sanity_model.py  (with PYTHONPATH set)
"""

from __future__ import annotations

import sys
import time

import torch

from anime_sr.config.schema import ModelSpec
from anime_sr.model.uflow import AnimeSRModel, count_parameters


def main() -> None:
    torch.manual_seed(0)
    model = AnimeSRModel(ModelSpec())
    model.eval()
    n = model.num_params
    print(f"[base] total params: {n:,}  ({n/1e6:.2f}M)")
    print(f"  pixel encoder: {count_parameters(model.pixel_encoder):,}")
    print(f"  trunk:         {count_parameters(model.trunk):,}")
    for name, blocks in model.trunk.stages.items():
        print(f"  stage {name}: depth={len(blocks)} blocks={sum(count_parameters(b) for b in blocks):,}")
    print(f"  conditioner: {count_parameters(model.trunk.conditioner):,}")
    print(f"  output head: {count_parameters(model.trunk.head):,}")

    r_t = torch.randn(1, 128, 64, 64)
    z_lr = torch.randn(1, 128, 64, 64)
    lq = torch.randn(1, 3, 256, 256)
    t = torch.tensor([0.5])
    sigma = torch.tensor([0.0])
    t0 = time.time()
    with torch.no_grad():
        v = model(r_t, z_lr, lq, t, sigma)
    dt = time.time() - t0
    print(f"[base] forward (1,128,64,64): out {tuple(v.shape)}, "
          f"max|v|={v.abs().max().item():.3e}, {dt:.1f}s CPU fp32")

    # smoke spec
    from anime_sr.config.loader import load_config

    cfg = load_config("anime-sr/config/base.toml", "anime-sr/config/smoke.toml")
    smoke = AnimeSRModel(cfg.model)
    n_smoke = smoke.num_params
    print(f"[smoke] total params: {n_smoke:,}  ({n_smoke/1e6:.2f}M)  budget 15-20M")
    with torch.no_grad():
        vs = smoke(r_t, z_lr, lq, t, sigma)
    print(f"[smoke] forward ok, out {tuple(vs.shape)}, finite={bool(torch.isfinite(vs).all())}")


if __name__ == "__main__":
    sys.exit(main())
