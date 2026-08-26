"""Train (or resume) the M2 pixel baseline (plan §M2, step 7).

Single GPU:

    python -m anime_sr.cli.train_pixel_baseline \
        --config config/base.toml config/data.toml \
        --index-dir <index> --webp-dir <webp> \
        --out-dir output_model/pixel-baseline --bucket-hr 512

Multi-GPU (DDP, one process per GPU):

    torchrun --nproc_per_node=N -m anime_sr.cli.train_pixel_baseline ...

Uncaught exceptions propagate (repo rule: the CLI prints the traceback,
never squashed JSON).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from anime_sr.config.loader import load_config
from anime_sr.train.pixel_baseline import run_pixel_baseline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the M2 pixel baseline.")
    ap.add_argument("--config", nargs="+", required=True, help="TOML config files")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="defaults to [pixel_baseline].out_dir")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--iterations", type=int, default=None, help="override [pixel_baseline].iterations")
    ap.add_argument("--resume", default=None, help="ckpt to resume from (step/model/optimizer)")
    args = ap.parse_args(argv)

    cfg = load_config(*args.config)
    if args.iterations is not None:
        cfg.pixel_baseline.iterations = args.iterations
    out_dir = args.out_dir or cfg.pixel_baseline.out_dir

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise SystemExit("error: WORLD_SIZE > 1 requires CUDA devices")
        dist.init_process_group(backend="nccl")

    run_pixel_baseline(
        cfg,
        index_dir=args.index_dir,
        webp_dir=args.webp_dir,
        out_dir=out_dir,
        bucket_hr=args.bucket_hr,
        rank=rank,
        world_size=world_size,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
