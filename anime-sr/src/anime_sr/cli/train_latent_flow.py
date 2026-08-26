"""M3/M4 latent flow-matching training CLI (plan §13, §14).

Usage (single HCU / multi-HCU DDP via torchrun):

    python -m anime_sr.cli.train_latent_flow \
        --config config/base.toml config/data.toml config/smoke.toml \
        --index-dir ... --webp-dir ... \
        --latent-dir /root/private_data/anime-sr/latents-10k-1024 \
        --vae .../mage-vae.safetensors --out-dir .../output_model/latent-flow-smoke
    torchrun --nproc_per_node=2 -m anime_sr.cli.train_latent_flow ...

Trains the latent-only UFlowSR on the pre-encoded z_hr store: z_lr is
computed on the fly as E_Mage(Bicubic4x(LQ)) (plan §4.3). See
``train/latent_flow.py`` for the M3 checklist mapping.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch.distributed as dist

from anime_sr.config.loader import load_config
from anime_sr.train.latent_flow import run_latent_flow


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AnimeSR-Mage-UFlow M3 latent flow loop")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument(
        "--latent-dir", required=True, help="LatentStore root (index-v1.json + z/)"
    )
    ap.add_argument(
        "--vae", default=None, help="Mage-VAE weights (default: [vae].path)"
    )
    ap.add_argument("--out-dir", default=None, help="default: [latent_flow].out_dir")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument(
        "--no-prefetch", action="store_true", help="disable double-buffered prefetch"
    )
    args = ap.parse_args(argv)

    cfg = load_config(*args.config)
    if args.no_prefetch:
        cfg.latent_flow.prefetch = False
    out_dir = args.out_dir or cfg.latent_flow.out_dir

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    final = run_latent_flow(
        cfg,
        index_dir=args.index_dir,
        webp_dir=args.webp_dir,
        latent_dir=args.latent_dir,
        out_dir=out_dir,
        vae_path=args.vae,
        bucket_hr=args.bucket_hr,
        rank=rank,
        world_size=world_size,
        resume=args.resume,
    )
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    print(f"[train-latent-flow] done: {final}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise
