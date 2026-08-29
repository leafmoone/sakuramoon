"""M3/M4 latent flow-matching training CLI (plan §13, §14).

Usage (single HCU / multi-HCU DDP via torchrun):

    python -m anime_sr.cli.train_latent_flow \
        --config config/base.toml config/data.toml config/smoke.toml \
        --index-dir ... --webp-dir ... \
        --latent-dir /root/private_data/anime-sr/latents-10k-1024 \
        --vae .../mage-vae.safetensors --out-dir .../output_model/latent-flow-smoke
    torchrun --nproc_per_node=2 -m anime_sr.cli.train_latent_flow ...

Trains the latent-only UFlowSR over z_hr (pre-encoded store, or P1 ④
on-the-fly VAE encode via ``[latent_flow].zhr_source = "onfly"``); z_lr
is always computed on the fly as E_Mage(Bicubic4x(LQ)) (plan §4.3). See
``train/latent_flow.py`` for the M3 checklist mapping.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch.distributed as dist

from anime_sr.config.loader import load_config
from anime_sr.train.latent_flow import (
    prepare_producer_prefork,
    run_latent_flow,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AnimeSR-Mage-UFlow M3 latent flow loop")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument(
        "--latent-dir",
        default=None,
        help="LatentStore root (index-v1.json + z/); required unless "
        "[latent_flow].zhr_source = \"onfly\" (P1 ④ on-the-fly z_hr)",
    )
    ap.add_argument(
        "--vae", default=None, help="Mage-VAE weights (default: [vae].path)"
    )
    ap.add_argument("--out-dir", default=None, help="default: [latent_flow].out_dir")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument(
        "--resume",
        default=None,
        help="same-stage full checkpoint to resume from (strict model + "
        "optimizer + EMA/RNG/exposure when present; pixel zero-init is "
        "NEVER re-applied); mutually exclusive with --init-trunk",
    )
    ap.add_argument(
        "--init-trunk",
        default=None,
        help="trunk-only checkpoint for the trunk-only -> pixel-stage "
        "transition (NEW stage: step=0, exposure=0, fresh optimizer, pixel "
        "zero-init when configured); requires "
        "[latent_flow].pixel_features = true; mutually exclusive with --resume",
    )
    ap.add_argument(
        "--stage-transition",
        default=None,
        help="legacy-full -> production-v2 stage transition (M4-1024): "
        "start a NEW long stage from a FULL pixel checkpoint of the "
        "previous stage (e.g. the Phase I-P v1 latest.pt). All trained "
        "weights are retained (the pixel path is NEVER re-zeroed), the "
        "optimizer is inherited when its AdamW state is complete and "
        "compatible (else a FRESH optimizer, reported as such), the EMA is "
        "seeded from the loaded weights, and step/exposure start at 0 "
        "under a fresh scheduler over this run's horizon. A legitimate v2 "
        "production checkpoint (with an EMA section) must use --resume "
        "instead. Mutually exclusive with --resume and --init-trunk.",
    )
    ap.add_argument(
        "--no-prefetch",
        action="store_true",
        help="synchronous data (prefetch_depth=0; M2-style canary)",
    )
    ap.add_argument(
        "--prefetch-depth",
        type=int,
        default=None,
        help="override [latent_flow].prefetch_depth (2=double, 4=quad)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(*args.config)
    if args.no_prefetch:
        cfg.latent_flow.prefetch_depth = 0
    elif args.prefetch_depth is not None:
        cfg.latent_flow.prefetch_depth = args.prefetch_depth
    if cfg.latent_flow.zhr_source == "store" and args.latent_dir is None:
        raise SystemExit("--latent-dir is required for zhr_source=\"store\"")
    if sum(x is not None for x in (args.resume, args.init_trunk, args.stage_transition)) > 1:
        raise SystemExit(
            "--resume / --init-trunk / --stage-transition are mutually "
            "exclusive (same-stage recovery vs trunk->pixel transition vs "
            "legacy-full->v2 stage transition)"
        )
    out_dir = args.out_dir or cfg.latent_flow.out_dir

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        # P1-WEDGE-FIX: fork the CPU producer pool BEFORE NCCL/HCU init so
        # workers never inherit accelerator runtime state (inheriting it
        # makes forked workers SIGSEGV on their first CPU op; observed in
        # the 2-rank smoke: SIGSEGV in torch.randn inside _noise_like).
        if cfg.latent_flow.producer == "process":
            prepare_producer_prefork(
                cfg,
                index_dir=args.index_dir,
                webp_dir=args.webp_dir,
                latent_dir=args.latent_dir,
                bucket_hr=args.bucket_hr,
                rank=rank,
            )
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
        init_trunk=args.init_trunk,
        stage_transition=args.stage_transition,
        config_names=list(args.config),
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
