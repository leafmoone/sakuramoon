"""Run deterministic Mage-VAE reconstruction metrics on a validation subset."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mage-VAE reconstructions")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--comparison-count", type=int, default=16)
    parser.add_argument(
        "--output-subdir",
        default="output_model/evaluation/vae-reconstruction",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current = os.environ.get("PYTORCH_ALLOC_CONF", "")
    options = [
        value
        for value in current.split(",")
        if value and not value.startswith("expandable_segments:")
    ]
    options.append("expandable_segments:True")
    os.environ["PYTORCH_ALLOC_CONF"] = ",".join(options)

    import torch

    from sakuramoon.config import load_config
    from sakuramoon.eval.reconstruction import evaluate_vae_reconstruction

    root = args.root.resolve(strict=True)
    config_root = (
        args.config_root if args.config_root.is_absolute() else root / args.config_root
    )
    loaded = load_config(
        args.config,
        config_root=config_root,
        validate_secrets=False,
    )
    if loaded.config.evaluation.enabled is not True:
        raise ValueError("VAE reconstruction requires evaluation.enabled=true")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("VAE reconstruction requires exactly one visible CUDA device")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    result = evaluate_vae_reconstruction(
        loaded.config,
        repository_root=root,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        comparison_count=args.comparison_count,
        output_subdir=args.output_subdir,
        device=device,
    )
    print(f"[vae-eval] result: {result.result_path}", flush=True)
    print(f"[vae-eval] comparison: {result.comparison_grid_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
