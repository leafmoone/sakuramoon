"""Run standalone FID/IS/KID/CMMD generation evaluation on a saved checkpoint.

Reuses the online ``TrainingEvaluator`` (generation, feature extraction,
metric computation, and result publication are all the online
implementation); this CLI only adds a checkpoint-only model load and an
explicit update label, so catch-up evaluations for updates that the live
counter skipped (e.g. resume gaps) can be run without touching a training
run.

Usage:
    python -m sakuramoon.cli.generation_eval \
        --config train_g1.toml --config-root config --root /sakuramoon-runtime \
        --checkpoint /sakuramoon-runtime/eval-ckpts/ckpt_61900_...
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone generation evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="complete checkpoint directory (manifest.json + model/ + train_state/)",
    )
    parser.add_argument(
        "--update",
        type=int,
        default=None,
        help="evaluation update label (defaults to the checkpoint manifest update)",
    )
    parser.add_argument(
        "--growth-alpha",
        type=float,
        default=None,
        help="growth alpha (defaults to train_state/growth_state.json when present)",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="override the evaluation output directory (default from config)",
    )
    return parser


def _resolve_alpha(checkpoint: Path, explicit: float | None) -> float:
    if explicit is not None:
        if not 0.0 <= explicit <= 1.0:
            raise ValueError("growth-alpha must be in [0, 1]")
        return explicit
    path = checkpoint / "train_state" / "growth_state.json"
    if path.is_file():
        document = json.loads(path.read_bytes())
        alpha = document.get("alpha")
        if type(alpha) is not float:
            raise ValueError("growth_state.json alpha is invalid")
        return alpha
    raise ValueError(
        "growth alpha is unavailable; pass --growth-alpha for a model-only directory"
    )


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

    from sakuramoon.checkpoint.load import (
        load_inference_artifact,
        read_checkpoint_manifest,
    )
    from sakuramoon.config import load_config
    from sakuramoon.encoders.mage_vae import load_local_mage_vae
    from sakuramoon.encoders.qwen import load_local_qwen
    from sakuramoon.eval.runtime import TrainingEvaluator

    root = args.root.resolve(strict=True)
    config_root = (
        args.config_root if args.config_root.is_absolute() else root / args.config_root
    )
    loaded = load_config(
        args.config,
        config_root=config_root,
        validate_secrets=False,
    )
    config = loaded.config
    if config.evaluation.enabled is not True:
        raise ValueError("generation evaluation requires evaluation.enabled=true")
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest = read_checkpoint_manifest(checkpoint)
    update = args.update if args.update is not None else manifest.identity.update
    if update < 0:
        raise ValueError("update must be nonnegative")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "standalone generation evaluation requires exactly one visible CUDA device"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    if args.output_subdir is not None:
        evaluation = config.evaluation.model_copy(
            update={"output_dir": str(Path(args.output_subdir))}
        )
        config = config.model_copy(update={"evaluation": evaluation})
    output_dir = (
        Path(args.output_subdir)
        if args.output_subdir is not None
        else root / config.evaluation.output_dir
    )
    every = config.evaluation.every_updates
    missing: list[int] = []
    if update % every != 0:
        print(
            f"[gen-eval] note: update={update} is not an online evaluation point "
            f"(every_updates={every}); this run is an explicit catch-up",
            flush=True,
        )
    for online_point in range(every, update + 1, every):
        if not (output_dir / f"step-{online_point}.toml").is_file():
            missing.append(online_point)
    if missing:
        print(
            f"[gen-eval] note: missing online evaluation results for updates: "
            f"{missing}",
            flush=True,
        )

    alpha = _resolve_alpha(checkpoint, args.growth_alpha)
    print(
        f"[gen-eval] loading checkpoint {checkpoint} (update={update}, "
        f"growth_alpha={alpha})",
        flush=True,
    )
    composite = load_inference_artifact(checkpoint, manifest.identity, device=device)
    qwen = load_local_qwen(root, device)
    vae = load_local_mage_vae(root, device)

    evaluator = TrainingEvaluator(
        config,
        repository_root=root,
        composite=composite,
        qwen=qwen,
        vae=vae,
        device=device,
        growth_alpha=alpha,
    )
    result = evaluator.evaluate(update, force=True)
    if result is None:
        raise RuntimeError("standalone evaluation returned no result")
    print(f"[gen-eval] result: {result.result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
