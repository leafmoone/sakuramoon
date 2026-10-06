"""Generate PNGs from local SakuraMoon weights without the training stack."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint root or its model/ directory",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="root containing model/qwen_3.5_2B and model/vae",
    )
    parser.add_argument(
        "--config", type=Path, help="defaults to the checkpoint's resolved_config.toml"
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="natural-language image description")
    prompt.add_argument(
        "--prompt-file", type=Path, help="UTF-8 natural-language description"
    )
    condition = parser.add_mutually_exclusive_group()
    condition.add_argument("--style", help="comma-separated style-reference tags")
    condition.add_argument(
        "--character", help="comma-separated character-identity tags"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="sequential images, seeds seed..seed+N-1",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--profile", help="sampling profile name from the TOML file")
    parser.add_argument(
        "--steps", type=int, help="override the selected profile's step count"
    )
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument(
        "--growth-alpha",
        type=float,
        help="required for in-flight growth without a growth sidecar",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="PyTorch device; DCU also uses cuda:N"
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "flash"),
        default="sdpa",
        help="explicit backend; flash requires FlashAttention-2 (DAS on DCU)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_images <= 0 or not 0 <= args.seed < 2**63 - args.num_images + 1:
        parser.error("num-images must be positive and all seeds must lie in [0, 2**63)")
    if any(size <= 0 or size % 16 for size in (args.width, args.height)):
        parser.error("width and height must be positive multiples of 16")
    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else args.prompt.strip()
    )
    if not prompt:
        parser.error("prompt must not be empty")
    checkpoint = args.checkpoint.resolve(strict=True)
    model_dir = checkpoint / "model" if (checkpoint / "model").is_dir() else checkpoint
    checkpoint_root = model_dir.parent
    config_path = args.config or checkpoint_root / "resolved_config.toml"
    if not config_path.is_file():
        parser.error("no resolved config found; pass --config config/inference.toml")
    outputs = [
        args.output_dir / f"seed-{args.seed + i}-{args.width}x{args.height}.png"
        for i in range(args.num_images)
    ]
    if any(path.exists() or path.with_suffix(".json").exists() for path in outputs):
        parser.error(
            "output already exists; choose a different output directory or seed"
        )

    import torch
    from PIL import Image, PngImagePlugin

    from sakuramoon.checkpoint.load import load_model_directory
    from sakuramoon.data.caption import (
        CaptionPlan,
        ConditionRequest,
        Tag,
        empty_caption_dropout_hits,
    )
    from sakuramoon.encoders.mage_vae import load_local_mage_vae
    from sakuramoon.encoders.qwen import load_local_qwen
    from sakuramoon.inference import generate_image, load_inference_settings
    from sakuramoon.model.attention import configure_attention_backend
    from sakuramoon.train.step import TrainableComposite

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("inference requires a supported NVIDIA CUDA or Hygon HIP device")
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    settings = load_inference_settings(
        config_path,
        profile=args.profile,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    print(
        f"[infer] device={torch.cuda.get_device_name(device)} backend={args.attention_backend} profile={settings.profile.name} NFE={settings.profile.nfe}",
        flush=True,
    )
    print(f"[infer] loading {model_dir}", flush=True)
    module, identity, kind = load_model_directory(model_dir, device=device)
    if not isinstance(module, TrainableComposite):
        raise TypeError("checkpoint does not contain a TrainableComposite")
    alpha = args.growth_alpha
    growth_path = checkpoint_root / "train_state/growth_state.json"
    if alpha is None and growth_path.is_file():
        alpha = json.loads(growth_path.read_text(encoding="utf-8"))["alpha"]
    if alpha is None:
        if module.dit.new_slot_ids:
            raise ValueError(
                "checkpoint has growing slots; provide --growth-alpha or growth_state.json"
            )
        alpha = 1.0
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("growth alpha must be finite and in [0, 1]")
    module.eval().requires_grad_(False)
    configure_attention_backend(
        module, cast(Literal["flash", "sdpa"], args.attention_backend)
    )
    qwen = load_local_qwen(
        args.root.resolve(strict=True), device, attention_backend="sdpa", math_sdpa=True
    )
    vae = load_local_mage_vae(args.root.resolve(strict=True), device)
    condition = None
    condition_text = args.style or args.character
    if condition_text is not None:
        tags = tuple(
            Tag(value.strip(), value.strip())
            for value in condition_text.split(",")
            if value.strip()
        )
        condition = ConditionRequest(
            source="artist_text" if args.style else "character_text",
            role="style" if args.style else "identity",
            tags=tags,
        )
    plan = CaptionPlan(
        tags=(),
        condition=condition,
        nl_text=prompt,
        selected_nl="long_names",
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, output in enumerate(outputs):
        seed = args.seed + i
        pixels = generate_image(
            plan,
            composite=module,
            qwen=qwen,
            vae=vae,
            settings=settings,
            width=args.width,
            height=args.height,
            seed=seed,
            growth_alpha=float(alpha),
            device=device,
        )
        metadata = {
            "prompt": prompt,
            "style": args.style,
            "character": args.character,
            "seed": seed,
            "width": args.width,
            "height": args.height,
            "checkpoint_id": identity.checkpoint_id,
            "checkpoint_update": identity.update,
            "checkpoint_kind": kind.value,
            "growth_alpha": alpha,
            "attention_backend": args.attention_backend,
            "profile": settings.profile.name,
            "solver": settings.profile.solver,
            "steps": settings.profile.steps,
            "nfe": settings.profile.nfe,
            "cfg_scale": settings.cfg_scale,
            "noise_scale": settings.noise_scale,
            "t_eps": settings.t_eps,
            "prediction_type": "x",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
        }
        encoded = json.dumps(metadata, ensure_ascii=False, indent=2)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("sakuramoon", encoded)
        with output.open("xb") as handle:
            pixel_array = cast(
                "NDArray[np.uint8]",
                pixels.permute(1, 2, 0).cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
            )
            Image.fromarray(pixel_array).save(handle, format="PNG", pnginfo=png_info)
        with output.with_suffix(".json").open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        print(f"[infer] {i + 1}/{args.num_images}: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
