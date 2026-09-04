"""Standalone camera-control evaluation for the shifted-square camera viewport.

Every point of the fixed manifest shares the identical checkpoint, prompts,
conditioning, initial noise (per-case manifest seeds), sampler profile and
CFG; the ONLY variable is the image-coordinate map, i.e. the exact
full-canvas camera transform of
``sakuramoon.conditioning.camera.transform_camera_coordinates``.

The manifest classes are statements about the hdm_shifted_square_v2 TRAINING
distribution (z in [1.10, 1.50], single-axis inclusive uniform offsets,
coupled shift magnitude <= z**2 - 1). HDM public sources are a mechanism
reference only and are never used as a correctness oracle here.

Usage:
    python -m sakuramoon.cli.camera_eval \\
        --config train_g1_camera_v2_p25.toml --config-root config \\
        --root /sakuramoon-runtime \\
        --checkpoint /sakuramoon-runtime/.../ckpt_N_... \\
        --prompts <canonical PromptManifest path> --count 4 \\
        --output artifacts/camera_eval/g1_camera_v2_p25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

CAMERA_EVAL_MANIFEST_VERSION = 1

CAMERA_EVAL_CLASSES: tuple[str, ...] = (
    "in_distribution_coupled",
    "interpolation",
    "extrapolation",
    "zoom_out_extrapolation",
    "decoupled_shift_extrapolation",
)

CAMERA_EVAL_TRAINING_ZOOM_MIN = 1.10
CAMERA_EVAL_TRAINING_ZOOM_MAX = 1.50

# Coupled anchor zooms of the training matrix (center / left-edge /
# right-edge per anchor). sqrt(2) is the 2:1 source-aspect zoom.
COUPLED_ANCHOR_ZOOMS: tuple[float, ...] = (
    1.10,
    1.225,
    math.sqrt(2.0),
    1.50,
)

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class CameraEvalPoint:
    """One fixed camera-control point (the only variable across the matrix)."""

    name: str
    group: str
    zoom: float
    x_shift: float
    y_shift: float
    manifest_class: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or self.name != self.name.strip():
            raise ValueError("camera eval point name must be a nonempty trimmed string")
        if self.group not in ("legacy", "hdm_demo", "training_coupled", "diagonal"):
            raise ValueError(f"unknown camera eval group: {self.group}")
        for name, value in (
            ("zoom", self.zoom),
            ("x_shift", self.x_shift),
            ("y_shift", self.y_shift),
        ):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"camera eval point {name} must be a finite float")
        if self.manifest_class not in CAMERA_EVAL_CLASSES:
            raise ValueError(f"unknown camera eval class: {self.manifest_class}")
        if self.zoom <= 0.0:
            raise ValueError("camera eval point zoom must be positive")


def classify_camera_eval_point(zoom: float, x_shift: float, y_shift: float) -> str:
    """Fixed manifest class of one camera control point.

    Sign convention: positive x_shift / y_shift means increasing crop
    left / top (the camera transform and the planner agree; see
    ``camera_transform_params``).
    """

    for name, value in (("zoom", zoom), ("x_shift", x_shift), ("y_shift", y_shift)):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"camera eval classification {name} must be a finite float")
    if zoom <= 0.0:
        raise ValueError("camera eval classification zoom must be positive")
    if x_shift == 0.0 and y_shift == 0.0:
        if zoom < 1.0:
            return "zoom_out_extrapolation"
        if zoom < CAMERA_EVAL_TRAINING_ZOOM_MIN:
            return "extrapolation"
        if zoom <= CAMERA_EVAL_TRAINING_ZOOM_MAX + _EPS:
            return (
                "in_distribution_coupled"
                if any(abs(zoom - anchor) <= _EPS for anchor in COUPLED_ANCHOR_ZOOMS)
                else "interpolation"
            )
        return "extrapolation"
    if x_shift != 0.0 and y_shift != 0.0:
        return "decoupled_shift_extrapolation"
    if zoom < CAMERA_EVAL_TRAINING_ZOOM_MIN:
        return "decoupled_shift_extrapolation"
    axis = abs(x_shift if x_shift != 0.0 else y_shift)
    if (
        zoom <= CAMERA_EVAL_TRAINING_ZOOM_MAX + _EPS
        and axis <= zoom * zoom - 1.0 + _EPS
    ):
        return "in_distribution_coupled"
    return "decoupled_shift_extrapolation"


def coupled_edge_shift(zoom: float) -> float:
    """Coupled single-axis shift magnitude at one training zoom (z**2 - 1)."""

    if type(zoom) is not float or not math.isfinite(zoom) or zoom <= 1.0:
        raise ValueError("coupled edge shift requires a zoom above 1.0")
    return zoom * zoom - 1.0


def build_camera_eval_manifest() -> tuple[CameraEvalPoint, ...]:
    """The fixed camera-control matrix (deterministic, versioned).

    LEGACY z=1; HDM DEMO reference grid (x/y pure shifts at z=1 plus the
    HDM reference zooms 0.75 / 1.33, z=1.0 folded into LEGACY);
    TRAINING-COUPLED anchor zooms at center / left-edge / right-edge;
    DIAGONAL generalization (+/-x +/-y at the 1.225 anchor).
    """

    points: list[CameraEvalPoint] = []

    def add(name: str, group: str, zoom: float, x: float, y: float) -> None:
        points.append(
            CameraEvalPoint(
                name=name,
                group=group,
                zoom=zoom,
                x_shift=x,
                y_shift=y,
                manifest_class=classify_camera_eval_point(zoom, x, y),
            )
        )

    add("legacy_center_z1", "legacy", 1.0, 0.0, 0.0)
    for shift in (-0.25, 0.25):
        add(
            f"hdm_demo_x{shift:+.2f}_z1",
            "hdm_demo",
            1.0,
            shift,
            0.0,
        )
    for shift in (-0.25, 0.25):
        add(
            f"hdm_demo_y{shift:+.2f}_z1",
            "hdm_demo",
            1.0,
            0.0,
            shift,
        )
    for zoom in (0.75, 1.33):
        add(f"hdm_demo_z{zoom:.2f}", "hdm_demo", zoom, 0.0, 0.0)
    for zoom in COUPLED_ANCHOR_ZOOMS:
        slug = f"z{zoom:.4f}".rstrip("0")
        edge = coupled_edge_shift(zoom)
        add(f"coupled_{slug}_center", "training_coupled", zoom, 0.0, 0.0)
        add(
            f"coupled_{slug}_left_edge",
            "training_coupled",
            zoom,
            -edge,
            0.0,
        )
        add(
            f"coupled_{slug}_right_edge",
            "training_coupled",
            zoom,
            edge,
            0.0,
        )
    diagonal_zoom = 1.225
    diagonal_edge = coupled_edge_shift(diagonal_zoom)
    for sx in (-diagonal_edge, diagonal_edge):
        for sy in (-diagonal_edge, diagonal_edge):
            add(
                f"diagonal_z{diagonal_zoom:.4f}_x{sx:+.4f}_y{sy:+.4f}",
                "diagonal",
                diagonal_zoom,
                sx,
                sy,
            )
    manifest = tuple(points)
    names = tuple(point.name for point in manifest)
    if len(set(names)) != len(names):
        raise ValueError("camera eval manifest contains duplicate point names")
    return manifest


def _point_output_dir(output: Path, point: CameraEvalPoint) -> Path:
    return output / point.group / point.name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone camera-control evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--points",
        default=None,
        help="comma-separated point-name filter (default: the full manifest)",
    )
    args = parser.parse_args(argv)

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

    # The eval runtime's private helpers are the only correct sampling path
    # (the fixed gallery kernel); reusing them keeps the matrix contract.
    from sakuramoon.cli.generation_eval import (
        _resolve_alpha,  # pyright: ignore[reportPrivateUsage]
    )
    from sakuramoon.conditioning.camera import transform_camera_coordinates
    from sakuramoon.conditioning.rope import image_coordinates
    from sakuramoon.config import load_config
    from sakuramoon.encoders.mage_vae import load_local_mage_vae
    from sakuramoon.encoders.qwen import load_local_qwen
    from sakuramoon.eval.runtime import (
        _generate,  # pyright: ignore[reportPrivateUsage]
        _stage_cases,  # pyright: ignore[reportPrivateUsage]
    )
    from sakuramoon.train.step import TrainableComposite

    root = args.root.resolve(strict=True)
    config_root = (
        args.config_root if args.config_root.is_absolute() else root / args.config_root
    )
    loaded = load_config(args.config, config_root=config_root, validate_secrets=False)
    config = loaded.config
    if config.evaluation.enabled is not True:
        raise ValueError("camera evaluation requires evaluation.enabled=true")
    if args.count <= 0:
        raise ValueError("count must be positive")

    checkpoint = args.checkpoint.resolve(strict=True)
    manifest = read_checkpoint_manifest(checkpoint)
    update = manifest.identity.update
    alpha = _resolve_alpha(checkpoint, None)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "camera evaluation requires exactly one visible CUDA device"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    resolution = config.stage.resolution
    if type(resolution) is not int or resolution <= 0 or resolution % 16:
        raise ValueError("camera evaluation resolution must be a positive multiple of 16")

    composite = load_inference_artifact(checkpoint, manifest.identity, device=device)
    # Training checkpoints always compose a TrainableComposite
    # (build_trainable_composite); the loader's nn.Module return annotation
    # is the shared baseline surface (checkpoint/load.py) and is not
    # widened here, so narrow it locally for _generate.
    composite = cast(TrainableComposite, composite)
    qwen = load_local_qwen(root, device)
    vae = load_local_mage_vae(root, device)

    cases = _stage_cases(args.prompts.resolve(strict=True), args.count, resolution=resolution)
    base_map = image_coordinates(resolution // 16, resolution // 16, device=device)

    points = build_camera_eval_manifest()
    if args.points is not None:
        wanted = tuple(name.strip() for name in args.points.split(",") if name.strip())
        known = tuple(point.name for point in points)
        unknown = tuple(name for name in wanted if name not in known)
        if unknown:
            raise ValueError(f"unknown camera eval points: {unknown}")
        points = tuple(point for point in points if point.name in wanted)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for point in points:
            coordinate_map = transform_camera_coordinates(
                base_map,
                zoom=point.zoom,
                x_shift=point.x_shift,
                y_shift=point.y_shift,
            )
            images = _generate(
                cases,
                config=config,
                evaluation=config.evaluation,
                composite=composite,
                qwen=qwen,
                vae=vae,
                device=device,
                growth_alpha=alpha,
                coordinate_map=coordinate_map,
            )
            point_dir = _point_output_dir(output, point)
            point_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            paths: list[str] = []
            for index, image in enumerate(images.unbind(0)):
                array = image.permute(1, 2, 0).cpu().numpy()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                target = point_dir / f"{index + 1:02d}.png"
                Image.fromarray(array).save(target)  # pyright: ignore[reportUnknownArgumentType]
                paths.append(str(target.relative_to(output)))
            records.append(
                {
                    "name": point.name,
                    "group": point.group,
                    "zoom": point.zoom,
                    "x_shift": point.x_shift,
                    "y_shift": point.y_shift,
                    "manifest_class": point.manifest_class,
                    "outputs": paths,
                }
            )
            print(
                f"[camera-eval] {point.name} class={point.manifest_class} "
                f"z={point.zoom} x={point.x_shift} y={point.y_shift} images={len(paths)}",
                flush=True,
            )

    resolved_hash = hashlib.sha256(
        loaded.resolved_toml.encode("utf-8")
    ).hexdigest()
    document = {
        "manifest_version": CAMERA_EVAL_MANIFEST_VERSION,
        "checkpoint": str(checkpoint),
        "update": update,
        "resolved_config_sha256": resolved_hash,
        "config": str(args.config),
        "prompts": str(args.prompts.resolve()),
        "prompts_count": len(cases),
        "growth_alpha": alpha,
        "fixed": {
            "checkpoint": str(checkpoint),
            "prompts": str(args.prompts.resolve()),
            "initial_noise": "per-case manifest seeds (identical across the matrix)",
            "sampler_profile": config.evaluation.sampling_profile,
            "cfg_scale": config.cfg.scale,
            "only_variable": "image coordinates",
        },
        "sign_convention": "positive x/y shift = increasing crop left/top",
        "points": records,
    }
    (output / "camera_eval_manifest.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[camera-eval] manifest: {output / 'camera_eval_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
