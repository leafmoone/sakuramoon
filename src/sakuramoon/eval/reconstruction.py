"""Deterministic Mage-VAE reconstruction evaluation on validation images."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import tomli_w
import torch
from PIL import Image

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.eval.features import (
    REAL_PREPROCESSING_ID,
    InceptionFeatureModels,
    validation_dataset_fingerprint,
    validation_image_batches,
)
from sakuramoon.eval.metrics import FeatureStats, frechet_inception_distance
from sakuramoon.storage import repository_directory


class ReconstructionEvaluationError(RuntimeError):
    """The VAE reconstruction experiment violated a strict contract."""


@dataclass(frozen=True, slots=True)
class ReconstructionEvaluationResult:
    sample_count: int
    resolution: int
    reconstruction_fid: float
    lpips: float
    psnr_db: float
    ms_ssim: float
    real_real_fid: float
    result_path: Path
    comparison_grid_path: Path


def _write_toml(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(tomli_w.dumps(payload).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_rgb(image: torch.Tensor, path: Path) -> None:
    if image.dtype != torch.uint8 or image.shape[0] != 3 or image.ndim != 3:
        raise ReconstructionEvaluationError("comparison image is not uint8 RGB")
    array = image.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(array).save(path)


def _comparison_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    path: Path,
) -> None:
    if originals.shape != reconstructions.shape or originals.ndim != 4:
        raise ReconstructionEvaluationError("comparison grid image tensors differ")
    count, channels, height, width = originals.shape
    if count <= 0 or channels != 3:
        raise ReconstructionEvaluationError("comparison grid is empty")
    columns = 4
    rows = (count + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width * 2, rows * height))
    for index in range(count):
        row, column = divmod(index, columns)
        for side, values in enumerate((originals[index], reconstructions[index])):
            array = values.permute(1, 2, 0).contiguous().numpy()
            canvas.paste(
                Image.fromarray(array), ((column * 2 + side) * width, row * height)
            )
    canvas.save(path)


@torch.inference_mode()
def evaluate_vae_reconstruction(
    config: RuntimeConfig,
    *,
    repository_root: Path,
    sample_count: int,
    batch_size: int,
    comparison_count: int,
    output_subdir: str,
    device: torch.device,
) -> ReconstructionEvaluationResult:
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("reconstruction sample count must be at least two")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("reconstruction batch size must be positive")
    if (
        type(comparison_count) is not int
        or comparison_count <= 0
        or comparison_count > sample_count
    ):
        raise ValueError("reconstruction comparison count is invalid")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("VAE reconstruction evaluation requires CUDA")
    resolution = config.stage.resolution
    if resolution % 16:
        raise ValueError("VAE reconstruction resolution must be divisible by 16")

    output_root = repository_directory(repository_root, output_subdir)
    experiment_root = output_root / f"n{sample_count}-r{resolution}"
    experiment_root.mkdir(parents=True, exist_ok=False)
    validation_root = repository_root / config.evaluation.validation_shard_root
    selection_path = repository_root / config.data.validation.selection_path
    dataset_fingerprint = validation_dataset_fingerprint(
        validation_root, selection_path
    )
    batches = validation_image_batches(
        validation_root,
        sample_count * 2,
        batch_size,
        output_size=resolution,
    )
    all_ids = tuple(sample_id for batch in batches for sample_id in batch.sample_ids)
    all_images = torch.cat(tuple(batch.images for batch in batches))
    if all_images.shape != (sample_count * 2, 3, resolution, resolution):
        raise ReconstructionEvaluationError(
            "selected reconstruction images are invalid"
        )
    if len(set(all_ids)) != sample_count * 2:
        raise ReconstructionEvaluationError("reconstruction sample IDs are duplicated")
    original_images = all_images[:sample_count]
    second_real_images = all_images[sample_count:]

    print("[vae-eval] loading Mage VAE", flush=True)
    vae = load_local_mage_vae(repository_root, device)
    print("[vae-eval] loading LPIPS AlexNet", flush=True)
    import lpips  # pyright: ignore[reportMissingTypeStubs]
    from pytorch_msssim import ms_ssim  # pyright: ignore[reportMissingTypeStubs]

    lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    lpips_model.requires_grad_(False)
    inception = InceptionFeatureModels(device)

    original_features: list[torch.Tensor] = []
    reconstructed_features: list[torch.Tensor] = []
    second_real_features: list[torch.Tensor] = []
    reconstruction_images: list[torch.Tensor] = []
    squared_error_sum = torch.zeros((), dtype=torch.float64, device=device)
    element_count = 0
    lpips_sum = torch.zeros((), dtype=torch.float64, device=device)
    ms_ssim_sum = torch.zeros((), dtype=torch.float64, device=device)

    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        original_uint8 = original_images[start:stop]
        second_real_uint8 = second_real_images[start:stop]
        original_01 = original_uint8.to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div(255.0)
        original_m11 = original_01.mul(2.0).sub(1.0)
        latent = vae.encode(original_m11.to(torch.bfloat16))
        decoded_m11 = vae.decode(latent).float()
        reconstructed_01 = decoded_m11.add(1.0).mul(0.5).clamp(0.0, 1.0)
        reconstructed_m11 = reconstructed_01.mul(2.0).sub(1.0)
        reconstructed_uint8 = (
            reconstructed_01.mul(255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8)
        )
        if reconstructed_uint8.shape != original_uint8.shape:
            raise ReconstructionEvaluationError("VAE reconstruction shape differs")
        if not bool(torch.isfinite(reconstructed_01).all().item()):
            raise ReconstructionEvaluationError("VAE reconstruction is nonfinite")

        difference = reconstructed_01 - original_01
        squared_error_sum += difference.double().square().sum()
        element_count += difference.numel()
        lpips_values = cast(
            torch.Tensor,
            lpips_model(original_m11, reconstructed_m11),
        ).flatten()
        ms_ssim_values = cast(
            torch.Tensor,
            ms_ssim(
                original_01,
                reconstructed_01,
                data_range=1.0,
                size_average=False,
            ),
        ).flatten()
        if lpips_values.shape != (stop - start,) or ms_ssim_values.shape != (
            stop - start,
        ):
            raise ReconstructionEvaluationError("paired metric batch shape differs")
        if not bool(torch.isfinite(lpips_values).all().item()) or not bool(
            torch.isfinite(ms_ssim_values).all().item()
        ):
            raise ReconstructionEvaluationError(
                "paired reconstruction metric is nonfinite"
            )
        lpips_sum += lpips_values.double().sum()
        ms_ssim_sum += ms_ssim_values.double().sum()
        original_features.append(inception.features(original_uint8))
        reconstructed_features.append(inception.features(reconstructed_uint8))
        second_real_features.append(inception.features(second_real_uint8))
        reconstruction_images.append(reconstructed_uint8.cpu())
        print(f"[vae-eval] processed {stop}/{sample_count}", flush=True)

    if element_count != sample_count * 3 * resolution * resolution:
        raise ReconstructionEvaluationError("PSNR element count differs")
    reconstructed_images_cpu = torch.cat(reconstruction_images)
    if reconstructed_images_cpu.shape != original_images.shape:
        raise ReconstructionEvaluationError("reconstruction output count differs")
    original_feature_values = torch.cat(original_features)
    reconstructed_feature_values = torch.cat(reconstructed_features)
    second_real_feature_values = torch.cat(second_real_features)
    original_stats = FeatureStats.from_features(original_feature_values, device=device)
    reconstruction_stats = FeatureStats.from_features(
        reconstructed_feature_values, device=device
    )
    second_real_stats = FeatureStats.from_features(
        second_real_feature_values, device=device
    )
    reconstruction_fid = frechet_inception_distance(
        reconstruction_stats, original_stats, device=device
    )
    real_real_fid = frechet_inception_distance(
        second_real_stats, original_stats, device=device
    )
    mean_squared_error = float((squared_error_sum / element_count).item())
    if not math.isfinite(mean_squared_error) or mean_squared_error <= 0.0:
        raise ReconstructionEvaluationError("reconstruction MSE is invalid")
    psnr_db = 10.0 * math.log10(1.0 / mean_squared_error)
    lpips_mean = float((lpips_sum / sample_count).item())
    ms_ssim_mean = float((ms_ssim_sum / sample_count).item())
    if not all(math.isfinite(value) for value in (psnr_db, lpips_mean, ms_ssim_mean)):
        raise ReconstructionEvaluationError(
            "aggregate reconstruction metric is nonfinite"
        )

    comparison_originals = original_images[:comparison_count]
    comparison_reconstructions = reconstructed_images_cpu[:comparison_count]
    for index in range(comparison_count):
        _save_rgb(
            comparison_originals[index],
            experiment_root / f"{index + 1:03d}-original.png",
        )
        _save_rgb(
            comparison_reconstructions[index],
            experiment_root / f"{index + 1:03d}-reconstruction.png",
        )
    comparison_grid_path = experiment_root / "comparison-grid.png"
    _comparison_grid(
        comparison_originals,
        comparison_reconstructions,
        comparison_grid_path,
    )
    metadata_path = experiment_root / "metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "preprocessing": REAL_PREPROCESSING_ID,
            "sample_count": sample_count,
            "resolution": resolution,
            "reconstruction_sample_ids": list(all_ids[:sample_count]),
            "real_baseline_sample_ids": list(all_ids[sample_count:]),
        },
    )
    result_path = experiment_root / "result.toml"
    _write_toml(
        result_path,
        {
            "schema_version": 1,
            "sample_count": sample_count,
            "resolution": resolution,
            "batch_size": batch_size,
            "comparison_count": comparison_count,
            "dataset_fingerprint": dataset_fingerprint,
            "preprocessing": REAL_PREPROCESSING_ID,
            "reconstruction_fid": reconstruction_fid,
            "lpips_alex": lpips_mean,
            "psnr_db": psnr_db,
            "ms_ssim": ms_ssim_mean,
            "real_real_fid": real_real_fid,
        },
    )
    print(
        f"[vae-eval] complete: reconstruction_FID={reconstruction_fid:.4f}, "
        f"LPIPS={lpips_mean:.6f}, PSNR={psnr_db:.4f} dB, "
        f"MS-SSIM={ms_ssim_mean:.6f}, real-real_FID={real_real_fid:.4f}",
        flush=True,
    )
    return ReconstructionEvaluationResult(
        sample_count,
        resolution,
        reconstruction_fid,
        lpips_mean,
        psnr_db,
        ms_ssim_mean,
        real_real_fid,
        result_path,
        comparison_grid_path,
    )


__all__ = [
    "ReconstructionEvaluationError",
    "ReconstructionEvaluationResult",
    "evaluate_vae_reconstruction",
]
