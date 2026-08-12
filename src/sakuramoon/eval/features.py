"""Shared image decoding and feature extraction for FID, IS, KID, and CMMD."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from PIL import Image
from torch import nn
from torch.nn import functional
from torchvision.models import (  # pyright: ignore[reportMissingTypeStubs]
    Inception_V3_Weights,
    inception_v3,
)
from torchvision.transforms import (  # pyright: ignore[reportMissingTypeStubs]
    functional as vision_functional,
)

from sakuramoon.assets import require_local_clip

FEATURE_CACHE_SCHEMA_VERSION = 2
REAL_PREPROCESSING_ID = "rgb-square-stretch-bilinear-v2"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"
CLIP_INPUT_SIZE = 336
CLIP_EMBEDDING_DIM = 768
INCEPTION_FEATURE_DIM = 2048
INCEPTION_LOGIT_DIM = 1000

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class EvaluationFeatureError(RuntimeError):
    """A validation image or feature-model contract was violated."""


class _FidExtractor(Protocol):
    def __call__(self, images: torch.Tensor) -> list[torch.Tensor]: ...


class _ClipOutput(Protocol):
    image_embeds: torch.Tensor


@dataclass(frozen=True, slots=True)
class ImageFeatureBatch:
    inception_features: torch.Tensor
    clip_features: torch.Tensor
    inception_logits: torch.Tensor

    def __post_init__(self) -> None:
        count = self.inception_features.shape[0]
        if (
            self.inception_features.ndim != 2
            or self.inception_features.shape != (count, INCEPTION_FEATURE_DIM)
            or self.clip_features.shape != (count, CLIP_EMBEDDING_DIM)
            or self.inception_logits.shape != (count, INCEPTION_LOGIT_DIM)
            or count <= 0
        ):
            raise EvaluationFeatureError("image feature batch shapes are invalid")
        for name, values in (
            ("Inception", self.inception_features),
            ("CLIP", self.clip_features),
            ("Inception logits", self.inception_logits),
        ):
            if not torch.is_floating_point(values):
                raise EvaluationFeatureError(f"{name} features must be floating point")
            if not bool(torch.isfinite(values).all().item()):
                raise EvaluationFeatureError(
                    f"{name} features contain nonfinite values"
                )
        clip_norms = torch.linalg.vector_norm(self.clip_features.float(), dim=1)
        if not bool(
            torch.allclose(
                clip_norms,
                torch.ones_like(clip_norms),
                atol=1e-4,
                rtol=1e-4,
            )
        ):
            raise EvaluationFeatureError("CLIP features are not L2-normalized")

    @property
    def count(self) -> int:
        return self.inception_features.shape[0]

    def to(self, device: torch.device | str) -> ImageFeatureBatch:
        return ImageFeatureBatch(
            self.inception_features.to(device=device, non_blocking=True),
            self.clip_features.to(device=device, non_blocking=True),
            self.inception_logits.to(device=device, non_blocking=True),
        )

    def cpu(self) -> ImageFeatureBatch:
        return ImageFeatureBatch(
            self.inception_features.detach().to("cpu").contiguous(),
            self.clip_features.detach().to("cpu").contiguous(),
            self.inception_logits.detach().to("cpu").contiguous(),
        )

    @classmethod
    def concatenate(cls, batches: tuple[ImageFeatureBatch, ...]) -> ImageFeatureBatch:
        if type(batches) is not tuple or not batches:
            raise EvaluationFeatureError("cannot concatenate an empty feature sequence")
        return cls(
            torch.cat(tuple(batch.inception_features for batch in batches)),
            torch.cat(tuple(batch.clip_features for batch in batches)),
            torch.cat(tuple(batch.inception_logits for batch in batches)),
        )


@dataclass(frozen=True, slots=True)
class ValidationImageBatch:
    sample_ids: tuple[str, ...]
    images: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.sample_ids) is not tuple
            or not self.sample_ids
            or len(set(self.sample_ids)) != len(self.sample_ids)
            or any(type(value) is not str or not value for value in self.sample_ids)
        ):
            raise EvaluationFeatureError("validation sample IDs are invalid")
        if (
            self.images.dtype != torch.uint8
            or self.images.ndim != 4
            or self.images.shape[0] != len(self.sample_ids)
            or self.images.shape[1] != 3
            or self.images.shape[2] != self.images.shape[3]
        ):
            raise EvaluationFeatureError("validation image batch is invalid")


class InceptionFeatureModels:
    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("Inception evaluation requires a CUDA device")
        self.device = device
        print(f"[eval] Inception inference device: {device}", flush=True)
        from pytorch_fid.inception import (  # pyright: ignore[reportMissingTypeStubs]
            InceptionV3,
        )

        block = InceptionV3.BLOCK_INDEX_BY_DIM[INCEPTION_FEATURE_DIM]
        self.fid = cast(_FidExtractor, InceptionV3([block]).eval().to(device))
        self.is_weights = Inception_V3_Weights.DEFAULT
        self.classifier = inception_v3(weights=self.is_weights).eval().to(device)
        self.classifier.requires_grad_(False)

    @torch.inference_mode()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        _require_images(images)
        values = images.to(self.device, dtype=torch.float32, non_blocking=True).div(
            255.0
        )
        result = self.fid(values)[0].flatten(1).float()
        if result.shape != (images.shape[0], INCEPTION_FEATURE_DIM):
            raise EvaluationFeatureError("FID Inception returned invalid features")
        return result

    @torch.inference_mode()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        _require_images(images)
        values = self.is_weights.transforms()(images.float().div(255.0)).to(
            self.device, non_blocking=True
        )
        result = self.classifier(values)
        if not isinstance(result, torch.Tensor) or result.shape != (
            images.shape[0],
            INCEPTION_LOGIT_DIM,
        ):
            raise EvaluationFeatureError("Inception classifier returned invalid logits")
        return result.float()


class ClipFeatureModel:
    def __init__(self, repository_root: Path, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("CMMD CLIP evaluation requires a CUDA device")
        model_path = require_local_clip(repository_root)
        print(f"[eval] loading CMMD CLIP from {model_path}", flush=True)
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection.from_pretrained(
            model_path,
            local_files_only=True,
        )
        if (
            model.config.projection_dim != CLIP_EMBEDDING_DIM
            or model.config.image_size != CLIP_INPUT_SIZE
        ):
            raise EvaluationFeatureError("CMMD CLIP configuration is not ViT-L/14@336")
        model.requires_grad_(False)
        self.model = cast(
            nn.Module, model.eval().to(device=device, dtype=torch.float32)
        )
        self.device = device
        self.mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor(_CLIP_STD, dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )

    @torch.inference_mode()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        _require_images(images)
        values = images.to(self.device, dtype=torch.float32, non_blocking=True).div(
            255.0
        )
        values = functional.interpolate(
            values,
            size=(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        values = (values - self.mean) / self.std
        output = cast(_ClipOutput, self.model(pixel_values=values))
        embeddings = output.image_embeds
        if embeddings.shape != (images.shape[0], CLIP_EMBEDDING_DIM):
            raise EvaluationFeatureError("CMMD CLIP returned invalid embeddings")
        return functional.normalize(embeddings.float(), dim=1)


class EvaluationFeatureModels:
    def __init__(
        self,
        repository_root: Path,
        device: torch.device,
    ) -> None:
        self.inception = InceptionFeatureModels(device)
        self.clip = ClipFeatureModel(repository_root, device)

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> ImageFeatureBatch:
        return ImageFeatureBatch(
            self.inception.features(images),
            self.clip.features(images),
            self.inception.logits(images),
        )


def _require_images(images: torch.Tensor) -> None:
    if (
        not isinstance(images, torch.Tensor)
        or images.dtype != torch.uint8
        or images.ndim != 4
        or images.shape[0] <= 0
        or images.shape[1] != 3
        or images.shape[2] <= 0
        or images.shape[3] <= 0
    ):
        raise EvaluationFeatureError("feature images must be uint8 [N,3,H,W]")


def validation_dataset_fingerprint(
    shard_root: Path,
    selection_path: Path,
) -> str:
    if not selection_path.is_file() or selection_path.is_symlink():
        raise EvaluationFeatureError("validation selection is missing or symlinked")
    archives = sorted(shard_root.rglob("*.tar"))
    if not archives:
        raise EvaluationFeatureError(f"validation tar files are absent: {shard_root}")
    digest = hashlib.sha256()
    digest.update(selection_path.read_bytes())
    for archive in archives:
        if not archive.is_file() or archive.is_symlink():
            raise EvaluationFeatureError("validation archive is not a regular file")
        stat = archive.stat()
        digest.update(archive.relative_to(shard_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validation_image_batches(
    root: Path,
    count: int,
    batch_size: int,
    *,
    output_size: int,
) -> tuple[ValidationImageBatch, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("validation image count must be positive")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("validation image batch size must be positive")
    if type(output_size) is not int or output_size <= 0:
        raise ValueError("validation output size must be positive")
    archives = sorted(root.rglob("*.tar"))
    if not archives:
        raise EvaluationFeatureError(f"validation tar files are absent: {root}")

    batches: list[ValidationImageBatch] = []
    current_images: list[torch.Tensor] = []
    current_ids: list[str] = []
    observed = 0
    for archive in archives:
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                if observed >= count:
                    break
                if (
                    not member.isfile()
                    or Path(member.name).suffix.casefold() not in _IMAGE_SUFFIXES
                ):
                    continue
                source = handle.extractfile(member)
                if source is None:
                    raise EvaluationFeatureError(
                        f"validation member cannot be read: {archive}:{member.name}"
                    )
                try:
                    with Image.open(io.BytesIO(source.read())) as image:
                        tensor = vision_functional.pil_to_tensor(image.convert("RGB"))
                except (OSError, ValueError) as error:
                    raise EvaluationFeatureError(
                        f"validation image cannot be decoded: {archive}:{member.name}"
                    ) from error
                tensor = (
                    functional.interpolate(
                        tensor.unsqueeze(0).float(),
                        size=(output_size, output_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .round()
                    .clamp(0.0, 255.0)
                    .to(torch.uint8)
                )
                current_images.append(tensor)
                current_ids.append(
                    f"{archive.relative_to(root).as_posix()}::{member.name}"
                )
                observed += 1
                if len(current_images) == batch_size:
                    batches.append(
                        ValidationImageBatch(
                            tuple(current_ids), torch.stack(current_images)
                        )
                    )
                    current_images = []
                    current_ids = []
            if observed >= count:
                break
    if current_images:
        batches.append(
            ValidationImageBatch(tuple(current_ids), torch.stack(current_images))
        )
    if observed != count:
        raise EvaluationFeatureError(
            f"validation images are incomplete: {observed}/{count}"
        )
    flattened_ids = tuple(
        sample_id for batch in batches for sample_id in batch.sample_ids
    )
    if len(flattened_ids) != count or len(set(flattened_ids)) != count:
        raise EvaluationFeatureError(
            "validation image selection is incomplete or duplicated"
        )
    return tuple(batches)


__all__ = [
    "CLIP_EMBEDDING_DIM",
    "CLIP_INPUT_SIZE",
    "CLIP_MODEL_ID",
    "FEATURE_CACHE_SCHEMA_VERSION",
    "INCEPTION_FEATURE_DIM",
    "INCEPTION_LOGIT_DIM",
    "REAL_PREPROCESSING_ID",
    "EvaluationFeatureError",
    "EvaluationFeatureModels",
    "ImageFeatureBatch",
    "InceptionFeatureModels",
    "ValidationImageBatch",
    "validation_dataset_fingerprint",
    "validation_image_batches",
]
