"""Strict checks for locally prepared frozen model directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QWEN_MODEL_PATH = Path("model/qwen_3.5_2B")
VAE_MODEL_PATH = Path("model/vae")
CLIP_MODEL_PATH = Path("model/clip-vit-large-patch14-336")

_QWEN_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors",
)
_VAE_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
)
_CLIP_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
)


@dataclass(frozen=True)
class LocalModelPaths:
    qwen: Path
    vae: Path
    clip: Path


def _require_local_component(
    repository_root: Path,
    model_path: Path,
    required_files: tuple[str, ...],
    component: str,
) -> Path:
    missing = [
        model_path / name
        for name in required_files
        if not (repository_root / model_path / name).is_file()
    ]
    if missing:
        rendered = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(
            f"required local {component} files are missing: {rendered}"
        )
    return repository_root / model_path


def require_local_qwen(repository_root: Path) -> Path:
    """Return the fixed Qwen path without requiring the VAE files."""

    return _require_local_component(
        repository_root,
        QWEN_MODEL_PATH,
        _QWEN_REQUIRED_FILES,
        "Qwen",
    )


def require_local_vae(repository_root: Path) -> Path:
    """Return the fixed VAE path without requiring the Qwen files."""

    return _require_local_component(
        repository_root,
        VAE_MODEL_PATH,
        _VAE_REQUIRED_FILES,
        "Mage-VAE",
    )


def require_local_clip(repository_root: Path) -> Path:
    """Return the fixed CMMD CLIP path or fail before evaluation starts."""

    return _require_local_component(
        repository_root,
        CLIP_MODEL_PATH,
        _CLIP_REQUIRED_FILES,
        "CLIP ViT-L/14@336",
    )


def require_local_models(repository_root: Path) -> LocalModelPaths:
    """Return fixed local model paths, or fail when a required file is absent."""

    return LocalModelPaths(
        qwen=require_local_qwen(repository_root),
        vae=require_local_vae(repository_root),
        clip=require_local_clip(repository_root),
    )


__all__ = [
    "CLIP_MODEL_PATH",
    "QWEN_MODEL_PATH",
    "VAE_MODEL_PATH",
    "LocalModelPaths",
    "require_local_clip",
    "require_local_models",
    "require_local_qwen",
    "require_local_vae",
]
