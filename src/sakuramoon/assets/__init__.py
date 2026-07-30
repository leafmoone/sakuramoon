"""Minimal checks for the two locally prepared model directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QWEN_MODEL_PATH = Path("model/qwen_3.5_2B")
VAE_MODEL_PATH = Path("model/vae")

_REQUIRED_FILES = (
    QWEN_MODEL_PATH / "config.json",
    QWEN_MODEL_PATH / "tokenizer.json",
    QWEN_MODEL_PATH / "model.safetensors",
    VAE_MODEL_PATH / "config.json",
    VAE_MODEL_PATH / "diffusion_pytorch_model.safetensors",
)


@dataclass(frozen=True)
class LocalModelPaths:
    qwen: Path
    vae: Path


def require_local_models(repository_root: Path) -> LocalModelPaths:
    """Return fixed local model paths, or fail when a required file is absent."""

    missing = [path for path in _REQUIRED_FILES if not (repository_root / path).is_file()]
    if missing:
        rendered = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"required local model files are missing: {rendered}")
    return LocalModelPaths(
        qwen=repository_root / QWEN_MODEL_PATH,
        vae=repository_root / VAE_MODEL_PATH,
    )


__all__ = [
    "QWEN_MODEL_PATH",
    "VAE_MODEL_PATH",
    "LocalModelPaths",
    "require_local_models",
]
