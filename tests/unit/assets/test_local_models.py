from __future__ import annotations

from pathlib import Path

import pytest

from sakuramoon.assets import (
    QWEN_MODEL_PATH,
    VAE_MODEL_PATH,
    require_local_models,
)


def _create_required_files(root: Path) -> None:
    for path in (
        QWEN_MODEL_PATH / "config.json",
        QWEN_MODEL_PATH / "tokenizer.json",
        QWEN_MODEL_PATH / "model.safetensors",
        VAE_MODEL_PATH / "config.json",
        VAE_MODEL_PATH / "diffusion_pytorch_model.safetensors",
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def test_require_local_models_returns_fixed_paths(tmp_path: Path) -> None:
    _create_required_files(tmp_path)

    models = require_local_models(tmp_path)

    assert models.qwen == tmp_path / "model/qwen_3.5_2B"
    assert models.vae == tmp_path / "model/vae"


def test_require_local_models_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model/qwen_3.5_2B/config.json"):
        require_local_models(tmp_path)


def test_prepared_repository_models_pass_minimal_check() -> None:
    repository_root = Path(__file__).parents[3]

    require_local_models(repository_root)
