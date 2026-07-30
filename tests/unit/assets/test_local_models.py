from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
import torch

from sakuramoon.assets import (
    QWEN_MODEL_PATH,
    VAE_MODEL_PATH,
    require_local_models,
    require_local_qwen,
    require_local_vae,
)
from sakuramoon.encoders import mage_vae, qwen


def _create_qwen_files(root: Path) -> None:
    for path in (
        QWEN_MODEL_PATH / "config.json",
        QWEN_MODEL_PATH / "tokenizer.json",
        QWEN_MODEL_PATH / "model.safetensors",
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def _create_vae_files(root: Path) -> None:
    for path in (
        VAE_MODEL_PATH / "config.json",
        VAE_MODEL_PATH / "diffusion_pytorch_model.safetensors",
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def test_require_local_models_returns_fixed_paths(tmp_path: Path) -> None:
    _create_qwen_files(tmp_path)
    _create_vae_files(tmp_path)

    models = require_local_models(tmp_path)

    assert models.qwen == tmp_path / "model/qwen_3.5_2B"
    assert models.vae == tmp_path / "model/vae"


def test_require_local_models_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model/qwen_3.5_2B/config.json"):
        require_local_models(tmp_path)


def test_qwen_check_does_not_require_vae_files(tmp_path: Path) -> None:
    _create_qwen_files(tmp_path)

    assert require_local_qwen(tmp_path) == tmp_path / QWEN_MODEL_PATH


def test_vae_check_does_not_require_qwen_files(tmp_path: Path) -> None:
    _create_vae_files(tmp_path)

    assert require_local_vae(tmp_path) == tmp_path / VAE_MODEL_PATH


def test_component_checks_report_only_their_own_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required local Qwen files.*qwen_3.5_2B"):
        require_local_qwen(tmp_path)
    with pytest.raises(FileNotFoundError, match="required local Mage-VAE files.*model/vae"):
        require_local_vae(tmp_path)


def test_qwen_loader_with_qwen_only_reaches_kernel_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_qwen_files(tmp_path)
    monkeypatch.setattr(qwen, "is_fast_path_available", False)

    with pytest.raises(RuntimeError, match="fast linear-attention kernels"):
        qwen.load_local_qwen(tmp_path, torch.device("cuda"))


def test_vae_loader_with_vae_only_reaches_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_vae_files(tmp_path)

    def construction_reached(checkpoint: Path) -> NoReturn:
        assert checkpoint == tmp_path / VAE_MODEL_PATH / "diffusion_pytorch_model.safetensors"
        raise RuntimeError("VAE construction reached")

    monkeypatch.setattr(mage_vae, "MageVAE", construction_reached)

    with pytest.raises(RuntimeError, match="VAE construction reached"):
        mage_vae.load_local_mage_vae(tmp_path, torch.device("cuda"))


def test_prepared_repository_models_pass_minimal_check() -> None:
    repository_root = Path(__file__).parents[3]

    assert require_local_qwen(repository_root) == repository_root / QWEN_MODEL_PATH
    assert require_local_vae(repository_root) == repository_root / VAE_MODEL_PATH
    require_local_models(repository_root)
