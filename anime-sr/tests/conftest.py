"""Shared fixtures: locate the local Mage-VAE weights (never committed to git)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _find_vae_weights() -> Path | None:
    env = os.environ.get("ANIME_SR_VAE_PATH")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    roots = [
        Path(r"C:\Users\PC\.cache\huggingface\hub\models--mage-flow-community--Mage-Flow\snapshots"),
        Path.home() / ".cache" / "huggingface" / "hub" / "models--mage-flow-community--Mage-Flow" / "snapshots",
    ]
    for root in roots:
        if root.exists():
            hits = sorted(root.glob("*/vae/diffusion_pytorch_model.safetensors"))
            if hits:
                return hits[0]
    return None


VAE_WEIGHTS = _find_vae_weights()

requires_vae = pytest.mark.skipif(
    VAE_WEIGHTS is None,
    reason="Mage-VAE weights not found (set ANIME_SR_VAE_PATH to the .safetensors file)",
)
