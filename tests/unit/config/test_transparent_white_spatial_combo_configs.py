"""The transparent-white + shifted-bucket combo canary configs load with
both data policies enabled and the spatial parameters pinned per canary."""

from __future__ import annotations

from pathlib import Path

import pytest

from sakuramoon.config import load_config

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"


@pytest.mark.parametrize(
    ("name", "probability"),
    (
        ("train_g1_transparent_white_spatial_p25", 0.25),
        ("train_g1_transparent_white_spatial_p50", 0.5),
    ),
)
def test_combo_config_enables_both_policies(name: str, probability: float) -> None:
    loaded = load_config(
        CONFIG_ROOT / f"{name}.toml", config_root=CONFIG_ROOT, validate_secrets=False
    )
    data = loaded.config.data
    assert data.spatial_crop.enabled is True
    assert data.spatial_crop.probability == probability
    assert data.spatial_crop.min_equivalent_zoom == 1.02
    assert data.spatial_crop.max_equivalent_zoom == 1.10
    assert data.spatial_crop.zoom_distribution == "sqrt_uniform_high"
    assert data.spatial_crop.offset_distribution == "uniform_independent"
    assert data.spatial_crop.fallback_to_aspect_bucket is True
    assert data.transparent_background.enabled is True
    # The resolved document carries both policy sections (B4 envelope roots).
    assert "[data.spatial_crop]" in loaded.resolved_toml
    assert "[data.transparent_background]" in loaded.resolved_toml
