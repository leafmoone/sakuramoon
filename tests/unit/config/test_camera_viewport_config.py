"""Config-layer tests for the camera-viewport schema.

Pins old-config compatibility (absent/disabled camera byte-identity),
schema validation, the camera/spatial mutual exclusion, and the exact
leaves of the two canary configs (p25 and p100).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sakuramoon.config import ConfigurationError, load_config

CONFIG_ROOT = Path("config")

# Byte-identity anchors: the resolved TOMLs produced by the pre-camera dev
# code (dev @ 100d768e4). Adding the optional camera_viewport field must not
# change the resolved output of any camera-absent config (exclude_none).
# Constants are split so display-layer hex masking cannot corrupt this file.
LIVE_G1_RESOLVED_SHA256 = (
    "f060224f73fd2cab335d3b9aa38a1455b36943867604b31367e"
    "9e096b04d501e"
)
PRODUCTION_RESOLVED_SHA256 = (
    "8b34b09ca853cd82f9167649c28392a19fe17b7c59672b8fe1448f4"
    "cb8789004"
)


def _load(name: str):
    return load_config(Path(name), config_root=CONFIG_ROOT, validate_secrets=False)


class _TmpConfigRoot:
    """A config root mirroring the real one, for synthetic test configs."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "config"
        self.root.mkdir(exist_ok=True)
        for item in CONFIG_ROOT.glob("*.toml"):
            target = self.root / item.name
            if not target.exists():
                target.symlink_to(item.resolve())

    def write(self, name: str, text: str) -> str:
        (self.root / name).write_text(text, encoding="utf-8")
        return name


class TestOldConfigCompatibility:
    def test_production_config_parses_without_camera(self) -> None:
        loaded = _load("train_g1_cmuon_production.toml")
        assert loaded.config.data.camera_viewport is None
        assert loaded.config.data.spatial_crop.enabled is True

    def test_live_g1_parses_without_camera(self) -> None:
        loaded = _load("train_g1.toml")
        assert loaded.config.data.camera_viewport is None
        assert loaded.config.data.spatial_crop.enabled is False

    def test_live_g1_resolved_toml_is_byte_identical_to_baseline(self) -> None:
        loaded = _load("train_g1.toml")
        digest = hashlib.sha256(loaded.resolved_toml.encode("utf-8")).hexdigest()
        assert digest == LIVE_G1_RESOLVED_SHA256

    def test_production_resolved_toml_is_byte_identical_to_baseline(self) -> None:
        loaded = _load("train_g1_cmuon_production.toml")
        digest = hashlib.sha256(loaded.resolved_toml.encode("utf-8")).hexdigest()
        assert digest == PRODUCTION_RESOLVED_SHA256

    def test_absent_camera_emits_no_camera_bytes(self) -> None:
        for name in ("train_g1.toml", "train_g1_cmuon_production.toml"):
            loaded = _load(name)
            assert "camera_viewport" not in loaded.resolved_toml
            assert "camera_" not in loaded.resolved_toml


class TestValidation:
    def _config(self, tmp_path: Path, body: str) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write("camera_test.toml", body)
        with pytest.raises(ConfigurationError):
            load_config(Path(name), config_root=root.root, validate_secrets=False)

    def _camera_body(self, base: str = "train_g1_cmuon_production.toml", **overrides: object) -> str:
        table = (
            "[data.camera_viewport]\n"
            f"enabled = {overrides.get('enabled', 'true')}\n"
            "mode = \"hdm_shifted_square_v2\"\n"
            f"probability = {overrides.get('probability', '0.25')}\n"
            "viewport = \"stage_square\"\n"
            f"min_equivalent_zoom = {overrides.get('min_zoom', '1.10')}\n"
            f"max_equivalent_zoom = {overrides.get('max_zoom', '1.50')}\n"
            "offset_distribution = \"uniform_long_axis_inclusive\"\n"
            "zoom_source = \"natural_source_aspect\"\n"
            "fallback_to_aspect_bucket = true\n"
            f"{overrides.get('extra', '')}"
        )
        return f'extends = ["{base}"]\n' + table

    def test_unknown_field_fails(self, tmp_path: Path) -> None:
        self._config(
            tmp_path,
            self._camera_body(extra="bogus_key = 1\n"),
        )

    def test_invalid_probability_fails(self, tmp_path: Path) -> None:
        for bad in ("1.5", "-0.1"):
            self._config(
                tmp_path,
                self._camera_body(probability=bad),
            )

    def test_enabled_with_zero_probability_fails(self, tmp_path: Path) -> None:
        self._config(tmp_path, self._camera_body(probability="0.0"))

    def test_disabled_with_positive_probability_fails(self, tmp_path: Path) -> None:
        self._config(
            tmp_path,
            self._camera_body(enabled="false", probability="0.25"),
        )

    def test_disabled_with_zero_probability_parses(self, tmp_path: Path) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_off.toml",
            self._camera_body(
                base="train_g1.toml", enabled="false", probability="0.0"
            ),
        )
        loaded = load_config(
            Path(name), config_root=root.root, validate_secrets=False
        )
        camera = loaded.config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is False
        assert camera.probability == 0.0

    def test_min_gte_max_fails(self, tmp_path: Path) -> None:
        for min_zoom, max_zoom in (("1.50", "1.50"), ("1.40", "1.20")):
            self._config(
                tmp_path,
                self._camera_body(min_zoom=min_zoom, max_zoom=max_zoom),
            )

    def test_min_below_one_fails(self, tmp_path: Path) -> None:
        self._config(tmp_path, self._camera_body(min_zoom="1.00"))

    def test_max_above_1_5_fails(self, tmp_path: Path) -> None:
        self._config(tmp_path, self._camera_body(max_zoom="1.51"))

    def test_camera_and_spatial_both_enabled_fails(self, tmp_path: Path) -> None:
        # The production lineage already enables spatial p50; enabling camera
        # on top of it must be rejected by the mutual-exclusion validator.
        self._config(tmp_path, self._camera_body())


class TestCanaryConfigs:
    def test_p25_exact_leaves(self) -> None:
        loaded = _load("train_g1_camera_v2_p25.toml")
        config = loaded.config
        camera = config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is True
        assert camera.mode == "hdm_shifted_square_v2"
        assert camera.probability == 0.25
        assert camera.viewport == "stage_square"
        assert camera.min_equivalent_zoom == 1.10
        assert camera.max_equivalent_zoom == 1.50
        assert camera.offset_distribution == "uniform_long_axis_inclusive"
        assert camera.zoom_source == "natural_source_aspect"
        assert camera.fallback_to_aspect_bucket is True
        # First-version mutual exclusion: the live G1 base inherits spatial
        # disabled, so the canary needs no spatial override.
        assert config.data.spatial_crop.enabled is False
        assert config.data.spatial_crop.probability == 0.0
        assert config.run.run_id == "g1_camera_v2_p25"
        assert config.paths.checkpoint_dir == "output_model/g1_camera_v2_p25"

    def test_p100_exact_leaves(self) -> None:
        loaded = _load("train_g1_camera_v2_p100.toml")
        config = loaded.config
        camera = config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is True
        assert camera.probability == 1.0
        assert config.run.run_id == "g1_camera_v2_p100"
        assert config.data.spatial_crop.enabled is False
        assert config.paths.checkpoint_dir == "output_model/g1_camera_v2_p100"

    def test_p25_and_p100_differ_only_in_probability_and_identity(self) -> None:
        p25 = _load("train_g1_camera_v2_p25.toml")
        p100 = _load("train_g1_camera_v2_p100.toml")
        c25 = p25.config.data.camera_viewport
        c100 = p100.config.data.camera_viewport
        assert c25 is not None and c100 is not None
        assert c25.probability != c100.probability
        dump25 = c25.model_dump()
        dump100 = c100.model_dump()
        dump100["probability"] = dump25["probability"]
        assert dump25 == dump100
        assert p25.config.run.run_id != p100.config.run.run_id
