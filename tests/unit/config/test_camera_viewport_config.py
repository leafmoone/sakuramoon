"""Config-layer tests for the camera-viewport schema (spec section 23).

Pins old-config compatibility (absent/disabled camera byte-identity),
schema validation, the camera/spatial mutual exclusion, and the exact
leaves of the two canary configs (p25 deployable, p50 LOCKED).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sakuramoon.config import ConfigurationError, load_config

CONFIG_ROOT = Path("config")

# Byte-identity anchor: the resolved production TOML produced by the
# pre-camera code (dev @ 3a341c0). The constant is split so display-layer
# hex masking cannot corrupt this file.
PROD_RESOLVED_SHA256 = (
    "833850a7d63f6e79c76bc46330c5d06be"
    "f285b635a60ea427c17f5ba70accd4b"
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

    def test_production_resolved_toml_is_byte_identical_to_baseline(self) -> None:
        loaded = _load("train_g1_cmuon_production.toml")
        digest = hashlib.sha256(
            loaded.resolved_toml.encode("utf-8")
        ).hexdigest()
        assert digest == PROD_RESOLVED_SHA256

    def test_absent_camera_emits_no_camera_bytes(self) -> None:
        loaded = _load("train_g1_cmuon_production.toml")
        assert "camera_viewport" not in loaded.resolved_toml
        assert "camera_" not in loaded.resolved_toml


class TestValidation:
    CAMERA_TABLE = """
[data.camera_viewport]
enabled = true
mode = "hdm_shifted_square_v2"
probability = 0.25
viewport = "stage_square"
min_equivalent_zoom = 1.10
max_equivalent_zoom = 1.50
offset_distribution = "uniform_long_axis_inclusive"
zoom_source = "natural_source_aspect"
fallback_to_aspect_bucket = true
"""

    def _config(self, tmp_path: Path, body: str) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write("camera_test.toml", body)
        with pytest.raises(ConfigurationError):
            load_config(Path(name), config_root=root.root, validate_secrets=False)

    def test_unknown_field_fails(self, tmp_path: Path) -> None:
        self._config(
            tmp_path,
            'extends = ["train_g1_cmuon_production.toml"]\n'
            "[data.camera_viewport]\n"
            "enabled = true\nmode = \"hdm_shifted_square_v2\"\n"
            "probability = 0.25\nviewport = \"stage_square\"\n"
            "min_equivalent_zoom = 1.10\nmax_equivalent_zoom = 1.50\n"
            "offset_distribution = \"uniform_long_axis_inclusive\"\n"
            "zoom_source = \"natural_source_aspect\"\n"
            "fallback_to_aspect_bucket = true\n"
            "bogus_key = 1\n",
        )

    def test_invalid_probability_fails(self, tmp_path: Path) -> None:
        for bad in ("probability = 1.5", "probability = 1", "probability = -0.1"):
            body = (
                'extends = ["train_g1_cmuon_production.toml"]\n'
                "[data.camera_viewport]\n"
                "enabled = true\nmode = \"hdm_shifted_square_v2\"\n"
                f"{bad}\n"
                "viewport = \"stage_square\"\n"
                "min_equivalent_zoom = 1.10\nmax_equivalent_zoom = 1.50\n"
                "offset_distribution = \"uniform_long_axis_inclusive\"\n"
                "zoom_source = \"natural_source_aspect\"\n"
                "fallback_to_aspect_bucket = true\n"
            )
            self._config(tmp_path, body)

    def test_enabled_with_zero_probability_fails(self, tmp_path: Path) -> None:
        body = (
            'extends = ["train_g1_cmuon_production.toml"]\n'
            "[data.camera_viewport]\n"
            "enabled = true\nmode = \"hdm_shifted_square_v2\"\n"
            "probability = 0.0\n"
            "viewport = \"stage_square\"\n"
            "min_equivalent_zoom = 1.10\nmax_equivalent_zoom = 1.50\n"
            "offset_distribution = \"uniform_long_axis_inclusive\"\n"
            "zoom_source = \"natural_source_aspect\"\n"
            "fallback_to_aspect_bucket = true\n"
        )
        self._config(tmp_path, body)

    def test_disabled_with_positive_probability_fails(self, tmp_path: Path) -> None:
        body = (
            'extends = ["train_g1_cmuon_production.toml"]\n'
            "[data.camera_viewport]\n"
            "enabled = false\nmode = \"hdm_shifted_square_v2\"\n"
            "probability = 0.25\n"
            "viewport = \"stage_square\"\n"
            "min_equivalent_zoom = 1.10\nmax_equivalent_zoom = 1.50\n"
            "offset_distribution = \"uniform_long_axis_inclusive\"\n"
            "zoom_source = \"natural_source_aspect\"\n"
            "fallback_to_aspect_bucket = true\n"
        )
        self._config(tmp_path, body)

    def test_min_gte_max_fails(self, tmp_path: Path) -> None:
        for min_zoom, max_zoom in ((1.5, 1.5), (1.4, 1.2)):
            body = (
                'extends = ["train_g1_cmuon_production.toml"]\n'
                "[data.camera_viewport]\n"
                "enabled = true\nmode = \"hdm_shifted_square_v2\"\n"
                "probability = 0.25\n"
                "viewport = \"stage_square\"\n"
                f"min_equivalent_zoom = {min_zoom}\n"
                f"max_equivalent_zoom = {max_zoom}\n"
                "offset_distribution = \"uniform_long_axis_inclusive\"\n"
                "zoom_source = \"natural_source_aspect\"\n"
                "fallback_to_aspect_bucket = true\n"
            )
            self._config(tmp_path, body)

    def test_max_above_1_5_fails(self, tmp_path: Path) -> None:
        body = (
            'extends = ["train_g1_cmuon_production.toml"]\n'
            "[data.camera_viewport]\n"
            "enabled = true\nmode = \"hdm_shifted_square_v2\"\n"
            "probability = 0.25\n"
            "viewport = \"stage_square\"\n"
            "min_equivalent_zoom = 1.10\nmax_equivalent_zoom = 1.51\n"
            "offset_distribution = \"uniform_long_axis_inclusive\"\n"
            "zoom_source = \"natural_source_aspect\"\n"
            "fallback_to_aspect_bucket = true\n"
        )
        self._config(tmp_path, body)

    def test_camera_and_spatial_both_enabled_fails(self, tmp_path: Path) -> None:
        # The production lineage already enables spatial p50; enabling camera
        # on top of it must be rejected by the mutual-exclusion validator.
        body = (
            'extends = ["train_g1_cmuon_production.toml"]\n'
            "[data.camera_viewport]\n"
            "enabled = true\nmode = \"hdm_shifted_square_v2\"\n"
            "probability = 0.25\n"
            "viewport = \"stage_square\"\n"
            "min_equivalent_zoom = 1.10\nmax_equivalent_zoom = 1.50\n"
            "offset_distribution = \"uniform_long_axis_inclusive\"\n"
            "zoom_source = \"natural_source_aspect\"\n"
            "fallback_to_aspect_bucket = true\n"
        )
        self._config(tmp_path, body)


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
        # Mutual exclusion: the canary lineage disables the inherited
        # production spatial p50 explicitly.
        assert config.data.spatial_crop.enabled is False
        assert config.data.spatial_crop.probability == 0.0
        assert config.run.run_id == "g1_camera_v2_p25"
        assert config.stage.automatic_transition is False

    def test_p50_exact_leaves_and_locked_identity(self) -> None:
        loaded = _load("train_g1_camera_v2_p50.toml")
        config = loaded.config
        camera = config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is True
        assert camera.probability == 0.5
        assert config.run.run_id == "g1_camera_v2_p50"
        assert config.data.spatial_crop.enabled is False
        raw = (CONFIG_ROOT / "train_g1_camera_v2_p50.toml").read_text(
            encoding="utf-8"
        )
        assert "LOCKED" in raw

    def test_p25_and_p50_differ_only_in_probability_and_identity(self) -> None:
        p25 = _load("train_g1_camera_v2_p25.toml")
        p50 = _load("train_g1_camera_v2_p50.toml")
        c25 = p25.config.data.camera_viewport
        c50 = p50.config.data.camera_viewport
        assert c25 is not None and c50 is not None
        assert c25.probability != c50.probability
        dump25 = c25.model_dump()
        dump50 = c50.model_dump()
        dump50["probability"] = dump25["probability"]
        assert dump25 == dump50
        assert p25.config.run.run_id != p50.config.run.run_id
