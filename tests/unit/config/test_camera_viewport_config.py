"""Config-layer tests for the shifted-square camera viewport schema.

Covers camera-absent compatibility (no new resolved table bytes), the
effectively-off shapes (absent / disabled-any-p / enabled p=0) against the
camera/spatial mutual exclusion that keys on the single activation
condition, exact canary leaves (p25 and p100), and field validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import RuntimeConfig

CONFIG_ROOT = Path("config")


def _load(name: str, config_root: Path | None = None) -> LoadedConfig:
    return load_config(
        Path(name),
        config_root=config_root or CONFIG_ROOT,
        validate_secrets=False,
    )


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


def _camera_body(
    base: str,
    *,
    enabled: bool,
    probability: str,
    mode: str = "hdm_shifted_square_v2",
    viewport: str = "stage_square",
    offset_distribution: str = "uniform_long_axis_inclusive",
    zoom_source: str = "natural_source_aspect",
) -> str:
    return (
        f'extends = ["{base}"]\n'
        "\n"
        "[data.camera_viewport]\n"
        f"enabled = {str(enabled).lower()}\n"
        f'mode = "{mode}"\n'
        f"probability = {probability}\n"
        f'viewport = "{viewport}"\n'
        f'offset_distribution = "{offset_distribution}"\n'
        f'zoom_source = "{zoom_source}"\n'
    )


def _is_active(config: RuntimeConfig) -> bool:
    camera = config.data.camera_viewport
    return camera is not None and camera.enabled and camera.probability > 0.0


class TestCameraAbsentCompatibility:
    def test_live_g1_parses_without_camera(self) -> None:
        loaded = _load("train_g1.toml")
        assert loaded.config.data.camera_viewport is None
        assert loaded.config.data.spatial_crop.enabled is False

    def test_production_config_parses_without_camera(self) -> None:
        loaded = _load("train_g1_cmuon_production.toml")
        assert loaded.config.data.camera_viewport is None
        assert loaded.config.data.spatial_crop.enabled is True

    def test_absent_camera_emits_no_camera_table_bytes(self) -> None:
        for name in ("train_g1.toml", "train_g1_cmuon_production.toml"):
            loaded = _load(name)
            assert "camera_viewport" not in loaded.resolved_toml
            assert "camera_" not in loaded.resolved_toml


class TestOffShapesParse:
    """Every effectively-off shape parses and is inactive."""

    def test_absent_table_is_inactive(self) -> None:
        loaded = _load("train_g1.toml")
        assert loaded.config.data.camera_viewport is None
        assert not _is_active(loaded.config)

    @pytest.mark.parametrize(
        ("enabled", "probability"),
        [
            (False, "0.0"),
            (False, "1.0"),
            (True, "0.0"),
        ],
    )
    def test_off_shapes_parse_and_are_inactive(
        self, tmp_path: Path, enabled: bool, probability: str
    ) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_off.toml",
            _camera_body(
                "train_g1.toml", enabled=enabled, probability=probability
            ),
        )
        loaded = _load(name, config_root=root.root)
        camera = loaded.config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is enabled
        assert camera.probability == float(probability)
        assert not _is_active(loaded.config)

    def test_active_shape_is_active(self, tmp_path: Path) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_on.toml",
            _camera_body("train_g1.toml", enabled=True, probability="0.25"),
        )
        loaded = _load(name, config_root=root.root)
        assert _is_active(loaded.config)


class TestMutualExclusion:
    """Exclusion keys on the single activation condition, not on the table."""

    def test_active_camera_conflicts_with_spatial(self, tmp_path: Path) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_conflict.toml",
            _camera_body(
                "train_g1_cmuon_production.toml",
                enabled=True,
                probability="0.25",
            ),
        )
        with pytest.raises(ConfigurationError, match="mutually exclusive"):
            _load(name, config_root=root.root)

    @pytest.mark.parametrize(
        ("enabled", "probability"),
        [
            (False, "1.0"),
            (True, "0.0"),
        ],
    )
    def test_off_camera_does_not_block_spatial(
        self, tmp_path: Path, enabled: bool, probability: str
    ) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_off_spatial.toml",
            _camera_body(
                "train_g1_cmuon_production.toml",
                enabled=enabled,
                probability=probability,
            ),
        )
        loaded = _load(name, config_root=root.root)
        assert loaded.config.data.spatial_crop.enabled is True
        assert not _is_active(loaded.config)

    def test_active_camera_with_spatial_disabled_parses(self, tmp_path: Path) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_only.toml",
            _camera_body("train_g1.toml", enabled=True, probability="1.0"),
        )
        loaded = _load(name, config_root=root.root)
        assert _is_active(loaded.config)
        assert loaded.config.data.spatial_crop.enabled is False


class TestCanaryLeaves:
    def test_p100_exact_leaves(self) -> None:
        loaded = _load("train_g1_camera_v2_p100.toml")
        camera = loaded.config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is True
        assert camera.mode == "hdm_shifted_square_v2"
        assert camera.probability == 1.0
        assert camera.viewport == "stage_square"
        assert camera.offset_distribution == "uniform_long_axis_inclusive"
        assert camera.zoom_source == "natural_source_aspect"
        # Low-resolution shifted-square target config: inherits the live G1
        # resolution; no legacy zoom-window or fallback leaves exist.
        assert loaded.config.train.resolution == 256
        legacy = set(camera.model_dump()) & {
            "min_equivalent_zoom",
            "max_equivalent_zoom",
            "fallback_to_aspect_bucket",
        }
        assert legacy == set()

    def test_p25_exact_leaves(self) -> None:
        loaded = _load("train_g1_camera_v2_p25.toml")
        camera = loaded.config.data.camera_viewport
        assert camera is not None
        assert camera.enabled is True
        assert camera.probability == 0.25
        assert camera.mode == "hdm_shifted_square_v2"
        assert loaded.config.train.resolution == 256
        assert _is_active(loaded.config)

    def test_canaries_do_not_override_training_parameters(self) -> None:
        baseline = _load("train_g1.toml").config
        for name in ("train_g1_camera_v2_p100.toml", "train_g1_camera_v2_p25.toml"):
            loaded = _load(name).config
            assert loaded.train == baseline.train
            assert loaded.optimizer == baseline.optimizer
            assert loaded.checkpoint == baseline.checkpoint


class TestFieldValidation:
    def _expect_error(self, tmp_path: Path, body: str, pattern: str) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write("camera_invalid.toml", body)
        with pytest.raises(ConfigurationError, match=pattern):
            _load(name, config_root=root.root)

    def test_probability_above_one(self, tmp_path: Path) -> None:
        self._expect_error(
            tmp_path,
            _camera_body("train_g1.toml", enabled=True, probability="1.1"),
            "probability",
        )

    def test_probability_below_zero(self, tmp_path: Path) -> None:
        self._expect_error(
            tmp_path,
            _camera_body("train_g1.toml", enabled=True, probability="-0.1"),
            "probability",
        )

    def test_toml_integer_probability_is_coerced(self, tmp_path: Path) -> None:
        root = _TmpConfigRoot(tmp_path)
        name = root.write(
            "camera_int_p.toml",
            _camera_body("train_g1.toml", enabled=True, probability="1"),
        )
        loaded = _load(name, config_root=root.root)
        camera = loaded.config.data.camera_viewport
        assert camera is not None
        assert camera.probability == 1.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mode", "hdm_shifted_square_v1"),
            ("viewport", "full_frame"),
            ("offset_distribution", "gaussian"),
            ("zoom_source", "config_window"),
        ],
    )
    def test_unknown_literal_rejected(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        body = _camera_body("train_g1.toml", enabled=True, probability="0.25")
        prefix = field + " = \""
        start = body.index(prefix)
        end = body.index("\n", start + len(prefix))
        body = body[:start] + f'{field} = "{value}"\n' + body[end + 1 :]
        self._expect_error(tmp_path, body, field)
