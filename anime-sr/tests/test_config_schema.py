"""Config schema + loader tests (plan §21 frozen values; repo rule: params live in TOML)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from anime_sr.config import dump_resolved, load_config, resolve
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config"


def test_base_loads_with_frozen_values() -> None:
    cfg = load_config(CFG / "base.toml")
    assert cfg.project.scale == 4
    assert cfg.model.latent_channels == 128
    assert cfg.model.latent_downsample == 16
    grids = [s.grid for s in cfg.model.uflow.stages]
    assert grids == [64, 32, 16, 32, 64]
    assert cfg.model.uflow.stages[2].attention == "global"
    assert cfg.flow.pred == "v"
    assert cfg.flow.solver_4step == "heun"
    assert cfg.buckets.lq_sizes == [128, 192, 256]
    assert cfg.inference.tile_lq == 256 and cfg.inference.tile_overlap_hr == 256


def test_stage1_overlay() -> None:
    cfg = load_config(CFG / "base.toml", CFG / "stage1_flow.toml")
    assert cfg.optimizer.lr == 0.00015
    assert cfg.optimizer.betas == [0.9, 0.95]
    assert len(cfg.phase1.curriculum) == 4
    assert abs(sum(p.fraction for p in cfg.phase1.curriculum) - 1.0) < 1e-9
    assert cfg.phase1.final_256_extra_exposures == 500_000
    assert cfg.hardware.target_latent_tokens_phase1 == 131_072
    assert cfg.ema.half_life_samples == 500_000


def test_stage2_overlay() -> None:
    cfg = load_config(CFG / "base.toml", CFG / "stage2_faithful.toml")
    assert cfg.phase2.loss.edge == 0.10 and cfg.phase2.loss.flat == 0.05
    assert cfg.phase2.loss.pixel_charbonnier_eps == 1e-3
    assert cfg.phase2.batch_mix == {"random_t_flow": 0.5, "one_step_decode": 0.5}
    assert cfg.phase2.optimizer_lrs["vae_base"] == 0.0
    assert cfg.hardware.target_latent_tokens_phase2 == 65_536


def test_smoke_overlay() -> None:
    cfg = load_config(CFG / "base.toml", CFG / "smoke.toml")
    dims = [s.dim for s in cfg.model.uflow.stages]
    assert dims == [192, 256, 384, 256, 192]
    assert cfg.model.param_budget_m == [15.0, 20.0]
    assert cfg.model.output_head.dim_in == 192
    assert cfg.phase1.exposure_target == 200_000


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[model]\nunknown_key = 1\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_curriculum_fraction_guard() -> None:
    cfg = load_config(CFG / "base.toml")
    cfg.phase1.curriculum[0].fraction = 0.5  # break the sum-to-1 invariant
    with pytest.raises(ValueError):
        cfg.validate_all()


def test_resolved_dump(tmp_path: Path) -> None:
    cfg = load_config(CFG / "base.toml", CFG / "stage1_flow.toml")
    out = dump_resolved(cfg, tmp_path / "resolved.json")
    text = json.loads(out.read_text(encoding="utf-8"))
    assert text["model"]["latent_channels"] == 128
    # resolve() output must be plain JSON types
    assert isinstance(resolve(cfg)["flow"]["time_points"], list)
