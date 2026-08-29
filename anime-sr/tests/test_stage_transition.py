"""M4-1024 stage-transition tests (``--stage-transition``).

Contract under test (2026-08-29 M4 work order, legacy-full -> M4-v2):

* strict FULL load from the previous stage's checkpoint — every trained
  weight retained, the pixel path is NEVER re-zeroed;
* a dead (all-zero) pixel-injection source is refused; a v2 production
  checkpoint WITH an EMA section is refused (that is a ``--resume`` source);
  a trunk-only file cannot fill the strict full load;
* optimizer: inherited when the explicit compatibility gate passes
  (complete shape-exact AdamW moments + identical betas/eps/wd), else a
  FRESH optimizer, recorded as such in the transition meta;
* provenance meta: transition tag, source SHA256, n_model_tensors,
  source_step, EMA seeding tag;
* the caller re-seeds a fresh ``SampleEMA`` from the loaded weights
  (shadow == source weights, ``n_samples_total == 0``);
* the fresh M4 scheduler starts warmup at step 0 (monotone non-decreasing
  through warmup — no LR jump — cosine to the floor), and inherited Adam
  moments under the fresh schedule produce finite updates (no NaN);
* milestone exposure arithmetic: the m4_1024 / m4_1024_canary configs land
  exactly on integer steps at the production global batch;
* CLI mutex: two of ``--resume`` / ``--init-trunk`` / ``--stage-transition``
  is a hard error;
* ``stamp_launch_decision`` writes a resolved-config.json carrying the
  2026-08-29 M4 resolution revision.

CPU-safe: smoke-sized model, no VAE weights, no HCU. The 596/596 production
tensor count is asserted against the smoke model's own count here and
verified on the real Phase I-P checkpoint by ``tools/verify_m4_canary.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from anime_sr.config.loader import load_config
from anime_sr.config.schema import Config
from anime_sr.model.uflow import AnimeSRModel
from anime_sr.train.ckpt_v2 import CKPT_VERSION
from anime_sr.train.ema_sample import SampleEMA
from anime_sr.train.latent_flow import (
    _PIXEL_ALIVE_KEYS,
    M4_LAUNCH_DECISION,
    _apply_stage_transition,
    _optimizer_state_compatible,
    _sha256_file,
    stamp_launch_decision,
)
from anime_sr.train.pixel_baseline import _cosine_lr, _optimizer_for

CFG_DIR = Path(__file__).resolve().parent.parent / "config"
DEVICE = torch.device("cpu")
M4_TOTAL_STEPS = 375_000  # 6,000,000 exposures / global batch 16
M4_BASE_LR = 0.00015
M4_WARM = 11_250  # floor(0.03 * 375,000)


def _cfg() -> Config:
    return load_config(str(CFG_DIR / "base.toml"), str(CFG_DIR / "smoke.toml"))


def _cfg_optimizer(**updates: object) -> Config:
    cfg = _cfg()
    return cfg.model_copy(update={"optimizer": cfg.optimizer.model_copy(update=updates)})


def _new_model(cfg: Config) -> AnimeSRModel:
    torch.manual_seed(0)
    return AnimeSRModel(cfg.model, zero_init_pixel=cfg.model.zero_init_pixel).to(DEVICE)


def _trained_model(cfg: Config) -> AnimeSRModel:
    """A 'previous stage' model: fresh init + a few REAL optimization steps
    (finite by construction; the zero-init pixel path becomes non-zero as
    gradients flow — exactly like the end of a real stage)."""
    m = _new_model(cfg)
    opt = _optimizer_for(cfg, m)
    _warm_steps(m, opt, 3)
    for k in _PIXEL_ALIVE_KEYS:
        assert float(m.state_dict()[k].abs().max()) > 0.0, f"pixel path still zero: {k}"
    return m


def _v1_ckpt(
    path: Path,
    cfg: Config,
    *,
    step: int = 18_750,
    pixel_zero: bool = False,
    trunk_only: bool = False,
    optimizer_cfg: Config | None = None,
    drop_optimizer: bool = False,
) -> Path:
    """Phase I-P v1 payload: {step, model, optimizer} (no version key)."""
    src = _new_model(cfg)
    opt = _optimizer_for(optimizer_cfg or cfg, src)
    # Phase I-P ran 18,750 steps: finite trained weights + complete AdamW state
    _warm_steps(src, opt, 3)
    sd = src.state_dict()
    if trunk_only:
        # drop the whole pixel path (both the trunk injection 1x1s and the
        # pixel_encoder block) -> a trunk-only payload
        sd = {
            k: v
            for k, v in sd.items()
            if "pixel_encoder" not in k and "proj_p" not in k and "gap_proj" not in k
        }
    if pixel_zero:
        sd = {k: torch.zeros_like(v) if k in _PIXEL_ALIVE_KEYS else v for k, v in sd.items()}
    payload: dict = {"step": step, "model": sd}
    if not drop_optimizer:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, path)
    return path


# ----------------------------------------------------------------------
# strict full load + provenance meta
# ----------------------------------------------------------------------
def test_transition_strict_full_load_and_meta(tmp_path: Path) -> None:
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "phase1p.pt", cfg, step=18_750)
    sd_src = torch.load(ck, map_location="cpu", weights_only=False)["model"]

    m = _new_model(cfg)
    opt = _optimizer_for(cfg, m)
    meta = _apply_stage_transition(m, opt, ck, DEVICE, 0, True, cfg)

    after = m.state_dict()
    assert set(after) == set(sd_src), "strict full load must keep exactly the source keys"
    for k, v in sd_src.items():
        assert torch.equal(after[k], v), f"tensor not carried bit-exact: {k}"
    assert meta["transition"] == "legacy-full->m4-v2"
    assert meta["n_model_tensors"] == len(sd_src)
    assert meta["source_step"] == 18_750
    assert meta["optimizer"] == "inherited"
    assert meta["ema"] == "seeded-from-source-weights"
    assert meta["source_sha256"] == _sha256_file(ck)
    # pixel path alive in the target (never re-zeroed)
    for k in _PIXEL_ALIVE_KEYS:
        assert k in after
        assert float(after[k].abs().max()) > 0.0, f"pixel weight re-zeroed: {k}"


def test_transition_ema_caller_reseed(tmp_path: Path) -> None:
    """After the transition the caller seeds a fresh EMA from the loaded
    weights: shadow == source weights, n_samples_total == 0."""
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "phase1p.pt", cfg)
    m = _new_model(cfg)
    _apply_stage_transition(m, _optimizer_for(cfg, m), ck, DEVICE, 0, True, cfg)
    ema = SampleEMA(m, decay=0.5, ref_samples=cfg.ema.half_life_samples)
    assert ema.n_samples_total == 0
    shadow = ema.avg_state_dict()
    live = dict(m.named_parameters())
    assert set(shadow) == set(live)
    for fqn, p in live.items():
        assert torch.equal(shadow[fqn], p.float()), f"EMA shadow != source weights: {fqn}"


def test_transition_rejects_dead_pixel_source(tmp_path: Path) -> None:
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "dead.pt", cfg, pixel_zero=True)
    m = _new_model(cfg)
    with pytest.raises(ValueError, match="dead pixel path"):
        _apply_stage_transition(m, _optimizer_for(cfg, m), ck, DEVICE, 0, True, cfg)


def test_transition_rejects_v2_with_ema(tmp_path: Path) -> None:
    """A production v2 ckpt WITH an EMA section is a --resume source."""
    cfg = _cfg()
    src = _trained_model(cfg)
    payload = {
        "version": CKPT_VERSION,
        "step": 100,
        "model": src.state_dict(),
        "optimizer": _optimizer_for(cfg, src).state_dict(),
        "ema": {"decay": 0.5, "ref_samples": 500_000, "n_samples_total": 10, "params": {}},
    }
    ck = tmp_path / "v2.pt"
    torch.save(payload, ck)
    m = _new_model(cfg)
    with pytest.raises(ValueError, match="Use --resume"):
        _apply_stage_transition(m, _optimizer_for(cfg, m), ck, DEVICE, 0, True, cfg)


def test_transition_rejects_trunk_only(tmp_path: Path) -> None:
    """Strict full load: a trunk-only payload cannot fill the full model."""
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "trunk.pt", cfg, trunk_only=True)
    m = _new_model(cfg)
    with pytest.raises(RuntimeError, match="Missing key"):
        _apply_stage_transition(m, _optimizer_for(cfg, m), ck, DEVICE, 0, True, cfg)


def test_transition_requires_pixel_stage(tmp_path: Path) -> None:
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "phase1p.pt", cfg)
    m = _new_model(cfg)
    with pytest.raises(ValueError, match="pixel_features"):
        _apply_stage_transition(m, _optimizer_for(cfg, m), ck, DEVICE, 0, False, cfg)


def test_transition_missing_sections_rejected(tmp_path: Path) -> None:
    cfg = _cfg()
    ck = tmp_path / "bad.pt"
    torch.save({"foo": 1}, ck)
    m = _new_model(cfg)
    with pytest.raises(ValueError, match="not a checkpoint payload"):
        _apply_stage_transition(m, _optimizer_for(cfg, m), ck, DEVICE, 0, True, cfg)


# ----------------------------------------------------------------------
# optimizer compatibility gate
# ----------------------------------------------------------------------
def _warm_steps(
    model: AnimeSRModel, opt: torch.optim.Optimizer, n: int = 2
) -> None:
    """Deterministic training steps so every parameter receives a gradient
    (AdamW populates per-param state only for params with a grad)."""
    torch.manual_seed(3)
    for _ in range(n):
        rt = torch.randn(1, 128, 64, 64, device=DEVICE)
        z_lr = torch.randn(1, 128, 64, 64, device=DEVICE)
        lq = torch.randn(1, 3, 256, 256, device=DEVICE)
        t = 0.1 + 0.5 * torch.rand(1, device=DEVICE)
        sigma = torch.zeros(1, device=DEVICE)
        v_hat = model(rt, z_lr, lq, t, sigma)
        loss = torch.nn.functional.mse_loss(v_hat, torch.randn_like(v_hat))
        opt.zero_grad()
        loss.backward()
        opt.step()


def _source_state(cfg: Config, optimizer_cfg: Config | None = None) -> dict:
    src = _new_model(cfg)
    opt = _optimizer_for(optimizer_cfg or cfg, src)
    _warm_steps(src, opt, 3)
    return opt.state_dict()


def test_optimizer_gate_inherit(tmp_path: Path) -> None:
    cfg = _cfg()
    st = _source_state(cfg)
    ok, why = _optimizer_state_compatible(st, _new_model(cfg), cfg)
    assert ok, why


def test_optimizer_gate_fresh_when_incompatible(tmp_path: Path) -> None:
    """Each incompatible variant: (False, reason) AND the transition records
    optimizer=fresh with the target optimizer left pristine."""
    cfg = _cfg()
    m = _new_model(cfg)
    opt = _optimizer_for(cfg, m)

    st_betas = _source_state(cfg, _cfg_optimizer(betas=[0.9, 0.99]))
    st_eps = _source_state(cfg, _cfg_optimizer(eps=2e-8))
    st_wd = _source_state(cfg, _cfg_optimizer(weight_decay=0.06))

    st = _source_state(cfg)
    st_decay_wd0 = copy.deepcopy(st)
    st_decay_wd0["param_groups"][0]["weight_decay"] = 0.0
    st_nodw = copy.deepcopy(st)
    st_nodw["param_groups"][1]["weight_decay"] = 0.5
    st_missing = copy.deepcopy(st)
    st_missing["state"][1] = {"step": torch.tensor(3)}
    st_shape = copy.deepcopy(st)
    st_shape["state"][1]["exp_avg"] = torch.zeros(3, 3)
    st_count = copy.deepcopy(st)
    n = max(max(g["params"]) for g in st_count["param_groups"] if g["params"])
    for g in st_count["param_groups"]:
        if n in g["params"]:
            g["params"] = [i for i in g["params"] if i != n]
    st_count["state"].pop(n)
    st_malformed = {"not": "an optimizer state"}

    for label, bad in (
        ("betas", st_betas),
        ("eps", st_eps),
        ("weight_decay", st_wd),
        ("decay-group wd=0", st_decay_wd0),
        ("no-decay wd!=0", st_nodw),
        ("missing exp_avg", st_missing),
        ("shape mismatch", st_shape),
        ("param count", st_count),
        ("malformed", st_malformed),
    ):
        ok, why = _optimizer_state_compatible(bad, _new_model(cfg), cfg)
        assert not ok and why, f"{label}: expected incompatible, got ok={ok} why={why!r}"

    # fresh path end-to-end: incompatible source -> meta optimizer=fresh,
    # target state pristine, model still strict-loaded
    ck = _v1_ckpt(tmp_path / "badopt.pt", cfg, optimizer_cfg=_cfg_optimizer(betas=[0.9, 0.99]))
    meta = _apply_stage_transition(m, opt, ck, DEVICE, 0, True, cfg)
    assert meta["optimizer"] == "fresh"
    assert len(opt.state_dict()["state"]) == 0, "fresh optimizer must stay pristine"
    src_sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    after = m.state_dict()
    for k, v in src_sd.items():
        assert torch.equal(after[k], v), f"model must still strict-load: {k}"


def test_transition_without_optimizer_key_is_fresh(tmp_path: Path) -> None:
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "noopt.pt", cfg, drop_optimizer=True)
    m = _new_model(cfg)
    opt = _optimizer_for(cfg, m)
    meta = _apply_stage_transition(m, opt, ck, DEVICE, 0, True, cfg)
    assert meta["optimizer"] == "fresh"
    assert len(opt.state_dict()["state"]) == 0


def test_optimizer_gate_inherited_state_bitexact(tmp_path: Path) -> None:
    cfg = _cfg()
    ck = _v1_ckpt(tmp_path / "phase1p.pt", cfg)
    src_st = torch.load(ck, map_location="cpu", weights_only=False)["optimizer"]
    m = _new_model(cfg)
    opt = _optimizer_for(cfg, m)
    meta = _apply_stage_transition(m, opt, ck, DEVICE, 0, True, cfg)
    assert meta["optimizer"] == "inherited"
    dst_st = opt.state_dict()
    assert dst_st["state"].keys() == src_st["state"].keys()
    for i, s in src_st["state"].items():
        assert torch.equal(dst_st["state"][i]["exp_avg"], s["exp_avg"]), f"param {i} m drift"
        assert torch.equal(dst_st["state"][i]["exp_avg_sq"], s["exp_avg_sq"]), f"param {i} v drift"
        assert dst_st["state"][i].get("step") == s.get("step"), f"param {i} step drift"


# ----------------------------------------------------------------------
# scheduler: fresh M4 schedule + inherited moments coexist without NaN
# ----------------------------------------------------------------------
def test_scheduler_fresh_warmup_no_lr_jump() -> None:
    cfg = _cfg()

    def lr(s: int) -> float:
        return _cosine_lr(s, M4_TOTAL_STEPS, M4_BASE_LR, cfg)
    # warmup: strictly increasing from step 0 (no jump, no plateau at 0)
    assert lr(0) > 0.0
    for s in range(0, M4_WARM, 97):
        assert lr(s + 1) >= lr(s), f"warmup not non-decreasing at {s}"
    # step-0 LR is tiny (fresh stage, not the inherited 1.5e-5 tail)
    assert lr(0) < 1e-7
    # peak at the warmup boundary, continuous into the cosine
    assert lr(M4_WARM - 1) == pytest.approx(M4_BASE_LR, rel=1e-9)
    assert lr(M4_WARM) == pytest.approx(M4_BASE_LR, rel=1e-9)
    # cosine to the floor, finite everywhere
    floor = M4_BASE_LR * 0.10
    for s in (M4_WARM, M4_TOTAL_STEPS // 2, M4_TOTAL_STEPS - 1):
        assert lr(s) >= floor * (1 - 1e-9)
    assert lr(M4_TOTAL_STEPS - 1) < M4_BASE_LR


def test_inherited_moments_finite_updates() -> None:
    """Inherited Adam m/v under the fresh M4 schedule: no NaN, finite
    params, step-0 update is small (lr ~1.3e-8)."""
    cfg = _cfg()
    torch.manual_seed(1)
    ms = nn_linear()
    os_ = _optimizer_for(cfg, ms)
    for s in range(5):
        _tiny_step(ms, os_, s, cfg)
    src_st = os_.state_dict()

    torch.manual_seed(1)
    mt = nn_linear()
    ot = _optimizer_for(cfg, mt)
    ot.load_state_dict(src_st)
    prev_norm = sum(p.detach().abs().sum().item() for p in mt.parameters())
    losses = []
    for s in range(3):
        losses.append(_tiny_step(mt, ot, s, cfg))
        for p in mt.parameters():
            assert torch.isfinite(p).all(), f"param non-finite at step {s}"
    for v in losses:
        assert torch.isfinite(torch.tensor(v)).item()
    # step-0 LR ~1.3e-8 => the first update must be a negligible nudge
    assert abs(sum(p.detach().abs().sum().item() for p in mt.parameters()) - prev_norm) < 1e-5


def nn_linear() -> torch.nn.Module:
    m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
    return m.to(DEVICE)


def _tiny_step(m: torch.nn.Module, opt: torch.optim.Optimizer, step: int, cfg: Config) -> float:
    x = torch.randn(4, 8, device=DEVICE)
    loss = torch.nn.functional.mse_loss(m(x), torch.randn(4, 8, device=DEVICE))
    lr = _cosine_lr(step, M4_TOTAL_STEPS, M4_BASE_LR, cfg)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    return float(loss)


# ----------------------------------------------------------------------
# launch-decision stamp + sha helper
# ----------------------------------------------------------------------
def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"anime-sr" * 1000)
    assert _sha256_file(p) == hashlib.sha256(b"anime-sr" * 1000).hexdigest()


def test_stamp_launch_decision(tmp_path: Path) -> None:
    cfg = _cfg()
    path = stamp_launch_decision(cfg, tmp_path / "run")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["launch_decision"] == M4_LAUNCH_DECISION
    assert "256→1024 only" in doc["launch_decision"]


# ----------------------------------------------------------------------
# M4-1024 / canary config arithmetic
# ----------------------------------------------------------------------
def test_m4_config_milestones_land_on_steps() -> None:
    cfg = load_config(
        str(CFG_DIR / "base.toml"),
        str(CFG_DIR / "data.toml"),
        str(CFG_DIR / "m4_1024.toml"),
    )
    lf = cfg.latent_flow
    bs, world = 8, 2  # production: per-rank 8 x 2 ranks -> global batch 16
    assert cfg.phase1.exposure_target == 6_000_000
    assert cfg.phase1.exposure_min == cfg.phase1.exposure_max == 6_000_000
    assert cfg.phase1.exposure_target // (bs * world) == M4_TOTAL_STEPS
    assert lf.batch_size == 8
    assert lf.pixel_features is True
    assert lf.zhr_source == "onfly"
    assert lf.producer == "process"
    assert lf.save_every_steps == 0  # periodic save off; milestones only
    assert lf.save_at_exposures == [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]
    expected_steps = {100_000: 6_250, 250_000: 15_625, 500_000: 31_250,
                      1_000_000: 62_500, 2_000_000: 125_000, 4_000_000: 250_000}
    for e in lf.save_at_exposures:
        assert e % (bs * world) == 0, f"milestone {e} is not a multiple of global batch 16"
        step = e // (bs * world)
        assert step == expected_steps[e]
        # the loop predicate: (step + 1) * bs * world lands EXACTLY on e
        assert (step) * bs * world == e
    # held-out probe grid: 500k/1M/2M/4M/6M exposures
    assert lf.val_heldout_every_steps == 31_250
    assert 31_250 * bs * world == 500_000
    assert 12 * 31_250 == M4_TOTAL_STEPS  # 12 * 500k = 6M = run end (deduplicated)
    # frozen Phase-I optimizer + fresh M4 schedule
    assert cfg.optimizer.lr == M4_BASE_LR
    assert cfg.optimizer.betas == [0.9, 0.95]
    assert cfg.optimizer.weight_decay == 0.05
    assert cfg.scheduler.warmup_fraction == 0.03
    assert cfg.scheduler.min_lr_ratio == 0.10
    assert cfg.ema.half_life_samples == 500_000
    assert cfg.hardware.attention_backend == "sdpa-correctness"
    assert cfg.filter.clean_score_min == -1.0  # report-only


def test_canary_config_arithmetic() -> None:
    cfg = load_config(
        str(CFG_DIR / "base.toml"),
        str(CFG_DIR / "data.toml"),
        str(CFG_DIR / "m4_1024.toml"),
        str(CFG_DIR / "m4_1024_canary.toml"),
    )
    bs, world = 8, 2
    assert cfg.phase1.exposure_target == 10_000
    total = cfg.phase1.exposure_target // (bs * world)
    assert total == 625
    assert cfg.latent_flow.save_at_exposures == [5_120]
    assert 5_120 % (bs * world) == 0 and 5_120 // (bs * world) == 320
    # run-end-only held-out probe (cadence beyond the 625-step horizon)
    assert cfg.latent_flow.val_heldout_every_steps > total
    assert cfg.latent_flow.out_dir.endswith("latent-flow-m4-1024-canary")


# ----------------------------------------------------------------------
# CLI mutex
# ----------------------------------------------------------------------
def test_cli_mutex_two_transition_flags() -> None:
    from anime_sr.cli import train_latent_flow as cli

    with pytest.raises(SystemExit) as ei:
        cli.main(
            [
                "--config",
                str(CFG_DIR / "base.toml"),
                str(CFG_DIR / "smoke.toml"),
                "--index-dir", "x",
                "--webp-dir", "y",
                "--latent-dir", "z",
                "--resume", "a.pt",
                "--stage-transition", "b.pt",
            ]
        )
    assert "mutually exclusive" in str(ei.value)


def test_cli_mutex_init_trunk_and_stage_transition() -> None:
    from anime_sr.cli import train_latent_flow as cli

    with pytest.raises(SystemExit) as ei:
        cli.main(
            [
                "--config",
                str(CFG_DIR / "base.toml"),
                str(CFG_DIR / "smoke.toml"),
                "--index-dir", "x",
                "--webp-dir", "y",
                "--latent-dir", "z",
                "--init-trunk", "a.pt",
                "--stage-transition", "b.pt",
            ]
        )
    assert "mutually exclusive" in str(ei.value)
