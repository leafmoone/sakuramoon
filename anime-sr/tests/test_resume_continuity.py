"""P0-2 resume-continuity tests: ``--init-trunk`` vs ``--resume`` split.

Covers the production save/load contract on a real (smoke-sized) model:

* N+M continuous steps  ==  N steps -> save v2 -> fresh model+opt ->
  ``_apply_resume`` -> M steps, with bit-exact model parameters, optimizer
  ``exp_avg``/``exp_avg_sq``/``step``, LR schedule, and RNG continuation
  (all randomness flows from the single global seed stream; the v2 RNG
  snapshot is the only bridge between the two halves);
* the exposure cursor round-trips through the v2 section;
* a trained pixel checkpoint resumed with ``_apply_resume`` keeps its
  pixel weights (zero-init is NEVER re-applied on resume);
* hard guards: ``--init-trunk`` rejects a full pixel checkpoint;
  ``--resume`` in the pixel stage rejects a trunk-only checkpoint;
  ``--init-trunk`` produces a fresh stage (step=0, trunk weights in,
  pixel zero-init applied).

CPU-safe: no VAE weights, no HCU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anime_sr.config.loader import load_config
from anime_sr.config.schema import Config
from anime_sr.data.pipeline import _EXPOSURE_PER_CYCLE
from anime_sr.model.uflow import AnimeSRModel
from anime_sr.train.ckpt_v2 import restore_rng, save_v2
from anime_sr.train.latent_flow import (
    _PIXEL_KEY_PREFIX,
    _apply_init_trunk,
    _apply_resume,
)
from anime_sr.train.pixel_baseline import _cosine_lr, _optimizer_for

CFG_DIR = Path(__file__).resolve().parent.parent / "config"

N_STEPS = 8
M_STEPS = 5
SEED = 7
TOTAL = 40


def _cfg() -> Config:
    return load_config(str(CFG_DIR / "base.toml"), str(CFG_DIR / "smoke.toml"))


def _new_model(cfg: Config, device: torch.device) -> AnimeSRModel:
    torch.manual_seed(0)
    # zero_init_pixel via the KWARG (as the trainer does): AnimeSRModel
    # applies it only from the kwarg, not from the spec field.
    return AnimeSRModel(cfg.model, zero_init_pixel=cfg.model.zero_init_pixel).to(device)


def _step(
    model: AnimeSRModel,
    opt: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    device: torch.device,
) -> float:
    """One deterministic training step; ALL randomness is drawn from the
    global seed stream (no re-seeding inside), so two runs continue
    identically only if the RNG state is carried over exactly."""
    rt = torch.randn(1, 128, 64, 64, device=device)
    z_lr = torch.randn(1, 128, 64, 64, device=device)
    lq = torch.randn(1, 3, 256, 256, device=device)
    t = 0.1 + 0.5 * torch.rand(1, device=device)
    sigma = torch.zeros(1, device=device)
    v_hat = model(rt, z_lr, lq, t, sigma)
    v_target = torch.randn_like(v_hat)
    loss = torch.nn.functional.mse_loss(v_hat, v_target)
    lr = _cosine_lr(step, TOTAL, cfg.optimizer.lr, cfg)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return lr


def _assert_optimizer_equal(a: torch.optim.Optimizer, b: torch.optim.Optimizer) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"], "LR schedule drift"
    # exp_avg/exp_avg_sq live per-param under state[i], not top-level
    assert set(sa["state"]) == set(sb["state"]), "optimizer param sets drift"
    for i, s_a in sa["state"].items():
        s_b = sb["state"][i]
        assert torch.equal(s_a["exp_avg"], s_b["exp_avg"]), f"param {i} exp_avg drift"
        assert torch.equal(
            s_a["exp_avg_sq"], s_b["exp_avg_sq"]
        ), f"param {i} exp_avg_sq drift"
        assert s_a.get("step") == s_b.get("step"), f"param {i} iteration drift"


def test_resume_continuity_n_plus_m(tmp_path: Path) -> None:
    """N+M == N + (save v2) + resume + M, bit-exact (params/opt/LR/RNG)."""
    cfg = _cfg()
    device = torch.device("cpu")
    n, m = N_STEPS, M_STEPS

    # path A: continuous
    torch.manual_seed(SEED)
    ma = _new_model(cfg, device)
    opta = _optimizer_for(cfg, ma)
    for s in range(n + m):
        _step(ma, opta, s, cfg, device)

    # path B: N steps -> save -> fresh -> resume -> M steps
    torch.manual_seed(SEED)
    mb = _new_model(cfg, device)
    optb = _optimizer_for(cfg, mb)
    for s in range(n):
        _step(mb, optb, s, cfg, device)
    ck = save_v2(
        tmp_path / "ck.pt",
        step=n,
        model=mb,
        opt=optb,
        exposure={
            "index": n,
            "cycle": n // _EXPOSURE_PER_CYCLE,
            "per_cycle": _EXPOSURE_PER_CYCLE,
        },
    )
    mc = _new_model(cfg, device)
    optc = _optimizer_for(cfg, mc)
    start, meta = _apply_resume(mc, optc, None, ck, device, 0, pixel_stage=True)
    assert start == n
    assert meta is not None and not meta["legacy"]
    assert meta["exposure"]["index"] == n  # exposure cursor round-trip
    restore_rng(meta["rng"])  # the only RNG bridge (path A never re-seeded)
    for s in range(n, n + m):
        _step(mc, optc, s, cfg, device)

    # model parameters bit-exact
    pa, pc = dict(ma.state_dict()), dict(mc.state_dict())
    assert set(pa) == set(pc)
    for k, value in pa.items():
        assert torch.equal(value, pc[k]), f"parameter drift: {k}"
    # optimizer state bit-exact
    _assert_optimizer_equal(opta, optc)


def test_resume_never_rezeros_pixel_weights(tmp_path: Path) -> None:
    """A trained (non-zero) pixel checkpoint survives --resume unchanged."""
    cfg = _cfg()
    device = torch.device("cpu")
    m = _new_model(cfg, device)
    with torch.no_grad():
        for p in m.pixel_encoder.parameters():
            p.add_(1.0)  # unmistakably trained: non-zero pixel path
    sd_before = {
        k: v.clone()
        for k, v in m.state_dict().items()
        if k.startswith(_PIXEL_KEY_PREFIX)
    }
    ck = save_v2(tmp_path / "ck.pt", step=3, model=m, opt=_optimizer_for(cfg, m))
    m2 = _new_model(cfg, device)
    _apply_resume(m2, _optimizer_for(cfg, m2), None, ck, device, 0, pixel_stage=True)
    after = m2.state_dict()
    assert len(sd_before) > 0, "smoke model has no pixel_encoder keys"
    for k, v in sd_before.items():
        assert torch.equal(after[k], v), f"pixel weight altered by resume: {k}"
        assert torch.count_nonzero(v) > 0, f"fixture lost non-zero pixel weights: {k}"


def test_init_trunk_rejects_full_checkpoint(tmp_path: Path) -> None:
    cfg = _cfg()
    device = torch.device("cpu")
    m = _new_model(cfg, device)
    ck = save_v2(tmp_path / "full.pt", step=3, model=m, opt=_optimizer_for(cfg, m))
    m2 = _new_model(cfg, device)
    with pytest.raises(RuntimeError, match="FULL pixel checkpoint"):
        _apply_init_trunk(m2, ck, cfg, device, 0)


def test_resume_rejects_trunk_only_in_pixel_stage(tmp_path: Path) -> None:
    cfg = _cfg()
    device = torch.device("cpu")
    m = _new_model(cfg, device)
    sd = {
        k: v
        for k, v in m.state_dict().items()
        if not k.startswith(_PIXEL_KEY_PREFIX)
    }
    ck = tmp_path / "trunk_only.pt"
    torch.save(
        {
            "step": 5,
            "model": sd,
            "optimizer": _optimizer_for(cfg, m).state_dict(),
        },
        ck,
    )
    m2 = _new_model(cfg, device)
    with pytest.raises(RuntimeError, match="TRUNK-ONLY"):
        _apply_resume(m2, _optimizer_for(cfg, m2), None, ck, device, 0, pixel_stage=True)


def test_init_trunk_fresh_stage(tmp_path: Path) -> None:
    """--init-trunk: trunk weights in, fresh stage (step 0), pixel zero-init."""
    cfg = _cfg()
    cfg2 = cfg.model_copy(
        update={"model": cfg.model.model_copy(update={"zero_init_pixel": True})}
    )
    device = torch.device("cpu")
    m = _new_model(cfg2, device)
    with torch.no_grad():
        for p in m.trunk.parameters():
            p.add_(0.5)  # simulate a trained trunk
    sd_trunk = {
        k: v
        for k, v in m.state_dict().items()
        if not k.startswith(_PIXEL_KEY_PREFIX)
    }
    ck = tmp_path / "trunk_only.pt"
    torch.save({"step": 12_345, "model": sd_trunk, "optimizer": {}}, ck)

    m2 = _new_model(cfg2, device)
    start = _apply_init_trunk(m2, ck, cfg2, device, 0)
    assert start == 0, "init-trunk must start a fresh stage at step 0"
    after = m2.state_dict()
    # the four pixel-path weights are re-zeroed by design on a fresh-stage
    # transition (apply_pixel_zero_init) — they are NOT expected to carry.
    pixel_path = (
        "trunk.proj_p64.weight",
        "trunk.proj_p32.weight",
        "trunk.proj_p16.weight",
        "trunk.conditioner.gap_proj.weight",
    )
    for k, v in sd_trunk.items():
        if k in pixel_path:
            continue  # intentionally re-zeroed (fresh-stage contract)
        assert torch.equal(after[k], v), f"trunk weight not carried: {k}"
    for p in pixel_path:
        assert p in after, f"missing pixel-path weight {p}"
        assert torch.count_nonzero(after[p]) == 0, f"{p} not zero-init on transition"
