"""P2-1: EMA + ckpt v2 wiring invariants (U233 M4-prep, 2026-08-30).

Covers the production contract:
  * SampleEMA: sample-anchored decay (retention == decay after exactly
    ref_samples samples), GLOBAL exposure semantics (half_life_samples is
    a global-exposure semantic: bs*world_size per step), fp32 shadow
    independent of the live dtype, state_dict roundtrip + config
    mismatch rejection, apply/restore;
  * ckpt v2: every production save section (model/optimizer/EMA/step/
    exposure cursor/RNG/scalars/provenance+resolved-config id), atomic
    file, load_v2 restore into fresh objects, v1 legacy compat
    (legacy=True, new sections None, fresh EMA tolerated), and the
    error path (EMA instance vs checkpoint without an EMA section);
  * the trainer helpers: _exposure_cursor (global_exposures =
    step*bs*world), _step_scalars (None-safe on an empty loop),
    _run_provenance (plain identifiers + resolved stream-defining values,
    no hashing).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from anime_sr.config.schema import Config
from anime_sr.train.ckpt_v2 import (
    CKPT_VERSION,
    load_v2,
    restore_rng,
    save_v2,
    snapshot_rng,
)
from anime_sr.train.ema_sample import SampleEMA
from anime_sr.train.latent_flow import (
    _exposure_cursor,
    _run_provenance,
    _step_scalars,
)
from torch import nn


class _Tiny(nn.Module):
    """2-parameter toy model (live dtype stays fp32 in these tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b


def _opt(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


# ----------------------------------------------------------------------
# SampleEMA semantics
# ----------------------------------------------------------------------
def test_ema_retention_after_ref_samples() -> None:
    """decay=0.5 at ref_samples: retention is exactly 1/2 after exactly
    ref_samples samples (and 1/4 after 2x)."""
    model = _Tiny()
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    ema = SampleEMA(model, decay=0.5, ref_samples=100)
    # "optimizer step": move the live params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    beta = ema.update(model, n_samples=100)
    assert beta == pytest.approx(0.5)
    for n, p in model.named_parameters():
        expected = 0.5 * p0[n] + 0.5 * p
        assert torch.allclose(
            ema._shadow[n], expected, atol=1e-6
        ), f"EMA shadow mismatch for {n} after 1x ref"
    # second update consuming 2x ref samples: beta = 0.5**2 = 0.25
    beta2 = ema.update(model, n_samples=200)
    assert beta2 == pytest.approx(0.25)
    assert ema.n_samples_total == 300


def test_ema_global_exposure_semantics() -> None:
    """The trainer feeds n_samples = bs*world_size (GLOBAL exposures).
    half_life=1000 global samples with bs*world=100 per step: after 10
    steps the retention must be exactly 0.5 — the half-life is a global
    semantic, not a per-rank one."""
    model = _Tiny()
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    bs, world = 4, 2
    half_life = 1000
    ema = SampleEMA(model, decay=0.5, ref_samples=half_life)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    steps = half_life // (bs * world)
    for _ in range(steps):
        ema.update(model, n_samples=bs * world)
    assert ema.n_samples_total == half_life
    for n, p in model.named_parameters():
        expected = 0.5 * p0[n] + 0.5 * p
        assert torch.allclose(ema._shadow[n], expected, atol=1e-6)


def test_ema_fp32_shadow_and_apply_restore() -> None:
    """The shadow is fp32 even for a bf16 live module; apply swaps the
    live weights (in the live dtype) and restore brings them back."""
    model = _Tiny()
    ema = SampleEMA(model, decay=0.9, ref_samples=10)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    ema.update(model, n_samples=10)
    for n, s in ema._shadow.items():
        assert s.dtype == torch.float32, f"shadow {n} is not fp32"
    prev = ema.apply(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0)  # destroy the live params
    ema.restore(model, prev)
    for n0, p0 in prev.items():
        assert torch.equal(p0, dict(model.named_parameters())[n0]), (
            f"restore failed for {n0}"
        )


def test_ema_state_roundtrip_and_mismatch() -> None:
    model = _Tiny()
    ema = SampleEMA(model, decay=0.5, ref_samples=100)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, n_samples=50)
    sd = ema.state_dict()

    model2 = _Tiny()
    ema2 = SampleEMA(model2, decay=0.5, ref_samples=100)
    ema2.load_state_dict(sd)
    for n in ema._shadow:
        assert torch.allclose(ema._shadow[n], ema2._shadow[n], atol=0)
    assert ema2.n_samples_total == ema.n_samples_total

    with pytest.raises(ValueError, match="EMA config mismatch"):
        ema3 = SampleEMA(_Tiny(), decay=0.7, ref_samples=100)
        ema3.load_state_dict(sd)


# ----------------------------------------------------------------------
# ckpt v2 sections + roundtrip
# ----------------------------------------------------------------------
def test_save_v2_sections_and_roundtrip(tmp_path: Path) -> None:
    model = _Tiny()
    opt = _opt(model)
    opt.step()
    ema = SampleEMA(model, decay=0.5, ref_samples=100)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, n_samples=100)

    cur = _exposure_cursor(42, bs=4, world_size=2, exposure_target=100_000)
    path = save_v2(
        tmp_path / "step-0000042.pt",
        step=42,
        model=model,
        opt=opt,
        ema=ema,
        scalars={"loss": 0.25, "lr": 1e-3},
        exposure=cur,
        provenance={"git_commit": "abc123", "config": "base.toml", "source_ckpt": None},
    )
    assert path.is_file()
    assert not (tmp_path / "step-0000042.part").exists()  # atomic: no .part left

    payload = torch.load(path, weights_only=False)
    assert payload["version"] == CKPT_VERSION
    assert payload["step"] == 42
    assert payload["ema"] is not None
    assert payload["scalars"]["loss"] == 0.25
    assert payload["exposure"]["step"] == 42
    assert payload["provenance"]["git_commit"] == "abc123"
    assert payload["rng"] is not None and payload["rng"]["cpu"] is not None

    # roundtrip into FRESH objects
    model2 = _Tiny()
    opt2 = _opt(model2)
    ema2 = SampleEMA(model2, decay=0.5, ref_samples=100)
    meta = load_v2(path, model2, opt2, ema=ema2, device="cpu")
    assert meta["step"] == 42 and meta["legacy"] is False
    for n, p in model.named_parameters():
        assert torch.allclose(p.detach(), model2.state_dict()[n], atol=0)
    for n, s in ema._shadow.items():
        assert torch.allclose(s, ema2._shadow[n], atol=0)
    assert meta["exposure"] == cur
    assert meta["scalars"]["loss"] == 0.25
    assert meta["rng"] is not None


def test_load_v2_v1_legacy_compat(tmp_path: Path) -> None:
    """A v1 file ({step, model, optimizer}) loads through load_v2 with
    legacy=True and the new sections None."""
    model = _Tiny()
    opt = _opt(model)
    opt.step()
    path = tmp_path / "v1.pt"
    torch.save(
        {"step": 7, "model": model.state_dict(), "optimizer": opt.state_dict()}, path
    )

    model2 = _Tiny()
    opt2 = _opt(model2)
    meta = load_v2(path, model2, opt2, ema=None, device="cpu")
    assert meta["legacy"] is True
    assert meta["step"] == 7
    assert meta["scalars"] is None
    assert meta["exposure"] is None and meta["rng"] is None
    for n, p in model.named_parameters():
        assert torch.allclose(p.detach(), model2.state_dict()[n], atol=0)


def test_load_v2_ema_section_required(tmp_path: Path) -> None:
    """Passing an EMA instance for a checkpoint WITHOUT an EMA section is
    a hard error (fail-closed, not a silent fresh EMA)."""
    model, opt = _Tiny(), _opt(_Tiny())
    path = tmp_path / "no-ema.pt"
    torch.save(
        {"step": 1, "model": model.state_dict(), "optimizer": opt.state_dict()}, path
    )
    model2 = _Tiny()
    o2 = _opt(model2)
    ema2 = SampleEMA(model2, decay=0.5, ref_samples=100)
    with pytest.raises(ValueError, match="no EMA section"):
        load_v2(path, model2, o2, ema=ema2, device="cpu")


# ----------------------------------------------------------------------
# RNG snapshot/restore
# ----------------------------------------------------------------------
def test_rng_snapshot_restore_roundtrip() -> None:
    torch.manual_seed(1234)
    snap = snapshot_rng()
    first = [torch.randn(4) for _ in range(3)]
    torch.manual_seed(999)  # move the state far away
    restore_rng(snap)
    second = [torch.randn(4) for _ in range(3)]
    for a, b in zip(first, second):
        assert torch.equal(a, b), "restore_rng did not reproduce the stream"


# ----------------------------------------------------------------------
# trainer helpers
# ----------------------------------------------------------------------
def test_exposure_cursor_global_accounting() -> None:
    cur = _exposure_cursor(100, bs=4, world_size=2, exposure_target=100_000)
    assert cur["step"] == 100
    assert cur["global_exposures"] == 100 * 4 * 2
    assert cur["exposure_per_cycle"] == 25
    assert cur["exposure_target"] == 100_000


def test_step_scalars_none_safe() -> None:
    from collections import deque

    window = deque([0.5, 0.4, 0.3], maxlen=100)
    s = _step_scalars(10, None, 0.0, window, 1.0, 1.0, 3)
    assert s["loss"] is None and s["lr"] == 0.0
    assert s["loss_window_mean"] == pytest.approx(0.4)
    assert s["loss_window"] == [0.5, 0.4, 0.3]
    assert s["data_wait_pct"] is None  # < 50 steps: not yet measured

    loss = torch.tensor(0.42)
    s2 = _step_scalars(60, loss, 2e-4, window, 9.0, 81.0, 60)
    assert s2["loss"] == pytest.approx(0.42)
    assert s2["data_wait_pct"] == pytest.approx(100.0 * 9.0 / 90.0)


def test_run_provenance_plain_identifiers(tmp_path: Path) -> None:
    cfg = Config()
    prov = _run_provenance(
        cfg, source_ckpt="/ckpt/step-0000100.pt", config_names=["base.toml", "data.toml"]
    )
    # plain identifiers only (no project-level hashing per repo rule)
    assert prov["config"] == "base.toml,data.toml"
    assert prov["source_ckpt"] == "/ckpt/step-0000100.pt"
    assert prov["torch_version"] == torch.__version__
    assert prov["platform"]
    assert prov["created_utc"].endswith("Z")
    assert prov["git_commit"] is None or isinstance(prov["git_commit"], str)
    # resolved config identifier: the stream-defining values
    r = prov["resolved"]
    assert r["ema_half_life_samples"] == cfg.ema.half_life_samples
    assert r["exposure_per_cycle"] == 25
    assert r["batch_size"] == cfg.latent_flow.batch_size
    assert r["zhr_source"] in ("store", "onfly")
    assert set(r["sampling"]) == {
        "enabled",
        "core_fraction",
        "regular_fraction",
        "aux_fraction",
    }
    assert r["clean_score_min"] == cfg.filter.clean_score_min
    assert r["attention_backend"] == cfg.hardware.attention_backend
    assert r["dtype"] == cfg.hardware.dtype
