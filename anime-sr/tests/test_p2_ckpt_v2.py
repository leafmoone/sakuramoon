"""P2-prep unit tests: checkpoint v2 schema (ckpt_v2).

Covers: full v2 round-trip (model / optimizer / EMA / scalars / exposure /
provenance / RNG), v1-legacy load through load_v2, v2 files readable by the
v1 loader (forward compat), RNG restore reproducibility, atomic write, and
bf16-model / fp32-EMA dtype preservation.  CPU; no VAE / weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from anime_sr.train import ckpt_v2
from anime_sr.train.ckpt_v2 import load_v2, make_provenance, restore_rng, save_v2
from anime_sr.train.ema_sample import SampleEMA
from anime_sr.train.pixel_baseline import _load_ckpt, _save_ckpt
from torch import nn


def _net(seed: int = 0, dtype: torch.dtype = torch.float32) -> tuple[nn.Sequential, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).to(dtype=dtype)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return m, opt


def test_v2_roundtrip(tmp_path: Path) -> None:
    m, opt = _net()
    ema = SampleEMA(m, decay=0.9, ref_samples=4)
    cast(nn.Linear, m[0]).weight.data.add_(0.1)
    ema.update(m, 4)
    loss = torch.stack([p.sum() for p in m.parameters()]).sum()
    loss.backward()  # a real step so the optimizer state exists
    opt.step()

    prov = make_provenance(git_commit="5a92ce1", config="anime-sr-p2prep-m4.toml",
                           source_ckpt="latest.pt", platform="hcu")
    out = save_v2(
        tmp_path / "step-0000123.pt",
        step=123,
        model=m,
        opt=opt,
        ema=ema,
        scalars={"loss": 0.37, "lr": 1e-3, "data_wait": 0.118},
        exposure={"index": 42, "cycle": 1, "per_cycle": 1024},
        provenance=prov,
        capture_rng=True,
    )
    assert out.exists() and not out.with_suffix(".part").exists()  # atomic, .part consumed

    m2, opt2 = _net(seed=999)  # different start weights
    ema2 = SampleEMA(m2, decay=0.9, ref_samples=4)
    meta = load_v2(out, m2, opt2, ema=ema2)
    assert meta["step"] == 123 and meta["legacy"] is False
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    st_a = next(iter(opt.state_dict()["state"].values()))
    st_b = next(iter(opt2.state_dict()["state"].values()))
    for k in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(st_a[k], st_b[k])
    for fqn in ema._shadow:
        assert torch.equal(ema._shadow[fqn], ema2._shadow[fqn])
    assert meta["scalars"]["data_wait"] == 0.118
    assert meta["exposure"] == {"index": 42, "cycle": 1, "per_cycle": 1024}
    assert meta["provenance"]["git_commit"] == "5a92ce1"
    assert meta["rng"] is not None and meta["rng"]["cpu"] is not None


def test_v1_legacy_load_via_load_v2(tmp_path: Path) -> None:
    m, opt = _net()
    _save_ckpt(tmp_path / "legacy.pt", 77, m, opt)
    m2, opt2 = _net(seed=999)
    meta = load_v2(tmp_path / "legacy.pt", m2, opt2)
    assert meta["legacy"] is True
    assert meta["step"] == 77
    assert meta["scalars"] is None and meta["provenance"] is None
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_v2_file_loadable_by_v1_loader(tmp_path: Path) -> None:
    """Forward compat: the production v1 reader ignores the extra sections."""
    m, opt = _net()
    save_v2(tmp_path / "v2.pt", step=5, model=m, opt=opt, capture_rng=False)
    m2, opt2 = _net(seed=999)
    step = _load_ckpt(tmp_path / "v2.pt", m2, opt2, torch.device("cpu"))
    assert step == 5
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_rng_restore_reproducibility() -> None:
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.randn(64)  # burn a few draws
    torch.randint(0, 9, (32,))
    st = ckpt_v2.snapshot_rng()
    z1 = torch.randn(64)
    z2 = np.random.rand(16)
    restore_rng(st)
    assert torch.equal(torch.randn(64), z1)
    assert np.array_equal(np.random.rand(16), z2)


def test_rng_cuda_roundtrip_when_available(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed_all(123)
    st = ckpt_v2.snapshot_rng()
    a = torch.randn(8, device="cuda")
    restore_rng(st)
    assert torch.equal(torch.randn(8, device="cuda"), a)


def test_bf16_model_fp32_ema(tmp_path: Path) -> None:
    m, opt = _net(dtype=torch.bfloat16)
    ema = SampleEMA(m, decay=0.9, ref_samples=4)
    save_v2(tmp_path / "bf16.pt", step=1, model=m, opt=opt, ema=ema, capture_rng=False)
    m2, opt2 = _net(seed=999, dtype=torch.bfloat16)
    ema2 = SampleEMA(m2, decay=0.9, ref_samples=4)
    load_v2(tmp_path / "bf16.pt", m2, opt2, ema=ema2)
    assert cast(nn.Linear, m2[0]).weight.dtype == torch.bfloat16
    for t in ema2._shadow.values():
        assert t.dtype == torch.float32


def test_ema_present_but_section_missing_raises(tmp_path: Path) -> None:
    m, opt = _net()
    _save_ckpt(tmp_path / "legacy.pt", 1, m, opt)
    m2, opt2 = _net(seed=999)
    ema2 = SampleEMA(m2, decay=0.9, ref_samples=4)
    try:
        load_v2(tmp_path / "legacy.pt", m2, opt2, ema=ema2)
        raise AssertionError("must raise when EMA passed but file has none")
    except ValueError:
        pass
