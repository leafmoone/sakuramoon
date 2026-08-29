"""P2-2: attention backend wiring + dtype flow + native-GQA flag.

Contract under test (U233 P2-2, 2026-08-30):
  * ``hardware.attention_backend`` truly selects the core at
    construction time in RestorationBlock / UFlowSR / AnimeSRModel:
    correctness|manual|sdpa-correctness -> frozen explicit core (the
    DEFAULT — SDPA never becomes the default just because it runs),
    sdpa-repeat -> SDPA repeat_interleave, sdpa-native-gqa -> SDPA
    native GQA; unknown values are a hard error;
  * the RoPE fp32-trig promotion is cast back: a bf16 module keeps q/k/v
    in bf16 through the SDPA front half (no silent fp32 attention core);
  * native-GQA SDPA calls pass ``enable_gqa=True`` EXPLICITLY (Hq != Hkv)
    — and never in repeat mode;
  * model-level fixed-seed short trajectory: the same small trunk built
    with correctness vs sdpa-repeat follows the same optimum (output
    rel-L2 + loss parity, not bit-exact — different kernels).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from anime_sr.config.loader import load_config
from anime_sr.model.restoration_block import (
    RestorationBlock,
    build_attention,
)
from anime_sr.model.uflow import UFlowSR
from anime_sr.model.window_attention import WindowAttention
from anime_sr.model.window_attention_sdpa import WindowAttentionSDPA
from torch import nn

_CFG_DIR = Path(__file__).resolve().parent.parent / "config"


# ----------------------------------------------------------------------
# construction-time backend selection
# ----------------------------------------------------------------------
def test_build_attention_mapping() -> None:
    for name in ("correctness", "manual", "sdpa-correctness"):
        m = build_attention(64, 4, 2, backend=name)
        assert type(m) is WindowAttention, name
    m = build_attention(64, 4, 2, backend="sdpa-repeat")
    assert type(m) is WindowAttentionSDPA and m.gqa_native is False
    m = build_attention(64, 4, 2, backend="sdpa-native-gqa")
    assert type(m) is WindowAttentionSDPA and m.gqa_native is True
    with pytest.raises(ValueError, match="unknown hardware.attention_backend"):
        build_attention(64, 4, 2, backend="flash")


def test_restoration_block_backend() -> None:
    b = RestorationBlock(64, 4, 2, attention_backend="sdpa-repeat")
    assert isinstance(b.attn, WindowAttentionSDPA) and b.attn.gqa_native is False
    b2 = RestorationBlock(64, 4, 2)  # default stays correctness
    assert type(b2.attn) is WindowAttention
    b3 = RestorationBlock(64, 4, 2, attention_backend="sdpa-native-gqa")
    assert isinstance(b3.attn, WindowAttentionSDPA) and b3.attn.gqa_native is True
    with pytest.raises(ValueError):
        RestorationBlock(64, 4, 2, attention_backend="nope")


def test_uflow_threads_backend_to_all_blocks() -> None:
    cfg = load_config(_CFG_DIR / "base.toml", _CFG_DIR / "smoke.toml")
    trunk = UFlowSR(cfg.model.uflow, cfg.model.output_head, attention_backend="sdpa-repeat")
    assert trunk.attention_backend == "sdpa-repeat"
    n = 0
    for name in trunk.stages:
        stage = cast(nn.ModuleList, trunk.stages[name])
        for block in stage:
            assert isinstance(block.attn, WindowAttentionSDPA)
            n += 1
    assert n > 0

    trunk2 = UFlowSR(cfg.model.uflow, cfg.model.output_head)  # default
    for name in trunk2.stages:
        stage = cast(nn.ModuleList, trunk2.stages[name])
        for block in stage:
            assert type(block.attn) is WindowAttention


# ----------------------------------------------------------------------
# dtype flow (P2-2 item 2)
# ----------------------------------------------------------------------
def test_bf16_module_keeps_bf16_through_rope() -> None:
    m = WindowAttentionSDPA(dim=64, num_heads=4, num_kv_heads=2, gqa_native=False)
    m = m.to(dtype=torch.bfloat16)
    x = torch.randn(1, 64, 64, dtype=torch.bfloat16)
    q, k, v = m._qkv_rope(x, 8, 8, (0, 0))
    assert q.dtype == torch.bfloat16, "RoPE output must be cast back to the module dtype"
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16


def test_bf16_sdpa_core_stays_bf16() -> None:
    m = WindowAttentionSDPA(dim=64, num_heads=4, num_kv_heads=2, gqa_native=False)
    m = m.to(dtype=torch.bfloat16)
    q = torch.randn(2, 4, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(2, 2, 64, 64, dtype=torch.bfloat16)
    v = torch.randn(2, 2, 64, 64, dtype=torch.bfloat16)
    out = m._sdpa_core(q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), None)
    assert out.dtype == torch.bfloat16


# ----------------------------------------------------------------------
# native-GQA enable_gqa flag (P2-2 item 1)
# ----------------------------------------------------------------------
def test_enable_gqa_flag_explicit(monkeypatch) -> None:
    import anime_sr.model.window_attention_sdpa as w

    captured: list[bool] = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, enable_gqa=False, **kw):
        captured.append(bool(enable_gqa))
        return real(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa, **kw)

    monkeypatch.setattr(w.F, "scaled_dot_product_attention", spy)
    x = torch.randn(1, 64, 64)

    native = WindowAttentionSDPA(dim=64, num_heads=4, num_kv_heads=2, gqa_native=True)
    try:
        native(x, 8, 8)  # windowed exact path
    except RuntimeError:
        pass  # backends without native-GQA support reject Hq != Hkv
    try:
        ng = WindowAttentionSDPA(dim=64, num_heads=4, num_kv_heads=2, gqa_native=True,
                                 global_attention=True)
        ng(x, 8, 8)  # global path
    except RuntimeError:
        pass
    assert captured, "native-GQA path never reached the SDPA call"
    assert all(c is True for c in captured), "enable_gqa must be passed EXPLICITLY in native mode"

    captured.clear()
    rep = WindowAttentionSDPA(dim=64, num_heads=4, num_kv_heads=2, gqa_native=False)
    rep(x, 8, 8)
    assert captured and all(c is False for c in captured), (
        "repeat mode must not set enable_gqa"
    )


# ----------------------------------------------------------------------
# model-level fixed-seed short trajectory (P2-2 parity gate item)
# ----------------------------------------------------------------------
def test_model_trajectory_fixed_seed_cpu() -> None:
    """The same smoke trunk, built with the correctness core vs the
    sdpa-repeat core, seeded identically, runs a short fixed-seed
    trajectory: outputs and losses must track each other (rel-L2 small —
    the kernels differ, so NOT bit-exact).

    Note: the OutputHead is zero-init (§7.7), so at step 0 the trunk
    output is exactly 0 and only the head weights receive gradients;
    from step 1 the head is non-zero and gradients flow into the trunk,
    where the backends actually differ. A random (non-zero) target is
    REQUIRED — a zero target gives zero gradients and a vacuous test."""
    cfg = load_config(_CFG_DIR / "base.toml", _CFG_DIR / "smoke.toml")
    H = W = 32  # latent 32x32 -> 256x256 pixel path
    outs: dict[str, list[torch.Tensor]] = {}
    losses: dict[str, list[float]] = {}
    for backend in ("correctness", "sdpa-repeat"):
        torch.manual_seed(20260830)
        trunk = UFlowSR(cfg.model.uflow, cfg.model.output_head, attention_backend=backend)
        trunk.train()
        opt = torch.optim.AdamW(trunk.parameters(), lr=1e-4)
        o: list[torch.Tensor] = []
        ls: list[float] = []
        for step in range(3):
            torch.manual_seed(1000 + step)
            r_t = torch.randn(1, 128, H, W)
            z_lr = torch.randn(1, 128, H, W)
            # trunk pixel-injection channels (uflow.forward): 128/192/256
            p64 = torch.randn(1, 128, H, W)
            p32 = torch.randn(1, 192, H // 2, W // 2)
            p16 = torch.randn(1, 256, H // 4, W // 4)
            gap16 = torch.randn(1, 256)
            t = torch.tensor([0.1 + 0.1 * step])
            sigma = torch.zeros(1)
            target = torch.randn(1, 128, H, W)
            opt.zero_grad()
            v_hat = trunk(r_t, z_lr, t, sigma, p64, p32, p16, gap16)
            loss = F.mse_loss(v_hat, target)
            loss.backward()
            opt.step()
            o.append(v_hat.detach().clone())
            ls.append(loss.item())
        outs[backend] = o
        losses[backend] = ls

    # step 0: zero-init head -> outputs exactly equal (both 0), losses equal
    assert torch.equal(outs["correctness"][0], outs["sdpa-repeat"][0])
    assert losses["correctness"][0] == pytest.approx(losses["sdpa-repeat"][0], abs=1e-12)
    # steps 1-2: trunk receives gradients -> backends differ by kernel
    # noise only; the trajectories must stay on the same path
    for step in (1, 2):
        a, b = outs["correctness"][step], outs["sdpa-repeat"][step]
        assert a.norm() > 0
        rel = ((a - b).norm() / a.norm()).item()
        assert rel < 5e-3, f"step {step}: backend outputs diverge rel={rel}"
        assert abs(losses["correctness"][step] - losses["sdpa-repeat"][step]) < 1e-3
