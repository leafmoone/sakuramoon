"""Model architecture tests (plan §7, §8; M3 checklist items).

Covers:
    * config-driven build (base spec + smoke spec), shape contract
    * v-hat = 0 at init (zero-init output head, plan §7.7)
    * gradient flow to every major component
    * window-attention correctness: partition isolation, shifted-window
      identity (roll commutes with per-token linear maps), RoPE norm
      preservation
    * tiled-inference offset wiring (plan §17.2)
    * zero-init FiLM identity (plan §7.6)

CPU-safe: no VAE weights required.
"""

from __future__ import annotations

from typing import cast

import torch
from anime_sr.config.schema import ModelSpec
from anime_sr.model import (
    GlobalConditioner,
    RoPE2D,
    UFlowSR,
    WindowAttention,
)
from anime_sr.model.restoration_block import RestorationBlock
from anime_sr.model.uflow import AnimeSRModel
from pytest import raises
from torch import nn


def _base_model() -> AnimeSRModel:
    torch.manual_seed(0)
    return AnimeSRModel(ModelSpec())


def _stage_blocks(model: AnimeSRModel, name: str) -> list[RestorationBlock]:
    """Concrete blocks of one stage (typed view over the ModuleDict of modules)."""
    return [cast(RestorationBlock, blk) for blk in cast(nn.ModuleList, model.trunk.stages[name])]


def test_base_param_count() -> None:
    """Structural canary: the frozen §7.2 trunk sits in the expected band.

    The plan's 121-128M figure is an estimate; the structure spec is
    authoritative (plan §15.1). Actual count is recorded in design.md.
    """
    model = _base_model()
    n = model.num_params
    assert 90_000_000 <= n <= 160_000_000, f"base model has {n} params"


def test_smoke_spec_build_and_forward() -> None:
    """smoke.toml dims [192, 256, 384] build + forward (M3 canary, plan §13)."""
    from anime_sr.config.loader import load_config

    cfg = load_config("anime-sr/config/base.toml", "anime-sr/config/smoke.toml")
    model = AnimeSRModel(cfg.model)
    n = model.num_params
    assert n < 35_000_000, f"smoke model too large: {n}"

    r_t = torch.randn(1, 128, 64, 64)
    z_lr = torch.randn(1, 128, 64, 64)
    lq = torch.randn(1, 3, 256, 256)
    t = torch.tensor([0.3])
    sigma = torch.zeros(1)
    v = model(r_t, z_lr, lq, t, sigma)
    assert v.shape == (1, 128, 64, 64)
    assert torch.isfinite(v).all()


def test_forward_shapes_base() -> None:
    model = _base_model()
    r_t = torch.randn(2, 128, 64, 64)
    z_lr = torch.randn(2, 128, 64, 64)
    lq = torch.randn(2, 3, 256, 256)
    t = torch.tensor([0.0, 1.0])
    sigma = torch.tensor([0.0, 0.1])
    v = model(r_t, z_lr, lq, t, sigma)
    assert v.shape == (2, 128, 64, 64)


def test_zero_velocity_at_init() -> None:
    """§7.7: zero-init exit conv => v-hat = 0 exactly at init."""
    model = _base_model()
    model.eval()
    with torch.no_grad():
        v = model(
            torch.randn(1, 128, 64, 64),
            torch.randn(1, 128, 64, 64),
            torch.randn(1, 3, 256, 256),
            torch.tensor([0.5]),
            torch.tensor([0.0]),
        )
    assert v.abs().max().item() == 0.0


def test_gradient_flow() -> None:
    model = _base_model()
    v = model(
        torch.randn(1, 128, 64, 64),
        torch.randn(1, 128, 64, 64),
        torch.randn(1, 3, 256, 256),
        torch.tensor([0.5]),
        torch.tensor([0.0]),
    )
    v.sum().backward()
    blocks64 = _stage_blocks(model, "enc64")
    blocks16 = _stage_blocks(model, "bottleneck16")
    spot = [
        model.pixel_encoder.stem,  # pixel encoder (M3: decoder grad reaches trunk)
        model.trunk.proj_r,
        model.trunk.proj_z,
        model.trunk.proj_p64,
        blocks64[0].w1,
        blocks16[0].attn.q_proj,
        model.trunk.conditioner.mlp,
        model.trunk.gate_64,
        model.trunk.head.conv1,
    ]
    for m in spot:
        param = m if isinstance(m, torch.nn.Parameter) else next(m.parameters())
        assert param.grad is not None, f"no gradient reached {m}"


def test_film_identity_at_init() -> None:
    """§7.6: zero-init stage FiLMs are the identity at init."""
    cond = GlobalConditioner(gap_dim=256, stage_dims=(384, 512, 768, 512, 384))
    films = cond(torch.tensor([0.3]), torch.tensor([0.0]), torch.randn(1, 256))
    x = torch.randn(1, 16, 384)
    out = cond.apply("enc64", x, films)
    assert torch.allclose(out, x, atol=1e-6)


def test_window_partition_isolation() -> None:
    """A spike affects only its 8x8 window (normal windowing, 16x16 grid)."""
    torch.manual_seed(1)
    attn = WindowAttention(64, 2, 2, head_dim=32)  # 2 kv heads x 32 = 64
    B, H, W = 1, 16, 16
    x = torch.zeros(B, H * W, 64)
    spike = (3 * W + 3)  # token (row 3, col 3) -> window (0,0)
    x[0, spike] = 1.0
    with torch.no_grad():
        out = attn(x, H, W, shift=False)
    out_grid = out.view(B, H, W, 64)
    # same window (0,0): rows 0-7, cols 0-7 -> position (5, 5) sees the spike
    assert out_grid[0, 5, 5].abs().max() > 1e-4
    # window (0,1): rows 0-7, cols 8-15 -> zero v's there -> exactly zero
    assert out_grid[0, 3, 10].abs().max() == 0.0


def test_shifted_window_roll_identity() -> None:
    """Plan §7.3: tokens keep their own RoPE coordinates through the window
    shift (rotation happens on the original grid, then the roll).

    Two consequences are checked:

    1. Interior roll identity: where the 8x8 window does not cross the cyclic
       wrap, shift(x) == roll(normal(roll(x,-s), offset=+s), +s) holds exactly.
       In the offset formulation the wrapped tokens get coordinates shifted by
       a full grid width (e.g. x+16 instead of x), which continuous RoPE is
       NOT invariant to — so the identity is asserted on the un-wrapped
       interior only (post-roll [0,8)^2 == original [4,12)^2).
    2. Uniform offset invariance (full grid, wrap included): RoPE logits
       depend only on p - j, so adding the same delta to every coordinate is
       an exact no-op of the attention, even in the shifted path."""
    torch.manual_seed(2)
    H = W = 16
    dim = 64
    attn = WindowAttention(dim, 2, 1, head_dim=32, qk_norm=True)
    x = torch.randn(1, H, W, dim)
    tokens = x.reshape(1, H * W, dim)
    s = 4  # window_size // 2
    rolled = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
    tokens_r = rolled.reshape(1, H * W, dim)
    with torch.no_grad():
        out_shift = attn(tokens, H, W, shift=True)
        out_norm = attn(tokens_r, H, W, shift=False, offset=(s, s))
    expected = torch.roll(
        out_norm.view(1, H, W, dim), shifts=(s, s), dims=(1, 2)
    ).view(1, H * W, dim)
    # (1) interior: original rows/cols [4, 12) — the un-wrapped 8x8 window region
    g_shift = out_shift.view(1, H, W, dim)
    g_expected = expected.view(1, H, W, dim)
    assert torch.allclose(g_shift[:, 4:12, 4:12], g_expected[:, 4:12, 4:12], atol=1e-4)

    # (2) uniform offset is an exact no-op in the shifted path (full grid)
    with torch.no_grad():
        v_a = attn(tokens, H, W, shift=True, offset=(0, 0))
        v_b = attn(tokens, H, W, shift=True, offset=(5, 7))
    assert torch.allclose(v_a, v_b, atol=1e-4)


def test_rope_preserves_norms() -> None:
    """RoPE is a per-axis rotation: |q|, |k| invariant."""
    rope = RoPE2D(head_dim=64)
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    coords = torch.randn(128, 2) * 4.0
    q2, k2 = rope.apply(q, k, coords)
    assert torch.allclose(q2.norm(dim=-1), q.norm(dim=-1), rtol=1e-4, atol=1e-5)
    assert torch.allclose(k2.norm(dim=-1), k.norm(dim=-1), rtol=1e-4, atol=1e-5)
    assert not torch.allclose(q2, q)  # coordinates actually rotate


def test_tiled_offset_wiring() -> None:
    """§17.2 tiled inference: a uniform coordinate offset must be a no-op.

    Continuous 2D RoPE makes attention logits depend only on p - j, so
    adding the same delta to every coordinate (tile offset) cancels exactly.
    This is what makes tile-wise inference valid (plan §17.2).

    The zero-init head makes v-hat identically zero, so perturb the head
    once to make the invariance check non-trivial."""
    model = _base_model()
    model.eval()
    with torch.no_grad():
        model.trunk.head.conv2.weight.data.fill_(0.01)  # break zero-init for this test
        z_lr = torch.zeros(1, 128, 64, 64)
        r_t = torch.zeros(1, 128, 64, 64)
        lq = torch.zeros(1, 3, 256, 256)
        t, sigma = torch.tensor([0.5]), torch.tensor([0.0])
        v0 = model(r_t, z_lr, lq, t, sigma, offset=(0, 0))
        v1 = model(r_t, z_lr, lq, t, sigma, offset=(128, 64))
    assert torch.allclose(v0, v1, atol=1e-4)  # constant features: offset-invariant

    # random input: uniform offset is still a no-op (logits depend on p - j only)
    torch.manual_seed(0)
    r2 = torch.randn(1, 128, 64, 64)
    with torch.no_grad():
        va = model(r2, z_lr, lq, t, sigma, offset=(0, 0))
        vb = model(r2, z_lr, lq, t, sigma, offset=(128, 64))
    assert torch.allclose(va, vb, atol=1e-4)  # RoPE relative invariance

    # attention-level wiring: same property on a small grid
    attn = WindowAttention(64, 2, 1, head_dim=32)
    x_small = torch.randn(1, 16 * 16, 64)
    with torch.no_grad():
        a0 = attn(x_small, 16, 16, offset=(0, 0))
        a1 = attn(x_small, 16, 16, offset=(128, 64))
    assert torch.allclose(a0, a1, atol=1e-4)

    # coordinates must actually enter the rotation (offset is not silently
    # ignored): shifted coords produce a different RoPE rotation
    rope = RoPE2D(head_dim=32)
    q = torch.randn(1, 2, 256, 32)
    k = torch.randn(1, 2, 256, 32)
    ys = torch.arange(16).view(16, 1).expand(16, 16)
    xs = torch.arange(16).view(1, 16).expand(16, 16)
    coords0 = torch.stack([xs, ys], dim=-1).reshape(-1, 2).float()
    coords_d = coords0 + torch.tensor([128.0, 64.0])
    q2a, k2a = rope.apply(q, k, coords0)
    q2b, k2b = rope.apply(q, k, coords_d)
    assert not torch.allclose(q2a, q2b, atol=1e-3)
    assert not torch.allclose(k2a, k2b, atol=1e-3)


def test_pixel_encoder_output_scales() -> None:
    model = _base_model()
    out = model.pixel_encoder(torch.randn(2, 3, 256, 256))
    assert out.p256.shape == (2, 48, 256, 256)
    assert out.p128.shape == (2, 96, 128, 128)
    assert out.p64.shape == (2, 128, 64, 64)
    assert out.p32.shape == (2, 192, 32, 32)
    assert out.p16.shape == (2, 256, 16, 16)
    assert out.gap16.shape == (2, 256)
    p64, p32, p16, gap16 = out.v1_subset()
    assert (p64, p32, p16, gap16) == (out.p64, out.p32, out.p16, out.gap16)


def test_trunk_rejects_bad_grid() -> None:
    model = _base_model()
    trunk: UFlowSR = model.trunk
    films = trunk.conditioner(
        torch.tensor([0.5]), torch.tensor([0.0]), torch.zeros(1, 256)
    )
    with raises(ValueError, match="expects grid"):
        trunk._run_stage(torch.randn(1, 512, 16, 16), "enc32", films, (0, 0))
