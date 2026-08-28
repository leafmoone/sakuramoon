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
from anime_sr.config.schema import ModelSpec, OutputHeadSpec, UFlowSpec
from anime_sr.model import (
    GlobalConditioner,
    RoPE2D,
    UFlowSR,
    WindowAttention,
)
from anime_sr.model.restoration_block import RestorationBlock
from anime_sr.model.uflow import AnimeSRModel, apply_pixel_zero_init
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


def test_shifted_window_boundary_no_wrap() -> None:
    """P1 boundary fix: shifted windows are clipped at the border, not
    cyclic. Under the unmasked roll, tokens (0,0) and (15,15) share rolled
    window (1,1) and attend across both borders; the boundary mask restores
    the clipped 8x8 neighbourhood (rows/cols 0..7 at the top-left corner,
    12..15 at the bottom-right corner)."""
    torch.manual_seed(3)
    H = W = 16
    attn = WindowAttention(64, 2, 1, head_dim=32)
    x = torch.zeros(1, H * W, 64)
    x[0, 15 * W + 15] = 1.0  # bottom-right corner spike
    with torch.no_grad():
        out = attn(x, H, W, shift=True)
    g = out.view(1, H, W, 64)
    # (0,0) neighbourhood = rows 0..3 x cols 0..3 -> the corner spike must
    # be masked out exactly (its v is the only non-zero v in the window)
    assert g[0, 0, 0].abs().max() == 0.0
    # (15,15) neighbourhood = rows 12..15 x cols 12..15 contains the spike
    assert g[0, 15, 15].abs().max() > 1e-4
    # mirror: a top-left spike must not leak into the bottom-right corner
    x2 = torch.zeros(1, H * W, 64)
    x2[0, 3 * W + 3] = 1.0
    with torch.no_grad():
        out2 = attn(x2, H, W, shift=True)
    g2 = out2.view(1, H, W, 64)
    assert g2[0, 15, 15].abs().max() == 0.0
    assert g2[0, 3, 3].abs().max() > 1e-4


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
    # dynamic U-Flow: the encoder also serves smaller / non-square LQ (S = 4x
    # the latent edge, S % 16 == 0). Field names keep the 1024-HR reference
    # scale, so the actual grid is RELATIVE: p256 = input, p128 = input/2,
    # p64 = input/4, p32 = input/8, p16 = input/16.
    out128 = model.pixel_encoder(torch.randn(1, 3, 128, 128))
    assert out128.p256.shape == (1, 48, 128, 128)
    assert out128.p64.shape == (1, 128, 32, 32)
    assert out128.p16.shape == (1, 256, 8, 8)
    out192 = model.pixel_encoder(torch.randn(1, 3, 192, 128))
    assert out192.p16.shape == (1, 256, 12, 8)
    with raises(ValueError, match="divisible by 16"):
        model.pixel_encoder(torch.randn(1, 3, 120, 120))


def test_trunk_input_validation() -> None:
    """Dynamic U-Flow: the trunk validates INPUT shapes (grids are derived
    from the input, not from a frozen table)."""
    model = _base_model()
    trunk: UFlowSR = model.trunk
    t = torch.tensor([0.5])
    sigma = torch.zeros(1)
    with raises(ValueError, match="at least 8x8"):
        trunk(torch.randn(1, 128, 6, 6), torch.randn(1, 128, 6, 6), t, sigma)
    with raises(ValueError, match="4-divisible"):
        trunk(torch.randn(1, 128, 10, 10), torch.randn(1, 128, 10, 10), t, sigma)
    with raises(ValueError, match="128 channels"):
        trunk(torch.randn(1, 64, 32, 32), torch.randn(1, 64, 32, 32), t, sigma)
    with raises(ValueError, match="z_lr shape"):
        trunk(torch.randn(1, 128, 32, 32), torch.randn(1, 128, 16, 16), t, sigma)


def test_smoke_dynamic_grids() -> None:
    """Dynamic U-Flow (option A): the same trunk serves 512/768/1024 (latent
    32/48/64) and non-square buckets; 40x40 exercises the window-padding path
    (stage grids 40/20/10 are not 8-divisible)."""
    from anime_sr.config.loader import load_config

    cfg = load_config("anime-sr/config/base.toml", "anime-sr/config/smoke.toml")
    model = AnimeSRModel(cfg.model)
    t = torch.tensor([0.3])
    sigma = torch.zeros(1)
    for H, W in [(32, 32), (48, 48), (64, 64), (48, 32)]:
        r_t = torch.randn(1, 128, H, W)
        z_lr = torch.randn(1, 128, H, W)
        lq = torch.randn(1, 3, 4 * H, 4 * W)
        v = model(r_t, z_lr, lq, t, sigma)
        assert v.shape == (1, 128, H, W), f"({H}, {W})"
        assert torch.isfinite(v).all()
    # padded stage grids (40 -> 20 -> 10): forward + finite gradients
    r_t = torch.randn(1, 128, 40, 40, requires_grad=True)
    z_lr = torch.randn(1, 128, 40, 40, requires_grad=True)
    lq = torch.randn(1, 3, 160, 160, requires_grad=True)
    v = model(r_t, z_lr, lq, t, sigma)
    assert v.shape == (1, 128, 40, 40)
    v.sum().backward()
    assert r_t.grad is not None and torch.isfinite(r_t.grad).all()
    assert lq.grad is not None and torch.isfinite(lq.grad).all()


def test_window_padding_matches_clipped_brute_force() -> None:
    """Padded window path == clipped (no-wrap) 8x8 neighbourhood attention on
    the REAL grid: brute-force check on a 12x12 grid (768-HR bottleneck
    scale, window 8 does not tile), both normal and shifted blocks. Real
    queries must see exactly their clipped real neighbourhood; the padded
    (fake) tokens contribute nothing."""
    torch.manual_seed(0)
    attn = WindowAttention(64, 1, 1, head_dim=64, window_size=8, qk_norm=False)
    H = W = 12
    N = H * W
    B = 1
    dev = torch.device("cpu")
    for shift in (False, True):
        x = torch.randn(B, N, 64, requires_grad=True)
        out = attn(x, H, W, shift=shift)
        assert torch.isfinite(out).all()
        # brute-force reference on the real (H, W) grid
        q = attn.q_proj(x).view(B, N, 1, 64).transpose(1, 2)
        k = attn.k_proj(x).view(B, N, 1, 64).transpose(1, 2)
        v = attn.v_proj(x).view(B, N, 1, 64).transpose(1, 2)
        q, k = attn.q_norm(q), attn.k_norm(k)
        qr, kr = attn.rope.apply(q, k, attn._coords(H, W, (0, 0), dev))
        Hp = 16  # padded grid (ceil(12/8) * 8); shifted semantics: the
        # post-roll 8x8 block mapped back to pre-roll coordinates (NOT a
        # centred clip) — identical to the exact-path _shift_mask semantics,
        # e.g. query row 11 in a 12x12 grid sees pre-roll rows 4..11.
        exp = torch.zeros(B, N, 64)
        for i in range(H):
            for j in range(W):
                if shift:
                    b = ((i - 4) % Hp) // 8
                    rows = sorted({(r + 4) % Hp for r in range(b * 8, b * 8 + 8) if (r + 4) % Hp < H})
                    c = ((j - 4) % Hp) // 8
                    cols = sorted({(c_ + 4) % Hp for c_ in range(c * 8, c * 8 + 8) if (c_ + 4) % Hp < W})
                else:
                    rows = range((i // 8) * 8, min((i // 8) * 8 + 8, H))
                    cols = range((j // 8) * 8, min((j // 8) * 8 + 8, W))
                keys = torch.tensor([r * W + c for r in rows for c in cols], device=dev)
                qi = qr[:, :, i * W + j, :]  # (B, 1, 64)
                kk = kr.index_select(2, keys)  # (B, 1, nk, 64); tokens are dim 2
                vv = v.index_select(2, keys)
                logits = (qi @ kk.transpose(-1, -2)) * 64**-0.5
                exp[:, i * W + j, :] = (logits.softmax(-1) @ vv).squeeze(1)
        assert torch.allclose(attn.o_proj(exp), out, atol=1e-4), f"shift={shift}"
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"shift={shift}"
        assert attn.q_proj.weight.grad is not None and torch.isfinite(attn.q_proj.weight.grad).all(), f"shift={shift}"


def test_window_padding_spike_isolation() -> None:
    """A (7,7)-corner spike in a 7x7 grid must NOT reach query (0,0) through
    a SHIFTED padded window (clipped neighbourhood), while the unshifted
    self-block of the spike does see itself."""
    torch.manual_seed(1)
    attn = WindowAttention(64, 1, 1, head_dim=64, window_size=4, qk_norm=False)
    H = W = 7
    x = torch.zeros(1, H * W, 64)
    x[0, 6 * W + 6, 0] = 1.0  # spike at (6, 6)
    out_shift = attn(x, H, W, shift=True)
    # shifted clipped neighbourhood of (0,0) = rows/cols 0..3 x 0..3 (no wrap):
    # the spike at (6,6) is excluded -> output at (0,0) is 0
    assert out_shift[0, 0, :].abs().max().item() == 0.0
    out_normal = attn(x, H, W, shift=False)
    # normal window block of the spike contains the spike itself
    assert out_normal[0, 6 * W + 6, :].abs().max().item() > 0.0


def test_pixel_zero_init_identity() -> None:
    """Phase I-P transition (plan §8): a trunk-only checkpoint loaded into the
    full model, with the pixel weights zeroed (biases kept), must produce
    bit-identical output to the old trunk-only model for ANY pixel input.

    The zeroed weights make every pixel term input-independent —
    ``0 * p' + b_old == b_old`` (IEEE-exact) — so the transition starts from
    exactly the checkpoint's behavior. The test mirrors the trainer flow:
    build -> load trunk state -> re-apply the zero-init (load_state_dict
    overwrites the pixel weights).
    """
    torch.manual_seed(0)
    ref_trunk = UFlowSR(UFlowSpec(), OutputHeadSpec())
    # Give the pixel path "trained" (nonzero) biases so preservation is
    # observable (fresh init zeroes them via _small_init).
    with torch.no_grad():
        ref_params = dict(ref_trunk.named_parameters())
        ref_params["proj_p64.bias"].fill_(0.5)
        ref_params["proj_p32.bias"].fill_(0.25)
        ref_params["proj_p16.bias"].fill_(0.125)
        ref_params["conditioner.gap_proj.bias"].fill_(0.0625)

    full = AnimeSRModel(ModelSpec(), zero_init_pixel=True)
    zeroed = apply_pixel_zero_init(full.trunk)
    assert zeroed == [
        "proj_p64.weight",
        "proj_p32.weight",
        "proj_p16.weight",
        "conditioner.gap_proj.weight",
    ]
    # Trainer flow: load the trunk-only state, then re-apply the zero-init.
    full.trunk.load_state_dict(ref_trunk.state_dict())
    apply_pixel_zero_init(full.trunk)

    # Weights zero; trained biases preserved.
    params = dict(full.trunk.named_parameters())
    for name in zeroed:
        assert params[name].abs().max().item() == 0.0
    assert params["proj_p64.bias"].abs().max().item() == 0.5
    assert params["proj_p32.bias"].abs().max().item() == 0.25
    assert params["proj_p16.bias"].abs().max().item() == 0.125
    assert params["conditioner.gap_proj.bias"].abs().max().item() == 0.0625

    torch.manual_seed(1)
    r_t = torch.randn(1, 128, 64, 64)
    z_lr = torch.randn(1, 128, 64, 64)
    t = torch.tensor([0.3])
    sigma = torch.zeros(1)
    with torch.no_grad():
        out_ref = ref_trunk(r_t, z_lr, t, sigma)
        out_trunk = full.trunk(
            r_t,
            z_lr,
            t,
            sigma,
            torch.randn(1, 128, 64, 64),
            torch.randn(1, 192, 32, 32),
            torch.randn(1, 256, 16, 16),
            torch.randn(1, 256),
        )
        assert torch.equal(out_trunk, out_ref)  # bit-exact (fp32)
        out_full = full(r_t, z_lr, torch.randn(1, 3, 256, 256), t, sigma)
    assert torch.equal(out_full, out_ref)  # any LQ RGB -> old-model output
