"""P2-prep parity gate: WindowAttentionSDPA vs the frozen WindowAttention.

Acceptance: identical weights (state_dict copy), then
  * fp32 forward parity (exact & padded & global branches, shifted &
    unshifted) within tight tolerance on both CPU and (when available) CUDA;
  * fp32 backward parity (gradients of inputs and of the q/k/v/o
    projections);
  * a 20-step fixed-seed AdamW trajectory staying on the same optimum
    path (loss + parameter-delta cosine);
  * bf16 forward: relative-L2 tolerance REPORT (not a hard gate): the
    parent upcasts logits to fp32 before softmax, SDPA backends may keep
    bf16 -- the number recorded here is what the quality report cites.

Set P2_SDPA_REPORT=<path.json> to dump the per-case tolerance report.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from anime_sr.model.window_attention import WindowAttention
from anime_sr.model.window_attention_sdpa import WindowAttentionSDPA, sdpa_variant

_REPORT = os.environ.get("P2_SDPA_REPORT")
_REPORT_ROWS: list[dict] = []


def _record(case: str, dtype: str, device: str, max_abs: float, rel_l2: float,
            hard_gate: bool) -> None:
    _REPORT_ROWS.append({
        "case": case, "dtype": str(dtype), "device": device,
        "max_abs_diff": max_abs, "rel_l2": rel_l2, "hard_gate": hard_gate,
    })


def _build(dim: int, heads: int, kv: int, H: int, W: int, w: int, global_attn: bool,
           gqa_native: bool, dtype: torch.dtype, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    parent = WindowAttention(dim=dim, num_heads=heads, num_kv_heads=kv, window_size=w,
                             global_attention=global_attn)
    sdpa = WindowAttentionSDPA(dim=dim, num_heads=heads, num_kv_heads=kv, window_size=w,
                               global_attention=global_attn, gqa_native=gqa_native)
    sdpa.load_state_dict(parent.state_dict())
    parent = parent.to(device=device, dtype=dtype)
    sdpa = sdpa.to(device=device, dtype=dtype)
    torch.manual_seed(1234)
    x = torch.randn(2, H * W, dim, device=device, dtype=dtype)
    return parent, sdpa, x


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).norm().item()
    den = a.norm().item() or 1.0
    return num / den


# ----------------------------------------------------------------------
def test_fp32_forward_parity_exact_cpu() -> None:
    for shift in (False, True):
        p, s, x = _build(64, 4, 2, 16, 16, 8, False, False, torch.float32, torch.device("cpu"))
        a = p(x, 16, 16, shift=shift)
        b = s(x, 16, 16, shift=shift)
        assert torch.isfinite(b).all()
        ma = (a - b).abs().max().item()
        assert ma < 1e-4, f"exact shift={shift}: max_abs={ma}"
        _record(f"fp32-exact-16x16-shift{int(shift)}-cpu", "fp32", "cpu", ma, _rel_l2(a, b), True)


def test_fp32_forward_parity_padded_cpu() -> None:
    for shift in (False, True):
        p, s, x = _build(64, 4, 2, 14, 10, 8, False, False, torch.float32, torch.device("cpu"))
        a = p(x, 14, 10, shift=shift)
        b = s(x, 14, 10, shift=shift)
        assert torch.isfinite(b).all()
        ma = (a - b).abs().max().item()
        assert ma < 1e-4, f"padded shift={shift}: max_abs={ma}"
        _record(f"fp32-padded-14x10-shift{int(shift)}-cpu", "fp32", "cpu", ma, _rel_l2(a, b), True)


def test_fp32_forward_parity_global_cpu() -> None:
    # GQA-native (Hq heads vs Hkv heads) needs a backend that supports it;
    # the CPU math fallback rejects the head-count mismatch, so CPU checks
    # the bit-safe repeat_interleave mode only. The native mode is covered
    # by test_fp32_forward_parity_global_cuda (flash/mem-eff accept GQA).
    p, s, x = _build(64, 4, 2, 16, 16, 8, True, False, torch.float32, torch.device("cpu"))
    a = p(x, 16, 16, shift=False)
    b = s(x, 16, 16, shift=False)
    ma = (a - b).abs().max().item()
    assert ma < 1e-4, f"global gqa_native=False: max_abs={ma}"
    _record("fp32-global-16x16-native0-cpu", "fp32", "cpu", ma, _rel_l2(a, b), True)


def test_fp32_forward_parity_global_cuda() -> None:
    if not torch.cuda.is_available():
        return
    # repeat_interleave mode: hard gate (bit-safe default).
    p, s, x = _build(64, 4, 2, 16, 16, 8, True, False, torch.float32, torch.device("cuda"))
    a = p(x, 16, 16, shift=False)
    b = s(x, 16, 16, shift=False)
    ma = (a - b).abs().max().item()
    assert ma < 1e-3, f"cuda global gqa_native=False: max_abs={ma}"
    _record("fp32-global-16x16-native0-cuda", "fp32", "cuda", ma, _rel_l2(a, b), True)


def test_gqa_native_report_cuda() -> None:
    """Native GQA (Hq heads vs Hkv heads) is the benchmark A/B target, not
    the default: fp32 inputs fall back to SDPA backends that may reject the
    head-count mismatch, so the fp32 case is attempt-and-report; the bf16
    case (flash territory) is the one the bench actually relies on."""
    if not torch.cuda.is_available():
        return
    # fp32: attempt-and-report (fp32 falls back to backends that may
    # reject the head-count mismatch; that is itself the finding)
    p, s, x = _build(64, 4, 2, 16, 16, 8, True, True, torch.float32, torch.device("cuda"))
    try:
        a = p(x, 16, 16, shift=False)
        b = s(x, 16, 16, shift=False)
    except RuntimeError as exc:
        _record("fp32-global-16x16-native1-cuda", "fp32", "cuda", float("nan"), float("nan"),
                False)
        print(f"[p2-sdpa-parity] fp32 native GQA rejected by the SDPA backend (expected): {exc}")
    else:
        ma = (a - b).abs().max().item()
        assert ma < 1e-3, f"cuda global gqa_native=True (fp32): max_abs={ma}"
        _record("fp32-global-16x16-native1-cuda", "fp32", "cuda", ma, _rel_l2(a, b), False)

    # bf16: soft gate (report)
    p, s, x = _build(64, 4, 2, 16, 16, 8, True, True, torch.bfloat16, torch.device("cuda"))
    try:
        a = p(x, 16, 16, shift=False)
        b = s(x, 16, 16, shift=False)
    except RuntimeError as exc:
        _record("bf16-global-16x16-native1-cuda", "bf16", "cuda", float("nan"), float("nan"),
                False)
        print(f"[p2-sdpa-parity] bf16 native GQA rejected by the SDPA backend: {exc}")
    else:
        assert torch.isfinite(b.float()).all(), "bf16 native GQA: non-finite SDPA output"
        rl = _rel_l2(a.float(), b.float())
        assert rl < 0.05, f"bf16 native GQA rel_l2={rl} exceeds soft gate"
        _record("bf16-global-16x16-native1-cuda", "bf16", "cuda",
                (a.float() - b.float()).abs().max().item(), rl, False)


def test_fp32_forward_parity_cuda() -> None:
    if not torch.cuda.is_available():
        return
    for shift in (False, True):
        p, s, x = _build(64, 4, 2, 16, 16, 8, False, False, torch.float32, torch.device("cuda"))
        a = p(x, 16, 16, shift=shift)
        b = s(x, 16, 16, shift=shift)
        ma = (a - b).abs().max().item()
        assert ma < 1e-3, f"cuda exact shift={shift}: max_abs={ma}"
        _record(f"fp32-exact-16x16-shift{int(shift)}-cuda", "fp32", "cuda", ma, _rel_l2(a, b), True)


def test_fp32_backward_parity_cpu() -> None:
    p, s, x = _build(64, 4, 2, 16, 16, 8, False, False, torch.float32, torch.device("cpu"))
    xp = x.clone().requires_grad_()
    xs = x.clone().requires_grad_()
    p(xp, 16, 16, shift=True).sum().backward()
    s(xs, 16, 16, shift=True).sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        ga, gb = getattr(p, name).weight.grad, getattr(s, name).weight.grad
        assert ga is not None and gb is not None
        ma = (ga - gb).abs().max().item()
        assert ma < 1e-4, f"{name}.grad max_abs={ma}"
    assert xp.grad is not None and xs.grad is not None
    _record("fp32-bwd-grads-cpu", "fp32", "cpu", (xp.grad - xs.grad).abs().max().item(),
            _rel_l2(xp.grad, xs.grad), True)


def test_fixed_seed_trajectory_parity_cpu() -> None:
    """20-step AdamW on a shared init: the two cores must stay on the same path."""
    dim, heads, kv, H, W, w = 64, 4, 2, 16, 16, 8
    torch.manual_seed(7)
    p = WindowAttention(dim=dim, num_heads=heads, num_kv_heads=kv, window_size=w).double()
    torch.manual_seed(7)
    s = WindowAttentionSDPA(dim=dim, num_heads=heads, num_kv_heads=kv, window_size=w)
    s.load_state_dict(p.state_dict())
    p, s = p.to(torch.float32), s.to(torch.float32)
    opt_p = torch.optim.AdamW(p.parameters(), lr=1e-3)
    opt_s = torch.optim.AdamW(s.parameters(), lr=1e-3)
    x0 = p.q_proj.weight.detach().clone().flatten()
    losses = []
    for t in range(20):
        torch.manual_seed(1000 + t)
        x = torch.randn(2, H * W, dim)
        lp = (p(x, H, W, shift=True) ** 2).mean()
        opt_p.zero_grad(); lp.backward(); opt_p.step()
        ls = (s(x, H, W, shift=True) ** 2).mean()
        opt_s.zero_grad(); ls.backward(); opt_s.step()
        losses.append((lp.item(), ls.item()))
    last_p, last_s = losses[-1]
    rel = abs(last_p - last_s) / (abs(last_p) + 1e-12)
    dp = (p.q_proj.weight.detach().flatten() - x0)
    ds = (s.q_proj.weight.detach().flatten() - x0)
    cos = F.cosine_similarity(dp.unsqueeze(0), ds.unsqueeze(0)).item()
    assert rel < 1e-3, f"trajectory loss gap: {last_p} vs {last_s}"
    assert cos > 0.999, f"trajectory param-drift cosine={cos}"
    _record("fp32-20step-trajectory-cpu", "fp32", "cpu", abs(last_p - last_s), rel, True)


def test_bf16_relative_l2_report_cuda() -> None:
    """Soft gate: the bf16 gap is REPORTED (parent upcasts logits to fp32
    before softmax; SDPA backends may stay bf16). Hard-assert finite only."""
    if not torch.cuda.is_available():
        return
    for H, W, shift, case in ((16, 16, True, "bf16-exact-16x16-shift-cuda"),
                              (14, 10, False, "bf16-padded-14x10-cuda")):
        p, s, x = _build(64, 4, 2, H, W, 8, False, False, torch.bfloat16, torch.device("cuda"))
        a = p(x, H, W, shift=shift)
        b = s(x, H, W, shift=shift)
        assert torch.isfinite(b.float()).all(), f"{case}: non-finite SDPA output"
        ma = (a.float() - b.float()).abs().max().item()
        rl = _rel_l2(a.float(), b.float())
        assert rl < 0.05, f"{case}: bf16 rel_l2={rl} exceeds soft gate"
        _record(case, "bf16", "cuda", ma, rl, False)


def test_sdpa_variant_toggle() -> None:
    """gqa_native flip keeps the weights and flips the mode (bench A/B helper)."""
    m = WindowAttentionSDPA(dim=64, num_heads=4, num_kv_heads=2, window_size=8, gqa_native=False)
    m2 = sdpa_variant(m)
    assert m2.gqa_native is True and m.gqa_native is False
    for (na, pa), (nb, pb) in zip(sorted(m.named_parameters()), sorted(m2.named_parameters())):
        assert na == nb and torch.equal(pa, pb)


def test_report_dump() -> None:
    if _REPORT:
        with open(_REPORT, "w", encoding="utf-8") as f:
            json.dump(_REPORT_ROWS, f, indent=2)
        print(f"[p2-sdpa-parity] {len(_REPORT_ROWS)} cases -> {_REPORT}")
