"""CPU numerical unit tests for the Hybrid CMuon backend.

Covers:
  A. Newton-Schulz: tall/wide/square x BF16/FP32, finite + deterministic.
  B. chunk=1: CMuon update matches the reference torch.optim.Muon.
  C. FFN chunk=2: fused update == concat(independent_muon(gate), independent_muon(up)).
  D. AdaLN chunk=6: fused update == concat of 6 independent Muon updates.
  E. Chunking does not change the original tensor shape / FQN.
  F. Parameter partition: disjoint + complete (CMuon + AdamW == all trainable).
  K. Full checkpoint path (CUDA/HCU): hybrid->hybrid state-exact resume,
     AdamW->hybrid fork, NS-map / routing / schema tamper rejection, and the
     opt-in NS safety telemetry wired into step().
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import pytest
import torch
from torch import nn

from sakuramoon.checkpoint.load import (
    _load_hybrid_optimizer_state,
    _load_hybrid_state_exact,
    _validate_hybrid_cmuon_state,
    _validate_hybrid_optimizer_schema,
    _validate_transition_optimizer_schema,
    _validate_transition_optimizer_state,
)
from sakuramoon.checkpoint.save import _hybrid_optimizer_schema, _optimizer_schema
from sakuramoon.checkpoint.schema import CheckpointError
from sakuramoon.optim import cmuon as cmuon_mod
from sakuramoon.optim.adamw8bit import build_adamw8bit
from sakuramoon.optim.cmuon import (
    CMUON_ROLES,
    CMuonChunkSpec,
    CMuonConfig,
    NSSafetyTelemetry,
    _cmuon_update,
    _cmuon_update_phase1,
    _cmuon_update_phase2,
    build_hybrid_cmuon,
    cmuon_moonlight_alpha,
    cmuon_zeroth_power,
    cmuon_zeroth_power_traced,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import (
    CMuonSafetyError,
    ForensicConfig,
)

DEVICE = "cpu"
LR = 0.00015625  # the JLT-scaled G1 LR (0.00005 * 800 / 256)
MOMENTUM = 0.95
NS_STEPS = 5  # legacy global reference depth; per-role tests use resolve_ns_map


def _make_spec(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    chunk_count: int,
    roles: tuple[str, ...],
    weight_decay: float = 0.0,
    role: str = "attention_q",
) -> tuple[nn.Parameter, CMuonChunkSpec]:
    param = nn.Parameter(torch.empty(shape, dtype=dtype))
    spec = CMuonChunkSpec(
        name=name,
        parameter=param,
        weight_decay=weight_decay,
        chunk_count=chunk_count,
        chunk_dim=0,
        roles=roles,
        role=role,
    )
    return param, spec


def _cfg(
    *, ns_steps: int | None = None, ns_steps_by_role: dict | None = None, **overrides
) -> CMuonConfig:
    """Build a CMuonConfig. Pass ns_steps (uniform) or ns_steps_by_role (canonical
    per-role map); the default keeps every role at NS_STEPS."""
    if ns_steps_by_role is None:
        ns_steps_by_role = resolve_ns_map(
            None, ns_steps if ns_steps is not None else NS_STEPS
        )
    base: dict = {
        "lr": LR,
        "momentum": MOMENTUM,
        "nesterov": True,
        "ns_steps_by_role": ns_steps_by_role,
        "momentum_dtype": "bfloat16",
        "chunk_rescale_sqrt_n": False,
    }
    base.update(overrides)
    return CMuonConfig(**base)


# ---------------------------------------------------------------------------
# A. Newton-Schulz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 16), (16, 64), (32, 32)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ns_finite_and_deterministic(shape, dtype):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(*shape, generator=g, dtype=dtype)
    out1 = cmuon_zeroth_power(x, NS_STEPS)
    out2 = cmuon_zeroth_power(x, NS_STEPS)
    assert out1.shape == shape, "NS must preserve the input shape"
    assert torch.isfinite(out1).all(), "NS output must be finite"
    assert out1.dtype == torch.bfloat16, "NS must always be BF16"
    assert torch.equal(out1, out2), "NS must be deterministic"


def test_ns_matches_native_muon_ortho():
    """The quintic NS must match torch.optim.Muon's internal orthogonalization."""
    from torch.optim._muon import _zeropower_via_newtonschulz

    g = torch.Generator().manual_seed(1)
    for shape in [(48, 24), (24, 48), (32, 32)]:
        x = torch.randn(*shape, generator=g, dtype=torch.float32)
        mine = cmuon_zeroth_power(x, NS_STEPS)
        ref = _zeropower_via_newtonschulz(x, (3.4445, -4.7750, 2.0315), NS_STEPS, 1e-7)
        assert torch.allclose(mine, ref, atol=1e-3, rtol=1e-3), (
            f"NS mismatch at {shape}"
        )


# ---------------------------------------------------------------------------
# B. chunk=1 matches the reference Muon
# ---------------------------------------------------------------------------


# Small shapes keep the CPU test fast; the production-shape (2560x2560 etc.)
# verification + BF16/FP32 perf comparison run on the HCU benchmark (salt1),
# since local NVIDIA GPU timings/precision do not transfer to the DCU.
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((256, 256), torch.bfloat16),
        ((64, 256), torch.bfloat16),
        ((256, 128), torch.float32),
    ],
)
def test_chunk1_matches_reference_muon(shape, dtype):
    torch.manual_seed(2)
    lr, wd = LR, 0.0
    # reference: native Muon on the whole tensor
    w_ref = nn.Parameter(torch.randn(*shape, dtype=dtype))
    w_mine = nn.Parameter(w_ref.data.clone())
    grad = torch.randn(*shape, dtype=dtype)
    w_ref.grad = grad.clone()
    w_mine.grad = grad.clone()
    ref_opt = torch.optim.Muon(
        [w_ref],
        lr=lr,
        weight_decay=wd,
        momentum=MOMENTUM,
        nesterov=True,
        ns_steps=NS_STEPS,
        adjust_lr_fn="match_rms_adamw",
    )
    ref_opt.step()
    # mine: chunk=1, momentum dtype == param dtype (the reference case)
    mdtype = "float32" if dtype == torch.float32 else "bfloat16"
    param, spec = _make_spec("w", shape, dtype, 1, ("x",))
    param.data.copy_(w_mine.data)
    cfg = _cfg(momentum_dtype=mdtype, chunk_rescale_sqrt_n=False)
    buf = torch.zeros_like(param, dtype=getattr(torch, mdtype))
    _cmuon_update(param, grad, buf, spec, cfg)
    assert param.shape == shape
    # BF16 has ~3 sig digits; use a loose tolerance for BF16, tight for FP32
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(param.data, w_ref.data, atol=atol, rtol=1e-2), (
        f"chunk=1 update diverges from reference Muon: "
        f"max diff {(param.data - w_ref.data).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# C. FFN chunk=2 == concat(independent_muon(gate), independent_muon(up))
# ---------------------------------------------------------------------------


def _reference_muon_on_chunk(chunk, grad, *, lr, wd, dtype) -> torch.Tensor:
    w = nn.Parameter(chunk.clone())
    w.grad = grad.clone()
    opt = torch.optim.Muon(
        [w],
        lr=lr,
        weight_decay=wd,
        momentum=MOMENTUM,
        nesterov=True,
        ns_steps=NS_STEPS,
        adjust_lr_fn="match_rms_adamw",
    )
    opt.step()
    return w.data


def test_fused_chunk2_equals_independent_muon():
    torch.manual_seed(3)
    inter, hidden = 256, 256  # small; production [13824,2560] covered by HCU bench
    dtype = torch.bfloat16
    lr, wd = LR, 0.0
    fused = torch.randn(2 * inter, hidden, dtype=dtype)
    grad = torch.randn(2 * inter, hidden, dtype=dtype)
    # reference: independent Muon on each chunk
    gate_ref = _reference_muon_on_chunk(
        fused[:inter], grad[:inter], lr=lr, wd=wd, dtype=dtype
    )
    up_ref = _reference_muon_on_chunk(
        fused[inter:], grad[inter:], lr=lr, wd=wd, dtype=dtype
    )
    ref = torch.cat([gate_ref, up_ref], dim=0)
    # mine: chunked update on the fused tensor
    param, spec = _make_spec(
        "mlp.in_proj.weight", fused.shape, dtype, 2, ("gate", "up"), role="ffn_in"
    )
    param.data.copy_(fused)
    cfg = _cfg(momentum_dtype="bfloat16", chunk_rescale_sqrt_n=False)
    buf = torch.zeros_like(param, dtype=torch.bfloat16)
    _cmuon_update(param, grad, buf, spec, cfg)
    assert param.shape == fused.shape, "chunking must not change the shape"
    diff = (param.data - ref).abs().max().item()
    assert diff < 2e-2, f"chunk=2 diverges from independent Muon: max diff {diff}"


# ---------------------------------------------------------------------------
# D. AdaLN chunk=6 == concat of 6 independent Muon updates
# ---------------------------------------------------------------------------


def test_fused_chunk6_equals_independent_muon():
    torch.manual_seed(4)
    model_dim, hidden = 256, 128  # small; production [15360,1024] covered by HCU bench
    dtype = torch.float32  # shared_block_projection is FP32
    lr, wd = LR, 0.0
    fused = torch.randn(6 * model_dim, hidden, dtype=dtype)
    grad = torch.randn(6 * model_dim, hidden, dtype=dtype)
    chunks_ref = []
    for i in range(6):
        c = _reference_muon_on_chunk(
            fused[i * model_dim : (i + 1) * model_dim],
            grad[i * model_dim : (i + 1) * model_dim],
            lr=lr,
            wd=wd,
            dtype=dtype,
        )
        chunks_ref.append(c)
    ref = torch.cat(chunks_ref, dim=0)
    roles = ("a_scale", "a_shift", "a_gate", "m_scale", "m_shift", "m_gate")
    param, spec = _make_spec(
        "conditioner.shared_block_projection.weight",
        fused.shape,
        dtype,
        6,
        roles,
        role="adaln_shared",
    )
    param.data.copy_(fused)
    cfg = _cfg(momentum_dtype="float32", chunk_rescale_sqrt_n=False)
    buf = torch.zeros_like(param, dtype=torch.float32)
    _cmuon_update(param, grad, buf, spec, cfg)
    assert param.shape == fused.shape
    diff = (param.data - ref).abs().max().item()
    assert diff < 1e-4, f"chunk=6 diverges from independent Muon: max diff {diff}"


# ---------------------------------------------------------------------------
# D2. chunk_rescale_sqrt_n changes the update (independent switch)
# ---------------------------------------------------------------------------


def test_chunk_rescale_switch_changes_update():
    torch.manual_seed(5)
    inter, hidden = 512, 256
    dtype = torch.bfloat16
    fused = torch.randn(2 * inter, hidden, dtype=dtype)
    grad = torch.randn(2 * inter, hidden, dtype=dtype)
    results = []
    for rescale in (False, True):
        param, spec = _make_spec(
            "w", fused.shape, dtype, 2, ("gate", "up"), role="ffn_in"
        )
        param.data.copy_(fused)
        cfg = _cfg(momentum_dtype="bfloat16", chunk_rescale_sqrt_n=rescale)
        buf = torch.zeros_like(param, dtype=torch.bfloat16)
        _cmuon_update(param, grad, buf, spec, cfg)
        results.append(param.data.clone())
    # rescale ON multiplies alpha by sqrt(2); the update must differ
    diff = (results[0] - results[1]).abs().max().item()
    assert diff > 0, "chunk_rescale_sqrt_n must change the update"
    # the rescaled UPDATE should be ~sqrt(2) larger in RMS (measure the delta,
    # not the updated param which is dominated by the initial value)
    update_off = (results[0] - fused).float()
    update_on = (results[1] - fused).float()
    rms_off = update_off.pow(2).mean().sqrt().item()
    rms_on = update_on.pow(2).mean().sqrt().item()
    ratio = rms_on / rms_off
    assert 1.3 < ratio < 1.8, f"expected ~sqrt(2) RMS ratio, got {ratio}"


# ---------------------------------------------------------------------------
# E. Chunking does not change shape / FQN
# ---------------------------------------------------------------------------


def test_chunking_preserves_shape_and_fqn():
    torch.manual_seed(6)
    role_by_chunk = {1: "attention_out", 2: "ffn_in", 6: "adaln_shared"}
    for chunk_count, (rows, cols) in [
        (1, (256, 256)),
        (2, (512, 256)),
        (6, (1536, 128)),
    ]:
        dtype = torch.bfloat16
        param, spec = _make_spec(
            "dit.blocks.slot_00.mlp.in_proj.weight",
            (rows, cols),
            dtype,
            chunk_count,
            ("r",) * chunk_count,
            role=role_by_chunk[chunk_count],
        )
        grad = torch.randn(rows, cols, dtype=dtype)
        cfg = _cfg(momentum_dtype="bfloat16")
        buf = torch.zeros_like(param, dtype=torch.bfloat16)
        shape_before = param.shape
        name_before = spec.name
        _cmuon_update(param, grad, buf, spec, cfg)
        assert param.shape == shape_before, "shape must not change"
        assert spec.name == name_before, "FQN must not change"


# ---------------------------------------------------------------------------
# Moonlight alpha
# ---------------------------------------------------------------------------


def test_moonlight_alpha_values():
    # core: no rescale
    assert cmuon_moonlight_alpha(2560, 2560, LR, 1) == pytest.approx(
        LR * 0.2 * math.sqrt(2560)
    )
    assert cmuon_moonlight_alpha(640, 2560, LR, 1) == pytest.approx(
        LR * 0.2 * math.sqrt(2560)
    )
    # rescale ON: sqrt(N) factor
    assert cmuon_moonlight_alpha(6912, 2560, LR, 2) == pytest.approx(
        LR * 0.2 * math.sqrt(6912) * math.sqrt(2)
    )
    assert cmuon_moonlight_alpha(2560, 1024, LR, 6) == pytest.approx(
        LR * 0.2 * math.sqrt(2560) * math.sqrt(6)
    )


# ---------------------------------------------------------------------------
# F. Parameter partition (disjoint + complete) on a mock model
# ---------------------------------------------------------------------------


class _MockDiTBlock(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.attention = nn.Module()
        self.attention.q_proj = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.k_proj = nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.v_proj = nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.content_gate = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.out_proj = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.mlp = nn.Module()
        self.mlp.in_proj = nn.Linear(
            hidden, 2 * inter, bias=False, dtype=torch.bfloat16
        )
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)
        self.attention_norm = nn.Parameter(torch.ones(hidden, dtype=torch.float32))
        self.mlp_norm = nn.Parameter(torch.ones(hidden, dtype=torch.float32))


class GlobalConditioner(nn.Module):
    """Mock named exactly 'GlobalConditioner' so the audit treats its params as
    the sensitive ancestor (FP32), matching the production model."""

    def __init__(self, hidden):
        super().__init__()
        self.shared_block_projection = nn.Linear(hidden // 4, 6 * hidden, bias=False)
        self.final_projection = nn.Linear(hidden // 4, 2 * hidden, bias=True)


class _MockDiT(nn.Module):
    def __init__(self, hidden, inter, n_blocks):
        super().__init__()
        self.input_projection = nn.Linear(128, hidden, bias=False, dtype=torch.bfloat16)
        self.conditioner = GlobalConditioner(hidden)
        self.blocks = nn.ModuleDict(
            {f"slot_{i:02d}": _MockDiTBlock(hidden, inter) for i in range(n_blocks)}
        )
        self.modality_image = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))


class _MockComposite(nn.Module):
    """Reproduces the production FQN layout (TrainableComposite.dit.*)."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit


def test_routing_partition_disjoint_and_complete():
    hidden, inter, n_blocks = 256, 512, 3
    model = _MockComposite(_MockDiT(hidden, inter, n_blocks))
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    total = len(routing.full_audit.specs)
    cmuon = len(routing.cmuon_specs)
    adamw = len(routing.adamw_specs)
    assert cmuon + adamw == total, "partition must be complete"
    names_c = set(routing.cmuon_names)
    names_a = set(routing.adamw_names)
    assert not (names_c & names_a), "partition must be disjoint"
    # every block contributes 7 CMuon params; the conditioner contributes 1
    expected_cmuon = n_blocks * 7 + 1
    assert cmuon == expected_cmuon, (
        f"expected {expected_cmuon} cmuon params, got {cmuon}"
    )
    # chunk counts
    in_proj = [s for s in routing.cmuon_specs if s.name.endswith("in_proj.weight")]
    assert all(s.chunk_count == 2 for s in in_proj)
    shared = [
        s
        for s in routing.cmuon_specs
        if s.name.endswith("shared_block_projection.weight")
    ]
    assert len(shared) == 1 and shared[0].chunk_count == 6


# ---------------------------------------------------------------------------
# H. AdamW8bit -> Hybrid checkpoint transition (requires CUDA: AdamW8bit is
#    CUDA-only). Verifies: AdamW state preserved per-FQN for the AdamW params,
#    CMuon momentum from zero, no fake 2nd-moment -> Muon conversion.
# ---------------------------------------------------------------------------

_OPT_COMMON: dict = {
    "lr": LR,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "block_size": 256,
    "bf16_stochastic_round": True,
    "matrix_weight_decay": 0.0,
    "sensitive_weight_decay": 0.0,
    "sr_seed": 44,
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU")
def test_adamw_to_hybrid_transition():
    hidden, inter, n_blocks = 256, 512, 3
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(hidden, inter, n_blocks)).to(device)
    model = model.to(dtype=torch.bfloat16) if False else model  # keep mixed dtypes

    # baseline AdamW8bit over ALL params; run a few steps to fill the moments
    baseline = build_adamw8bit(model, **_OPT_COMMON)
    torch.manual_seed(7)
    for _ in range(3):
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p.float()).to(p.dtype)
        baseline.step()
        for p in model.parameters():
            p.grad = None
    baseline_state = baseline.state_dict()

    # hybrid; load the baseline state (transition path: no "cmuon" section)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps=5,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    hybrid.load_state_dict(baseline_state)

    assert hybrid.transition_from_adamw8bit is True
    routing = hybrid.routing
    # CMuon momentum must be exactly zero (from zero, no conversion)
    for spec in routing.cmuon_specs:
        m = hybrid._momenta[spec.parameter]
        assert torch.all(m == 0).item(), f"CMuon momentum not zero for {spec.name}"
    # AdamW state preserved: the inner AdamW8bit has state for every AdamW param
    preserved = 0
    for spec in routing.adamw_specs:
        st = hybrid.adamw.optimizer.state.get(spec.parameter)
        if st:
            preserved += 1
    assert preserved == len(routing.adamw_specs), (
        f"expected {len(routing.adamw_specs)} AdamW params preserved, got {preserved}"
    )
    # the transition record
    assert hybrid.transition_preserved_adamw_params == len(routing.adamw_specs)
    assert hybrid.transition_dropped_cmuon_params == len(routing.cmuon_specs)
    # a subsequent state_dict round-trips (Hybrid -> Hybrid)
    sd = hybrid.state_dict()
    assert sd["transition"]["from_adamw8bit"] is True
    hybrid2 = build_hybrid_cmuon(
        model,
        ns_steps=5,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    hybrid2.load_state_dict(sd)
    assert hybrid2.transition_from_adamw8bit is False  # Hybrid -> Hybrid is native
    for spec in routing.cmuon_specs:
        assert torch.equal(
            hybrid._momenta[spec.parameter], hybrid2._momenta[spec.parameter]
        )


# ---------------------------------------------------------------------------
# I. Per-spec (per-role) Newton-Schulz depth: resolution + validation
# ---------------------------------------------------------------------------

# Production shapes (rows, cols) keyed by canonical role. The ffn_in / adaln
# entries are the PER-CHUNK shape (gate/up == ffn_in; each AdaLN chunk ==
# adaln_shared). The full production-shape NS4 finiteness + ns5 instability
# are HCU-gated (CPU BF16 matmul is too slow / does not reproduce the HCU
# BF16 numerical artifact).
PRODUCTION_SHAPES = {
    "attention_q": (2560, 2560),
    "attention_k": (640, 2560),
    "attention_v": (640, 2560),
    "attention_content_gate": (2560, 2560),
    "attention_out": (2560, 2560),
    "ffn_in": (6912, 2560),
    "ffn_down": (2560, 6912),
    "adaln_shared": (2560, 1024),
}
TARGET_DELTA = 0.2 * LR


def _delta_rms(shape, ns_steps, *, seed=0, dtype=torch.bfloat16, device=None):
    """Applied parameter-delta RMS for one NS depth on an iid gradient.

    delta = -alpha * NS(grad); at step 1 the nesterov update is a scalar
    multiple of grad, so NS(grad) == NS(nesterov) (NS is scale-invariant after
    its one-time Frobenius normalization).

    ``device=None`` targets the accelerator when one is present: these
    bf16 invariants are platform dependent (the ns5 artifact is HCU-specific),
    and the CPU bf16 matmul path is orders of magnitude slower, so the
    "_hcu" tests must not silently fall back to CPU. Pass ``device="cpu"``
    explicitly for the CPU-pinned invariants.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=dtype).to(device)
    ns = cmuon_zeroth_power(x, ns_steps)
    alpha = cmuon_moonlight_alpha(shape[0], shape[1], LR, 1)
    return ((-alpha) * ns).pow(2).mean().sqrt().item()


def _reference_muon_ns(chunk, grad, *, lr, wd, ns):
    w = nn.Parameter(chunk.clone())
    w.grad = grad.clone()
    opt = torch.optim.Muon(
        [w],
        lr=lr,
        weight_decay=wd,
        momentum=MOMENTUM,
        nesterov=True,
        ns_steps=ns,
        adjust_lr_fn="match_rms_adamw",
    )
    opt.step()
    return w.data


def test_resolve_ns_map_uniform_and_override():
    m = resolve_ns_map(None, 4)
    assert m == {r: 4 for r in CMUON_ROLES}
    m = resolve_ns_map(None, None)  # default -> DEFAULT_NS_STEPS (5)
    assert set(m) == set(CMUON_ROLES) and all(v == 5 for v in m.values())
    m = resolve_ns_map({"ffn_in": 3, "adaln_shared": 3}, 4)
    assert m["ffn_in"] == 3 and m["adaln_shared"] == 3
    assert m["attention_q"] == 4 and m["ffn_down"] == 4
    m = resolve_ns_map({r: 2 for r in CMUON_ROLES}, 4)
    assert all(v == 2 for v in m.values())
    with pytest.raises(ValueError):
        resolve_ns_map({"bogus_role": 3}, 4)
    with pytest.raises(ValueError):
        resolve_ns_map({"ffn_in": 0}, 4)
    with pytest.raises(ValueError):
        resolve_ns_map({"ffn_in": 100}, 4)


def test_cmuon_config_ns_map_validation_and_lookup():
    m = resolve_ns_map({"ffn_in": 3}, 4)
    cfg = _cfg(ns_steps_by_role=m)
    assert cfg.ns_steps_for_role("ffn_in") == 3
    assert cfg.ns_steps_for_role("attention_q") == 4
    assert cfg.canonical_ns_map() == {r: m[r] for r in CMUON_ROLES}
    bad = dict(m)
    del bad["adaln_shared"]
    with pytest.raises(ValueError):
        CMuonConfig(lr=LR, ns_steps_by_role=bad)
    bad = dict(m)
    bad["ffn_down"] = 200
    with pytest.raises(ValueError):
        CMuonConfig(lr=LR, ns_steps_by_role=bad)


def test_chunk1_native_equivalence_ns234():
    """custom chunk=1 == native torch.optim.Muon for ns in (2,3,4)."""
    for ns in (2, 3, 4):
        for shape, dtype in [
            ((256, 256), torch.bfloat16),
            ((64, 256), torch.bfloat16),
            ((128, 128), torch.float32),
        ]:
            torch.manual_seed(ns)
            w_ref = nn.Parameter(torch.randn(*shape, dtype=dtype))
            w_mine = nn.Parameter(w_ref.data.clone())
            grad = torch.randn(*shape, dtype=dtype)
            w_ref.grad = grad.clone()
            ref_opt = torch.optim.Muon(
                [w_ref],
                lr=LR,
                weight_decay=0.0,
                momentum=MOMENTUM,
                nesterov=True,
                ns_steps=ns,
                adjust_lr_fn="match_rms_adamw",
            )
            ref_opt.step()
            mdtype = "float32" if dtype == torch.float32 else "bfloat16"
            param, spec = _make_spec("w", shape, dtype, 1, ("x",), role="attention_q")
            param.data.copy_(w_mine.data)
            cfg = _cfg(ns_steps=ns, momentum_dtype=mdtype, chunk_rescale_sqrt_n=False)
            buf = torch.zeros_like(param, dtype=getattr(torch, mdtype))
            _cmuon_update(param, grad, buf, spec, cfg)
            atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
            assert torch.allclose(param.data, w_ref.data, atol=atol, rtol=1e-2), (
                f"chunk=1 ns{ns} diverges from reference Muon at {shape}: "
                f"max diff {(param.data - w_ref.data).abs().max().item()}"
            )


def test_chunk2_per_spec_ns_equals_independent_muon():
    """FFN in_proj (ffn_in) with per-role ns=3 == concat of 2 independent Muon(ns3)."""
    torch.manual_seed(30)
    inter, hidden = 256, 256
    dtype = torch.bfloat16
    ns = 3
    fused = torch.randn(2 * inter, hidden, dtype=dtype)
    grad = torch.randn(2 * inter, hidden, dtype=dtype)
    gate_ref = _reference_muon_ns(fused[:inter], grad[:inter], lr=LR, wd=0.0, ns=ns)
    up_ref = _reference_muon_ns(fused[inter:], grad[inter:], lr=LR, wd=0.0, ns=ns)
    ref = torch.cat([gate_ref, up_ref], dim=0)
    param, spec = _make_spec(
        "mlp.in_proj.weight", fused.shape, dtype, 2, ("gate", "up"), role="ffn_in"
    )
    param.data.copy_(fused)
    cfg = _cfg(ns_steps=ns, momentum_dtype="bfloat16", chunk_rescale_sqrt_n=False)
    buf = torch.zeros_like(param, dtype=torch.bfloat16)
    _cmuon_update(param, grad, buf, spec, cfg)
    diff = (param.data - ref).abs().max().item()
    assert diff < 2e-2, f"chunk=2 per-spec ns{ns} diverges: max diff {diff}"


def test_chunk6_per_spec_ns_equals_independent_muon():
    """AdaLN (adaln_shared) with per-role ns=4 == concat of 6 independent Muon(ns4)."""
    torch.manual_seed(40)
    model_dim, hidden = 256, 128
    dtype = torch.float32
    ns = 4
    fused = torch.randn(6 * model_dim, hidden, dtype=dtype)
    grad = torch.randn(6 * model_dim, hidden, dtype=dtype)
    chunks_ref = [
        _reference_muon_ns(
            fused[i * model_dim : (i + 1) * model_dim],
            grad[i * model_dim : (i + 1) * model_dim],
            lr=LR,
            wd=0.0,
            ns=ns,
        )
        for i in range(6)
    ]
    ref = torch.cat(chunks_ref, dim=0)
    roles = ("a_scale", "a_shift", "a_gate", "m_scale", "m_shift", "m_gate")
    param, spec = _make_spec(
        "cond.weight", fused.shape, dtype, 6, roles, role="adaln_shared"
    )
    param.data.copy_(fused)
    cfg = _cfg(ns_steps=ns, momentum_dtype="float32", chunk_rescale_sqrt_n=False)
    buf = torch.zeros_like(param, dtype=torch.float32)
    _cmuon_update(param, grad, buf, spec, cfg)
    diff = (param.data - ref).abs().max().item()
    assert diff < 1e-4, f"chunk=6 per-spec ns{ns} diverges: max diff {diff}"


def test_kv_adaln_ns4_finite_and_scaled_cpu():
    """The two small-short-edge production shapes are SAFE at ns4 (~0.2*lr)."""
    for shape in [(640, 2560), (2560, 1024)]:
        r = _delta_rms(shape, 4, device="cpu")
        assert math.isfinite(r)
        assert r < 5 * TARGET_DELTA, (
            f"ns4 delta_rms for {shape} should be ~0.2*lr, got {r}"
        )


def _mock_dit_for_routing():
    hidden, inter, n_blocks = 2560, 6912, 1
    return _MockComposite(_MockDiT(hidden, inter, n_blocks))


def test_global_ns4_routing_roles():
    model = _mock_dit_for_routing()
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    for spec in routing.cmuon_specs:
        assert spec.role in CMUON_ROLES, f"unexpected role {spec.role} for {spec.name}"
    counts = {r: 0 for r in CMUON_ROLES}
    for spec in routing.cmuon_specs:
        counts[spec.role] += 1
    for r in CMUON_ROLES:
        assert counts[r] == 1, f"expected exactly 1 {r} param, got {counts[r]}"
    cfg = _cfg(ns_steps=4)
    for spec in routing.cmuon_specs:
        assert cfg.ns_steps_for_role(spec.role) == 4


def test_mixed_ns_routing_per_role_depth():
    model = _mock_dit_for_routing()
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    mixed = resolve_ns_map(
        {
            "attention_q": 3,
            "attention_k": 3,
            "attention_v": 3,
            "attention_content_gate": 3,
            "attention_out": 3,
            "ffn_in": 4,
            "ffn_down": 4,
            "adaln_shared": 4,
        },
        4,
    )
    cfg = _cfg(ns_steps_by_role=mixed)
    for spec in routing.cmuon_specs:
        want = 3 if spec.role.startswith("attention") else 4
        assert cfg.ns_steps_for_role(spec.role) == want, spec.role


# ---------------------------------------------------------------------------
# J. Full production shapes + ns5 instability + checkpoint (require HCU/CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="full production shapes need HCU (CPU BF16 too slow)",
)
def test_all_production_shapes_ns4_finite_hcu():
    for role, shape in PRODUCTION_SHAPES.items():
        r = _delta_rms(shape, 4)
        assert math.isfinite(r), f"ns4 not finite for {role} {shape}"
        assert r < 5 * TARGET_DELTA, f"ns4 delta_rms for {role} {shape} off target: {r}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ns5 instability is an HCU BF16 artifact"
)
def test_kv_adaln_ns5_finite_hcu():
    """ns4 safety (the deployed invariant) + ns5 finiteness on the HCU.

    The quintic NS is a BF16 numerical artifact that is spectrum/platform
    context dependent (in exact arithmetic f^5(sigma) converges for every
    normalized sigma). The 08-28 sweep (reports/cmuon-ns-depth-audit.md,
    24-seed characterization) documented an ns5 blow-up for the small
    short-edge shapes (k/v, adaln) on the then-current HCU context; the
    08-29 salt1 context (DTK 26.04 / torch 2.9.0) does NOT reproduce it
    (all 8 seeds finite at ~2.96e-05, i.e. ns4-like). The production
    candidate is ns4 either way, so this test guards the invariant that
    must hold on a healthy HCU: ns4 stays at the Moonlight target order and
    ns5 stays finite. A ns5 finiteness failure flags a platform regression
    worth re-running the full sweep before ever considering ns5.
    """
    for shape, tag in [((640, 2560), "k/v"), ((2560, 1024), "adaln")]:
        for seed in range(8):
            r4 = _delta_rms(shape, 4, seed=seed)
            assert math.isfinite(r4) and r4 < 5 * TARGET_DELTA, (
                f"ns4 not safe for {tag} seed={seed}: {r4}"
            )
            r5 = _delta_rms(shape, 5, seed=seed)
            assert math.isfinite(r5), f"ns5 not finite for {tag} seed={seed}: {r5}"


def _opt_common_lr(lr):
    d = dict(_OPT_COMMON)
    d["lr"] = lr
    return d


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_checkpoint_ns_map_roundtrip():
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    torch.manual_seed(11)
    for _ in range(2):
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p.float()).to(p.dtype)
        hybrid.step()
        hybrid.zero_grad(set_to_none=True)
    sd = hybrid.state_dict()
    assert sd["cmuon"]["ns_steps"] == ns4, "cmuon state must store the canonical NS map"
    # fresh model, same ns map -> exact restore
    model2 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid2 = build_hybrid_cmuon(
        model2,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    hybrid2.load_state_dict(sd)
    assert hybrid2.transition_from_adamw8bit is False
    m1 = sd["cmuon"]["momenta"]
    m2 = hybrid2.state_dict()["cmuon"]["momenta"]
    assert set(m1) == set(m2)
    for name in m1:
        assert torch.equal(m1[name], m2[name]), f"momentum mismatch for {name}"
    assert hybrid2.state_dict()["cmuon"]["ns_steps"] == ns4


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_checkpoint_ns_map_mismatch_hard_fail():
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=resolve_ns_map(None, 4),
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    torch.manual_seed(12)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p.float()).to(p.dtype)
    hybrid.step()
    sd = hybrid.state_dict()
    # a different per-role map (all 5) is a semantic incompatibility
    model2 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid2 = build_hybrid_cmuon(
        model2,
        ns_steps_by_role=resolve_ns_map(None, 5),
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    with pytest.raises(ValueError, match="ns_steps mismatch"):
        hybrid2.load_state_dict(sd)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_checkpoint_legacy_scalar_ns_migration():
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=resolve_ns_map(None, 4),
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    torch.manual_seed(13)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p.float()).to(p.dtype)
    hybrid.step()
    sd = hybrid.state_dict()
    # simulate a legacy checkpoint that stored a scalar ns_steps
    sd["cmuon"]["ns_steps"] = 4
    model2 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid2 = build_hybrid_cmuon(
        model2,
        ns_steps_by_role=resolve_ns_map(None, 4),
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    hybrid2.load_state_dict(sd)  # scalar 4 == all-4 map -> allowed
    assert hybrid2.state_dict()["cmuon"]["ns_steps"] == resolve_ns_map(None, 4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_lr_change_still_allows_resume():
    """LR is a variable training hyperparameter, not an optimizer-state semantic;
    a per-role NS map match must not be rejected when only the LR differs."""
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_opt_common_lr(LR),
    )
    torch.manual_seed(14)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p.float()).to(p.dtype)
    hybrid.step()
    sd = hybrid.state_dict()
    # different LR, same per-role NS map -> must NOT hard-fail
    model2 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid2 = build_hybrid_cmuon(
        model2,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_opt_common_lr(LR * 2),
    )
    hybrid2.load_state_dict(sd)  # must not raise


def test_ns_safety_telemetry_accumulates_and_logs():
    """The opt-in NS telemetry accumulates on-device (no per-step sync) and
    reports per-representative-role ns/delta RMS + nonfinite count every N
    steps; delta_rms == alpha * ns_rms exactly (delta = -alpha * ns)."""
    shape = (64, 256)
    param, spec = _make_spec("w", shape, torch.bfloat16, 1, ("x",), role="attention_k")
    cfg = _cfg(ns_steps=4, momentum_dtype="bfloat16", chunk_rescale_sqrt_n=False)
    tel = NSSafetyTelemetry({"attention_k": spec}, device=param.device, log_every_n=2)
    buf = torch.zeros_like(param, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    alpha = cmuon_moonlight_alpha(shape[0], shape[1], LR, 1)
    for i in range(4):
        grad = torch.randn(*shape, generator=g, dtype=torch.bfloat16)
        _cmuon_update(param, grad, buf, spec, cfg, ns_telemetry=tel)
        samples = tel.step()  # advance + maybe log
        if (i + 1) % 2 != 0:
            assert samples is None, "off-cycle step must not read/sync"
        else:
            assert samples is not None and len(samples) == 1
            s = samples[0]
            assert s.role == "attention_k"
            assert math.isfinite(s.ns_output_rms) and s.ns_output_rms > 0
            assert math.isfinite(s.applied_delta_rms)
            assert s.nonfinite_count == 0
            assert abs(s.applied_delta_rms - alpha * s.ns_output_rms) < 1e-3
    # default (no telemetry) leaves the update path untouched
    param2, spec2 = _make_spec(
        "w2", shape, torch.bfloat16, 1, ("x",), role="attention_k"
    )
    buf2 = torch.zeros_like(param2, dtype=torch.bfloat16)
    g2 = torch.Generator().manual_seed(99)
    grad = torch.randn(*shape, generator=g2, dtype=torch.bfloat16)
    d1 = _cmuon_update(param2, grad, buf2, spec2, cfg)  # no telemetry
    assert math.isfinite(d1.float().abs().max().item())


# ---------------------------------------------------------------------------
# K. Full checkpoint path (requires CUDA/HCU: AdamW8bit is CUDA-only).
#    Exercises the save-side schema/state builders + the load-side validators
#    exactly as production wires them:
#      - hybrid -> hybrid: file-level state-exact resume (lr/wd override,
#        routing manifest equality, momentum + AdamW state restored)
#      - AdamW -> hybrid: v1 full-audit schema/state validated, transition
#        loaded (per-FQN AdamW preservation, zero CMuon momentum), then the
#        transition checkpoint re-saved and resumed as a native hybrid
#      - tamper rejection: per-role NS mismatch, routing mismatch, schema
#        structural mismatches, outer-state key/version violations
#      - telemetry wiring: opt-in logger through step(), absolute update
#        numbers, per-role coverage, LR sync into the Moonlight scale
# ---------------------------------------------------------------------------

_TELEMETRY_LINE = re.compile(
    r"update=(\d+) role=(\w+) ns_rms=(\S+) delta_rms=(\S+) nonfinite=(\d+)"
)


def _write_optimizer_sidecars(root: Path, schema: dict, state_doc) -> Path:
    """Write train_state/{optimizer.pt, optimizer_schema.json} like the save path."""
    train_state = root / "train_state"
    train_state.mkdir(parents=True, exist_ok=True)
    torch.save(state_doc, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(json.dumps(schema))
    return train_state


def _seed_grads(model, seed: int) -> None:
    g = torch.Generator(device=model.dit.input_projection.weight.device).manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(
                *p.shape, generator=g, dtype=torch.float32, device=p.device
            ).to(p.dtype)


def _set_group_lr(optimizer, lr: float) -> None:
    """Set the per-group LR exactly like the production scheduler: torchao
    AdamW8bit keeps lr as a 0-dim Tensor after construction, so a plain float
    assignment is rejected at step() — write floats only where the current
    value is a float, otherwise fill_ the tensor in place."""
    for group in optimizer.param_groups:
        current = group["lr"]
        if isinstance(current, torch.Tensor):
            with torch.no_grad():
                current.fill_(lr)
        else:
            group["lr"] = lr


def _moment_equal(a, b) -> bool:
    """Value equality for an optimizer moment that may be a plain tensor or a
    torchao 8-bit OptimState. OptimState8bit is a Tensor SUBCLASS whose
    aten.equal dispatch is unimplemented, so it is compared via its raw
    codes/scale/qmap storage instead."""
    if type(a).__name__ == "OptimState8bit" or type(b).__name__ == "OptimState8bit":
        for attr in ("codes", "scale", "qmap"):
            x, y = getattr(a, attr, None), getattr(b, attr, None)
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                return False
            if x.to("cpu").shape != y.to("cpu").shape:
                return False
            if not x.to("cpu").equal(y.to("cpu")):
                return False
        return True
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.detach().cpu().equal(b.detach().cpu())
    return False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_hybrid_checkpoint_full_path_resume(tmp_path):
    """Hybrid -> hybrid: the file-level resume is state-exact and the
    runtime lr/wd override the saved per-group values (same contract as v1)."""
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    for seed in (21, 22):
        _seed_grads(model, seed)
        hybrid.step()
    schema = _hybrid_optimizer_schema(hybrid)
    outer = hybrid.state_dict()
    train_state = _write_optimizer_sidecars(tmp_path / "good", schema, outer)

    model2 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid2 = build_hybrid_cmuon(
        model2,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_opt_common_lr(LR * 2),
    )
    # The scheduler/checkpoint-restore only writes the inner torch groups:
    # the resumed run starts from a DIFFERENT (current) lr/weight-decay.
    _set_group_lr(hybrid2.adamw.optimizer, LR * 2)
    for group in hybrid2.adamw.optimizer.param_groups:
        group["weight_decay"] = 0.01

    _load_hybrid_state_exact(
        train_state, schema, hybrid2, hybrid.sr_rng.state_dict(), successful_updates=2
    )

    assert hybrid2.transition_from_adamw8bit is False
    # CMuon momentum restored exactly (bitwise, BF16) for every allowlisted
    # param. The two optimizers own DIFFERENT parameter objects (model vs
    # model2), so compare spec-pairs aligned by FQN.
    by_name_2 = {spec.name: spec for spec in hybrid2.routing.cmuon_specs}
    for spec in hybrid.routing.cmuon_specs:
        spec2 = by_name_2[spec.name]
        before = hybrid._momenta[spec.parameter].detach().cpu()
        after = hybrid2._momenta[spec2.parameter].detach().cpu()
        assert torch.equal(before, after), f"momentum mismatch for {spec.name}"
    # AdamW state restored per-FQN with the right step count.
    for spec in hybrid2.routing.adamw_specs:
        st = hybrid2.adamw.optimizer.state.get(spec.parameter)
        assert st is not None, f"AdamW state missing for {spec.name}"
        assert float(st["step"].item()) == 2.0, spec.name
    # lr/wd come from the CURRENT config, not the saved groups. (lr rides a
    # float32 tensor in the group, so compare with float32 tolerance.)
    for group in hybrid2.adamw.optimizer.param_groups:
        lr_now = group["lr"]
        lr_now = float(lr_now.item()) if isinstance(lr_now, torch.Tensor) else lr_now
        assert math.isclose(lr_now, LR * 2, rel_tol=1e-6) and group["weight_decay"] == 0.01
    # The routing contract is state-exact.
    assert hybrid2.routing.routing_manifest() == outer["routing"]
    # The CMuon algorithm scalars followed the checkpoint.
    assert hybrid2.cfg.canonical_ns_map() == ns4
    assert hybrid2.cfg.momentum_dtype == "bfloat16"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_adamw_to_hybrid_checkpoint_full_path(tmp_path):
    """AdamW -> hybrid FORK: a pure full-param AdamW8bit v1 checkpoint is
    validated against the FULL audit, transitioned (AdamW state preserved
    per-FQN, CMuon momentum from zero), and the resulting checkpoint resumes
    natively as a hybrid."""
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    baseline = build_adamw8bit(model, **_OPT_COMMON)
    for seed in (31, 32, 33):
        _seed_grads(model, seed)
        baseline.step()
        for p in model.parameters():
            p.grad = None

    schema_v1 = _optimizer_schema(baseline)
    v1_state = baseline.optimizer.state_dict()
    # Production reads v1 state from the checkpoint file (torch.save moves the
    # tensors to CPU and load reconstructs the 8-bit state subclass); round-trip
    # through a file so the validators see the production form, not the live
    # HCU-resident state.
    buf = io.BytesIO()
    torch.save(v1_state, buf)
    buf.seek(0)
    # map_location="cpu" is required: the weights_only unpickler rebuilds the
    # 8-bit TensorSubclass on its recorded device (cuda) otherwise — the
    # production load path (load.py) always passes map_location="cpu".
    v1_state = torch.load(buf, map_location="cpu", weights_only=True)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    _validate_transition_optimizer_schema(schema_v1, hybrid.audit)
    _validate_transition_optimizer_state(v1_state, hybrid.audit, 3)
    hybrid.load_state_dict(
        {"optimizer": v1_state, "sr_rng": baseline.sr_rng.state_dict()}
    )

    assert hybrid.transition_from_adamw8bit is True
    assert hybrid.transition_preserved_adamw_params == len(hybrid.routing.adamw_specs)
    assert hybrid.transition_dropped_cmuon_params == len(hybrid.routing.cmuon_specs)
    for spec in hybrid.routing.cmuon_specs:
        assert torch.all(hybrid._momenta[spec.parameter] == 0).item(), (
            f"CMuon momentum not from zero for {spec.name}"
        )
    for spec in hybrid.routing.adamw_specs:
        st = hybrid.adamw.optimizer.state.get(spec.parameter)
        assert st is not None, f"AdamW state missing for {spec.name}"
        assert float(st["step"].item()) == 3.0, spec.name
    # Per-FQN correctness (regression guard): each preserved AdamW moment must
    # be the moment of the SAME-named baseline param, not a neighbor's. The
    # baseline torch state id follows the flattened [decay, sensitive] group
    # order; index the saved state by that id and compare exp_avg values.
    name_to_baseline_id: dict[str, int] = {}
    for rank, spec in enumerate(hybrid.audit.decay):
        name_to_baseline_id[spec.name] = rank
    offset = len(hybrid.audit.decay)
    for rank, spec in enumerate(hybrid.audit.sensitive):
        name_to_baseline_id[spec.name] = offset + rank
    for spec in hybrid.routing.adamw_specs:
        baseline_state = v1_state["state"][name_to_baseline_id[spec.name]]
        preserved = hybrid.adamw.optimizer.state[spec.parameter]
        assert _moment_equal(baseline_state["exp_avg"], preserved["exp_avg"]), (
            f"exp_avg preserved from the wrong baseline param: {spec.name}"
        )
        assert _moment_equal(baseline_state["exp_avg_sq"], preserved["exp_avg_sq"]), (
            f"exp_avg_sq preserved from the wrong baseline param: {spec.name}"
        )

    # The forked checkpoint is a NATIVE hybrid checkpoint (schema v2) and
    # resumes state-exactly into a fresh hybrid.
    schema_v2 = _hybrid_optimizer_schema(hybrid)
    outer = hybrid.state_dict()
    train_state = _write_optimizer_sidecars(tmp_path / "fork", schema_v2, outer)
    model3 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid3 = build_hybrid_cmuon(
        model3,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    _load_hybrid_state_exact(
        train_state, schema_v2, hybrid3, hybrid.sr_rng.state_dict(), successful_updates=3
    )
    assert hybrid3.transition_from_adamw8bit is False
    for spec in hybrid3.routing.cmuon_specs:
        assert torch.all(hybrid3._momenta[spec.parameter] == 0).item()
    for spec in hybrid3.routing.adamw_specs:
        st = hybrid3.adamw.optimizer.state.get(spec.parameter)
        assert st is not None and float(st["step"].item()) == 3.0, spec.name


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_hybrid_checkpoint_tamper_rejection(tmp_path):
    """State-exact resume must HARD-FAIL on semantic tampering: a per-role
    NS depth change, a routing-manifest change, an outer-state key change,
    and a schema/outer mismatch. No silent optimizer-state downgrade."""
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    _seed_grads(model, 41)
    hybrid.step()
    schema = _hybrid_optimizer_schema(hybrid)
    outer = hybrid.state_dict()

    def fresh_hybrid():
        m = _MockComposite(_MockDiT(256, 512, 3)).to(device)
        return build_hybrid_cmuon(
            m,
            ns_steps_by_role=ns4,
            momentum_dtype="bfloat16",
            chunk_rescale_sqrt_n=False,
            **_OPT_COMMON,
        )

    # 1) Per-role NS depth tampered in the saved CMuon state (semantic).
    cmuon_bad = dict(outer["cmuon"])
    ns_bad = dict(cmuon_bad["ns_steps"])
    ns_bad["ffn_in"] = 3
    cmuon_bad["ns_steps"] = ns_bad
    with pytest.raises(CheckpointError):
        _validate_hybrid_cmuon_state(cmuon_bad, fresh_hybrid())

    # 2) Per-role NS depth tampered in the schema contract.
    schema_bad = dict(schema)
    block = dict(schema["hybrid_cmuon"])
    ns_bad2 = dict(block["ns_steps"])
    ns_bad2["adaln_shared"] = 6
    block["ns_steps"] = ns_bad2
    schema_bad["hybrid_cmuon"] = block
    with pytest.raises(CheckpointError):
        _validate_hybrid_optimizer_schema(schema_bad, fresh_hybrid())

    # 3) Momentum dtype tampered in the schema contract.
    block2 = dict(schema["hybrid_cmuon"])
    block2["momentum_dtype"] = "float32"
    schema_bad2 = dict(schema)
    schema_bad2["hybrid_cmuon"] = block2
    with pytest.raises(CheckpointError):
        _validate_hybrid_optimizer_schema(schema_bad2, fresh_hybrid())

    # 4) A pure-AdamW v1 schema document is not a hybrid v2 schema.
    model_v1 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    baseline = build_adamw8bit(model_v1, **_OPT_COMMON)
    with pytest.raises(CheckpointError):
        _validate_hybrid_optimizer_schema(_optimizer_schema(baseline), fresh_hybrid())

    # 5) Routing manifest tampered in the outer state (file-level load path).
    routing_bad = dict(outer["routing"])
    routing_bad["cmuon"] = [dict(e) for e in outer["routing"]["cmuon"]][1:]
    outer_bad = {**outer, "routing": routing_bad}
    train_state = _write_optimizer_sidecars(tmp_path / "tamper", schema, outer_bad)
    with pytest.raises(CheckpointError):
        _load_hybrid_state_exact(
            train_state, schema, fresh_hybrid(), hybrid.sr_rng.state_dict(), 1
        )

    # 6) Outer-state top-level structure violations (file-level load path).
    outer_missing = {k: v for k, v in outer.items() if k != "transition"}
    train_state2 = _write_optimizer_sidecars(tmp_path / "badkeys", schema, outer_missing)
    with pytest.raises(CheckpointError):
        _load_hybrid_optimizer_state(train_state2 / "optimizer.pt")
    outer_ver = {**outer, "hybrid_cmuon_schema_version": 2}
    train_state3 = _write_optimizer_sidecars(tmp_path / "badver", schema, outer_ver)
    with pytest.raises(CheckpointError):
        _load_hybrid_optimizer_state(train_state3 / "optimizer.pt")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_hybrid_checkpoint_schema_rejects_group_mismatch(tmp_path):
    """The v2 schema validates the INNER AdamW group contract exactly: a
    reordered/renamed param_names list inside a group is a checkpoint/
    optimizer boundary mismatch and must hard-fail."""
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    schema = _hybrid_optimizer_schema(hybrid)
    groups = [dict(g) for g in schema["groups"]]
    # Pick whichever inner group carries >=2 params (the mock's matrix_decay
    # group may hold a single param).
    idx = next(i for i, g in enumerate(groups) if len(g["param_names"]) >= 2)
    names = list(groups[idx]["param_names"])
    names[0], names[1] = names[1], names[0]
    groups[idx]["param_names"] = names
    schema_bad = dict(schema)
    schema_bad["groups"] = groups
    m2 = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid2 = build_hybrid_cmuon(
        m2,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    with pytest.raises(CheckpointError):
        _validate_hybrid_optimizer_schema(schema_bad, hybrid2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_hybrid_telemetry_wired_into_step():
    """The opt-in NS safety telemetry is wired through HybridCMuon.step():
    absolute update numbers (offset aware), one batched log every N steps,
    full 8-role coverage, zero nonfinite, order-of-magnitude delta RMS, and
    the Moonlight scale tracks the inner optimizer's scheduled LR."""
    device = torch.device("cuda:0")
    ns4 = resolve_ns_map(None, 4)
    logs: list[str] = []
    model = _MockComposite(_MockDiT(256, 512, 3)).to(device)
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=ns4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        ns_telemetry_enabled=True,
        ns_telemetry_log_every_n=2,
        ns_telemetry_update_offset=100,
        ns_telemetry_logger=logs.append,
        **_OPT_COMMON,
    )
    tel = hybrid.ns_telemetry
    assert tel is not None
    assert set(tel.roles) == set(CMUON_ROLES), "default roles must cover all 8"
    assert len(logs) == 0, "construction must not log"

    for seed in (51, 52, 53, 54):
        _seed_grads(model, seed)
        hybrid.step()

    # log_every_n=2 -> one batched log after steps 2 and 4 (updates 102, 104).
    parsed = []
    for line in logs:
        m = _TELEMETRY_LINE.fullmatch(line.removeprefix("[cmuon-ns-telemetry] "))
        assert m is not None, f"unparseable telemetry line: {line!r}"
        parsed.append(tuple(m.groups()))
    assert len(parsed) == 2 * len(CMUON_ROLES), (
        f"expected {2 * len(CMUON_ROLES)} lines (2 cycles x 8 roles), got {len(parsed)}"
    )
    updates = {int(p[0]) for p in parsed}
    assert updates == {102, 104}, f"absolute update numbers wrong: {updates}"
    for update, role, ns_rms, delta_rms, nonfinite in parsed:
        assert role in CMUON_ROLES
        assert float(ns_rms) > 0.0, f"ns_rms not positive for {role} @ {update}"
        assert 0.0 < float(delta_rms) < 5e-3, (
            f"delta_rms for {role} @ {update} off Moonlight order: {delta_rms}"
        )
        assert int(nonfinite) == 0, f"nonfinite NS output for {role} @ {update}"
    for u in (102, 104):
        assert {p[1] for p in parsed if int(p[0]) == u} == set(CMUON_ROLES)

    # The scheduler only writes the inner torch groups; step() must sync the
    # Moonlight scale from them (cfg.lr follows the scheduled rate).
    new_lr = LR * 3
    _set_group_lr(hybrid.adamw.optimizer, new_lr)
    _seed_grads(model, 55)
    hybrid.step()
    # The inner lr group holds a float32 tensor; the sync reads it back through
    # float32, so compare with float32 tolerance, not exact equality.
    assert math.isclose(hybrid.cfg.lr, new_lr, rel_tol=1e-6), (
        f"Moonlight scale did not follow the inner LR: {hybrid.cfg.lr} != {new_lr}"
    )
    # step 5 is off-cycle: no extra log lines.
    assert len(logs) == 2 * len(CMUON_ROLES)


# ---------------------------------------------------------------------------
# L. Forensic instrumentation (fail-closed guard + two-phase equivalence)
# ---------------------------------------------------------------------------


def test_traced_ns_matches_plain_ns():
    """The traced NS (forensic) is bit-identical to the plain NS; the trace
    holds one finite fp32 norm per iteration."""
    g = torch.Generator().manual_seed(3)
    coeffs = (3.4445, -4.7750, 2.0315)
    for shape in [(48, 24), (24, 48), (32, 32)]:
        x = torch.randn(*shape, generator=g, dtype=torch.bfloat16)
        plain = cmuon_zeroth_power(x, NS_STEPS, coeffs, 1e-7)
        trace: list[torch.Tensor] = []
        traced = cmuon_zeroth_power_traced(x, NS_STEPS, coeffs, 1e-7, trace)
        assert torch.equal(plain, traced), f"traced NS diverged from plain NS at {shape}"
        assert len(trace) == NS_STEPS, "one trace entry per NS iteration"
        assert all(bool(torch.isfinite(t).all()) for t in trace), "trace norms finite"


def test_two_phase_matches_original_update():
    """Phase1+phase2 (forensic) must reproduce the original single-phase
    update bit-exactly: same momentum buffer, same parameter, for both
    nesterov modes and with/without weight decay."""
    g = torch.Generator().manual_seed(4)
    for shape in [(64, 128), (128, 64)]:
        for wd in (0.0, 0.01):
            for nesterov in (True, False):
                p_orig = nn.Parameter(
                    torch.randn(*shape, generator=g, dtype=torch.bfloat16)
                )
                p_two = nn.Parameter(p_orig.detach().clone())
                grad = torch.randn(*shape, generator=g, dtype=torch.bfloat16)
                spec_o = CMuonChunkSpec(
                    name="orig", parameter=p_orig, weight_decay=wd,
                    chunk_count=1, chunk_dim=0, roles=("attention_q",), role="attention_q",
                )
                spec_t = CMuonChunkSpec(
                    name="two", parameter=p_two, weight_decay=wd,
                    chunk_count=1, chunk_dim=0, roles=("attention_q",), role="attention_q",
                )
                cfg = _cfg(ns_steps=4, nesterov=nesterov)
                buf_o = torch.zeros_like(p_orig, dtype=torch.bfloat16)
                buf_t = torch.zeros_like(p_two, dtype=torch.bfloat16)
                _cmuon_update(p_orig, grad.clone(), buf_o, spec_o, cfg)
                buf_cand, _nesterov_t, _ns_full, delta, trace = _cmuon_update_phase1(
                    p_two, grad.clone(), buf_t, spec_t, cfg
                )
                _cmuon_update_phase2(p_two, buf_t, buf_cand, delta, spec_t, cfg)
                assert torch.equal(buf_o, buf_t), (
                    f"momentum mismatch {shape} wd={wd} nesterov={nesterov}"
                )
                assert torch.equal(p_orig, p_two), (
                    f"parameter mismatch {shape} wd={wd} nesterov={nesterov}"
                )
                # sanity: phase1 outputs are consistent with the applied delta
                assert delta.dtype == torch.bfloat16
                assert len(trace) == 4


def test_routing_exclude_roles():
    """cmuon_routing_exclude moves the excluded roles to the AdamW fallback;
    the partition stays disjoint + complete; unknown roles are rejected."""
    model = _MockComposite(_MockDiT(256, 512, 3))
    common = {"matrix_weight_decay": 0.0, "sensitive_weight_decay": 0.0}
    full = route_cmuon_parameters(model, **common)
    assert len(full.cmuon_specs) == 3 * 7 + 1
    excl = route_cmuon_parameters(
        model, exclude_roles=("attention_k", "attention_v", "adaln_shared"), **common
    )
    # 3 blocks: 3 k_proj + 3 v_proj + 1 shared AdaLN moved to AdamW
    assert len(excl.cmuon_specs) == len(full.cmuon_specs) - 7
    assert len(excl.adamw_specs) == len(full.adamw_specs) + 7
    names = {s.name for s in excl.cmuon_specs}
    assert not any("k_proj" in n or "v_proj" in n for n in names)
    assert not any("shared_block_projection" in n for n in names)
    with pytest.raises(ValueError, match="unknown cmuon exclude roles"):
        route_cmuon_parameters(model, exclude_roles=("bogus",), **common)


def _build_forensic_hybrid(tmp_path, *, logs, **overrides):
    model = _MockComposite(_MockDiT(256, 512, 3)).to(torch.device("cuda:0"))
    hybrid = build_hybrid_cmuon(
        model,
        ns_steps_by_role=resolve_ns_map(None, 4),
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        ns_telemetry_logger=logs.append,
        forensic=ForensicConfig(enabled=True, ring_size=10, dump_dir=str(tmp_path)),
        **overrides,
        **_OPT_COMMON,
    )
    assert hybrid.forensic is not None
    return model, hybrid


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_forensic_healthy_steps_and_step0_check(tmp_path):
    """Healthy grads run through the two-phase step without tripping: the
    step-0 momentum check logs ZERO-OK, the ring accumulates one entry per
    step, and the applied-delta RMS stays in Moonlight order."""
    logs: list[str] = []
    model, hybrid = _build_forensic_hybrid(tmp_path, logs=logs)
    mon = hybrid.forensic
    for seed in (51, 52, 53, 54):
        _seed_grads(model, seed)
        hybrid.step()
        hybrid.zero_grad(set_to_none=True)
    step0_lines = [l for l in logs if "step0 momentum check" in l]
    assert len(step0_lines) == 1, f"step0 check must log exactly once: {step0_lines}"
    assert "ZERO-OK" in step0_lines[0]
    assert len(mon._ring) == 4, "one ring entry per successful step"
    for entry in mon._ring:
        for i in range(mon.n_specs):
            row = entry["fp"][i]
            delta_rms = row[12]
            assert 0.0 < delta_rms < 5e-3, f"delta_rms off Moonlight order: {delta_rms}"
    assert not (tmp_path / "cmuon-forensic-crash-1.json").exists(), (
        "no dump on a healthy run"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_forensic_guard_trips_on_nonfinite_ns(tmp_path, monkeypatch):
    """A nonfinite NS output (from a FINITE gradient) must trip the
    fail-closed guard BEFORE any parameter write: CMuonSafetyError (not
    FloatingPointError), parameters bit-unchanged, crash dump written with
    the offending spec + the 10-step ring."""

    def evil_ns(grad, ns_steps, coefficients, eps, trace):
        trace.append(torch.tensor(float("inf")))
        return torch.full_like(grad, float("inf"))

    monkeypatch.setattr(cmuon_mod, "cmuon_zeroth_power_traced", evil_ns)
    logs: list[str] = []
    model, hybrid = _build_forensic_hybrid(tmp_path, logs=logs)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _seed_grads(model, 51)
    with pytest.raises(CMuonSafetyError):
        hybrid.step()
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p), f"fail-closed must not write {n}"
    dump = tmp_path / "cmuon-forensic-crash-1.json"
    assert dump.exists(), "crash dump must be written at the trip step"
    payload = json.loads(dump.read_text())
    assert payload["first_local_failure"]["kind"] == "nonfinite"
    assert payload["update"] == 1
    assert len(payload["ring"]) >= 1
    # the offending spec's phase-1 tensors are saved alongside the JSON
    tensor_files = [p for p in tmp_path.iterdir() if p.suffix == ".pt"]
    assert len(tensor_files) == 1
    blob = torch.load(tensor_files[0], weights_only=False)
    assert blob["ns_output"].shape == model.dit.blocks["slot_00"].attention.q_proj.weight.shape
    assert not torch.isfinite(blob["ns_output"].float()).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_hybrid_cmuon requires CUDA/HCU"
)
def test_forensic_guard_trips_on_delta_rms_ceiling(tmp_path, monkeypatch):
    """A finite-but-catastrophic NS output (delta RMS far above
    10 x 0.2 x lr) must also trip the guard, with the parameters untouched."""

    def evil_ns(grad, ns_steps, coefficients, eps, trace):
        trace.append(grad.float().norm())
        return (grad.float() * 1e9).to(torch.bfloat16)

    monkeypatch.setattr(cmuon_mod, "cmuon_zeroth_power_traced", evil_ns)
    logs: list[str] = []
    model, hybrid = _build_forensic_hybrid(tmp_path, logs=logs)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _seed_grads(model, 52)
    with pytest.raises(CMuonSafetyError):
        hybrid.step()
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p), f"fail-closed must not write {n}"
    dump = tmp_path / "cmuon-forensic-crash-1.json"
    assert dump.exists()
    payload = json.loads(dump.read_text())
    assert payload["first_local_failure"]["kind"] == "delta_rms_ceiling"
    assert payload["first_local_failure"]["delta_rms"] > payload["first_local_failure"]["ceiling"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
