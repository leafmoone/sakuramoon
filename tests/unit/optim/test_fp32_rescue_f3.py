"""F3 soft below-floor FP32 rescue — state-machine tests.

Covers the F3 semantic change (reports/cmuon-f3-soft-below-floor.md):
  finite FP32 below-floor  -> ACCEPT ORIGINAL DELTA + soft telemetry
  finite zero FP32 delta   -> ACCEPT + zero_delta_soft_rescue
  nonfinite / above ceiling -> hard fail (preserved, unchanged)

Determinism: the FP32 NS result is constructed directly (spec section 7:
"can construct the FP32 rescue result directly or use a deterministic
test harness") so test classification never depends on HCU BF16 addmm
nondeterminism. The BF16 failure is forced with a deterministic constant
NS stand-in (finite, always above ceiling).

Boundary construction: the delta rms is ``(-alpha)*v`` for a constant NS
``v``; a deterministic bisection over the v-ulp lattice (on the same
device the production read-back uses) lands the rms within a couple of
fp32 ulps of the requested target. Targets are chosen with comfortable
margin from every boundary (0.99x / 1.01x / 2x multiples), so the
classification side is unambiguous. The strict ``<`` comparison itself
is exercised by the 0.99x (soft) vs 1.01x (not soft) pair.

  A. exact zero            -> accept, zero_delta_soft_rescue
  B. in-band (2x floor)    -> accept, NOT soft
  C. just above floor      -> accept, NOT soft
  D. just below floor      -> accept unchanged + below_floor_soft_rescue
  E. above ceiling         -> hard fail, capsule, zero commit (preserved)
  F. nonfinite             -> hard fail, capsule, zero commit (preserved)
  G. synthetic matrix sweep (REAL fp32 NS): zero / constant / rank-1 /
     rank-8 / weak low-rank / full-rank -> all accepted, no capsule
  H. committed value == reference built from the ORIGINAL fp32 rescue +
     the existing single bf16 rounding (no amplification/clamp/normalize)
  I. checkpoint schema unchanged (no new persisted keys)
  J. floor numeric value unchanged (0.05)
  K. hard-fail path must NOT emit soft telemetry
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from test_cmuon import (
    LR,
    _MockComposite,
    _MockDiT,
    _seed_grads,
)

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.optim.cmuon import (
    cmuon_moonlight_alpha,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import (
    HybridCMuonCanonicalNS4FP32Rescue,
    build_fp32_rescue,
)
from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig

NS4 = resolve_ns_map(None, 4)
TARGET = 0.2 * LR
CEILING = 10.0 * TARGET
FLOOR = fr._RESCUE_SANITY_LOW * TARGET  # the diagnostic boundary


def _guard() -> GuardedCanonicalGuardConfig:
    return GuardedCanonicalGuardConfig(
        guard_ratio=0.05,
        reference_decay=0.999,
        min_reference=1e-12,
        numerical_floor=1e-20,
        warmup_observations=0,
        invariant_check=True,
    )


def _bootstrap_refs(model) -> dict[str, float]:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    refs: dict[str, float] = {}
    for spec in routing.cmuon_specs:
        g = torch.randn_like(spec.parameter)
        for ci in range(spec.chunk_count):
            chunk_size = spec.chunk_size()
            if spec.chunk_count == 1:
                sig = g.float().pow(2).mean().sqrt().item()
            else:
                start = ci * chunk_size
                end = start + chunk_size
                sl = [slice(None)] * g.ndim
                sl[spec.chunk_dim] = slice(start, end)
                sig = g[tuple(sl)].float().pow(2).mean().sqrt().item()
            refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)
    return refs


def _build(
    model,
    *,
    rank: int = 0,
    world_size: int = 1,
    logs: list[str] | None = None,
    local_capsule_root: str | None = None,
    mirror_root: str | None = None,
    forensic_dir: str | None = None,
) -> HybridCMuonCanonicalNS4FP32Rescue:
    return build_fp32_rescue(
        model,
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
        sr_seed=44,
        ns_steps_by_role=NS4,
        guard_cfg=_guard(),
        guard_bootstrap_refs=_bootstrap_refs(model),
        rank=rank,
        world_size=world_size,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        stats_logger=(logs.append if logs is not None else None),
        # F2/F3 semantics: emergency_capsule_root = LOCAL durable publish;
        # hard_fail_artifact_root = BEST-EFFORT shared mirror. Kept on
        # distinct paths so a local+mirror collision cannot fake -r2
        # event names.
        hard_fail_artifact_root=(str(mirror_root) if mirror_root else None),
        legacy_forensic_dir=(str(forensic_dir) if forensic_dir else None),
        emergency_capsule_root=(str(local_capsule_root) if local_capsule_root else None),
    )


def _n_inputs(model) -> int:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    return sum(spec.chunk_count for spec in routing.cmuon_specs)


def _alpha_for(shape) -> float:
    """The Moonlight alpha step() would apply to a chunk of this shape
    (rescale = 1 in this harness: chunk_rescale_sqrt_n=False)."""
    return cmuon_moonlight_alpha(shape[0], shape[1], LR, 1)


def _evil_bf16_const(value: float = 1e9):
    """Deterministic BF16 failure stand-in: finite, far above ceiling.

    Replaces the HCU-nondeterministic chaos stand-in so the BF16 side of
    every scenario is bit-deterministic."""

    def evil(grad, ns_steps, ns_coefficients, eps):
        return torch.full(
            tuple(grad.shape), value, dtype=torch.bfloat16, device=grad.device
        )

    return evil


def _make_const_ns(target: float):
    """An FP32 NS override returning a per-shape constant matrix whose
    ``(-alpha) * v`` delta rms lands as close to ``target`` as the fp32
    read-back chain allows (deterministic bisection over the v-ulp
    lattice, on the same device the production verdict reads back from —
    no randomness). ``ns._v_for_shape(shape, device)`` exposes the solved
    constants for reference computations."""
    cache: dict[tuple[int, ...], float] = {}

    def solve(shape, device) -> float:
        key = tuple(shape)
        if key in cache:
            return cache[key]
        alpha = _alpha_for(key)
        lo = 0.0
        hi = float(torch.tensor(4.0 * target / alpha, dtype=torch.float32))
        best_v, best_r = None, None
        for _ in range(64):
            mid = (lo + hi) / 2.0
            v = float(torch.tensor(mid, dtype=torch.float32))
            d = (-alpha) * torch.full(key, v, dtype=torch.float32, device=device)
            r = d.pow(2).mean().sqrt().item()
            if best_r is None or abs(r - target) < abs(best_r - target):
                best_v, best_r = v, r
            if r == target:
                break
            if r < target:
                lo = v
            else:
                hi = v
        cache[key] = best_v
        return best_v

    def ns(grad, ns_steps, ns_coefficients, eps):
        v = solve(tuple(grad.shape), grad.device)
        return torch.full(tuple(grad.shape), v, dtype=torch.float32, device=grad.device)

    ns._v_for_shape = solve  # type: ignore[attr-defined]
    return ns


def _param_snapshot(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def _snap_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(k in a and a[k].equal(b[k]) for k in b)


def _capsule_events(root: str) -> list[str]:
    p = Path(root)
    return sorted(e.name for e in p.iterdir()) if p.is_dir() else []


def _staged_for_chunk(const_ns, shape, device) -> torch.Tensor:
    """The exact staged delta step() would build for a constant-NS chunk:
    original fp32 delta -> ONE bf16 rounding (the existing path)."""
    alpha = _alpha_for(shape)
    v = const_ns._v_for_shape(tuple(shape), device)  # type: ignore[attr-defined]
    return ((-alpha) * torch.full(tuple(shape), v, dtype=torch.float32, device=device)).bfloat16()


def _chunk_shape(spec, ci: int) -> tuple[int, ...]:
    if spec.chunk_count == 1:
        return tuple(spec.parameter.shape)
    chunk_size = spec.chunk_size()
    shape = list(spec.parameter.shape)
    shape[spec.chunk_dim] = chunk_size
    return tuple(shape)


# ---------------------------------------------------------------------------
# A. exact zero
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_A_zero_delta_soft_rescue(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    before = _param_snapshot(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(
        fr, "cmuon_zeroth_power_fp32", lambda *a, **k: torch.zeros_like(a[0])
    )
    logs: list[str] = []
    local = tmp_path / "local"
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 101)
    opt.step()  # F3: must NOT raise
    assert opt.observations == 1
    assert opt.bf16_safety_failures == n
    assert opt.fp32_attempts == n
    assert opt.fp32_rescues == n
    assert opt.fp32_rescue_failures == 0
    assert opt.fp32_low_delta_rescues == n
    assert sum(opt.fp32_low_delta_by_role.values()) == n
    soft = [l for l in logs if l.startswith("[fp32-soft-rescue]")]
    assert len(soft) == n
    assert all("reason=zero_delta_soft_rescue" in l for l in soft)
    # zero staged delta -> CMuon parameters byte-unchanged (weight decay 0)
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    cmuon_names = set(routing.cmuon_names)
    after = _param_snapshot(model)
    for name in cmuon_names:
        assert before[name].equal(after[name]), f"{name} must be unchanged"
    # NO hard-fail capsule, no legacy forensic dump
    assert _capsule_events(str(local)) == []
    assert not (tmp_path / "fx" / "guard-forensic-rank0.json").exists()
    # telemetry JSON schema carries the soft counters
    rescue_lines = [l for l in logs if l.startswith('{"fp32_rescue_obs"')]
    assert rescue_lines
    rec = json.loads(rescue_lines[0])
    assert rec["fp32_low_delta_rescues"] == n
    assert "fp32_low_delta_by_role" in rec


# ---------------------------------------------------------------------------
# B. in-band (2x floor): accepted, NOT soft
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_B_in_band_accepted_not_soft(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", _make_const_ns(2.0 * FLOOR))
    logs: list[str] = []
    local = tmp_path / "local"
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 102)
    opt.step()
    assert opt.observations == 1
    assert opt.fp32_rescues == n
    assert opt.fp32_rescue_failures == 0
    assert opt.fp32_low_delta_rescues == 0, "in-band rescue must not be soft"
    assert not [l for l in logs if l.startswith("[fp32-soft-rescue]")]
    assert _capsule_events(str(local)) == []


# ---------------------------------------------------------------------------
# C. just above floor (1.01x): accepted, NOT soft
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_C_just_above_floor_accepted_not_soft(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", _make_const_ns(1.01 * FLOOR))
    logs: list[str] = []
    local = tmp_path / "local"
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 103)
    opt.step()
    assert opt.observations == 1
    assert opt.fp32_rescues == n
    assert opt.fp32_rescue_failures == 0
    assert opt.fp32_low_delta_rescues == 0
    assert not [l for l in logs if l.startswith("[fp32-soft-rescue]")]


# ---------------------------------------------------------------------------
# D. just below floor (0.99x) -> accept UNCHANGED + soft telemetry
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_D_below_floor_accepted_unchanged(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    const_ns = _make_const_ns(0.99 * FLOOR)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", const_ns)
    logs: list[str] = []
    local = tmp_path / "local"
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    before = _param_snapshot(model)
    _seed_grads(model, 104)
    opt.step()  # F3: accepted, not a safety error
    assert opt.observations == 1
    assert opt.fp32_rescues == n
    assert opt.fp32_rescue_failures == 0
    assert opt.fp32_low_delta_rescues == n
    assert sum(opt.fp32_low_delta_by_role.values()) == n
    soft = [l for l in logs if l.startswith("[fp32-soft-rescue]")]
    assert len(soft) == n
    for l in soft:
        assert "reason=below_floor_soft_rescue" in l
        for field in (
            "fqn=",
            "role=",
            "fp32_delta_rms=",
            "target_delta_rms=",
            "rescue_floor=",
            "delta/target=",
            "delta/floor=",
            "u_t_rms=",
        ):
            assert field in l, l
    # H. committed CMuon parameters == reference built from the ORIGINAL
    # fp32 rescue + the existing single bf16 rounding (no correction).
    # (reference computed on CPU — same elementwise bf16 values, same
    # rounding; before/after snapshots are CPU)
    after = _param_snapshot(model)
    for spec in routing.cmuon_specs:
        if spec.chunk_count == 1:
            staged = _staged_for_chunk(
                const_ns, tuple(spec.parameter.shape), device
            ).cpu()
            ref = before[spec.name].clone()
            ref.add_(staged)
        else:
            parts = [
                _staged_for_chunk(const_ns, _chunk_shape(spec, ci), device).cpu()
                for ci in range(spec.chunk_count)
            ]
            ref = before[spec.name].clone()
            ref.add_(torch.cat(parts, dim=spec.chunk_dim))
        assert after[spec.name].equal(ref), (
            f"{spec.name}: committed value must equal the reference "
            "(original fp32 delta + one bf16 rounding)"
        )
    # NO hard-fail capsule (no hard failure happened)
    assert _capsule_events(str(local)) == []
    assert not (tmp_path / "fx" / "guard-forensic-rank0.json").exists()


# ---------------------------------------------------------------------------
# E. above ceiling -> hard fail (preserved)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_E_above_ceiling_hardfail_preserved(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", _make_const_ns(2.0 * CEILING))
    before = _param_snapshot(model)
    local = tmp_path / "local"
    logs: list[str] = []
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 105)
    with pytest.raises(CMuonSafetyError, match="both failed"):
        opt.step()
    assert _snap_equal(before, _param_snapshot(model)), (
        "above-ceiling must commit NOTHING (atomic)"
    )
    assert opt.observations == 0
    assert opt.fp32_rescue_failures == n
    assert opt.fp32_low_delta_rescues == 0
    # F2 fast capture preserved for REAL hard failures (local root):
    events = _capsule_events(str(local))
    assert len(events) == n, f"one owner capsule per failed input, got {events}"
    meta = json.loads((Path(local) / events[0] / "metadata.json").read_text())
    assert meta["fp32_failure_reason"] == "above_ceiling"
    assert meta["fp32_finite"] is True
    # best-effort mirror got the same events (distinct root, no -r2)
    mirror_events = _capsule_events(str(tmp_path / "mirror"))
    assert mirror_events == events or len(mirror_events) == 0, (
        f"mirror must mirror the local events, got {mirror_events}"
    )


# ---------------------------------------------------------------------------
# F. nonfinite -> hard fail (preserved)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_F_nonfinite_hardfail_preserved(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(
        fr,
        "cmuon_zeroth_power_fp32",
        lambda grad, *a, **k: torch.full_like(grad, float("inf"), device=grad.device),
    )
    before = _param_snapshot(model)
    local = tmp_path / "local"
    logs: list[str] = []
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 106)
    with pytest.raises(CMuonSafetyError, match="both failed"):
        opt.step()
    assert _snap_equal(before, _param_snapshot(model))
    assert opt.observations == 0
    assert opt.fp32_rescue_failures == n
    events = _capsule_events(str(local))
    assert len(events) == n
    meta = json.loads((Path(local) / events[0] / "metadata.json").read_text())
    assert meta["fp32_failure_reason"] == "nonfinite"
    assert meta["fp32_finite"] is False


# ---------------------------------------------------------------------------
# G. synthetic input matrix sweep — REAL FP32 NS (deterministic inputs)
# ---------------------------------------------------------------------------
def _matrix_class(grad: torch.Tensor, kind: str, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = tuple(grad.shape)
    if grad.ndim != 2:
        m = torch.randn(shape, generator=g) * 1e-3
    elif kind == "zero":
        m = torch.zeros(shape)
    elif kind == "constant":
        m = torch.full(shape, 1e-3)
    elif kind == "rank-1":
        u = torch.randn(shape[0], 1, generator=g)
        v = torch.randn(1, shape[1], generator=g)
        m = (u @ v) * 1e-3
    elif kind == "rank-8":
        u = torch.randn(shape[0], 8, generator=g)
        v = torch.randn(8, shape[1], generator=g)
        m = (u @ v) * 1e-4
    elif kind == "weak-low-rank":
        u = torch.randn(shape[0], 8, generator=g)
        v = torch.randn(8, shape[1], generator=g)
        m = (u @ v) * 1e-4 + torch.randn(shape, generator=g) * 1e-5
    elif kind == "full-rank":
        m = torch.randn(shape, generator=g) * 1e-3
    else:
        raise ValueError(kind)
    grad.copy_(m.to(grad.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
@pytest.mark.parametrize(
    "kind", ["zero", "constant", "rank-1", "rank-8", "weak-low-rank", "full-rank"]
)
def test_G_synthetic_matrix_sweep_real_fp32(monkeypatch, tmp_path, kind: str) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    # deterministic BF16 failure on every chunk; REAL fp32 NS on the
    # synthetic input class (deterministic: fixed inputs, same backend)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    logs: list[str] = []
    local = tmp_path / "local"
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    for p in model.parameters():
        p.grad = torch.empty_like(p)
        _matrix_class(p.grad, kind, seed=20260904)
    opt.step()  # real fp32 NS on a finite synthetic matrix is finite -> accepted
    assert opt.observations == 1, f"{kind}: real fp32 NS must be accepted"
    assert opt.fp32_rescues == n
    assert opt.fp32_rescue_failures == 0
    assert _capsule_events(str(local)) == []
    for p in model.parameters():
        assert torch.isfinite(p.float()).all()
    rec = json.loads(
        next(l for l in logs if l.startswith('{"fp32_rescue_obs"'))
    )
    print(
        f"[f3-sweep] kind={kind} low_delta={rec['fp32_low_delta_rescues']}"
        f"/{rec['fp32_rescues']} by_role={rec['fp32_low_delta_by_role']}"
    )


# ---------------------------------------------------------------------------
# I. checkpoint schema unchanged
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_I_checkpoint_schema_unchanged(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(
        fr, "cmuon_zeroth_power_fp32", lambda *a, **k: torch.zeros_like(a[0])
    )
    local = tmp_path / "local"
    opt = _build(
        model,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 107)
    opt.step()  # a soft-rescue step
    state = opt.state_dict()
    keys = set(state["guard"]["fp32_rescue"].keys())
    assert keys == {
        "bf16_attempts",
        "bf16_safety_failures",
        "fp32_attempts",
        "fp32_rescues",
        "fp32_rescue_failures",
        "rescue_by_role",
    }, f"F3 must not change the persisted schema, got {sorted(keys)}"
    assert "fp32_low_delta_rescues" not in state["guard"]
    assert "fp32_low_delta_rescues" not in state
    # round-trip: persisted counters exact; process-local soft counters
    # restart at zero (documented semantics)
    model2 = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    opt2 = _build(
        model2,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    opt2.load_state_dict(state)
    for k in (
        "bf16_attempts",
        "bf16_safety_failures",
        "fp32_attempts",
        "fp32_rescues",
        "fp32_rescue_failures",
    ):
        assert getattr(opt2, k) == getattr(opt, k)
    assert opt2.rescue_by_role == opt.rescue_by_role
    assert opt2.fp32_low_delta_rescues == 0
    assert opt2.fp32_low_delta_by_role == {}
    # parent-class checkpoint (no fp32_rescue block) still loads
    state2 = dict(state)
    state2["guard"] = dict(state["guard"])
    del state2["guard"]["fp32_rescue"]
    opt3 = _build(
        model2,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    opt3.load_state_dict(state2)


# ---------------------------------------------------------------------------
# J. floor numeric value unchanged
# ---------------------------------------------------------------------------
def test_J_floor_numeric_unchanged() -> None:
    assert fr._RESCUE_SANITY_LOW == 0.05
    # and the safety constants: ceiling 10x target, target 0.2*lr
    assert abs(CEILING - 10.0 * 0.2 * LR) < 1e-20
    assert abs(FLOOR - 0.05 * 0.2 * LR) < 1e-20


# ---------------------------------------------------------------------------
# K. hard-fail path must NOT emit soft telemetry
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HCU")
def test_K_no_soft_telemetry_on_hardfail(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_bf16_const())
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", _make_const_ns(2.0 * CEILING))
    logs: list[str] = []
    local = tmp_path / "local"
    opt = _build(
        model,
        logs=logs,
        local_capsule_root=str(local),
        mirror_root=str(tmp_path / "mirror"),
        forensic_dir=str(tmp_path / "fx"),
    )
    _seed_grads(model, 108)
    with pytest.raises(CMuonSafetyError):
        opt.step()
    assert not [l for l in logs if l.startswith("[fp32-soft-rescue]")]
    assert opt.fp32_low_delta_rescues == 0
