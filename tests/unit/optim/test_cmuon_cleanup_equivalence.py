"""Cleanup equivalence tests for the batched post-NS safety verdict
(cleanup spec sections 7-9 of the production cleanup round).

Proves that the batched device-side safety flags are DECISION-IDENTICAL to
the pre-cleanup per-chunk host-read implementation:

  P. predicate equivalence: old vs new flag code EXACTLY IDENTICAL for the
     same delta tensor — finite / nonfinite / below floor / above ceiling /
     exactly-at-threshold / threshold +/- ulp, canonical chunk shapes
  F. forced catastrophe: NaN / Inf / very-large / too-small / healthy,
     old and new rescue decision identical (step level)
  M. multi-failure: 1 / 2 / 10 / all chunks forced unsafe — the packed mask
     has no index shift, chunk/FQN mismatch, role or owner mismatch
  A. atomicity: forced failure on the LAST chunk -> zero partial mutation
     (parameter bytes untouched, observations not advanced)
  E. FP32 rescue equivalence: fixed FP32 inputs -> bit-identical NS output,
     identical verdict readback (same on-device scalars), production-
     intercepted BF16 staging (cross-allocation NS calls are not
     bit-reproducible on DTK, so the production output is captured, not
     recomputed)
  G. AdamW default path: no CMuon hot-path state is created

The "old" reference implementations below are verbatim transcriptions of
the D4 validated per-chunk code (37602e4): they are the contract, not a
re-implementation. Any drift between the transcription and the validated
source is itself a bug (kept in one place on purpose).
"""

from __future__ import annotations

import math

import pytest
import torch
from test_cmuon import LR, _MockComposite, _MockDiT, _seed_grads

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.optim.cmuon import (
    CMuonChunkSpec,
    cmuon_moonlight_alpha,
    cmuon_zeroth_power_bf16,
    cmuon_zeroth_power_fp32,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import (
    _CEILING,
    _NONFINITE,
    GuardedCanonicalGuardConfig,
)

NS4 = resolve_ns_map(None, 4)
CEILING = 10.0 * 0.2 * LR


# ---------------------------------------------------------------------------
# "Old" reference implementations (D4 validated per-chunk decision rules)
# ---------------------------------------------------------------------------


def old_chunk_flag(delta: torch.Tensor, ceiling: float) -> int:
    """Pre-cleanup (37602e4) per-chunk safety decision, verbatim semantics:
    host bool() of isfinite-all, then host float(rms) > ceiling."""
    if not bool(torch.isfinite(delta).all()):
        return _NONFINITE
    rms = float(delta.float().pow(2).mean().sqrt())
    if rms > ceiling:
        return _CEILING
    return 0


def new_chunk_flag(delta: torch.Tensor, ceiling: float, device: torch.device) -> int:
    """Cleanup batched device-side decision, verbatim production semantics:
    torch.where chain with the FP64 ceiling constant, ceiling flag guarded
    to fire only where the chunk is not already flagged (production
    inf-guard): an inf delta has rms == inf (inf > ceiling is TRUE), so
    without the guard _CEILING would clobber _NONFINITE."""
    flags = torch.zeros(1, dtype=torch.int64, device=device)
    ceiling_t = torch.tensor(ceiling, device=device, dtype=torch.float64)
    rms = delta.float().pow(2).mean().sqrt()
    flags[0] = torch.where(~torch.isfinite(delta).all(), _NONFINITE, flags[0])
    flags[0] = torch.where(
        (rms.double() > ceiling_t) & (flags[0] == 0), _CEILING, flags[0]
    )
    return int(flags[0].item())


def _const_delta(
    value: float, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.bfloat16, device=device)


def _edge_battery(
    shape: tuple[int, ...], device: torch.device, ceiling: float
) -> list[tuple[str, torch.Tensor]]:
    """Exactly-at-threshold and threshold +/- ulp edge cases.

    A constant tensor c*ones has rms == |c| up to the fp32
    pow2/mean/sqrt rounding chain, so the edge family is built around the
    closest fp32 value to the FP64 ceiling and its neighbours.
    """
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    inf32 = torch.tensor(float("inf"), dtype=torch.float32)
    c32 = torch.tensor(ceiling, dtype=torch.float32)
    cases: list[tuple[str, torch.Tensor]] = []
    for name, c in (
        (
            "edge_minus2ulp",
            c32.nextafter(zero32).nextafter(zero32),
        ),
        ("edge_minus1ulp", c32.nextafter(zero32)),
        ("edge_fp32_of_ceiling", c32),
        ("edge_plus1ulp", c32.nextafter(inf32)),
        (
            "edge_plus2ulp",
            c32.nextafter(inf32).nextafter(inf32),
        ),
    ):
        cases.append((name, _const_delta(float(c), shape, device)))
    # random tensor whose rms lands near the ceiling (scaled unit-rms draw)
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(*shape, generator=g, dtype=torch.float32, device=device)
    x = x / x.pow(2).mean().sqrt().clamp(min=1e-12)
    for scale in (0.5 * ceiling, 0.9 * ceiling, 1.1 * ceiling, 2.0 * ceiling):
        cases.append((f"rand_rms_x{scale / ceiling:g}", (x * scale).to(torch.bfloat16)))
    return cases


def _battery(device: torch.device) -> list[tuple[str, torch.Tensor]]:
    """Canonical chunk shapes x finite/nonfinite/edge cases (spec 8.1)."""
    shapes = {
        "q": (256, 256),
        "k": (256, 64),
        "v": (64, 256),  # tall
        "content_gate": (256, 256),
        "out": (256, 256),
        "ffn_in_half": (512, 256),
        "ffn_down": (256, 512),
        "adaln_shared": (256, 128),
        "square_small": (64, 64),
    }
    cases: list[tuple[str, torch.Tensor]] = []
    for role, shape in shapes.items():
        g = torch.Generator(device=device).manual_seed(1)
        cases.append(
            (
                f"{role}_healthy",
                torch.randn(
                    *shape, generator=g, dtype=torch.bfloat16, device=device
                ),
            )
        )
        cases.append((f"{role}_zero", _const_delta(0.0, shape, device)))
        x = torch.zeros(*shape, dtype=torch.bfloat16, device=device)
        x[0, 0] = float("nan")
        cases.append((f"{role}_nan", x))
        x = torch.zeros(*shape, dtype=torch.bfloat16, device=device)
        x[0, 0] = float("inf")
        cases.append((f"{role}_inf", x))
        x = torch.zeros(*shape, dtype=torch.bfloat16, device=device)
        x[0, 0] = float("-inf")
        cases.append((f"{role}_neginf", x))
        cases.append((f"{role}_huge", _const_delta(1e30, shape, device)))
        cases.append((f"{role}_tiny", _const_delta(1e-30, shape, device)))
        cases.extend(
            (f"{role}_{n}", t) for n, t in _edge_battery(shape, device, CEILING)
        )
    return cases


# ---------------------------------------------------------------------------
# P. predicate equivalence (no optimizer required)
# ---------------------------------------------------------------------------


def test_P_predicate_equivalence_battery() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mismatches = []
    total = 0
    for name, delta in _battery(device):
        old = old_chunk_flag(delta, CEILING)
        new = new_chunk_flag(delta, CEILING, device)
        total += 1
        if old != new:
            mismatches.append((name, old, new))
    assert not mismatches, f"predicate mask mismatch: {mismatches}"
    assert total >= 8 * 13, f"battery too small: {total}"


def test_P_full_vector_mask_identity() -> None:
    """Full-vector: old and new unsafe masks EXACTLY identical element by
    element (spec 8.1), including both failure classes and healthy chunks."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(99)
    n_chunks = 166
    deltas = []
    for i in range(n_chunks):
        shape = (64 + (i % 5) * 32, 64 + ((i * 7) % 5) * 32)
        # healthy unit-rms draws scaled well below the ceiling (a raw randn
        # delta has rms ~1.0 >> CEILING ~2e-4 and would trip the ceiling)
        x = (
            torch.randn(*shape, generator=g, dtype=torch.bfloat16, device=device)
            * (0.1 * CEILING)
        )
        # sprinkle forced defects every 17th chunk
        if i % 17 == 3:
            x[0, 0] = float("nan")
        elif i % 17 == 5:
            x = x * (1e9 / (0.1 * CEILING))  # finite but far above the ceiling
        deltas.append(x)
    old_mask = [old_chunk_flag(d, CEILING) for d in deltas]
    new_mask = [new_chunk_flag(d, CEILING, device) for d in deltas]
    assert old_mask == new_mask, "full-vector unsafe mask differs"
    assert sum(1 for f in old_mask if f == _NONFINITE) >= 9
    assert sum(1 for f in old_mask if f == _CEILING) >= 9
    assert sum(1 for f in old_mask if f == 0) > 0


# ---------------------------------------------------------------------------
# step-level helpers (single rank: NS calls arrive in flat-chunk order)
# ---------------------------------------------------------------------------


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


def _build_opt(model):
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
        guard_cfg=GuardedCanonicalGuardConfig(
            guard_ratio=0.05,
            reference_decay=0.999,
            min_reference=1e-12,
            numerical_floor=1e-20,
            warmup_observations=0,
            invariant_check=True,
        ),
        guard_bootstrap_refs=_bootstrap_refs(model),
        rank=0,
        world_size=1,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
    )


def _n_inputs(model) -> int:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    return sum(spec.chunk_count for spec in routing.cmuon_specs)


def _flat_input_keys(model) -> list[tuple[str, int]]:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    keys: list[tuple[str, int]] = []
    for spec in routing.cmuon_specs:
        for ci in range(spec.chunk_count):
            keys.append((spec.name, ci))
    return keys


def _forced_ns(pattern: dict[int, str]):
    """Call-order BF16 NS stand-in: single-rank builds call NS in flat-chunk
    order 0..n-1. Pattern values: 'nan' | 'inf' | 'huge' | 'zero' | absent."""
    real = cmuon_zeroth_power_bf16
    call = {"i": 0}

    def forced(grad, ns_steps, ns_coefficients, eps):
        i = call["i"]
        call["i"] += 1
        kind = pattern.get(i)
        if kind is None:
            return real(grad, ns_steps, ns_coefficients, eps)
        out = real(grad, ns_steps, ns_coefficients, eps)
        if kind == "nan":
            return out * float("nan")
        if kind == "inf":
            return out * float("inf")
        if kind == "huge":
            return out * 1e9
        if kind == "zero":
            return torch.zeros_like(out)
        raise ValueError(kind)

    return forced


@pytest.fixture()
def single_rank_env():
    """Fresh mock model; single rank (world_size=1 => rank 0 owns every
    chunk, so NS call order == flat chunk order)."""
    if not torch.cuda.is_available():
        pytest.skip("optimizer build requires CUDA/HCU")
    device = torch.device("cuda:0")
    torch.manual_seed(20260901)
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    return model


# ---------------------------------------------------------------------------
# F. forced catastrophe (spec 8.2)
# ---------------------------------------------------------------------------


def test_F_forced_catastrophe_decision_identity(
    monkeypatch, single_rank_env
) -> None:
    model = single_rank_env
    n = _n_inputs(model)
    # force: 0 NaN, 1 Inf, 2 huge (ceiling), 3 zero (too small), rest healthy
    pattern = {0: "nan", 1: "inf", 2: "huge", 3: "zero"}
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _forced_ns(pattern))
    opt = _build_opt(model)
    _seed_grads(model, 11)
    opt.step()  # must NOT raise: every BF16 trip is FP32-rescued (fp32 sane)
    # the old decision rule applied to the same forced outputs:
    expected_unsafe = {
        i for i, k in pattern.items() if k in ("nan", "inf", "huge")
    }
    assert opt.bf16_attempts == n
    assert opt.bf16_safety_failures == len(expected_unsafe)
    assert opt.fp32_attempts == len(expected_unsafe)
    assert opt.fp32_rescues == len(expected_unsafe)
    assert opt.fp32_rescue_failures == 0
    assert opt.observations == 1
    # the 'zero' chunk passed the BF16 predicate exactly as the old code
    # decided (finite, rms 0 < ceiling) — no new predicate behaviour
    assert 3 not in expected_unsafe
    for p in model.parameters():
        assert torch.isfinite(p.float()).all()


# ---------------------------------------------------------------------------
# M. multi-failure packed mask (spec 8.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 10, "all"])
def test_M_multi_failure_mask_no_shift(
    monkeypatch, single_rank_env, count: int | str
) -> None:
    model = single_rank_env
    n = _n_inputs(model)
    keys = _flat_input_keys(model)
    if count == "all":
        idxs = list(range(n))
    else:
        g = torch.Generator().manual_seed(5)
        idxs = sorted(torch.randperm(n, generator=g)[:count].tolist())
    pattern = {i: "nan" for i in idxs}
    # fp32 also fails on EXACTLY the forced set -> hard fail-closed. The
    # rescue loop visits unsafe chunks in ascending flat-idx order, so the
    # kth FP32 call maps to the kth unsafe idx; failing all of them proves
    # the mask -> FQN naming end to end (no index shift).
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _forced_ns(pattern))
    real32 = cmuon_zeroth_power_fp32

    def evil32(grad, ns_steps, ns_coefficients, eps):
        return real32(grad, ns_steps, ns_coefficients, eps) * float("nan")

    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", evil32)
    opt = _build_opt(model)
    _seed_grads(model, 13)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(CMuonSafetyError) as excinfo:
        opt.step()
    msg = str(excinfo.value)
    # every forced chunk named exactly once with its true FQN
    for i in idxs:
        fqn, ci = keys[i]
        needle = f"{fqn}#chunk{ci}"
        assert msg.count(needle) == 1, (needle, msg)
    assert opt.fp32_rescue_failures == len(idxs)
    assert opt.fp32_rescues == 0
    assert opt.observations == 0
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p.detach()), (
            "failed step must commit nothing"
        )


def test_M_rescue_set_exact(monkeypatch, single_rank_env) -> None:
    """BF16 trips on a scattered set, FP32 healthy: the RESCUED set must be
    exactly the forced set (packed mask, no shift) and the role histogram
    must match the forced chunks' roles (no role/owner mismatch)."""
    model = single_rank_env
    n = _n_inputs(model)
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    role_of: dict[int, str] = {}
    k = 0
    for spec in routing.cmuon_specs:
        for _ci in range(spec.chunk_count):
            role_of[k] = spec.role
            k += 1
    idxs = sorted({0, max(1, n // 4), max(2, n // 2), max(3, (3 * n) // 4), n - 1})
    pattern = {i: "nan" for i in idxs}
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _forced_ns(pattern))
    opt = _build_opt(model)
    _seed_grads(model, 17)
    opt.step()
    assert opt.bf16_safety_failures == len(idxs)
    assert opt.fp32_rescues == len(idxs)
    assert opt.fp32_rescue_failures == 0
    expected_roles: dict[str, int] = {}
    for i in idxs:
        expected_roles[role_of[i]] = expected_roles.get(role_of[i], 0) + 1
    assert opt.rescue_by_role == expected_roles, (
        f"role histogram mismatch: {opt.rescue_by_role} vs {expected_roles}"
    )


# ---------------------------------------------------------------------------
# A. atomicity: forced failure on the LAST chunk (spec 7)
# ---------------------------------------------------------------------------


def test_A_last_chunk_failure_zero_commit(
    monkeypatch, single_rank_env
) -> None:
    model = single_rank_env
    n = _n_inputs(model)
    last = n - 1
    # every chunk BEFORE the last computes and (ws=1) stages normally; only
    # the last chunk fails both BF16 and FP32 -> zero commit even though all
    # prior chunks are fully computed.
    pattern = {last: "nan"}
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _forced_ns(pattern))

    def evil32(grad, ns_steps, ns_coefficients, eps):
        return cmuon_zeroth_power_fp32(
            grad, ns_steps, ns_coefficients, eps
        ) * float("nan")

    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", evil32)
    opt = _build_opt(model)
    _seed_grads(model, 23)
    before_params = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }
    with pytest.raises(CMuonSafetyError):
        opt.step()
    for name, p in model.named_parameters():
        assert torch.equal(before_params[name], p.detach()), (
            "parameters must be byte-identical after a final-chunk failure"
        )
    assert opt.observations == 0
    assert opt.fp32_rescue_failures == 1
    assert opt.fp32_rescues == 0


# ---------------------------------------------------------------------------
# E. FP32 rescue bit-identity (spec 9)
# ---------------------------------------------------------------------------


def test_E_fp32_ns_bit_deterministic() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(3)
    for shape in ((128, 256), (256, 128), (64, 64)):
        x = torch.randn(*shape, generator=g, dtype=torch.float32, device=device)
        a = cmuon_zeroth_power_fp32(x, 4, (3.4445, -4.7750, 2.4304), 1e-7)
        b = cmuon_zeroth_power_fp32(x, 4, (3.4445, -4.7750, 2.4304), 1e-7)
        assert torch.equal(a, b), f"FP32 NS not bit-deterministic for {shape}"


def test_E_rescue_verdict_readback_identical() -> None:
    """Old-style (float()/bool()) vs new-style (packed tolist) rescue
    verdict: identical values and identical decisions on a battery of FP32
    deltas, including floor/ceiling edges.

    Both readback styles consume the SAME on-device scalars (one reduction
    per case): the cleanup's claim under test is that the packed tolist()
    readback yields exactly the Python floats the old float()/bool() calls
    produced. Running the reduction twice (old vs new as independent calls)
    is not valid on DTK: cross-allocation results are not bit-reproducible
    (see the E-staging interception note).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(4)
    rescue_floor = 0.05 * 0.2 * LR
    shape = (64, 128)
    base = torch.randn(*shape, generator=g, dtype=torch.float32, device=device)
    base = base / base.pow(2).mean().sqrt().clamp(min=1e-12)
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    inf32 = torch.tensor(float("inf"), dtype=torch.float32)
    f_floor = torch.tensor(rescue_floor, dtype=torch.float32)
    f_ceiling = torch.tensor(CEILING, dtype=torch.float32)
    cases: dict[str, torch.Tensor] = {
        "healthy": base * (0.2 * LR),
        "at_floor": base * float(f_floor),
        "floor_m1ulp": base * float(f_floor.nextafter(zero32)),
        "floor_p1ulp": base * float(f_floor.nextafter(inf32)),
        "at_ceiling": base * float(f_ceiling),
        "ceiling_m1ulp": base * float(f_ceiling.nextafter(zero32)),
        "ceiling_p1ulp": base * float(f_ceiling.nextafter(inf32)),
        "nan": base * float("nan"),
        "inf": base * float("inf"),
        "zero": torch.zeros_like(base),
    }
    for name, d32 in cases.items():
        # one reduction per case; both readback styles consume it
        rms_t = d32.pow(2).mean().sqrt()
        fin_t = torch.isfinite(d32).all()
        # old style (37602e4 rescue verdict)
        rms_old = float(rms_t)
        verdict_old = (
            not bool(fin_t)
        ) or rms_old < rescue_floor or rms_old > CEILING
        # new style (cleanup packed readback of the same scalars)
        rms_new, fin_new = torch.stack(
            (rms_t, fin_t.to(torch.float32))
        ).tolist()
        verdict_new = (
            not bool(fin_new)
        ) or rms_new < rescue_floor or rms_new > CEILING
        # NaN-aware: the "nan" case yields NaN from both readback styles and
        # NaN is never == itself, so compare via isnan for that case.
        if math.isnan(rms_old) or math.isnan(rms_new):
            assert math.isnan(rms_old) and math.isnan(rms_new), name
        else:
            assert rms_old == rms_new, name
        assert verdict_old == verdict_new, name


def test_E_rescue_staging_bit_identical(
    monkeypatch, single_rank_env
) -> None:
    """End-to-end: a forced-NaN chunk must commit exactly
    (-alpha) * fp32_NS(nesterov_chunk) rounded once to BF16, applied
    through the same bf16 add path.

    The FP32 NS output is INTERCEPTED from the production call rather than
    recomputed in the test: DTK's FP32 GEMM is not bit-reproducible across
    different memory allocations (allocator alignment can select a
    different GEMM path), so a second NS call in the test may differ by
    1 ulp even for bit-identical inputs. The interception verifies the
    contract that matters for the cleanup:
      (1) the rescue input is bit-identical to the PREPARE chain
          (fresh-momentum nesterov chunk) — no semantic drift;
      (2) the committed bytes equal the production PHASE-2 reassembly: the
          rescued chunk staged as ((-alpha) * captured_fp32_output).bfloat16()
          (single-rounding formula intact) plus the sibling chunks' captured
          bf16 staged deltas, cat'ed and applied with one add_.
    """
    model = single_rank_env
    target = 7
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    flat = 0
    spec_t: CMuonChunkSpec | None = None
    ci_t = 0
    for spec in routing.cmuon_specs:
        for ci in range(spec.chunk_count):
            if flat == target:
                spec_t, ci_t = spec, ci
            flat += 1
    assert spec_t is not None

    # Capture EVERY bf16 NS call (single rank: call order == flat-chunk
    # order) so the sibling chunks' staged deltas can be reconstructed;
    # force NaN only on the target chunk.
    real_bf16 = fr.cmuon_zeroth_power_bf16
    call_i = {"i": 0}
    bf16_captured: dict[int, torch.Tensor] = {}

    def spy_bf16(grad, ns_steps, ns_coefficients, eps):
        i = call_i["i"]
        call_i["i"] += 1
        out = real_bf16(grad, ns_steps, ns_coefficients, eps)
        bf16_captured[i] = out.detach().clone()
        if i == target:
            return out * float("nan")
        return out

    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", spy_bf16)

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    real32 = fr.cmuon_zeroth_power_fp32

    def spy32(grad, ns_steps, ns_coefficients, eps):
        out = real32(grad, ns_steps, ns_coefficients, eps)
        captured.append((grad.detach().clone(), out.detach().clone()))
        return out

    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", spy32)
    opt = _build_opt(model)
    _seed_grads(model, 31)
    # clone: state_dict tensors share parameter storage (in-place add_ would
    # corrupt a "before" view)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt.step()
    # exactly one FP32 NS call: the forced-NaN chunk, nothing else
    assert len(captured) == 1, (
        f"expected exactly one FP32 rescue call, got {len(captured)}"
    )
    in32, out32 = captured[0]
    # (1) rescue input == PREPARE semantics (fresh momentum, first step):
    mu = opt.cfg.momentum
    grad = spec_t.parameter.grad
    assert grad is not None
    grad_md = grad.to(torch.bfloat16)
    buf0 = torch.zeros_like(grad_md)
    buf0.lerp_(grad_md, 1.0 - mu)
    nesterov = grad_md.lerp(buf0, mu)
    chunk_size = spec_t.chunk_size()
    if spec_t.chunk_count == 1:
        chunks = (nesterov,)
        chunk = nesterov
    else:
        chunks = tuple(nesterov.split(chunk_size, dim=spec_t.chunk_dim))
        chunk = chunks[ci_t]
    assert torch.equal(in32, chunk.float()), (
        "rescue input is not bit-identical to the prepare nesterov chunk"
    )
    # (2) committed bytes: per-chunk staged deltas reassembled EXACTLY like
    # production PHASE 2 (cat on chunk_dim + one parameter.add_). The
    # rescued chunk goes through the single-rounding FP32 staging formula
    # with the production-captured NS output; the sibling chunks (normal
    # BF16 path of the same step) go through the production bf16 formula
    # with their production-captured NS outputs. Pointwise scalar
    # multiplies are deterministic on the same tensors, so this is a
    # bit-exact reconstruction (no GEMM/reduction is re-run).
    rescale = spec_t.chunk_count if opt.cfg.chunk_rescale_sqrt_n else 1
    spec_flat_start = target - ci_t
    delta_parts: list[torch.Tensor] = []
    for ci, c in enumerate(chunks):
        alpha_ci = cmuon_moonlight_alpha(c.shape[0], c.shape[1], opt.cfg.lr, rescale)
        if ci == ci_t:
            delta_parts.append(((-alpha_ci) * out32).bfloat16().contiguous())
        else:
            ns_bf16 = bf16_captured[spec_flat_start + ci]
            delta_parts.append(((-alpha_ci) * ns_bf16).contiguous())
    update_ortho = (
        delta_parts[0]
        if spec_t.chunk_count == 1
        else torch.cat(delta_parts, dim=spec_t.chunk_dim)
    )
    name = spec_t.name
    expected_param = before[name]
    expected_param.add_(update_ortho)
    actual = model.state_dict()[name]
    assert torch.equal(actual, expected_param), (
        "committed bytes differ from the production-captured staging "
        "formula (rescued chunk single-rounding + sibling bf16 deltas)"
    )


# ---------------------------------------------------------------------------
# G. AdamW default path (spec 12)
# ---------------------------------------------------------------------------


def test_G_adamw_default_path_no_cmuon_state() -> None:
    """The torchao_adamw8bit default path must not instantiate any CMuon hot
    path state (no momentum map, no rescue counters, no guard refs)."""
    if not torch.cuda.is_available():
        pytest.skip("optimizer build requires CUDA/HCU")
    device = torch.device("cuda:0")
    torch.manual_seed(20260902)
    model = _MockComposite(_MockDiT(256, 512, 1)).to(device)
    from sakuramoon.optim.adamw8bit import build_adamw8bit

    opt = build_adamw8bit(
        model,
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
        sr_seed=44,
    )
    assert type(opt).__name__ == "IsolatedAdamW8bit"
    for attr in (
        "_momenta",
        "bf16_attempts",
        "fp32_attempts",
        "fp32_rescues",
        "fp32_rescue_failures",
        "rescue_by_role",
        "_refs",
        "routing",
        "observations",
    ):
        assert not hasattr(opt, attr), f"AdamW path must not carry {attr}"
    _seed_grads(model, 41)
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p.float()).all()
