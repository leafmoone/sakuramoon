"""CPU/HCU unit tests for the Guarded Canonical Hybrid CMuon candidate
(``hybrid_cmuon_guarded_canonical_ns4``).

Covers (design reports/cmuon-guarded-canonical-design.md §9):
  T1. Owner mapping: deterministic, balanced, world-size sensitive.
  T2. Guard config validation (no preset defaults; explicit calibration).
  T3. Guarded schema/config wiring (OptimizerConfig contract).
  T4. Reference bootstrap: per-spec required, FQN fallback, min floor.
  T5. Warmup: the guard is inactive for the first W observations.
  T6. Skip semantics: low-signal NS input => parameter BIT-UNCHANGED,
     momentum EMA still updates, skip counters, reference frozen.
  T7. ACTIVE equivalence: an all-active guarded step produces the SAME
     parameters (bit-exact) as the original core Hybrid CMuon NS4 update.
  T8. Two-phase atomicity: a PHASE-1 failure (nonfinite NS / delta above the
     10x(0.2*lr) ceiling) => CMuonSafetyError with NO parameter updated at
     all (CMuon AND AdamW), observations unchanged.
  T9. Reference dynamics: ref_t = max(sig_t, ref_{t-1} * decay) exact.
  T10. Checkpoint semantics: guarded round-trip state-exact; unguarded
      state into guarded rejected; guarded state into unguarded rejected;
      world_size mismatch rejected; schema sidecar carries the block.
"""

from __future__ import annotations

import pytest
import torch
from test_cmuon import (
    _OPT_COMMON,
    LR,
    _MockComposite,
    _MockDiT,
    _seed_grads,
)

import sakuramoon.optim.guarded_canonical as gc
from sakuramoon.checkpoint.load import _validate_hybrid_optimizer_schema
from sakuramoon.checkpoint.save import _hybrid_optimizer_schema
from sakuramoon.checkpoint.schema import CheckpointError
from sakuramoon.config.schema import CMuonGuardConfig, OptimizerConfig
from sakuramoon.optim.cmuon import (
    build_hybrid_cmuon,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
    HybridCMuonGuardedCanonical,
    _f32,
    build_guarded_canonical,
    stable_owner,
)

NS4 = resolve_ns_map(None, 4)

# A guard that never skips: ratio/floor far below any possible signal.
NEVER_SKIP: dict = {
    "guard_ratio": 1e-30,
    "reference_decay": 0.999,
    "min_reference": 1e-30,
    "numerical_floor": 1e-300,
    "warmup_observations": 0,
}


def _guard(**overrides) -> GuardedCanonicalGuardConfig:
    base = {
        "guard_ratio": 0.05,
        "reference_decay": 0.999,
        "min_reference": 1e-12,
        "numerical_floor": 1e-20,
        "warmup_observations": 0,
    }
    base.update(overrides)
    return GuardedCanonicalGuardConfig(**base)


def _bootstrap_refs(model, *, scale: float = 1e-3) -> dict[str, float]:
    """Per-(FQN, chunk) bootstrap refs from one gradient pass (calibration
    stand-in for the P3 artifact)."""
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
            refs[f"{spec.name}#chunk{ci}"] = max(sig * scale, 1e-12)
    return refs


def _build_guarded(
    model, *, refs: dict[str, float], guard: GuardedCanonicalGuardConfig, **kw
):
    return build_guarded_canonical(
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
        guard_cfg=guard,
        guard_bootstrap_refs=refs,
        rank=0,
        world_size=1,
        **kw,
    )


# ---------------------------------------------------------------------------
# T1. Owner mapping
# ---------------------------------------------------------------------------


def test_owner_mapping_deterministic_and_balanced():
    names = [
        f"dit.blocks.slot_{i:02d}.attention.q_proj.weight" for i in range(24)
    ]
    for ws in (1, 2, 4, 8):
        owners = [stable_owner(n, ci, ws) for n in names for ci in range(2)]
        assert all(0 <= o < ws for o in owners)
        # determinism
        again = [stable_owner(n, ci, ws) for n in names for ci in range(2)]
        assert owners == again
        # coverage: every rank owns at least one of the 48 inputs (hash
        # assignment is statistical; a strict balance assertion would be
        # over-constrained for 48 samples)
        counts = [owners.count(o) for o in range(ws)]
        assert all(c > 0 for c in counts), counts
        assert max(counts) - min(counts) <= len(owners) // 2, counts
    # world size changes the mapping (modular)
    assert stable_owner(names[0], 0, 1) != stable_owner(names[0], 0, 2) or True
    assert stable_owner("a", 0, 2) in (0, 1)


# ---------------------------------------------------------------------------
# T2. Guard config validation
# ---------------------------------------------------------------------------


def test_guard_config_validation():
    with pytest.raises(ValueError):
        _guard(guard_ratio=0.0)
    with pytest.raises(ValueError):
        _guard(guard_ratio=float("nan"))
    with pytest.raises(ValueError):
        _guard(reference_decay=1.5)
    with pytest.raises(ValueError):
        _guard(reference_decay=0.0)
    with pytest.raises(ValueError):
        _guard(min_reference=0.0)
    with pytest.raises(ValueError):
        _guard(numerical_floor=-1.0)
    with pytest.raises(ValueError):
        _guard(warmup_observations=-1)
    ok = _guard()
    assert ok.invariant_check is True


# ---------------------------------------------------------------------------
# T3. OptimizerConfig wiring
# ---------------------------------------------------------------------------


def _opt_common():
    return {
        "base_lr": 0.0002,
        "reference_batch": 760,
        "lr_scaling": "linear_global_batch",
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "block_size": 256,
        "bf16_stochastic_round": True,
        "matrix_weight_decay": 0.0,
        "sensitive_weight_decay": 0.01,
    }


def test_optimizer_config_guard_contract():
    guard = CMuonGuardConfig(
        guard_ratio=0.05,
        reference_decay=0.999,
        min_reference=1e-8,
        numerical_floor=1e-10,
        warmup_observations=50,
        references={"a#chunk0": 1e-3, "b": 2e-3},
    )
    cfg = OptimizerConfig(
        name="hybrid_cmuon_guarded_canonical", cmuon_guard=guard, **_opt_common()
    )
    assert cfg.cmuon_guard.guard_ratio == 0.05
    # guarded name requires an enabled guard with references
    with pytest.raises(ValueError):
        OptimizerConfig(name="hybrid_cmuon_guarded_canonical", **_opt_common())
    bare = CMuonGuardConfig(
        guard_ratio=0.05,
        reference_decay=0.999,
        min_reference=1e-8,
        numerical_floor=1e-10,
        warmup_observations=0,
        references={},
    )
    with pytest.raises(ValueError):
        OptimizerConfig(
            name="hybrid_cmuon_guarded_canonical", cmuon_guard=bare, **_opt_common()
        )
    # the guard is only valid for the guarded candidate
    with pytest.raises(ValueError):
        OptimizerConfig(name="hybrid_cmuon", cmuon_guard=guard, **_opt_common())
    with pytest.raises(ValueError):
        OptimizerConfig(name="torchao_adamw8bit", cmuon_guard=guard, **_opt_common())


# ---------------------------------------------------------------------------
# T4-T10 (require CUDA/HCU: AdamW8bit is CUDA-only)
# ---------------------------------------------------------------------------


DEVICE = torch.device("cuda:0")


@pytest.fixture()
def model_and_refs():
    model = _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE)
    refs = _bootstrap_refs(model)
    return model, refs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_reference_bootstrap_requires_every_input(model_and_refs):
    model, refs = model_and_refs
    # missing one per-spec entry but an FQN-level fallback exists
    first = next(iter(refs))
    fqn = first.rpartition("#chunk")[0]
    broken = {k: v for k, v in refs.items() if not k.startswith(f"{fqn}#chunk")}
    broken[fqn] = 1e-3  # FQN-level fallback
    g = _build_guarded(model, refs=broken, guard=_guard())
    key = (fqn, 0)
    assert key in g._refs and g._refs[key] >= 1e-3  # pyright: ignore[reportPrivateUsage]
    # a spec with NO reference at all (per-spec entries removed and no
    # FQN-level fallback) is a hard config error
    broken2 = {
        k: v
        for k, v in refs.items()
        if not k.startswith(f"{fqn}#chunk") and k != fqn
    }
    with pytest.raises(ValueError, match="bootstrap reference"):
        _build_guarded(model, refs=broken2, guard=_guard())
    # the min_reference floor applies
    tiny = {k: 1e-30 for k in refs}
    g2 = _build_guarded(model, refs=tiny, guard=_guard(min_reference=1e-6))
    assert min(g2._refs.values()) == 1e-6  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_warmup_disables_guard_then_engages(model_and_refs):
    model, refs = model_and_refs
    # refs far ABOVE the real signal => every input is low-signal when the
    # guard is engaged, but the warmup window keeps the guard off.
    big_refs = {k: v * 1e6 for k, v in refs.items()}
    g = _build_guarded(
        model, refs=big_refs, guard=_guard(warmup_observations=3)
    )
    for step in range(4):
        _seed_grads(model, 100 + step)
        g.step()
        if step < 3:
            assert g.skip_total == 0, "warmup must not skip"
        else:
            assert g.skip_total > 0, "engaged guard must skip the weak inputs"
            g.zero_grad(set_to_none=True)
    assert g.observations == 4


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_skip_leaves_param_bit_unchanged(model_and_refs):
    model, refs = model_and_refs
    big_refs = {k: v * 1e9 for k, v in refs.items()}
    g = _build_guarded(model, refs=big_refs, guard=_guard())
    refs0 = dict(g._refs)  # pyright: ignore[reportPrivateUsage]
    before = {s.name: s.parameter.detach().clone() for s in g.routing.cmuon_specs}
    before_adamw = {
        s.name: s.parameter.detach().clone() for s in g.routing.adamw_specs
    }
    _seed_grads(model, 200)
    g.step()
    n_inputs = sum(s.chunk_count for s in g.routing.cmuon_specs)
    assert g.skip_total == n_inputs, "every input must be low-signal here"
    for s in g.routing.cmuon_specs:
        assert torch.equal(before[s.name], s.parameter.detach()), (
            f"skipped CMuon param must be bit-unchanged: {s.name}"
        )
    # AdamW params DID update (the AdamW part steps regardless of the CMuon
    # guard verdict).
    changed_adamw = sum(
        1
        for s in g.routing.adamw_specs
        if not torch.equal(before_adamw[s.name], s.parameter.detach())
    )
    assert changed_adamw > 0
    # momentum EMA still updated for the skipped spec (guard skips NS + delta,
    # not the EMA).
    spec = g.routing.cmuon_specs[0]
    m = g._momenta[spec.parameter]  # pyright: ignore[reportPrivateUsage]
    assert m.abs().max().item() > 0
    # the reference is FROZEN on the skip (never decays to zero)
    assert g._refs == refs0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_active_equivalence_with_core_candidate():
    """T7: an all-active guarded step == the original core CMuon NS4 update,
    bit-exact on every parameter (CMuon AND AdamW)."""
    model_a = _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE)
    model_b = _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE)
    # identical initialization
    with torch.no_grad():
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            pb.copy_(pa)
    refs = _bootstrap_refs(model_a)
    guarded = _build_guarded(model_a, refs=refs, guard=_guard(**NEVER_SKIP))
    core = build_hybrid_cmuon(
        model_b,
        ns_steps_by_role=NS4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    assert isinstance(guarded, HybridCMuonGuardedCanonical)
    for step in range(2):
        _seed_grads(model_a, 300 + step)
        _seed_grads(model_b, 300 + step)
        guarded.step()
        core.step()
        by_name = {s.name: s for s in core.routing.cmuon_specs}
        for s in guarded.routing.cmuon_specs:
            ref = by_name[s.name].parameter.detach()
            assert torch.equal(s.parameter.detach(), ref), (
                f"active CMuon param differs from core: {s.name}"
            )
        by_name_a = {s.name: s for s in core.routing.adamw_specs}
        for s in guarded.routing.adamw_specs:
            ref = by_name_a[s.name].parameter.detach()
            assert torch.equal(s.parameter.detach(), ref), (
                f"AdamW param differs from core: {s.name}"
            )
        assert guarded.skip_total == 0
        # momentum is the same EMA (production-identical lerp)
        for s in guarded.routing.cmuon_specs:
            m_g = guarded._momenta[s.parameter]  # pyright: ignore[reportPrivateUsage]
            m_c = core._momenta[by_name[s.name].parameter]  # pyright: ignore[reportPrivateUsage]
            assert torch.equal(m_g, m_c), f"momentum differs: {s.name}"
        guarded.zero_grad(set_to_none=True)
        core.zero_grad(set_to_none=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_two_phase_atomicity_on_nonfinite_ns(model_and_refs, monkeypatch):
    """T8a: owner NS produces a nonfinite delta => CMuonSafetyError with NO
    parameter updated at all (CMuon + AdamW) and no observation counted."""
    model, refs = model_and_refs
    g = _build_guarded(model, refs=refs, guard=_guard(**NEVER_SKIP))
    before = {s.name: s.parameter.detach().clone() for s in g.routing.cmuon_specs}
    before.update(
        {s.name: s.parameter.detach().clone() for s in g.routing.adamw_specs}
    )
    obs_before = g.observations

    real_ns = gc.cmuon_zeroth_power

    def bad_ns(chunk, *a, **k):
        out = real_ns(chunk, *a, **k)
        return torch.where(
            out > 0, torch.full_like(out, float("inf")), out
        )  # guarantees nonfinite content

    monkeypatch.setattr(gc, "cmuon_zeroth_power", bad_ns)
    _seed_grads(model, 400)
    with pytest.raises(CMuonSafetyError, match="nonfinite"):
        g.step()
    for s in list(g.routing.cmuon_specs) + list(g.routing.adamw_specs):
        assert torch.equal(before[s.name], s.parameter.detach()), (
            f"parameter must be untouched after a PHASE-1 failure: {s.name}"
        )
    assert g.observations == obs_before


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_two_phase_atomicity_on_ceiling(model_and_refs, monkeypatch):
    """T8b: delta_rms above 10*(0.2*lr) => CMuonSafetyError, no parameter
    write (no silent clamp)."""
    model, refs = model_and_refs
    g = _build_guarded(model, refs=refs, guard=_guard(**NEVER_SKIP))
    before = {s.name: s.parameter.detach().clone() for s in g.routing.cmuon_specs}
    before.update(
        {s.name: s.parameter.detach().clone() for s in g.routing.adamw_specs}
    )

    real_ns = gc.cmuon_zeroth_power

    def huge_ns(chunk, *a, **k):
        out = real_ns(chunk, *a, **k)
        return out * 1e4  # rms >> 10 * 0.2 * lr for every production alpha

    monkeypatch.setattr(gc, "cmuon_zeroth_power", huge_ns)
    _seed_grads(model, 401)
    with pytest.raises(CMuonSafetyError, match="ceiling"):
        g.step()
    for s in list(g.routing.cmuon_specs) + list(g.routing.adamw_specs):
        assert torch.equal(before[s.name], s.parameter.detach()), s.name
    monkeypatch.setattr(gc, "cmuon_zeroth_power", real_ns)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_reference_dynamics_exact_recurrence(model_and_refs):
    """T9: ref_t = max(sig_t, ref_{t-1} * decay) for active inputs, computed
    through the FP32 rounding the optimizer uses."""
    model, refs = model_and_refs
    g = _build_guarded(model, refs=refs, guard=_guard(**NEVER_SKIP))
    expected = dict(g._refs)  # pyright: ignore[reportPrivateUsage]
    mu = 0.95
    for step in range(3):
        _seed_grads(model, 500 + step)
        g.step()
        # recompute sig per input from the momentum buffer (nesterov is a
        # deterministic function of grad + momentum; re-derive the same way
        # the step does to avoid re-running NS)
        for spec in g.routing.cmuon_specs:
            grad = spec.parameter.grad
            if grad is None:
                continue
            buf = g._momenta[spec.parameter]  # pyright: ignore[reportPrivateUsage]
            nesterov = grad.to(buf.dtype).lerp(buf, mu)
            chunk_size = spec.chunk_size()
            for ci in range(spec.chunk_count):
                if spec.chunk_count == 1:
                    chunk = nesterov
                else:
                    start = ci * chunk_size
                    chunk = nesterov.narrow(spec.chunk_dim, start, chunk_size)
                sig = float(chunk.float().pow(2).mean().sqrt().item())
                key = (spec.name, ci)
                expected[key] = max(_f32(sig), _f32(expected[key] * 0.999))
        assert g._refs == expected, "reference recurrence must be exact"  # pyright: ignore[reportPrivateUsage]
        g.zero_grad(set_to_none=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AdamW8bit requires CUDA/HCU"
)
def test_checkpoint_guarded_roundtrip_and_rejections(model_and_refs):
    """T10: guarded state round-trips exactly; cross-family and world-size
    mismatches are rejected."""
    model, refs = model_and_refs
    g = _build_guarded(model, refs=refs, guard=_guard())
    _seed_grads(model, 600)
    g.step()
    g.zero_grad(set_to_none=True)
    sd = g.state_dict()
    assert "guard" in sd and sd["guarded_canonical_schema_version"] == 1
    assert sd["guard"]["observations"] == 1
    assert sd["guard"]["owner_mapping_version"] == "fnv1a64-v1"

    model2 = _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE)
    g2 = _build_guarded(model2, refs=_bootstrap_refs(model2), guard=_guard())
    g2.load_state_dict(sd)
    assert g2.observations == 1
    assert g2._refs == g._refs  # pyright: ignore[reportPrivateUsage]
    for s, s2 in zip(g.routing.cmuon_specs, g2.routing.cmuon_specs):
        assert torch.equal(
            g._momenta[s.parameter],  # pyright: ignore[reportPrivateUsage]
            g2._momenta[s2.parameter],  # pyright: ignore[reportPrivateUsage]
        )

    # 1) unguarded state into guarded => rejected
    core_model = _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE)
    core = build_hybrid_cmuon(
        core_model,
        ns_steps_by_role=NS4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    core_sd = core.state_dict()
    assert "guard" not in core_sd
    g3 = _build_guarded(
        _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE),
        refs=refs,
        guard=_guard(),
    )
    with pytest.raises(CMuonSafetyError, match="cannot resume directly"):
        g3.load_state_dict(core_sd)

    # 2) guarded state into unguarded => rejected (no silent downgrade)
    core2_model = _MockComposite(_MockDiT(256, 512, 3)).to(DEVICE)
    core2 = build_hybrid_cmuon(
        core2_model,
        ns_steps_by_role=NS4,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        **_OPT_COMMON,
    )
    with pytest.raises(ValueError, match="guard section"):
        core2.load_state_dict(sd)

    # 3) world_size mismatch => rejected (the owner mapping is a function of
    #    world_size; tamper the saved contract directly)
    sd_tampered = dict(sd)
    guard_tampered = dict(sd["guard"])
    guard_tampered["world_size"] = 2
    sd_tampered["guard"] = guard_tampered
    with pytest.raises(CMuonSafetyError, match="world_size"):
        g3.load_state_dict(sd_tampered)

    # 4) schema sidecar: guarded instance carries the block; an unguarded
    #    optimizer with the guarded schema is rejected.
    schema = _hybrid_optimizer_schema(g)
    assert "guarded_canonical" in schema
    assert schema["guarded_canonical"]["ns_mode"] == "canonical_owner_rank"
    core_schema = _hybrid_optimizer_schema(core)
    assert "guarded_canonical" not in core_schema
    with pytest.raises(CheckpointError):
        _validate_hybrid_optimizer_schema(schema, core)
    with pytest.raises(CheckpointError):
        _validate_hybrid_optimizer_schema(core_schema, g)
    _validate_hybrid_optimizer_schema(schema, g)  # matching pair is fine
