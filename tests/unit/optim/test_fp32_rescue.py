"""CPU/HCU unit tests for the FP32-rescue candidate
(``hybrid_cmuon_canonical_ns4_fp32_rescue``, D2 spec section 11).

A. safe step never invokes the FP32 path
B. mocked bad BF16 -> FP32 rescue: step succeeds, delta is the FP32 one,
   counters consistent
C. mocked bad BF16 + bad FP32 -> CMuonSafetyError, ZERO parameter change
   (atomic), observations unchanged
D. owner-only rescue across 2 ranks (each rank rescues only its own inputs)
E. broadcast consistency after rescue: rank-invariant + zero delta spread
   (2 ranks)
F. checkpoint round-trip: rescue counters state-exact; parent-class
   checkpoint (no fp32_rescue block) resumes with zeroed telemetry
G. AdamW part identical to the core hybrid (same seed/grads, bit-exact)
H. retired mechanisms are not selectable: no pre-NS low-signal skip
   (low signal still enters NS), the base skip gate is never consulted,
   config requires the guard section for the new name
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
from test_cmuon import (
    LR,
    _MockComposite,
    _MockDiT,
    _seed_grads,
)

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.config.schema import CMuonGuardConfig, OptimizerConfig
from sakuramoon.optim.cmuon import (
    build_hybrid_cmuon,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import (
    HybridCMuonCanonicalNS4FP32Rescue,
    build_fp32_rescue,
)
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
    stable_owner,
)

NS4 = resolve_ns_map(None, 4)
CEILING = 10.0 * 0.2 * LR


def _guard() -> GuardedCanonicalGuardConfig:
    return GuardedCanonicalGuardConfig(
        guard_ratio=0.05,
        reference_decay=0.999,
        min_reference=1e-12,
        numerical_floor=1e-20,
        warmup_observations=0,
        invariant_check=True,
    )


def _bootstrap_refs(model, *, scale: float = 1e-3) -> dict[str, float]:
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


def _build(
    model,
    *,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    logs: list[str] | None = None,
) -> HybridCMuonCanonicalNS4FP32Rescue:
    refs = _bootstrap_refs(model)
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
        guard_bootstrap_refs=refs,
        rank=rank,
        world_size=world_size,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        stats_logger=(logs.append if logs is not None else None),
    )


def _n_inputs(model) -> int:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    return sum(spec.chunk_count for spec in routing.cmuon_specs)


def _evil_ns_amplified(amplify: float):
    """A stand-in NS that returns the REAL bf16 NS amplified (chaos
    stand-in: finite but far above the safety ceiling)."""
    from sakuramoon.optim.cmuon import cmuon_zeroth_power_bf16

    def evil(grad, ns_steps, ns_coefficients, eps):
        return cmuon_zeroth_power_bf16(grad, ns_steps, ns_coefficients, eps) * amplify

    return evil


def _param_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
    }


def _snap_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(k in a and a[k].equal(b[k]) for k in b)


def test_H_schema_rejects_name_without_guard() -> None:
    base = {
        "name": "hybrid_cmuon_canonical_ns4_fp32_rescue",
        "base_lr": 5e-5,
        "reference_batch": 800,
        "lr_scaling": "linear_global_batch",
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "block_size": 256,
        "bf16_stochastic_round": True,
        "matrix_weight_decay": 0.0,
        "sensitive_weight_decay": 0.0,
        "cmuon_ns": {"attention_q": 4},
    }
    with pytest.raises(ValueError, match="cmuon_guard"):
        OptimizerConfig(**base)
    with pytest.raises(ValueError, match="references"):
        OptimizerConfig(
            **base,
            cmuon_guard=CMuonGuardConfig(
                guard_ratio=0.05,
                reference_decay=0.999,
                min_reference=1e-12,
                numerical_floor=1e-20,
                warmup_observations=0,
                references={},
            ),
        )
    # the guard section IS accepted for the new name
    OptimizerConfig(
        **base,
        cmuon_guard=CMuonGuardConfig(
            guard_ratio=0.05,
            reference_decay=0.999,
            min_reference=1e-12,
            numerical_floor=1e-20,
            warmup_observations=0,
            references={"x#chunk0": 1e-3},
        ),
    )


# A. safe step never invokes the FP32 path
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer build requires CUDA/HCU"
)
def test_A_safe_step_never_invokes_fp32() -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    logs: list[str] = []
    opt = _build(model, logs=logs)
    for seed in (1, 2, 3):
        _seed_grads(model, seed)
        opt.step()
    assert opt.observations == 3
    assert opt.bf16_attempts == 3 * _n_inputs(model)
    assert opt.fp32_attempts == 0, "safe steps must never call the FP32 path"
    assert opt.fp32_rescues == 0
    assert opt.fp32_rescue_failures == 0
    for p in model.parameters():
        assert torch.isfinite(p.float()).all()
    # the rescue telemetry line is only emitted when a rescue happened
    assert not any(l.startswith('{"fp32_rescue_obs"') for l in logs)


# B. mocked bad BF16 -> FP32 rescue
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer build requires CUDA/HCU"
)
def test_B_bad_bf16_rescued_by_fp32(monkeypatch) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    before = _param_snapshot(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_ns_amplified(1e9))
    opt = _build(model)
    _seed_grads(model, 11)
    opt.step()  # must NOT raise: every BF16 failure is FP32-rescued
    after = _param_snapshot(model)
    assert opt.observations == 1
    assert opt.bf16_attempts == n
    assert opt.bf16_safety_failures == n
    assert opt.fp32_attempts == n
    assert opt.fp32_rescues == n
    assert opt.fp32_rescue_failures == 0
    changed = [
        k for k in before if not before[k].equal(after[k])
    ]
    assert changed, "the rescued step must commit the FP32 deltas"
    # every committed CMuon delta sits inside the Moonlight-sane band
    # (the rescue floor rejected anything degenerate)
    for name, p in model.named_parameters():
        d = (after[name] - before[name]).float()
        if d.abs().max() == 0:
            continue
        assert d.pow(2).mean().sqrt().item() <= CEILING * 2, name


# C. mocked bad BF16 + bad FP32 -> atomic zero-commit failure
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer build requires CUDA/HCU"
)
def test_C_bad_fp32_too_fails_atomically(monkeypatch) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    n = _n_inputs(model)
    before = _param_snapshot(model)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_ns_amplified(1e9))
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", _evil_ns_amplified(1e9))
    opt = _build(model)
    _seed_grads(model, 12)
    with pytest.raises(CMuonSafetyError, match="both failed"):
        opt.step()
    after = _param_snapshot(model)
    assert _snap_equal(before, after), (
        "a failed step must commit NOTHING (CMuon and AdamW parameters)"
    )
    assert opt.observations == 0, "observations must not advance on failure"
    assert opt.fp32_attempts == n
    assert opt.fp32_rescues == 0
    assert opt.fp32_rescue_failures == n


# D+E. 2 ranks: owner-only rescue + broadcast/rank consistency
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="2-rank test requires CUDA/HCU (nccl)"
)
def test_DE_two_rank_owner_only_and_consistency(tmp_path) -> None:
    import torch.distributed as dist
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory(dir=tmp_path) as td:
        init_file = os.path.join(td, "init")

        def worker(rank: int, world_size: int, i: str) -> None:
            device = torch.device("cuda", rank)
            torch.cuda.set_device(device)
            dist.init_process_group(
                backend="nccl",
                init_method=f"file://{i}",
                world_size=world_size,
                rank=rank,
                device_id=device,
            )
            try:
                torch.manual_seed(20260831)
                model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
                fr.cmuon_zeroth_power_bf16 = _evil_ns_amplified(1e9)
                opt = _build(model, rank=rank, world_size=world_size)
                routing = route_cmuon_parameters(
                    model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
                )
                owned = sum(
                    1
                    for spec in routing.cmuon_specs
                    for ci in range(spec.chunk_count)
                    if stable_owner(spec.name, ci, world_size) == rank
                )
                _seed_grads(model, 21)
                opt.step()
                (tmp_path / f"per-rank-{rank}.json").write_text(
                    json.dumps(
                        {
                            "rank": rank,
                            "observations": opt.observations,
                            "bf16_attempts": opt.bf16_attempts,
                            "fp32_attempts": opt.fp32_attempts,
                            "fp32_rescues": opt.fp32_rescues,
                            "fp32_rescue_failures": opt.fp32_rescue_failures,
                            "owned_inputs": owned,
                            "max_delta_rank_spread": opt.max_delta_rank_spread,
                            "max_param_rank_diff": opt.max_param_rank_diff,
                        }
                    )
                )
            finally:
                dist.destroy_process_group()

        mp.spawn(worker, args=(2, init_file), nprocs=2, join=True)
        r0 = json.loads((tmp_path / "per-rank-0.json").read_text())
        r1 = json.loads((tmp_path / "per-rank-1.json").read_text())

    for r in (r0, r1):
        # D. owner-only: a rank attempts the FP32 path ONLY for its inputs
        assert r["fp32_attempts"] == r["owned_inputs"], r
        assert r["bf16_attempts"] == r["owned_inputs"], r
        assert r["fp32_rescues"] == r["owned_inputs"], r
        assert r["fp32_rescue_failures"] == 0, r
        assert r["observations"] == 1, r
        # E. broadcast + rank consistency after rescue
        assert r["max_delta_rank_spread"] == 0.0, r
        assert r["max_param_rank_diff"] == 0.0, r
    assert r0["fp32_attempts"] + r1["fp32_attempts"] == _n_inputs(
        _MockComposite(_MockDiT(256, 512, 2))
    )


# F. checkpoint round-trip (rescue counters state-exact)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer build requires CUDA/HCU"
)
def test_F_checkpoint_roundtrip(monkeypatch, tmp_path) -> None:
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", _evil_ns_amplified(1e9))
    opt = _build(model)
    _seed_grads(model, 31)
    opt.step()  # a rescued step (counters non-zero)
    monkeypatch.undo()
    _seed_grads(model, 32)
    opt.step()  # a plain safe step after restoring the real bf16 path

    state = opt.state_dict()
    assert "fp32_rescue" in state["guard"]
    torch.save(state, tmp_path / "opt.pt")

    model2 = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    opt2 = _build(model2)
    opt2.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=False))
    for k in (
        "bf16_attempts",
        "bf16_safety_failures",
        "fp32_attempts",
        "fp32_rescues",
        "fp32_rescue_failures",
    ):
        assert getattr(opt2, k) == getattr(opt, k), k
    assert opt2.rescue_by_role == opt.rescue_by_role
    assert opt2.observations == opt.observations == 2
    # parent-class checkpoint (no fp32_rescue block) resumes cleanly
    del state["guard"]["fp32_rescue"]
    opt3 = _build(model2)
    opt3.load_state_dict(state)
    assert opt3.fp32_attempts == 0
    assert opt3.observations == 2


# G. AdamW part identical to the core hybrid
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer build requires CUDA/HCU"
)
def test_G_adamw_path_untouched() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    model_a = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    torch.manual_seed(7)
    model_b = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    common = {
        "lr": LR,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "block_size": 256,
        "bf16_stochastic_round": True,
        "matrix_weight_decay": 0.0,
        "sensitive_weight_decay": 0.0,
        "sr_seed": 44,
    }
    core = build_hybrid_cmuon(
        model_b, ns_steps_by_role=NS4, momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False, **common,
    )
    rescue = _build(model_a)
    for seed in (41, 42):
        _seed_grads(model_a, seed)
        _seed_grads(model_b, seed)
        rescue.step()
        core.step()
    for (na, pa), (nb, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert na == nb
        assert pa.detach().cpu().equal(pb.detach().cpu()), (
            f"parameter {na} differs: the rescue candidate must be "
            "bit-identical to the core update on safe inputs"
        )


# H. retired mechanisms are not selectable
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="optimizer build requires CUDA/HCU"
)
def test_H_no_pre_ns_skip_gate(monkeypatch) -> None:
    # H1: the base-class low-signal skip gate must NEVER be consulted.
    device = torch.device("cuda:0")
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    opt = _build(model)

    def explode(*a, **k):  # pyright: ignore[reportUnusedFunction]
        raise AssertionError("the retired pre-NS skip gate must not be called")

    monkeypatch.setattr(opt, "_is_low_signal", explode)
    _seed_grads(model, 51)
    opt.step()
    assert opt.bf16_attempts == _n_inputs(model), (
        "every gradient-bearing chunk must enter NS (no low-signal skip)"
    )
    assert opt.skip_total == 0

    # H2: a pathologically low signal still enters NS (every input gets a
    # BF16 attempt) and the step commits — the protection is POST-NS
    # (ceiling + FP32 rescue), never a pre-NS skip.
    n0 = opt.bf16_attempts
    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(1e-30)
    opt.step()
    assert opt.bf16_attempts == n0 + _n_inputs(model), (
        "a low-signal input must NOT be skipped before NS"
    )
    assert opt.fp32_rescue_failures == 0
    assert opt.observations >= 2
