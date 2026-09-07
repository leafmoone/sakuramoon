"""Post-commit parameter fingerprint: no-grad contract + BASE/FIX parity.

The F3 post-commit rank invariant (``HybridCMuonCanonicalNS4FP32Rescue.step``)
computes cross-rank parameter fingerprints (per-spec FP32 RMS/max) from
``spec.parameter.float()`` and all-reduces them. Training parameters carry
``requires_grad=True`` and the trainer does not wrap optimizer steps in
``torch.no_grad()``, so the pre-fix implementation built an autograd graph
rooted at the training parameters and saved large FP32 intermediates (full
parameter copies) for backward for the lifetime of the fingerprint list.

Contract under test: the diagnostic must be graph-free and must not save
tensors for backward. The fix wraps the existing fingerprint block in
``torch.no_grad()`` and changes nothing else (same statements, order, dtypes,
collectives, and error message).

The tests drive the REAL F3 ``step()`` with ``world_size=2`` on one process
(collectives mocked: identity reduce == identical ranks; a controlled mismatch
is injected only where asserted). ``torch.enable_grad()`` is asserted at entry
and after every step. BASE (a88c20c) is loaded from the sibling main clone
when present, so the parity/semantics tests compare both versions in the same
environment.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from test_cmuon import LR, _MockComposite, _MockDiT, _seed_grads
from test_fp32_rescue_f3 import NS4, _alpha_for, _bootstrap_refs

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.optim.cmuon import route_cmuon_parameters
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig

_SKIP = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HCU"
)

_BASE_PATH = Path(
    os.environ.get(
        "SAKURAMOON_BASE_FP32_RESCUE",
        "/sakuramoon-runtime/sakuramoon-irepa/src/sakuramoon/optim/fp32_rescue.py",
    )
)


def _load_base_module():
    """Load the pre-fix (BASE, a88c20c) fp32_rescue as a standalone module.

    Its imports resolve against the same ``sakuramoon`` package on
    ``sys.path``; only ``fp32_rescue.py`` differs between BASE and FIXED, so
    every other dependency is identical. Returns None when the BASE tree is
    absent (the parity/semantics tests then skip).
    """
    if not _BASE_PATH.is_file():
        return None
    name = "fp32_rescue_base_under_test"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    # register before exec: module-level dataclasses resolve annotations
    # through sys.modules by class __module__
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_model(device: torch.device) -> _MockComposite:
    torch.manual_seed(1234)
    model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
    # plain detached grads from a device Generator (no create_graph)
    _seed_grads(model, 101)
    return model


def _const_ns_bf16(target: float):
    """Deterministic BF16 NS stand-in: a constant matrix whose staged-delta
    RMS lands at ~``target`` (comfortably in-band: > 0, < ceiling)."""

    def _v_for_shape(shape, device) -> float:
        alpha = _alpha_for(shape)
        return float(torch.tensor(target / abs(alpha), dtype=torch.bfloat16))

    def ns(grad, ns_steps, ns_coefficients, eps):
        v = _v_for_shape(tuple(grad.shape), grad.device)
        return torch.full(
            tuple(grad.shape), v, dtype=torch.bfloat16, device=grad.device
        )

    ns._v_for_shape = _v_for_shape  # type: ignore[attr-defined]
    return ns


class _SpyCollectives:
    """Mock ``dist.all_reduce``/``dist.broadcast`` for a single-process
    world_size=2. Identity reduce semantics (both ranks hold identical
    values). With ``perturb_param_lo > 0`` the parameter-fingerprint lo (MIN)
    reduce emulates the other rank holding a strictly smaller value."""

    def __init__(self, perturb_param_lo: float = 0.0):
        self.calls: list[dict] = []
        self.perturb = perturb_param_lo

    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        op_name = getattr(op, "name", str(op))
        fp32_before = sum(
            1
            for c in self.calls
            if c["kind"] == "all_reduce" and c["dtype"] == torch.float32
        )
        # per-step fp32 all_reduce order:
        # [delta lo, delta hi, param lo, param hi]
        role = None
        if tensor.dtype == torch.float32:
            if fp32_before == 0 and op_name == "MIN":
                role = "delta_lo"
            elif fp32_before == 1 and op_name == "MAX":
                role = "delta_hi"
            elif fp32_before == 2 and op_name == "MIN":
                role = "param_lo"
            elif fp32_before == 3 and op_name == "MAX":
                role = "param_hi"
        rec = {
            "kind": "all_reduce",
            "op": op_name,
            "dtype": tensor.dtype,
            "numel": tensor.numel(),
            "shape": tuple(tensor.shape),
            "requires_grad": bool(tensor.requires_grad),
            "grad_fn": (
                type(tensor.grad_fn).__name__
                if tensor.grad_fn is not None
                else None
            ),
            "role": role,
            "value": tensor.detach().clone(),
        }
        self.calls.append(rec)
        if self.perturb > 0.0 and role == "param_lo":
            tensor.sub_(self.perturb)

    def broadcast(self, tensor, src=0, group=None, async_op=False):
        self.calls.append(
            {
                "kind": "broadcast",
                "dtype": tensor.dtype,
                "numel": tensor.numel(),
                "src": src,
            }
        )

    def param_fp(self) -> list[dict]:
        return [c for c in self.calls if c.get("role") in ("param_lo", "param_hi")]


def _patch_module(module, spy: _SpyCollectives, ns) -> list:
    """Patch the NS entry points + collectives for one module; return an undo
    list (the NS names are module attributes; ``dist`` is the shared
    ``torch.distributed`` module used by every fp32_rescue module)."""
    saved = [
        (module, "cmuon_zeroth_power_bf16", module.cmuon_zeroth_power_bf16),
        (module, "cmuon_zeroth_power_fp32", module.cmuon_zeroth_power_fp32),
        (torch.distributed, "all_reduce", torch.distributed.all_reduce),
        (torch.distributed, "broadcast", torch.distributed.broadcast),
    ]
    module.cmuon_zeroth_power_bf16 = ns
    module.cmuon_zeroth_power_fp32 = ns
    torch.distributed.all_reduce = spy.all_reduce
    torch.distributed.broadcast = spy.broadcast
    return saved


def _unpatch(saved: list) -> None:
    for target, name, value in reversed(saved):
        setattr(target, name, value)


def _build_opt(
    model,
    module,
    tmp_path: Path,
    *,
    rank: int = 0,
    world_size: int = 2,
    invariant: bool = True,
    logs: list[str] | None = None,
):
    guard = GuardedCanonicalGuardConfig(
        guard_ratio=0.05,
        reference_decay=0.999,
        min_reference=1e-12,
        numerical_floor=1e-20,
        warmup_observations=0,
        invariant_check=invariant,
    )
    torch.manual_seed(888)  # identical bootstrap refs across versions
    refs = _bootstrap_refs(model)
    return module.build_fp32_rescue(
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
        rank=rank,
        world_size=world_size,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        stats_logger=(logs.append if logs is not None else None),
        hard_fail_artifact_root=str(tmp_path / "mirror"),
        legacy_forensic_dir=str(tmp_path / "forensic"),
        emergency_capsule_root=str(tmp_path / "capsule"),
    )


def _all_owner(opt) -> None:
    # single-process test: make every chunk owner-local (rank 0)
    opt.owner_of = lambda fqn, chunk: 0


def _snapshot(model, opt) -> dict:
    return {
        "params": {n: p.detach().cpu() for n, p in model.named_parameters()},
        "grads": {
            n: p.grad.detach().cpu()
            for n, p in model.named_parameters()
            if p.grad is not None
        },
        "refs": {k: float(v) for k, v in opt._refs.items()},
        "counters": (
            opt.observations,
            opt.bf16_attempts,
            opt.bf16_safety_failures,
            opt.fp32_attempts,
            opt.fp32_rescues,
            opt.fp32_rescue_failures,
            opt.fp32_low_delta_rescues,
            dict(opt.fp32_low_delta_by_role),
            opt._steps_this_process,
        ),
        "max_param_rank_diff": float(opt.max_param_rank_diff),
        "sr_rng": opt.sr_rng.state.clone().cpu(),
        "state_dict": opt.state_dict(),
    }


def _assert_state_dicts_equal(a, b, path: str = "") -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: key set differs"
        for k in a:
            _assert_state_dicts_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_state_dicts_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        assert a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)
        if type(a) is torch.Tensor and type(b) is torch.Tensor:
            same = bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
            assert same, f"{path}: tensor differs"
        else:
            # tensor subclass (torchao OptimState8bit): aten.equal is
            # unimplemented; compare the constituent tensors
            # (codes/scale/qmap) and scalar attrs exactly
            attrs = list(getattr(a, "tensor_attrs", []) or [])
            assert attrs, f"{path}: unknown tensor subclass {type(a)}"
            assert attrs == list(getattr(b, "tensor_attrs", []) or [])
            for name in attrs:
                va, vb = getattr(a, name), getattr(b, name)
                assert isinstance(va, torch.Tensor) and isinstance(
                    vb, torch.Tensor
                ), f"{path}.{name}: not a plain tensor"
                assert (
                    va.dtype == vb.dtype
                    and tuple(va.shape) == tuple(vb.shape)
                    and bool(torch.equal(va.detach().cpu(), vb.detach().cpu()))
                ), f"{path}.{name}: differs"
            extra_a = {k: v for k, v in vars(a).items() if k not in attrs}
            extra_b = {k: v for k, v in vars(b).items() if k not in attrs}
            assert set(extra_a) == set(extra_b), (
                f"{path}: scalar attrs differ: {set(extra_a) ^ set(extra_b)}"
            )
            for k, va in extra_a.items():
                vb = extra_b[k]
                if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
                    continue
                assert va == vb, f"{path}.{k}: {va!r} != {vb!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------------------
# A. autograd boundary: the diagnostic must be graph-free (RED anchor)
# ---------------------------------------------------------------------------
@_SKIP
def test_fingerprint_no_grad_contract(monkeypatch, tmp_path) -> None:
    """Pre-fix: the post-commit fingerprint tensors carry a live autograd
    graph (requires_grad=True / grad_fn set) and the diagnostic saves
    parameter-sized FP32 tensors for backward. Post-fix: graph-free.
    """
    torch.enable_grad()
    assert torch.is_grad_enabled() is True
    device = torch.device("cuda:0")
    model = _make_model(device)
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    param_numels = {spec.parameter.numel() for spec in routing.cmuon_specs}
    assert all(spec.parameter.requires_grad for spec in routing.cmuon_specs)
    assert all(
        spec.parameter.grad is not None
        and spec.parameter.grad.requires_grad is False
        for spec in routing.cmuon_specs
    )

    spy = _SpyCollectives()
    saved = _patch_module(fr, spy, _const_ns_bf16(0.2 * LR))
    try:
        logs: list[str] = []
        opt = _build_opt(model, fr, tmp_path, logs=logs)
        _all_owner(opt)
        records: list[dict] = []
        seen: set[tuple] = set()

        def pack(t, *a, **k):
            key = (t.device.index, t.data_ptr(), t.nelement(), t.element_size())
            if key not in seen:
                seen.add(key)
                records.append(
                    {
                        "shape": tuple(t.shape),
                        "dtype": str(t.dtype),
                        "numel": t.nelement(),
                        "bytes": t.nelement() * t.element_size(),
                    }
                )
            return t

        def unpack(t, *a, **k):
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            opt.step()
    finally:
        _unpatch(saved)

    # the post-commit parameter fingerprint actually ran
    # (world_size=2, invariant_check=True)
    pfp = spy.param_fp()
    assert len(pfp) == 2
    lo, hi = pfp
    assert lo["op"] == "MIN" and hi["op"] == "MAX"
    # contract: graph-free diagnostic tensors
    assert lo["requires_grad"] is False, (
        "param-fingerprint lo must not require grad (diagnostic built an "
        f"autograd graph; grad_fn={lo['grad_fn']})"
    )
    assert lo["grad_fn"] is None, (
        f"param-fingerprint lo has a live graph: {lo['grad_fn']}"
    )
    assert hi["requires_grad"] is False
    assert hi["grad_fn"] is None
    # contract: no parameter-sized tensor saved for backward by the
    # diagnostic (dedup by unique storage: data_ptr + size)
    param_sized = [r for r in records if r["numel"] in param_numels]
    assert not param_sized, (
        "diagnostic saved param-sized tensors for backward: "
        f"{param_sized[:4]}..."
    )
    # invariants preserved
    assert all(spec.parameter.requires_grad for spec in routing.cmuon_specs)
    assert torch.is_grad_enabled() is True


# ---------------------------------------------------------------------------
# B/E. bit-exact parity of fingerprints and full optimizer state, BASE vs FIX
# ---------------------------------------------------------------------------
@_SKIP
def test_fingerprint_and_state_bitexact_base_vs_fixed(tmp_path) -> None:
    fr_base = _load_base_module()
    if fr_base is None:
        pytest.skip(f"BASE module not present at {_BASE_PATH}")
    device = torch.device("cuda:0")
    results: dict[str, dict] = {}
    for tag, module in (("base", fr_base), ("fixed", fr)):
        model = _make_model(device)
        spy = _SpyCollectives()
        saved = _patch_module(module, spy, _const_ns_bf16(0.2 * LR))
        rng_before = torch.get_rng_state()
        try:
            opt = _build_opt(model, module, tmp_path / tag)
            _all_owner(opt)
            opt.step()
        finally:
            _unpatch(saved)
        pfp = spy.param_fp()
        assert len(pfp) == 2
        lo, hi = pfp
        results[tag] = {
            "lo": lo["value"].cpu(),
            "hi": hi["value"].cpu(),
            "snapshot": _snapshot(model, opt),
            "torch_rng_before": rng_before,
            "torch_rng_after": torch.get_rng_state(),
        }
    b, f = results["base"], results["fixed"]
    # B: the fingerprint values are bit-exact between versions
    assert torch.equal(b["lo"], f["lo"]), "param-fingerprint lo values differ"
    assert torch.equal(b["hi"], f["hi"]), "param-fingerprint hi values differ"
    bs, fs = b["snapshot"], f["snapshot"]
    for name in bs["params"]:
        assert torch.equal(bs["params"][name], fs["params"][name]), (
            f"param {name} differs"
        )
    for name in bs["grads"]:
        assert torch.equal(bs["grads"][name], fs["grads"][name]), (
            f"grad {name} differs"
        )
    assert bs["refs"] == fs["refs"], "guard refs differ"
    assert bs["counters"] == fs["counters"], f"counters differ: {bs['counters']} != {fs['counters']}"
    assert bs["max_param_rank_diff"] == fs["max_param_rank_diff"]
    assert torch.equal(bs["sr_rng"], fs["sr_rng"]), "SR RNG state differs"
    # get_rng_state() returns a multi-element uint8 tensor
    assert torch.equal(b["torch_rng_before"], f["torch_rng_before"])
    assert torch.equal(b["torch_rng_after"], f["torch_rng_after"])
    _assert_state_dicts_equal(bs["state_dict"], fs["state_dict"])


# ---------------------------------------------------------------------------
# C. branch execution matrix
# ---------------------------------------------------------------------------
@_SKIP
def test_branch_execution_matrix(tmp_path) -> None:
    device = torch.device("cuda:0")

    # C1: invariant_check=True + world_size=2 -> param fingerprint runs
    model = _make_model(device)
    spy = _SpyCollectives()
    saved = _patch_module(fr, spy, _const_ns_bf16(0.2 * LR))
    try:
        opt = _build_opt(model, fr, tmp_path / "c1")
        _all_owner(opt)
        opt.step()
    finally:
        _unpatch(saved)
    assert len(spy.param_fp()) == 2

    # C2: invariant_check=False + world_size=2 -> param fingerprint absent
    # (the delta fingerprint still runs, unchanged)
    model = _make_model(device)
    spy = _SpyCollectives()
    saved = _patch_module(fr, spy, _const_ns_bf16(0.2 * LR))
    try:
        opt = _build_opt(model, fr, tmp_path / "c2", invariant=False)
        _all_owner(opt)
        opt.step()
    finally:
        _unpatch(saved)
    assert spy.param_fp() == []
    fp32_reduces = [
        c
        for c in spy.calls
        if c["kind"] == "all_reduce" and c["dtype"] == torch.float32
    ]
    assert [c["role"] for c in fp32_reduces] == ["delta_lo", "delta_hi"]

    # C3: world_size=1 -> no collectives at all (original behavior)
    model = _make_model(device)
    spy = _SpyCollectives()
    saved = _patch_module(fr, spy, _const_ns_bf16(0.2 * LR))
    try:
        opt = _build_opt(model, fr, tmp_path / "c3", world_size=1)
        opt.step()
    finally:
        _unpatch(saved)
    assert spy.calls == []

    # C4: caller under an outer no_grad: the step completes and step()
    # itself must not change the caller grad mode (still disabled after
    # step() returns, inside the caller's no_grad block)
    model = _make_model(device)
    spy = _SpyCollectives()
    saved = _patch_module(fr, spy, _const_ns_bf16(0.2 * LR))
    try:
        opt = _build_opt(model, fr, tmp_path / "c4")
        _all_owner(opt)
        with torch.no_grad():
            assert torch.is_grad_enabled() is False
            opt.step()
            assert torch.is_grad_enabled() is False, (
                "step() changed the caller grad mode under no_grad"
            )
    finally:
        _unpatch(saved)
    # the no_grad context manager restores the outer mode (enabled here)
    assert torch.is_grad_enabled() is True
    assert len(spy.param_fp()) == 2


# ---------------------------------------------------------------------------
# D. safety semantics preserved: a controlled cross-rank mismatch still
# raises the same error (pre-fix and post-fix)
# ---------------------------------------------------------------------------
@_SKIP
def test_controlled_mismatch_still_fails(tmp_path) -> None:
    fr_base = _load_base_module()
    modules: list[tuple[str, object]] = [("fixed", fr)]
    if fr_base is not None:
        modules.append(("base", fr_base))
    device = torch.device("cuda:0")
    for tag, module in modules:
        torch.enable_grad()
        model = _make_model(device)
        spy = _SpyCollectives(perturb_param_lo=1e-3)
        saved = _patch_module(module, spy, _const_ns_bf16(0.2 * LR))
        opt = None
        try:
            opt = _build_opt(model, module, tmp_path / tag)
            _all_owner(opt)
            with pytest.raises(CMuonSafetyError) as ei:
                opt.step()
        finally:
            _unpatch(saved)
        msg = str(ei.value)
        assert msg.startswith(
            "fp32-rescue rank invariant violated: "
            "cross-rank parameter fingerprint diff"
        ), msg
        # the message is rounded to 3 significant digits (%.3e); the
        # telemetry attribute keeps the full-precision value
        diff = float(msg.split("diff ")[1].split(" ")[0])
        assert diff > 0.0
        assert opt.max_param_rank_diff > 0.0
        assert opt.max_param_rank_diff == pytest.approx(diff, rel=1e-3)
        # not the pre-commit hard-fail message
        assert "safety violation" not in msg
        # the caller grad mode is restored even after the raise
        assert torch.is_grad_enabled() is True
