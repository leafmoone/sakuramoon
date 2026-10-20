"""CPU index/step regressions, NOT native TorchAO, SR or accelerator acceptance.

Only the unit kernel boundary is substituted. Routing, momentum preparation,
reference updates, safety verdicts and commit execute the production step.
The CPU AdamW/SR adapters are explicit and also reused by the Gloo tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
import torch
from torch import nn

import sakuramoon.optim.fp32_rescue as rescue
import sakuramoon.optim.guarded_canonical as guarded
from sakuramoon.optim.cmuon import (
    CMuonChunkSpec,
    CMuonConfig,
    CMuonRouting,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec

_T = TypeVar("_T")
VARIANTS = ("guarded", "rescue")
# Q/K/V are separate parameters; the two fused matrices exercise both chunk axes.
_LAYOUT = (
    ("dit.blocks.slot_00.attention.q_proj.weight", (3, 2), 1, 0, "attention_q"),
    ("dit.blocks.slot_00.attention.k_proj.weight", (2, 4), 1, 0, "attention_k"),
    ("dit.blocks.slot_00.attention.v_proj.weight", (4, 3), 1, 0, "attention_v"),
    ("dit.blocks.slot_00.mlp.in_proj.weight", (3, 8), 2, 1, "ffn_in"),
    ("dit.conditioner.shared_block_projection.weight", (12, 3), 6, 0, "adaln_shared"),
    ("dit.blocks.slot_00.attention.out_proj.weight", (2, 5), 1, 0, "attention_out"),
)
MASKS = (
    (), (0,), (1,), (2,), (5,), (0, 1), (1, 2, 3), (0, 3), (2, 4),
    tuple(range(len(_LAYOUT))),
)


class _CPUSR:
    """Stateful call adapter; does not initialize CUDA or claim native SR."""

    def __init__(self) -> None:
        self.calls = 0

    def run_step(self, fn: Callable[[], _T]) -> _T:
        self.calls += 1
        return fn()

    def state_dict(self) -> dict[str, int]:
        return {"cpu_sr_calls": self.calls}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.calls = state["cpu_sr_calls"]


class _CPUAdamW:
    def __init__(self, specs: tuple[ParameterSpec, ...], sr: _CPUSR) -> None:
        self.optimizer = torch.optim.AdamW(
            [s.parameter for s in specs], lr=0.01, weight_decay=0.1, foreach=False
        )
        self.audit, self.sr_rng = ParameterAudit(specs), sr

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.sr_rng.load_state_dict(state["sr_rng"])


def make_optimizer(root: Path, variant: str, rank: int = 0, world: int = 1) -> Any:
    specs: list[CMuonChunkSpec] = []
    audit: list[ParameterSpec] = []
    for index, (name, shape, count, dim, role) in enumerate(_LAYOUT):
        parameter = nn.Parameter(torch.full(shape, 0.5 + index / 8, dtype=torch.bfloat16))
        specs.append(CMuonChunkSpec(
            name, parameter, 0.1, count, dim,
            tuple(f"synthetic_{i}" for i in range(count)), role,
        ))
        audit.append(ParameterSpec(name, parameter, "matrix_decay", 0.1))
    adam_specs = tuple(
        ParameterSpec(f"synthetic.bias_{i}", nn.Parameter(torch.ones(2)),
                      "sensitive_no_decay", 0.0)
        for i in range(2)
    )
    routing = CMuonRouting(ParameterAudit(tuple(audit) + adam_specs), tuple(specs), adam_specs)
    sr = _CPUSR()
    kwargs: dict[str, Any] = {
        "routing": routing,
        "cfg": CMuonConfig(
            lr=0.01, momentum=0.5, momentum_dtype="bfloat16",
            ns_steps_by_role=resolve_ns_map({"ffn_in": 3, "adaln_shared": 5}, 4),
            chunk_rescale_sqrt_n=True,
        ),
        "adamw": _CPUAdamW(adam_specs, sr),
        "sr_rng": sr,
        "guard_cfg": guarded.GuardedCanonicalGuardConfig(0.1, 0.9, 1e-6, 1e-10, 0, True),
        "rank": rank, "world_size": world,
        "guard_bootstrap_refs": {s.name: 0.125 for s in specs},
    }
    # Collaborators satisfy the runtime protocol, not CUDA-specific nominal types.
    opt: Any
    if variant == "guarded":
        opt = guarded.HybridCMuonGuardedCanonical(**kwargs)
    else:
        opt = rescue.HybridCMuonCanonicalNS4FP32Rescue(
            **kwargs, legacy_forensic_dir=root / "legacy",
            emergency_capsule_root=root / "unused-capsules",
            hard_fail_artifact_root=root / "unused-mirror",
        )
    for index, spec in enumerate(specs):
        opt._momenta[spec.parameter].fill_(0.125 * (index + 1))
    return opt


def _chunks(tensor: torch.Tensor, spec: CMuonChunkSpec) -> tuple[torch.Tensor, ...]:
    size = tensor.shape[spec.chunk_dim] // spec.chunk_count
    return tuple(tensor.narrow(spec.chunk_dim, ci * size, size) for ci in range(spec.chunk_count))


def seed_gradients(opt: Any, missing: tuple[int, ...], *, adam_none: bool = False, step: int = 0) -> None:
    for index, spec in enumerate(opt.routing.cmuon_specs):
        if index in missing:
            spec.parameter.grad = None
            continue
        grad = torch.empty_like(spec.parameter)
        for ci, chunk in enumerate(_chunks(grad, spec)):
            chunk.fill_(0.25 * (index + 1) + ci / 16 + step / 8)
        spec.parameter.grad = grad
    for spec in opt.routing.adamw_specs:
        spec.parameter.grad = None if adam_none else torch.full_like(spec.parameter, 0.3)


def _serialized_clone(value: Any) -> Any:
    # Serialization, not deepcopy: also safe for quantized optimizer tensor state.
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False, map_location="cpu")


def assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        a_dict, b_dict = cast(dict[Any, Any], left), cast(dict[Any, Any], right)
        assert a_dict.keys() == b_dict.keys()
        for key in a_dict:
            assert_equal(a_dict[key], b_dict[key])
    elif isinstance(left, (list, tuple)):
        a_seq, b_seq = cast(list[Any], left), cast(list[Any], right)
        assert type(a_seq) is type(b_seq)
        assert len(a_seq) == len(b_seq)
        for a, b in zip(a_seq, b_seq):
            assert_equal(a, b)
    else:
        assert left == right


def snapshot(opt: Any) -> Any:
    attributes = cast(dict[str, Any], vars(opt))
    counters = {
        name: value for name, value in attributes.items()
        if isinstance(value, (int, float, bool)) or name in (
            "skip_by_role", "skip_by_fqn", "rescue_by_role", "fp32_low_delta_by_role"
        )
    }
    return _serialized_clone({
        "state": opt.state_dict(), "counters": counters,
        "weights": [s.parameter.detach() for s in opt.audit.specs],
    })


def expected_index(opt: Any) -> list[tuple[Any, int, tuple[int, ...], int]]:
    entries: list[tuple[Any, int, tuple[int, ...], int]] = []
    for spec in opt.routing.cmuon_specs:
        if spec.parameter.grad is None:
            continue
        shape = list(spec.parameter.shape)
        shape[spec.chunk_dim] //= spec.chunk_count
        entries.extend((spec, ci, tuple(shape), guarded.stable_owner(spec.name, ci, opt.world_size))
                       for ci in range(spec.chunk_count))
    return entries


def assert_index(opt: Any) -> None:
    active = opt._active_inputs()
    expected = expected_index(opt)
    assert len(active) == len(expected)
    for i, (spec, ci, shape, owner) in enumerate(expected):
        entry = active[i]
        assert entry.spec is spec
        assert (entry.chunk_idx, entry.key, entry.shape, entry.owner) == (
            ci, (spec.name, ci), shape, owner,
        )
        assert opt._input_key(i) == opt._input_key(i, active) == (spec.name, ci)
        assert opt._chunk_shape(i) == opt._chunk_shape(i, active) == shape
    with pytest.raises(IndexError):
        opt._input_key(len(active), active)
    with pytest.raises(IndexError):
        opt._chunk_shape(len(active), active)


def _unit_kernel(
    tensor: torch.Tensor, _steps: int, _coefficients: tuple[float, float, float], _eps: float,
) -> torch.Tensor:
    """Deliberately substituted CPU unit boundary; not a native NS claim."""
    return (tensor * 0.05).bfloat16()


def _check_manual_step(opt: Any, patch: pytest.MonkeyPatch) -> None:
    """Independent full-route arithmetic oracle, never uses the active index."""
    expected_weights: list[torch.Tensor] = []
    expected_momenta: list[torch.Tensor] = []
    expected_refs = dict(opt._refs)
    expected_calls: list[tuple[torch.Tensor, int]] = []
    for spec in opt.routing.cmuon_specs:
        weight = spec.parameter.detach().clone()
        momentum = opt._momenta[spec.parameter].clone()
        if spec.parameter.grad is not None:
            grad = spec.parameter.grad.to(momentum.dtype)
            momentum.lerp_(grad, 1 - opt.cfg.momentum)
            nesterov = grad.lerp(momentum, opt.cfg.momentum)
            deltas: list[torch.Tensor] = []
            for ci, chunk in enumerate(_chunks(nesterov, spec)):
                expected_calls.append((chunk.clone(), opt.cfg.ns_steps_for_role(spec.role)))
                alpha = 0.2 * opt.cfg.lr * math.sqrt(max(chunk.shape))
                if opt.cfg.chunk_rescale_sqrt_n:
                    alpha *= math.sqrt(spec.chunk_count)
                deltas.append((-alpha * (chunk * 0.05).bfloat16()).contiguous())
                sig = float(chunk.float().square().mean().sqrt())
                key = (spec.name, ci)
                expected_refs[key] = max(sig, float(torch.tensor(
                    expected_refs[key] * opt.guard_cfg.reference_decay, dtype=torch.float32
                )))
            weight.mul_(1 - opt.cfg.lr * spec.weight_decay)
            weight.add_(torch.cat(deltas, dim=spec.chunk_dim))
        expected_weights.append(weight)
        expected_momenta.append(momentum)
    calls: list[tuple[torch.Tensor, int]] = []

    def kernel(tensor: torch.Tensor, steps: int, coefficients: tuple[float, float, float],
               eps: float) -> torch.Tensor:
        assert coefficients == opt.cfg.ns_coefficients and eps == opt.cfg.eps
        calls.append((tensor.clone(), steps))
        return _unit_kernel(tensor, steps, coefficients, eps)

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("single-rank healthy step must not use collectives or FP32 rescue")

    with patch.context() as p:
        p.setattr(guarded, "cmuon_zeroth_power", kernel)
        p.setattr(rescue, "cmuon_zeroth_power_bf16", kernel)
        p.setattr(rescue, "cmuon_zeroth_power_fp32", unexpected)
        p.setattr(guarded.dist, "all_reduce", unexpected)
        p.setattr(guarded.dist, "broadcast", unexpected)
        before_sr, before_obs = opt.sr_rng.calls, opt.observations
        opt.step()
    assert opt.sr_rng.calls == before_sr + 1
    assert opt.observations == before_obs + 1
    assert_equal(calls, expected_calls)
    for spec, weight, momentum in zip(opt.routing.cmuon_specs, expected_weights, expected_momenta):
        assert_equal(spec.parameter.detach(), weight)
        assert_equal(opt._momenta[spec.parameter], momentum)
    assert opt._refs == expected_refs
    assert opt.skip_total == 0  # missing gradients are not low-signal/zero inputs
    if isinstance(opt, rescue.HybridCMuonCanonicalNS4FP32Rescue):
        assert opt.fp32_attempts == opt.fp32_rescues == 0


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("missing", MASKS)
def test_masked_step_exact_manual_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, missing: tuple[int, ...],
) -> None:
    opt = make_optimizer(tmp_path, variant)
    route = opt.routing
    manifest = _serialized_clone(route.routing_manifest())
    seed_gradients(opt, missing)
    assert_index(opt)
    adam_before = [s.parameter.detach().clone() for s in route.adamw_specs]
    _check_manual_step(opt, monkeypatch)
    assert opt.routing is route
    assert_equal(route.routing_manifest(), manifest)
    state = opt.state_dict()
    assert set(state["cmuon"]["momenta"]) == {s.name for s in route.cmuon_specs}
    assert set(state["guard"]["references"]) == {
        f"{s.name}#chunk{ci}" for s in route.cmuon_specs for ci in range(s.chunk_count)
    }
    for spec, before in zip(route.adamw_specs, adam_before):
        assert not torch.equal(spec.parameter, before)
        assert float(opt.optimizer.state[spec.parameter]["step"]) == 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_consecutive_mask_changes_and_state_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    opt = make_optimizer(tmp_path, variant)
    route = opt.routing
    manifest = _serialized_clone(route.routing_manifest())
    for step, missing in enumerate(((0, 3), (1, 2, 4), tuple(range(6)), (), (5,))):
        seed_gradients(opt, missing, step=step)
        assert_index(opt)
        _check_manual_step(opt, monkeypatch)
        assert opt.routing is route
        assert_equal(route.routing_manifest(), manifest)
    restored = make_optimizer(tmp_path / "restore", variant)
    # Model weights are saved separately from optimizer state, as in training.
    with torch.no_grad():
        for src, dst in zip(opt.audit.specs, restored.audit.specs):
            dst.parameter.copy_(src.parameter)
    restored.load_state_dict(_serialized_clone(opt.state_dict()))
    assert_equal(restored.state_dict(), opt.state_dict())
    for instance in (opt, restored):
        seed_gradients(instance, (0, 2, 5), step=6)
        _check_manual_step(instance, monkeypatch)
    assert_equal(restored.state_dict(), opt.state_dict())
    assert_equal([s.parameter for s in restored.audit.specs], [s.parameter for s in opt.audit.specs])


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_parameters_none_does_not_inject_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    opt = make_optimizer(tmp_path, variant)
    seed_gradients(opt, tuple(range(6)), adam_none=True)
    before = snapshot(opt)
    _check_manual_step(opt, monkeypatch)
    assert_equal(before["weights"], [s.parameter for s in opt.audit.specs])
    assert_equal(before["state"]["optimizer"], opt.optimizer.state_dict())
    assert all(s.parameter.grad is None for s in opt.audit.specs)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("world", (1, 2, 4))
def test_index_owner_mapping_and_snapshot_are_step_local(tmp_path: Path, variant: str, world: int) -> None:
    opt = make_optimizer(tmp_path, variant, world=world)
    route = opt.routing
    manifest = _serialized_clone(route.routing_manifest())
    seed_gradients(opt, (0, 3))
    assert_index(opt)
    old = opt._active_inputs()
    keys = [entry.key for entry in old]
    seed_gradients(opt, (1, 4, 5))
    assert_index(opt)
    assert [opt._input_key(i, old) for i in range(len(old))] == keys
    assert [entry.key for entry in opt._active_inputs()] != keys
    assert opt.routing is route
    assert_equal(route.routing_manifest(), manifest)


def test_guard_failure_forensics_uses_active_key_and_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opt = make_optimizer(tmp_path, "guarded")
    seed_gradients(opt, (0, 1, 2, 3))  # first active input is the six-way fused matrix
    target = opt.routing.cmuon_specs[4]
    calls = 0

    def faulty(tensor: torch.Tensor, steps: int, coefficients: tuple[float, float, float],
               eps: float) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.full_like(tensor, float("nan")) if calls == 4 else _unit_kernel(
            tensor, steps, coefficients, eps
        )

    monkeypatch.setenv("SAKURAMOON_FORENSIC_DIR", str(tmp_path))
    monkeypatch.setattr(guarded, "cmuon_zeroth_power", faulty)
    before = snapshot(opt)
    with pytest.raises(CMuonSafetyError, match=f"{target.name}#chunk3"):
        opt.step()
    assert_equal(before["weights"], [s.parameter for s in opt.audit.specs])
    assert_equal(before["state"]["optimizer"], opt.optimizer.state_dict())
    assert opt.sr_rng.calls == 0 and opt.observations == 0
    records = json.loads((tmp_path / "guard-forensic-rank0.json").read_text())["records"]
    assert len(records) == 1
    assert (records[0]["fqn"], records[0]["chunk"], records[0]["shape"], records[0]["owner"]) == (
        target.name, 3, [2, 3], 0,
    )


def test_soft_rescue_log_routes_past_missing_multichunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opt = make_optimizer(tmp_path, "rescue")
    seed_gradients(opt, (0, 3, 4))
    calls = 0
    logs: list[str] = []
    opt.stats_logger = logs.append

    def faulty(tensor: torch.Tensor, steps: int, coefficients: tuple[float, float, float],
               eps: float) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.full_like(tensor, float("nan")) if calls == 3 else _unit_kernel(
            tensor, steps, coefficients, eps
        )

    def zero_rescue(tensor: torch.Tensor, *_args: Any) -> torch.Tensor:
        return torch.zeros_like(tensor)

    monkeypatch.setattr(rescue, "cmuon_zeroth_power_bf16", faulty)
    monkeypatch.setattr(rescue, "cmuon_zeroth_power_fp32", zero_rescue)
    last = opt.routing.cmuon_specs[-1]
    before = last.parameter.detach().clone()
    opt.step()
    expected = before.mul_(1 - opt.cfg.lr * last.weight_decay)
    assert_equal(last.parameter, expected)  # original zero rescue accepted; no rescaling
    assert opt.fp32_attempts == opt.fp32_rescues == opt.fp32_low_delta_rescues == 1
    assert opt.rescue_by_role == {last.role: 1}
    assert any(f"fqn={last.name}#chunk0 role={last.role}" in line for line in logs)


def _make_routed_failure_optimizer(root: Path, variant: str) -> Any:
    """Production FQN routing and optimizer classes; CPU unit adapters only."""
    module = nn.Module()
    for index, (name, shape, _count, dim, _role) in enumerate(_LAYOUT):
        # Production routing splits output rows, unlike the synthetic dim-1 case.
        rows, cols = shape if dim == 0 else (shape[1], shape[0])
        parts = name.split(".")
        parent = module
        for part in parts[:-2]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = parent.get_submodule(part)
        linear = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.fill_(0.5 + index / 8)
        parent.add_module(parts[-2], linear)
    module.register_parameter("bias", nn.Parameter(torch.ones(2)))
    route = route_cmuon_parameters(
        module, matrix_weight_decay=0.1, sensitive_weight_decay=0.0,
    )
    sr = _CPUSR()
    kwargs: dict[str, Any] = {
        "routing": route,
        "cfg": CMuonConfig(
            lr=0.01, momentum=0.5, momentum_dtype="bfloat16",
            ns_steps_by_role=resolve_ns_map({"ffn_in": 3, "adaln_shared": 5}, 4),
            chunk_rescale_sqrt_n=True,
        ),
        "adamw": _CPUAdamW(route.adamw_specs, sr), "sr_rng": sr,
        "guard_cfg": guarded.GuardedCanonicalGuardConfig(0.1, 0.9, 1e-6, 1e-10, 0, True),
        "rank": 0, "world_size": 1,
        "guard_bootstrap_refs": {spec.name: 0.125 for spec in route.cmuon_specs},
    }
    if variant == "guarded":
        return guarded.HybridCMuonGuardedCanonical(**kwargs)
    return rescue.HybridCMuonCanonicalNS4FP32Rescue(
        **kwargs, legacy_forensic_dir=root / "legacy",
        emergency_capsule_root=root / "capsules", hard_fail_artifact_root=root / "mirror",
        checkpoint_source="unit-checkpoint", run_id="cpu-unit-failure-attribution",
    )


@pytest.mark.parametrize("prefix", ("missing_start", "missing_multichunk", "active_multichunk"))
@pytest.mark.parametrize(("variant", "bf16_mode", "fp32_mode"), (
    ("guarded", "nonfinite", None),
    ("guarded", "above_ceiling", None),
    ("rescue", "nonfinite", "nonfinite"),
    ("rescue", "nonfinite", "above_ceiling"),
    ("rescue", "above_ceiling", "nonfinite"),
    ("rescue", "above_ceiling", "above_ceiling"),
))
def test_cpu_unit_hard_failure_attribution_after_missing_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str,
    variant: str, bf16_mode: str, fp32_mode: str | None,
) -> None:
    """Injected CPU kernels exercise real routing/step, not native acceptance."""
    opt = _make_routed_failure_optimizer(tmp_path, variant)
    route = opt.routing
    manifest = _serialized_clone(route.routing_manifest())
    seed_gradients(opt, ())
    _check_manual_step(opt, monkeypatch)  # populate AdamW state, refs, SR and observations
    target = next(spec for spec in route.cmuon_specs if spec.role == "adaln_shared")
    chunk_idx = 3 if prefix == "missing_start" else 4
    if prefix == "missing_start":
        missing = tuple(i for i, spec in enumerate(route.cmuon_specs) if spec is not target)
    else:
        missing = tuple(i for i, spec in enumerate(route.cmuon_specs) if spec.role in (
            ("attention_k", "ffn_in") if prefix == "missing_multichunk" else ("attention_k",)
        ))
    seed_gradients(opt, missing, step=1)
    before = snapshot(opt)
    gradients = _serialized_clone([spec.parameter.grad for spec in opt.audit.specs])
    # Independent Nesterov/chunk oracle: never uses the production active index.
    inputs: list[tuple[CMuonChunkSpec, int, torch.Tensor]] = []
    momenta: list[torch.Tensor] = []
    for spec in route.cmuon_specs:
        momentum = opt._momenta[spec.parameter].clone()
        if spec.parameter.grad is not None:
            grad = spec.parameter.grad.to(momentum.dtype)
            momentum.lerp_(grad, 1 - opt.cfg.momentum)
            nesterov = grad.lerp(momentum, opt.cfg.momentum)
            inputs.extend((spec, ci, chunk.clone()) for ci, chunk in enumerate(_chunks(nesterov, spec)))
        momenta.append(momentum)
    target_idx = next(i for i, (spec, ci, _chunk) in enumerate(inputs)
                      if spec is target and ci == chunk_idx)
    expected_input = inputs[target_idx][2]
    full_keys = [(spec.name, ci) for spec in route.cmuon_specs for ci in range(spec.chunk_count)]
    # This would be misattributed by the old full-routing flat-index helpers.
    assert full_keys[target_idx] != (target.name, chunk_idx)
    assert route.cmuon_specs[0].parameter.grad is None
    assert tuple(expected_input.shape) == (2, 3)
    assert_index(opt)
    bf16_calls: list[tuple[torch.Tensor, int]] = []
    fp32_calls: list[tuple[torch.Tensor, int]] = []
    bf16_outputs: list[torch.Tensor] = []
    fp32_outputs: list[torch.Tensor] = []
    logs: list[str] = []
    opt.stats_logger = logs.append
    if variant == "rescue":
        opt.note_forensic_update(last_successful_update=77, attempted_update=78)

    def bad_output(tensor: torch.Tensor, mode: str) -> torch.Tensor:
        # Inf also exceeds the ceiling: NONFINITE must retain priority.
        return torch.full_like(tensor, float("inf") if mode == "nonfinite" else 100.0)

    def bf16_kernel(tensor: torch.Tensor, steps: int, coefficients: tuple[float, float, float],
                    eps: float) -> torch.Tensor:
        assert tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16
        assert coefficients == opt.cfg.ns_coefficients and eps == opt.cfg.eps
        bf16_calls.append((tensor.clone(), steps))
        output = bad_output(tensor, bf16_mode) if len(bf16_calls) - 1 == target_idx else _unit_kernel(
            tensor, steps, coefficients, eps
        )
        if len(bf16_calls) - 1 == target_idx:
            bf16_outputs.append(output.clone())
        return output

    def fp32_kernel(tensor: torch.Tensor, steps: int, coefficients: tuple[float, float, float],
                    eps: float) -> torch.Tensor:
        assert variant == "rescue" and fp32_mode is not None
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        assert coefficients == opt.cfg.ns_coefficients and eps == opt.cfg.eps
        fp32_calls.append((tensor.clone(), steps))
        output = bad_output(tensor, fp32_mode)
        fp32_outputs.append(output.clone())
        return output

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("hard-failed CPU unit step reached commit or a collective")

    monkeypatch.setenv("SAKURAMOON_FORENSIC_DIR", str(tmp_path / "legacy"))
    monkeypatch.setattr(guarded, "cmuon_zeroth_power", bf16_kernel)
    monkeypatch.setattr(rescue, "cmuon_zeroth_power_bf16", bf16_kernel)
    monkeypatch.setattr(rescue, "cmuon_zeroth_power_fp32", fp32_kernel)
    monkeypatch.setattr(guarded.dist, "all_reduce", forbidden)
    monkeypatch.setattr(guarded.dist, "broadcast", forbidden)
    monkeypatch.setattr(opt.sr_rng, "run_step", forbidden)
    monkeypatch.setattr(opt.optimizer, "step", forbidden)
    failure = "nonfinite NS delta" if bf16_mode == "nonfinite" else "delta_rms above 10x(0.2*lr) ceiling"
    failure_message = f"{failure}: {target.name}#chunk{chunk_idx}"
    header = "guarded canonical safety violation: " if variant == "guarded" else (
        "fp32-rescue safety violation (BF16 and FP32 both failed): "
    )
    with pytest.raises(CMuonSafetyError) as error:
        opt.step()
    assert type(error.value) is CMuonSafetyError
    assert str(error.value) == header + failure_message
    assert_equal(bf16_calls, [(chunk, opt.cfg.ns_steps_for_role(spec.role)) for spec, _ci, chunk in inputs])
    assert len(bf16_outputs) == 1
    # Fail-closed promises no parameter/AdamW/SR/reference/observation commit,
    # NOT momentum rollback: preparation has already updated every present EMA.
    assert_equal(before["weights"], [spec.parameter for spec in opt.audit.specs])
    assert_equal(before["state"]["optimizer"], opt.optimizer.state_dict())
    assert_equal(before["state"]["sr_rng"], opt.sr_rng.state_dict())
    assert_equal(before["state"]["guard"]["references"], opt.state_dict()["guard"]["references"])
    assert opt.observations == before["counters"]["observations"] == 1
    assert opt.skip_total == 0 and opt.skip_by_role == opt.skip_by_fqn == {}
    for spec, momentum in zip(route.cmuon_specs, momenta):
        assert_equal(opt._momenta[spec.parameter], momentum)
    assert_equal(gradients, [spec.parameter.grad for spec in opt.audit.specs])
    assert opt.routing is route
    assert_equal(route.routing_manifest(), manifest)
    payload = json.loads((tmp_path / "legacy" / "guard-forensic-rank0.json").read_text())
    assert payload["observations"] == 1 and len(payload["records"]) == 1
    record = payload["records"][0]
    identity = {
        "fqn": target.name, "chunk": chunk_idx, "shape": [2, 3],
        "numel": 6, "owner": 0, "this_rank": 0,
    }
    assert {key: record[key] for key in identity} == identity
    assert record["failure"] == failure
    input_rms = float(expected_input.float().square().mean().sqrt())
    assert record["u_t_rms"] == input_rms
    assert record["lr"] == opt.cfg.lr
    assert record["target_delta_rms"] == 0.2 * opt.cfg.lr
    assert record["ceiling"] == 10 * (0.2 * opt.cfg.lr)
    alpha = 0.2 * opt.cfg.lr * math.sqrt(3) * math.sqrt(target.chunk_count)
    bf16_rms = float(((-alpha) * bf16_outputs[0]).float().square().mean().sqrt())
    if variant == "guarded":
        assert fp32_calls == fp32_outputs == []
        assert record["delta_rms"] == (None if bf16_mode == "nonfinite" else bf16_rms)
        assert not (tmp_path / "capsules").exists()
        assert not (tmp_path / "mirror").exists()
        assert any(line.startswith("[guard-forensic]") for line in logs)
        return

    assert_equal(fp32_calls, [(expected_input.float(), opt.cfg.ns_steps_for_role(target.role))])
    assert len(fp32_outputs) == 1
    fp32_rms = float(((-alpha) * fp32_outputs[0]).square().mean().sqrt())
    finite32 = fp32_mode != "nonfinite"
    assert opt.bf16_attempts == before["counters"]["bf16_attempts"] + len(inputs)
    assert opt.bf16_safety_failures == opt.fp32_attempts == opt.fp32_rescue_failures == 1
    assert opt.fp32_rescues == opt.fp32_low_delta_rescues == 0
    assert opt.rescue_by_role == opt.fp32_low_delta_by_role == {}
    assert opt._steps_this_process == 2
    assert record["bf16_delta_rms"] == record["delta_rms"] == bf16_rms
    assert record["fp32_delta_rms"] == fp32_rms
    assert record["fp32_finite"] is finite32
    assert record["fp32_failure_reason"] == fp32_mode
    assert record["fp32_rescue_floor"] == 0.05 * (0.2 * opt.cfg.lr)
    assert record["fp32_ceiling"] == record["ceiling"]
    assert (record["fp32_attempts"], record["fp32_rescues"], record["fp32_rescue_failures"]) == (1, 0, 1)
    event_name = f"obs-1-rank0-{target.name.replace('.', '_')}-chunk{chunk_idx}"
    events = list((tmp_path / "capsules").iterdir())
    assert len(events) == 1 and events[0].name == event_name
    event = events[0]
    metadata = json.loads((event / "metadata.json").read_text())
    expected_metadata = {
        **identity, "schema": "sakuramoon.cmuon_minimal_hardfail_capsule.v1",
        "role": target.role, "dtype": "torch.bfloat16", "contiguous": True,
        "world_size": 1, "observations": 1, "process_steps": 2,
        "run_id": "cpu-unit-failure-attribution", "checkpoint_source": "unit-checkpoint",
        "last_successful_update": 77, "attempted_update": 78,
        "input_rms": input_rms, "u_t_rms": input_rms, "input_finite": True,
        "input_frobenius_norm": float(cast(Any, expected_input.float()).norm()),
        "input_abs_max": float(expected_input.float().abs().max()),
        "alpha": alpha, "ns_steps": opt.cfg.ns_steps_for_role(target.role),
        "ns_coefficients": list(opt.cfg.ns_coefficients), "eps": opt.cfg.eps,
        "lr": opt.cfg.lr, "target_delta_rms": record["target_delta_rms"],
        "ceiling": record["ceiling"], "rescue_floor": record["fp32_rescue_floor"],
        "bf16_delta_rms": None if bf16_mode == "nonfinite" else bf16_rms,
        "original_fp32_delta_rms": fp32_rms if finite32 else None,
        "fp32_delta_rms": fp32_rms if finite32 else None,
        "original_fp32_finite": finite32, "fp32_finite": finite32,
        "fp32_failure_reason": fp32_mode, "bf16_failure": failure,
        "failure_message": failure_message, "local_artifact_path": str(event),
        "shared_mirror_root": str(tmp_path / "mirror"),
    }
    assert {key: metadata[key] for key in expected_metadata} == expected_metadata
    assert isinstance(metadata["hostname"], str) and metadata["hostname"]
    assert isinstance(metadata["pid"], int) and metadata["pid"] > 0
    assert metadata["wall_clock_unix_seconds"] > 0
    if metadata["tensor_format"] == "safetensors":
        from safetensors.torch import load

        tensor_path = event / "input.safetensors"
        saved = load(tensor_path.read_bytes())["input"]
    else:
        assert metadata["tensor_format"] == "torch_pt"
        tensor_path = event / "input.pt"
        saved = torch.load(tensor_path, weights_only=True, map_location="cpu")["input"]
    assert metadata["tensor_sha256"] == hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    assert_equal(saved, expected_input)
    assert_equal(saved, bf16_calls[target_idx][0])
    assert_equal(saved.float(), fp32_calls[0][0])
    mirror = json.loads((event / "mirror.json").read_text())
    assert mirror["status"] == "ok" and mirror["error"] is None
    mirrored = list((tmp_path / "mirror").iterdir())
    assert len(mirrored) == 1 and mirrored[0].name == event_name
    assert (mirrored[0] / tensor_path.name).read_bytes() == tensor_path.read_bytes()
    assert (mirrored[0] / "metadata.json").read_bytes() == (event / "metadata.json").read_bytes()
    assert any(line == f"[guard-forensic] {failure_message}" for line in logs)
    assert any(line == f"[hard-fail-capsule] {event_name} local=durable mirror=ok" for line in logs)
