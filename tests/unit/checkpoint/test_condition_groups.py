# pyright: reportPrivateUsage=false
"""CPU checkpoint boundaries for opt-in contiguous AdamW condition runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn
from torchao.optim.subclass_8bit import (  # pyright: ignore[reportMissingTypeStubs]
    OptimState8bit,
)

import sakuramoon.checkpoint.load as loader
from sakuramoon.checkpoint.save import _hybrid_optimizer_schema, _optimizer_schema
from sakuramoon.checkpoint.schema import CheckpointError, CheckpointIdentity
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.cmuon import (
    CMuonChunkSpec,
    CMuonConfig,
    CMuonRouting,
    HybridCMuon,
)
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG

NAMES = ("a.weight", "b.weight", "c.weight", "a.bias", "b.bias", "c.bias")
SELECTED = frozenset({"b.weight", "b.bias"})


def _build(
    selected: frozenset[str] | None = None,
    *,
    names: tuple[str, ...] = NAMES,
    quantized: bool = False,
) -> HybridCMuon:
    specs = tuple(
        ParameterSpec(
            name,
            nn.Parameter(
                torch.zeros((64, 64), dtype=torch.bfloat16)
                if quantized and name.endswith("weight")
                else torch.zeros(4)
            ),
            "matrix_decay" if name.endswith("weight") else "sensitive_no_decay",
            0.0,
        )
        for name in names
    )
    audit = ParameterAudit(specs)
    groups: list[dict[str, Any]] = []
    for base, subset in (
        ("matrix_decay", audit.decay),
        ("sensitive_no_decay", audit.sensitive),
    ):
        runs: list[dict[str, Any]] = []
        for spec in subset:
            tag = selected is not None and spec.name in selected
            if not runs or (
                selected is not None and runs[-1]["condition_lr_group"] != tag
            ):
                group: dict[str, Any] = {
                    "params": [],
                    "param_names": [],
                    "group_name": base,
                    "weight_decay": 0.0,
                }
                if selected is not None:
                    group["condition_lr_group"] = tag
                runs.append(group)
            runs[-1]["params"].append(spec.parameter)
            runs[-1]["param_names"].append(spec.name)
        if not runs:
            runs = [
                {
                    "params": [],
                    "param_names": [],
                    "group_name": base,
                    "weight_decay": 0.0,
                }
            ]
            if selected is not None:
                runs[0]["condition_lr_group"] = False
        groups.extend(runs)
    inner = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
    rng = StochasticRoundingRNG(
        torch.device("cuda:0"), torch.arange(16, dtype=torch.uint8)
    )
    adamw = IsolatedAdamW8bit(inner, audit, rng)
    cmuon_parameter = nn.Parameter(torch.zeros(4, 4))
    cmuon_spec = CMuonChunkSpec(
        "cmuon.weight", cmuon_parameter, 0.0, 1, 0, ("q",), "attention_q"
    )
    full = ParameterAudit(
        (*specs, ParameterSpec("cmuon.weight", cmuon_parameter, "matrix_decay", 0.0))
    )
    return HybridCMuon(
        routing=CMuonRouting(full, (cmuon_spec,), specs),
        cfg=CMuonConfig(lr=1e-4),
        adamw=adamw,
        sr_rng=rng,
    )


def _seed(optimizer: HybridCMuon, *, quantized: bool = False) -> None:
    for index, spec in enumerate(optimizer.adamw.audit.specs):
        moments: list[torch.Tensor] = []
        for signed in (True, False):
            if quantized and spec.parameter.numel() >= 4096:
                moment = cast(Any, OptimState8bit).zeros(
                    spec.parameter.shape, signed=signed, device="cpu"
                )
                moment.codes.fill_(index + 1)
                moment.scale.fill_(index + 0.25)
                moment.qmap.copy_(torch.linspace(-1 if signed else 0, 1, 256))
                moments.append(moment)
            else:
                moments.append(torch.full_like(spec.parameter, index + 0.25))
        optimizer.optimizer.state[spec.parameter] = {
            "step": torch.tensor(float(index % 3 + 1)),
            "exp_avg": moments[0],
            "exp_avg_sq": moments[1],
        }
    for momentum in optimizer._momenta.values():
        momentum.fill_(0.75)


def _by_name(document: dict[str, Any]) -> dict[str, Any]:
    return {
        name: document["state"][pid]
        for group in document["param_groups"]
        for name, pid in zip(group["param_names"], group["params"], strict=True)
        if pid in document["state"]
    }


def _assert_moments(before: dict[str, Any], after: dict[str, Any]) -> None:
    for name, state in _by_name(before).items():
        restored = _by_name(after)[name]
        for key, value in state.items():
            other = restored[key]
            assert type(value) is type(other)
            assert value.dtype == other.dtype
            if isinstance(value, OptimState8bit):
                for field in ("codes", "qmap", "scale"):
                    left, right = getattr(value, field), getattr(other, field)
                    assert left.dtype == right.dtype
                    assert torch.equal(left, right)
                assert value.signed is other.signed
                assert value.block_size == other.block_size
            else:
                assert torch.equal(value, other)


@pytest.mark.parametrize("source_selected", [None, SELECTED])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_hybrid_restore_preserves_exact_state_and_current_rates(
    tmp_path: Path,
    source_selected: frozenset[str] | None,
    quantized: bool,
) -> None:
    old = _build(source_selected, quantized=quantized)
    _seed(old, quantized=quantized)
    before = cast(dict[str, Any], old.state_dict())
    schema = _hybrid_optimizer_schema(old)
    torch.save(before, tmp_path / "optimizer.pt")
    current = _build(frozenset({"a.weight", "c.bias"}), quantized=quantized)
    rates: list[float] = []
    for index, group in enumerate(current.optimizer.param_groups):
        group["lr"] = (index + 1) * 0.002
        group["weight_decay"] = (index + 1) * 0.01
        rates.append(group["lr"])
    loader._load_hybrid_state_exact(
        tmp_path, schema, current, old.sr_rng.state_dict(), 3
    )
    after = cast(dict[str, Any], current.state_dict())
    _assert_moments(before["optimizer"], after["optimizer"])
    assert [group["lr"] for group in current.optimizer.param_groups] == rates
    assert [group["weight_decay"] for group in current.optimizer.param_groups] == [
        (index + 1) * 0.01 for index in range(len(rates))
    ]
    assert after["routing"] == before["routing"]
    for name, momentum in before["cmuon"]["momenta"].items():
        assert torch.equal(momentum, after["cmuon"]["momenta"][name])
    assert torch.equal(before["sr_rng"]["state"], after["sr_rng"]["state"])
    assert [
        name
        for group in after["optimizer"]["param_groups"]
        for name in group["param_names"]
    ] == list(NAMES)


@pytest.mark.parametrize("source_selected", [None, SELECTED])
@pytest.mark.parametrize("updates", [0, 3])
def test_pure_adamw_schema_and_restore(
    source_selected: frozenset[str] | None, updates: int
) -> None:
    source = _build(source_selected)
    if updates:
        _seed(source)
    old, current = source.adamw, _build(SELECTED).adamw
    schema = _optimizer_schema(old)
    loader._validate_optimizer_schema(schema, current, cast(CheckpointIdentity, None))
    saved = cast(dict[str, object], old.optimizer.state_dict())
    prepared = loader._prepare_condition_optimizer_state(schema, saved, current)
    assert prepared is not None
    loader._validate_optimizer_state(prepared, current, updates)
    current.load_state_dict({"optimizer": prepared, "sr_rng": old.sr_rng.state_dict()})
    assert len(current.optimizer.param_groups) == 6
    _assert_moments(saved, current.optimizer.state_dict())
    assert torch.equal(old.sr_rng.state, current.sr_rng.state)


@pytest.mark.parametrize(
    "case",
    [
        "count",
        "duplicate_name",
        "duplicate_id",
        "unknown_id",
        "unknown_name",
        "order",
        "schema",
        "base_name",
        "base_order",
        "betas",
        "eps",
        "amsgrad",
        "extra_field",
        "missing_field",
        "bad_tag",
        "partial_tag",
        "all_false_tag",
    ],
)
def test_native_hybrid_rejects_corrupt_saved_groups(tmp_path: Path, case: str) -> None:
    old = _build(SELECTED)
    _seed(old)
    outer = cast(dict[str, Any], old.state_dict())
    schema = _hybrid_optimizer_schema(old)
    groups = outer["optimizer"]["param_groups"]
    if case == "count":
        groups[0]["params"].append(99)
    elif case == "duplicate_name":
        groups[1]["param_names"][0] = groups[0]["param_names"][0]
    elif case == "duplicate_id":
        groups[1]["params"][0] = groups[0]["params"][0]
    elif case == "unknown_id":
        outer["optimizer"]["state"][99] = outer["optimizer"]["state"][0]
    elif case == "unknown_name":
        groups[0]["param_names"][0] = "rogue.weight"
    elif case == "order":
        groups[0]["param_names"], groups[1]["param_names"] = (
            groups[1]["param_names"],
            groups[0]["param_names"],
        )
    elif case == "schema":
        cast(list[dict[str, Any]], schema["groups"])[0]["param_names"] = ["wrong"]
    elif case == "base_name":
        groups[0]["group_name"] = "condition"
    elif case == "base_order":
        groups[0]["group_name"] = "sensitive_no_decay"
    elif case == "extra_field":
        groups[0]["rogue"] = True
    elif case == "missing_field":
        del groups[0]["eps"]
    elif case == "bad_tag":
        groups[0]["condition_lr_group"] = 1
    elif case == "partial_tag":
        del groups[0]["condition_lr_group"]
    elif case == "all_false_tag":
        for group in groups:
            group["condition_lr_group"] = False
    else:
        groups[0][case] = {"betas": (0.8, 0.95), "eps": 1e-6, "amsgrad": True}[case]
    torch.save(outer, tmp_path / "optimizer.pt")
    current = _build(SELECTED)
    with pytest.raises(CheckpointError):
        loader._load_hybrid_state_exact(
            tmp_path, schema, current, old.sr_rng.state_dict(), 3
        )
    assert not current.optimizer.state


def test_growth_and_declared_drop_remap_preserves_shared_fqns() -> None:
    dropped = "irepa_alignment.projector.bias"
    old = _build(SELECTED, names=(*NAMES, dropped))
    _seed(old)
    current = _build(
        SELECTED, names=("a.weight", "b.weight", "b_new.weight", *NAMES[2:])
    )
    schema = _optimizer_schema(old.adamw)
    before = old.optimizer.state_dict()
    after = loader._prepare_condition_optimizer_state(
        schema,
        before,
        current.adamw,
        new_fqns=frozenset({"b_new.weight"}),
        dropped_fqns=frozenset({dropped}),
    )
    assert after is not None
    loader._validate_optimizer_state(after, current.adamw, 3)
    expected = _by_name(before)
    del expected[dropped]
    assert set(_by_name(after)) == set(expected)
    assert "b_new.weight" not in _by_name(after)
    for name, state in expected.items():
        for key, value in state.items():
            assert value is _by_name(after)[name][key]


@pytest.mark.parametrize("tag", [None, 1, False])
def test_schema_relaxation_requires_complete_live_opt_in(tag: object) -> None:
    old, current = _build(), _build(SELECTED)
    if tag is None:
        del current.optimizer.param_groups[0]["condition_lr_group"]
    else:
        for group in current.optimizer.param_groups:
            group["condition_lr_group"] = tag
    with pytest.raises(CheckpointError):
        loader._validate_optimizer_schema(
            _optimizer_schema(old.adamw), current.adamw, cast(CheckpointIdentity, None)
        )


def test_legacy_schema_stays_exact_and_downgrade_is_rejected() -> None:
    old, current = _build(SELECTED), _build()
    schema = _optimizer_schema(old.adamw)
    with pytest.raises(CheckpointError, match="group count"):
        loader._validate_optimizer_schema(
            schema, current.adamw, cast(CheckpointIdentity, None)
        )
    with pytest.raises(CheckpointError, match="legacy"):
        loader._prepare_condition_optimizer_state(
            schema, cast(dict[str, object], old.optimizer.state_dict()), current.adamw
        )
    legacy = cast(dict[str, object], current.optimizer.state_dict())
    assert (
        loader._prepare_condition_optimizer_state(
            _optimizer_schema(current.adamw), legacy, current.adamw
        )
        is None
    )


def test_condition_adamw_to_hybrid_fails_before_loading(tmp_path: Path) -> None:
    current = _build(SELECTED)
    with pytest.raises(CheckpointError, match="transition.*unsupported"):
        loader._load_adamw_transition_state(
            tmp_path, {}, current, current.sr_rng.state_dict(), 3
        )
