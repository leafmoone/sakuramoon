# pyright: reportPrivateUsage=false
"""CPU tests of real grouping/scheduling/restoration with CUDA execution replaced."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_cmuon import _MockComposite, _MockDiT
from test_learning_rate_multipliers import BASE, NAMES, _config, _cpu_adamw, _seed_grads
from torch import nn

import sakuramoon.optim.cmuon as core
import sakuramoon.optim.fp32_rescue as rescue
import sakuramoon.optim.guarded_canonical as guarded
from sakuramoon.checkpoint.load import _load_hybrid_state_exact
from sakuramoon.checkpoint.save import _hybrid_optimizer_schema
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.optim.condition import (
    condition_learning_rate,
    split_condition_lr_groups,
)
from sakuramoon.train import production
from sakuramoon.train.runtime import _optimizer_learning_rates
from sakuramoon.train.step import TrainableComposite


def _model() -> nn.Module:
    model = _MockComposite(_MockDiT(16, 32, 1))
    model.condition_tokens = nn.Linear(8, 8, dtype=torch.bfloat16)
    model.condition_tokens.bias = nn.Parameter(torch.zeros(8, dtype=torch.float32))
    return model


def _configured(name: str, multiplier: float | None, **overrides: Any) -> RuntimeConfig:
    source = name if name in NAMES else "hybrid_cmuon"
    document = _config(source, cmuon=1.3, adamw=0.73).model_dump(mode="python", exclude_computed_fields=True)
    document["optimizer"]["name"] = name
    if multiplier is not None:
        document["optimizer"]["condition"] = dict(
            parameter_prefixes=["condition_tokens"], independent_lr=True,
            lr_multiplier=multiplier, **overrides,
        )
    return RuntimeConfig.model_validate(document)


@pytest.fixture(autouse=True)
def cpu_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (core, guarded, rescue):
        monkeypatch.setattr(module, "_build_adamw8bit_for_specs", _cpu_adamw)
    def pure(model: nn.Module, **kwargs: Any) -> Any:
        audit = core.route_cmuon_parameters(model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0).full_audit
        return _cpu_adamw(audit.specs, **kwargs)
    monkeypatch.setattr(production, "build_adamw8bit", pure)


def _build(config: RuntimeConfig) -> tuple[nn.Module, Any]:
    model = _model()
    return model, production._build_optimizer(config, cast(TrainableComposite, model))


@pytest.mark.parametrize("name", (*NAMES, "torchao_adamw8bit"))
@pytest.mark.parametrize("multiplier", [0.17, 1.0, 2.3])
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_common_schedule_independent_condition_rate(name: str, multiplier: float, tensor_lr: bool) -> None:
    config = _configured(name, multiplier)
    model, opt = _build(config)
    flat = [n for group in opt.optimizer.param_groups for n in group["param_names"]]
    _, legacy = _build(_configured(name, None))
    assert flat == [n for group in legacy.optimizer.param_groups for n in group["param_names"]]
    assert all("condition_lr_group" not in group for group in legacy.optimizer.param_groups)
    assert {n for group in opt.optimizer.param_groups if group["condition_lr_group"] for n in group["param_names"]} == {"condition_tokens.weight", "condition_tokens.bias"}
    if tensor_lr:
        for group in opt.optimizer.param_groups:
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
    schedule = production._SuccessfulUpdateLrScheduler(config, opt, production._config_learning_rate, restored_successful_update=0, fresh=True)
    for update in range(1, 7):
        base = production._config_learning_rate(config, update)
        assert production._optimizer_learning_rate_matches(opt, base, config)
        rate = condition_learning_rate(opt)
        assert rate is not None
        assert math.isclose(rate, base * 0.73 * multiplier, rel_tol=1e-6)
        rates = _optimizer_learning_rates(opt)
        assert math.isclose(rates[2], base * 0.73, rel_tol=1e-6)
        assert rates == _optimizer_learning_rates(opt)  # retry does not advance
        _seed_grads(model, update)
        opt.step()
        schedule(update)


def _by_name(state: dict[str, Any]) -> dict[str, Any]:
    return {name: state["state"].get(pid) for g in state["param_groups"] for pid, name in zip(g["params"],g["param_names"],strict=True)}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("old_multiplier", [None, 0.37])
def test_old_to_new_and_new_to_new_restore(name: str, old_multiplier: float | None, tmp_path: Path) -> None:
    model, old = _build(_configured(name, old_multiplier))
    for seed in (1, 2):
        _seed_grads(model, seed)
        old.step()
    state = copy.deepcopy(old.state_dict())
    schema = _hybrid_optimizer_schema(old)
    torch.save(state, tmp_path / "optimizer.pt")
    (tmp_path / "optimizer_schema.json").write_text(json.dumps(schema),encoding="utf8")
    current_config = _configured(name, 1.91)
    _, current = _build(current_config)
    base = production._config_learning_rate(current_config, 3)
    production._set_optimizer_learning_rate(current, base, current_config)
    _load_hybrid_state_exact(tmp_path, schema, current, old.sr_rng.state_dict(), successful_updates=2)
    production._SuccessfulUpdateLrScheduler(current_config,current,production._config_learning_rate,restored_successful_update=2,fresh=False)
    restored = current.state_dict()
    before, after = _by_name(state["optimizer"]), _by_name(restored["optimizer"])
    assert set(before) == set(after)
    for fqn, entry in before.items():
        assert set(entry) == set(after[fqn])
        for field,value in entry.items():
            assert torch.equal(value, after[fqn][field]),(fqn,field)
    for fqn, momentum in state["cmuon"]["momenta"].items():
        assert torch.equal(momentum, restored["cmuon"]["momenta"][fqn])
    assert torch.equal(state["sr_rng"]["state"],restored["sr_rng"]["state"])
    assert state["routing"] == restored["routing"]
    if "guard" in state:
        for field, value in state["guard"].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored["guard"][field])
            else:
                assert value == restored["guard"][field]
    rate = condition_learning_rate(current)
    assert rate is not None
    assert math.isclose(rate, base*0.73*1.91,rel_tol=1e-6)


@pytest.mark.parametrize("explicit_prefixes", [None, ("condition_tokens", "dit.input_projection"), ("condition_tokens", "dit.conditioner.final_projection")])
def test_contiguous_runs_do_not_reorder_shared_draws_or_alias_lr(explicit_prefixes: tuple[str, ...] | None) -> None:
    config = _configured("hybrid_cmuon",1.0)
    _, optimizer = _build(_configured("hybrid_cmuon",None))
    # Interleave synthetic membership to exercise more than head-only condition params.
    names = [n for g in optimizer.optimizer.param_groups for n in g["param_names"]]
    prefix = names[len(names)//2]
    assert config.optimizer.condition is not None
    policy = config.optimizer.condition.model_copy(update={"parameter_prefixes": explicit_prefixes or (prefix,)})
    for group in optimizer.optimizer.param_groups:
        group["lr"] = torch.tensor(BASE,dtype=torch.float32)
    split_condition_lr_groups(optimizer,policy)
    assert [n for g in optimizer.optimizer.param_groups for n in g["param_names"]] == names
    assert len({g["lr"].data_ptr() for g in optimizer.optimizer.param_groups}) == len(optimizer.optimizer.param_groups)
    for index, group in enumerate(optimizer.optimizer.param_groups):
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter) if index % 2 == 0 else None
    expected = [n for g in optimizer.optimizer.param_groups for n, p in zip(g["param_names"], g["params"], strict=True) if p.dtype == torch.bfloat16 and p.grad is not None]
    specs = {s.name: s for s in optimizer.audit.specs}
    assert expected == [n for n in names if specs[n].parameter.dtype == torch.bfloat16 and specs[n].parameter.grad is not None]


def test_missing_selector_or_cmuon_selection_fails_before_training() -> None:
    for prefix in ("missing", "dit.blocks"):
        config = _configured("hybrid_cmuon",0.5)
        document=config.model_dump(mode="python",exclude_computed_fields=True)
        document["optimizer"]["condition"]["parameter_prefixes"]=[prefix]
        with pytest.raises(ValueError,match="prefixes|AdamW"):
            _build(RuntimeConfig.model_validate(document))


def test_default_does_not_change_group_state_or_rng() -> None:
    _, opt = _build(_configured("hybrid_cmuon",None))
    before = copy.deepcopy(opt.state_dict())
    groups=opt.optimizer.param_groups
    split_condition_lr_groups(opt,None)
    assert opt.optimizer.param_groups is groups
    assert before["optimizer"]["param_groups"] == opt.state_dict()["optimizer"]["param_groups"]
    assert torch.equal(before["sr_rng"]["state"],opt.sr_rng.state)


@pytest.mark.parametrize("condition_multiplier", [1.0, 0.27])
def test_lr_change_does_not_change_rest_update_or_sr_draw_order(condition_multiplier: float) -> None:
    torch.manual_seed(12)  # pyright: ignore[reportUnknownMemberType]
    model, legacy = _build(_configured("hybrid_cmuon",None))
    torch.manual_seed(12)  # pyright: ignore[reportUnknownMemberType]
    split_model, split = _build(_configured("hybrid_cmuon",condition_multiplier))
    draws: list[list[str]]=[]
    for opt in (legacy,split):
        # A CPU spy models a draw for every visited param; actual TorchAO traversal
        # is checked by the preserved flattened order, not claimed GPU bit parity.
        original=opt.optimizer.step
        def step(original: Any=original, opt: Any=opt) -> Any:
            draws.append([n for g in opt.optimizer.param_groups for n,p in zip(g["param_names"],g["params"],strict=True) if p.grad is not None])
            return original()
        opt.optimizer.step=step
    for seed in (7,8):
        _seed_grads(model,seed);_seed_grads(split_model,seed)
        legacy.step();split.step()
        assert draws[-1]==draws[-2]
    for name,p in model.named_parameters():
        other=dict(split_model.named_parameters())[name]
        if condition_multiplier == 1.0 or not name.startswith("condition_tokens."):
            assert torch.equal(p,other),name
        else:
            assert not torch.equal(p,other),name
