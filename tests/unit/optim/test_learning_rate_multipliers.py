# pyright: reportPrivateUsage=false
"""CPU boundary tests; only CUDA-only AdamW construction/SR execution is replaced."""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest
import torch
from test_cmuon import _MockComposite, _MockDiT
from torch import nn

import sakuramoon.optim.cmuon as core
import sakuramoon.optim.fp32_rescue as rescue
import sakuramoon.optim.guarded_canonical as guarded
from sakuramoon.checkpoint.load import _load_hybrid_state_exact
from sakuramoon.checkpoint.save import _hybrid_optimizer_schema
from sakuramoon.config.load import load_config
from sakuramoon.config.schema import OptimizerConfig, RuntimeConfig
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG
from sakuramoon.train.production import (
    _build_optimizer,
    _config_learning_rate,
    _optimizer_learning_rate_matches,
    _set_optimizer_learning_rate,
    _SuccessfulUpdateLrScheduler,
)
from sakuramoon.train.runtime import _optimizer_learning_rates
from sakuramoon.train.step import TrainableComposite

ROOT = Path(__file__).parents[3]
BASE = 1.5e-4
SECRETS = {
    "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
    "WANDB_API_KEY": "synthetic-wandb-secret",
}
NAMES = (
    "hybrid_cmuon",
    "hybrid_cmuon_guarded_canonical",
    "hybrid_cmuon_canonical_ns4_fp32_rescue",
)


class _AdamWGroup(TypedDict):
    param_names: list[str]
    lr: float


class _AdamWState(TypedDict):
    param_groups: list[_AdamWGroup]
    state: dict[int, dict[str, torch.Tensor]]


class _CMuonState(TypedDict):
    momenta: dict[str, torch.Tensor]


class _SRState(TypedDict):
    state: torch.Tensor


class _CpuSR(StochasticRoundingRNG):
    def run_step(self, step: object) -> object:
        assert callable(step)
        self.state.add_(1)
        return step()


def _cpu_adamw(specs: Sequence[ParameterSpec], **kwargs: Any) -> IsolatedAdamW8bit:
    audit = ParameterAudit(tuple(specs))
    groups = [
        {
            "params": [s.parameter for s in subset],
            "param_names": [s.name for s in subset],
            "group_name": name,
            "weight_decay": 0.0,
        }
        for name, subset in (
            ("matrix_decay", audit.decay),
            ("sensitive_no_decay", audit.sensitive),
        )
    ]
    inner = torch.optim.AdamW(groups, lr=kwargs["lr"], betas=kwargs["betas"], eps=kwargs["eps"])
    rng = _CpuSR(torch.device("cuda:0"), torch.zeros(16, dtype=torch.uint8))
    return IsolatedAdamW8bit(inner, audit, rng)


def _seed_grads(model: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ).to(parameter.dtype)


@pytest.fixture(autouse=True)
def cpu_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (core, guarded, rescue):
        monkeypatch.setattr(module, "_build_adamw8bit_for_specs", _cpu_adamw)


def _document() -> dict[str, Any]:
    """Resolved document of the tracked CMuon production config.

    The tracked config is the repository copy of the deployed optimizer setup, so
    the multipliers asserted here are the ones production actually runs with.
    """

    loaded = load_config(
        Path("train_g1_cmuon_production.toml"),
        config_root=ROOT / "config",
        environment=SECRETS,
    )
    return loaded.config.model_dump(
        mode="python",
        by_alias=True,
        exclude_computed_fields=True,
        exclude_none=True,
    )


def _config(name: str = "hybrid_cmuon", cmuon: float = 1.0, adamw: float = 1.0) -> RuntimeConfig:
    document = _document()
    opt = document["optimizer"]
    opt.update(name=name, base_lr=BASE, lr_scaling="none", cmuon_lr_multiplier=cmuon, adamw_lr_multiplier=adamw)
    if name == "hybrid_cmuon":
        del opt["cmuon_guard"]
    else:
        model = _MockComposite(_MockDiT(16, 32, 1))
        routing = core.route_cmuon_parameters(model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0)
        opt["cmuon_guard"]["references"] = {s.name: 1e-3 for s in routing.cmuon_specs}
        opt["cmuon_guard"].update(warmup_observations=0, numerical_floor=1e-20)
    document["scheduler"].update(name="linear_warmup_constant", warmup_updates=4)
    return RuntimeConfig.model_validate(document)


def _build(config: RuntimeConfig) -> tuple[nn.Module, core.HybridCMuon]:
    model = _MockComposite(_MockDiT(16, 32, 1))
    optimizer = _build_optimizer(config, cast(TrainableComposite, model))
    assert isinstance(optimizer, core.HybridCMuon)
    return model, optimizer


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("multipliers", [(1.0, 1.0), (1.0, 0.6), (1.7, 0.4)])
@pytest.mark.parametrize("tensor_lr", [False, True])
def test_schedule_explicitly_drives_both_rates(name: str, multipliers: tuple[float, float], tensor_lr: bool) -> None:
    config = _config(name, *multipliers)
    model, opt = _build(config)
    if tensor_lr:
        for group in opt.optimizer.param_groups:
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
    scheduler = _SuccessfulUpdateLrScheduler(config, opt, _config_learning_rate, restored_successful_update=0, fresh=True)
    for update in range(1, 7):
        base = _config_learning_rate(config, update)
        assert _optimizer_learning_rate_matches(opt, base, config)
        assert opt.cfg.lr == base * multipliers[0]
        captured = _optimizer_learning_rates(opt)
        expected = (base, base * multipliers[0], base * multipliers[1])
        assert all(
            math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-12)
            for actual, target in zip(captured, expected, strict=True)
        )
        # No successful edge: retry keeps exactly the same rates.
        assert _optimizer_learning_rates(opt) == captured
        _seed_grads(model, update)
        opt.step()
        assert opt.cfg.lr == base * multipliers[0]
        scheduler(update)
        assert _optimizer_learning_rate_matches(opt, _config_learning_rate(config, update + 1), config)
    with pytest.raises(ValueError, match="consecutive"):
        scheduler(6)
    opt.cfg = replace(opt.cfg, lr=BASE * 2)
    assert not _optimizer_learning_rate_matches(opt, BASE, config)
    with pytest.raises(ValueError, match="restored optimizer"):
        _SuccessfulUpdateLrScheduler(config, opt, _config_learning_rate, restored_successful_update=6, fresh=False)


@pytest.mark.parametrize("name", NAMES)
def test_step_never_reads_cmuon_lr_from_adamw(name: str) -> None:
    model, opt = _build(_config(name))
    original_lr = opt.cfg.lr
    for group in opt.optimizer.param_groups:
        group["lr"] = original_lr / 100
    _seed_grads(model, 7)
    opt.step()
    assert opt.cfg.lr == original_lr


def test_fp32_rescue_thresholds_and_moonlight_use_cmuon_lr(monkeypatch: pytest.MonkeyPatch) -> None:
    model, opt = _build(_config(NAMES[2], 1.7, 0.4))
    lr = opt.cfg.lr
    alphas: list[float] = []
    constants: list[float] = []
    original_alpha, original_tensor = rescue.cmuon_moonlight_alpha, torch.tensor

    def alpha(rows: int, cols: int, rate: float, rescale: int) -> float:
        alphas.append(rate)
        return original_alpha(rows, cols, rate, rescale)

    def tensor(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("dtype") == torch.float64 and isinstance(data, float):
            constants.append(data)
        return original_tensor(data, *args, **kwargs)

    monkeypatch.setattr(rescue, "cmuon_moonlight_alpha", alpha)
    monkeypatch.setattr(torch, "tensor", tensor)
    _seed_grads(model, 8)
    opt.step()
    assert alphas and all(rate == lr for rate in alphas)
    assert math.isclose(10.0 * 0.2 * lr, constants[0])


def test_old_checkpoint_resume_preserves_state_and_current_rates(tmp_path: Path) -> None:
    config = _config()
    model, old = _build(config)
    for seed in (1, 2):
        _seed_grads(model, seed)
        old.step()
    state = copy.deepcopy(old.state_dict())
    schema = _hybrid_optimizer_schema(old)
    train_state = tmp_path / "train_state"
    train_state.mkdir()
    torch.save(state, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    new_config = _config(cmuon=1.7, adamw=0.6)
    _, current = _build(new_config)
    resume_base = _config_learning_rate(new_config, 3)
    _set_optimizer_learning_rate(current, resume_base, new_config)
    _load_hybrid_state_exact(train_state, schema, current, old.sr_rng.state_dict(), successful_updates=2)
    _SuccessfulUpdateLrScheduler(new_config, current, _config_learning_rate, restored_successful_update=2, fresh=False)
    after = current.state_dict()
    assert set(after) == set(state)  # no new persisted multipliers/LR metadata
    assert current.cfg.lr == resume_base * 1.7
    assert _optimizer_learning_rate_matches(current, resume_base, new_config)
    before_adamw = cast(_AdamWState, state["optimizer"])
    after_adamw = cast(_AdamWState, after["optimizer"])
    for before_group, after_group in zip(before_adamw["param_groups"], after_adamw["param_groups"], strict=True):
        assert before_group["param_names"] == after_group["param_names"]
        assert after_group["lr"] == resume_base * 0.6
    for key, before_param in before_adamw["state"].items():
        for field, before_value in before_param.items():
            assert torch.equal(before_value, after_adamw["state"][key][field])
    before_cmuon = cast(_CMuonState, state["cmuon"])
    after_cmuon = cast(_CMuonState, after["cmuon"])
    for name, momentum in before_cmuon["momenta"].items():
        assert torch.equal(momentum, after_cmuon["momenta"][name])
    assert torch.equal(cast(_SRState, state["sr_rng"])["state"], cast(_SRState, after["sr_rng"])["state"])
    assert state["routing"] == after["routing"]


@pytest.mark.parametrize("field", ["cmuon_lr_multiplier", "adamw_lr_multiplier"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "1.0"])
def test_config_rejects_invalid_multipliers(field: str, bad: object) -> None:
    document = _config().optimizer.model_dump()
    document[field] = bad
    with pytest.raises((ValueError, TypeError)):
        OptimizerConfig.model_validate(document)


def test_config_defaults_and_deployed_rates() -> None:
    document = _config().optimizer.model_dump()
    del document["cmuon_lr_multiplier"], document["adamw_lr_multiplier"]
    defaults = OptimizerConfig.model_validate(document)
    assert defaults.cmuon_lr_multiplier == defaults.adamw_lr_multiplier == 1.0
    deployed = RuntimeConfig.model_validate(_document())
    assert deployed.optimizer.cmuon_lr_multiplier == 1.0
    assert deployed.optimizer.adamw_lr_multiplier == 0.6
    # Deployed production recipe: local 96 x accumulation 4 x world 2 = 768 global,
    # on the tracked base_lr/reference_batch of this config.
    deployed_global_batch = 768
    deployed_base = deployed.optimizer.base_lr * deployed_global_batch / deployed.optimizer.reference_batch
    assert math.isclose(deployed_base, BASE)
    assert math.isclose(deployed_base * deployed.optimizer.cmuon_lr_multiplier, 1.5e-4)
    assert math.isclose(deployed_base * deployed.optimizer.adamw_lr_multiplier, 9e-5)


def test_nonhybrid_default_and_scaled_paths() -> None:
    config = _config()
    model = _MockComposite(_MockDiT(16, 32, 1))
    routing = core.route_cmuon_parameters(model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0)
    opt = _cpu_adamw(routing.full_audit.specs, lr=BASE, betas=(0.9, 0.95), eps=1e-8)
    for multiplier in (1.0, 0.6):
        configured = config.model_copy(update={"optimizer": config.optimizer.model_copy(update={"adamw_lr_multiplier": multiplier})})
        _set_optimizer_learning_rate(opt, BASE, configured)
        assert _optimizer_learning_rate_matches(opt, BASE, configured)
        assert _optimizer_learning_rates(opt) == (BASE * multiplier, 0.0, BASE * multiplier)
