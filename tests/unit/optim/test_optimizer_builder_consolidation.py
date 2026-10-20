# pyright: reportPrivateUsage=false
"""CPU regression tests for the consolidated ``_build_optimizer``.

The five builder entry points (and the split/LR tail) are mocked so each
family's exact kwargs, the fail-closed error boundaries, and the
split-then-LR tail order are pinned to the pre-consolidation
``_build_optimizer`` / ``_build_optimizer_unscaled`` behavior.
"""
from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from typing import Any, cast

import pytest
from test_cmuon import _MockComposite, _MockDiT
from test_learning_rate_multipliers import BASE, _config
from torch import nn

import sakuramoon.optim.cmuon as core
import sakuramoon.optim.fp32_rescue as rescue
import sakuramoon.optim.guarded_canonical as guarded
from sakuramoon.config.schema import (
    CMuonForensicConfig,
    CMuonNSTelemetryConfig,
    RuntimeConfig,
)
from sakuramoon.optim import condition
from sakuramoon.train import production
from sakuramoon.train.step import TrainableComposite

# Real builder signature defaults, captured before any fixture patches the
# entry points; the consolidated call passes these explicitly instead of
# relying on omission.
_REAL_HYBRID_PARAMS = inspect.signature(core.build_hybrid_cmuon).parameters
REAL_NS_STEPS_DEFAULT: int = _REAL_HYBRID_PARAMS["ns_steps"].default
REAL_NS_BY_ROLE_DEFAULT: object = _REAL_HYBRID_PARAMS["ns_steps_by_role"].default


class _Built:
    """Sentinel optimizer returned by the mocked builders."""

    def __init__(self, family: str) -> None:
        self.family = family


class _Capture:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.guard_kwargs: list[dict[str, Any]] = []
        self.tail_order: list[str] = []
        self.split_args: list[tuple[Any, Any]] = []
        self.lr_args: list[tuple[Any, Any, Any]] = []

    def install(self, family: str, target: Any, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(module_arg: nn.Module, **kwargs: Any) -> _Built:
            self.calls.append((family, kwargs))
            return _Built(family)

        monkeypatch.setattr(target, name, fake)

    def guard_config(self, **kwargs: Any) -> Any:
        self.guard_kwargs.append(kwargs)
        return kwargs

    def split(self, optimizer: Any, condition_config: Any) -> None:
        self.tail_order.append("split")
        self.split_args.append((optimizer, condition_config))

    def set_lr(self, optimizer: Any, rate: Any, config: Any) -> None:
        self.tail_order.append("lr")
        self.lr_args.append((optimizer, rate, config))


@pytest.fixture()
def cap(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    capture = _Capture()
    capture.install("hybrid_cmuon", core, "build_hybrid_cmuon", monkeypatch)
    capture.install("guarded_canonical", guarded, "build_guarded_canonical", monkeypatch)
    capture.install("fp32_rescue", rescue, "build_fp32_rescue", monkeypatch)
    capture.install("adamw8bit", production, "build_adamw8bit", monkeypatch)
    monkeypatch.setattr(guarded, "GuardedCanonicalGuardConfig", capture.guard_config)
    monkeypatch.setattr(condition, "split_condition_lr_groups", capture.split)
    monkeypatch.setattr(production, "_set_optimizer_learning_rate", capture.set_lr)
    return capture


def _model() -> nn.Module:
    return _MockComposite(_MockDiT(16, 32, 1))


def _variant(config: RuntimeConfig, **opt_updates: Any) -> RuntimeConfig:
    # Frozen models: build variants without re-validation on purpose.
    optimizer = config.optimizer.model_copy(update=opt_updates)
    return config.model_copy(update={"optimizer": optimizer})


def _build(
    config: RuntimeConfig,
    cap: _Capture,
    telemetry_logger: Callable[[str], None] | None = None,
    rank: int = 0,
    world_size: int = 1,
    checkpoint_source: str | None = None,
) -> _Built:
    optimizer = production._build_optimizer(
        config,
        cast(TrainableComposite, _model()),
        telemetry_logger=telemetry_logger,
        rank=rank,
        world_size=world_size,
        checkpoint_source=checkpoint_source,
    )
    assert isinstance(optimizer, _Built)
    return optimizer


def _common_kwargs(config: RuntimeConfig) -> dict[str, Any]:
    spec = config.optimizer
    return {
        "lr": config.scaled_learning_rate(),
        "betas": spec.betas,
        "eps": spec.eps,
        "block_size": spec.block_size,
        "bf16_stochastic_round": spec.bf16_stochastic_round,
        "matrix_weight_decay": spec.matrix_weight_decay,
        "sensitive_weight_decay": spec.sensitive_weight_decay,
        "sr_seed": config.run.seed,
    }


def test_signature_preserved_and_unscaled_removed() -> None:
    assert "_build_optimizer_unscaled" not in vars(production)
    parameters = list(inspect.signature(production._build_optimizer).parameters)
    assert parameters == [
        "config", "module", "telemetry_logger", "rank", "world_size",
        "checkpoint_source",
    ]


def test_default_adamw_kwargs(cap: _Capture) -> None:
    config = _variant(_config("hybrid_cmuon"), name="torchao_adamw8bit")
    built = _build(config, cap)
    assert len(cap.calls) == 1
    family, kwargs = cap.calls[0]
    assert family == "adamw8bit"
    assert kwargs == _common_kwargs(config)
    assert kwargs["lr"] == BASE
    assert built.family == "adamw8bit"


def test_hybrid_per_role_kwargs(cap: _Capture) -> None:
    config = _config("hybrid_cmuon")
    # A non-default legacy scalar must NOT leak into the per-role path.
    config = _variant(config, cmuon_ns_steps=9)
    assert config.optimizer.cmuon_ns is not None
    _build(config, cap)
    family, kwargs = cap.calls[0]
    assert family == "hybrid_cmuon"
    assert kwargs["ns_steps"] == REAL_NS_STEPS_DEFAULT
    assert kwargs["ns_steps"] != 9
    assert kwargs["ns_steps_by_role"] == config.optimizer.cmuon_ns.canonical_map()
    assert kwargs["ns_telemetry_enabled"] is False
    assert kwargs["ns_telemetry_log_every_n"] == 100
    assert kwargs["ns_telemetry_roles"] is None
    assert kwargs["ns_telemetry_logger"] is None
    assert kwargs["exclude_roles"] == ()
    assert kwargs["forensic"] is None
    assert "rank" not in kwargs and "world_size" not in kwargs
    for key, value in _common_kwargs(config).items():
        assert kwargs[key] == value


def test_hybrid_scalar_kwargs(cap: _Capture) -> None:
    config = _variant(_config("hybrid_cmuon"), cmuon_ns=None, cmuon_ns_steps=7)
    _build(config, cap)
    family, kwargs = cap.calls[0]
    assert family == "hybrid_cmuon"
    assert kwargs["ns_steps"] == 7
    assert kwargs["ns_steps_by_role"] is None
    assert kwargs["ns_steps_by_role"] == REAL_NS_BY_ROLE_DEFAULT


def test_hybrid_forensic_context(cap: _Capture) -> None:
    config = _variant(
        _config("hybrid_cmuon"),
        cmuon_forensic=CMuonForensicConfig(enabled=True),
    )
    _build(config, cap)
    family, kwargs = cap.calls[0]
    assert family == "hybrid_cmuon"
    forensic = kwargs["forensic"]
    assert forensic is not None
    assert forensic.enabled is True
    assert forensic.ring_size == 10
    assert forensic.dump_dir is None


def test_guarded_canonical_kwargs(cap: _Capture) -> None:
    config = _config("hybrid_cmuon_guarded_canonical")
    assert config.optimizer.cmuon_ns is not None
    _build(config, cap)
    family, kwargs = cap.calls[0]
    assert family == "guarded_canonical"
    guard = config.optimizer.cmuon_guard
    assert guard is not None
    assert kwargs["ns_steps_by_role"] == config.optimizer.cmuon_ns.canonical_map()
    assert cap.guard_kwargs == [{
        "guard_ratio": guard.guard_ratio,
        "reference_decay": guard.reference_decay,
        "min_reference": guard.min_reference,
        "numerical_floor": guard.numerical_floor,
        "warmup_observations": guard.warmup_observations,
        "invariant_check": guard.invariant_check,
    }]
    assert kwargs["guard_cfg"] is cap.guard_kwargs[0]
    assert kwargs["guard_bootstrap_refs"] == dict(guard.references)
    assert kwargs["rank"] == 0 and kwargs["world_size"] == 1
    assert kwargs["stats_logger"] is None
    assert "checkpoint_source" not in kwargs and "run_id" not in kwargs
    for key, value in _common_kwargs(config).items():
        assert kwargs[key] == value


def test_rescue_kwargs(cap: _Capture) -> None:
    config = _config("hybrid_cmuon_canonical_ns4_fp32_rescue")

    def logger(line: str) -> None:
        pass

    _build(
        config, cap, telemetry_logger=logger, rank=0, world_size=2,
        checkpoint_source="ckpt-7",
    )
    family, kwargs = cap.calls[0]
    assert family == "fp32_rescue"
    assert kwargs["rank"] == 0 and kwargs["world_size"] == 2
    assert kwargs["checkpoint_source"] == "ckpt-7"
    assert kwargs["run_id"] == config.run.run_id
    assert kwargs["stats_logger"] is logger
    assert "ns_steps" not in kwargs
    for key, value in _common_kwargs(config).items():
        assert kwargs[key] == value


def test_tail_split_before_lr_with_args(cap: _Capture) -> None:
    config = _config("hybrid_cmuon")
    built = _build(config, cap)
    assert cap.tail_order == ["split", "lr"]
    (split_optimizer, split_condition), = cap.split_args
    (lr_optimizer, lr_rate, lr_config), = cap.lr_args
    assert split_optimizer is built and lr_optimizer is built
    assert split_condition is config.optimizer.condition
    assert lr_rate == config.scaled_learning_rate()
    assert lr_config is config


@pytest.mark.parametrize(
    ("opt_updates", "match"),
    [
        ({"cmuon_forensic": CMuonForensicConfig(enabled=True)},
         "two-phase fail-closed"),
        ({"cmuon_ns_telemetry": CMuonNSTelemetryConfig(enabled=True)},
         "per-step safety"),
        ({"cmuon_ns": None}, "requires an explicit"),
    ],
    ids=["forensic-enabled", "ns-telemetry-enabled", "ns-map-missing"],
)
def test_guarded_fail_closed_boundaries(
    cap: _Capture,
    opt_updates: dict[str, Any],
    match: str,
) -> None:
    config = _variant(_config("hybrid_cmuon_guarded_canonical"), **opt_updates)
    with pytest.raises(ValueError, match=match):
        _build(config, cap)
    assert cap.calls == []
    assert cap.tail_order == []


def test_guard_config_raise_before_constructor(
    cap: _Capture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**kwargs: Any) -> Any:
        raise RuntimeError("guard-cfg-invalid")

    monkeypatch.setattr(guarded, "GuardedCanonicalGuardConfig", _raise)
    config = _config("hybrid_cmuon_guarded_canonical")
    with pytest.raises(RuntimeError, match="guard-cfg-invalid"):
        _build(config, cap)
    assert cap.calls == []


def test_canonical_map_raise_and_constructor_raise(
    cap: _Capture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("hybrid_cmuon_guarded_canonical")
    assert config.optimizer.cmuon_ns is not None
    ns_type = type(config.optimizer.cmuon_ns)
    original_canonical_map: Callable[[Any], dict[str, int]] = ns_type.canonical_map

    def _ns_raise(self: Any) -> dict[str, int]:
        raise ValueError("ns-map-invalid")

    monkeypatch.setattr(ns_type, "canonical_map", _ns_raise)
    with pytest.raises(ValueError, match="ns-map-invalid"):
        _build(config, cap)
    assert cap.calls == []

    monkeypatch.setattr(ns_type, "canonical_map", original_canonical_map)

    def _ctor_raise(module_arg: nn.Module, **kwargs: Any) -> _Built:
        raise ValueError("ctor-invalid")

    monkeypatch.setattr(guarded, "build_guarded_canonical", _ctor_raise)
    with pytest.raises(ValueError, match="ctor-invalid"):
        _build(config, cap)
    assert cap.calls == []
    assert cap.tail_order == []


def test_rescue_import_fail_precedes_map_fail(
    cap: _Capture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Break the lazy fp32_rescue import AND make canonical_map fail; the
    # original order requires the import error to surface first.
    config = _config("hybrid_cmuon_canonical_ns4_fp32_rescue")
    assert config.optimizer.cmuon_ns is not None
    ns_type = type(config.optimizer.cmuon_ns)

    def _ns_raise(self: Any) -> dict[str, int]:
        raise ValueError("ns-map-invalid")

    monkeypatch.setattr(ns_type, "canonical_map", _ns_raise)
    monkeypatch.setitem(sys.modules, "sakuramoon.optim.fp32_rescue", None)
    with pytest.raises(ImportError):
        _build(config, cap)
    assert cap.calls == []


def test_rescue_map_fail_after_successful_import(
    cap: _Capture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("hybrid_cmuon_canonical_ns4_fp32_rescue")
    assert config.optimizer.cmuon_ns is not None
    ns_type = type(config.optimizer.cmuon_ns)

    def _ns_raise(self: Any) -> dict[str, int]:
        raise ValueError("ns-map-invalid")

    monkeypatch.setattr(ns_type, "canonical_map", _ns_raise)
    with pytest.raises(ValueError, match="ns-map-invalid"):
        _build(config, cap)
    assert cap.calls == []


def test_guarded_map_fail_precedes_guard_config_fail(
    cap: _Capture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**kwargs: Any) -> Any:
        raise RuntimeError("guard-cfg-invalid")

    monkeypatch.setattr(guarded, "GuardedCanonicalGuardConfig", _raise)
    config = _config("hybrid_cmuon_guarded_canonical")
    assert config.optimizer.cmuon_ns is not None
    ns_type = type(config.optimizer.cmuon_ns)

    def _ns_raise(self: Any) -> dict[str, int]:
        raise ValueError("ns-map-invalid")

    monkeypatch.setattr(ns_type, "canonical_map", _ns_raise)
    with pytest.raises(ValueError, match="ns-map-invalid"):
        _build(config, cap)
    assert cap.calls == []
