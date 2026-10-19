"""Opt-in condition AdamW subgroups without changing parameter/SR traversal order."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from sakuramoon.config.schema import ConditionOptimizerConfig
from sakuramoon.optim.groups import ParameterSpec


def parameter_matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def select_condition_names(
    specs: Sequence[ParameterSpec], prefixes: tuple[str, ...]
) -> frozenset[str]:
    """Every explicit selector must match trainable canonical FQNs; never broaden it."""
    names = frozenset(spec.name for spec in specs if parameter_matches_prefix(spec.name, prefixes))
    missing = [prefix for prefix in prefixes if not any(parameter_matches_prefix(name, (prefix,)) for name in names)]
    if not names or missing:
        raise ValueError(f"condition parameter prefixes match no trainable parameters: {missing or prefixes}")
    return names


def split_condition_lr_groups(optimizer: Any, policy: ConditionOptimizerConfig | None) -> None:
    """Split *contiguous runs*, not sorted condition/rest buckets.

    TorchAO visits param_groups then params in order under one shared SR stream.
    Retaining this flattened order avoids reassigning stochastic-rounding draws.
    The opt-out branch does not touch groups, defaults, state, or RNG.
    """
    if policy is None or not (policy.independent_clipping or policy.independent_lr):
        return
    selected = select_condition_names(optimizer.audit.specs, policy.parameter_prefixes)
    if not policy.independent_lr:
        return
    inner = optimizer.optimizer
    adamw_names: frozenset[str] = frozenset(name for group in inner.param_groups for name in group["param_names"])
    if not selected <= adamw_names:
        raise ValueError("independent condition LR requires all selected parameters in AdamW (CMuon routing is unchanged)")
    if selected == adamw_names:
        raise ValueError("independent condition LR requires remaining AdamW parameters")
    groups: list[dict[str, Any]] = []
    for group in inner.param_groups:
        if "condition_lr_group" in group:
            raise ValueError("condition LR groups are already configured")
        names, params = group["param_names"], group["params"]
        if len(names) != len(params):
            raise ValueError("AdamW group names and parameter count differ")
        if not names:
            groups.append({**group, "condition_lr_group": False})
            continue
        run: dict[str, Any] | None = None
        for name, parameter in zip(names, params, strict=True):
            condition = name in selected
            if run is None or run["condition_lr_group"] != condition:
                run = {**group, "params": [], "param_names": [], "condition_lr_group": condition}
                # Each run needs its own scalar storage: scheduler fills tensor LRs in place.
                if isinstance(run["lr"], torch.Tensor):
                    run["lr"] = run["lr"].clone()
                groups.append(run)
            run["params"].append(parameter)
            run["param_names"].append(name)
    inner.param_groups[:] = groups


def condition_learning_rate(optimizer: object) -> float | None:
    inner = getattr(optimizer, "optimizer", None)
    if inner is None:
        return None
    groups = getattr(inner, "param_groups", ())
    rates = {
        float(group["lr"].detach().item()) if isinstance(group["lr"], torch.Tensor) else float(group["lr"])
        for group in groups if group.get("condition_lr_group") is True
    }
    if not rates:
        return None
    if len(rates) != 1:
        raise ValueError("condition AdamW learning rates differ across groups")
    rate = rates.pop()
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("condition AdamW learning rate must be finite and nonnegative")
    return rate
