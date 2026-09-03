"""Phase-4 iREPA shadow-gradient diagnostics (isolated, non-telemetry).

Separate ``g_main`` and ``g_irepa`` gradients for the SAME per-sample loss
vectors in two backward passes, so an early-training audit can report, per
representative parameter, how the main JLT gradient and the iREPA gradient
compare in norm and direction.  This is a research/canary diagnostic:

- it never steps an optimizer, publishes a checkpoint, or touches W&B;
- it restores ``set_to_none`` grad state after both passes, so a failed
  audit leaves the module exactly as it found it (no parameter mutation);
- it consumes per-sample loss tensors with live graphs — the loss wiring
  (flow JLT + cosine alignment) is built by the caller and tested elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ShadowGradientFacts:
    """Separate main / iREPA gradient facts for one parameter."""

    norm_main: float
    norm_irepa: float
    cosine: float | None  # None when either gradient is absent or zero
    main_absent: bool  # None grad (parameter not reached by the main graph)
    irepa_absent: bool  # None grad (parameter not reached by the iREPA graph)


def _flat_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """FP32 L2 norm of a flat vector via the sum-of-squares root form."""

    return (x.float().square().sum()).sqrt()


def _collect_norms(
    module: nn.Module,
    names: tuple[str, ...],
) -> dict[str, torch.Tensor | None]:
    facts: dict[str, torch.Tensor | None] = {}
    available = dict(module.named_parameters())
    for name in names:
        parameter = available.get(name)
        if parameter is None:
            raise KeyError(f"unknown parameter name: {name}")
        grad = parameter.grad
        if grad is None:
            facts[name] = None
        elif grad.is_sparse:
            raise RuntimeError(f"parameter {name} has a sparse gradient")
        else:
            facts[name] = grad.detach().reshape(-1)
    return facts


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = _flat_l2_norm(a) * _flat_l2_norm(b)
    if denominator <= 0.0:
        return 0.0
    dot = (a.float() * b.float()).sum()
    return float((dot / denominator).item())


def shadow_gradient_audit(
    *,
    module: nn.Module,
    main_per_sample: torch.Tensor,
    irepa_per_sample: torch.Tensor | None,
    parameter_names: tuple[str, ...] | None = None,
    lambda_weight: float = 1.0,
) -> dict[str, ShadowGradientFacts]:
    """Run the two-shadow backward audit and restore grad state.

    ``main_per_sample`` / ``irepa_per_sample`` are float32 ``[B]`` vectors
    with live autograd graphs.  The main pass backpropagates
    ``main_per_sample.sum()``; the iREPA pass backpropagates
    ``(lambda_weight * irepa_per_sample).sum()``.  At ``lambda_weight == 0.0``
    the iREPA pass still runs (the graph must stay traversable) and yields
    exact zero gradients for every parameter the iREPA graph reaches.

    Grad state on the module is restored to ``set_to_none`` afterwards;
    parameters are never mutated.
    """

    if main_per_sample.ndim != 1 or main_per_sample.numel() == 0:
        raise ValueError("main_per_sample must be a nonempty one-dimensional tensor")
    if main_per_sample.dtype != torch.float32:
        raise TypeError("main_per_sample must use float32")
    if irepa_per_sample is not None:
        if irepa_per_sample.shape != main_per_sample.shape:
            raise ValueError("irepa_per_sample shape differs from main_per_sample")
        if irepa_per_sample.dtype != torch.float32:
            raise TypeError("irepa_per_sample must use float32")
    if type(lambda_weight) is not float or lambda_weight < 0.0:
        raise ValueError("lambda_weight must be a finite nonnegative float")

    if parameter_names is None:
        parameter_names = tuple(name for name, _ in module.named_parameters())
    if not parameter_names:
        raise ValueError("parameter_names must not be empty")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names contain duplicates")

    module.zero_grad(set_to_none=True)
    try:
        main_per_sample.sum().backward(retain_graph=True)  # pyright: ignore[reportUnknownMemberType]
        main_facts = _collect_norms(module, parameter_names)
    finally:
        module.zero_grad(set_to_none=True)
    try:
        if irepa_per_sample is not None:
            (lambda_weight * irepa_per_sample).sum().backward()  # pyright: ignore[reportUnknownMemberType]
        irepa_facts = _collect_norms(module, parameter_names)
    finally:
        module.zero_grad(set_to_none=True)

    audit: dict[str, ShadowGradientFacts] = {}
    for name in parameter_names:
        main_grad = main_facts[name]
        irepa_grad = irepa_facts[name]
        norm_main = (
            0.0
            if main_grad is None
            else float(_flat_l2_norm(main_grad).item())
        )
        norm_irepa = (
            0.0
            if irepa_grad is None
            else float(_flat_l2_norm(irepa_grad).item())
        )
        cosine: float | None = None
        if main_grad is not None and irepa_grad is not None:
            cosine = _cosine(main_grad, irepa_grad)
        audit[name] = ShadowGradientFacts(
            norm_main=norm_main,
            norm_irepa=norm_irepa,
            cosine=cosine,
            main_absent=main_grad is None,
            irepa_absent=irepa_grad is None,
        )
    return audit


__all__ = [
    "ShadowGradientFacts",
    "shadow_gradient_audit",
]
