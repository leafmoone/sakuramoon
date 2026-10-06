"""Public exports, loaded on demand so inference needs no training extras."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sakuramoon.optim.adamw8bit import (
        IsolatedAdamW8bit,
        OptimizerStateSpec,
        build_adamw8bit,
    )
    from sakuramoon.optim.clip import ClipResult, clip_grad_norm_fp32
    from sakuramoon.optim.groups import (
        ParameterAudit,
        ParameterSpec,
        audit_trainable_parameters,
    )
    from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG

_EXPORT_MODULES = {
    "IsolatedAdamW8bit": "sakuramoon.optim.adamw8bit",
    "OptimizerStateSpec": "sakuramoon.optim.adamw8bit",
    "build_adamw8bit": "sakuramoon.optim.adamw8bit",
    "ClipResult": "sakuramoon.optim.clip",
    "clip_grad_norm_fp32": "sakuramoon.optim.clip",
    "ParameterAudit": "sakuramoon.optim.groups",
    "ParameterSpec": "sakuramoon.optim.groups",
    "audit_trainable_parameters": "sakuramoon.optim.groups",
    "StochasticRoundingRNG": "sakuramoon.optim.stochastic_rounding",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ClipResult",
    "IsolatedAdamW8bit",
    "OptimizerStateSpec",
    "ParameterAudit",
    "ParameterSpec",
    "StochasticRoundingRNG",
    "audit_trainable_parameters",
    "build_adamw8bit",
    "clip_grad_norm_fp32",
]
