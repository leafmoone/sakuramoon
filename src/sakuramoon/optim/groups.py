"""Canonical-FQN parameter precision and decay audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

ParameterGroupName = Literal["matrix_decay", "sensitive_no_decay"]

# P5 spec-11 (AdamW SR RNG audit, HARD GATE): FQN prefix of the iREPA
# projector parameters.  The AdamW8bit step consumes ONE shared
# stochastic-rounding CUDA RNG stream in parameter-group order (one draw per
# BF16 decay parameter, including exact-zero gradients), so a parameter
# interleaved into the middle of the order shifts the SR consumption of every
# pre-existing parameter drawn after it.  The projector must therefore be
# APPENDED AFTER every existing parameter in the canonical audit order —
# never FQN-interleaved — which keeps the first lambda=0 update (and every
# later old-parameter SR draw) bit-identical to the no-iREPA baseline
# (spec-18 parity).
_IREPA_FQN_PREFIX = "irepa_alignment."

_SENSITIVE_ANCESTORS = {"FinalOutputHead", "GlobalConditioner"}
_SENSITIVE_RANKED_PARAMETERS = {
    ("ModalityEmbedding", "image"),
    ("ModalityEmbedding", "condition"),
    ("ModalityEmbedding", "text"),
    ("ConditionTokenEncoder", "layer_embedding"),
    ("ConditionTokenEncoder", "null_tokens"),
    ("ConditionTokenEncoder", "queries"),
    ("TextConditioner", "gate_bias"),
    ("TextConditioner", "gate_weight"),
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    parameter: nn.Parameter
    group: ParameterGroupName
    weight_decay: float


@dataclass(frozen=True)
class ParameterAudit:
    specs: tuple[ParameterSpec, ...]

    @property
    def decay(self) -> tuple[ParameterSpec, ...]:
        return tuple(spec for spec in self.specs if spec.group == "matrix_decay")

    @property
    def sensitive(self) -> tuple[ParameterSpec, ...]:
        return tuple(spec for spec in self.specs if spec.group == "sensitive_no_decay")


def audit_trainable_parameters(
    module: nn.Module,
    *,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
) -> ParameterAudit:
    """Reject parameters outside the locked BF16-matrix/FP32-sensitive policy."""

    decay_values = (matrix_weight_decay, sensitive_weight_decay)
    if any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in decay_values
    ):
        raise ValueError("parameter decay values must be finite TOML-style fractions")
    named = sorted(
        (
            (name, parameter)
            for name, parameter in module.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    if not named:
        raise ValueError("trainable module has no parameters")
    named_modules = dict(
        module.named_modules(  # pyright: ignore[reportUnknownArgumentType]
            remove_duplicate=False
        )
    )
    identities: dict[int, str] = {}
    specs: list[ParameterSpec] = []
    for name, parameter in named:
        previous = identities.setdefault(id(parameter), name)
        if previous != name:
            raise ValueError(
                f"trainable parameter is aliased by both {previous!r} and {name!r}"
            )
        owner_path, _, local_name = name.rpartition(".")
        owner = named_modules[owner_path]
        ancestor_paths = [""]
        if owner_path:
            parts = owner_path.split(".")
            ancestor_paths.extend(".".join(parts[:index]) for index in range(1, len(parts) + 1))
        ancestor_names = {type(named_modules[path]).__name__ for path in ancestor_paths}
        owner_role = (type(owner).__name__, local_name)
        is_sensitive_ancestor = bool(ancestor_names & _SENSITIVE_ANCESTORS)
        is_matrix_projection = (
            isinstance(owner, nn.Linear) and local_name == "weight"
        ) or (
            isinstance(owner, nn.MultiheadAttention)
            and local_name == "in_proj_weight"
        ) or (
            isinstance(owner, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
            and local_name == "weight"
        )
        is_sensitive = (
            parameter.ndim <= 1
            or is_sensitive_ancestor
            or owner_role in _SENSITIVE_RANKED_PARAMETERS
        )
        if is_matrix_projection and not is_sensitive:
            group: ParameterGroupName = "matrix_decay"
            weight_decay = matrix_weight_decay
            expected_dtype = torch.bfloat16
        elif is_sensitive:
            group = "sensitive_no_decay"
            weight_decay = sensitive_weight_decay
            expected_dtype = torch.float32
        else:
            raise TypeError(f"trainable parameter has no locked module role: {name}")
        if parameter.dtype != expected_dtype:
            raise TypeError(
                f"parameter role requires {expected_dtype}, got {parameter.dtype}: {name}"
            )
        specs.append(
            ParameterSpec(
                name=name,
                parameter=parameter,
                group=group,
                weight_decay=weight_decay,
            )
        )
    # P5 spec-11 (AdamW SR RNG audit): append the iREPA projector parameters
    # AFTER every existing parameter (never FQN-interleaved).  Pre-existing
    # parameters keep their exact canonical FQN-sorted relative order, so the
    # AdamW parameter-group order — and therefore the shared SR RNG
    # consumption order (one draw per BF16 decay parameter, including
    # exact-zero gradients) — is bit-identical to the no-iREPA model for the
    # whole pre-existing set.  This is the single source of truth: it
    # propagates into every AdamW group construction (build_adamw8bit,
    # _build_adamw8bit_for_specs), the routing manifest, and the checkpoint
    # migration's target group order.
    irepa_specs = [
        spec for spec in specs if spec.name.startswith(_IREPA_FQN_PREFIX)
    ]
    if irepa_specs:
        specs = [
            spec for spec in specs if not spec.name.startswith(_IREPA_FQN_PREFIX)
        ] + irepa_specs
    return ParameterAudit(specs=tuple(specs))


__all__ = [
    "ParameterAudit",
    "ParameterGroupName",
    "ParameterSpec",
    "audit_trainable_parameters",
]
