"""Canonical-FQN parameter precision and decay audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

ParameterGroupName = Literal["matrix_decay", "sensitive_no_decay"]

_SENSITIVE_ANCESTORS = {"FinalOutputHead", "GlobalConditioner"}
_SENSITIVE_RANKED_PARAMETERS = {
    ("ModalityEmbedding", "image"),
    ("ModalityEmbedding", "style"),
    ("ModalityEmbedding", "text"),
    ("StyleResampler", "layer_embedding"),
    ("StyleResampler", "null_tokens"),
    ("StyleResampler", "queries"),
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
    schema_sha256: str

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

    if matrix_weight_decay != 0.01 or sensitive_weight_decay != 0.0:
        raise ValueError("parameter decay values differ from the locked policy")
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
    schema_rows: list[dict[str, object]] = []
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
        schema_rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "group": group,
                "weight_decay": weight_decay,
            }
        )
    encoded = json.dumps(
        schema_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ParameterAudit(
        specs=tuple(specs),
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )


__all__ = [
    "ParameterAudit",
    "ParameterGroupName",
    "ParameterSpec",
    "audit_trainable_parameters",
]
