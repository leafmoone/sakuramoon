"""Self-describing trainable-composite model artifact contract."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.train.step import TrainableComposite

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_ARCHITECTURE_SCHEMA_VERSION = 3
_ROOT_KEYS = {"schema_version", "class", "dit", "text", "condition_tokens"}
_DIT_META_KEYS = {"active_slot_ids", "attention_backend"}
_STATE_COMPATIBLE_ATTENTION_BACKENDS = {
    "dense_sdpa",
    "fa4_varlen",
    "das_fa2_varlen",
}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _decode_dtypes(config: dict[str, Any], name: str) -> dict[str, Any]:
    decoded = dict(config)
    for key in ("linear_dtype", "sensitive_dtype"):
        raw = decoded.get(key)
        if raw not in _DTYPES:
            raise ValueError(f"{name}.{key} is invalid")
        decoded[key] = _DTYPES[cast(str, raw)]
    return decoded


def export_trainable_composite(module: nn.Module) -> dict[str, object]:
    if type(module) is not TrainableComposite:
        raise TypeError("checkpoint module must be an unwrapped TrainableComposite")
    composite = module
    if (
        type(composite.dit) not in {DenseDiT, PackedDiT}
        or type(composite.text) is not TextConditioner
        or type(composite.condition_tokens) is not ConditionTokenEncoder
        or set(dict(composite.named_children()))
        != {"dit", "text", "condition_tokens"}
    ):
        raise TypeError(
            "checkpoint composite must contain only a supported DiT, text and "
            "condition-token encoder"
        )
    if composite.dit.condition_token_count != composite.condition_tokens.token_count:
        raise ValueError("DiT and condition-token encoder counts differ")
    if composite.dit.hidden_size != composite.condition_tokens.output_size:
        raise ValueError("DiT and condition-token encoder widths differ")
    parameters = tuple(composite.named_parameters(remove_duplicate=False))
    if not parameters or any(not parameter.requires_grad for _, parameter in parameters):
        raise ValueError("checkpoint composite parameters must all be trainable")
    if any(
        name.split(".", 1)[0] not in {"dit", "text", "condition_tokens"}
        for name, _ in parameters
    ):
        raise ValueError("checkpoint parameter is outside the trainable composite")
    metadata = composite.dit.model_metadata()
    if metadata.get("prediction_type") != "x" or metadata.get("out_channels") != 128:
        raise ValueError("checkpoint model must use the locked x-prediction head")
    return {
        "schema_version": _ARCHITECTURE_SCHEMA_VERSION,
        "class": "TrainableComposite",
        "dit": composite.dit.artifact_config(),
        "condition_tokens": composite.condition_tokens.artifact_config(),
        "text": composite.text.artifact_config(),
    }


def validate_optimizer_coverage(
    module: nn.Module,
    canonical_parameters: tuple[tuple[str, nn.Parameter], ...],
) -> None:
    export_trainable_composite(module)
    module_parameters = tuple(
        sorted(module.named_parameters(remove_duplicate=False), key=lambda item: item[0])
    )
    if tuple(name for name, _ in module_parameters) != tuple(
        name for name, _ in canonical_parameters
    ) or any(
        module_parameter is not optimizer_parameter
        for (_, module_parameter), (_, optimizer_parameter) in zip(
            module_parameters, canonical_parameters, strict=True
        )
    ):
        raise ValueError("checkpoint module and optimizer canonical parameters differ")


def active_slot_ids_from_module(module: nn.Module) -> tuple[int, ...]:
    export_trainable_composite(module)
    slots = cast(TrainableComposite, module).dit.active_slot_ids
    if not all(type(slot) is int for slot in slots):
        raise ValueError("checkpoint module active slots are invalid")
    return slots


def build_trainable_composite(
    value: object,
    *,
    device: torch.device | str,
) -> TrainableComposite:
    document = _mapping(value, "model architecture")
    if (
        set(document) != _ROOT_KEYS
        or document.get("schema_version") != _ARCHITECTURE_SCHEMA_VERSION
        or document.get("class") != "TrainableComposite"
    ):
        raise ValueError("model architecture has unknown or missing fields")
    dit_config = _mapping(document["dit"], "DiT architecture")
    recorded_slots = dit_config.get("active_slot_ids")
    if not isinstance(recorded_slots, list) or not all(
        type(slot) is int for slot in cast(list[object], recorded_slots)
    ):
        raise ValueError("DiT active slots are invalid")
    backend = dit_config.get("attention_backend")
    dit_class: type[DenseDiT | PackedDiT]
    if backend in {"fa4_varlen", "das_fa2_varlen"}:
        dit_class = PackedDiT
    elif backend == "dense_sdpa":
        dit_class = DenseDiT
    else:
        raise ValueError("model artifact attention backend is invalid")
    dit_arguments = {
        key: item for key, item in dit_config.items() if key not in _DIT_META_KEYS
    }
    text_arguments = _decode_dtypes(
        _mapping(document["text"], "text architecture"), "text"
    )
    condition_token_arguments = _decode_dtypes(
        _mapping(document["condition_tokens"], "condition-token architecture"),
        "condition_tokens",
    )
    dit_arguments = _decode_dtypes(dit_arguments, "dit")
    try:
        with torch.device(device):
            module = TrainableComposite(
                dit=dit_class(**dit_arguments),  # pyright: ignore[reportArgumentType]
                text=TextConditioner(**text_arguments),
                condition_tokens=ConditionTokenEncoder(**condition_token_arguments),
            )
    except (TypeError, ValueError):
        raise ValueError("model architecture cannot construct the locked composite") from None
    if list(module.dit.active_slot_ids) != recorded_slots:
        raise ValueError("model architecture active slots differ from depth")
    if export_trainable_composite(module) != document:
        raise ValueError("model architecture is not canonical")
    return module


def architectures_share_parameter_contract(left: object, right: object) -> bool:
    """Compare artifacts while treating parameter-free attention backends alike."""

    try:
        left_document = _mapping(left, "left model architecture")
        right_document = _mapping(right, "right model architecture")
        left_dit = _mapping(left_document.get("dit"), "left DiT architecture")
        right_dit = _mapping(right_document.get("dit"), "right DiT architecture")
    except TypeError:
        return False
    left_backend = left_dit.get("attention_backend")
    right_backend = right_dit.get("attention_backend")
    if (
        left_backend not in _STATE_COMPATIBLE_ATTENTION_BACKENDS
        or right_backend not in _STATE_COMPATIBLE_ATTENTION_BACKENDS
    ):
        return False
    normalized_left = {
        **left_document,
        "dit": {**left_dit, "attention_backend": "state_compatible_gqa"},
    }
    normalized_right = {
        **right_document,
        "dit": {**right_dit, "attention_backend": "state_compatible_gqa"},
    }
    return normalized_left == normalized_right


__all__ = [
    "active_slot_ids_from_module",
    "architectures_share_parameter_contract",
    "build_trainable_composite",
    "export_trainable_composite",
    "validate_optimizer_coverage",
]
