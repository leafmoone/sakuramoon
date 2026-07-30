"""Self-describing trainable-composite model artifact contract."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.train.step import TrainableComposite

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_ROOT_KEYS = {"class", "dit", "text", "style"}
_DIT_META_KEYS = {"active_slot_ids", "attention_backend"}


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
        or type(composite.style) is not StyleResampler
        or set(dict(composite.named_children())) != {"dit", "text", "style"}
    ):
        raise TypeError("checkpoint composite must contain only a supported DiT, text and style")
    parameters = tuple(composite.named_parameters(remove_duplicate=False))
    if not parameters or any(not parameter.requires_grad for _, parameter in parameters):
        raise ValueError("checkpoint composite parameters must all be trainable")
    if any(name.split(".", 1)[0] not in {"dit", "text", "style"} for name, _ in parameters):
        raise ValueError("checkpoint parameter is outside the trainable composite")
    metadata = composite.dit.model_metadata()
    if metadata.get("prediction_type") != "x" or metadata.get("out_channels") != 128:
        raise ValueError("checkpoint model must use the locked x-prediction head")
    return {
        "class": "TrainableComposite",
        "dit": composite.dit.artifact_config(),
        "style": composite.style.artifact_config(),
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
    if set(document) != _ROOT_KEYS or document.get("class") != "TrainableComposite":
        raise ValueError("model architecture has unknown or missing fields")
    dit_config = _mapping(document["dit"], "DiT architecture")
    recorded_slots = dit_config.get("active_slot_ids")
    if not isinstance(recorded_slots, list) or not all(
        type(slot) is int for slot in cast(list[object], recorded_slots)
    ):
        raise ValueError("DiT active slots are invalid")
    backend = dit_config.get("attention_backend")
    dit_class: type[DenseDiT | PackedDiT]
    if backend == "fa4_varlen":
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
    style_arguments = _decode_dtypes(
        _mapping(document["style"], "style architecture"), "style"
    )
    dit_arguments = _decode_dtypes(dit_arguments, "dit")
    try:
        with torch.device(device):
            module = TrainableComposite(
                dit=dit_class(**dit_arguments),  # pyright: ignore[reportArgumentType]
                text=TextConditioner(**text_arguments),
                style=StyleResampler(**style_arguments),
            )
    except (TypeError, ValueError):
        raise ValueError("model architecture cannot construct the locked composite") from None
    if list(module.dit.active_slot_ids) != recorded_slots:
        raise ValueError("model architecture active slots differ from depth")
    if export_trainable_composite(module) != document:
        raise ValueError("model architecture is not canonical")
    return module


__all__ = [
    "active_slot_ids_from_module",
    "build_trainable_composite",
    "export_trainable_composite",
    "validate_optimizer_coverage",
]
