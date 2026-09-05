"""Self-describing trainable-composite model artifact contract."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT, PackedDiT
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.train.step import TrainableComposite

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_ARCHITECTURE_SCHEMA_VERSION = 3
# v4 is the iREPA-enabled training-auxiliary track; v3 stays the permanent
# legacy no-iREPA contract.
_ARCHITECTURE_SCHEMA_VERSION_V4 = 4
_ROOT_KEYS = {"schema_version", "class", "dit", "text", "condition_tokens"}
_ROOT_KEYS_V4 = _ROOT_KEYS | {"training_auxiliaries"}
_DIT_META_KEYS = {"active_slot_ids", "attention_backend"}
_IREPA_META_KEYS = {
    "class",
    "schema_version",
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "bias",
    "weight_dtype",
    "bias_dtype",
}
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
    irepa = composite.irepa_alignment
    has_irepa = irepa is not None
    if has_irepa and type(irepa) is not IRepaAlignment:
        raise TypeError(
            "checkpoint composite training auxiliary must be an IRepaAlignment"
        )
    expected_children = {"dit", "text", "condition_tokens"}
    allowed_roots = {"dit", "text", "condition_tokens"}
    if has_irepa:
        expected_children = expected_children | {"irepa_alignment"}
        allowed_roots = allowed_roots | {"irepa_alignment"}
    if (
        type(composite.dit) not in {DenseDiT, PackedDiT}
        or type(composite.text) is not TextConditioner
        or type(composite.condition_tokens) is not ConditionTokenEncoder
        or (has_irepa and type(irepa) is not IRepaAlignment)
        or set(dict(composite.named_children())) != expected_children
    ):
        raise TypeError(
            "checkpoint composite must contain only a supported DiT, text and "
            "condition-token encoder"
            + (" plus the iREPA alignment auxiliary" if has_irepa else "")
        )
    if composite.dit.condition_token_count != composite.condition_tokens.token_count:
        raise ValueError("DiT and condition-token encoder counts differ")
    if composite.dit.hidden_size != composite.condition_tokens.output_size:
        raise ValueError("DiT and condition-token encoder widths differ")
    if has_irepa and irepa.projector.in_channels != composite.dit.hidden_size:
        raise ValueError(
            "DiT hidden width and iREPA projector input width differ"
        )
    parameters = tuple(composite.named_parameters(remove_duplicate=False))
    if not parameters or any(not parameter.requires_grad for _, parameter in parameters):
        raise ValueError("checkpoint composite parameters must all be trainable")
    if any(
        name.split(".", 1)[0] not in allowed_roots for name, _ in parameters
    ):
        raise ValueError("checkpoint parameter is outside the trainable composite")
    metadata = composite.dit.model_metadata()
    if metadata.get("prediction_type") != "x" or metadata.get("out_channels") != 128:
        raise ValueError("checkpoint model must use the locked x-prediction head")
    document: dict[str, object] = {
        "schema_version": (
            _ARCHITECTURE_SCHEMA_VERSION_V4
            if has_irepa
            else _ARCHITECTURE_SCHEMA_VERSION
        ),
        "class": "TrainableComposite",
        "dit": composite.dit.artifact_config(),
        "condition_tokens": composite.condition_tokens.artifact_config(),
        "text": composite.text.artifact_config(),
    }
    if has_irepa:
        document["training_auxiliaries"] = {"irepa": irepa.artifact_config()}
    return document


def validate_optimizer_coverage(
    module: nn.Module,
    canonical_parameters: tuple[tuple[str, nn.Parameter], ...],
) -> None:
    export_trainable_composite(module)
    # Coverage is order-independent by contract: the canonical audit order
    # deliberately appends introduced auxiliary parameters (e.g. the iREPA
    # projector, P5 spec-11) after every pre-existing parameter, so the
    # optimizer's audit order is not a plain FQN sort on v4 modules.  Both
    # sides are compared in FQN order; the names must match exactly and each
    # name must bind to the identical parameter object.
    canonical_sorted = tuple(sorted(canonical_parameters, key=lambda item: item[0]))
    module_parameters = tuple(
        sorted(module.named_parameters(remove_duplicate=False), key=lambda item: item[0])
    )
    if tuple(name for name, _ in module_parameters) != tuple(
        name for name, _ in canonical_sorted
    ) or any(
        module_parameter is not optimizer_parameter
        for (_, module_parameter), (_, optimizer_parameter) in zip(
            module_parameters, canonical_sorted, strict=True
        )
    ):
        raise ValueError("checkpoint module and optimizer canonical parameters differ")


def active_slot_ids_from_module(module: nn.Module) -> tuple[int, ...]:
    export_trainable_composite(module)
    slots = cast(TrainableComposite, module).dit.active_slot_ids
    if not all(type(slot) is int for slot in slots):
        raise ValueError("checkpoint module active slots are invalid")
    return slots


def _decode_irepa_auxiliary(value: object) -> int:
    """Strictly decode the locked iREPA v1 auxiliary document.

    Returns the projector input width; the module itself is constructed on
    the requested device by ``build_trainable_composite``.
    """

    auxiliaries = _mapping(value, "training auxiliaries")
    if set(auxiliaries) != {"irepa"}:
        raise ValueError("model architecture has an unknown training auxiliary")
    meta = _mapping(auxiliaries["irepa"], "irepa auxiliary metadata")
    if set(meta) != _IREPA_META_KEYS:
        raise ValueError(
            "irepa auxiliary metadata has unknown or missing fields"
        )
    if (
        meta["class"] != "IRepaAlignment"
        or meta["schema_version"] != 1
        or meta["out_channels"] != 768
        or meta["kernel_size"] != 3
        or meta["stride"] != 1
        or meta["padding"] != 1
        or meta["dilation"] != 1
        or meta["groups"] != 1
        or meta["bias"] is not True
        or meta["weight_dtype"] != "bfloat16"
        or meta["bias_dtype"] != "float32"
    ):
        raise ValueError(
            "irepa auxiliary metadata is not the locked v1 projector contract"
        )
    in_channels = meta["in_channels"]
    if type(in_channels) is not int or in_channels <= 0:
        raise ValueError("irepa auxiliary input width is invalid")
    return in_channels


def build_trainable_composite(
    value: object,
    *,
    device: torch.device | str,
) -> TrainableComposite:
    document = _mapping(value, "model architecture")
    version = document.get("schema_version")
    irepa_in_channels: int | None = None
    if version == _ARCHITECTURE_SCHEMA_VERSION:
        if (
            set(document) != _ROOT_KEYS
            or document.get("class") != "TrainableComposite"
        ):
            raise ValueError("model architecture has unknown or missing fields")
    elif version == _ARCHITECTURE_SCHEMA_VERSION_V4:
        if (
            set(document) != _ROOT_KEYS_V4
            or document.get("class") != "TrainableComposite"
        ):
            raise ValueError(
                "v4 model architecture has unknown or missing fields"
            )
        irepa_in_channels = _decode_irepa_auxiliary(document["training_auxiliaries"])
    else:
        raise ValueError("model architecture schema version is unsupported")
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
    # iREPA projector: construct on the same requested device as the rest of
    # the canonical composite so the builder returns a fully placed trainable
    # composite (the production contract forbids a post-build device move).
    irepa_alignment: IRepaAlignment | None = None
    if irepa_in_channels is not None:
        with torch.device(device):
            try:
                irepa_alignment = IRepaAlignment(irepa_in_channels)
            except (TypeError, ValueError):
                raise ValueError(
                    "irepa auxiliary metadata cannot construct the locked projector"
                ) from None
    try:
        with torch.device(device):
            module = TrainableComposite(
                dit=dit_class(**dit_arguments),  # pyright: ignore[reportArgumentType]
                text=TextConditioner(**text_arguments),
                condition_tokens=ConditionTokenEncoder(**condition_token_arguments),
                irepa_alignment=irepa_alignment,
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
