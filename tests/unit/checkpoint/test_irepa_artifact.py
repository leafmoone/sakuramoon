from __future__ import annotations

import copy
from typing import Any, cast

import torch

from sakuramoon.checkpoint.artifact import (
    architectures_share_parameter_contract,
    build_trainable_composite,
    export_trainable_composite,
)
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.model.irepa import IRepaAlignment, irepa_alignment_metadata
from sakuramoon.train.step import TrainableComposite

HIDDEN_SIZE = 2560


def _production_dit() -> PackedDiT:
    with torch.device("meta"):
        return PackedDiT(
            depth=16,
            input_channels=128,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=6912,
            q_heads=20,
            kv_heads=5,
            head_dim=128,
            rope_nope_dim=32,
            rope_y_dim=48,
            rope_x_dim=48,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=5120,
            out_channels=128,
            condition_token_count=8,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        )


def _production_text() -> TextConditioner:
    with torch.device("meta"):
        return TextConditioner(
            input_size=2048,
            adapter_size=1024,
            output_size=HIDDEN_SIZE,
            groups=8,
            attention_heads=8,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        )


def _production_condition_tokens() -> ConditionTokenEncoder:
    with torch.device("meta"):
        return ConditionTokenEncoder(
            input_size=2048,
            hidden_size=1024,
            intermediate_size=2048,
            output_size=HIDDEN_SIZE,
            token_count=8,
            attention_heads=8,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        )


def _composite(irepa: IRepaAlignment | None = None) -> TrainableComposite:
    with torch.device("meta"):
        return TrainableComposite(
            dit=_production_dit(),
            text=_production_text(),
            condition_tokens=_production_condition_tokens(),
            irepa_alignment=irepa,
        )


def _irepa() -> IRepaAlignment:
    with torch.device("meta"):
        return IRepaAlignment(HIDDEN_SIZE)


def _as_document(value: dict[str, object]) -> dict[str, Any]:
    return cast(dict[str, Any], value)


def test_v3_roundtrip_is_exact() -> None:
    module = _composite()

    document = export_trainable_composite(module)
    assert document["schema_version"] == 3
    assert set(document) == {
        "schema_version",
        "class",
        "dit",
        "text",
        "condition_tokens",
    }

    rebuilt = build_trainable_composite(document, device="meta")
    assert rebuilt.irepa_alignment is None
    assert export_trainable_composite(rebuilt) == document


def test_v4_roundtrip_is_exact() -> None:
    module = _composite(_irepa())

    document = export_trainable_composite(module)
    assert document["schema_version"] == 4
    assert set(document) == {
        "schema_version",
        "class",
        "dit",
        "text",
        "condition_tokens",
        "training_auxiliaries",
    }
    auxiliaries = cast(dict[str, Any], document["training_auxiliaries"])
    assert set(auxiliaries) == {"irepa"}
    assert auxiliaries["irepa"] == irepa_alignment_metadata(HIDDEN_SIZE)

    rebuilt = build_trainable_composite(document, device="meta")
    assert type(rebuilt.irepa_alignment) is IRepaAlignment
    assert rebuilt.irepa_alignment.projector.in_channels == HIDDEN_SIZE
    assert export_trainable_composite(rebuilt) == document


def test_v3_v4_parameter_contracts_are_incompatible() -> None:
    v3 = export_trainable_composite(_composite())
    v4 = export_trainable_composite(_composite(_irepa()))

    assert architectures_share_parameter_contract(v3, v4) is False
    assert architectures_share_parameter_contract(v4, v3) is False
    assert architectures_share_parameter_contract(v3, copy.deepcopy(v3)) is True
    assert architectures_share_parameter_contract(v4, copy.deepcopy(v4)) is True


def test_v4_unknown_fields_fail_closed() -> None:
    document = _as_document(export_trainable_composite(_composite(_irepa())))

    unknown_root = copy.deepcopy(document)
    unknown_root["training_auxiliaries_extra"] = {}
    try:
        build_trainable_composite(unknown_root, device="meta")
    except ValueError as error:
        assert "unknown or missing fields" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown v4 root key")

    unknown_auxiliary = copy.deepcopy(document)
    unknown_auxiliary["training_auxiliaries"] = {"other": document["training_auxiliaries"]}
    try:
        build_trainable_composite(unknown_auxiliary, device="meta")
    except ValueError as error:
        assert "unknown training auxiliary" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown training auxiliary")

    v3_with_auxiliaries = _as_document(export_trainable_composite(_composite()))
    v3_with_auxiliaries["training_auxiliaries"] = document["training_auxiliaries"]
    try:
        build_trainable_composite(v3_with_auxiliaries, device="meta")
    except ValueError as error:
        assert "unknown or missing fields" in str(error)
    else:
        raise AssertionError("expected ValueError for v3 root with auxiliaries")

    bad_version = copy.deepcopy(document)
    bad_version["schema_version"] = 5
    try:
        build_trainable_composite(bad_version, device="meta")
    except ValueError as error:
        assert "schema version is unsupported" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown schema version")


def test_v4_corrupted_projector_metadata_fails_closed() -> None:
    document = _as_document(export_trainable_composite(_composite(_irepa())))

    def corrupt(key: str, value: object) -> dict[str, Any]:
        broken = copy.deepcopy(document)
        metadata = cast(dict[str, Any], broken["training_auxiliaries"])
        irepa_meta = cast(dict[str, Any], metadata["irepa"])
        irepa_meta[key] = value
        return broken

    for key, value in (
        ("kernel_size", 5),
        ("out_channels", 512),
        ("stride", 2),
        ("groups", 2),
        ("bias", False),
        ("weight_dtype", "float16"),
        ("bias_dtype", "bfloat16"),
        ("in_channels", 0),
        ("class", "OtherProjector"),
        ("schema_version", 2),
    ):
        try:
            build_trainable_composite(corrupt(key, value), device="meta")
        except ValueError as error:
            assert "irepa" in str(error).lower()
        else:
            raise AssertionError(f"expected ValueError for corrupted {key}")

    broken = copy.deepcopy(document)
    metadata = cast(dict[str, Any], broken["training_auxiliaries"])
    irepa_meta = cast(dict[str, Any], metadata["irepa"])
    del irepa_meta["bias"]
    try:
        build_trainable_composite(broken, device="meta")
    except ValueError as error:
        assert "irepa" in str(error).lower()
    else:
        raise AssertionError("expected ValueError for missing projector metadata key")


def test_v4_export_rejects_width_mismatch() -> None:
    module = _composite()
    irepa = _irepa()
    # forge a width mismatch without touching the real DiT:
    irepa.projector.in_channels = HIDDEN_SIZE + 1  # type: ignore[misc]
    module.irepa_alignment = irepa
    try:
        export_trainable_composite(module)
    except ValueError as error:
        assert "iREPA projector input width differ" in str(error)
    else:
        raise AssertionError("expected ValueError for projector/DiT width mismatch")
