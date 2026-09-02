from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.checkpoint.artifact import validate_optimizer_coverage
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.optim.cmuon import route_cmuon_parameters
from sakuramoon.train.step import TrainableComposite

HIDDEN_SIZE = 2560

IREPA_FQNS = (
    "irepa_alignment.projector.bias",
    "irepa_alignment.projector.weight",
)


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


def _routing(module: nn.Module):
    return route_cmuon_parameters(
        module,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
    )


def test_irepa_projector_routes_to_adamw_not_cmuon() -> None:
    module = _composite(_irepa())

    routing = _routing(module)

    assert "irepa_alignment.projector.weight" in routing.adamw_names
    assert "irepa_alignment.projector.bias" in routing.adamw_names
    assert "irepa_alignment.projector.weight" not in routing.cmuon_names
    assert "irepa_alignment.projector.bias" not in routing.cmuon_names
    groups = {spec.name: spec.group for spec in routing.full_audit.specs}
    assert groups["irepa_alignment.projector.weight"] == "matrix_decay"
    assert groups["irepa_alignment.projector.bias"] == "sensitive_no_decay"
    # every trainable parameter is covered exactly once
    module_fqns = {name for name, _ in module.named_parameters()}
    assert set(routing.cmuon_names) | set(routing.adamw_names) == module_fqns
    assert not set(routing.cmuon_names) & set(routing.adamw_names)


def test_cmuon_allowlist_unchanged_when_auxiliary_enabled() -> None:
    legacy = _routing(_composite())
    enabled = _routing(_composite(_irepa()))

    assert enabled.cmuon_names == legacy.cmuon_names
    assert set(enabled.adamw_names) == set(legacy.adamw_names) | set(IREPA_FQNS)
    # canonical role distribution of the frozen CMuon set is untouched
    legacy_roles = sorted(spec.role for spec in legacy.cmuon_specs)
    enabled_roles = sorted(spec.role for spec in enabled.cmuon_specs)
    assert enabled_roles == legacy_roles


def test_optimizer_coverage_fails_closed_without_auxiliary_parameters() -> None:
    module = _composite(_irepa())
    full_canonical = tuple(
        sorted(module.named_parameters(remove_duplicate=False), key=lambda p: p[0])
    )

    # full coverage (the v4 lifecycle contract) passes
    validate_optimizer_coverage(module, full_canonical)

    # an optimizer built for the legacy v3 composite misses the auxiliary
    legacy_canonical = tuple(
        (name, parameter)
        for name, parameter in full_canonical
        if name not in IREPA_FQNS
    )
    with pytest.raises(ValueError, match="canonical parameters differ"):
        validate_optimizer_coverage(module, legacy_canonical)
