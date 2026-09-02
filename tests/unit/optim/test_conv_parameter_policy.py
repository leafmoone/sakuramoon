from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.model.dit import PackedDiT
from sakuramoon.model.mixed_precision_conv import MixedPrecisionConv2d
from sakuramoon.optim.cmuon import route_cmuon_parameters
from sakuramoon.optim.groups import audit_trainable_parameters


def _audit(module: nn.Module):
    return audit_trainable_parameters(
        module,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
    )


def _conv2d_bf16_bias_fp32() -> nn.Conv2d:
    conv = nn.Conv2d(16, 16, 3, dtype=torch.bfloat16)
    conv.bias = nn.Parameter(torch.zeros(16, dtype=torch.float32))
    return conv


def test_linear_weight_still_matrix_decay_and_conv_weight_is_matrix() -> None:
    module = nn.Module()
    module.linear = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    module.conv1d = nn.Conv1d(16, 16, 3, bias=False, dtype=torch.bfloat16)
    module.conv2d = nn.Conv2d(16, 16, 3, bias=False, dtype=torch.bfloat16)
    module.conv3d = nn.Conv3d(8, 8, 3, bias=False, dtype=torch.bfloat16)

    audit = _audit(module)
    groups = {spec.name: spec.group for spec in audit.specs}

    assert groups["linear.weight"] == "matrix_decay"
    assert groups["conv1d.weight"] == "matrix_decay"
    assert groups["conv2d.weight"] == "matrix_decay"
    assert groups["conv3d.weight"] == "matrix_decay"


def test_conv_bias_is_sensitive_no_decay() -> None:
    module = nn.Module()
    module.conv = _conv2d_bf16_bias_fp32()

    audit = _audit(module)
    groups = {spec.name: spec.group for spec in audit.specs}
    dtypes = {
        spec.name: spec.parameter.dtype for spec in audit.specs
    }

    assert groups["conv.weight"] == "matrix_decay"
    assert groups["conv.bias"] == "sensitive_no_decay"
    assert dtypes["conv.weight"] is torch.bfloat16
    assert dtypes["conv.bias"] is torch.float32


def test_mixed_precision_conv2d_satisfies_parameter_audit() -> None:
    module = nn.Module()
    module.conv = MixedPrecisionConv2d(16, 16, 3)

    audit = _audit(module)
    groups = {spec.name: spec.group for spec in audit.specs}
    dtypes = {spec.name: spec.parameter.dtype for spec in audit.specs}

    assert groups["conv.weight"] == "matrix_decay"
    assert groups["conv.bias"] == "sensitive_no_decay"
    assert dtypes["conv.weight"] is torch.bfloat16
    assert dtypes["conv.bias"] is torch.float32


def test_audit_rejects_fp32_conv_weight() -> None:
    module = nn.Conv2d(16, 16, 3, bias=False, dtype=torch.float32)

    with pytest.raises(TypeError, match="requires torch.bfloat16"):
        _audit(module)


def test_audit_rejects_bf16_conv_bias() -> None:
    module = nn.Conv2d(16, 16, 3, dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="requires torch.float32"):
        _audit(module)


def test_unknown_2d_parameter_still_fails_closed() -> None:
    module = nn.Module()
    module.register_parameter(
        "matrix_like",
        nn.Parameter(torch.zeros(8, 16, dtype=torch.bfloat16)),
    )

    with pytest.raises(TypeError, match="no locked module role"):
        _audit(module)


def test_unknown_3d_parameter_still_fails_closed() -> None:
    module = nn.Module()
    module.register_parameter(
        "tensor_like",
        nn.Parameter(torch.zeros(4, 8, 16, dtype=torch.bfloat16)),
    )

    with pytest.raises(TypeError, match="no locked module role"):
        _audit(module)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)


class _Slot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _Attention()


class _Blocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.slot_0 = _Slot()


class _DiTWrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = _Blocks()
        self.conv = MixedPrecisionConv2d(16, 16, 3)


class _RoutingRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = _DiTWrap()


def test_conv_parameters_route_to_adamw_not_cmuon() -> None:
    module = _RoutingRoot()

    routing = route_cmuon_parameters(
        module,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
    )

    assert routing.cmuon_names == ("dit.blocks.slot_0.attention.q_proj.weight",)
    assert routing.cmuon_specs[0].role == "attention_q"
    assert "dit.conv.weight" in routing.adamw_names
    assert "dit.conv.bias" in routing.adamw_names
    assert "dit.conv.weight" not in routing.cmuon_names
    assert "dit.conv.bias" not in routing.cmuon_names
    assert set(routing.cmuon_names) | set(routing.adamw_names) == {
        "dit.blocks.slot_0.attention.q_proj.weight",
        "dit.conv.weight",
        "dit.conv.bias",
    }
    audit_groups = {
        spec.name: spec.group for spec in routing.full_audit.specs
    }
    assert audit_groups["dit.conv.weight"] == "matrix_decay"
    assert audit_groups["dit.conv.bias"] == "sensitive_no_decay"


def _production_dit_root() -> nn.Module:
    with torch.device("meta"):
        dit = PackedDiT(
            depth=16,
            input_channels=128,
            hidden_size=2560,
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

    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = dit

    return _Root()


def test_production_dit_cmuon_allowlist_set_is_unchanged() -> None:
    routing = route_cmuon_parameters(
        _production_dit_root(),
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
    )
    manifest = routing.routing_manifest()

    # Baseline pinned before the Conv policy generalization: the canonical
    # FQN allowlist selects exactly these parameters (no Conv exists in the
    # current production DiT, so the set must be byte-identical).
    assert manifest["counts"] == {"total": 208, "cmuon": 113, "adamw": 95}
    roles: dict[str, int] = {}
    for spec in routing.cmuon_specs:
        roles[spec.role] = roles.get(spec.role, 0) + 1
    assert roles == {
        "adaln_shared": 1,
        "attention_content_gate": 16,
        "attention_k": 16,
        "attention_out": 16,
        "attention_q": 16,
        "attention_v": 16,
        "ffn_down": 16,
        "ffn_in": 16,
    }
