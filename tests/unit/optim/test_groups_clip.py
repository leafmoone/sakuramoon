from __future__ import annotations

import pytest
import torch
from torch import nn

from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.optim.clip import clip_grad_norm_fp32
from sakuramoon.optim.groups import audit_trainable_parameters


class _PolicyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.norm = nn.Parameter(torch.ones(64, dtype=torch.float32))


class _TrainableComposite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = _production_model()
        self.text = TextConditioner(
            input_size=2048,
            adapter_size=1024,
            output_size=2560,
            groups=8,
            attention_heads=16,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        )
        self.style = StyleResampler(
            input_size=2048,
            hidden_size=1024,
            intermediate_size=2048,
            output_size=2560,
            query_count=4,
            attention_heads=16,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        )


def _production_model() -> PackedDiT:
    return PackedDiT(
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
        modality_init_std=0.02,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    )


def test_parameter_audit_is_canonical_and_assigns_exact_decay() -> None:
    module = _PolicyModule()
    audit = audit_trainable_parameters(
        module,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )

    assert tuple(spec.name for spec in audit.specs) == tuple(
        sorted(spec.name for spec in audit.specs)
    )
    assert tuple(spec.name for spec in audit.decay) == ("matrix.weight",)
    assert {spec.name for spec in audit.sensitive} == {"norm"}
    assert {spec.weight_decay for spec in audit.decay} == {0.01}
    assert {spec.weight_decay for spec in audit.sensitive} == {0.0}


def test_full_16l_production_parameter_schema_is_locked() -> None:
    with torch.device("meta"):
        model = _production_model()
    audit = audit_trainable_parameters(
        model,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )

    assert len(audit.specs) == 206
    assert len(audit.decay) == 113
    assert len(audit.sensitive) == 93
    assert sum(spec.parameter.numel() for spec in audit.decay) == 1_216_675_840
    assert sum(spec.parameter.numel() for spec in audit.sensitive) == 23_090_304


def test_full_trainable_composite_uses_role_based_precision_groups() -> None:
    with torch.device("meta"):
        composite = _TrainableComposite()
    audit = audit_trainable_parameters(
        composite,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )
    specs = {spec.name: spec for spec in audit.specs}

    assert specs["text.shared_projection.weight"].group == "matrix_decay"
    assert specs["text.gate_weight"].group == "sensitive_no_decay"
    assert specs["style.cross_attention.in_proj_weight"].group == "matrix_decay"
    assert specs["style.queries"].group == "sensitive_no_decay"
    assert specs["style.layer_embedding"].group == "sensitive_no_decay"
    assert specs["style.null_tokens"].group == "sensitive_no_decay"
    assert specs["dit.conditioner.condition_mlp.0.weight"].group == (
        "sensitive_no_decay"
    )
    assert specs["dit.output_head.projection.weight"].group == (
        "sensitive_no_decay"
    )
    assert len(audit.specs) == 239
    assert len(audit.decay) == 126
    assert len(audit.sensitive) == 113
    assert sum(spec.parameter.numel() for spec in audit.decay) == 1_240_793_088
    assert sum(spec.parameter.numel() for spec in audit.sensitive) == 23_145_786


def test_parameter_audit_rejects_fp32_matrix_projection() -> None:
    module = nn.Linear(64, 64, bias=False, dtype=torch.float32)

    with pytest.raises(TypeError, match="requires torch.bfloat16"):
        audit_trainable_parameters(
            module,
            matrix_weight_decay=0.01,
            sensitive_weight_decay=0.0,
        )


def test_parameter_audit_rejects_unknown_ranked_gate_role() -> None:
    module = nn.Module()
    module.register_parameter(
        "gate_weight",
        nn.Parameter(torch.zeros(7, 8, 128, dtype=torch.bfloat16)),
    )

    with pytest.raises(TypeError, match="no locked module role"):
        audit_trainable_parameters(
            module,
            matrix_weight_decay=0.01,
            sensitive_weight_decay=0.0,
        )


def test_parameter_audit_rejects_aliased_modules_by_fqn() -> None:
    module = nn.Module()
    shared = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    module.add_module("first", shared)
    module.add_module("second", shared)

    with pytest.raises(ValueError, match="aliased"):
        audit_trainable_parameters(
            module,
            matrix_weight_decay=0.01,
            sensitive_weight_decay=0.0,
        )


def test_parameter_audit_rejects_bf16_sensitive_vectors() -> None:
    module = nn.Module()
    module.register_parameter("gate", nn.Parameter(torch.ones(8, dtype=torch.bfloat16)))

    with pytest.raises(TypeError, match="requires torch.float32"):
        audit_trainable_parameters(
            module,
            matrix_weight_decay=0.01,
            sensitive_weight_decay=0.0,
        )


def test_fp32_global_clip_scales_mixed_dtype_gradients() -> None:
    first = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    second = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    first.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    second.grad = torch.tensor([12.0], dtype=torch.float32)

    result = clip_grad_norm_fp32((first, second), max_norm=1.0)

    torch.testing.assert_close(result.pre_clip_norm, torch.tensor(13.0))
    torch.testing.assert_close(result.post_clip_norm, torch.tensor(1.0))
    torch.testing.assert_close(second.grad, torch.tensor([12.0 / 13.0]))


def test_fp32_global_clip_rejects_nonfinite_before_mutation() -> None:
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1.0, float("nan")])
    before = parameter.grad.clone()

    with pytest.raises(FloatingPointError, match="nonfinite"):
        clip_grad_norm_fp32((parameter,), max_norm=1.0)

    torch.testing.assert_close(parameter.grad, before, equal_nan=True)
