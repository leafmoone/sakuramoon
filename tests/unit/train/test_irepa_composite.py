from __future__ import annotations

import pytest
import torch

from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import PackedDiT
from sakuramoon.model.irepa import IRepaAlignment
from sakuramoon.train.step import TrainableComposite

HIDDEN_SIZE = 2560


def _production_dit(depth: int = 16) -> PackedDiT:
    with torch.device("meta"):
        return PackedDiT(
            depth=depth,
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


def test_none_auxiliary_keeps_legacy_children_fqns_and_state_dict() -> None:
    default = _composite()
    explicit_none = _composite(irepa=None)

    assert default.irepa_alignment is None
    assert explicit_none.irepa_alignment is None
    legacy_children = {"dit", "text", "condition_tokens"}
    assert set(dict(default.named_children())) == legacy_children
    assert set(dict(explicit_none.named_children())) == legacy_children

    legacy_fqns = [name for name, _ in default.named_parameters()]
    assert [name for name, _ in explicit_none.named_parameters()] == legacy_fqns
    assert all(name.split(".", 1)[0] in legacy_children for name in legacy_fqns)
    assert list(default.state_dict()) == list(explicit_none.state_dict())


def test_enabled_auxiliary_becomes_canonical_trainable_child() -> None:
    legacy = _composite()
    irepa = _irepa()
    module = _composite(irepa)

    assert module.irepa_alignment is irepa
    assert set(dict(module.named_children())) == {
        "dit",
        "text",
        "condition_tokens",
        "irepa_alignment",
    }
    legacy_fqns = [name for name, _ in legacy.named_parameters()]
    new_fqns = [name for name, _ in module.named_parameters()]
    assert new_fqns[: len(legacy_fqns)] == legacy_fqns
    assert set(new_fqns) == set(legacy_fqns) | {
        "irepa_alignment.projector.weight",
        "irepa_alignment.projector.bias",
    }
    states = dict(module.state_dict())
    assert "irepa_alignment.projector.weight" in states
    assert "irepa_alignment.projector.bias" in states
    assert len(states) == len(legacy.state_dict()) + 2


def test_no_parameter_aliasing_with_auxiliary() -> None:
    module = _composite(_irepa())

    parameters = tuple(module.named_parameters(remove_duplicate=False))
    identities = [id(parameter) for _, parameter in parameters]
    assert len(set(identities)) == len(identities)


def test_tap_slot_requires_an_installed_alignment() -> None:
    with pytest.raises(ValueError, match="requires an installed irepa_alignment"):
        _composite_tapped(irepa=None, tap=8)


def test_tap_slot_must_be_active_at_the_current_depth() -> None:
    # depth 16 has no slot 8 (G1 growth slot); depth 20 does
    with (
        pytest.raises(ValueError, match="not an active stable slot"),
        torch.device("meta"),
    ):
        TrainableComposite(
            dit=_production_dit(depth=16),
            text=_production_text(),
            condition_tokens=_production_condition_tokens(),
            irepa_alignment=_irepa(),
            irepa_tap_slot_id=8,
        )
    module = _composite_tapped(irepa=_irepa(), tap=8, depth=20)
    assert module.irepa_tap_slot_id == 8
    # an active but different slot is accepted verbatim
    module = _composite_tapped(irepa=_irepa(), tap=0, depth=16)
    assert module.irepa_tap_slot_id == 0
    with pytest.raises(ValueError, match="not an active stable slot"):
        _composite_tapped(irepa=_irepa(), tap=11, depth=16)
    with pytest.raises(ValueError, match="must be an int"):
        _composite_tapped(irepa=_irepa(), tap="8", depth=20)


def test_tap_slot_is_a_plain_runtime_attribute() -> None:
    module = _composite_tapped(irepa=_irepa(), tap=8, depth=20)

    assert module.irepa_tap_slot_id == 8
    assert type(module.irepa_tap_slot_id) is int
    # runtime binding only: never a child module, never checkpointed
    assert "irepa_tap_slot_id" not in dict(module.named_children())
    assert all(not name.startswith("irepa_tap") for name in module.state_dict())


def _composite_tapped(
    *,
    irepa: IRepaAlignment | None,
    tap: object,
    depth: int = 16,
) -> TrainableComposite:
    with torch.device("meta"):
        return TrainableComposite(
            dit=_production_dit(depth=depth),
            text=_production_text(),
            condition_tokens=_production_condition_tokens(),
            irepa_alignment=irepa,
            irepa_tap_slot_id=tap,  # type: ignore[arg-type]
        )


def test_bind_irepa_tap_slot_artifact_first_contract() -> None:
    # artifact-first construction leaves the tap unbound
    module = _composite(_irepa())
    assert module.irepa_tap_slot_id is None

    # binding an inactive slot is rejected before any mutation
    with pytest.raises(ValueError, match="not an active stable slot"):
        module.bind_irepa_tap_slot(8)  # depth 16: slot 8 inactive
    assert module.irepa_tap_slot_id is None

    # an active slot binds; rebinding is explicit and idempotent
    module.bind_irepa_tap_slot(0)
    assert module.irepa_tap_slot_id == 0
    module.bind_irepa_tap_slot(0)
    assert module.irepa_tap_slot_id == 0
    module.bind_irepa_tap_slot(16)
    assert module.irepa_tap_slot_id == 16
    # the tap stays outside the checkpoint contract after binding
    assert all(not name.startswith("irepa_tap") for name in module.state_dict())

    # no projector -> binding fails closed
    bare = _composite()
    with pytest.raises(ValueError, match="requires an installed projector"):
        bare.bind_irepa_tap_slot(0)
    with pytest.raises(ValueError, match="must be an int"):
        module.bind_irepa_tap_slot(0.0)  # type: ignore[arg-type]
