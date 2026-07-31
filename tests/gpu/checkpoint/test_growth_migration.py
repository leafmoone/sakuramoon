from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sakuramoon.checkpoint import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    load_raw_checkpoint,
    save_raw_checkpoint,
)
from sakuramoon.checkpoint.migrate import (
    load_and_migrate_growth,
    migrate_loaded_growth,
)
from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import (
    active_slot_ids,
    half_cosine_growth_alpha,
    is_new_slot_fqn,
    new_slot_fqn_prefixes,
)
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit
from sakuramoon.train.stage import GrowthProgress, StageTransitionRequest
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_RESOLVED_CONFIG = b'[run]\nname = "t044-growth"\n'
_CONFIG_SHA256 = hashlib.sha256(_RESOLVED_CONFIG).hexdigest()


def _composite(depth: int) -> TrainableComposite:
    return TrainableComposite(
        dit=DenseDiT(  # pyright: ignore[reportArgumentType]
            depth=depth,
            input_channels=128,
            hidden_size=256,
            intermediate_size=256,
            q_heads=8,
            kv_heads=2,
            head_dim=32,
            rope_nope_dim=8,
            rope_y_dim=12,
            rope_x_dim=12,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=512,
            out_channels=128,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        ),
        text=TextConditioner(
            input_size=256,
            adapter_size=256,
            output_size=256,
            groups=8,
            attention_heads=8,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
        style=StyleResampler(
            input_size=256,
            hidden_size=256,
            intermediate_size=256,
            output_size=256,
            query_count=4,
            attention_heads=8,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
    ).cuda()


def _optimizer(module: TrainableComposite, seed: int) -> IsolatedAdamW8bit:
    return build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=seed,
    )


def _identity(name: str, update: int, optimizer: IsolatedAdamW8bit) -> CheckpointIdentity:
    return CheckpointIdentity(
        checkpoint_id=name,
        update=update,
        config_sha256=_CONFIG_SHA256,
        dependency_sha256="b" * 64,
        parameter_schema_sha256=optimizer.audit.schema_sha256,
    )


def _source_state(depth: int, stage: str) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(1, 1, 4),
        growth=GrowthCheckpointState(
            active_slot_ids(depth),
            1.0,
            stage,
            4,
            256 if stage in {"S0", "S1", "G1"} else 512,
            None,
            None,
        ),
    )


def _assert_loaded_growth_point(
    root: Path,
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    state: RawCheckpointState,
    elapsed_updates: int,
    target_depth: int,
) -> None:
    progress = GrowthProgress.from_checkpoint(state)
    update = progress.start_successful_update + elapsed_updates
    alpha = progress.alpha(update)
    point = replace(
        state,
        trainer=SingleGpuUpdateState(update, update, state.trainer.effective_samples),
        growth=replace(state.growth, alpha=alpha),
    )
    identity = _identity(f"alpha-{elapsed_updates}", update, optimizer)
    result = save_raw_checkpoint(
        root, identity, module, optimizer, point,
        resolved_config=_RESOLVED_CONFIG,
    )
    restored = _composite(target_depth)
    restored_optimizer = _optimizer(restored, 999)
    loaded = load_raw_checkpoint(result.path, restored, restored_optimizer, identity)
    assert loaded == point
    assert GrowthProgress.from_checkpoint(loaded).alpha(update + 1) == progress.alpha(
        update + 1
    )
    assert restored_optimizer.audit_state() == optimizer.audit_state()
    for left, right in zip(
        restored.parameters(), module.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def _assert_optimizer_state_equal(left: dict[object, object], right: dict[object, object]) -> None:
    assert set(left) == set(right)
    for key, left_value in left.items():
        right_value = right[key]
        assert isinstance(left_value, torch.Tensor)
        assert isinstance(right_value, torch.Tensor)
        assert type(left_value) is type(right_value)
        if type(left_value).__name__ == "OptimState8bit":
            for attribute in ("codes", "scale", "qmap"):
                torch.testing.assert_close(
                    getattr(left_value, attribute),
                    getattr(right_value, attribute),
                    atol=0,
                    rtol=0,
                )
            assert left_value.signed == right_value.signed  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
        else:
            torch.testing.assert_close(left_value, right_value, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("source_depth", "target_depth", "source_stage", "target_stage"),
    [
        (16, 20, "S1", "G1"),
        (20, 24, "S2", "G2"),
    ],
)
def test_raw_growth_migration_preserves_old_state_and_restores_growth_points(
    tmp_path: Path,
    source_depth: int,
    target_depth: int,
    source_stage: str,
    target_stage: str,
) -> None:
    torch.manual_seed(4301)  # pyright: ignore[reportUnknownMemberType]
    source = _composite(source_depth)
    source_optimizer = _optimizer(source, 4302)
    initialized_names = (
        "dit.input_projection.weight",
        "style.null_tokens",
    )
    source_specs = {spec.name: spec for spec in source_optimizer.audit.specs}
    for name in initialized_names:
        source_specs[name].parameter.grad = torch.ones_like(source_specs[name].parameter)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    source_identity = _identity("source", 1, source_optimizer)
    source_checkpoint = save_raw_checkpoint(
        tmp_path / "source",
        source_identity,
        source,
        source_optimizer,
        _source_state(source_depth, source_stage),
        resolved_config=_RESOLVED_CONFIG,
    ).path
    source_parameters = {
        name: parameter.detach().clone() for name, parameter in source.named_parameters()
    }
    source_audit = {spec.name: spec for spec in source_optimizer.audit_state()}
    source_sr = source_optimizer.sr_rng.state.clone()

    torch.manual_seed(4303)  # pyright: ignore[reportUnknownMemberType]
    target = _composite(target_depth)
    target_optimizer = _optimizer(target, 4304)
    prefixes = new_slot_fqn_prefixes(target_depth)
    random_new = {
        name: parameter.detach().clone()
        for name, parameter in target.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    request = StageTransitionRequest(
        source_stage=source_stage,
        target_stage=target_stage,
        source_checkpoint=source_checkpoint,
        source_checkpoint_id="source",
        planned_updates=50_000,
        manual_approval=True,
    )

    rejected_target = _composite(target_depth)
    rejected_optimizer = _optimizer(rejected_target, 4306)
    rejected_parameters = tuple(
        parameter.detach().clone() for parameter in rejected_target.parameters()
    )
    rejected_sr = rejected_optimizer.sr_rng.state.clone()
    with pytest.raises(ValueError, match="completed growth ramp"):
        migrate_loaded_growth(
            source,
            source_optimizer,
            RawCheckpointState(
                trainer=SingleGpuUpdateState(501, 501, 4),
                growth=GrowthCheckpointState(
                    active_slot_ids(source_depth),
                    half_cosine_growth_alpha(500, 1000),
                    source_stage,
                    4,
                    256 if source_stage in {"S1", "G1"} else 512,
                    1,
                    1000,
                ),
            ),
            rejected_target,
            rejected_optimizer,
            request,
        )
    assert not rejected_optimizer.optimizer.state
    assert torch.equal(rejected_optimizer.sr_rng.state, rejected_sr)
    for parameter, before in zip(
        rejected_target.parameters(), rejected_parameters, strict=True
    ):
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)

    state, report = load_and_migrate_growth(
        source_checkpoint,
        source_identity,
        target,
        target_optimizer,
        request,
    )

    assert report.source_depth == source_depth
    assert report.target_depth == target_depth
    assert set(report.preserved_optimizer_fqns) == set(initialized_names)
    assert state.trainer == _source_state(source_depth, source_stage).trainer
    assert state.growth == GrowthCheckpointState(
        active_slot_ids(target_depth),
        0.0,
        target_stage,
        4,
        256 if target_stage == "G1" else 512,
        1,
        1000,
    )
    target_parameters = dict(target.named_parameters())
    for name, expected in source_parameters.items():
        torch.testing.assert_close(target_parameters[name], expected, atol=0, rtol=0)
    for name, expected in random_new.items():
        torch.testing.assert_close(target_parameters[name], expected, atol=0, rtol=0)
    target_audit = {spec.name: spec for spec in target_optimizer.audit_state()}
    for name in initialized_names:
        assert target_audit[name] == source_audit[name]
        _assert_optimizer_state_equal(
            source_optimizer.optimizer.state[source_specs[name].parameter],
            target_optimizer.optimizer.state[
                {spec.name: spec for spec in target_optimizer.audit.specs}[name].parameter
            ],
        )
    assert all(not target_audit[name].initialized for name in target_audit if any(name.startswith(prefix) for prefix in prefixes))
    assert torch.equal(target_optimizer.sr_rng.state, source_sr)

    _assert_loaded_growth_point(
        tmp_path / "post", target, target_optimizer, state, 0, target_depth
    )
    _assert_loaded_growth_point(
        tmp_path / "mid", target, target_optimizer, state, 500, target_depth
    )
    _assert_loaded_growth_point(
        tmp_path / "end", target, target_optimizer, state, 1000, target_depth
    )


def _assert_unchanged(
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    parameters: tuple[torch.Tensor, ...],
    sr_state: torch.Tensor,
) -> None:
    assert not optimizer.optimizer.state
    assert torch.equal(optimizer.sr_rng.state, sr_state)
    for parameter, before in zip(module.parameters(), parameters, strict=True):
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)


@pytest.mark.parametrize("detached_side", ["source", "target"])
def test_growth_migration_rejects_optimizer_owned_by_another_module(
    tmp_path: Path, detached_side: str
) -> None:
    source = _composite(16)
    target = _composite(20)
    source_optimizer = _optimizer(source, 4401)
    target_optimizer = _optimizer(target, 4402)
    if detached_side == "source":
        source_optimizer = _optimizer(_composite(16), 4403)
    else:
        target_optimizer = _optimizer(_composite(20), 4404)
    parameters = tuple(parameter.detach().clone() for parameter in target.parameters())
    sr_state = target_optimizer.sr_rng.state.clone()
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    request = StageTransitionRequest("S1", "G1", checkpoint, "source", 50_000, True)

    with pytest.raises(ValueError, match="canonical parameters differ"):
        migrate_loaded_growth(
            source,
            source_optimizer,
            _source_state(16, "S1"),
            target,
            target_optimizer,
            request,
        )

    _assert_unchanged(target, target_optimizer, parameters, sr_state)


def test_growth_migration_rejects_new_slot_prefix_collision(tmp_path: Path) -> None:
    source = _composite(16)
    source_optimizer = _optimizer(source, 4501)
    target = _composite(20)
    bias = target.dit.conditioner.block_biases["slot_02"]
    target.dit.conditioner.block_biases["slot_02evil"] = torch.nn.Parameter(
        torch.zeros_like(bias)
    )
    target_optimizer = _optimizer(target, 4502)
    parameters = tuple(parameter.detach().clone() for parameter in target.parameters())
    sr_state = target_optimizer.sr_rng.state.clone()
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    request = StageTransitionRequest("S1", "G1", checkpoint, "source", 50_000, True)

    assert not is_new_slot_fqn(20, "dit.conditioner.block_biases.slot_02evil")
    with pytest.raises(ValueError, match="new-slot allowlist"):
        migrate_loaded_growth(
            source,
            source_optimizer,
            _source_state(16, "S1"),
            target,
            target_optimizer,
            request,
        )

    _assert_unchanged(target, target_optimizer, parameters, sr_state)
