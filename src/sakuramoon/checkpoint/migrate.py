"""Raw-only canonical-FQN migration for approved depth growth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torchao.optim.subclass_8bit import (  # pyright: ignore[reportMissingTypeStubs]
    OptimState8bit,
)

from sakuramoon.checkpoint.artifact import (
    active_slot_ids_from_module,
    validate_optimizer_coverage,
)
from sakuramoon.checkpoint.load import (
    load_inference_artifact,
    load_raw_checkpoint,
    read_checkpoint_manifest,
)
from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    RawCheckpointState,
)
from sakuramoon.model.growth import active_slot_ids, is_new_slot_fqn
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit
from sakuramoon.train.stage import (
    StageTransitionRequest,
    stage_spec,
    transition_checkpoint_state,
)
from sakuramoon.train.step import TrainableComposite


@dataclass(frozen=True, slots=True)
class GrowthMigrationReport:
    source_depth: int
    target_depth: int
    preserved_fqns: tuple[str, ...]
    new_fqns: tuple[str, ...]
    preserved_optimizer_fqns: tuple[str, ...]


def _depth(module: TrainableComposite) -> int:
    slots = active_slot_ids_from_module(module)
    for depth, expected in ((16, active_slot_ids(16)), (20, active_slot_ids(20)), (24, active_slot_ids(24))):
        if slots == expected:
            return depth
    raise ValueError("module does not use a canonical growth topology")


def _clone_optimizer_state(state: dict[object, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("optimizer state must use the locked string-to-tensor schema")
        if isinstance(value, OptimState8bit):
            cloned[key] = OptimState8bit(
                value.codes.clone(),
                value.scale.clone(),
                value.qmap.clone(),
                value.signed,
            )
        else:
            cloned[key] = value.detach().clone()
    return cloned


def migrate_loaded_growth(
    source_module: TrainableComposite,
    source_optimizer: IsolatedAdamW8bit,
    source_state: RawCheckpointState,
    target_module: TrainableComposite,
    target_optimizer: IsolatedAdamW8bit,
    request: StageTransitionRequest,
    *,
    checkpoint_cadence: CheckpointCadence,
) -> tuple[RawCheckpointState, GrowthMigrationReport]:
    source_depth = _depth(source_module)
    target_depth = _depth(target_module)
    if not request.is_growth or (
        source_depth,
        target_depth,
    ) not in {(16, 20), (20, 24)}:
        raise ValueError("growth migration only accepts 16->20 or 20->24")
    if target_depth != stage_spec(request.target_stage).depth:
        raise ValueError("target module depth differs from the transition request")
    if target_optimizer.optimizer.state:
        raise ValueError("target optimizer state must be empty before migration")
    validate_optimizer_coverage(
        source_module,
        tuple((spec.name, spec.parameter) for spec in source_optimizer.audit.specs),
    )
    validate_optimizer_coverage(
        target_module,
        tuple((spec.name, spec.parameter) for spec in target_optimizer.audit.specs),
    )
    source_optimizer.audit_state()
    target_state = transition_checkpoint_state(
        source_state,
        request,
        checkpoint_cadence=checkpoint_cadence,
    )

    source_tensors = source_module.state_dict()
    target_tensors = target_module.state_dict()
    preserved = tuple(sorted(set(source_tensors) & set(target_tensors)))
    removed = tuple(sorted(set(source_tensors) - set(target_tensors)))
    new = tuple(sorted(set(target_tensors) - set(source_tensors)))
    disallowed_new = tuple(
        name for name in new if not is_new_slot_fqn(target_depth, name)
    )
    if removed or not new or disallowed_new:
        raise ValueError(
            "growth model FQN delta differs from the new-slot allowlist: "
            f"removed={removed!r}, disallowed_new={disallowed_new!r}"
        )
    for name in preserved:
        if (
            source_tensors[name].shape != target_tensors[name].shape
            or source_tensors[name].dtype != target_tensors[name].dtype
        ):
            raise ValueError(f"preserved model tensor changed shape or dtype: {name}")
    source_specs = {spec.name: spec for spec in source_optimizer.audit.specs}
    target_specs = {spec.name: spec for spec in target_optimizer.audit.specs}
    if set(source_specs) - set(target_specs):
        raise ValueError("growth removed optimizer parameters")
    optimizer_new = tuple(sorted(set(target_specs) - set(source_specs)))
    if any(not is_new_slot_fqn(target_depth, name) for name in optimizer_new):
        raise ValueError("growth optimizer delta differs from the new-slot allowlist")
    cloned_optimizer_states: dict[str, dict[str, object]] = {}
    for name, source_spec in source_specs.items():
        target_spec = target_specs[name]
        if (
            source_spec.group != target_spec.group
            or source_spec.parameter.shape != target_spec.parameter.shape
            or source_spec.parameter.dtype != target_spec.parameter.dtype
        ):
            raise ValueError(f"preserved optimizer parameter policy changed: {name}")
        state = source_optimizer.optimizer.state.get(source_spec.parameter)
        if state:
            cloned_optimizer_states[name] = _clone_optimizer_state(
                cast(dict[object, object], state)
            )
    source_sr_state = source_optimizer.sr_rng.state_dict()

    with torch.no_grad():
        for name in preserved:
            target_tensors[name].copy_(source_tensors[name])
    for name, state in cloned_optimizer_states.items():
        target_optimizer.optimizer.state[target_specs[name].parameter] = state
    target_optimizer.sr_rng.load_state_dict(source_sr_state)

    return target_state, GrowthMigrationReport(
        source_depth=source_depth,
        target_depth=target_depth,
        preserved_fqns=preserved,
        new_fqns=new,
        preserved_optimizer_fqns=tuple(sorted(cloned_optimizer_states)),
    )


def load_and_migrate_growth(
    checkpoint: Path,
    expected: CheckpointIdentity,
    target_module: TrainableComposite,
    target_optimizer: IsolatedAdamW8bit,
    request: StageTransitionRequest,
    *,
    checkpoint_cadence: CheckpointCadence,
) -> tuple[RawCheckpointState, GrowthMigrationReport]:
    if (
        checkpoint.resolve(strict=True)
        != request.source_checkpoint.resolve(strict=True)
        or expected.checkpoint_id != request.source_checkpoint_id
    ):
        raise CheckpointError("growth source differs from the approved transition request")
    manifest = read_checkpoint_manifest(checkpoint)
    if manifest.kind is not CheckpointKind.RAW or manifest.identity != expected:
        raise CheckpointError("growth transition requires the exact raw predecessor")
    device = next(target_module.parameters()).device
    source = load_inference_artifact(checkpoint, expected, device=device)
    if not isinstance(source, TrainableComposite):
        raise CheckpointError("growth source is not a trainable composite")
    source_optimizer = build_adamw8bit(
        source,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=0,
    )
    source_state = load_raw_checkpoint(checkpoint, source, source_optimizer, expected)
    return migrate_loaded_growth(
        source,
        source_optimizer,
        source_state,
        target_module,
        target_optimizer,
        request,
        checkpoint_cadence=checkpoint_cadence,
    )


__all__ = [
    "GrowthMigrationReport",
    "load_and_migrate_growth",
    "migrate_loaded_growth",
]
