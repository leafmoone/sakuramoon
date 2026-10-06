"""Public exports, loaded on demand so inference needs no training extras."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sakuramoon.checkpoint.load import (
        discover_complete_checkpoints,
        load_inference_artifact,
        load_model_directory,
        load_model_only,
        load_raw_checkpoint,
    )
    from sakuramoon.checkpoint.policy import (
        FORCED_CHECKPOINT_REASONS,
        CheckpointCadence,
        CheckpointReason,
        RawRetentionPlan,
        apply_raw_retention,
        plan_raw_retention,
    )
    from sakuramoon.checkpoint.save import (
        save_model_only,
        save_raw_checkpoint,
    )
    from sakuramoon.checkpoint.schema import (
        CheckpointIdentity,
        CheckpointKind,
        CheckpointSaveResult,
        GrowthCheckpointState,
        RawCheckpointState,
        StageBudgetCheckpointState,
    )

_EXPORT_MODULES = {
    "discover_complete_checkpoints": "sakuramoon.checkpoint.load",
    "load_inference_artifact": "sakuramoon.checkpoint.load",
    "load_model_directory": "sakuramoon.checkpoint.load",
    "load_model_only": "sakuramoon.checkpoint.load",
    "load_raw_checkpoint": "sakuramoon.checkpoint.load",
    "FORCED_CHECKPOINT_REASONS": "sakuramoon.checkpoint.policy",
    "CheckpointCadence": "sakuramoon.checkpoint.policy",
    "CheckpointReason": "sakuramoon.checkpoint.policy",
    "RawRetentionPlan": "sakuramoon.checkpoint.policy",
    "apply_raw_retention": "sakuramoon.checkpoint.policy",
    "plan_raw_retention": "sakuramoon.checkpoint.policy",
    "save_model_only": "sakuramoon.checkpoint.save",
    "save_raw_checkpoint": "sakuramoon.checkpoint.save",
    "CheckpointIdentity": "sakuramoon.checkpoint.schema",
    "CheckpointKind": "sakuramoon.checkpoint.schema",
    "CheckpointSaveResult": "sakuramoon.checkpoint.schema",
    "GrowthCheckpointState": "sakuramoon.checkpoint.schema",
    "RawCheckpointState": "sakuramoon.checkpoint.schema",
    "StageBudgetCheckpointState": "sakuramoon.checkpoint.schema",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "FORCED_CHECKPOINT_REASONS",
    "CheckpointCadence",
    "CheckpointIdentity",
    "CheckpointKind",
    "CheckpointReason",
    "CheckpointSaveResult",
    "GrowthCheckpointState",
    "RawCheckpointState",
    "RawRetentionPlan",
    "StageBudgetCheckpointState",
    "apply_raw_retention",
    "discover_complete_checkpoints",
    "load_inference_artifact",
    "load_model_directory",
    "load_model_only",
    "load_raw_checkpoint",
    "plan_raw_retention",
    "save_model_only",
    "save_raw_checkpoint",
]
