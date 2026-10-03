"""Public exports, loaded on demand so inference needs no training extras."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
    from sakuramoon.train.loop import (
        LoopResult,
        SingleGpuTrainingLoop,
        SuccessfulLoopObservation,
    )
    from sakuramoon.train.preflight import (
        AcceptedPreflight,
        PreflightError,
        ProductionSingleGpuCheckpointPublisher,
        RestoredSingleGpuCheckpoint,
        build_single_gpu_preflight_checks,
        require_accepted_preflight,
        restore_single_gpu_checkpoint,
        run_single_gpu_preflight,
    )
    from sakuramoon.train.runtime import (
        DenseDiTAdapter,
        PreparedTrainingBatch,
        RuntimeMeasurement,
        SingleGpuBatchRuntime,
        SuccessfulTrainingObservation,
        require_train_topology,
        run_single_gpu_training,
    )
    from sakuramoon.train.scheduler import CheckpointDecision, CheckpointScheduler
    from sakuramoon.train.step import (
        SingleGpuStep,
        SingleGpuUpdateResult,
        SingleGpuUpdateState,
        StepOptimizer,
        TrainableComposite,
        TrainableCompositeInputs,
    )

_EXPORT_MODULES = {
    "FailureSnapshot": "sakuramoon.train.failures",
    "write_failure_bundle": "sakuramoon.train.failures",
    "LoopResult": "sakuramoon.train.loop",
    "SingleGpuTrainingLoop": "sakuramoon.train.loop",
    "SuccessfulLoopObservation": "sakuramoon.train.loop",
    "AcceptedPreflight": "sakuramoon.train.preflight",
    "PreflightError": "sakuramoon.train.preflight",
    "ProductionSingleGpuCheckpointPublisher": "sakuramoon.train.preflight",
    "RestoredSingleGpuCheckpoint": "sakuramoon.train.preflight",
    "build_single_gpu_preflight_checks": "sakuramoon.train.preflight",
    "require_accepted_preflight": "sakuramoon.train.preflight",
    "restore_single_gpu_checkpoint": "sakuramoon.train.preflight",
    "run_single_gpu_preflight": "sakuramoon.train.preflight",
    "DenseDiTAdapter": "sakuramoon.train.runtime",
    "PreparedTrainingBatch": "sakuramoon.train.runtime",
    "RuntimeMeasurement": "sakuramoon.train.runtime",
    "SingleGpuBatchRuntime": "sakuramoon.train.runtime",
    "SuccessfulTrainingObservation": "sakuramoon.train.runtime",
    "require_train_topology": "sakuramoon.train.runtime",
    "run_single_gpu_training": "sakuramoon.train.runtime",
    "CheckpointDecision": "sakuramoon.train.scheduler",
    "CheckpointScheduler": "sakuramoon.train.scheduler",
    "SingleGpuStep": "sakuramoon.train.step",
    "SingleGpuUpdateResult": "sakuramoon.train.step",
    "SingleGpuUpdateState": "sakuramoon.train.step",
    "StepOptimizer": "sakuramoon.train.step",
    "TrainableComposite": "sakuramoon.train.step",
    "TrainableCompositeInputs": "sakuramoon.train.step",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "AcceptedPreflight",
    "CheckpointDecision",
    "CheckpointScheduler",
    "DenseDiTAdapter",
    "FailureSnapshot",
    "LoopResult",
    "PreflightError",
    "PreparedTrainingBatch",
    "ProductionSingleGpuCheckpointPublisher",
    "RestoredSingleGpuCheckpoint",
    "RuntimeMeasurement",
    "SingleGpuBatchRuntime",
    "SingleGpuStep",
    "SingleGpuTrainingLoop",
    "SingleGpuUpdateResult",
    "SingleGpuUpdateState",
    "StepOptimizer",
    "SuccessfulLoopObservation",
    "SuccessfulTrainingObservation",
    "TrainableComposite",
    "TrainableCompositeInputs",
    "build_single_gpu_preflight_checks",
    "require_accepted_preflight",
    "require_train_topology",
    "restore_single_gpu_checkpoint",
    "run_single_gpu_preflight",
    "run_single_gpu_training",
    "write_failure_bundle",
]
