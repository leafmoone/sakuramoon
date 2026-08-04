"""Single-GPU training primitives."""
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
    require_single_gpu_config,
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
    "require_single_gpu_config",
    "restore_single_gpu_checkpoint",
    "run_single_gpu_preflight",
    "run_single_gpu_training",
    "write_failure_bundle",
]
