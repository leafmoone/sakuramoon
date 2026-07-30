"""Fail-closed single-GPU training primitives."""

from sakuramoon.train.benchmark import (
    MeasuredMicrobatch,
    SingleGpuStepBenchmarkAdapter,
)
from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.loop import LoopResult, SingleGpuTrainingLoop
from sakuramoon.train.preflight import (
    PREFLIGHT_CHECKS,
    PreflightCheckResult,
    PreflightError,
    PreflightReport,
    run_single_gpu_preflight,
)
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateResult,
    SingleGpuUpdateState,
    StepOptimizer,
    TrainableComposite,
    TrainableCompositeInputs,
)

__all__ = [
    "PREFLIGHT_CHECKS",
    "FailureSnapshot",
    "LoopResult",
    "MeasuredMicrobatch",
    "PreflightCheckResult",
    "PreflightError",
    "PreflightReport",
    "SingleGpuStep",
    "SingleGpuStepBenchmarkAdapter",
    "SingleGpuTrainingLoop",
    "SingleGpuUpdateResult",
    "SingleGpuUpdateState",
    "StepOptimizer",
    "TrainableComposite",
    "TrainableCompositeInputs",
    "run_single_gpu_preflight",
    "write_failure_bundle",
]
