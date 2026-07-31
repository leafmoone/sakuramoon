"""Fault-injection control plane for bounded CPU and single-GPU tests."""

from sakuramoon.fault_injection.driver import (
    FaultProcessError,
    ProcessFaultEvidence,
    run_expected_exit,
    run_until_ready_and_sigkill,
    signal_ready_from_environment,
)
from sakuramoon.fault_injection.evidence import (
    ExecutedFaultEvidence,
    load_executed_fault_evidence,
    publish_fault_matrix_from_evidence,
    write_executed_fault_evidence,
)
from sakuramoon.fault_injection.recovery import (
    ResumeParent,
    select_complete_raw_parent,
)
from sakuramoon.fault_injection.report import write_fault_matrix
from sakuramoon.fault_injection.schema import (
    ALL_SCENARIOS,
    CPU_SCENARIOS,
    FOUR_GPU_SCENARIOS,
    ONE_GPU_SCENARIOS,
    FaultMatrixReport,
    FaultOutcome,
    FaultScenario,
    FaultStatus,
    HardwareLevel,
    ReplayEvidence,
    TrainingControlSnapshot,
)

__all__ = [
    "ALL_SCENARIOS",
    "CPU_SCENARIOS",
    "FOUR_GPU_SCENARIOS",
    "ONE_GPU_SCENARIOS",
    "ExecutedFaultEvidence",
    "FaultMatrixReport",
    "FaultOutcome",
    "FaultProcessError",
    "FaultScenario",
    "FaultStatus",
    "HardwareLevel",
    "ProcessFaultEvidence",
    "ReplayEvidence",
    "ResumeParent",
    "TrainingControlSnapshot",
    "load_executed_fault_evidence",
    "publish_fault_matrix_from_evidence",
    "run_expected_exit",
    "run_until_ready_and_sigkill",
    "select_complete_raw_parent",
    "signal_ready_from_environment",
    "write_executed_fault_evidence",
    "write_fault_matrix",
]
