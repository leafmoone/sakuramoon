"""Lazy public API for bounded fault-injection workers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_EXPORTS = {
    "ALL_SCENARIOS": ("sakuramoon.fault_injection.schema", "ALL_SCENARIOS"),
    "CPU_SCENARIOS": ("sakuramoon.fault_injection.schema", "CPU_SCENARIOS"),
    "ExecutedFaultEvidence": (
        "sakuramoon.fault_injection.evidence",
        "ExecutedFaultEvidence",
    ),
    "FOUR_GPU_SCENARIOS": (
        "sakuramoon.fault_injection.schema",
        "FOUR_GPU_SCENARIOS",
    ),
    "FaultMatrixReport": (
        "sakuramoon.fault_injection.schema",
        "FaultMatrixReport",
    ),
    "FaultOutcome": ("sakuramoon.fault_injection.schema", "FaultOutcome"),
    "FaultProcessError": (
        "sakuramoon.fault_injection.driver",
        "FaultProcessError",
    ),
    "FaultScenario": ("sakuramoon.fault_injection.schema", "FaultScenario"),
    "FaultStatus": ("sakuramoon.fault_injection.schema", "FaultStatus"),
    "HardwareLevel": ("sakuramoon.fault_injection.schema", "HardwareLevel"),
    "ONE_GPU_SCENARIOS": (
        "sakuramoon.fault_injection.schema",
        "ONE_GPU_SCENARIOS",
    ),
    "ProcessFaultEvidence": (
        "sakuramoon.fault_injection.driver",
        "ProcessFaultEvidence",
    ),
    "ReplayEvidence": ("sakuramoon.fault_injection.schema", "ReplayEvidence"),
    "ResumeParent": ("sakuramoon.fault_injection.recovery", "ResumeParent"),
    "TrainingControlSnapshot": (
        "sakuramoon.fault_injection.schema",
        "TrainingControlSnapshot",
    ),
    "load_executed_fault_evidence": (
        "sakuramoon.fault_injection.evidence",
        "load_executed_fault_evidence",
    ),
    "publish_fault_matrix_from_evidence": (
        "sakuramoon.fault_injection.evidence",
        "publish_fault_matrix_from_evidence",
    ),
    "run_expected_exit": (
        "sakuramoon.fault_injection.driver",
        "run_expected_exit",
    ),
    "run_until_ready_and_sigkill": (
        "sakuramoon.fault_injection.driver",
        "run_until_ready_and_sigkill",
    ),
    "select_complete_raw_parent": (
        "sakuramoon.fault_injection.recovery",
        "select_complete_raw_parent",
    ),
    "signal_ready_from_environment": (
        "sakuramoon.fault_injection.driver",
        "signal_ready_from_environment",
    ),
    "write_executed_fault_evidence": (
        "sakuramoon.fault_injection.evidence",
        "write_executed_fault_evidence",
    ),
    "write_fault_matrix": (
        "sakuramoon.fault_injection.report",
        "write_fault_matrix",
    ),
}

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


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
