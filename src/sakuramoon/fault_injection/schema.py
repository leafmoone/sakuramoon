"""Strict evidence schema for the CPU and single-GPU fault matrix."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

_HEX64 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_EVIDENCE_FILE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,126}\.json")


class FaultScenario(StrEnum):
    DOWNLOAD_INTERRUPTION = "download_interruption"
    TRUNCATED_SHARD = "truncated_shard"
    TOKEN_EXPIRED = "token_expired"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    WORKER_EXIT = "worker_exit"
    DISK_FULL = "disk_full"
    MICROBATCH_SIGKILL = "microbatch_sigkill"
    OPTIMIZER_SIGKILL = "optimizer_sigkill"
    CHECKPOINT_SIGKILL = "checkpoint_sigkill"
    NONFINITE_LOSS = "nonfinite_loss"
    NONFINITE_GRADIENT = "nonfinite_gradient"
    CUDA_OOM = "cuda_oom"
    DDP_REDUCTION_SIGKILL = "ddp_reduction_sigkill"
    SR_RANK_DIVERGENCE = "sr_rank_divergence"
    NCCL_RANK_FAILURE = "nccl_rank_failure"
    ALL_RANK_STOP = "all_rank_stop"
    FOUR_RANK_CHECKPOINT_RECOVERY = "four_rank_checkpoint_recovery"


class FaultStatus(StrEnum):
    PASSED = "passed"
    BLOCKED = "blocked"


class HardwareLevel(StrEnum):
    CPU = "CPU"
    ONE_GPU = "1GPU"
    FOUR_GPU = "4GPU"


CPU_SCENARIOS = (
    FaultScenario.DOWNLOAD_INTERRUPTION,
    FaultScenario.TRUNCATED_SHARD,
    FaultScenario.TOKEN_EXPIRED,
    FaultScenario.CHECKSUM_MISMATCH,
    FaultScenario.WORKER_EXIT,
    FaultScenario.DISK_FULL,
    FaultScenario.NONFINITE_LOSS,
    FaultScenario.NONFINITE_GRADIENT,
)
ONE_GPU_SCENARIOS = (
    FaultScenario.MICROBATCH_SIGKILL,
    FaultScenario.OPTIMIZER_SIGKILL,
    FaultScenario.CHECKPOINT_SIGKILL,
    FaultScenario.CUDA_OOM,
)
FOUR_GPU_SCENARIOS = (
    FaultScenario.DDP_REDUCTION_SIGKILL,
    FaultScenario.SR_RANK_DIVERGENCE,
    FaultScenario.NCCL_RANK_FAILURE,
    FaultScenario.ALL_RANK_STOP,
    FaultScenario.FOUR_RANK_CHECKPOINT_RECOVERY,
)
ALL_SCENARIOS = CPU_SCENARIOS + ONE_GPU_SCENARIOS + FOUR_GPU_SCENARIOS


@dataclass(frozen=True, slots=True)
class TrainingControlSnapshot:
    """Fields that a fault must never silently alter."""

    resolved_config_sha256: str
    local_batch: int
    accumulation_steps: int
    attention_backend: str
    world_size: int
    optimizer_name: str
    learning_rate: float
    checkpoint_every_updates: int

    def __post_init__(self) -> None:
        if (
            type(self.resolved_config_sha256) is not str
            or _HEX64.fullmatch(self.resolved_config_sha256) is None
        ):
            raise ValueError("resolved config hash must be lowercase SHA-256")
        if (
            type(self.local_batch) is not int
            or self.local_batch <= 0
            or type(self.accumulation_steps) is not int
            or self.accumulation_steps <= 0
            or type(self.world_size) is not int
            or self.world_size <= 0
            or type(self.checkpoint_every_updates) is not int
            or self.checkpoint_every_updates <= 0
        ):
            raise ValueError("fault control counts must be positive integers")
        if (
            type(self.attention_backend) is not str
            or not self.attention_backend
            or type(self.optimizer_name) is not str
            or not self.optimizer_name
        ):
            raise ValueError("fault control backend and optimizer must be explicit")
        if (
            type(self.learning_rate) is not float
            or not 0.0 < self.learning_rate < 1.0
        ):
            raise ValueError("fault control learning rate must be an explicit float")


@dataclass(frozen=True, slots=True)
class ReplayEvidence:
    parent_checkpoint_id: str
    parent_successful_update: int
    completed_shards: tuple[str, ...]
    active_shard: str | None
    replayed_shards: int
    replayed_samples: int

    def __post_init__(self) -> None:
        if (
            type(self.parent_checkpoint_id) is not str
            or _CHECKPOINT_ID.fullmatch(self.parent_checkpoint_id) is None
        ):
            raise ValueError("recovery parent checkpoint ID is invalid")
        if (
            type(self.parent_successful_update) is not int
            or self.parent_successful_update < 0
            or type(self.replayed_shards) is not int
            or self.replayed_shards < 0
            or type(self.replayed_samples) is not int
            or self.replayed_samples < 0
        ):
            raise ValueError("recovery counters must be nonnegative integers")
        if (
            type(self.completed_shards) is not tuple
            or any(type(shard) is not str for shard in self.completed_shards)
            or self.completed_shards != tuple(sorted(set(self.completed_shards)))
        ):
            raise ValueError("completed shards must be sorted and unique")
        if (
            any(not shard for shard in self.completed_shards)
            or self.active_shard in self.completed_shards
            or (
                self.active_shard is not None
                and (type(self.active_shard) is not str or not self.active_shard)
            )
        ):
            raise ValueError("recovery shard state is inconsistent")


@dataclass(frozen=True, slots=True)
class FaultOutcome:
    scenario: FaultScenario
    status: FaultStatus
    hardware_level: HardwareLevel
    failure_type: str | None
    control_before: TrainingControlSnapshot | None
    control_after: TrainingControlSnapshot | None
    replay: ReplayEvidence | None
    evidence_file: str | None
    evidence_sha256: str | None
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status is FaultStatus.PASSED:
            if (
                type(self.failure_type) is not str
                or not self.failure_type
                or self.control_before is None
                or self.control_after is None
                or self.control_before != self.control_after
                or type(self.evidence_file) is not str
                or _EVIDENCE_FILE.fullmatch(self.evidence_file) is None
                or type(self.evidence_sha256) is not str
                or _HEX64.fullmatch(self.evidence_sha256) is None
                or type(self.blockers) is not tuple
                or self.blockers
            ):
                raise ValueError("passed fault outcome lacks invariant evidence")
            expected_hardware = (
                HardwareLevel.CPU
                if self.scenario in CPU_SCENARIOS
                else HardwareLevel.ONE_GPU
            )
            if self.scenario in FOUR_GPU_SCENARIOS or self.hardware_level is not expected_hardware:
                raise ValueError("passed fault outcome uses the wrong hardware level")
            if self.scenario in ONE_GPU_SCENARIOS and self.replay is None:
                raise ValueError("single-GPU fault outcome requires recovery evidence")
            return

        if (
            self.scenario not in FOUR_GPU_SCENARIOS
            or self.hardware_level is not HardwareLevel.FOUR_GPU
            or self.failure_type is not None
            or self.control_before is not None
            or self.control_after is not None
            or self.replay is not None
            or self.evidence_file is not None
            or self.evidence_sha256 is not None
            or type(self.blockers) is not tuple
            or any(type(blocker) is not str or not blocker for blocker in self.blockers)
            or not self.blockers
            or self.blockers != tuple(sorted(set(self.blockers)))
            or "FOUR-GPU-AVAILABLE" not in self.blockers
        ):
            raise ValueError("only unexecuted four-GPU faults may be blocked")


@dataclass(frozen=True, slots=True)
class FaultMatrixReport:
    task_id: str
    outcomes: tuple[FaultOutcome, ...]

    def __post_init__(self) -> None:
        if self.task_id != "T054":
            raise ValueError("fault matrix task ID must be T054")
        scenarios = tuple(outcome.scenario for outcome in self.outcomes)
        if scenarios != ALL_SCENARIOS:
            raise ValueError("fault matrix must contain every scenario in canonical order")
        if any(
            outcome.status is not FaultStatus.PASSED
            for outcome in self.outcomes[: len(CPU_SCENARIOS) + len(ONE_GPU_SCENARIOS)]
        ):
            raise ValueError("CPU and single-GPU matrix contains an unpassed scenario")
        if any(
            outcome.status is not FaultStatus.BLOCKED
            for outcome in self.outcomes[-len(FOUR_GPU_SCENARIOS) :]
        ):
            raise ValueError("four-GPU matrix must remain explicitly blocked")

    def to_dict(self) -> dict[str, object]:
        outcomes: list[dict[str, object]] = []
        for outcome in self.outcomes:
            item = asdict(outcome)
            item["scenario"] = outcome.scenario.value
            item["status"] = outcome.status.value
            item["hardware_level"] = outcome.hardware_level.value
            outcomes.append(cast(dict[str, object], item))
        return {
            "outcomes": outcomes,
            "schema_version": 2,
            "status": "cpu_single_gpu_complete_four_gpu_blocked",
            "task_id": self.task_id,
        }


__all__ = [
    "ALL_SCENARIOS",
    "CPU_SCENARIOS",
    "FOUR_GPU_SCENARIOS",
    "ONE_GPU_SCENARIOS",
    "FaultMatrixReport",
    "FaultOutcome",
    "FaultScenario",
    "FaultStatus",
    "HardwareLevel",
    "ReplayEvidence",
    "TrainingControlSnapshot",
]
