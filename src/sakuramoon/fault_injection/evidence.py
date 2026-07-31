"""Strict per-scenario evidence and canonical T054 matrix binding."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from sakuramoon.fault_injection.report import write_fault_matrix
from sakuramoon.fault_injection.schema import (
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

_CONTROL_KEYS = {
    "accumulation_steps",
    "attention_backend",
    "checkpoint_every_updates",
    "learning_rate",
    "local_batch",
    "optimizer_name",
    "resolved_config_sha256",
    "world_size",
}
_REPLAY_KEYS = {
    "active_shard",
    "completed_shards",
    "parent_checkpoint_id",
    "parent_successful_update",
    "replayed_samples",
    "replayed_shards",
}
_EVIDENCE_KEYS = {
    "control_after",
    "control_before",
    "failure_type",
    "hardware_level",
    "replay",
    "scenario",
    "schema_version",
    "task_id",
    "test_report_sha256",
    "test_selector",
}


@dataclass(frozen=True, slots=True)
class ExecutedFaultEvidence:
    """One actual CPU or single-GPU fault result before matrix publication."""

    scenario: FaultScenario
    hardware_level: HardwareLevel
    failure_type: str
    control_before: TrainingControlSnapshot
    control_after: TrainingControlSnapshot
    replay: ReplayEvidence | None
    test_selector: str
    test_report_sha256: str

    def __post_init__(self) -> None:
        expected_hardware = (
            HardwareLevel.CPU
            if self.scenario in CPU_SCENARIOS
            else HardwareLevel.ONE_GPU
        )
        if (
            self.scenario in FOUR_GPU_SCENARIOS
            or self.hardware_level is not expected_hardware
            or type(self.failure_type) is not str
            or not self.failure_type
            or self.control_before != self.control_after
            or (self.scenario in ONE_GPU_SCENARIOS and self.replay is None)
            or type(self.test_selector) is not str
            or not self.test_selector.startswith("tests/")
            or "\n" in self.test_selector
            or type(self.test_report_sha256) is not str
            or len(self.test_report_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.test_report_sha256)
        ):
            raise ValueError("executed fault evidence is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "control_after": asdict(self.control_after),
            "control_before": asdict(self.control_before),
            "failure_type": self.failure_type,
            "hardware_level": self.hardware_level.value,
            "replay": None if self.replay is None else asdict(self.replay),
            "scenario": self.scenario.value,
            "schema_version": 1,
            "task_id": "T054",
            "test_report_sha256": self.test_report_sha256,
            "test_selector": self.test_selector,
        }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_executed_fault_evidence(
    path: Path, evidence: ExecutedFaultEvidence
) -> None:
    """Durably publish one no-clobber scenario record."""

    expected_name = f"{evidence.scenario.value}.json"
    if path.name != expected_name:
        raise ValueError("fault evidence filename does not match its scenario")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("fault scenario evidence already exists")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    body = (
        json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        published = True
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise


def _mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"fault evidence {label} must be an object")
    mapping = cast(dict[object, object], value)
    if set(mapping) != keys or not all(type(key) is str for key in mapping):
        raise ValueError(f"fault evidence {label} fields are invalid")
    return cast(dict[str, Any], mapping)


def _control(value: object) -> TrainingControlSnapshot:
    document = _mapping(value, _CONTROL_KEYS, "control")
    return TrainingControlSnapshot(**document)


def _replay(value: object) -> ReplayEvidence | None:
    if value is None:
        return None
    document = _mapping(value, _REPLAY_KEYS, "replay")
    completed = document["completed_shards"]
    if type(completed) is not list:
        raise ValueError("fault evidence completed shards must be a list")
    document["completed_shards"] = tuple(cast(list[object], completed))
    return ReplayEvidence(**document)


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("fault evidence could not be opened") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise ValueError("fault evidence must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
    finally:
        os.close(descriptor)
    return body


def _decode_executed_fault_evidence(
    body: bytes, path: Path
) -> ExecutedFaultEvidence:
    try:
        document = _mapping(json.loads(body), _EVIDENCE_KEYS, "root")
        if document["schema_version"] != 1 or document["task_id"] != "T054":
            raise ValueError
        evidence = ExecutedFaultEvidence(
            scenario=FaultScenario(document["scenario"]),
            hardware_level=HardwareLevel(document["hardware_level"]),
            failure_type=document["failure_type"],
            control_before=_control(document["control_before"]),
            control_after=_control(document["control_after"]),
            replay=_replay(document["replay"]),
            test_selector=document["test_selector"],
            test_report_sha256=document["test_report_sha256"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("fault scenario evidence is invalid") from None
    if path.name != f"{evidence.scenario.value}.json":
        raise ValueError("fault evidence filename does not match its scenario")
    return evidence


def load_executed_fault_evidence(path: Path) -> ExecutedFaultEvidence:
    """Load one strict scenario record without accepting unknown fields."""

    return _decode_executed_fault_evidence(_read_regular_file(path), path)


def publish_fault_matrix_from_evidence(
    evidence_directory: Path, verification_artifact: Path, output: Path
) -> FaultMatrixReport:
    """Bind every executed result, block every 4GPU item, and publish the matrix."""

    verification_sha256 = hashlib.sha256(
        _read_regular_file(verification_artifact)
    ).hexdigest()
    outcomes: list[FaultOutcome] = []
    for scenario in CPU_SCENARIOS + ONE_GPU_SCENARIOS:
        path = evidence_directory / f"{scenario.value}.json"
        body = _read_regular_file(path)
        evidence = _decode_executed_fault_evidence(body, path)
        if evidence.scenario is not scenario:
            raise ValueError("fault evidence is not in canonical scenario order")
        if evidence.test_report_sha256 != verification_sha256:
            raise ValueError("fault evidence verification artifact hash is stale")
        outcomes.append(
            FaultOutcome(
                scenario=evidence.scenario,
                status=FaultStatus.PASSED,
                hardware_level=evidence.hardware_level,
                failure_type=evidence.failure_type,
                control_before=evidence.control_before,
                control_after=evidence.control_after,
                replay=evidence.replay,
                evidence_file=path.name,
                evidence_sha256=hashlib.sha256(body).hexdigest(),
                blockers=(),
            )
        )
    outcomes.extend(
        FaultOutcome(
            scenario=scenario,
            status=FaultStatus.BLOCKED,
            hardware_level=HardwareLevel.FOUR_GPU,
            failure_type=None,
            control_before=None,
            control_after=None,
            replay=None,
            evidence_file=None,
            evidence_sha256=None,
            blockers=("FOUR-GPU-AVAILABLE",),
        )
        for scenario in FOUR_GPU_SCENARIOS
    )
    report = FaultMatrixReport("T054", tuple(outcomes))
    write_fault_matrix(output, report)
    return report


__all__ = [
    "ExecutedFaultEvidence",
    "load_executed_fault_evidence",
    "publish_fault_matrix_from_evidence",
    "write_executed_fault_evidence",
]
