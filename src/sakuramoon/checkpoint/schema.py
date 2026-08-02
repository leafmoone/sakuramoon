"""Typed checkpoint identity, manifest and training-state schemas."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, cast

from sakuramoon.model.growth import ACTIVE_SLOT_IDS, half_cosine_growth_alpha
from sakuramoon.train.step import SingleGpuUpdateState

SCHEMA_VERSION = 1
RAW_SCHEMA_VERSION = 4
MAX_MODEL_SHARD_BYTES = 2 * 1024**3
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class CheckpointError(RuntimeError):
    """A checkpoint is invalid, incomplete or incompatible."""


class CheckpointKind(StrEnum):
    RAW = "raw"
    MODEL_ONLY = "model-only"
    PMA = "pma"
    RELEASE = "release"


class CheckpointReason(StrEnum):
    UPDATE_CADENCE = "update-cadence"
    WALL_CADENCE = "wall-cadence"
    STAGE_FINALIZE = "stage-finalize"
    PRE_TRANSITION = "pre-transition"
    POST_TRANSITION = "post-transition"
    PRE_GROWTH = "pre-growth"
    POST_GROWTH = "post-growth"
    RAMP_MIDPOINT = "ramp-midpoint"
    RAMP_END = "ramp-end"
    PRE_DECAY = "pre-decay"


FORCED_CHECKPOINT_REASONS = frozenset(
    {
        CheckpointReason.STAGE_FINALIZE,
        CheckpointReason.PRE_TRANSITION,
        CheckpointReason.POST_TRANSITION,
        CheckpointReason.PRE_GROWTH,
        CheckpointReason.POST_GROWTH,
        CheckpointReason.RAMP_MIDPOINT,
        CheckpointReason.RAMP_END,
        CheckpointReason.PRE_DECAY,
    }
)


@dataclass(frozen=True, slots=True)
class CheckpointCadence:
    """Durable update cadence plus the wall-clock time of its last commit."""

    last_successful_update: int
    last_wall_clock_unix_seconds: float
    every_successful_updates: int

    def __post_init__(self) -> None:
        if (
            type(self.last_successful_update) is not int
            or self.last_successful_update < 0
            or type(self.last_wall_clock_unix_seconds) is not float
            or not math.isfinite(self.last_wall_clock_unix_seconds)
            or self.last_wall_clock_unix_seconds < 0.0
            or type(self.every_successful_updates) is not int
            or self.every_successful_updates <= 0
        ):
            raise ValueError("checkpoint cadence fields are invalid")

    def due(
        self,
        *,
        successful_update: int,
        wall_clock_unix_seconds: float,
        forced: CheckpointReason | None = None,
    ) -> CheckpointReason | None:
        if (
            type(successful_update) is not int
            or successful_update < self.last_successful_update
            or type(wall_clock_unix_seconds) is not float
            or not math.isfinite(wall_clock_unix_seconds)
            or wall_clock_unix_seconds < self.last_wall_clock_unix_seconds
        ):
            raise ValueError("checkpoint cadence input is invalid")
        if forced is not None:
            if forced not in FORCED_CHECKPOINT_REASONS:
                raise ValueError("checkpoint reason is not a forced checkpoint")
            return forced
        if (
            successful_update > self.last_successful_update
            and successful_update % self.every_successful_updates == 0
        ):
            return CheckpointReason.UPDATE_CADENCE
        return None

    def committed(
        self,
        *,
        successful_update: int,
        wall_clock_unix_seconds: float,
        reason: CheckpointReason,
    ) -> CheckpointCadence:
        if reason != self.due(
            successful_update=successful_update,
            wall_clock_unix_seconds=wall_clock_unix_seconds,
            forced=reason if reason in FORCED_CHECKPOINT_REASONS else None,
        ):
            raise ValueError("committed checkpoint reason differs from the due reason")
        return CheckpointCadence(
            successful_update,
            wall_clock_unix_seconds,
            self.every_successful_updates,
        )


def _require_hash(name: str, value: str) -> None:
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    checkpoint_id: str
    update: int
    config_sha256: str
    dependency_sha256: str
    parameter_schema_sha256: str

    def __post_init__(self) -> None:
        if _CHECKPOINT_ID.fullmatch(self.checkpoint_id) is None:
            raise ValueError("checkpoint_id is invalid")
        if type(self.update) is not int or self.update < 0:
            raise ValueError("checkpoint update must be a nonnegative integer")
        _require_hash("config_sha256", self.config_sha256)
        _require_hash("dependency_sha256", self.dependency_sha256)
        _require_hash("parameter_schema_sha256", self.parameter_schema_sha256)


@dataclass(frozen=True, slots=True)
class GrowthCheckpointState:
    active_slot_ids: tuple[int, ...]
    alpha: float
    stage: str
    world_size: int
    resolution: int
    ramp_start_successful_update: int | None
    ramp_updates: int | None

    def __post_init__(self) -> None:
        if self.active_slot_ids not in ACTIVE_SLOT_IDS.values():
            raise ValueError(
                "growth active_slot_ids must be a canonical 16/20/24 slot set"
            )
        if type(self.alpha) is not float or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("growth alpha must be a float in [0, 1]")
        if not self.stage or type(self.stage) is not str:
            raise ValueError("growth stage must be a nonempty string")
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("growth world size must be a positive integer")
        if type(self.resolution) is not int or self.resolution <= 0:
            raise ValueError("growth resolution must be a positive integer")
        has_start = self.ramp_start_successful_update is not None
        has_updates = self.ramp_updates is not None
        if has_start != has_updates:
            raise ValueError(
                "growth ramp origin and duration must both be present or absent"
            )
        if has_start:
            if (
                type(self.ramp_start_successful_update) is not int
                or self.ramp_start_successful_update < 0
                or type(self.ramp_updates) is not int
                or not 1000 <= self.ramp_updates <= 5000
            ):
                raise ValueError("growth ramp state is invalid")
        elif self.alpha != 1.0:
            raise ValueError("non-growth checkpoint alpha must be complete")


@dataclass(frozen=True, slots=True)
class StageBudgetCheckpointState:
    start_successful_update: int
    terminal_successful_update: int

    def __post_init__(self) -> None:
        if (
            type(self.start_successful_update) is not int
            or self.start_successful_update < 0
            or type(self.terminal_successful_update) is not int
            or self.terminal_successful_update <= self.start_successful_update
        ):
            raise ValueError("checkpoint stage budget is invalid")


@dataclass(frozen=True, slots=True)
class RawCheckpointState:
    trainer: SingleGpuUpdateState
    growth: GrowthCheckpointState
    stage_budget: StageBudgetCheckpointState
    checkpoint_cadence: CheckpointCadence

    def __post_init__(self) -> None:
        if (
            type(self.trainer.attempted_updates) is not int
            or type(self.trainer.successful_updates) is not int
            or type(self.trainer.effective_samples) is not int
        ):
            raise ValueError("checkpoint trainer state is inconsistent")
        if type(self.stage_budget) is not StageBudgetCheckpointState:
            raise TypeError("checkpoint stage budget has an invalid type")
        if type(self.checkpoint_cadence) is not CheckpointCadence:
            raise TypeError("checkpoint cadence has an invalid type")
        if not (
            self.stage_budget.start_successful_update
            <= self.trainer.successful_updates
            <= self.stage_budget.terminal_successful_update
        ):
            raise ValueError("checkpoint update is outside the persisted stage budget")
        if (
            self.checkpoint_cadence.last_successful_update
            != self.trainer.successful_updates
        ):
            raise ValueError(
                "checkpoint cadence is not committed at the checkpoint update"
            )
        growth = self.growth
        if growth.ramp_start_successful_update is not None:
            if (
                growth.ramp_start_successful_update
                != self.stage_budget.start_successful_update
            ):
                raise ValueError(
                    "growth ramp origin differs from the persisted stage start"
                )
            if self.trainer.successful_updates < growth.ramp_start_successful_update:
                raise ValueError("checkpoint update precedes growth ramp origin")
            assert growth.ramp_updates is not None
            if (
                growth.ramp_start_successful_update + growth.ramp_updates
                > self.stage_budget.terminal_successful_update
            ):
                raise ValueError("growth ramp exceeds the persisted stage budget")
            expected_alpha = half_cosine_growth_alpha(
                self.trainer.successful_updates - growth.ramp_start_successful_update,
                growth.ramp_updates,
            )
            if growth.alpha != expected_alpha:
                raise ValueError(
                    "checkpoint growth alpha differs from persisted ramp progress"
                )


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.as_posix() != self.path
        ):
            raise ValueError("checkpoint file path must be relative and normalized")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("checkpoint file size must be nonnegative")
        _require_hash("file sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    kind: CheckpointKind
    identity: CheckpointIdentity
    files: tuple[FileRecord, ...]

    def __post_init__(self) -> None:
        paths = tuple(record.path for record in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ValueError(
                "checkpoint manifest files must be nonempty, sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class CheckpointSaveResult:
    path: Path
    kind: CheckpointKind
    payload_bytes: int
    files: int


def identity_to_dict(identity: CheckpointIdentity) -> dict[str, object]:
    return {
        "checkpoint_id": identity.checkpoint_id,
        "config_sha256": identity.config_sha256,
        "dependency_sha256": identity.dependency_sha256,
        "parameter_schema_sha256": identity.parameter_schema_sha256,
        "update": identity.update,
    }


def manifest_to_dict(manifest: CheckpointManifest) -> dict[str, object]:
    return {
        "files": [
            {"path": record.path, "sha256": record.sha256, "size": record.size}
            for record in manifest.files
        ],
        "identity": identity_to_dict(manifest.identity),
        "kind": manifest.kind.value,
        "schema_version": (
            RAW_SCHEMA_VERSION
            if manifest.kind is CheckpointKind.RAW
            else SCHEMA_VERSION
        ),
    }


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(document: dict[str, Any], expected: set[str], name: str) -> None:
    if set(document) != expected:
        raise CheckpointError(f"{name} has unknown or missing fields")


def _has_schema_version(document: dict[str, Any], expected: int) -> bool:
    value = document.get("schema_version")
    return type(value) is int and value == expected


def identity_from_dict(value: object) -> CheckpointIdentity:
    document = _mapping(value, "checkpoint identity")
    _exact_keys(
        document,
        {
            "checkpoint_id",
            "config_sha256",
            "dependency_sha256",
            "parameter_schema_sha256",
            "update",
        },
        "checkpoint identity",
    )
    try:
        if not all(
            isinstance(document[key], str)
            for key in (
                "checkpoint_id",
                "config_sha256",
                "dependency_sha256",
                "parameter_schema_sha256",
            )
        ):
            raise TypeError
        return CheckpointIdentity(
            checkpoint_id=cast(str, document["checkpoint_id"]),
            update=document["update"],
            config_sha256=cast(str, document["config_sha256"]),
            dependency_sha256=cast(str, document["dependency_sha256"]),
            parameter_schema_sha256=cast(str, document["parameter_schema_sha256"]),
        )
    except (TypeError, ValueError):
        raise CheckpointError("checkpoint identity is invalid") from None


def manifest_from_dict(value: object) -> CheckpointManifest:
    document = _mapping(value, "checkpoint manifest")
    _exact_keys(
        document, {"schema_version", "kind", "identity", "files"}, "checkpoint manifest"
    )
    try:
        kind = CheckpointKind(document["kind"])
    except (TypeError, ValueError):
        raise CheckpointError("checkpoint kind is invalid") from None
    expected_schema = (
        RAW_SCHEMA_VERSION if kind is CheckpointKind.RAW else SCHEMA_VERSION
    )
    if not _has_schema_version(document, expected_schema):
        if kind is CheckpointKind.RAW:
            raise CheckpointError("legacy raw checkpoint schema is unsupported")
        raise CheckpointError("checkpoint manifest schema version is unsupported")
    raw_files = document["files"]
    if not isinstance(raw_files, list):
        raise CheckpointError("checkpoint manifest files must be an array")
    records: list[FileRecord] = []
    try:
        for raw_record in cast(list[object], raw_files):
            record = _mapping(raw_record, "checkpoint file record")
            _exact_keys(record, {"path", "size", "sha256"}, "checkpoint file record")
            if not isinstance(record["path"], str) or not isinstance(
                record["sha256"], str
            ):
                raise TypeError
            records.append(
                FileRecord(
                    path=record["path"],
                    size=record["size"],
                    sha256=record["sha256"],
                )
            )
        return CheckpointManifest(
            kind=kind,
            identity=identity_from_dict(document["identity"]),
            files=tuple(records),
        )
    except (TypeError, ValueError):
        raise CheckpointError("checkpoint manifest is invalid") from None


def raw_state_to_dict(
    state: RawCheckpointState,
) -> tuple[dict[str, object], dict[str, object]]:
    trainer: dict[str, object] = {
        "attempted_updates": state.trainer.attempted_updates,
        "checkpoint_cadence": {
            "every_successful_updates": state.checkpoint_cadence.every_successful_updates,
            "last_successful_update": state.checkpoint_cadence.last_successful_update,
            "last_wall_clock_unix_seconds": (
                state.checkpoint_cadence.last_wall_clock_unix_seconds
            ),
        },
        "effective_samples": state.trainer.effective_samples,
        "schema_version": RAW_SCHEMA_VERSION,
        "stage_budget": {
            "start_successful_update": state.stage_budget.start_successful_update,
            "terminal_successful_update": state.stage_budget.terminal_successful_update,
        },
        "successful_updates": state.trainer.successful_updates,
    }
    growth: dict[str, object] = {
        "active_slot_ids": list(state.growth.active_slot_ids),
        "alpha": state.growth.alpha,
        "ramp_start_successful_update": state.growth.ramp_start_successful_update,
        "ramp_updates": state.growth.ramp_updates,
        "resolution": state.growth.resolution,
        "schema_version": RAW_SCHEMA_VERSION,
        "stage": state.growth.stage,
        "world_size": state.growth.world_size,
    }
    return trainer, growth


def raw_state_from_dicts(
    trainer_value: object, growth_value: object
) -> RawCheckpointState:
    trainer = _mapping(trainer_value, "trainer state")
    growth = _mapping(growth_value, "growth state")
    _exact_keys(
        trainer,
        {
            "schema_version",
            "attempted_updates",
            "successful_updates",
            "effective_samples",
            "stage_budget",
            "checkpoint_cadence",
        },
        "trainer state",
    )
    _exact_keys(
        growth,
        {
            "schema_version",
            "active_slot_ids",
            "alpha",
            "stage",
            "world_size",
            "resolution",
            "ramp_start_successful_update",
            "ramp_updates",
        },
        "growth state",
    )
    if not all(
        _has_schema_version(document, RAW_SCHEMA_VERSION)
        for document in (trainer, growth)
    ):
        raise CheckpointError("checkpoint state schema version is unsupported")
    stage_budget = _mapping(trainer["stage_budget"], "checkpoint stage budget")
    _exact_keys(
        stage_budget,
        {"start_successful_update", "terminal_successful_update"},
        "checkpoint stage budget",
    )
    cadence = _mapping(trainer["checkpoint_cadence"], "checkpoint cadence")
    _exact_keys(
        cadence,
        {
            "last_successful_update",
            "last_wall_clock_unix_seconds",
            "every_successful_updates",
        },
        "checkpoint cadence",
    )
    slots = growth["active_slot_ids"]
    if not isinstance(slots, list) or not all(
        type(item) is int for item in cast(list[object], slots)
    ):
        raise CheckpointError("growth active slots are invalid")
    try:
        return RawCheckpointState(
            trainer=SingleGpuUpdateState(
                attempted_updates=trainer["attempted_updates"],
                successful_updates=trainer["successful_updates"],
                effective_samples=trainer["effective_samples"],
            ),
            growth=GrowthCheckpointState(
                active_slot_ids=tuple(cast(list[int], slots)),
                alpha=growth["alpha"],
                stage=growth["stage"],
                world_size=growth["world_size"],
                resolution=growth["resolution"],
                ramp_start_successful_update=growth["ramp_start_successful_update"],
                ramp_updates=growth["ramp_updates"],
            ),
            stage_budget=StageBudgetCheckpointState(
                start_successful_update=stage_budget["start_successful_update"],
                terminal_successful_update=stage_budget["terminal_successful_update"],
            ),
            checkpoint_cadence=CheckpointCadence(
                last_successful_update=cadence["last_successful_update"],
                last_wall_clock_unix_seconds=cadence["last_wall_clock_unix_seconds"],
                every_successful_updates=cadence["every_successful_updates"],
            ),
        )
    except (TypeError, ValueError):
        raise CheckpointError("checkpoint training state is invalid") from None


__all__ = [
    "FORCED_CHECKPOINT_REASONS",
    "MAX_MODEL_SHARD_BYTES",
    "RAW_SCHEMA_VERSION",
    "CheckpointCadence",
    "CheckpointError",
    "CheckpointIdentity",
    "CheckpointKind",
    "CheckpointManifest",
    "CheckpointReason",
    "CheckpointSaveResult",
    "FileRecord",
    "GrowthCheckpointState",
    "RawCheckpointState",
    "StageBudgetCheckpointState",
    "identity_from_dict",
    "identity_to_dict",
    "manifest_from_dict",
    "manifest_to_dict",
    "raw_state_from_dicts",
    "raw_state_to_dict",
]
