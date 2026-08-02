"""Explicit S0 readiness blockers and caller-supplied capacity sweep rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, order=True, slots=True)
class ProductionReadinessBlocker:
    """One non-TOML production contract that cannot be inferred or defaulted."""

    code: str
    subject: str
    required_evidence: str

    def __post_init__(self) -> None:
        if (
            not self.code.startswith("S0_")
            or not self.code.isupper()
            or not self.subject
            or self.subject != self.subject.strip()
            or not self.required_evidence
            or self.required_evidence != self.required_evidence.strip()
        ):
            raise ValueError("production readiness blocker is invalid")


S0_GOVERNED_SEMANTIC_BLOCKERS = (
    ProductionReadinessBlocker(
        code="S0_WARMUP_FUNCTION_UNRESOLVED",
        subject="scheduler warmup function and starting learning rate",
        required_evidence=(
            "a current confirmed decision; only the 2,000 successful-update length, "
            "2e-5 stable rate, and manual cosine decay target are locked"
        ),
    ),
    ProductionReadinessBlocker(
        code="S0_PASS_INDEX_OWNERSHIP_UNRESOLVED",
        subject="trainer-owned production data pass_index semantics",
        required_evidence=(
            "a current confirmed ownership and resume rule; the data service mainset "
            "must not be inferred as a trainer pass index"
        ),
    ),
    ProductionReadinessBlocker(
        code="S0_FORMAL_PROMPT_CONDITION_CONTRACT_UNRESOLVED",
        subject="formal evaluator prompt condition-to-caption-field semantics",
        required_evidence=(
            "a current typed prompt-condition manifest contract; flat condition "
            "strings cannot be guessed into caption field groups"
        ),
    ),
)


S0_RUNTIME_INTEGRATION_BLOCKERS = (
    ProductionReadinessBlocker(
        code="S0_LIVE_READY_QUEUE_DEPTH_UNBOUND",
        subject="telemetry ready_queue_depth observation",
        required_evidence=(
            "a live trainer-side queue-depth observation at each successful update; "
            "configured ready-batch capacity is not an observation"
        ),
    ),
    ProductionReadinessBlocker(
        code="S0_DIT_FLOPS_OBSERVATION_UNBOUND",
        subject="telemetry actual_dit_flops observation",
        required_evidence=(
            "a measured per-update DiT FLOPs observation from the production runtime; "
            "planned_dit_flops and planned_updates cannot be used to derive it"
        ),
    ),
)


ActivationCheckpointMode = Literal["none", "alternating", "all"]


@dataclass(frozen=True, slots=True)
class S0CapacitySweepRow:
    """One fully explicit S0 capacity candidate with no production defaults."""

    candidate_id: str
    local_batch: int
    accumulation: int
    global_batch: int
    activation_checkpoint_mode: ActivationCheckpointMode
    cache_low_watermark_gib: int
    cache_high_watermark_gib: int
    download_concurrency: int
    verified_shard_lookahead: int
    persistent_workers_per_rank: int
    ready_batches_per_rank: int
    lease_channel_capacity: int
    ack_channel_capacity: int

    def __post_init__(self) -> None:
        positive = (
            self.local_batch,
            self.accumulation,
            self.global_batch,
            self.cache_high_watermark_gib,
            self.download_concurrency,
            self.verified_shard_lookahead,
            self.persistent_workers_per_rank,
            self.ready_batches_per_rank,
            self.lease_channel_capacity,
            self.ack_channel_capacity,
        )
        workers = self.persistent_workers_per_rank
        if (
            not self.candidate_id
            or self.candidate_id != self.candidate_id.strip()
            or type(self.cache_low_watermark_gib) is not int
            or self.cache_low_watermark_gib < 0
            or any(type(value) is not int or value <= 0 for value in positive)
            or self.cache_low_watermark_gib >= self.cache_high_watermark_gib
            or self.cache_high_watermark_gib > 500
            or self.global_batch != self.local_batch * self.accumulation
            or self.activation_checkpoint_mode not in {"none", "alternating", "all"}
            or workers not in {1, 2, 3}
            or self.verified_shard_lookahead < workers
            or self.ready_batches_per_rank < workers
            or self.ready_batches_per_rank % workers
            or self.lease_channel_capacity < workers
            or self.ack_channel_capacity < workers
        ):
            raise ValueError("S0 capacity sweep row is invalid")


def validate_s0_capacity_sweep_matrix(
    rows: tuple[S0CapacitySweepRow, ...],
) -> tuple[S0CapacitySweepRow, ...]:
    """Validate an explicit matrix without selecting or promoting a candidate."""

    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not S0CapacitySweepRow for row in rows)
        or len({row.candidate_id for row in rows}) != len(rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("S0 capacity sweep requires unique explicit rows")
    observed_workers = {row.persistent_workers_per_rank for row in rows}
    if observed_workers != {1, 2, 3}:
        raise ValueError("S0 capacity sweep must cover 1/2/3 persistent workers")
    return rows


__all__ = [
    "S0_GOVERNED_SEMANTIC_BLOCKERS",
    "S0_RUNTIME_INTEGRATION_BLOCKERS",
    "ProductionReadinessBlocker",
    "S0CapacitySweepRow",
    "validate_s0_capacity_sweep_matrix",
]
