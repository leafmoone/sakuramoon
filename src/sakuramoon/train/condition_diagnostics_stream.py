"""Versioned per-rank condition-side diagnostics, per successful update.

Independent opt-in stream (config section ``[condition_diagnostics]``,
absent by default).  It writes ONE append-only JSONL file per rank under
``<artifact_dir>/diagnostics/rank-<n>/`` and is versioned independently of
the TrainingMetric schema (``schema_version`` field per record; evolution
only adds fields).  When the section is absent the training path, the
optimizer flow, and the schema-12 output are byte-identical to a run
without this feature.

Quantity definitions (v1).  Activation facts come from the LAST
accumulated microbatch of the update; the runtime's ``measure()`` computes
them when the record cadence is due — mid-update, BEFORE
``optimizer.step()`` — so they always use the weights and condition tokens
that forward fed to the DiT.  Gradient shares are measured at the update
boundary:

- ``condition_token_rms_active_pre_norm``: RMS of the raw (pre-norm) Qwen
  states gathered at the condition positions of active samples — exactly
  the gather the encoder applies before ``shared_norm``.  Masked
  (padding) slots are excluded.
- ``condition_token_rms_active_post_norm``: RMS of the encoder output
  tokens of active samples.
- ``condition_token_rms_null_pre_norm``: RMS of the learned ``null_tokens``
  parameter (native fp32 precision; the null path applies no norm).
- ``condition_token_rms_null_post_norm``: RMS of the tokens of null
  samples (identical values in practice — the null path is identity).
- ``condition_global_norm_weight_rms``: RMS of the global-path RMSNorm
  weight (``condition_global_norm.weight``).
- ``base_hidden_rms``: per-sample RMS of the base (condition-free) global
  hidden, mean over the whole sampled microbatch.
- ``condition_residual_rms``: per-sample RMS of the condition residual,
  mean over ACTIVE samples only (null samples are structurally zero and
  would dilute the signal).
- ``slot_cosine_<slot:02d>``: cosine between the per-slot modulation vector
  WITH the condition and the BASE-ONLY per-slot modulation (mean over
  active samples).  Approaches 1.0 when the condition does not steer that
  slot; ``null`` when there are no active samples or a zero-norm vector
  makes the cosine undefined.
- ``condition_grad_share_{pre,post}_clip``: L2 norm over the union of the
  condition-side parameters (``condition_tokens.*`` and ``dit.conditioner.*``)
  divided by the global gradient norm, at the update boundary BEFORE and
  AFTER the global clip (both after the sample-mean gradient scale).
  ``null`` when the global norm is exactly zero.

Diagnostics are advisory: a nonfinite activation value is recorded as
``null`` (never a training failure).  Structural contract violations
(shape or key mismatch) raise, because they indicate a code bug rather
than a data condition.

The stream is append-only and durable per write.  If an update is later
retried (data-service redelivery of the same attempt), its record is
appended to the SAME per-rank file, so a failed update may appear more
than once; readers should treat ``successful_update`` as non-monotonic
exactly like the main schema-12 metrics stream.
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch

from sakuramoon.conditioning.condition_tokens import ConditionTokenOutput
from sakuramoon.telemetry.metrics import DurableJsonlSink
from sakuramoon.train.step import (
    SingleGpuUpdateResult,
    TrainableComposite,
    TrainableCompositeInputs,
)
from sakuramoon.train.tensor_metrics import (
    TrainingSamplingError,
    metric_float,
    tensor_cosine_rows,
    tensor_rms,
)

if TYPE_CHECKING:
    from sakuramoon.train.runtime import SuccessfulTrainingObservation

CONDITION_DIAGNOSTICS_SCHEMA_VERSION = 1

# Fixed key set per record (slot keys are added from the run's active slots
# at construction time); unknown keys are rejected at record build time.
_BASE_FACT_KEYS = frozenset(
    {
        "active_samples",
        "null_samples",
        "condition_token_rms_active_pre_norm",
        "condition_token_rms_active_post_norm",
        "condition_token_rms_null_pre_norm",
        "condition_token_rms_null_post_norm",
        "condition_global_norm_weight_rms",
        "base_hidden_rms",
        "condition_residual_rms",
    }
)


def diagnostics_file_name(version: int = CONDITION_DIAGNOSTICS_SCHEMA_VERSION) -> str:
    return f"condition-diagnostics-v{version}.jsonl"


def _nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a nonnegative integer")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{name} must be a positive integer")


def _finite_float_or_none(name: str, value: torch.Tensor) -> float | None:
    """``metric_float`` with the advisory policy: a nonfinite value is
    recorded as ``null`` (never a training failure)."""

    try:
        result = metric_float(name, value)
    except TrainingSamplingError:
        return None
    if result < 0.0 and name.endswith("_rms"):
        raise ValueError(f"{name} must not be negative")
    return result


def _masked_rms(values: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """RMS over the elements selected by a same-shape boolean mask."""

    if values.shape != keep.shape:
        raise ValueError("masked RMS tensors must share one shape")
    flat_values = values.float().flatten()
    flat_keep = keep.flatten().to(flat_values.dtype)
    count = int(flat_keep.sum().item())
    if count == 0:
        raise ValueError("masked RMS has no selected elements")
    return torch.sqrt((flat_values.square() * flat_keep).sum()) / math.sqrt(count)


def _per_sample_rms_mean(values: torch.Tensor) -> torch.Tensor:
    """Per-row RMS, mean over rows.  ``values`` is ``[B, ...]``."""

    rows = values.float().reshape(values.shape[0], -1)
    return rows.square().mean(dim=1).sqrt().mean()


def _slot_cosine(
    shared_with_condition: torch.Tensor,
    shared_base_only: torch.Tensor,
    bias: torch.Tensor,
    active_indices: torch.Tensor,
) -> float | None:
    """Mean over active samples of cosine(slot modulation w/ condition,
    slot modulation base-only).

    The cosine math lives in the shared batched primitive
    ``tensor_metrics.tensor_cosine_rows`` — the scalar ``tensor_cosine``
    definition applied per row, as a handful of whole-tensor kernels on
    the inputs' device (no per-row kernel launches, no per-row
    device-to-host copy; the scalar ``tensor_cosine`` itself is
    untouched).  This wrapper adds only the diagnostic advisory policy
    on top: nonfinite input or result -> ``None`` (never a training
    failure), no active samples -> ``None``.

    ``shared_*`` are ``[B, M]``; ``bias`` is ``[M]``.
    """

    if active_indices.numel() == 0:
        return None
    bias_f = bias.float()
    left = (
        shared_with_condition.index_select(0, active_indices).float() + bias_f
    ).flatten(1)
    right = (shared_base_only.index_select(0, active_indices).float() + bias_f).flatten(1)
    # Nonfinite input -> advisory None (the shared primitive's float() cast
    # would otherwise let NaN/inf through to the record and the JSONL writer).
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        return None
    try:
        cosines = tensor_cosine_rows(left, right)
    except TrainingSamplingError:
        # any row at/below the 1e-12 zero-norm threshold: undefined
        return None
    if not torch.isfinite(cosines).all():
        return None
    # double-precision mean keeps the accumulation error at float64 level
    return float(cosines.double().mean())


def compute_condition_activation_facts(
    composite: TrainableComposite,
    inputs: TrainableCompositeInputs,
    condition: ConditionTokenOutput,
) -> dict[str, float | int | None]:
    """The per-microbatch activation facts for one successful update.

    All tensor math is detached (``no_grad``); the global-path quantities
    re-run the SAME conditioner module on the SAME inputs/weights — its
    forward is a pure function of both, so the values match what the DiT
    consumed internally without touching the training graph.
    """

    encoder = composite.condition_tokens
    dit = composite.dit
    batch = inputs.qwen_states.shape[0]
    active_mask = condition.active_mask
    if active_mask.shape != (batch,):
        raise ValueError("condition active mask batch differs from Qwen states")
    active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
    null_indices = torch.nonzero(~active_mask, as_tuple=False).flatten()
    facts: dict[str, float | int | None] = {
        "active_samples": int(active_indices.numel()),
        "null_samples": int(null_indices.numel()),
    }
    if active_indices.numel():
        # Exactly the gather the encoder applies before ``shared_norm``
        # (masked slots are zero-indexed, then excluded from the RMS).
        safe_indices = inputs.condition_token_indices.index_select(
            0, active_indices
        )
        slot_valid = inputs.condition_mask.index_select(0, active_indices)
        safe_indices = safe_indices.masked_fill(~slot_valid, 0)
        gather_index = safe_indices[:, :, None, None].expand(
            -1, -1, 7, encoder.input_size
        )
        gathered = torch.gather(
            inputs.qwen_states.index_select(0, active_indices).detach(),
            dim=1,
            index=gather_index,
        )
        slot_keep = slot_valid[:, :, None, None].expand_as(gathered)
        facts["condition_token_rms_active_pre_norm"] = _finite_float_or_none(
            "condition_token_rms_active_pre_norm", _masked_rms(gathered, slot_keep)
        )
        facts["condition_token_rms_active_post_norm"] = _finite_float_or_none(
            "condition_token_rms_active_post_norm",
            tensor_rms(condition.tokens.index_select(0, active_indices)),
        )
    else:
        facts["condition_token_rms_active_pre_norm"] = None
        facts["condition_token_rms_active_post_norm"] = None
    # The null tokens are the learned parameter; the null path applies no
    # norm, so pre- and post-norm differ only by the dtype cast.
    facts["condition_token_rms_null_pre_norm"] = _finite_float_or_none(
        "condition_token_rms_null_pre_norm", tensor_rms(encoder.null_tokens)
    )
    if null_indices.numel():
        facts["condition_token_rms_null_post_norm"] = _finite_float_or_none(
            "condition_token_rms_null_post_norm",
            tensor_rms(condition.tokens.index_select(0, null_indices)),
        )
    else:
        facts["condition_token_rms_null_post_norm"] = None

    conditioner = dit.conditioner
    slot_ids = dit.active_slot_ids
    with torch.no_grad():
        global_output = conditioner(
            inputs.timestep,
            inputs.size_scale,
            inputs.aspect,
            condition.tokens,
            condition.active_mask,
            slot_ids,
        )
        facts["condition_global_norm_weight_rms"] = _finite_float_or_none(
            "condition_global_norm_weight_rms",
            tensor_rms(conditioner.condition_global_norm.weight),
        )
        facts["base_hidden_rms"] = _finite_float_or_none(
            "base_hidden_rms", _per_sample_rms_mean(global_output.base_hidden)
        )
        if active_indices.numel():
            facts["condition_residual_rms"] = _finite_float_or_none(
                "condition_residual_rms",
                _per_sample_rms_mean(
                    global_output.condition_residual.index_select(0, active_indices)
                ),
            )
        else:
            facts["condition_residual_rms"] = None
        shared_with_condition = conditioner.shared_block_projection(
            global_output.total_hidden
        )
        shared_base_only = conditioner.shared_block_projection(
            global_output.base_hidden
        )
        for slot_id in slot_ids:
            facts[f"slot_cosine_{slot_id:02d}"] = _slot_cosine(
                shared_with_condition,
                shared_base_only,
                conditioner.block_biases[f"slot_{slot_id:02d}"],
                active_indices,
            )
    return facts


def _required_facts_keys(slot_ids: tuple[int, ...]) -> set[str]:
    return set(_BASE_FACT_KEYS) | {
        f"slot_cosine_{slot_id:02d}" for slot_id in slot_ids
    }


def _grad_share(condition_norm: float, total_norm: float) -> float | None:
    if not (math.isfinite(condition_norm) and math.isfinite(total_norm)):
        raise ValueError("gradient norms must be finite")
    if total_norm <= 0.0:
        return None
    return condition_norm / total_norm


def build_condition_diagnostic_record(
    *,
    successful_update: int,
    rank: int,
    world_size: int,
    update: SingleGpuUpdateResult,
    activation_facts: Mapping[str, float | int | None],
    slot_ids: tuple[int, ...],
) -> dict[str, object]:
    """Assemble the versioned record for one sampled update."""

    _positive_int("successful_update", successful_update)
    _nonnegative_int("rank", rank)
    _positive_int("world_size", world_size)
    pre_condition = update.condition_grad_norm_pre_clip
    post_condition = update.condition_grad_norm_post_clip
    if pre_condition is None or post_condition is None:
        raise ValueError(
            "condition grad norms are missing from the update result; "
            "the step must run with condition_grad_share enabled"
        )
    required = _required_facts_keys(slot_ids)
    actual = set(activation_facts)
    if actual != required:
        missing = sorted(required - actual)
        unknown = sorted(actual - required)
        raise ValueError(
            "activation fact keys do not match the record contract; "
            f"missing={missing}, unknown={unknown}"
        )
    for key in sorted(required - {"active_samples", "null_samples"}):
        value = activation_facts[key]
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError(f"activation fact {key} must be finite or null")
    for key in ("active_samples", "null_samples"):
        if type(activation_facts[key]) is not int:
            raise ValueError(f"activation fact {key} must be an integer")
    total_pre = float(update.clip.pre_clip_norm.detach().cpu().item())
    total_post = float(update.clip.post_clip_norm.detach().cpu().item())
    pre_condition_value = float(pre_condition.detach().cpu().item())
    post_condition_value = float(post_condition.detach().cpu().item())
    record: dict[str, object] = {
        "schema_version": CONDITION_DIAGNOSTICS_SCHEMA_VERSION,
        "successful_update": successful_update,
        "recorded_at_unix_ns": time.time_ns(),
        "rank": rank,
        "world_size": world_size,
        "microbatches": update.microbatches,
        "effective_samples": update.effective_samples,
        **dict(activation_facts),
        "condition_grad_norm_pre_clip": pre_condition_value,
        "condition_grad_norm_post_clip": post_condition_value,
        "total_grad_norm_pre_clip": total_pre,
        "total_grad_norm_post_clip": total_post,
        "condition_grad_share_pre_clip": _grad_share(
            pre_condition_value, total_pre
        ),
        "condition_grad_share_post_clip": _grad_share(
            post_condition_value, total_post
        ),
    }
    return record


def _warn_condition_io(message: str, error: Exception) -> None:
    # Same best-effort stderr/traceback pattern as Guard Window: even the
    # warning must not replace a training error when stderr is broken.
    try:
        print(f"[condition_diagnostics] {message}", file=sys.stderr, flush=True)
        traceback.print_exception(error, file=sys.stderr)
    except Exception:  # noqa: BLE001,S110 - a broken stderr must not mask training
        pass


class ConditionDiagnosticsRecorder:
    """Per-rank durable writer for the condition diagnostics stream.

    One instance per rank, each bound to its own file, so records from
    different ranks never interleave.  The update cadence is
    ``log_every_n``; fsync follows the sink's ``flush_every_records``.
    The record's activation facts are the SCALAR dict the runtime's
    ``measure()`` computed (mid-update, before ``optimizer.step()``); a
    retried update appends to the same per-rank file, so a failed update
    may appear more than once and ``successful_update`` is not strictly
    monotonic within a file (the main metrics stream has the same
    behavior).  Ordinary sink I/O faults warn once and disable this
    recorder; structural/record contract errors still propagate.
    """

    def __init__(
        self,
        path: Path,
        *,
        rank: int,
        world_size: int,
        log_every_n: int,
        flush_every_records: int,
        slot_ids: tuple[int, ...],
    ) -> None:
        _nonnegative_int("rank", rank)
        _positive_int("world_size", world_size)
        _positive_int("log_every_n", log_every_n)
        _positive_int("flush_every_records", flush_every_records)
        if not slot_ids or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slot_ids must be nonempty and unique")
        self._rank = rank
        self._world_size = world_size
        self._log_every_n = log_every_n
        self._slot_ids = slot_ids
        self._disabled = False
        self._closed = False
        self._sink = DurableJsonlSink(
            path, fsync_every_records=flush_every_records
        )

    def record_update(
        self,
        *,
        successful_update: int,
        update: SingleGpuUpdateResult,
        activation_facts: Mapping[str, float | int | None],
    ) -> None:
        """Write one record when the update cadence is due (no-op otherwise)."""

        if self._disabled or successful_update % self._log_every_n != 0:
            return
        record = build_condition_diagnostic_record(
            successful_update=successful_update,
            rank=self._rank,
            world_size=self._world_size,
            update=update,
            activation_facts=activation_facts,
            slot_ids=self._slot_ids,
        )
        try:
            self._sink.write(record)
        except OSError as error:
            # Catch only the optional sink I/O, never record validation or
            # model/loss/optimizer work.  Disable before cleanup/logging so
            # later observations cannot repeatedly hit the broken stream.
            self._disabled = True
            _warn_condition_io("write failed; disabling this rank's stream", error)
            self.close()

    def maybe_record(self, observation: SuccessfulTrainingObservation) -> None:
        """Callback bound to the runtime's per-update observation hook.

        The cadence gate comes FIRST: off-cadence updates pay nothing.
        When due, the record carries the scalar facts the runtime
        computed in ``measure()`` — mid-update, before
        ``optimizer.step()`` — from the LAST microbatch; if the runtime
        did not carry them the stream fails closed instead of guessing.
        """

        if self._disabled:
            return
        successful_update = observation.loop.update.state.successful_updates
        if successful_update % self._log_every_n != 0:
            return
        microbatches = observation.microbatches
        if not microbatches:
            raise ValueError("observation carries no microbatch measurements")
        condition_facts = microbatches[-1].condition_diagnostics
        if condition_facts is None:
            raise ValueError(
                "the runtime did not carry condition activation facts; "
                "the stream is enabled without the runtime capture flag"
            )
        self.record_update(
            successful_update=successful_update,
            update=observation.loop.update,
            activation_facts=condition_facts,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._disabled = True
        self._closed = True
        try:
            self._sink.close()
        except Exception as error:  # noqa: BLE001 - optional cleanup must not mask training
            _warn_condition_io("close failed; ignoring (diagnostics only)", error)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CONDITION_DIAGNOSTICS_SCHEMA_VERSION",
    "ConditionDiagnosticsRecorder",
    "build_condition_diagnostic_record",
    "compute_condition_activation_facts",
    "diagnostics_file_name",
]
