"""BF16-first canonical NS4 with owner-rank FP32 rescue (D2 candidate).

Candidate name: ``hybrid_cmuon_canonical_ns4_fp32_rescue``.

Mechanism (offline-validated; artifacts-fp32-rescue/, see
reports/cmuon-fp32-rescue-audit.md):

  1. BF16 NS4 exactly as the guarded canonical production path (canonical
     owner-rank NS, single batched ceiling sync, pre-write fail-closed).
  2. A chunk whose BF16 result trips the safety ceiling or goes nonfinite
     is recomputed by the OWNER rank only, in pure FP32
     (``cmuon_zeroth_power_fp32``, same coefficients/steps/normalization).
  3. The FP32 result is re-checked (finite + ceiling) and, when safe,
     staged as BF16 (a single rounding at the update boundary) and flows
     through the unchanged canonical broadcast / cross-rank fingerprint /
     two-phase atomic commit. F3: the Moonlight-sane lower band
     (``_RESCUE_SANITY_LOW`` x target, numerically unchanged) is a
     DIAGNOSTIC boundary, not a safety boundary: a finite FP32 delta below
     it (including exactly zero) is accepted UNCHANGED (no clamp, rescale,
     normalization, or fallback) and emits scalar soft-rescue telemetry.
     Rationale (R2 evidence, reports/cmuon-f2-real-capture-r2.md): the
     production below-floor case was a finite structurally-small update on
     a near-rank-1 input (stable rank 1.001, top32 energy 0.99999,
     LOW_RANK_CONFIRMED) — a zero/small finite parameter update is not a
     parameter-safety violation.
  4. Only an FP32-also-failed chunk (nonfinite or above ceiling) fails the
     step: ``CMuonSafetyError`` with zero commits. A bad BF16 result alone
     is NEVER an optimizer failure (no clamp-and-continue, no skip-on-bad,
     no AdamW fallback).

Offline evidence (16-obs shadow at ckpt_97100 + repeats + align + bench):
  - BF16 chaos on the D1 danger class: 15-51% catastrophic per input over
    100 hard-synchronized repeats (delta-rms spikes ~5 orders of magnitude).
  - FP32: 0/700 catastrophic, zero output spread (bit-deterministic).
  - 2657-tensor replay: every BF16 catastrophic input rescued (1/1,
    fp32_failed = 0).
  - 1958 stratified SAFE samples: FP32/BF16 delta-rms ratio p50 0.983,
    update cosine p50 0.9966 (same Muon geometry, not bit-identical).
  - Bench (5 production shapes): FP32 NS4 at 0.92-1.00x BF16 wall clock on
    the target HCU -> no measurable added cost at the expected rescue
    frequency (~1 per observation).

The RETIRED pre-NS low-signal structural guard is NOT part of this
candidate: there is no skip-on-low-signal and no amplitude gate. Every
gradient-bearing chunk enters NS and is protected exclusively by the
post-NS safety checks plus the FP32 rescue. The guard config section is
still required (its per-input reference table drives the checkpoint
schema contract inherited from the guarded canonical base class).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sakuramoon.optim.cmuon import (
    CMuonConfig,
    _build_adamw8bit_for_specs,  # pyright: ignore[reportPrivateUsage]
    cmuon_moonlight_alpha,
    cmuon_zeroth_power_bf16,
    cmuon_zeroth_power_fp32,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.cmuon_hardfail import (
    DEFAULT_EMERGENCY_CAPSULE_ROOT,
    DEFAULT_HARD_FAIL_ROOT,
    HardFailArtifactError,
    _write_tensor_bytes,  # pyright: ignore[reportPrivateUsage]
    build_minimal_capsule_metadata,
    classify_fp32_verdict,
    mirror_capsule,
    publish_minimal_capsule,
)
from sakuramoon.optim.guarded_canonical import (
    _CEILING,
    _FAILURE_NAMES,
    _NONFINITE,
    GUARD_SCHEMA_VERSION,  # noqa: F401 - re-exported for test parity
    OWNER_MAPPING_VERSION,  # noqa: F401 - re-exported for test parity
    GuardedCanonicalGuardConfig,
    HybridCMuonGuardedCanonical,
    _as_dict,
    _f32,
    _PreparedSpec,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sakuramoon.train.step import TrainableComposite

# Rescue band constants (validated offline, D2): the ceiling (10x target,
# identical to the BF16 path) is the SAFETY boundary on top. The
# Moonlight-sane floor (0.05x target) — F3: a DIAGNOSTIC boundary, not a
# safety boundary. A finite FP32 delta below it (including exactly zero)
# is accepted unchanged (the R2 evidence: below-floor production events
# were finite, bit-deterministic, structurally-small updates on
# near-rank-1 inputs); the floor value itself is unchanged. Both are
# fixed constants of the validated criterion, NOT tunable parameters.
_RESCUE_SANITY_LOW = 0.05

# Production location of the LEGACY per-rank forensic JSON (the analysis-only
# failure dump that predates F1). F1 keeps this path by default and only
# redirects it (``legacy_forensic_dir``) so isolated tests never touch the
# live artifacts tree. Telemetry-only; never feeds the safety verdict.
LEGACY_FORENSIC_DIR_DEFAULT = "/sakuramoon-runtime/artifacts/g1"


@dataclass
class _RescueMeta:
    """Owner-local rescue context for one NS input."""

    chunk: torch.Tensor  # BF16 nesterov chunk (fp32-derivable)
    alpha: float
    ns_steps: int
    role: str
    spec_name: str
    chunk_idx: int


@dataclass(frozen=True)
class _FP32Verdict:
    """F1 telemetry: the ORIGINAL FP32 rescue verdict values for one failed
    input (owner rank only). Captured strictly after the verdict branch;
    consumed by the forensic record and the hard-fail artifact. Never feeds
    the verdict itself."""

    original_fp32_delta_rms: float
    original_fp32_finite: bool
    # ``str | None``: the classifier's contract allows None (the "ok" case);
    # the capture site is only reachable on a failed verdict, where it is a
    # reason string — an assert here would risk masking the safety error.
    fp32_failure_reason: str | None


class HybridCMuonCanonicalNS4FP32Rescue(HybridCMuonGuardedCanonical):
    """BF16-first canonical NS4 + owner-rank FP32 rescue (D2).

    Subclasses the guarded canonical base (routing, owner mapping,
    references, state/checkpoint contract) and replaces ``step()`` with the
    BF16-first / FP32-rescue flow. The structural pre-NS skip of the base
    class is intentionally NOT used.
    """

    def __init__(self, **kwargs: object) -> None:
        # F1/F2 telemetry parameters (telemetry-only: never feed the
        # verdict). Popped BEFORE super() because the base __init__ is
        # explicit.
        hard_fail_root = kwargs.pop("hard_fail_artifact_root", None)
        legacy_forensic_dir = kwargs.pop("legacy_forensic_dir", None)
        emergency_capsule_root = kwargs.pop("emergency_capsule_root", None)
        checkpoint_source = kwargs.pop("checkpoint_source", None)
        run_id = kwargs.pop("run_id", None)
        super().__init__(**kwargs)  # pyright: ignore[reportArgumentType]
        self.bf16_attempts = 0
        self.bf16_safety_failures = 0
        self.fp32_attempts = 0
        self.fp32_rescues = 0
        self.fp32_rescue_failures = 0
        self.rescue_by_role: dict[str, int] = {}
        # F3 process-local telemetry (NOT persisted — deliberately absent
        # from _guard_state/_guard load_state_dict to keep the checkpoint
        # schema byte-identical to bb41292). Counts finite FP32 rescues
        # whose delta sat below the diagnostic floor (including exactly
        # zero) and were therefore accepted unchanged.
        self.fp32_low_delta_rescues = 0
        self.fp32_low_delta_by_role: dict[str, int] = {}
        self.hard_fail_artifact_root = (
            Path(str(hard_fail_root))
            if hard_fail_root is not None and str(hard_fail_root)
            else Path(DEFAULT_HARD_FAIL_ROOT)
        )
        # F1: redirect the LEGACY per-rank forensic JSON (analysis-only dump)
        # for isolated test environments; None keeps the production path.
        # Never feeds the safety verdict.
        self.legacy_forensic_dir = (
            Path(str(legacy_forensic_dir))
            if legacy_forensic_dir is not None and str(legacy_forensic_dir)
            else Path(LEGACY_FORENSIC_DIR_DEFAULT)
        )
        # F2: LOCAL-first durable emergency root for the minimal hard-fail
        # capsule (the ONLY write in the failure critical path). Production
        # default is a verified local filesystem on the production host;
        # isolated tests MUST redirect it. F2: the shared forensic root
        # (``hard_fail_artifact_root``) is now the BEST-EFFORT mirror
        # target only — it is never on the critical path.
        self.emergency_capsule_root = (
            Path(str(emergency_capsule_root))
            if emergency_capsule_root is not None and str(emergency_capsule_root)
            else Path(DEFAULT_EMERGENCY_CAPSULE_ROOT)
        )
        # F2 telemetry-only identity (never feeds the verdict): the resume
        # checkpoint path and the run id, recorded in the capsule so a
        # capsule found after a crash is self-identifying.
        self.checkpoint_source = (
            None if checkpoint_source is None else str(checkpoint_source)
        )
        self.run_id = None if run_id is None else str(run_id)
        # F2 process-local step counter (1-based at the failing step;
        # NOT persisted — the persisted identity is ``observations``).
        self._steps_this_process = 0
        # F2 trainer-noted global update identity, refreshed by
        # ``note_forensic_update`` before each ``step()`` (None when the
        # trainer does not call the hook, e.g. isolated harnesses).
        self._forensic_update: tuple[int, int] | None = None

    # -- step: BF16 first, FP32 rescue, two-phase atomic ---------------------

    def step(self) -> None:
        # F2 process-local counter (telemetry only; not persisted — the
        # persisted event identity is ``observations``).
        self._steps_this_process += 1
        self._sync_learning_rate()  # pyright: ignore[reportPrivateUsage]
        self._validate_finite_gradients()  # pyright: ignore[reportPrivateUsage]
        mu = self.cfg.momentum
        lr = self.cfg.lr
        target_delta_rms = 0.2 * lr
        ceiling = 10.0 * target_delta_rms
        rescue_floor = _RESCUE_SANITY_LOW * target_delta_rms
        specs = self.routing.cmuon_specs
        device = self._first_device()
        # Device-side ceiling constant for the batched safety flags: the
        # predicate compares FP64 values (fp32 rms -> fp64 is exact), so the
        # flag decision is bit-identical to the old host-side comparison.
        ceiling_t = torch.tensor(ceiling, device=device, dtype=torch.float64)

        # ---- PHASE 1: PREPARE ---------------------------------------------
        # Identical to the guarded canonical base (momentum EMA + nesterov +
        # chunking + Moonlight alphas + one batched signal sync). The base
        # class's low-signal skip is deliberately absent here: every
        # gradient-bearing chunk is ACTIVE.
        prepared: list[_PreparedSpec | None] = []
        for spec in specs:
            grad = spec.parameter.grad
            if grad is None:
                prepared.append(None)
                continue
            buf = self._momenta[spec.parameter]  # pyright: ignore[reportPrivateUsage]
            grad_md = grad.to(buf.dtype)
            buf.lerp_(grad_md, 1.0 - mu)
            nesterov = grad_md.lerp(buf, mu)
            chunk_size = spec.chunk_size()
            if spec.chunk_count == 1:
                chunks = (nesterov,)
            else:
                chunks = tuple(nesterov.split(chunk_size, dim=spec.chunk_dim))
            rescale = spec.chunk_count if self.cfg.chunk_rescale_sqrt_n else 1
            alphas = tuple(
                cmuon_moonlight_alpha(c.shape[0], c.shape[1], lr, rescale)
                for c in chunks
            )
            cf_chunks = [c.float() for c in chunks]
            prepared.append(
                _PreparedSpec(
                    spec=spec,
                    chunks=chunks,
                    alphas=alphas,
                    ns_steps=self.cfg.ns_steps_for_role(spec.role),
                    sig=[c.pow(2).mean().sqrt() for c in cf_chunks],
                    # sigf (D1 fro floor) is retired for this candidate: the
                    # base step's low-signal skip is intentionally absent, so
                    # no per-chunk L2 norm is computed or read back here.
                    sigf=[],
                )
            )

        n_inputs = sum(p.spec.chunk_count for p in prepared if p is not None)
        sig_flat = [s for p in prepared if p is not None for s in p.sig]
        sig_vals = torch.stack(sig_flat).tolist() if sig_flat else []

        # Owner NS (BF16) + staged deltas + failure flags.
        fail_flags = torch.zeros(n_inputs, dtype=torch.int64, device=device)
        staged: list[torch.Tensor | None] = []
        owners: list[int] = []
        is_active: list[bool] = []
        sig_by_input: list[float | None] = []
        rescue_meta: dict[int, _RescueMeta] = {}
        d_rms_list: list[torch.Tensor] = []
        d_rms_owner_idx: list[int] = []
        fi = 0
        for prep in prepared:
            if prep is None:
                continue
            spec = prep.spec
            for ci in range(spec.chunk_count):
                owner = self.owner_of(spec.name, ci)
                sig = sig_vals[fi]
                fi += 1
                sig_by_input.append(sig)
                owners.append(owner)
                is_active.append(True)
                staged.append(None)
                if owner == self.rank:
                    chunk = prep.chunks[ci]
                    self.bf16_attempts += 1
                    ns = cmuon_zeroth_power_bf16(
                        chunk, prep.ns_steps, self.cfg.ns_coefficients, self.cfg.eps
                    )
                    delta = ((-prep.alphas[ci]) * ns).contiguous()
                    staged[fi - 1] = delta
                    # Device-side safety flags (no host sync per chunk):
                    # nonfinite first, ceiling second (only where not already
                    # flagged) — owner-local codes identical to the pre-cleanup
                    # path. The ceiling compare runs in FP64 exactly like the
                    # old host-side `float(rms) > ceiling` (fp32->fp64 is
                    # exact, so the predicate is bit-identical).
                    rms = delta.float().pow(2).mean().sqrt()
                    d_rms_list.append(rms)
                    d_rms_owner_idx.append(fi - 1)
                    fail_flags[fi - 1] = torch.where(
                        ~torch.isfinite(delta).all(),
                        _NONFINITE,
                        fail_flags[fi - 1],
                    )
                    # ceiling only where not already flagged: an inf delta has
                    # rms == inf (inf > ceiling is TRUE), so without the guard
                    # _CEILING would clobber _NONFINITE and the flag code would
                    # drift from the pre-cleanup decision (the old code
                    # `continue`d nonfinite chunks before any ceiling check).
                    fail_flags[fi - 1] = torch.where(
                        (rms.double() > ceiling_t) & (fail_flags[fi - 1] == 0),
                        _CEILING,
                        fail_flags[fi - 1],
                    )
                    rescue_meta[fi - 1] = _RescueMeta(
                        chunk=chunk,
                        alpha=prep.alphas[ci],
                        ns_steps=prep.ns_steps,
                        role=spec.role,
                        spec_name=spec.name,
                        chunk_idx=ci,
                    )

        # ---- FP32 RESCUE (owner rank only) ---------------------------------
        # Runs on the OWNER-LOCAL flags, before the cross-rank all_reduce:
        # the owner is the only rank that has the NS input and the only rank
        # that may recompute it (canonical owner semantics). A successful
        # rescue clears the flag so the rescued (single-rounded) BF16 delta
        # flows through the unchanged broadcast/fingerprint/commit path.
        # The owner-local flags are read back ONCE, packed (the pre-cleanup
        # loop called .item() per chunk on every rank); non-owner ranks skip
        # the read entirely and reach the same all_reduce below.
        rescued_this_step = 0
        # F1 telemetry: the ORIGINAL FP32 verdict values per failed input
        # (owner rank only). Captured strictly after the verdict branch;
        # consumed by the forensic record and the hard-fail artifact.
        fp32_verdicts: dict[int, _FP32Verdict] = {}
        if any(owner == self.rank for owner in owners):
            flags_host = fail_flags.tolist()
            for idx, flag in enumerate(flags_host):
                if flag <= 0 or not is_active[idx] or owners[idx] != self.rank:
                    continue
                meta = rescue_meta[idx]
                self.fp32_attempts += 1
                self.bf16_safety_failures += 1
                ns32 = cmuon_zeroth_power_fp32(
                    meta.chunk.float(),
                    meta.ns_steps,
                    self.cfg.ns_coefficients,
                    self.cfg.eps,
                )
                delta32 = (-meta.alpha) * ns32
                # One packed readback for the rescue verdict (was two
                # separate syncs); tolist() on FP32 scalars yields the exact
                # same Python floats the old float()/bool() calls produced,
                # so the verdict comparisons are bit-identical.
                rms32, finite32 = torch.stack(
                    (
                        delta32.pow(2).mean().sqrt(),
                        torch.isfinite(delta32).all().to(torch.float32),
                    )
                ).tolist()
                if not bool(finite32) or rms32 > ceiling:
                    # FP32 also failed: keep the flag -> hard fail-closed
                    # below. F3: ONLY nonfinite and above_ceiling are safety
                    # failures; a finite below-floor delta is NOT a failure
                    # (accept unchanged, below).
                    self.fp32_rescue_failures += 1
                    # F1 telemetry: persist the ORIGINAL FP32 verdict values
                    # (rms32/finite32 are locals of this loop and would
                    # otherwise be lost). The failure reason follows the
                    # verdict's condition order exactly (nonfinite ->
                    # below_floor -> above_ceiling); it only ever appears in
                    # report fields and never feeds the verdict. After F3 the
                    # branch is only reachable for nonfinite / above_ceiling.
                    fp32_verdicts[idx] = _FP32Verdict(
                        original_fp32_delta_rms=float(rms32),
                        original_fp32_finite=bool(finite32),
                        fp32_failure_reason=classify_fp32_verdict(
                            bool(finite32), float(rms32), rescue_floor, ceiling
                        ),
                    )
                    continue
                # Accept the ORIGINAL FP32 delta: single BF16 rounding at the
                # update boundary, then the unchanged owner broadcast /
                # fingerprint / two-phase commit. No amplitude correction of
                # any kind (F3: finite below-floor is accepted unchanged).
                staged[idx] = delta32.bfloat16().contiguous()
                fail_flags[idx] = 0
                self.fp32_rescues += 1
                rescued_this_step += 1
                self.rescue_by_role[meta.role] = (
                    self.rescue_by_role.get(meta.role, 0) + 1
                )
                if rms32 < rescue_floor:
                    # F3 diagnostic telemetry for the soft below-floor band
                    # (scalar only: no replay, SVD, or tensor dump; never
                    # feeds the verdict; process-local, not persisted).
                    # A finite zero delta is a degenerate-but-safe update,
                    # not a parameter-safety violation.
                    reason = (
                        "zero_delta_soft_rescue"
                        if rms32 == 0.0
                        else "below_floor_soft_rescue"
                    )
                    self.fp32_low_delta_rescues += 1
                    self.fp32_low_delta_by_role[meta.role] = (
                        self.fp32_low_delta_by_role.get(meta.role, 0) + 1
                    )
                    if self.stats_logger is not None:
                        fqn, cix = self._input_key(idx)
                        self.stats_logger(
                            f"[fp32-soft-rescue] reason={reason} "
                            f"fqn={fqn}#chunk{cix} role={meta.role} "
                            f"fp32_delta_rms={rms32:.6e} "
                            f"target_delta_rms={target_delta_rms:.6e} "
                            f"rescue_floor={rescue_floor:.6e} "
                            f"delta/target={rms32 / target_delta_rms:.6g} "
                            f"delta/floor={rms32 / rescue_floor:.6g} "
                            f"u_t_rms={sig_by_input[idx]}"
                        )
        if rescued_this_step and self.stats_logger is not None:
            self.stats_logger(
                json.dumps(
                    {
                        "fp32_rescue_obs": self.observations,
                        "rescued_this_step": rescued_this_step,
                        "bf16_attempts": self.bf16_attempts,
                        "bf16_safety_failures": self.bf16_safety_failures,
                        "fp32_attempts": self.fp32_attempts,
                        "fp32_rescues": self.fp32_rescues,
                        "fp32_rescue_failures": self.fp32_rescue_failures,
                        "rescue_by_role": dict(sorted(self.rescue_by_role.items())),
                        # F3: process-local (not persisted) soft below-floor
                        # counts, so production logs can separate normal
                        # rescues from accepted small-delta rescues.
                        "fp32_low_delta_rescues": self.fp32_low_delta_rescues,
                        "fp32_low_delta_by_role": dict(
                            sorted(self.fp32_low_delta_by_role.items())
                        ),
                    }
                )
            )

        # Rank-consistent failure verdict (every rank learns every failure).
        if self.world_size > 1:
            dist.all_reduce(fail_flags, op=dist.ReduceOp.MAX)
        # Single host verdict read (the pre-cleanup path read the same tensor
        # back twice, once for the messages and once for the `failed` set).
        final_flags = fail_flags.tolist()
        failure_msgs = []
        for idx, flag in enumerate(final_flags):
            if flag > 0:
                fqn, chunk_idx = self._input_key(idx)
                failure_msgs.append(
                    f"{_FAILURE_NAMES[flag]}: {fqn}#chunk{chunk_idx}"
                )

        # Broadcast: every ACTIVE, non-failed chunk is sent owner -> all
        # ranks (rescued deltas included; they are plain BF16 tensors).
        failed = {idx for idx, flag in enumerate(final_flags) if flag > 0}
        if self.world_size > 1:
            for idx, (a, owner) in enumerate(zip(is_active, owners)):
                if not a or idx in failed:
                    staged[idx] = None  # zero delta on every rank (consensus)
                    continue
                shape = self._chunk_shape(idx)
                if owner == self.rank:
                    tensor = staged[idx]
                    assert tensor is not None
                    dist.broadcast(tensor.contiguous(), src=owner)
                else:
                    buf = torch.empty(shape, dtype=torch.bfloat16, device=device)
                    dist.broadcast(buf, src=owner)
                    staged[idx] = buf
        else:
            for idx, a in enumerate(is_active):
                if not a or idx in failed:
                    staged[idx] = None

        # Cross-rank delta fingerprint (rescued deltas must fingerprint
        # identically on every rank after the canonical broadcast).
        if self.world_size > 1:
            fp = []
            for idx, a in enumerate(is_active):
                if not a or idx in failed:
                    continue
                df = staged[idx].float()  # pyright: ignore[union-attr]
                fp.append(df.pow(2).mean().sqrt())
                fp.append(df.abs().max())
            if fp:
                flat = torch.stack(fp)
                lo = flat.clone()
                hi = flat.clone()
                dist.all_reduce(lo, op=dist.ReduceOp.MIN)
                dist.all_reduce(hi, op=dist.ReduceOp.MAX)
                spread = float((hi - lo).max().item())
                self.max_delta_rank_spread = max(self.max_delta_rank_spread, spread)
                if spread != 0.0:
                    failure_msgs.append(
                        f"cross-rank delta fingerprint spread {spread:.3e} "
                        "after canonical broadcast"
                    )

        if failure_msgs:
            # F1: owner-local BF16 delta rms by input (pure readback, no
            # I/O) — hoisted out of the dump try so the exact-input
            # artifact publish below can use it even if the legacy JSON
            # write fails.
            d_rms_vals = torch.stack(d_rms_list).tolist() if d_rms_list else []
            d_rms_by_idx = dict(zip(d_rms_owner_idx, d_rms_vals))
            # ---- forensic dump (analysis-only; fail-closed unchanged) ----
            try:
                recs = []
                for idx, flag in enumerate(fail_flags.tolist()):
                    if flag <= 0:
                        continue
                    fqn, cix = self._input_key(idx)
                    shape = self._chunk_shape(idx)
                    verdict = fp32_verdicts.get(idx)
                    recs.append(
                        {
                            "failure": _FAILURE_NAMES.get(int(flag), str(flag)),
                            "fqn": fqn,
                            "chunk": cix,
                            "owner": owners[idx] if idx < len(owners) else None,
                            "this_rank": self.rank,
                            "shape": list(shape),
                            "numel": int(math.prod(shape)),
                            "u_t_rms": sig_by_input[idx],
                            "lr": lr,
                            "target_delta_rms": target_delta_rms,
                            "ceiling": ceiling,
                            # F1: explicit BF16 field; `delta_rms` is kept as
                            # a DEPRECATED alias (old consumers) — both are
                            # the ORIGINAL BF16 NS attempt delta rms (null on
                            # non-owner ranks, which never run the BF16 NS).
                            "bf16_delta_rms": d_rms_by_idx.get(idx),
                            "delta_rms": d_rms_by_idx.get(idx),
                            # F1: the ORIGINAL FP32 rescue verdict values
                            # (owner rank only; null on non-owner ranks).
                            "fp32_delta_rms": (
                                None
                                if verdict is None
                                else verdict.original_fp32_delta_rms
                            ),
                            "fp32_finite": (
                                None
                                if verdict is None
                                else verdict.original_fp32_finite
                            ),
                            "fp32_failure_reason": (
                                None
                                if verdict is None
                                else verdict.fp32_failure_reason
                            ),
                            "fp32_rescue_floor": rescue_floor,
                            "fp32_ceiling": ceiling,
                            "fp32_attempts": self.fp32_attempts,
                            "fp32_rescues": self.fp32_rescues,
                            "fp32_rescue_failures": self.fp32_rescue_failures,
                        }
                    )
                import os as _os

                out_dir = str(self.legacy_forensic_dir)
                _os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/guard-forensic-rank{self.rank}.json"
                with open(out_path, "w") as _f:
                    json.dump(
                        {"observations": self.observations, "records": recs},
                        _f,
                        indent=1,
                    )
                if self.stats_logger is not None:
                    self.stats_logger("[guard-forensic] " + " | ".join(failure_msgs))
            except Exception as exc:  # noqa: BLE001 - analysis must never mask the real failure
                try:
                    if self.stats_logger is not None:
                        self.stats_logger(f"[guard-forensic] dump failed: {exc!r}")
                except Exception:  # noqa: BLE001
                    import sys as _sys

                    print(f"[guard-forensic] log failed: {exc!r}", file=_sys.stderr)
            # ---- F2: minimal exact-input capsule (owner rank only) -------
            # Published strictly after the rank-consistent failure verdict.
            # Telemetry-only: nothing here feeds the flag, staged delta,
            # momentum, parameters, owner broadcast, or commit. The
            # critical path is now: freeze exact input -> ONE device->CPU
            # transfer -> serialize+sha -> minimal metadata -> LOCAL atomic
            # publish (durable) -> best-effort shared mirror. NO CPU NS
            # replay / trace / SVD (F1's pre-write replays were what let
            # the 2-rank elastic teardown kill the owner before any byte
            # landed). A publish failure is logged; the original
            # CMuonSafetyError below still raises (telemetry I/O can never
            # replace the root cause).
            for idx, flag in enumerate(fail_flags.tolist()):
                if flag <= 0 or owners[idx] != self.rank:
                    continue
                meta = rescue_meta.get(idx)
                if meta is None:
                    continue
                try:
                    self._publish_minimal_hardfail_capsule(
                        idx=idx,
                        meta=meta,
                        flag=int(flag),
                        verdict=fp32_verdicts.get(idx),
                        bf16_delta_rms=d_rms_by_idx.get(idx),
                        failure_message=" | ".join(failure_msgs),
                        ceiling=ceiling,
                        rescue_floor=rescue_floor,
                        lr=lr,
                        target_delta_rms=target_delta_rms,
                    )
                except HardFailArtifactError as exc:
                    try:
                        if self.stats_logger is not None:
                            self.stats_logger(f"[hard-fail-artifact] {exc!r}")
                    except Exception:  # noqa: BLE001
                        import sys as _sys

                        print(f"[hard-fail-artifact] {exc!r}", file=_sys.stderr)
            raise CMuonSafetyError(
                "fp32-rescue safety violation (BF16 and FP32 both failed): "
                + " | ".join(failure_msgs)
            )

        # ---- PHASE 2: COMMIT ------------------------------------------------
        # AdamW part (production path under the isolated SR RNG).
        self.sr_rng.run_step(self.optimizer.step)
        # CMuon deltas: reassemble per spec exactly like the core update.
        with torch.no_grad():
            flat = 0
            for prep in prepared:
                if prep is None:
                    continue
                spec = prep.spec
                spec_active = any(
                    is_active[flat + ci] for ci in range(spec.chunk_count)
                )
                if not spec_active:
                    flat += spec.chunk_count
                    continue
                deltas = [
                    staged[flat + ci]
                    if is_active[flat + ci]
                    else torch.zeros_like(prep.chunks[ci])
                    for ci in range(spec.chunk_count)
                ]
                update_ortho = (
                    deltas[0]
                    if spec.chunk_count == 1
                    else torch.cat(deltas, dim=spec.chunk_dim)
                )
                if spec.weight_decay != 0.0:
                    spec.parameter.mul_(1.0 - lr * spec.weight_decay)
                spec.parameter.add_(update_ortho)
                flat += spec.chunk_count

        # Reference updates (all inputs are active in this candidate; the
        # reference table stays in the checkpoint contract).
        fi = 0
        for prep in prepared:
            if prep is None:
                continue
            spec = prep.spec
            for ci in range(spec.chunk_count):
                key = (spec.name, ci)
                sig = sig_by_input[fi]
                assert sig is not None
                ref = self._refs[key]
                self._refs[key] = max(
                    _f32(sig), _f32(ref * self.guard_cfg.reference_decay)
                )
                fi += 1

        self.observations += 1

        # Optional per-step rank invariant: post-commit parameter fingerprints
        # must be identical on every rank (rescued deltas included).
        if self.guard_cfg.invariant_check and self.world_size > 1:
            fp = []
            for spec in specs:
                pf = spec.parameter.float()
                fp.append(pf.pow(2).mean().sqrt())
                fp.append(pf.abs().max())
            flat = torch.stack(fp)
            lo = flat.clone()
            hi = flat.clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            diff = float((hi - lo).max().item())
            self.max_param_rank_diff = max(self.max_param_rank_diff, diff)
            if diff != 0.0:
                raise CMuonSafetyError(
                    "fp32-rescue rank invariant violated: "
                    f"cross-rank parameter fingerprint diff {diff:.3e} after commit"
                )

        if self.stats_logger is not None and (
            self.observations % self.stats_log_every_n == 0
        ):
            self.stats_logger(
                json.dumps(
                    {
                        "observations": self.observations,
                        "bf16_attempts": self.bf16_attempts,
                        "bf16_safety_failures": self.bf16_safety_failures,
                        "fp32_attempts": self.fp32_attempts,
                        "fp32_rescues": self.fp32_rescues,
                        "fp32_rescue_failures": self.fp32_rescue_failures,
                        "rescue_by_role": dict(sorted(self.rescue_by_role.items())),
                        # F3: process-local (not persisted) soft below-floor
                        # counts (see the per-rescue block above).
                        "fp32_low_delta_rescues": self.fp32_low_delta_rescues,
                        "fp32_low_delta_by_role": dict(
                            sorted(self.fp32_low_delta_by_role.items())
                        ),
                        "ref_min": min(self._refs.values()),
                        "ref_max": max(self._refs.values()),
                        "max_delta_rank_spread": self.max_delta_rank_spread,
                        "max_param_rank_diff": self.max_param_rank_diff,
                    }
                )
            )

    # -- F2: fast minimal hard-fail capsule (owner rank, post-verdict) ------

    def note_forensic_update(
        self, *, last_successful_update: int, attempted_update: int
    ) -> None:
        """Trainer hook (F2 telemetry): record the global update identity
        immediately before ``step()`` so the minimal hard-fail capsule can
        carry it. Consumed ONLY by the capsule; never feeds the verdict.
        Duck-typed by the trainer (``getattr`` no-op for optimizers that
        do not implement it)."""
        self._forensic_update = (int(last_successful_update), int(attempted_update))

    def _publish_minimal_hardfail_capsule(
        self,
        *,
        idx: int,
        meta: _RescueMeta,
        flag: int,
        verdict: _FP32Verdict | None,
        bf16_delta_rms: float | None,
        failure_message: str,
        ceiling: float,
        rescue_floor: float,
        lr: float,
        target_delta_rms: float,
    ) -> None:
        """Owner-only MINIMAL exact-input capsule (F2 telemetry).

        Runs strictly AFTER the rank-consistent failure verdict is settled:
        it cannot change the fail flag, staged delta, momentum, parameters,
        owner broadcast, or commit. The critical path is exactly:

          1. freeze the exact NS input (``detach().contiguous().clone()``
             on the original device/dtype — a fresh storage that live
             momentum/grad mutations can never touch),
          2. ONE device->CPU transfer,
          3. serialize + sha256 over the exact file bytes,
          4. minimal metadata (the four existing O(n) scalar reductions
             plus the recorded original verdict values),
          5. LOCAL atomic capsule publish (durable, emergency root),
          6. best-effort shared mirror (never raises, never on the
             critical path's durability guarantee).

        There is NO CPU NS replay, trace, SVD, or other expensive scan
        before the local capsule is durable — that is what made F1 lose
        the 2-rank elastic-teardown race (112105). All diagnostics are
        OFFLINE via ``dev-tools/cmuon_hardfail_enrich.py``. ANY failure in
        this method is converted to ``HardFailArtifactError``; the caller
        logs it and the original ``CMuonSafetyError`` is still raised
        (telemetry can never replace the root cause).
        """
        try:
            self._publish_minimal_hardfail_capsule_inner(
                idx=idx,
                meta=meta,
                flag=flag,
                verdict=verdict,
                bf16_delta_rms=bf16_delta_rms,
                failure_message=failure_message,
                ceiling=ceiling,
                rescue_floor=rescue_floor,
                lr=lr,
                target_delta_rms=target_delta_rms,
            )
        except HardFailArtifactError:
            raise
        except Exception as exc:
            raise HardFailArtifactError(
                f"minimal hard-fail capsule preparation failed for "
                f"{meta.spec_name}#chunk{meta.chunk_idx}: {exc!r}"
            ) from exc

    def _publish_minimal_hardfail_capsule_inner(
        self,
        *,
        idx: int,
        meta: _RescueMeta,
        flag: int,
        verdict: _FP32Verdict | None,
        bf16_delta_rms: float | None,
        failure_message: str,
        ceiling: float,
        rescue_floor: float,
        lr: float,
        target_delta_rms: float,
    ) -> None:
        # 1. Freeze the EXACT NS input IMMEDIATELY: a fresh contiguous
        #    storage on the original device/dtype. It is the BF16 nesterov
        #    chunk the production BF16 NS consumed (and, via the exact
        #    BF16->FP32 cast, the input the FP32 rescue recomputed from);
        #    the clone cannot be altered by any later live mutation.
        frozen_input = meta.chunk.detach().contiguous().clone()
        # 2. The ONE device->CPU transfer (the only synchronizing op here).
        input_tensor = frozen_input.cpu()
        del frozen_input
        # 3. Serialize + sha256 over the exact file bytes (so the capsule
        #    is self-verifying end to end).
        tensor_bytes, tensor_format, _tensor_name = _write_tensor_bytes(input_tensor)
        tensor_sha = hashlib.sha256(tensor_bytes).hexdigest()
        # 4. Minimal metadata (existing scalars only — no replay/trace/SVD).
        forensic_update = self._forensic_update
        metadata = build_minimal_capsule_metadata(
            observations=self.observations,
            this_rank=self.rank,
            world_size=self.world_size,
            fqn=meta.spec_name,
            chunk_idx=meta.chunk_idx,
            role=meta.role,
            owner=self.rank,
            run_id=self.run_id,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            process_steps=self._steps_this_process,
            last_successful_update=(
                None if forensic_update is None else forensic_update[0]
            ),
            attempted_update=None if forensic_update is None else forensic_update[1],
            checkpoint_source=self.checkpoint_source,
            input_tensor=input_tensor,
            alpha=meta.alpha,
            ns_steps=meta.ns_steps,
            ns_coefficients=tuple(self.cfg.ns_coefficients),
            eps=self.cfg.eps,
            lr=lr,
            target_delta_rms=target_delta_rms,
            ceiling=ceiling,
            rescue_floor=rescue_floor,
            bf16_delta_rms=bf16_delta_rms,
            original_fp32_delta_rms=(
                None if verdict is None else verdict.original_fp32_delta_rms
            ),
            original_fp32_finite=(
                None if verdict is None else verdict.original_fp32_finite
            ),
            fp32_failure_reason=(
                None if verdict is None else verdict.fp32_failure_reason
            ),
            bf16_failure_name=_FAILURE_NAMES.get(flag, str(flag)),
            failure_message=failure_message,
            tensor_sha256=tensor_sha,
            tensor_format=tensor_format,
            shared_mirror_root=str(self.hard_fail_artifact_root),
        )
        # 5. LOCAL-first durable capsule (the only critical-path write).
        event_dir = publish_minimal_capsule(
            root=self.emergency_capsule_root,
            observations=self.observations,
            rank=self.rank,
            world_size=self.world_size,
            fqn=meta.spec_name,
            chunk_idx=meta.chunk_idx,
            role=meta.role,
            owner=self.rank,
            input_tensor=input_tensor,
            metadata=metadata,
        )
        # 6. Best-effort shared mirror (never raises; the local capsule is
        #    already durable and remains the success evidence either way).
        mirror_result = mirror_capsule(event_dir, self.hard_fail_artifact_root)
        try:
            if self.stats_logger is not None:
                self.stats_logger(
                    f"[hard-fail-capsule] {event_dir.name} "
                    f"local=durable mirror={mirror_result.get('status')}"
                )
        except Exception:  # noqa: BLE001, S110 - a logger failure can never
            # mask the safety failure; the capsule is already durable.
            pass

    # -- state (rescue counters ride inside the inherited guard block) -------

    def _guard_state(self) -> dict[str, object]:
        state = super()._guard_state()
        state["fp32_rescue"] = {
            "bf16_attempts": self.bf16_attempts,
            "bf16_safety_failures": self.bf16_safety_failures,
            "fp32_attempts": self.fp32_attempts,
            "fp32_rescues": self.fp32_rescues,
            "fp32_rescue_failures": self.fp32_rescue_failures,
            "rescue_by_role": dict(sorted(self.rescue_by_role.items())),
        }
        return state

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        super().load_state_dict(state_dict)
        if "guard" in state_dict:
            guard = _as_dict(state_dict["guard"], "guard state")
            rescue = guard.get("fp32_rescue")
            if rescue is not None:
                r = _as_dict(rescue, "fp32_rescue state")
                self.bf16_attempts = int(r.get("bf16_attempts", 0))
                self.bf16_safety_failures = int(r.get("bf16_safety_failures", 0))
                self.fp32_attempts = int(r.get("fp32_attempts", 0))
                self.fp32_rescues = int(r.get("fp32_rescues", 0))
                self.fp32_rescue_failures = int(r.get("fp32_rescue_failures", 0))
                self.rescue_by_role = {
                    str(k): int(v)
                    for k, v in _as_dict(r.get("rescue_by_role"), "rescue_by_role").items()
                }
        # Absent fp32_rescue block = a parent-class checkpoint: counters
        # stay at their construction values (telemetry restarts at zero;
        # the resume semantics are unaffected).


def build_fp32_rescue(
    module: TrainableComposite,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    block_size: int,
    bf16_stochastic_round: bool,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
    sr_seed: int,
    ns_steps_by_role: Mapping[str, int],
    guard_cfg: GuardedCanonicalGuardConfig,
    guard_bootstrap_refs: Mapping[str, float],
    rank: int,
    world_size: int,
    momentum_dtype: str = "bfloat16",
    chunk_rescale_sqrt_n: bool = False,
    stats_logger: Callable[[str], None] | None = None,
    stats_log_every_n: int = 10,
    hard_fail_artifact_root: str | None = None,
    legacy_forensic_dir: str | None = None,
    emergency_capsule_root: str | None = None,
    checkpoint_source: str | None = None,
    run_id: str | None = None,
) -> HybridCMuonCanonicalNS4FP32Rescue:
    """Build the FP32-rescue candidate (same routing/AdamW policy as the
    guarded canonical base; BF16-first NS + owner-rank FP32 rescue).

    ``hard_fail_artifact_root`` (F1/F2 telemetry only): the BEST-EFFORT
    shared mirror target for hard-fail capsules; None keeps the production
    default (``/sakuramoon-runtime/artifacts/g1/cmuon-hard-fail``). Since
    F2 it is never on the failure critical path (the local capsule lands
    first). It never feeds the safety verdict.

    ``legacy_forensic_dir`` (F1 telemetry only): redirect the legacy
    per-rank forensic JSON (the analysis-only failure dump) for isolated
    test environments; None keeps the production default (``/sakuramoon-
    runtime/artifacts/g1``). It never feeds the safety verdict.

    ``emergency_capsule_root`` (F2 telemetry only): the LOCAL-first
    durable emergency root where the minimal hard-fail capsule is
    published BEFORE any shared mirror; None keeps the production default
    (``/sakuramoon-runtime/cmuon-f1-emergency``, a verified local
    filesystem on the production host). Isolated tests MUST redirect it.
    It never feeds the safety verdict.

    ``checkpoint_source`` / ``run_id`` (F2 telemetry only): identity
    strings recorded in the capsule (the resume checkpoint path and the
    run id); None keeps null fields. They never feed the safety verdict.
    """
    routing = route_cmuon_parameters(
        module,
        matrix_weight_decay=matrix_weight_decay,
        sensitive_weight_decay=sensitive_weight_decay,
    )
    canonical_ns = resolve_ns_map(dict(ns_steps_by_role), None)
    cfg = CMuonConfig(
        lr=lr,
        ns_steps_by_role=canonical_ns,
        momentum_dtype=momentum_dtype,  # type: ignore[arg-type]
        chunk_rescale_sqrt_n=chunk_rescale_sqrt_n,
    )
    adamw = _build_adamw8bit_for_specs(
        routing.adamw_specs,
        lr=lr,
        betas=betas,
        eps=eps,
        block_size=block_size,
        bf16_stochastic_round=bf16_stochastic_round,
        matrix_weight_decay=matrix_weight_decay,
        sensitive_weight_decay=sensitive_weight_decay,
        sr_seed=sr_seed,
    )
    return HybridCMuonCanonicalNS4FP32Rescue(
        routing=routing,
        cfg=cfg,
        adamw=adamw,
        sr_rng=adamw.sr_rng,
        guard_cfg=guard_cfg,
        rank=rank,
        world_size=world_size,
        guard_bootstrap_refs=guard_bootstrap_refs,
        stats_logger=stats_logger,
        stats_log_every_n=stats_log_every_n,
        hard_fail_artifact_root=hard_fail_artifact_root,
        legacy_forensic_dir=legacy_forensic_dir,
        emergency_capsule_root=emergency_capsule_root,
        checkpoint_source=checkpoint_source,
        run_id=run_id,
    )


__all__ = [
    "HybridCMuonCanonicalNS4FP32Rescue",
    "build_fp32_rescue",
]
