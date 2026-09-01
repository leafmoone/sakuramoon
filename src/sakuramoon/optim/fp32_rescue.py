"""BF16-first canonical NS4 with owner-rank FP32 rescue (D2 candidate).

Candidate name: ``hybrid_cmuon_canonical_ns4_fp32_rescue``.

Mechanism (offline-validated; artifacts-fp32-rescue/, see
reports/cmuon-fp32-rescue-audit.md):

  1. BF16 NS4 exactly as the guarded canonical production path (canonical
     owner-rank NS, single batched ceiling sync, pre-write fail-closed).
  2. A chunk whose BF16 result trips the safety ceiling or goes nonfinite
     is recomputed by the OWNER rank only, in pure FP32
     (``cmuon_zeroth_power_fp32``, same coefficients/steps/normalization).
  3. The FP32 result is re-checked (finite + Moonlight-sane lower band +
     ceiling) and, when safe, staged as BF16 (a single rounding at the
     update boundary) and flows through the unchanged canonical
     broadcast / cross-rank fingerprint / two-phase atomic commit.
  4. Only an FP32-also-failed chunk fails the step: ``CMuonSafetyError``
     with zero commits. A bad BF16 result alone is NEVER an optimizer
     failure (no clamp-and-continue, no skip-on-bad, no AdamW fallback).

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

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

# Rescue safety band (validated offline, D2): the FP32 delta must be finite
# and sit inside [0.05, 10.0] x (0.2*lr) — the ceiling (10x, identical to
# the BF16 path) on top, a Moonlight-sane floor (0.05x) that rejects
# degenerate zero-energy NS outputs. These are fixed constants of the
# validated criterion, NOT tunable parameters.
_RESCUE_SANITY_LOW = 0.05


@dataclass
class _RescueMeta:
    """Owner-local rescue context for one NS input."""

    chunk: torch.Tensor  # BF16 nesterov chunk (fp32-derivable)
    alpha: float
    ns_steps: int
    role: str
    spec_name: str
    chunk_idx: int


class HybridCMuonCanonicalNS4FP32Rescue(HybridCMuonGuardedCanonical):
    """BF16-first canonical NS4 + owner-rank FP32 rescue (D2).

    Subclasses the guarded canonical base (routing, owner mapping,
    references, state/checkpoint contract) and replaces ``step()`` with the
    BF16-first / FP32-rescue flow. The structural pre-NS skip of the base
    class is intentionally NOT used.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # pyright: ignore[reportArgumentType]
        self.bf16_attempts = 0
        self.bf16_safety_failures = 0
        self.fp32_attempts = 0
        self.fp32_rescues = 0
        self.fp32_rescue_failures = 0
        self.rescue_by_role: dict[str, int] = {}

    # -- step: BF16 first, FP32 rescue, two-phase atomic ---------------------

    def step(self) -> None:
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
                    fail_flags[fi - 1] = torch.where(
                        rms.double() > ceiling_t,
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
                if (
                    not bool(finite32)
                    or rms32 < rescue_floor
                    or rms32 > ceiling
                ):
                    # FP32 also failed: keep the flag -> hard fail-closed below.
                    self.fp32_rescue_failures += 1
                    continue
                staged[idx] = delta32.bfloat16().contiguous()
                fail_flags[idx] = 0
                self.fp32_rescues += 1
                rescued_this_step += 1
                self.rescue_by_role[meta.role] = (
                    self.rescue_by_role.get(meta.role, 0) + 1
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
            # ---- forensic dump (analysis-only; fail-closed unchanged) ----
            try:
                d_rms_vals = (
                    torch.stack(d_rms_list).tolist() if d_rms_list else []
                )
                d_rms_by_idx = dict(zip(d_rms_owner_idx, d_rms_vals))
                recs = []
                for idx, flag in enumerate(fail_flags.tolist()):
                    if flag <= 0:
                        continue
                    fqn, cix = self._input_key(idx)
                    shape = self._chunk_shape(idx)
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
                            "delta_rms": d_rms_by_idx.get(idx),
                            "fp32_attempts": self.fp32_attempts,
                            "fp32_rescues": self.fp32_rescues,
                            "fp32_rescue_failures": self.fp32_rescue_failures,
                        }
                    )
                import os as _os

                out_dir = "/sakuramoon-runtime/artifacts/g1"
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
                        "ref_min": min(self._refs.values()),
                        "ref_max": max(self._refs.values()),
                        "max_delta_rank_spread": self.max_delta_rank_spread,
                        "max_param_rank_diff": self.max_param_rank_diff,
                    }
                )
            )

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
) -> HybridCMuonCanonicalNS4FP32Rescue:
    """Build the FP32-rescue candidate (same routing/AdamW policy as the
    guarded canonical base; BF16-first NS + owner-rank FP32 rescue)."""
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
    )


__all__ = [
    "HybridCMuonCanonicalNS4FP32Rescue",
    "build_fp32_rescue",
]
