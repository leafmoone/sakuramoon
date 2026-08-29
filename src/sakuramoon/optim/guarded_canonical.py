"""Guarded Canonical Hybrid CMuon v1 — the ``hybrid_cmuon_guarded_canonical_ns4``
candidate.

Replaces the PERMANENTLY RETIRED original candidate (``hybrid_cmuon_ns4_core``)
with four simultaneous fixes for the root-caused failure mode (see
``reports/cmuon-root-cause.md``):

A. **Low-signal guard (pre-NS)** — the NS input is the Nesterov matrix
   ``u = (1-mu)·g + mu·m``. When ``rms(u) < guard_ratio · ref`` (per NS
   input, rank-consistent) or ``fro(u) < numerical_floor``, that NS input
   is skipped: NO Newton-Schulz, ZERO parameter delta, momentum EMA still
   updates. v1 forbids falling back to AdamW on low signal. This eliminates
   the chaotic-boundary class: rank-1 weak-signal matrices (top-1 sigma^2
   ~ 89%) whose normalized form sits on the NS4 convergence boundary, where
   iteration 4 bifurcates chaotically in the Gram bits (same input bits ->
   HCU realizations fro {125235, 16.3, 15.9, ...}).
B. **Canonical owner-rank NS** — one deterministic owner per NS input
   (``stable_hash(fqn#chunk) % world_size``) runs the NS; the canonical
   delta is broadcast; ALL ranks apply the SAME bits. The platform NS
   nondeterminism can no longer convert into DDP parameter nondeterminism.
C. **Fail-closed before any parameter write** — PHASE 1 (prepare) validates
   gradients, updates the momentum candidates, makes guard decisions, runs
   the owner NS, safety-validates EVERY delta (finite, ``delta_rms <=
   10 * 0.2 * lr``, no clamp), reaches a rank-consistent failure verdict
   (all-reduce of per-NS-input failure flags), broadcasts, and cross-rank
   fingerprint checks. Any failure => NO parameter is updated (CMuon
   deltas unapplied, AdamW NOT stepped) and the run stops
   (``CMuonSafetyError`` on every rank).
D. **Atomic commit** — PHASE 2 (commit) applies the AdamW step, all CMuon
   deltas, and the momentum state only when PHASE 1 fully passed, followed
   by a parameter-fingerprint verification (config-gated; on for S1/S2).

The signal reference ``ref`` is a per-(FQN, chunk) FP32 scalar:

    active:   ref_t = max(sig_t, ref_{t-1} * reference_decay)
    inactive: ref_t = ref_{t-1}            (never decays to zero)

initialized by the calibration bootstrap (``[optimizer.cmuon_guard]
references`` — the P3 shadow-gradient calibration artifact; see
``reports/cmuon-guarded-canonical-design.md`` §5). The guard is inactive
for the first ``warmup_observations`` updates after the AdamW -> guarded
transition (optimizer-transition bootstrap, NOT LR warmup).

Checkpoint semantics: guarded -> guarded resume is state-exact
(references / counters / momentum / owner-mapping version / world_size must
all match). An old UNGUARDED CMuon checkpoint can NOT resume directly into
this candidate (explicit optimizer transition + guard re-bootstrap
required). A pure AdamW checkpoint transitions: AdamW state preserved
per-FQN, CMuon momentum from zero, references from the bootstrap table.
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
    CMuonChunkSpec,
    CMuonConfig,
    HybridCMuon,
    _build_adamw8bit_for_specs,  # pyright: ignore[reportPrivateUsage]
    cmuon_moonlight_alpha,
    cmuon_zeroth_power,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
    from sakuramoon.optim.cmuon import CMuonRouting
    from sakuramoon.train.step import TrainableComposite


#: Guard state schema (checkpoint "guard" block + optimizer_schema block).
GUARD_SCHEMA_VERSION = 1
#: Owner-mapping rule version (checkpoint contract; a change breaks
#: guarded->guarded resume compatibility by design).
OWNER_MAPPING_VERSION = "fnv1a64-v1"
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

#: Per-NS-input failure codes (rank-consistent via all-reduce).
_OK = 0
_NONFINITE = 1
_CEILING = 2
_FAILURE_NAMES = {
    _NONFINITE: "nonfinite NS delta",
    _CEILING: "delta_rms above 10x(0.2*lr) ceiling",
}


def stable_owner(fqn: str, chunk: int, world_size: int) -> int:
    """Deterministic canonical owner rank for one NS input.

    FNV-1a 64 over ``f"{fqn}#chunk{chunk}"`` modulo world_size: stable
    across processes / launch orders; identical for identical
    (fqn, chunk, world_size).
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    h = _FNV_OFFSET
    for byte in f"{fqn}#chunk{chunk}".encode():
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK64
    return h % world_size


def _f32(value: float) -> float:
    """Round through float32 (the checkpoint stores FP32 references)."""
    return float(torch.tensor(value, dtype=torch.float32).item())


def _as_dict(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CMuonSafetyError(f"{what} must be a mapping, got {type(value)!r}")
    return {str(k): v for k, v in value.items()}


@dataclass(frozen=True)
class GuardedCanonicalGuardConfig:
    """Guard parameters (``[optimizer.cmuon_guard]``).

    Values are calibration-derived (design §5.4); there are no preset
    defaults — every field is explicit in the config.
    """

    guard_ratio: float
    reference_decay: float  # (0, 1]; slow relaxation of the reference
    min_reference: float  # absolute floor for every reference
    numerical_floor: float  # secondary absolute floor on fro(u)
    warmup_observations: int  # guard inactive for the first W updates
    invariant_check: bool = True  # per-step cross-rank param fingerprints

    def __post_init__(self) -> None:
        if not math.isfinite(self.guard_ratio) or self.guard_ratio <= 0.0:
            raise ValueError("guard_ratio must be a positive finite float")
        if not math.isfinite(self.reference_decay) or not 0.0 < self.reference_decay <= 1.0:
            raise ValueError("reference_decay must be in (0, 1]")
        if not math.isfinite(self.min_reference) or self.min_reference <= 0.0:
            raise ValueError("min_reference must be a positive finite float")
        if not math.isfinite(self.numerical_floor) or self.numerical_floor <= 0.0:
            raise ValueError("numerical_floor must be a positive finite float")
        if not isinstance(self.warmup_observations, int) or self.warmup_observations < 0:
            raise ValueError("warmup_observations must be a non-negative int")


@dataclass
class _PreparedSpec:
    """PHASE-1 working state for one parameter (all its chunks)."""

    spec: CMuonChunkSpec
    chunks: tuple[torch.Tensor, ...]
    alphas: tuple[float, ...]
    ns_steps: int
    sig: list[torch.Tensor]  # per-chunk device rms (synced once, batched)
    sigf: list[torch.Tensor]  # per-chunk device fro


class HybridCMuonGuardedCanonical(HybridCMuon):
    """Guarded canonical Hybrid CMuon (v1). See module docstring."""

    def __init__(
        self,
        *,
        routing: CMuonRouting,
        cfg: CMuonConfig,
        adamw: IsolatedAdamW8bit,
        sr_rng: StochasticRoundingRNG,
        guard_cfg: GuardedCanonicalGuardConfig,
        rank: int,
        world_size: int,
        guard_bootstrap_refs: Mapping[str, float],
        ns_telemetry: object | None = None,
        forensic: object | None = None,
        stats_logger: Callable[[str], None] | None = None,
        stats_log_every_n: int = 10,
    ) -> None:
        if ns_telemetry is not None:
            raise ValueError(
                "the guarded canonical candidate has its own per-step safety "
                "checks; the legacy NS telemetry is not wired (leave it disabled)"
            )
        if forensic is not None:
            raise ValueError(
                "the guarded canonical candidate is two-phase fail-closed by "
                "construction; the forensic monitor belongs to the retired "
                "original candidate (disable [optimizer.cmuon_forensic])"
            )
        super().__init__(
            routing=routing,
            cfg=cfg,
            adamw=adamw,
            sr_rng=sr_rng,
            ns_telemetry=None,
            forensic=None,
        )
        self.guard_cfg = guard_cfg
        self.rank = rank
        self.world_size = world_size
        self.stats_logger = stats_logger
        self.stats_log_every_n = stats_log_every_n
        self.observations = 0
        self.skip_total = 0
        self.skip_by_role: dict[str, int] = {}
        self.skip_by_fqn: dict[str, int] = {}
        self.bootstrap_mode = "calibration"
        self.max_param_rank_diff = 0.0
        self.max_delta_rank_spread = 0.0
        # Per-NS-input signal references (FP32-rounded scalars).
        self._refs: dict[tuple[str, int], float] = {}
        for spec in routing.cmuon_specs:
            for ci in range(spec.chunk_count):
                key = f"{spec.name}#chunk{ci}"
                init = guard_bootstrap_refs.get(key)
                if init is None:
                    init = guard_bootstrap_refs.get(spec.name)
                if init is None or not math.isfinite(float(init)) or float(init) <= 0.0:
                    raise ValueError(
                        "guarded canonical requires a positive calibration "
                        f"bootstrap reference for every NS input; missing: {key}"
                    )
                self._refs[(spec.name, ci)] = max(
                    _f32(float(init)), guard_cfg.min_reference
                )

    # -- owner mapping -----------------------------------------------------

    def owner_of(self, fqn: str, chunk: int) -> int:
        return stable_owner(fqn, chunk, self.world_size)

    # -- guard decision ----------------------------------------------------

    def _is_low_signal(self, sig: float, sig_fro: float, key: tuple[str, int]) -> bool:
        """True => low signal => skip NS and parameter delta for this input."""
        if self.observations < self.guard_cfg.warmup_observations:
            return False  # bootstrap window: observe, do not skip
        ref = self._refs[key]
        return sig < self.guard_cfg.guard_ratio * ref or (
            sig_fro < self.guard_cfg.numerical_floor
        )

    # -- step: two-phase atomic --------------------------------------------

    def step(self) -> None:
        self._sync_learning_rate()  # pyright: ignore[reportPrivateUsage]
        self._validate_finite_gradients()  # pyright: ignore[reportPrivateUsage]
        mu = self.cfg.momentum
        lr = self.cfg.lr
        target_delta_rms = 0.2 * lr
        ceiling = 10.0 * target_delta_rms
        specs = self.routing.cmuon_specs
        device = self._first_device()

        # ---- PHASE 1: PREPARE ---------------------------------------------
        prepared: list[_PreparedSpec | None] = []
        for spec in specs:
            grad = spec.parameter.grad
            if grad is None:
                prepared.append(None)
                continue
            buf = self._momenta[spec.parameter]  # pyright: ignore[reportPrivateUsage]
            grad_md = grad.to(buf.dtype)
            # Momentum EMA (production-identical in-place lerp).
            # Fail-closed semantics: a PHASE-1 failure stops the run, so no
            # rollback is needed (the dump records the state as-is).
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
                    sigf=[c.norm() for c in cf_chunks],
                )
            )

        n_inputs = sum(p.spec.chunk_count for p in prepared if p is not None)
        # Single batched sync: all per-chunk signal magnitudes.
        sig_flat = [s for p in prepared if p is not None for s in p.sig]
        sigf_flat = [s for p in prepared if p is not None for s in p.sigf]
        sig_vals = torch.stack(sig_flat).tolist() if sig_flat else []
        sigf_vals = torch.stack(sigf_flat).tolist() if sigf_flat else []

        # Guard decisions + owner NS + staged deltas.
        # fail_flags: per NS input, owner-local failure code (all-reduced).
        fail_flags = torch.zeros(n_inputs, dtype=torch.int64, device=device)
        staged: list[torch.Tensor | None] = []
        owners: list[int] = []
        is_active: list[bool] = []
        sig_by_input: list[float | None] = []
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
                sigf = sigf_vals[fi]
                fi += 1
                key = (spec.name, ci)
                sig_by_input.append(sig)
                owners.append(owner)
                if self._is_low_signal(sig, sigf, key):
                    is_active.append(False)
                    staged.append(None)
                    self.skip_total += 1
                    self.skip_by_role[spec.role] = (
                        self.skip_by_role.get(spec.role, 0) + 1
                    )
                    self.skip_by_fqn[spec.name] = self.skip_by_fqn.get(spec.name, 0) + 1
                    continue
                is_active.append(True)
                staged.append(None)
                if owner == self.rank:
                    chunk = prep.chunks[ci]
                    ns = cmuon_zeroth_power(
                        chunk, prep.ns_steps, self.cfg.ns_coefficients, self.cfg.eps
                    )
                    # .contiguous(): tall chunks return from NS as a .T
                    # view; staged deltas must have the identical layout on
                    # every rank (owner: computed here, receiver: fresh
                    # broadcast buffer) so the cross-rank fp32 fingerprint
                    # reductions run in the same order and compare exact.
                    delta = ((-prep.alphas[ci]) * ns).contiguous()
                    if not bool(torch.isfinite(delta).all()):
                        fail_flags[fi - 1] = _NONFINITE
                        continue
                    # Keep the device-side rms for the batched ceiling sync.
                    d_rms_list.append(delta.float().pow(2).mean().sqrt())
                    d_rms_owner_idx.append(fi - 1)
                    staged[-1] = delta
                # non-owner: filled by broadcast below

        # One batched sync of the owner delta RMS values, then ceiling check.
        if d_rms_list:
            for rms, idx in zip(torch.stack(d_rms_list).tolist(), d_rms_owner_idx):
                if rms > ceiling:
                    fail_flags[idx] = _CEILING

        # Rank-consistent failure verdict (every rank learns every failure).
        if self.world_size > 1:
            dist.all_reduce(fail_flags, op=dist.ReduceOp.MAX)
        failure_msgs = []
        for idx, flag in enumerate(fail_flags.tolist()):
            if flag > 0:
                fqn, chunk_idx = self._input_key(idx)
                failure_msgs.append(
                    f"{_FAILURE_NAMES[flag]}: {fqn}#chunk{chunk_idx}"
                )

        # Broadcast: every ACTIVE, non-failed chunk is sent owner -> all
        # ranks (v1 is spec-by-spec, correctness first, no overlap).
        # world_size == 1: the owner is always this rank; no collective.
        failed = {idx for idx, flag in enumerate(fail_flags.tolist()) if flag > 0}
        if self.world_size > 1:
            for idx, (a, owner) in enumerate(zip(is_active, owners)):
                if not a or idx in failed:
                    staged[idx] = None  # zero delta on every rank (consensus)
                    continue
                shape = self._chunk_shape(idx)
                if owner == self.rank:
                    tensor = staged[idx]
                    assert tensor is not None
                    # Tall chunks come back from cmuon_zeroth_power as a
                    # .T view (non-contiguous); broadcast requires
                    # contiguous storage. .contiguous() is a no-op copy
                    # when already contiguous; broadcast does not modify
                    # the sender, so staged[idx] keeps the same values.
                    dist.broadcast(tensor.contiguous(), src=owner)
                else:
                    buf = torch.empty(shape, dtype=torch.bfloat16, device=device)
                    dist.broadcast(buf, src=owner)
                    staged[idx] = buf
        else:
            for idx, a in enumerate(is_active):
                if not a or idx in failed:
                    staged[idx] = None

        # Cross-rank delta fingerprint (exact equality is expected: the
        # broadcast delivers identical bits and the fp32 reductions are
        # deterministic on HCU for identical inputs).
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
            raise CMuonSafetyError(
                "guarded canonical safety violation: " + " | ".join(failure_msgs)
            )

        # ---- PHASE 2: COMMIT ------------------------------------------------
        # AdamW part (production path under the isolated SR RNG).
        self.sr_rng.run_step(self.optimizer.step)
        # CMuon deltas: reassemble per spec exactly like the core update
        # (cat along chunk_dim, single in-place add_) and apply. A skipped
        # chunk contributes zeros (param regions are bit-unchanged; with
        # matrix_weight_decay=0 a fully skipped spec gets no parameter
        # write at all).
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

        # Reference updates (active inputs only; inactive refs are frozen —
        # they never decay to zero).
        fi = 0
        for prep in prepared:
            if prep is None:
                continue
            spec = prep.spec
            for ci in range(spec.chunk_count):
                if not is_active[fi]:
                    fi += 1
                    continue
                key = (spec.name, ci)
                sig = sig_by_input[fi]
                assert sig is not None
                ref = self._refs[key]
                self._refs[key] = max(
                    _f32(sig), _f32(ref * self.guard_cfg.reference_decay)
                )
                fi += 1

        self.observations += 1

        # Optional per-step rank invariant (S1/S2): post-commit parameter
        # fingerprints must be identical on every rank (canonical deltas
        # guarantee this; any spread is a defect => fail closed).
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
                    "guarded canonical rank invariant violated: "
                    f"cross-rank parameter fingerprint diff {diff:.3e} after commit"
                )

        if self.stats_logger is not None and (
            self.observations % self.stats_log_every_n == 0
        ):
            self.stats_logger(
                json.dumps(
                    {
                        "observations": self.observations,
                        "skip_total": self.skip_total,
                        "skip_rate": self.skip_total / max(1, self.observations * n_inputs),
                        "skip_by_role": dict(sorted(self.skip_by_role.items())),
                        "skip_by_fqn": dict(sorted(self.skip_by_fqn.items())),
                        "ref_min": min(self._refs.values()),
                        "ref_max": max(self._refs.values()),
                        "max_delta_rank_spread": self.max_delta_rank_spread,
                        "max_param_rank_diff": self.max_param_rank_diff,
                    }
                )
            )

    # -- helpers -------------------------------------------------------------

    def _first_device(self) -> torch.device:
        for spec in self.routing.cmuon_specs:
            if spec.parameter.grad is not None:
                return spec.parameter.device
        return self.routing.cmuon_specs[0].parameter.device

    def _input_key(self, flat_idx: int) -> tuple[str, int]:
        idx = flat_idx
        for spec in self.routing.cmuon_specs:
            if idx < spec.chunk_count:
                return (spec.name, idx)
            idx -= spec.chunk_count
        raise IndexError(flat_idx)

    def _chunk_shape(self, flat_idx: int) -> tuple[int, ...]:
        idx = flat_idx
        for spec in self.routing.cmuon_specs:
            if idx < spec.chunk_count:
                shape = list(spec.parameter.shape)
                shape[spec.chunk_dim] = spec.chunk_size()
                return tuple(shape)
            idx -= spec.chunk_count
        raise IndexError(flat_idx)

    # -- state ---------------------------------------------------------------

    def _guard_state(self) -> dict[str, object]:
        return {
            "schema_version": GUARD_SCHEMA_VERSION,
            "config": {
                "guard_ratio": self.guard_cfg.guard_ratio,
                "reference_decay": self.guard_cfg.reference_decay,
                "min_reference": self.guard_cfg.min_reference,
                "numerical_floor": self.guard_cfg.numerical_floor,
                "warmup_observations": self.guard_cfg.warmup_observations,
                "invariant_check": self.guard_cfg.invariant_check,
            },
            "references": {
                f"{fqn}#chunk{ci}": _f32(ref)
                for (fqn, ci), ref in sorted(self._refs.items())
            },
            "skip_total": self.skip_total,
            "skip_by_role": dict(sorted(self.skip_by_role.items())),
            "skip_by_fqn": dict(sorted(self.skip_by_fqn.items())),
            "observations": self.observations,
            "bootstrap_mode": self.bootstrap_mode,
            "owner_mapping_version": OWNER_MAPPING_VERSION,
            "world_size": self.world_size,
            "canonical_ns_mode": True,
            "ns_map": self.cfg.canonical_ns_map(),
        }

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["guard"] = self._guard_state()
        state["guarded_canonical_schema_version"] = 1
        return state

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        has_guard = "guard" in state_dict
        has_cmuon = "cmuon" in state_dict
        if has_guard and not has_cmuon:
            raise CMuonSafetyError("guarded checkpoint without CMuon state is invalid")
        if not has_guard and has_cmuon:
            raise CMuonSafetyError(
                "an unguarded Hybrid CMuon checkpoint cannot resume directly "
                "into the guarded canonical candidate; an explicit optimizer "
                "transition with guard reference re-bootstrap is required "
                "(design §10)"
            )
        if has_guard:
            guard = _as_dict(state_dict["guard"], "guard state")
            if guard.get("schema_version") != GUARD_SCHEMA_VERSION:
                raise CMuonSafetyError("guard schema version mismatch")
            if guard.get("world_size") != self.world_size:
                raise CMuonSafetyError(
                    f"guard world_size mismatch: saved {guard.get('world_size')} "
                    f"vs current {self.world_size}; the canonical owner mapping "
                    "changed — explicit migration required"
                )
            if guard.get("owner_mapping_version") != OWNER_MAPPING_VERSION:
                raise CMuonSafetyError("owner mapping version mismatch")
            if guard.get("config") != self._guard_state()["config"]:
                raise CMuonSafetyError(
                    "guard config mismatch: saved "
                    f"{guard.get('config')} vs current {self._guard_state()['config']}"
                )
        # Delegate the base hybrid state to the parent; the guard section is
        # this subclass's own state and must not reach the unguarded parent
        # (which rejects it by contract).
        base_state = {k: v for k, v in state_dict.items() if k != "guard"}
        base_state.pop("guarded_canonical_schema_version", None)
        super().load_state_dict(base_state)
        if has_guard:
            guard = _as_dict(state_dict["guard"], "guard state")
            refs = _as_dict(guard.get("references"), "guard references")
            restored: dict[tuple[str, int], float] = {}
            for key, value in refs.items():
                fqn, _, chunk = str(key).rpartition("#chunk")
                restored[(fqn, int(chunk))] = _f32(float(value))
            if set(restored) != set(self._refs):
                raise CMuonSafetyError("guard reference key set mismatch")
            self._refs = restored
            self.skip_total = int(guard.get("skip_total", 0))
            self.skip_by_role = {
                str(k): int(v)
                for k, v in _as_dict(guard.get("skip_by_role"), "skip_by_role").items()
            }
            self.skip_by_fqn = {
                str(k): int(v)
                for k, v in _as_dict(guard.get("skip_by_fqn"), "skip_by_fqn").items()
            }
            self.observations = int(guard.get("observations", 0))
            mode = guard.get("bootstrap_mode")
            if mode not in ("calibration", "observation"):
                raise CMuonSafetyError("unknown guard bootstrap mode")
            self.bootstrap_mode = str(mode)


def build_guarded_canonical(
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
) -> HybridCMuonGuardedCanonical:
    """Build the guarded canonical candidate (same routing/AdamW policy as
    the hybrid core; guard + canonical owner NS on top)."""
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
    return HybridCMuonGuardedCanonical(
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
    "GUARD_SCHEMA_VERSION",
    "OWNER_MAPPING_VERSION",
    "GuardedCanonicalGuardConfig",
    "HybridCMuonGuardedCanonical",
    "build_guarded_canonical",
    "stable_owner",
]
