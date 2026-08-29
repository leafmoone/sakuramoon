"""Forensic instrumentation for the Hybrid CMuon optimizer (root-cause round).

This module is a FAIL-CLOSED, diagnosis-only layer on top of the hybrid
optimizer. It does NOT change the optimizer arithmetic:

  * The two-phase CMuon step (``HybridCMuon.step`` with a monitor attached)
    computes the exact same per-element values as the original single-phase
    path (momentum lerp, Nesterov lerp, per-chunk quintic Newton-Schulz,
    Moonlight scaling); it only splits "compute all candidates" from "commit
    momentum + parameters" so that a nonfinite / oversized delta can be
    detected BEFORE any parameter write (the original path writes the
    parameter in place and only the next clip_grad_norm reports nonfinite).

What it records (per successful CMuon step, device-side, one batched CPU
sync per step + two small all_gathers for the rank comparison):

  * all 141 CMuon specs: grad / momentum-candidate / nesterov / NS-output /
    applied-delta rms, sum, max_abs, finite flag  (P0-B: first offending
    tensor, not the next global grad norm)
  * per-NS-iteration (0..ns_steps-1) output norm per spec (NS blow-up trace)
  * rank consistency: the same fingerprints are all_gathered and compared
    across ranks; the first (update, stage, tensor) with a nonzero relative
    difference is the first rank divergence  (P0-A)
  * deterministic probe dot-products (fixed-seed untrainable probe vectors)
    for the K / V / shared AdaLN / one Q / one FFN-in specs: catches
    element-level differences that rms/sum can mask
  * a lightweight ring buffer (last N updates) that is dumped, together with
    the offending tensor artifacts, when the fail-closed guard trips or a
    rank divergence is detected

Fail-closed guard (detection -> stop -> dump; NEVER clamp-and-continue):

  * any nonfinite value in a spec's NS output or applied delta
  * applied-delta RMS above ``ceiling_multiplier * (0.2 * lr)`` — a
    catastrophic-abnormal threshold, not a clip (expected delta RMS is
    ~0.2*lr from the NS-depth audit)
  * applied-delta max_abs above a learned baseline (median of the first
    ``max_abs_learn_steps`` healthy steps) times ``max_abs_alarm_mult``
  * any cross-rank fingerprint divergence (rel diff > 0)

When the guard trips, every rank raises ``CMuonSafetyError`` (the failure
flag is all_reduced so both ranks stop at the same step — no partial
DDP continuation) and the main rank writes
``<dump_dir>/cmuon-forensic-crash-<update>.json`` plus the offending
tensor artifacts.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from sakuramoon.optim.cmuon import CMuonChunkSpec, CMuonConfig, CMuonRouting

# Fingerprint layout: (grad, momentum, nesterov, ns, delta) x (rms, sum, max)
# + 1 finite flag + 1 has-grad flag.
_FINGERPRINT_STAGES: tuple[str, ...] = ("grad", "momentum", "nesterov", "ns", "delta")
_N_FP_COLS: int = 5 * 3 + 2  # 17
# Probe stages: grad / momentum / ns / delta / param.
_PROBE_STAGES: tuple[str, ...] = ("grad", "momentum", "ns", "delta", "param")
_MAX_NS_ITERS: int = 99  # cmuon ns_steps upper bound


class CMuonSafetyError(RuntimeError):
    """Fail-closed abort of a CMuon step (nonfinite / oversized delta /
    cross-rank divergence). Raised on EVERY rank at the same step."""


@dataclass(frozen=True)
class ForensicConfig:
    """Forensic run settings (see the module docstring)."""

    enabled: bool = True
    ring_size: int = 10
    ceiling_multiplier: float = 10.0
    max_abs_learn_steps: int = 50
    max_abs_alarm_mult: float = 20.0
    dump_dir: str | None = None
    probe_seed: int = 20260829


def _dist() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return (
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
        )
    return 0, 1


def _all_gather_last_dim(x: torch.Tensor) -> list[torch.Tensor]:
    """all_gather a small 2D CPU tensor across ranks (identity for world=1)."""
    _, world = _dist()
    if world == 1:
        return [x]
    outs = [torch.empty_like(x) for _ in range(world)]
    torch.distributed.all_gather(outs, x.contiguous())
    return outs


def _all_reduce_bool(flag: bool) -> bool:
    _, world = _dist()
    if world == 1:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return bool(t.item() > 0)


class ForensicMonitor:
    """Per-step forensic instrumentation for all CMuon specs (see module doc)."""

    def __init__(
        self,
        routing: CMuonRouting,
        cfg: CMuonConfig,
        fcfg: ForensicConfig,
        *,
        device: torch.device | int | str,
        logger: Callable[[str], None] | None = None,
        update_offset: int = 0,
    ) -> None:
        self.cfg = cfg
        self.fcfg = fcfg
        self.device = device

        def _silent(_m: str) -> None:
            return None

        self._logger: Callable[[str], None] = logger if logger is not None else _silent
        self.update_offset = int(update_offset)
        self._steps = 0
        self._rank, self._world = _dist()
        self._lr = float(cfg.lr)
        self._ceiling = fcfg.ceiling_multiplier * (0.2 * self._lr)
        specs = routing.cmuon_specs
        self.specs: tuple[CMuonChunkSpec, ...] = tuple(specs)
        self.n_specs = len(specs)
        self._fqn = [s.name for s in specs]
        self._role = [s.role for s in specs]
        # device-side fingerprint accumulator: [n_specs, _N_FP_COLS] fp32
        self._fp = torch.zeros(self.n_specs, _N_FP_COLS, device=device, dtype=torch.float32)
        # per-NS-iteration output norms: [n_specs, _MAX_NS_ITERS] fp32
        self._ns_trace = torch.zeros(
            self.n_specs, _MAX_NS_ITERS, device=device, dtype=torch.float32
        )
        # probe dot-products for the selected specs
        probe_idx = self._probe_indices(routing)
        self._probe_specs: list[tuple[int, str]] = [
            (i, self._fqn[i]) for i in probe_idx
        ]
        self._probes: dict[int, torch.Tensor] = {}
        for i, _fqn in self._probe_specs:
            n = specs[i].parameter.numel()
            g = torch.Generator(device="cpu").manual_seed(fcfg.probe_seed + i)
            self._probes[i] = torch.randn(n, generator=g, dtype=torch.float32).to(
                device
            )
            self._probes[i].div_(self._probes[i].norm().clamp(min=1e-30))
        self._probe_fp = torch.zeros(
            max(1, len(self._probe_specs)),
            len(_PROBE_STAGES),
            device=device,
            dtype=torch.float32,
        )
        # ring buffer of per-step CPU rows (filled at collect time)
        self._ring: deque[dict[str, object]] = deque(maxlen=max(1, fcfg.ring_size))
        # learned max_abs baseline (median over the first healthy steps)
        self._delta_max_history: list[float] = []
        self._max_abs_baseline: float | None = None
        # first-divergence / failure state (set at collect time)
        self.first_rank_divergence: dict[str, object] | None = None
        self.first_local_failure: dict[str, object] | None = None
        self._step_failed = False
        self._step_diverged = False
        # phase-1 tensor stash for the dump (set at collect time)
        self._stashed: list[tuple[int, dict[str, torch.Tensor]]] = []

    # -- spec selection ----------------------------------------------------

    @staticmethod
    def _probe_indices(routing: CMuonRouting) -> list[int]:
        """All K, all V, the shared AdaLN projection, the first Q, the first
        FFN-in (spec index order = routing order)."""
        idx: list[int] = []
        first_q = False
        first_ffn = False
        for i, spec in enumerate(routing.cmuon_specs):
            if spec.role in ("attention_k", "attention_v", "adaln_shared"):
                idx.append(i)
            elif spec.role == "attention_q" and not first_q:
                idx.append(i)
                first_q = True
            elif spec.role == "ffn_in" and not first_ffn:
                idx.append(i)
                first_ffn = True
        return idx

    def _set_row(
        self,
        i: int,
        *,
        grad: torch.Tensor | None,
        momentum: torch.Tensor | None,
        nesterov: torch.Tensor | None,
        ns: torch.Tensor | None,
        delta: torch.Tensor | None,
    ) -> None:
        row = self._fp[i]
        row.zero_()
        has = grad is not None
        row[_N_FP_COLS - 1] = 1.0 if has else 0.0
        row[_N_FP_COLS - 2] = 0.0  # finite: optimistic until proven
        col = 0
        for tensor in (grad, momentum, nesterov, ns, delta):
            if tensor is None:
                col += 3
                continue
            t = tensor.float()
            row[col] = (t.pow(2).mean().sqrt())
            row[col + 1] = t.sum()
            row[col + 2] = t.abs().max()
            if not bool(torch.isfinite(t).all().item()):
                row[_N_FP_COLS - 2] = 0.0
            col += 3

    def record_spec(
        self,
        i: int,
        *,
        grad: torch.Tensor | None,
        momentum: torch.Tensor | None,
        nesterov: torch.Tensor | None,
        ns: torch.Tensor | None,
        delta: torch.Tensor | None,
        ns_trace: list[torch.Tensor] | None = None,
    ) -> None:
        """Phase-1 recording for one spec (device-side, no CPU sync).

        ``ns`` is the full (chunk-concatenated) NS output BEFORE Moonlight
        scaling; ``delta`` is the applied update (-alpha * NS). The probe
        dot-products are computed here while the tensors are alive.
        """
        self._set_row(
            i,
            grad=grad,
            momentum=momentum,
            nesterov=nesterov,
            ns=ns,
            delta=delta,
        )
        if ns_trace:
            for k, v in enumerate(ns_trace[:_MAX_NS_ITERS]):
                self._ns_trace[i, k] = v
        for p, (pi, _fqn) in enumerate(self._probe_specs):
            if pi != i:
                continue
            probe = self._probes[i]
            row = self._probe_fp[p]
            for s, tensor in enumerate(
                (grad, momentum, ns, delta, None)
            ):  # param stage filled after commit
                if tensor is None:
                    continue
                row[s] = (tensor.float().flatten() @ probe)

    def record_param_after(self) -> None:
        """Phase-2: param-after fingerprint (dedicated small tensor, compared
        across ranks) + the probe 'param' dot stage."""
        # dedicated param-after fingerprint tensor (rms/sum/max per spec)
        cols = torch.empty(self.n_specs, 3, device=self.device, dtype=torch.float32)
        for i, spec in enumerate(self.specs):
            t = spec.parameter.float()
            cols[i, 0] = t.pow(2).mean().sqrt()
            cols[i, 1] = t.sum()
            cols[i, 2] = t.abs().max()
        for p, (pi, _fqn) in enumerate(self._probe_specs):
            t = self.specs[pi].parameter.float().flatten()
            self._probe_fp[p, len(_PROBE_STAGES) - 1] = t @ self._probes[pi]
        gathered = _all_gather_last_dim(cols.cpu())
        if self._world > 1:
            worst: dict[str, object] | None = None
            worst_rel = -1.0
            for a in range(self._world):
                for b in range(a + 1, self._world):
                    diff = (gathered[a] - gathered[b]).abs()
                    rel = diff / gathered[a].abs().clamp(min=1e-30)
                    bad = (rel > 0.0).nonzero().flatten().tolist()
                    if bad:
                        i = int(bad[0])
                        stage_col = int(diff[i].argmax())
                        rel_diff = float(rel[i, stage_col])
                        cand: dict[str, object] = {
                            "stage": "param",
                            "spec": i,
                            "fqn": self._fqn[i],
                            "role": self._role[i],
                            "col": _PARAM_COL_NAMES[stage_col],
                            "abs_diff": float(diff[i, stage_col]),
                            "rel_diff": rel_diff,
                        }
                        if rel_diff > worst_rel:
                            worst = cand
                            worst_rel = rel_diff
            if worst is not None and self.first_rank_divergence is None:
                self._record_divergence(worst, rel_diff=worst_rel)
        self._param_fp = cols

    # -- collect / compare / guard ----------------------------------------

    def collect_and_compare(
        self,
        lr: float,
        *,
        stashed: list[tuple[int, dict[str, torch.Tensor]]] | None = None,
    ) -> None:
        """End of phase 1: one batched CPU sync, local guard, rank comparison.

        ``stashed`` maps spec index -> {grad, momentum, nesterov, ns_output,
        delta} (the live phase-1 tensors); used by the dump when the guard
        trips. Raises ``CMuonSafetyError`` (on every rank, after the flag is
        all_reduced) when the guard trips or any rank diverges.
        """
        self._stashed = stashed or []
        self._steps += 1
        self._lr = float(lr)
        self._ceiling = self.fcfg.ceiling_multiplier * (0.2 * self._lr)
        fp_cpu = self._fp.cpu()
        ns_cpu = self._ns_trace.cpu()
        probe_cpu = self._probe_fp.cpu()
        divergence: dict[str, object] | None = None
        # -- local fail-closed checks (per spec) --
        failure: dict[str, object] | None = None
        finite_col = _N_FP_COLS - 2
        delta_rms_col = 12  # delta stage starts at col 12: [12,13,14]
        for i in range(self.n_specs):
            row = fp_cpu[i]
            if bool(row[finite_col].item() == 0.0):
                failure = {
                    "kind": "nonfinite",
                    "spec": i,
                    "fqn": self._fqn[i],
                    "role": self._role[i],
                }
                break
            delta_rms = float(row[delta_rms_col].item())
            if not math.isfinite(delta_rms):
                failure = {
                    "kind": "nonfinite_delta_rms",
                    "spec": i,
                    "fqn": self._fqn[i],
                    "role": self._role[i],
                    "delta_rms": delta_rms,
                }
                break
            if delta_rms > self._ceiling:
                failure = {
                    "kind": "delta_rms_ceiling",
                    "spec": i,
                    "fqn": self._fqn[i],
                    "role": self._role[i],
                    "delta_rms": delta_rms,
                    "ceiling": self._ceiling,
                }
                break
            delta_max = float(row[14].item())
            if self._steps <= self.fcfg.max_abs_learn_steps:
                if math.isfinite(delta_max):
                    self._delta_max_history.append(delta_max)
                if len(self._delta_max_history) >= self.fcfg.max_abs_learn_steps:
                    self._max_abs_baseline = float(
                        sorted(self._delta_max_history)[
                            len(self._delta_max_history) // 2
                        ]
                    )
            elif (
                self._max_abs_baseline is not None
                and delta_max > self.fcfg.max_abs_alarm_mult * self._max_abs_baseline
            ):
                failure = {
                    "kind": "delta_max_abs_alarm",
                    "spec": i,
                    "fqn": self._fqn[i],
                    "role": self._role[i],
                    "delta_max": delta_max,
                    "baseline": self._max_abs_baseline,
                }
                break
        self.first_local_failure = failure
        self._step_failed = failure is not None
        # -- rank comparison (all ranks participate; identical call pattern) --
        gathered = _all_gather_last_dim(fp_cpu)
        probe_gathered = _all_gather_last_dim(probe_cpu)
        if self._world > 1:
            for a in range(self._world):
                for b in range(a + 1, self._world):
                    d = (gathered[a] - gathered[b]).abs()
                    rel = d / gathered[a].abs().clamp(min=1e-30)
                    # has-grad / finite flag columns are exact: any diff = divergence
                    flag_cols = (_N_FP_COLS - 2, _N_FP_COLS - 1)
                    mask = torch.zeros(self.n_specs, _N_FP_COLS, dtype=torch.bool)
                    mask[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] = (
                        rel[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] > 0.0
                    )
                    mask[:, flag_cols] = d[:, flag_cols] > 0.0
                    bad = mask.nonzero()
                    if bad.numel() > 0:
                        i, c = int(bad[0, 0]), int(bad[0, 1])
                        divergence = {
                            "stage": _FP_COL_STAGES[c],
                            "spec": i,
                            "fqn": self._fqn[i],
                            "role": self._role[i],
                            "col": _FP_COL_NAMES[c],
                            "abs_diff": float(d[i, c].item()),
                            "rel_diff": float(rel[i, c].item()),
                            "ranks": [a, b],
                        }
                        break
                    if divergence:
                        break
                if divergence:
                    break
            # probe comparison (exact fp32 dots; any diff = divergence)
            if divergence is None:
                for a in range(self._world):
                    for b in range(a + 1, self._world):
                        pd = (probe_gathered[a] - probe_gathered[b]).abs()
                        bad = (pd > 0.0).nonzero()
                        if bad.numel() > 0:
                            p, s = int(bad[0, 0]), int(bad[0, 1])
                            divergence = {
                                "stage": _PROBE_STAGES[s],
                                "spec": self._probe_specs[p][0],
                                "fqn": self._probe_specs[p][1],
                                "role": self._role[self._probe_specs[p][0]],
                                "col": "probe_dot",
                                "abs_diff": float(pd[p, s].item()),
                                "rel_diff": float(
                                    (pd[p, s] / probe_gathered[a][p, s].abs().clamp(min=1e-30)).item()
                                ),
                                "ranks": [a, b],
                                "probe": True,
                            }
                            break
                    if divergence:
                        break
        if divergence is not None and self.first_rank_divergence is None:
            rel_v = divergence.get("rel_diff")
            self._record_divergence(
                divergence,
                rel_diff=float(rel_v) if isinstance(rel_v, (int, float)) else 0.0,
            )
        # -- ring buffer (CPU row per step) --
        self._ring.append(
            {
                "update": self._steps + self.update_offset,
                "rank": self._rank,
                "lr": self._lr,
                "fp": fp_cpu.tolist(),
                "ns_trace": ns_cpu.tolist(),
                "probe": probe_cpu.tolist(),
                "max_abs_baseline": self._max_abs_baseline,
                "local_failure": failure,
                "divergence": self.first_rank_divergence,
            }
        )
        # -- fail-closed decision (all ranks see the same verdict) --
        any_failed = _all_reduce_bool(self._step_failed)
        any_diverged = _all_reduce_bool(divergence is not None)
        if any_failed or any_diverged:
            self._dump(failure, divergence)
            msg = (
                f"[cmuon-forensic] FAIL-CLOSED abort at update="
                f"{self._steps + self.update_offset} rank={self._rank} "
                f"local_failure={failure} first_divergence={self.first_rank_divergence}"
            )
            self._logger(msg)
            raise CMuonSafetyError(msg)

    def _record_divergence(
        self, div: dict[str, object], *, rel_diff: float
    ) -> None:
        div = dict(div)
        div["update"] = self._steps + self.update_offset
        self.first_rank_divergence = div
        self._step_diverged = True
        self._logger(
            f"[cmuon-forensic] RANK DIVERGENCE update={div['update']} "
            f"stage={div['stage']} tensor={div['fqn']} rel_diff={rel_diff:.3e}"
        )

    def _offending_idx(self) -> int | None:
        """The spec to attribute the failure to (failure first, then the
        first rank divergence)."""
        for entry in (self.first_local_failure, self.first_rank_divergence):
            if entry is None:
                continue
            spec = entry.get("spec")
            if isinstance(spec, int):
                return spec
        return None

    # -- dump ---------------------------------------------------------------

    def _dump(
        self,
        failure: dict[str, object] | None,
        divergence: dict[str, object] | None,
    ) -> None:
        if self._rank != 0:
            return
        if not self.fcfg.dump_dir:
            return
        out = Path(self.fcfg.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        update = self._steps + self.update_offset
        payload = {
            "update": update,
            "run_step": self._steps,
            "update_offset": self.update_offset,
            "lr": self._lr,
            "delta_ceiling": self._ceiling,
            "max_abs_baseline": self._max_abs_baseline,
            "first_local_failure": failure,
            "first_rank_divergence": self.first_rank_divergence,
            "ring": list(self._ring),
        }
        path = out / f"cmuon-forensic-crash-{update}.json"
        path.write_text(json.dumps(payload, indent=1))
        self._logger(f"[cmuon-forensic] dump written: {path}")
        idx = self._offending_idx()
        if idx is not None:
            for spec_idx, tensors in self._stashed:
                if spec_idx == idx:
                    tag = self._fqn[idx].replace(".", "_")[:80]
                    blob: dict[str, object] = {
                        **tensors,
                        "_spec_index": idx,
                        "_fqn": self._fqn[idx],
                        "_role": self._role[idx],
                    }
                    torch.save(blob, out / f"cmuon-forensic-crash-{update}-{tag}.pt")
                    break

    # -- step-0 momentum check ----------------------------------------------

    def check_initial_momenta(
        self, momenta: Mapping[nn.Parameter, torch.Tensor]
    ) -> None:
        """AdamW->CMuon transition audit: at step 0 every rank's momentum
        buffers must be exactly zero (fresh init). Records the max abs."""
        worst = 0.0
        for spec in self.specs:
            buf = momenta.get(spec.parameter)
            if buf is None:
                continue
            worst = max(worst, float(buf.abs().max().item()))
        gathered = _all_gather_last_dim(torch.tensor([worst], dtype=torch.float32))
        vals = [float(v.item()) for v in gathered]
        verdict = "ZERO-OK" if all(v == 0.0 for v in vals) else "NOT-ZERO!"
        if self._rank == 0:
            self._logger(
                f"[cmuon-forensic] step0 momentum check: per-rank max_abs={vals} {verdict}"
            )
        self._step0_momentum_max = vals


# fingerprint column layout helpers
_FP_COL_NAMES: tuple[str, ...] = (
    "grad_rms",
    "grad_sum",
    "grad_max",
    "momentum_rms",
    "momentum_sum",
    "momentum_max",
    "nesterov_rms",
    "nesterov_sum",
    "nesterov_max",
    "ns_rms",
    "ns_sum",
    "ns_max",
    "delta_rms",
    "delta_sum",
    "delta_max",
    "finite",
    "has_grad",
)
_FP_COL_STAGES: tuple[str, ...] = (
    "grad",
    "grad",
    "grad",
    "momentum",
    "momentum",
    "momentum",
    "nesterov",
    "nesterov",
    "nesterov",
    "ns",
    "ns",
    "ns",
    "delta",
    "delta",
    "delta",
    "finite",
    "has_grad",
)
_PARAM_COL_NAMES: tuple[str, ...] = ("param_rms", "param_sum", "param_max")
