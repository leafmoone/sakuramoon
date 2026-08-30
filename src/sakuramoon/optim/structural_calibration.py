"""Structural/SNR pre-NS classifier calibration for the Guarded Canonical
CMuon candidate (D1 round, 08-31 user spec).

Goal: BEFORE any structural threshold exists, prove that dangerous NS4
inputs (the pre-write-ceiling violators: nonfinite or delta_rms > 10*0.2*lr)
can be identified from pre-NS features alone — at a skip rate far below the
amplitude-floor's 64-70% — using only deterministic, interpretable rules.

Mechanism (mirrors ``guard_calibration.py``): replace the step of a fully
built production ``HybridCMuon`` optimizer with a shadow observation
routine.  Per observation, per (FQN, chunk):

* momentum EMA + Nesterov exactly like production (bf16, mu from cfg);
  NO parameter update, NO AdamW step;
* features:
  - amplitude: element_rms / frobenius / max_abs / (rms / calibrated ref);
  - structure: sigma1..sigma4 by power iteration WITH DEFLATION on the
    fp32 chunk, top1/top2/top4 cumulative energy (sigma_j^2 / fro^2),
    stable_rank = fro^2 / sigma1^2, effective_rank = exp(entropy of the
    top-4 energy shares) (4-mode approximation, documented);
  - temporal: cos(grad, momentum_now), cos(grad, nesterov),
    cos(nesterov_t, momentum_{t-1}), cos(nesterov_t, nesterov_{t-1});
  - metadata: fqn / slot / role / shape / chunk / observation id / lr.
* safety label (the ground truth the classifier must predict): run the
  EXACT production NS4 (``cmuon_zeroth_power`` bf16, same coefficients /
  eps / steps) K times on the same nesterov chunk; delta_rms =
  alpha * ns_rms with the production Moonlight alpha; a run is
  CATASTROPHIC iff nonfinite or delta_rms > 10 * 0.2 * lr (the validated
  fail-closed rule, no clamp); label = DANGEROUS iff any run is
  catastrophic, else SAFE.  The K repeats estimate the per-input chaotic
  branch rate (HCU bf16 GEMM nondeterminism selects the branch; a dummy
  GEMM between repeats perturbs scheduling so the repeats are not
  bit-locked to one schedule).
* cross-rank consistency of every nesterov_rms (two device all-reduces,
  hard fail on any spread — momentum recursion must be rank-exact);
* HCU cost accounting per observation: feature extraction split into
  power-iteration time, extra reductions, temporal cosines, NS4 label
  time.

Record files (JSONL, one per rank):

    structural-calibration-rank{rank}.jsonl
      {"obs": i, "abs_update": ..., "lr": ..., "ceiling": ...,
       "cost_ms": {"feat_pi": ..., "feat_reduce": ..., "feat_cos": ...,
                   "ns_label": ...},
       "rank_consistency": {...},
       "rows": [{"fqn": ..., "slot": ..., "role": ..., "chunk": ci,
                 "shape": [...], "rms": ..., "fro": ..., "max_abs": ...,
                 "ref": ..., "rel_sig": ...,
                 "sigma1": ..., "sigma2": ..., "sigma3": ..., "sigma4": ...,
                 "top1_energy": ..., "top2_cum_energy": ...,
                 "top4_cum_energy": ..., "stable_rank": ...,
                 "eff_rank4": ...,
                 "cos_grad_mom": ..., "cos_grad_nest": ...,
                 "cos_nest_mom_prev": ..., "cos_nest_nest_prev": ...,
                 "ns_runs": [{"ns_rms": ..., "ns_max": ...,
                              "delta_rms": ..., "catastrophic": bool,
                              "ns_hash": "..."} ...],
                 "hazard_rate": ..., "label": "SAFE"|"DANGEROUS"} ...]}

After ``observations`` the shadow step raises
:class:`StructuralCalibrationComplete` — a clean stop (no failure bundle),
handled by the training loop / production lifecycle exactly like
``GuardCalibrationComplete``.

SVD reference samples: on observation 1, rank 0 saves the first
``svd_samples`` chunks (bf16, as NS sees them) to
``artifact_dir/svd-samples/sample-{i:03d}.pt`` for the offline
power-iteration accuracy audit (spec section 4).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sakuramoon.optim.cmuon import HybridCMuon

from sakuramoon.optim.cmuon import cmuon_moonlight_alpha, cmuon_zeroth_power

_SLOT_RE = re.compile(r"\.slot_(\d+)\.")


class StructuralCalibrationComplete(Exception):
    """Clean stop after the requested number of structural observations."""

    def __init__(self, observations: int) -> None:
        super().__init__(
            f"structural calibration complete after {observations} observations"
        )
        self.observations = observations


def _slot_of(fqn: str) -> str:
    m = _SLOT_RE.search(fqn)
    return f"slot_{int(m.group(1)):02d}" if m else "nonslot"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """fp32 cosine of two same-shaped tensors; None if either norm is 0."""
    a = a.flatten().float()
    b = b.flatten().float()
    na = a.norm()
    nb = b.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return None
    return float((a @ b).item() / (na * nb).item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_fingerprint(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().float().view(-1).cpu().numpy().tobytes()
    ).hexdigest()[:16]


class _PowerIteration:
    """sigma1..sigma4 by power iteration with rank-1 deflation.

    v <- A^T A v on the column side (n dims).  After each singular value,
    A <- A - s * v w^T (w = A v) removes that mode so the next iteration
    converges to the next singular vector.  Deterministic v0 (fixed seed
    per chunk + mode) so features are rank-reproducible for identical
    chunks.  All arithmetic fp32 on the input device.

    Convergence is gap-dependent: for rank-1-dominant inputs (the
    pathological class) sigma1 converges in a few iterations; for
    near-isotropic inputs it is slow.  ``sigma_method="svd"`` is the
    exact path; PI is kept for the spec section-4 accuracy audit and as
    a candidate cheap production path.
    """

    def __init__(self, a: torch.Tensor, iters: int, seed_base: int) -> None:
        self.a = a
        self.iters = iters
        self.seed_base = seed_base

    def _one(self, a: torch.Tensor, mode: int) -> tuple[float, torch.Tensor]:
        n = a.size(1)
        gen = torch.Generator(device=a.device).manual_seed(self.seed_base + mode)
        v = torch.randn(n, generator=gen, dtype=torch.float32, device=a.device)
        v = v / v.norm()
        for _ in range(self.iters):
            y = a @ v
            v = a.T @ y
            nv = v.norm()
            if nv.item() == 0.0:
                return 0.0, v
            v = v / nv
        y = a @ v
        return float(y.norm().item()), v

    def top4(self) -> list[float]:
        sigmas: list[float] = []
        a = self.a
        for mode in range(4):
            s, v = self._one(a, mode)
            sigmas.append(s)
            if s <= 0.0:
                sigmas.extend([0.0] * (4 - mode - 1))
                break
            w = a @ v  # w = A v1 = s1 * u1 (already scaled by s1)
            a = a - w.unsqueeze(1) @ v.unsqueeze(0)  # remove s1*u1*v1^T
        return sigmas


def singular_top4(
    a: torch.Tensor, method: str, pi_iters: int, seed_base: int
) -> list[float]:
    """Top-4 singular values of the fp32 matrix ``a``.

    method="svd": exact (torch.linalg.svdvals, truncated to 4) — default,
    no convergence assumptions.  method="pi": power iteration with
    deflation (see :class:`_PowerIteration`).
    """
    if method == "svd":
        s = torch.linalg.svdvals(a)
        out = [float(v) for v in s[:4].flatten().tolist()]
        out.extend([0.0] * (4 - len(out)))
        return out
    if method == "pi":
        return _PowerIteration(a, pi_iters, seed_base=seed_base).top4()
    raise ValueError(f"unknown sigma method {method!r} (use 'svd' or 'pi')")


class StructuralCalibration:
    """Handle to an installed structural calibration (diagnostics)."""

    def __init__(
        self,
        *,
        observations: int,
        ns_repeat: int,
        pi_iters: int,
        sigma_method: str,
        output_path: Path,
        artifact_dir: Path,
        rank: int,
        world_size: int,
        update_offset: int,
        full_sample_obs: int,
    ) -> None:
        self.observations_requested = observations
        self.observations = 0
        self.ns_repeat = ns_repeat
        self.pi_iters = pi_iters
        self.sigma_method = sigma_method
        self.output_path = output_path
        self.artifact_dir = artifact_dir
        self.rank = rank
        self.world_size = world_size
        self.update_offset = update_offset
        self.full_sample_obs = full_sample_obs
        self.max_rank_spread = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "output_path": str(self.output_path),
            "rank": self.rank,
            "world_size": self.world_size,
        }


def install_structural_calibration(
    optimizer: HybridCMuon,
    *,
    observations: int,
    ns_repeat: int,
    pi_iters: int,
    sigma_method: str = "pi",
    output_path: Path,
    artifact_dir: Path,
    rank: int,
    world_size: int,
    update_offset: int = 0,
    refs: dict[tuple[str, int], float] | None = None,
    full_sample_obs: int = 5,
) -> StructuralCalibration:
    """Replace ``optimizer.step`` with the structural shadow routine.

    ``optimizer`` must be the fully built production ``HybridCMuon``
    (routing / momentum buffers / config restored from the checkpoint).
    ``refs`` is the P3 calibration reference per (fqn, chunk) (may be
    None; rows then carry ref=None and rel_sig=None).  The original step
    is preserved on the instance as ``_struct_cal_original_step`` and is
    never called.
    """

    from sakuramoon.optim.cmuon import HybridCMuon

    if not isinstance(optimizer, HybridCMuon):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"structural calibration requires HybridCMuon, got {type(optimizer)!r}"
        )
    if observations < 1:
        raise ValueError("calibration observations must be >= 1")
    if ns_repeat < 1:
        raise ValueError("ns_repeat must be >= 1")
    if pi_iters < 1:
        raise ValueError("pi_iters must be >= 1")
    if full_sample_obs < 0:
        raise ValueError("full_sample_obs must be >= 0")
    if world_size > 1 and not dist.is_initialized():
        raise RuntimeError("world_size > 1 requires an initialized process group")

    handle = StructuralCalibration(
        observations=observations,
        ns_repeat=ns_repeat,
        pi_iters=pi_iters,
        sigma_method=sigma_method,
        output_path=Path(output_path),
        artifact_dir=Path(artifact_dir),
        rank=rank,
        world_size=world_size,
        update_offset=update_offset,
        full_sample_obs=full_sample_obs,
    )
    if rank == 0:
        handle.output_path.parent.mkdir(parents=True, exist_ok=True)
        handle.output_path.write_text("", encoding="utf-8")
        handle.artifact_dir.mkdir(parents=True, exist_ok=True)

    # per-(fqn, chunk) history for the temporal features
    prev_nesterov: dict[tuple[str, int], torch.Tensor] = {}
    prev_mom: dict[tuple[str, int], torch.Tensor] = {}
    # scheduling perturbation buffer for the NS label repeats
    dummy = None

    def structural_step() -> None:
        optimizer._sync_learning_rate()  # pyright: ignore[reportPrivateUsage]
        optimizer._validate_finite_gradients()  # pyright: ignore[reportPrivateUsage]
        mu = optimizer.cfg.momentum
        lr = float(optimizer.cfg.lr)
        ceiling = 10.0 * 0.2 * lr
        first_spec = optimizer.routing.cmuon_specs[0]
        device = first_spec.parameter.data.device
        nonlocal dummy
        if dummy is None:
            dummy = torch.randn(
                1024, 1024, dtype=torch.bfloat16, device=device
            )
        nonlocal_t0 = time.perf_counter()
        t_pi = 0.0
        t_reduce = 0.0
        t_cos = 0.0
        t_ns = 0.0
        rows: list[dict[str, object]] = []
        rms_flat: list[torch.Tensor] = []
        rms_owner: list[str] = []

        for spec in optimizer.routing.cmuon_specs:
            grad = spec.parameter.grad
            if grad is None:
                continue
            buf = optimizer._momenta[spec.parameter]  # pyright: ignore[reportPrivateUsage]
            grad_md = grad.to(buf.dtype)
            # Production-identical momentum EMA + Nesterov (no param update,
            # no NS in the parameter path; NS4 below is label-only).
            buf.lerp_(grad_md, 1.0 - mu)
            nesterov = grad_md.lerp(buf, mu)
            chunk_size = spec.chunk_size()
            n_chunks = (
                (nesterov,)
                if spec.chunk_count == 1
                else tuple(nesterov.split(chunk_size, dim=spec.chunk_dim))
            )
            g_chunks = (
                (grad_md,)
                if spec.chunk_count == 1
                else tuple(grad_md.split(chunk_size, dim=spec.chunk_dim))
            )
            shape = [int(s) for s in spec.parameter.shape]
            ns_steps = optimizer.cfg.ns_steps_for_role(spec.role)
            rescale = (
                spec.chunk_count
                if optimizer.cfg.chunk_rescale_sqrt_n  # pyright: ignore[reportPrivateUsage]
                else 1
            )
            slot = _slot_of(spec.name)
            for ci, (nchunk, gchunk) in enumerate(zip(n_chunks, g_chunks)):
                key = (spec.name, ci)
                prev_n = prev_nesterov.get(key)
                prev_m = prev_mom.get(key)
                prev_nesterov[key] = nchunk.detach().clone()
                prev_mom[key] = buf.narrow(
                    spec.chunk_dim, ci * chunk_size, chunk_size
                ).detach().clone()

                t0 = time.perf_counter()
                _sync(device)
                nf = nchunk.float()
                rms = nf.pow(2).mean().sqrt()
                fro = nf.norm()
                max_abs = float(nf.abs().max().item())
                t_reduce += time.perf_counter() - t0

                t0 = time.perf_counter()
                _sync(device)
                sigmas = singular_top4(
                    nf, handle.sigma_method, handle.pi_iters, seed_base=1000 + ci
                )
                t_pi += time.perf_counter() - t0
                fro2 = float(fro.item()) ** 2
                top1_energy = (sigmas[0] ** 2) / fro2 if fro2 > 0 else None
                top2_cum = (sigmas[0] ** 2 + sigmas[1] ** 2) / fro2 if fro2 > 0 else None
                top4_cum = (sum(s * s for s in sigmas) / fro2) if fro2 > 0 else None
                stable_rank = fro2 / (sigmas[0] ** 2) if sigmas[0] > 0 else None
                eff_rank4 = None
                if fro2 > 0 and all(s > 0 for s in sigmas):
                    shares = [s * s / fro2 for s in sigmas]
                    eff_rank4 = math.exp(
                        -sum(p * math.log(p) for p in shares if p > 0)
                    )

                t0 = time.perf_counter()
                _sync(device)
                c_gm = _cosine(gchunk, buf.narrow(spec.chunk_dim, ci * chunk_size, chunk_size))
                c_gn = _cosine(gchunk, nchunk)
                c_nm = _cosine(nchunk, prev_m) if prev_m is not None else None
                c_nn = _cosine(nchunk, prev_n) if prev_n is not None else None
                t_cos += time.perf_counter() - t0

                # ---- safety label: K production NS4 runs (no param write) ----
                alpha = cmuon_moonlight_alpha(
                    nchunk.shape[0], nchunk.shape[1], lr, rescale
                )
                ns_runs: list[dict[str, object]] = []
                for k in range(handle.ns_repeat):
                    t0 = time.perf_counter()
                    _sync(device)
                    ns = cmuon_zeroth_power(
                        nchunk, ns_steps, optimizer.cfg.ns_coefficients, optimizer.cfg.eps
                    )
                    if k + 1 < handle.ns_repeat:
                        _ = dummy @ dummy  # perturb scheduling between repeats
                    nsf = ns.float()
                    ns_rms = float(nsf.pow(2).mean().sqrt().item())
                    ns_max = float(nsf.abs().max().item())
                    delta_rms = alpha * ns_rms
                    finite = math.isfinite(ns_rms) and math.isfinite(ns_max)
                    cat = (not finite) or delta_rms > ceiling
                    ns_runs.append(
                        {
                            "ns_rms": ns_rms,
                            "ns_max": ns_max,
                            "delta_rms": delta_rms,
                            "catastrophic": bool(cat),
                            "ns_hash": _tensor_fingerprint(ns),
                        }
                    )
                    t_ns += time.perf_counter() - t0
                hazard_rate = (
                    sum(1 for r in ns_runs if r["catastrophic"]) / len(ns_runs)
                )
                label = "DANGEROUS" if hazard_rate > 0 else "SAFE"

                ref = refs.get(key) if refs is not None else None
                rms_v = float(rms.item())
                rows.append(
                    {
                        "fqn": spec.name,
                        "slot": slot,
                        "role": spec.role,
                        "chunk": ci,
                        "n_chunks": spec.chunk_count,
                        "shape": shape,
                        "rms": rms_v,
                        "fro": float(fro.item()),
                        "max_abs": max_abs,
                        "ref": ref,
                        "rel_sig": (rms_v / ref) if ref else None,
                        "sigma1": sigmas[0],
                        "sigma2": sigmas[1],
                        "sigma3": sigmas[2],
                        "sigma4": sigmas[3],
                        "top1_energy": top1_energy,
                        "top2_cum_energy": top2_cum,
                        "top4_cum_energy": top4_cum,
                        "stable_rank": stable_rank,
                        "eff_rank4": eff_rank4,
                        "cos_grad_mom": c_gm,
                        "cos_grad_nest": c_gn,
                        "cos_nest_mom_prev": c_nm,
                        "cos_nest_nest_prev": c_nn,
                        "alpha": alpha,
                        "ns_steps": ns_steps,
                        "ns_runs": ns_runs,
                        "hazard_rate": hazard_rate,
                        "label": label,
                    }
                )
                rms_flat.append(rms)
                rms_owner.append(f"{spec.name}#chunk{ci}")

        max_spread = 0.0
        max_spread_chunk = ""
        ok = True
        if world_size > 1 and rms_flat:
            flat = torch.stack(rms_flat)
            lo = flat.clone()
            hi = flat.clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            spread = (hi - lo).max()
            max_spread = float(spread.item())
            ok = max_spread == 0.0
            if not ok:
                idx = int((hi - lo).argmax().item())
                max_spread_chunk = rms_owner[idx]
            handle.max_rank_spread = max(handle.max_rank_spread, max_spread)
            if not ok:
                raise FloatingPointError(
                    "structural calibration: momentum/Nesterov state diverges "
                    f"across ranks (spread={max_spread:.3e} at "
                    f"{max_spread_chunk}); the classifier input would not be "
                    "rank-consistent"
                )

        handle.observations += 1
        obs = handle.observations
        # Full nesterov-chunk dump (rank 0, observations 1..full_sample_obs):
        # the offline exact-SVD reference for the spec section-4 power-
        # iteration accuracy audit (HCU/CPU SVD of these tensors vs the PI
        # features recorded in the JSONL rows for the same obs).
        if rank == 0 and 1 <= obs <= handle.full_sample_obs:
            sample_dir = handle.artifact_dir / f"full-samples/obs-{obs:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for spec in optimizer.routing.cmuon_specs:
                g = spec.parameter.grad
                if g is None:
                    continue
                buf = optimizer._momenta[spec.parameter]  # pyright: ignore[reportPrivateUsage]
                grad_md = g.to(buf.dtype)
                nest = grad_md.lerp(buf, mu)
                cs = spec.chunk_size()
                chunk_t = (
                    (nest,)
                    if spec.chunk_count == 1
                    else tuple(
                        nest.narrow(spec.chunk_dim, ci * cs, cs)
                        for ci in range(spec.chunk_count)
                    )
                )
                for ci, ct in enumerate(chunk_t):
                    torch.save(
                        {
                            "fqn": spec.name,
                            "chunk": ci,
                            "tensor": ct.detach().cpu().float(),
                            "meta": {
                                "shape": [int(x) for x in spec.parameter.shape],
                                "chunk_dim": spec.chunk_dim,
                                "chunk_size": cs,
                            },
                        },
                        sample_dir / f"chunk-{spec.name.replace('.', '_').replace('/', '_')}-c{ci}.pt",
                    )
        if rank >= 0:
            record = {
                "obs": obs,
                "abs_update": update_offset + obs,
                "lr": lr,
                "ceiling": ceiling,
                "cost_ms": {
                    "feat_pi": (t_pi) * 1000.0,
                    "feat_reduce": (t_reduce) * 1000.0,
                    "feat_cos": (t_cos) * 1000.0,
                    "ns_label": (t_ns) * 1000.0,
                    "step_total": (time.perf_counter() - nonlocal_t0) * 1000.0,
                },
                "rank_consistency": {
                    "max_spread": max_spread,
                    "max_spread_chunk": max_spread_chunk,
                    "ok": ok,
                },
                "rows": rows,
            }
            with handle.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        if obs >= handle.observations_requested:
            raise StructuralCalibrationComplete(obs)

    optimizer._struct_cal_original_step = optimizer.step  # type: ignore[attr-defined]
    optimizer.step = structural_step  # type: ignore[method-assign]
    return handle
