"""Shadow-gradient calibration for the Guarded Canonical CMuon candidate.

Purpose (design Section 3/22): before any guard threshold is fixed, observe
the REAL signal distribution — per (FQN, chunk) Nesterov-matrix magnitudes
over a bounded number of production updates — from a healthy complete
checkpoint, on an isolated host, with the production model / data / batch
policy, without updating any parameter.

Mechanism
---------
:func:`install_guard_calibration` replaces the ``step`` of a fully built
production ``HybridCMuon`` optimizer with a shadow observation routine:

* the momentum EMA is updated exactly like production
  (``buf.lerp_(grad_md, 1 - mu)`` in the configured momentum dtype),
* the Nesterov matrix is computed exactly like production
  (``grad_md.lerp(buf, mu)``) — this is the true NS input, i.e. the guard
  signal,
* per (FQN, chunk) ``grad_rms`` / ``nesterov_rms`` / ``nesterov_fro`` are
  recorded (chunks follow the same chunking NS uses: ``spec.chunk_dim`` /
  ``spec.chunk_count``),
* cross-rank consistency of every per-chunk ``nesterov_rms`` is verified
  with two device all-reduces (MIN/MAX) each update.  A nonzero spread
  means the momentum recursion is NOT rank-exact and the run fails closed
  (Section 8 requires hard failure above the HCU element-wise noise
  tolerance; the expectation is a spread of exactly 0.0 for the bf16 lerp,
  which is recorded as evidence),
* NO parameter is updated, NO Newton-Schulz is run, NO AdamW step is taken
  (the inner torch optimizer is never stepped).

After ``steps`` observations the shadow step raises
:class:`GuardCalibrationComplete`, which the training loop and the
production lifecycle treat as a clean stop (no failure bundle).

The production side effects (W&B telemetry, checkpoint publishing,
sampling, evaluation) are neutralized by the calibration hooks in
``sakuramoon.train.production``; this module only owns the shadow step and
the record file.

Record file (JSONL, written by rank 0 only)::

    {"update": <1-based observation index>,
     "lr": <float>, "wall_ms": <float>,
     "rank_consistency": {"max_spread": <float>, "max_spread_chunk": "...",
                          "ok": <bool>},
     "rows": [{"fqn": ..., "role": ..., "chunk": i, "n_chunks": n,
               "shape": [r, c], "grad_rms": ..., "nesterov_rms": ...,
               "nesterov_fro": ...}, ...]}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sakuramoon.optim.cmuon import HybridCMuon


class GuardCalibrationComplete(Exception):
    """Clean stop after the requested number of calibration observations.

    Carries the number of completed observations; the loop/lifecycle must
    treat this as success, not as a failure.
    """

    def __init__(self, observations: int) -> None:
        super().__init__(
            f"guard calibration complete after {observations} observations"
        )
        self.observations = observations


class GuardCalibration:
    """Handle to an installed shadow calibration (diagnostics + restore)."""

    def __init__(
        self,
        *,
        steps: int,
        output_path: Path,
        rank: int,
        world_size: int,
        update_offset: int,
    ) -> None:
        self.steps = steps
        self.output_path = output_path
        self.rank = rank
        self.world_size = world_size
        self.update_offset = update_offset
        self.observations = 0
        self.max_rank_spread = 0.0
        self.max_rank_spread_chunk = ""

    def summary(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "observations": self.observations,
            "output_path": str(self.output_path),
            "rank": self.rank,
            "world_size": self.world_size,
            "update_offset": self.update_offset,
            "momentum_rank_exact": (
                None if self.world_size == 1 else self.max_rank_spread == 0.0
            ),
            "max_rank_spread": self.max_rank_spread,
            "max_rank_spread_chunk": self.max_rank_spread_chunk,
        }


def _chunk_tensors(
    tensor: torch.Tensor, chunk_dim: int, chunk_count: int, chunk_size: int
) -> tuple[torch.Tensor, ...]:
    if chunk_count == 1:
        return (tensor,)
    return tuple(tensor.split(chunk_size, dim=chunk_dim))


def install_guard_calibration(
    optimizer: HybridCMuon,
    *,
    steps: int,
    output_path: Path,
    rank: int,
    world_size: int,
    update_offset: int = 0,
) -> GuardCalibration:
    """Replace ``optimizer.step`` with the shadow observation routine.

    ``optimizer`` must be the fully built production ``HybridCMuon``
    (routing / momentum buffers / config restored from the checkpoint).
    The original step is preserved on the instance as
    ``_calibration_original_step`` and is never called.
    """

    from sakuramoon.optim.cmuon import HybridCMuon

    # Defensive runtime check: the annotation documents the contract, but
    # production builds either optimizer kind from config.
    if not isinstance(optimizer, HybridCMuon):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"guard calibration requires HybridCMuon, got {type(optimizer)!r}"
        )
    if steps < 1:
        raise ValueError("calibration steps must be >= 1")
    if world_size > 1 and not dist.is_initialized():
        raise RuntimeError("world_size > 1 requires an initialized process group")

    handle = GuardCalibration(
        steps=steps,
        output_path=Path(output_path),
        rank=rank,
        world_size=world_size,
        update_offset=update_offset,
    )
    if rank == 0:
        handle.output_path.parent.mkdir(parents=True, exist_ok=True)
        handle.output_path.write_text("", encoding="utf-8")

    def shadow_step() -> None:
        # Same pre-step checks as production (finite grads; LR sync). The
        # private calls are the sanctioned seam: the shadow step must reuse
        # the exact production logic, not a copy.
        optimizer._sync_learning_rate()  # pyright: ignore[reportPrivateUsage]
        optimizer._validate_finite_gradients()  # pyright: ignore[reportPrivateUsage]
        mu = optimizer.cfg.momentum
        started = time.perf_counter()
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
            # no NS).
            buf.lerp_(grad_md, 1.0 - mu)
            nesterov = grad_md.lerp(buf, mu)
            chunk_size = spec.chunk_size()
            n_chunks = _chunk_tensors(
                nesterov, spec.chunk_dim, spec.chunk_count, chunk_size
            )
            g_chunks = _chunk_tensors(
                grad_md, spec.chunk_dim, spec.chunk_count, chunk_size
            )
            shape = [int(s) for s in spec.parameter.shape]
            for ci, (nchunk, gchunk) in enumerate(zip(n_chunks, g_chunks)):
                nf = nchunk.float()
                gf = gchunk.float()
                g_rms = float(gf.pow(2).mean().sqrt().item())
                n_rms = nf.pow(2).mean().sqrt()
                n_fro = float(nf.norm().item())
                rows.append(
                    {
                        "fqn": spec.name,
                        "role": spec.role,
                        "chunk": ci,
                        "n_chunks": spec.chunk_count,
                        "shape": shape,
                        "grad_rms": g_rms,
                        "nesterov_rms": float(n_rms.item()),
                        "nesterov_fro": n_fro,
                    }
                )
                rms_flat.append(n_rms)
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
            if max_spread > handle.max_rank_spread:
                handle.max_rank_spread = max_spread
                handle.max_rank_spread_chunk = max_spread_chunk
            if not ok:
                # Hard fail: the momentum recursion must be rank-exact.
                raise FloatingPointError(
                    "guard calibration: momentum/Nesterov state diverges "
                    f"across ranks (spread={max_spread:.3e} at "
                    f"{max_spread_chunk}); the guard decision would not be "
                    "rank-consistent"
                )

        handle.observations += 1
        if rank == 0:
            record = {
                "update": handle.observations,
                "abs_update": update_offset + handle.observations,
                "lr": float(optimizer.cfg.lr),
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "rank_consistency": {
                    "max_spread": max_spread,
                    "max_spread_chunk": max_spread_chunk,
                    "ok": ok,
                },
                "rows": rows,
            }
            with handle.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        if handle.observations >= handle.steps:
            raise GuardCalibrationComplete(handle.observations)

    optimizer._calibration_original_step = optimizer.step  # type: ignore[attr-defined]
    optimizer.step = shadow_step  # type: ignore[method-assign]
    return handle
