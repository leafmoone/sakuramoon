"""M3 latent flow-matching loop (plan §5, §13 M3; §4.3 pre-encode design).

Data contract (frozen with the M3 z_hr pre-encode, commit 8fa778c):

* ``z_hr`` — read from a :class:`LatentStore` (fp16 ``[128, g, g]`` per
  crop; the crop is pinned at box ``(0, 0)`` of the sample's HR bucket).
* ``z_lr`` — ``E_Mage(Bicubic4x(LQ))`` computed on the fly by the frozen
  VAE (plan §4.3 anchor), batched, ``no_grad``.
* The crop stays pinned at box ``(0, 0)`` for every exposure of a sample;
  only the degradation draw varies with ``(data_cycle, exposure_index)``,
  so the pre-encoded ``z_hr`` remains the target of every exposure.

Deterministic schedule (plan §11.5, mirrors the M2 loop):

    step s (global, across all ranks)
        exposure_index = s % _EXPOSURE_PER_CYCLE
        data_cycle     = s // _EXPOSURE_PER_CYCLE
        slot i of rank r:  (s * (bs * world) + r * bs + i) % n

A resume at step ``s0`` reproduces the exact exposure stream (bit-exact
degradation draws; the stochastic flow noise is outside the §11.5
contract, as in M2).

Prefetch: ``[latent_flow].prefetch_depth`` sets how many step-batches the
CPU producer keeps ready ahead of the accelerator consumer (2 =
double-buffered, the M3 default; 4 = quad buffer, the M1 #8 data-wait fix
for Phase I; 0 = synchronous, M2-style canary). ``[latent_flow].producer``
selects the backend: ``"thread"`` (default; a thread pool in this
process, GIL-bound) or ``"process"`` (a forked process pool created BEFORE
any HCU context exists; the dataset/store context is inherited
copy-on-write and each worker re-tunes its own torch intra-op pool from
OMP_NUM_THREADS, so the pool is not GIL-bound). Every fetch is a pure
function of ``(step, slot)`` in either backend, so the §11.5 stream stays
bit-exact. The producer records per-stage wall-times (shard/decode/crop/
degradation/z_hr) and the loop exposes producer/consumer throughput plus
ready-queue occupancy (M1 #8 gate: data-wait fraction, producer >= 1.25x
consumer, ready queue not persistently empty).

Flow math (plan §5, ``anime_sr.flow``):

    delta = z_hr - z_lr ; r0 = sigma * eps ; rt = (1-t) r0 + t delta
    v*    = delta - r0   ; loss = MSE(v_hat, v*)      (v-prediction)

M3 checklist mapping (plan §13, "M3 未通过，不启动正式模型"):

1. flow direction correct      -> ``cos_v`` rises above ~0.5 in validation
2. 1-step moves toward HR      -> ``toward_frac_1step`` > 0.5 and rising
3. 4-step numerically stable  -> 1-step significantly better than the
   anchor, ``l1_4step / l1_1step <= 1.05``, no NaN/Inf or trajectory
   explosion (revised §13 #3; "4-step must beat 1-step" moved to the
   quality-mode release gate, no longer blocking Phase I)
4. non-square buckets correct  -> M4 (single 1024 bucket in M3 smoke)
5. window mask correct         -> no seam artifacts in decoded val samples
6. resume continuous           -> short re-run from a ckpt, loss continues
7. DDP == single card          -> M4 (2 cards)
8. decoder gradient to trunk   -> Stage II (M5)
9. no window seams             -> decode val samples, visual check
"""

from __future__ import annotations

import json
import multiprocessing.pool
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing.pool import AsyncResult as _MPAsyncResult
from multiprocessing.pool import Pool as _MPPool
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from anime_sr.config.schema import Config
from anime_sr.data.clean_score import (
    CleanScoreCache,
    build_clean_score_report,
    clean_score_gate_retained,
)
from anime_sr.data.degradation import degrade_hr
from anime_sr.data.latent_store import LatentStore, read_index
from anime_sr.data.pipeline import _EXPOSURE_PER_CYCLE, SampleMeta, SRDataset
from anime_sr.data.pool_sampler import SlotMap
from anime_sr.flow.path import (
    interpolate,
    sample_sigma,
    sample_source_noise,
    target_velocity,
)
from anime_sr.flow.sampling import FlowSampler
from anime_sr.flow.solver import euler_trajectory, heun_trajectory
from anime_sr.model.uflow import (
    AnimeSRModel,
    UFlowSR,
    apply_pixel_zero_init,
    count_parameters,
)
from anime_sr.train.ckpt_v2 import load_v2
from anime_sr.train.ema_sample import SampleEMA
from anime_sr.train.pixel_baseline import (
    _cosine_lr,
    _optimizer_for,
    _save_ckpt,
    _unwrap,
)
from anime_sr.vae.mage import load_frozen_vae

#: Fixed validation exposure (distinct from the train stream's cycling e).
_VAL_CYCLE = 0
_VAL_EXPOSURE = 24


class _LatentVelocity(nn.Module):
    """Adapts ``UFlowSR`` to the FlowSampler ``VelocityModel`` protocol
    ``v = model(rt, t, sigma, cond)`` with ``cond = z_lr`` and the pixel
    features disabled (M3 latent-only smoke, plan §13)."""

    def __init__(self, trunk: UFlowSR) -> None:
        super().__init__()
        self.trunk = trunk

    def forward(
        self,
        rt: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        z_lr = cond
        return self.trunk(rt, z_lr, t, sigma)


class _PixelVelocity(nn.Module):
    """Adapts ``AnimeSRModel`` to the FlowSampler ``VelocityModel`` protocol
    ``v = model(rt, t, sigma, cond)`` with ``cond = (z_lr, lq_rgb)`` — the
    Phase I-P pixel path is active (plan §8 transition)."""

    def __init__(self, model: AnimeSRModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        rt: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        z_lr, lq = cond
        return self.model(rt, z_lr, lq, t, sigma)


def latent_sample_index(
    step: int, rank: int, i: int, bs: int, world: int, n: int
) -> int:
    """M2-style deterministic stream slot (plan §11.5): the global slot
    ``step * bs * world + rank * bs + i`` wrapped by the set size. A resume
    at step ``s0`` reproduces the same stream position at step ``s``."""
    return (step * (bs * world) + rank * bs + i) % n


def build_flow_targets(
    z_hr: torch.Tensor,
    z_lr: torch.Tensor,
    cfg: Config,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One flow-matching batch of targets (plan §5.2/§5.3/§5.6).

    Returns ``(rt, v_star, sigma, t)``: the §5.6 sigma mix (zero fraction +
    noise range from ``[flow]``), a uniform time sample, ``r0 = sigma*eps``,
    ``rt = (1-t) r0 + t delta`` and ``v* = delta - r0`` (delta = z_hr - z_lr).
    With a fixed ``generator`` the whole draw is reproducible (tests/resume).
    """
    f = cfg.flow
    b = z_hr.shape[0]
    out_dtype = dtype if dtype is not None else z_lr.dtype
    sigma_range = (
        float(f.train_sigma_noise_range[0]),
        float(f.train_sigma_noise_range[1]),
    )
    sigma = sample_sigma(
        b,
        f.train_sigma_zero_fraction,
        sigma_range,
        generator=generator,
        device=device,
        dtype=out_dtype,
    )
    t = torch.rand(b, device=device, dtype=torch.float32, generator=generator)
    delta = z_hr - z_lr
    r0 = sample_source_noise(
        sigma, delta.shape, generator=generator, device=device, dtype=delta.dtype
    )
    rt = interpolate(r0, delta, t)
    v_star = target_velocity(delta, r0)
    return rt, v_star, sigma, t


def latent_val_metrics(
    z_hr: torch.Tensor,
    z_lr: torch.Tensor,
    z_hat: torch.Tensor,
) -> dict[str, float]:
    """Validation metrics for one solver output (plan §13 checklist).

    ``toward_frac`` = fraction of samples whose predicted latent is closer
    to ``z_hr`` than the LQ anchor ``z_lr`` ("1-step output moves toward
    HR", checklist #2)."""
    diff_hat = (z_hat - z_hr).float()
    diff_anchor = (z_lr - z_hr).float()
    per_hat = diff_hat.abs().mean(dim=(1, 2, 3))
    per_anchor = diff_anchor.abs().mean(dim=(1, 2, 3))
    return {
        "l1": diff_hat.abs().mean().item(),
        "l1_anchor": diff_anchor.abs().mean().item(),
        "toward_frac": float((per_hat < per_anchor).float().mean().item()),
    }


def velocity_cosine(v_hat: torch.Tensor, v_star: torch.Tensor) -> float:
    """Mean per-sample cosine between predicted and target velocity
    (checklist #1: flow direction correct -> rising above ~0.5)."""
    a = v_hat.reshape(v_hat.shape[0], -1).float()
    b = v_star.reshape(v_star.shape[0], -1).float()
    cos = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
    return float(cos.mean().item())


#: Revised M3 #3 probe grid: the endpoint-consistency / on-path times.
_ENDPOINT_TS = (0.0, 0.25, 0.5, 0.75)
_ENDPOINT_LABELS = {0.0: "t0", 0.25: "t25", 0.5: "t50", 0.75: "t75"}


def endpoint_consistency(
    mod: nn.Module,
    z_hr: torch.Tensor,
    z_lr: torch.Tensor,
    autocast: Callable[[], Any],
    device: torch.device,
    lq: torch.Tensor | None = None,
) -> dict[str, float]:
    """Endpoint consistency of the learned field (revised plan §13 #3).

    With r0 = 0 (Faithful sigma=0) the exact path is r_t = t*delta, so the
    endpoint estimate is ``delta_hat_t = r_t + (1-t) v_theta(r_t, t)`` and
    the endpoint L1 ``|delta_hat_t - delta|`` is reported at t in
    {0, .25, .5, .75} (at t=0 it equals the 1-step L1; for an exact field
    it goes to 0 as t -> 1). Deterministic: fixed t grid, sigma=0.

    ``lq`` (Phase I-P pixel model): when given, ``mod`` is the full
    AnimeSRModel and the velocity call feeds the pixel path
    (``mod(r_t, z_lr, lq, t_vec, sigma)``); trunk-only models leave it
    None.
    """
    b = z_hr.shape[0]
    delta = z_hr - z_lr
    sigma = torch.zeros(b, device=device)
    out: dict[str, float] = {}
    with torch.no_grad(), autocast():
        for t in _ENDPOINT_TS:
            r_t = t * delta
            t_vec = torch.full((b,), t, device=device, dtype=torch.float32)
            v_t = mod(r_t, z_lr, lq, t_vec, sigma) if lq is not None else mod(r_t, z_lr, t_vec, sigma)
            delta_hat = r_t + (1.0 - t) * v_t
            out[f"ep_l1_{_ENDPOINT_LABELS[t]}"] = (
                (delta_hat - delta).abs().float().mean().item()
            )
    return out


def trajectory_deviation(
    mod: nn.Module,
    z_hr: torch.Tensor,
    z_lr: torch.Tensor,
    *,
    solver: str,
    n_steps: int,
    autocast: Callable[[], Any],
    device: torch.device,
    lq: torch.Tensor | None = None,
) -> dict[str, float]:
    """D_t = mean|r_hat(t_k) - r_true(t_k)| for a val sample with GT
    (revised plan §13 #3). With r0 = 0 the true path is r_true(t) = t*delta.

    ``solver`` is ``"euler"`` or ``"heun"`` (the project's last-euler
    quality mode); ``n_steps`` splits [0, 1]. Returns ``D_t<100*t_k>`` at
    every sub-step end (key ``D_t100`` is the endpoint deviation, which
    equals the 1-step L1 for ``solver="euler", n_steps=1``)."""
    b = z_hr.shape[0]
    delta = z_hr - z_lr
    sigma = torch.zeros(b, device=device)
    r0 = torch.zeros_like(delta)

    def v_fn(r: torch.Tensor, t: float) -> torch.Tensor:
        t_vec = torch.full((b,), t, device=device, dtype=torch.float32)
        if lq is not None:
            return mod(r, z_lr, lq, t_vec, sigma)
        return mod(r, z_lr, t_vec, sigma)

    out: dict[str, float] = {}
    with torch.no_grad(), autocast():
        if solver == "euler":
            _final, states = euler_trajectory(r0, v_fn, n_steps)
        elif solver == "heun":
            _final, states = heun_trajectory(r0, v_fn, n_steps, last_euler=True)
        else:
            raise ValueError(f"unknown solver {solver!r}")
        for k, r_k in enumerate(states):
            t_k = (k + 1) / n_steps
            out[f"D_t{round(t_k * 100):03d}"] = (
                (r_k - t_k * delta).abs().float().mean().item()
            )
    return out


def _validate_latent(
    model: nn.Module,
    vae,
    ds: SRDataset,
    order: list[int],
    n: int,
    store: LatentStore | None,
    cfg: Config,
    device: torch.device,
    rank: int,
    step: int,
    bucket_hr: int,
    autocast: Callable[[], Any],
) -> None:
    """Quantitative M3 checklist subset on a fixed val slice (rank 0 only).

    ``store=None`` (P1 ④ ``zhr_source="onfly"``): the val slice is built
    from the train index, so the samples have no pre-encoded rows — z_hr is
    encoded on the fly with the frozen VAE, like the held-out val path."""
    if rank != 0:
        return
    lf = cfg.latent_flow
    mod = _unwrap(model)
    was_training = mod.training
    mod.eval()
    pixel = lf.pixel_features
    if pixel:
        sampler = FlowSampler(_PixelVelocity(cast(AnimeSRModel, mod)))
    else:
        sampler = FlowSampler(_LatentVelocity(cast(UFlowSR, mod)))
    g = torch.Generator(device=str(device)).manual_seed(0x5EED ^ (step % (2**31)))
    n_val = min(lf.val_samples, n)
    js = [order[k % n] for k in range(n_val)]
    z_hrs: list[torch.Tensor] = []
    z_lrs: list[torch.Tensor] = []
    lqs: list[torch.Tensor] = []
    with torch.no_grad():
        for j in js:
            meta = ds.samples[j]
            hr_full = ds.decode_hr(meta)
            x, y = ds.crop(meta, 0, 0)
            hr_crop = hr_full[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()
            lq, _ = degrade_hr(
                hr_crop,
                cfg,
                global_seed=ds.global_seed,
                sample_id=meta.sample_id,
                data_cycle=_VAL_CYCLE,
                exposure_index=_VAL_EXPOSURE,
            )
            # degrade_hr returns the unbatched LQ [3, h, w]; interpolate and
            # the frozen VAE both take a batch dim (plan §4.3 anchor). The
            # val slice is built on CPU (decode_hr), but the frozen VAE
            # lives on its device — move in, like the train path (l387).
            lq_up = F.interpolate(
                lq.unsqueeze(0), size=(bucket_hr, bucket_hr), mode="bicubic"
            ).squeeze(0)
            z_lr = vae.encode(lq_up.unsqueeze(0).to(vae.device, vae.dtype)).squeeze(0)
            if store is not None:
                z_hr = store.read(meta.sample_id).to(vae.dtype)
            else:
                # P1 ④ on-fly: no store — encode the HR crop (frozen VAE, §4.3)
                z_hr = vae.encode(
                    hr_crop.unsqueeze(0).to(vae.device, vae.dtype)
                ).squeeze(0)
            z_hrs.append(z_hr)
            z_lrs.append(z_lr)
            if pixel:
                lqs.append(lq)  # (3, h, w) fp32 at LQ resolution
        z_hr_b = torch.stack(z_hrs).to(device)
        z_lr_b = torch.stack(z_lrs).to(device)
        lq_b: torch.Tensor | None = (
            torch.stack(lqs).to(device) if pixel else None
        )
        # Run the model forwards under the same autocast as the train loop:
        # build_flow_targets emits fp32 ``t``/``rt`` (the §5.3 uniform-t draw
        # is fp32) which would otherwise hit the bf16 trunk weights raw.
        with autocast():
            if pixel:
                z1 = sampler.one_step(z_lr_b, (z_lr_b, lq_b), sigma=0.0, generator=g)
                z4 = sampler.four_step(z_lr_b, (z_lr_b, lq_b), sigma=0.0, generator=g)
            else:
                z1 = sampler.one_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
                z4 = sampler.four_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
        m1 = latent_val_metrics(z_hr_b, z_lr_b, z1)
        m4 = latent_val_metrics(z_hr_b, z_lr_b, z4)
        # flow-direction probe at a random t (reproducible per step)
        with autocast():
            rt, v_star, sigma, t = build_flow_targets(
                z_hr_b, z_lr_b, cfg, generator=g, device=device
            )
            v_hat = (
                mod(rt, z_lr_b, lq_b, t, sigma) if pixel else mod(rt, z_lr_b, t, sigma)
            )
        cos_v = velocity_cosine(v_hat, v_star)
        # Revised M3 #3: 4-step stability is judged by ratio + trajectory
        # behavior (all deterministic: fixed val slice, sigma=0, fixed t grid;
        # the per-step seeded draw above only feeds the cos_v probe).
        ep = endpoint_consistency(mod, z_hr_b, z_lr_b, autocast, device, lq=lq_b)
        d4 = trajectory_deviation(
            mod,
            z_hr_b,
            z_lr_b,
            solver="heun",
            n_steps=4,
            autocast=autocast,
            device=device,
            lq=lq_b,
        )
        ratio_41 = m4["l1"] / m1["l1"] if m1["l1"] > 1e-8 else float("inf")
    if was_training:
        mod.train()
    print(
        f"[latent] val step {step}: l1_1={m1['l1']:.4f} l1_4={m4['l1']:.4f} "
        f"ratio_4_1={ratio_41:.4f} l1_anchor={m1['l1_anchor']:.4f} "
        f"toward_1={m1['toward_frac']:.2f} toward_4={m4['toward_frac']:.2f} "
        f"cos_v={cos_v:.3f} (n={n_val})",
        flush=True,
    )
    print(
        f"[latent] val step {step}: endpoint_l1 "
        + " ".join(f"{k.replace('ep_l1_', '')}={v:.4f}" for k, v in ep.items())
        + " | D4(heun) "
        + " ".join(f"{k}={v:.4f}" for k, v in d4.items()),
        flush=True,
    )


def _validate_heldout(
    model: nn.Module,
    vae,
    val_ds: SRDataset,
    cfg: Config,
    device: torch.device,
    rank: int,
    step: int,
    bucket_hr: int,
    autocast: Callable[[], Any],
) -> None:
    """P1 ① held-out validation (rank 0 only).

    Sampled from the VALIDATION split (never the train stream) with a fully
    fixed seed, so the numbers are reproducible run-to-run and comparable
    across runs. Held-out samples have no pre-encoded store row (the
    LatentStore covers train crops only), so z_hr is encoded on the fly by
    the frozen VAE — the same path the P1 ④ on-the-fly design generalizes.
    """
    if rank != 0:
        return
    lf = cfg.latent_flow
    n = min(lf.val_heldout_samples, len(val_ds.samples))
    if n <= 0:
        return
    mod = _unwrap(model)
    was_training = mod.training
    mod.eval()
    pixel = lf.pixel_features
    if pixel:
        sampler = FlowSampler(_PixelVelocity(cast(AnimeSRModel, mod)))
    else:
        sampler = FlowSampler(_LatentVelocity(cast(UFlowSR, mod)))
    # Fixed seed (no step mix): the held-out slice must be reproducible.
    g = torch.Generator(device=str(device)).manual_seed(0x5EED)
    z_hrs: list[torch.Tensor] = []
    z_lrs: list[torch.Tensor] = []
    lqs: list[torch.Tensor] = []
    with torch.no_grad():
        for k in range(n):
            # validation split order = index order (deterministic artifact)
            meta = val_ds.samples[k]
            hr_full = val_ds.decode_hr(meta)
            x, y = val_ds.crop(meta, _VAL_CYCLE, _VAL_EXPOSURE)
            hr_crop = hr_full[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()
            lq, _ = degrade_hr(
                hr_crop,
                cfg,
                global_seed=val_ds.global_seed,
                sample_id=meta.sample_id,
                data_cycle=_VAL_CYCLE,
                exposure_index=_VAL_EXPOSURE,
            )
            lq_up = F.interpolate(
                lq.unsqueeze(0), size=(bucket_hr, bucket_hr), mode="bicubic"
            ).squeeze(0)
            z_lr = vae.encode(lq_up.unsqueeze(0).to(vae.device, vae.dtype)).squeeze(0)
            # held-out samples have no store row: encode the HR crop with the
            # frozen VAE (mirrors the train-path z_hr construction, §4.3)
            z_hr = vae.encode(
                hr_crop.unsqueeze(0).to(vae.device, vae.dtype)
            ).squeeze(0)
            z_hrs.append(z_hr)
            z_lrs.append(z_lr)
            if pixel:
                lqs.append(lq)
        z_hr_b = torch.stack(z_hrs).to(device)
        z_lr_b = torch.stack(z_lrs).to(device)
        lq_b: torch.Tensor | None = (
            torch.stack(lqs).to(device) if pixel else None
        )
        with autocast():
            if pixel:
                z1 = sampler.one_step(z_lr_b, (z_lr_b, lq_b), sigma=0.0, generator=g)
                z4 = sampler.four_step(z_lr_b, (z_lr_b, lq_b), sigma=0.0, generator=g)
            else:
                z1 = sampler.one_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
                z4 = sampler.four_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
        m1 = latent_val_metrics(z_hr_b, z_lr_b, z1)
        m4 = latent_val_metrics(z_hr_b, z_lr_b, z4)
        with autocast():
            rt, v_star, sigma, t = build_flow_targets(
                z_hr_b, z_lr_b, cfg, generator=g, device=device
            )
            v_hat = (
                mod(rt, z_lr_b, lq_b, t, sigma) if pixel else mod(rt, z_lr_b, t, sigma)
            )
        cos_v = velocity_cosine(v_hat, v_star)
        ep = endpoint_consistency(mod, z_hr_b, z_lr_b, autocast, device, lq=lq_b)
        d4 = trajectory_deviation(
            mod,
            z_hr_b,
            z_lr_b,
            solver="heun",
            n_steps=4,
            autocast=autocast,
            device=device,
            lq=lq_b,
        )
        ratio_41 = m4["l1"] / m1["l1"] if m1["l1"] > 1e-8 else float("inf")
    if was_training:
        mod.train()
    print(
        f"[latent] heldout-val step {step}: l1_1={m1['l1']:.4f} "
        f"l1_4={m4['l1']:.4f} ratio_4_1={ratio_41:.4f} "
        f"l1_anchor={m1['l1_anchor']:.4f} toward_1={m1['toward_frac']:.2f} "
        f"toward_4={m4['toward_frac']:.2f} cos_v={cos_v:.3f} "
        f"(n={n}, validation split)",
        flush=True,
    )
    print(
        f"[latent] heldout-val step {step}: endpoint_l1 "
        + " ".join(f"{k.replace('ep_l1_', '')}={v:.4f}" for k, v in ep.items())
        + " | D4(heun) "
        + " ".join(f"{k}={v:.4f}" for k, v in d4.items()),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Process-pool producer (producer="process"): the worker body is a
# MODULE-LEVEL function so the pickled task payload is just (slot, step);
# the heavy dataset/store/cfg context is inherited copy-on-write through
# the fork and lives in _PRODUCER_CTX. The body is line-for-line the same
# per-stage work as the in-loop _fetch closure -> the §11.5 stream is
# unchanged (only the transport differs).
# ---------------------------------------------------------------------------
_PRODUCER_CTX: dict[str, Any] | None = None


def _pp_worker_init() -> None:
    """Pool initializer (fork start method, runs once per worker).

    The producer context is inherited COW — nothing is pickled. The parent
    pinned torch intra-op threads to 1 (P1 ⑤); a worker that re-runs small
    CPU torch ops (degrade convs) with a single intra-op thread leaves its
    cores idle, so re-tune from OMP_NUM_THREADS (the launch env, inherited
    through the fork)."""
    import os

    try:
        n = int(os.environ.get("OMP_NUM_THREADS", "1"))
    except ValueError:
        n = 1
    if n > 1 and torch.get_num_threads() != n:
        torch.set_num_threads(n)


def _build_slot_map(ds: SRDataset, cfg: Config, legacy_order: list[int]) -> SlotMap:
    """P1 pool sampler (2026-08-29): the train stream's slot->dataset-index
    map. Pool membership comes from the index ``sampling_pool`` column on
    each sample (priority/regular/aux, §10.4); unknown names defensively
    fall into the regular pool. ``legacy_order`` is the pre-sampler stream
    (index/store order) used when sampling is disabled, and kept as-is for
    the val probe's separate contract."""
    members: dict[str, list[int]] = {}
    for i, m in enumerate(ds.samples):
        pool = m.sampling_pool if m.sampling_pool in ("priority", "regular", "aux") else "regular"
        members.setdefault(pool, []).append(i)
    return SlotMap(len(legacy_order), members, cfg, legacy_order, salt=str(ds.global_seed))


def _train_crop_box(
    ds: SRDataset,
    meta: SampleMeta,
    store: LatentStore | None,
    step: int,
    exposure_per_cycle: int,
) -> tuple[int, int]:
    """Train-path crop box (P1-④ dynamic crop, 2026-08-29).

    * store mode: pinned (0,0) — the pre-encoded z_hr was built from
      ``box_seed(sample_id, 0, 0)`` (encode_latents.py), so the crop must
      stay aligned with the store;
    * on-fly mode: dynamic per exposure — z_hr is encoded by the consumer
      from THIS exact crop, so the box follows the §11.5 exposure identity:
      same (sample_id, data_cycle, exposure_index) -> same box, and a
      resume to the same step reproduces the same crop. The crop-box
      stream (``box_seed``) is independent of the degradation seed.

    Val keeps its own pinned contract (``_VAL_CYCLE``/``_VAL_EXPOSURE``) so
    the #3 gate metrics stay comparable across steps."""
    if store is not None:
        return ds.crop(meta, 0, 0)
    data_cycle = step // exposure_per_cycle
    exposure_index = step % exposure_per_cycle
    return ds.crop(meta, data_cycle, exposure_index)


def _pp_fetch(args: tuple[int, int]) -> tuple:
    """Process-worker body: returns (hr | None, lq, z_hr | None, meta, st).

    ``hr`` is None in store mode (the consumer never consumes it there;
    skipping it halves the pipe payload)."""
    ctx = _PRODUCER_CTX
    assert ctx is not None, "producer context unset (pool must fork before use)"
    slot, step = args
    ds: SRDataset = ctx["ds"]
    cfg: Config = ctx["cfg"]
    store: LatentStore | None = ctx["store"]
    j = ctx["slot_map"][slot]  # P1 pool stream (legacy order when disabled)
    meta = ds.samples[j]
    st: dict[str, float] = {}
    hr_full, dec = ds.decode_hr_timed(meta)  # shard/decode stage split
    st["shard"] = dec["shard"]
    st["decode"] = dec["decode"]
    t_c0 = time.perf_counter()
    x, y = _train_crop_box(ds, meta, store, step, int(ctx["exposure_per_cycle"]))
    bucket_hr = int(ctx["bucket_hr"])
    hr_crop = hr_full[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()
    st["crop"] = time.perf_counter() - t_c0
    t_d0 = time.perf_counter()
    lq, _ = degrade_hr(
        hr_crop,
        cfg,
        global_seed=ctx["global_seed"],
        sample_id=meta.sample_id,
        data_cycle=step // int(ctx["exposure_per_cycle"]),
        exposure_index=step % int(ctx["exposure_per_cycle"]),
    )
    st["degradation"] = time.perf_counter() - t_d0
    t_z0 = time.perf_counter()
    z_hr_s = store.read(meta.sample_id) if store is not None else None
    st["z_hr"] = time.perf_counter() - t_z0
    return (None if store is not None else hr_crop, lq, z_hr_s, meta, st)


def _make_process_pool(n_workers: int) -> _MPPool:
    """Fork n_workers producer workers (Linux start method only)."""
    ctx = multiprocessing.get_context("fork")
    return ctx.Pool(processes=n_workers, initializer=_pp_worker_init)


def _fut_done(f: Future | _MPAsyncResult) -> bool:
    """Done-check across both producer backends.

    ``multiprocessing.pool.AsyncResult`` exposes ``ready()``/``get(timeout)``
    — NOT ``concurrent.futures.Future``'s ``done()``/``result()`` — so the
    process backend must be dispatched on its own API (p1formal
    P1-PRODUCER-PORT-V2 verified form; the duck-typed variant crashed the
    process path on the first telemetry poll)."""
    if isinstance(f, _MPAsyncResult):
        return f.ready()
    return cast("Future", f).done()


def _fut_result(f: Future | _MPAsyncResult, timeout: float | None = None) -> Any:
    """Blocking collect across both producer backends.

    AsyncResult raises the worker's exception on ``get()`` exactly like
    ``Future.result()`` does, so the consumer fails loud on task errors."""
    if isinstance(f, _MPAsyncResult):
        return f.get(timeout)
    return cast("Future", f).result(timeout)


# P1-WORKER-RECOVERY ---------------------------------------------------------
# A forked producer worker can die (observed on sakrua10 HCU: SIGSEGV inside
# a CPU op on a task, silent without PYTHONFAULTHANDLER). The lost in-flight
# task's AsyncResult never completes and the consumer would block forever.
# CPython's Pool auto-restarts dead workers (_handle_workers repopulates), so
# a lost task is recovered by resubmitting the same (slot, step) payload:
# _pp_fetch is deterministic (seeded from the exposure index), the restarted
# worker recomputes the identical tensor, and in-place future replacement
# keeps batch order with exactly-once consumption (the dead future is
# dropped; only the resubmitted future is ever collected).
_PP_RECOVER_POLL_S = 0.5  # liveness poll interval while waiting for a batch
_PP_MAX_WORKER_CRASHES = 4  # then abort loudly instead of crash-looping


def _pp_recover_lost_tasks(
    ppool: _MPPool,
    fs: list[Future | _MPAsyncResult],
    ready: deque[list[Future | _MPAsyncResult]],
    inflight: dict[Any, tuple[int, int]],
    crash_state: list[int],
    seen_workers: set,
) -> None:
    """Resubmit in-flight tasks after a worker death (in-place replacement).

    The Pool's maintenance thread reaps dead workers OUT of ``ppool._pool``
    (``_join_exited_workers`` does ``del pool[i]``; ``_repopulate_pool_static``
    appends the replacements), so a dead worker is never observable by a
    liveness scan of the list. Deaths are therefore detected by membership
    diff against ``seen_workers`` (the previously observed set of Process
    objects): anything that disappeared was reaped (P1-WORKER-RECOVERY r2).
    No-op while the set is unchanged. ``crash_state[0]`` accumulates DISTINCT
    worker deaths seen by this rank (a reaped worker is removed from
    ``seen_workers`` after being counted, so one death counts exactly once
    across the poll loop — an unbounded union would re-count the same corpse
    every poll and trip the guard ~2 s after a single death); at
    ``_PP_MAX_WORKER_CRASHES`` the pool is crash-looping and the run is
    aborted (fail loud, not hang)."""
    # ``_pool`` is a CPython/DTK Pool implementation detail (undeclared in
    # typeshed): the list of live worker Process objects — the only reliable
    # liveness signal for the membership-diff detector.
    current = set(cast(Any, ppool)._pool)
    gone = [w for w in seen_workers if w not in current]
    seen_workers.update(current)
    if not gone:
        return
    seen_workers.difference_update(gone)
    crash_state[0] += len(gone)
    codes = ", ".join(str(w.exitcode) for w in gone)
    print(
        f"[latent] producer: {len(gone)} pool worker(s) died "
        f"(exit codes: {codes}); resubmitting in-flight tasks to the "
        f"restarted workers (total crashes: {crash_state[0]}/"
        f"{_PP_MAX_WORKER_CRASHES})",
        flush=True,
    )
    if crash_state[0] >= _PP_MAX_WORKER_CRASHES:
        raise RuntimeError(
            f"producer worker crash-loop: {crash_state[0]} worker deaths "
            f"(last exit codes: {codes}); in-flight tasks are not recoverable. "
            "Relaunch with PYTHONFAULTHANDLER=1 to capture the crashing "
            "worker's stack (last observed crash: SIGSEGV in a CPU op inside "
            "_pp_fetch's degradation path)"
        )
    replaced: dict[Future | _MPAsyncResult, _MPAsyncResult] = {}
    for f, task in list(inflight.items()):
        if _fut_done(f):
            continue
        nf = ppool.apply_async(_pp_fetch, (task,))
        replaced[f] = nf
        inflight[nf] = task
    for f in replaced:
        inflight.pop(f, None)
    for q in ready:
        for i in range(len(q)):
            if q[i] in replaced:
                q[i] = replaced[q[i]]
    fs[:] = [replaced.get(f, f) for f in fs]


_PRE_FORK_POOL: _MPPool | None = None

#: model state keys that mark a full (pixel-stage) checkpoint
_PIXEL_KEY_PREFIX = "pixel_encoder."


def _ckpt_is_full_pixel(path: str | Path, device: torch.device) -> bool:
    """True when the checkpoint's model state carries pixel_encoder.* keys."""
    payload = torch.load(path, map_location=device, weights_only=False)
    return any(k.startswith(_PIXEL_KEY_PREFIX) for k in payload["model"])


def _apply_init_trunk(
    model: nn.Module,
    path: str | Path,
    cfg: Config,
    device: torch.device,
    rank: int,
) -> int:
    """Stage-initialization transition: trunk-only (M4-L0) checkpoint -> full
    AnimeSRModel pixel stage.

    Semantics (P0-2): NEW stage only — trunk weights loaded non-strict (the
    missing keys must be exactly the pixel_encoder.* set), a FRESH optimizer
    (the caller's), stage step = 0, exposure = 0, and
    ``apply_pixel_zero_init()`` re-applied when configured (load_state_dict
    overwrote the pixel weights).  A full pixel checkpoint passed here is a
    stage checkpoint and is rejected: use ``--resume`` instead."""
    payload = torch.load(path, map_location=device, weights_only=False)
    sd = payload["model"]
    if any(k.startswith(_PIXEL_KEY_PREFIX) for k in sd):
        n_pix = sum(1 for k in sd if k.startswith(_PIXEL_KEY_PREFIX))
        raise RuntimeError(
            f"--init-trunk received a FULL pixel checkpoint ({path}: "
            f"{n_pix} pixel_encoder keys). It is a stage checkpoint, not a "
            "trunk-only source — use --resume for same-stage recovery."
        )
    res = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith(_PIXEL_KEY_PREFIX)]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(
            f"--init-trunk mismatch at {path}: non-pixel missing "
            f"{bad_missing[:8]}, unexpected {res.unexpected_keys[:8]}"
        )
    if cfg.model.zero_init_pixel:
        zeroed = apply_pixel_zero_init(cast(AnimeSRModel, model).trunk)
        if rank == 0:
            print(f"[latent] init-trunk: pixel zero-init applied: {zeroed}")
    if rank == 0:
        print(
            f"[latent] init-trunk transition from {path}: trunk weights in, "
            f"{len(res.missing_keys)} pixel_encoder keys absent by design, "
            "fresh optimizer, stage step=0, exposure=0"
        )
    return 0  # fresh stage


def _apply_resume(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    ema: SampleEMA | None,
    path: str | Path,
    device: torch.device,
    rank: int,
    pixel_stage: bool,
) -> tuple[int, dict | None]:
    """Same-stage FULL resume (P0-2): strict model load + optimizer + EMA
    (when present) + step.  NEVER re-applies pixel zero-init and never
    creates a fresh optimizer.  v2 payloads restore the RNG/exposure cursor
    through the returned meta (``{"step", "legacy", "scalars", "exposure",
    "provenance", "rng"}``); v1 legacy payloads load with ``legacy=True``.

    Stage guards:
    * a FULL pixel checkpoint in a trunk-only stage -> error;
    * a trunk-only checkpoint in the pixel stage -> error (direct the
      operator to --init-trunk for the transition).

    Returns ``(start_step, v2_meta)`` — ``v2_meta`` is ``None`` for v1
    legacy files (no RNG/exposure sections to restore)."""
    if _ckpt_is_full_pixel(path, device) != pixel_stage:
        if pixel_stage:
            raise RuntimeError(
                f"--resume in the pixel stage received a TRUNK-ONLY checkpoint "
                f"({path}: no pixel_encoder keys). Trunk weights cannot "
                "reconstruct a trained pixel model — if this starts a NEW "
                "pixel stage, use --init-trunk instead."
            )
        raise RuntimeError(
            f"--resume in the trunk-only stage received a FULL pixel "
            f"checkpoint ({path}: pixel_encoder keys present); the strict "
            "trunk load cannot accept them."
        )
    payload = torch.load(path, map_location=device, weights_only=False)
    has_ema = payload.get("ema") is not None
    meta = load_v2(
        path,
        model,
        opt,
        ema=ema if has_ema else None,
        device=device,
    )
    start_step = int(meta["step"])
    if ema is not None and not has_ema and rank == 0:
        print(
            f"[latent] resume {path}: no EMA section (v1 legacy) — "
            "keeping the fresh EMA (decays in from the live weights)"
        )
    if rank == 0:
        extra = "" if meta["legacy"] else " (v2: RNG/exposure cursor restorable)"
        print(f"[latent] resumed at step {start_step} from {path}{extra}")
    v2_meta = None if meta["legacy"] else meta
    return start_step, v2_meta


def prepare_producer_prefork(
    cfg: Config,
    *,
    index_dir: str | Path,
    webp_dir: str | Path,
    latent_dir: str | Path | None,
    bucket_hr: int,
    rank: int,
) -> None:
    """P1-WEDGE-FIX: build the producer ctx and fork the worker pool BEFORE
    ``dist.init_process_group`` (call from the CLI).

    A forked worker that inherits NCCL/HCU runtime state SIGSEGVs on its
    first heavy CPU op (observed: torch.randn in the degradation path,
    2-rank smoke; the inherited OMP pool is a required factor — OMP=1 runs
    are crash-free).  The dataset/store/order construction is CPU-only, so
    it can run before any accelerator initialization; the forked pool then
    carries a clean (pre-accelerator) address space.  ``run_latent_flow``
    reuses the pool via ``_PRE_FORK_POOL``; its ``_PRODUCER_CTX`` is the
    very dict the workers inherited at fork time."""
    global _PRE_FORK_POOL, _PRODUCER_CTX
    lf = cfg.latent_flow
    onfly = lf.zhr_source == "onfly"
    store: LatentStore | None = None
    sids: list[str] = []
    if not onfly:
        if latent_dir is None:
            raise RuntimeError("zhr_source=store requires latent_dir")
        store = LatentStore(latent_dir, bucket_hr)
        doc = read_index(latent_dir)
        sids = sorted(doc["samples"].keys())
    ds = SRDataset(index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="train")
    # P1-4 (2026-08-29): the clean-score gate is FROZEN + read-only here —
    # the sidecar was precomputed offline; the identical filter on every
    # rank keeps the DDP stream consistent. (The start-up report + rank-0
    # logging live in run_latent_flow, which owns the sidecar read.)
    retained = clean_score_gate_retained(
        [m.sample_id for m in ds.samples], index_dir, cfg.filter.clean_score_min
    )
    if retained is not None:
        n_before = len(ds.samples)
        ds.samples = [m for m in ds.samples if m.sample_id in retained]
        if not ds.samples:
            raise RuntimeError(
                f"clean-score gate (min={cfg.filter.clean_score_min}) removed all "
                f"{n_before} train samples — check the sidecar coverage and threshold"
            )
    if onfly:
        order = list(range(len(ds.samples)))
    else:
        # gate-excluded samples are gone from ds.samples: intersect the
        # store ids so their latent rows are simply unused
        sids = [s for s in sids if s in {m.sample_id for m in ds.samples}]
        sid_to_idx = {m.sample_id: i for i, m in enumerate(ds.samples)}
        missing = [s for s in sids if s not in sid_to_idx]
        if missing:
            raise RuntimeError(
                f"{len(missing)} latent sample ids missing from the train index "
                f"(e.g. {missing[:3]}); rebuild the latent store from this index"
            )
        order = [sid_to_idx[s] for s in sids]
    n = len(order)
    slot_map = _build_slot_map(ds, cfg, order)
    _PRODUCER_CTX = {
        "ds": ds,
        "order": order,
        "n": n,
        "slot_map": slot_map,
        "cfg": cfg,
        "store": store,
        "global_seed": ds.global_seed,
        "bucket_hr": bucket_hr,
        "exposure_per_cycle": _EXPOSURE_PER_CYCLE,
    }
    n_pp = max(1, (lf.prefetch_depth or 1) * lf.batch_size)
    _PRE_FORK_POOL = _make_process_pool(n_pp)
    if rank == 0:
        print(
            f"[latent] producer=process: {n_pp} forked workers BEFORE "
            f"NCCL/HCU init (intra-op re-tuned from OMP_NUM_THREADS)",
            flush=True,
        )


def run_latent_flow(
    cfg: Config,
    *,
    index_dir: str | Path,
    webp_dir: str | Path,
    latent_dir: str | Path | None,
    out_dir: str | Path,
    vae_path: str | None = None,
    bucket_hr: int = 1024,
    rank: int = 0,
    world_size: int = 1,
    start_step: int = 0,
    resume: str | Path | None = None,
    init_trunk: str | Path | None = None,
) -> int:
    """Train (or resume) the M3/M4 latent flow model; returns the final step.

    ``latent_dir`` may be ``None`` for P1 ④ ``zhr_source="onfly"`` (no
    pre-encoded store; z_hr is encoded in the consumer)."""
    lf = cfg.latent_flow
    p1 = cfg.phase1
    # plan §15.1: the exposure budget is a SAMPLE budget (hardware-invariant);
    # one step consumes batch_size * world_size samples across all ranks.
    total = p1.exposure_target // (lf.batch_size * world_size)
    if not (p1.exposure_min <= p1.exposure_target <= p1.exposure_max):
        raise ValueError(
            f"phase1.exposure_target {p1.exposure_target} outside the frozen "
            f"[{p1.exposure_min}, {p1.exposure_max}] window"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dtype = (
        torch.bfloat16
        if (cfg.hardware.dtype == "bf16" and device.type == "cuda")
        else torch.float32
    )

    # data: pre-encoded z_hr store (index order = deterministic stream order)
    # — or P1 ④ on-fly mode: no store, z_hr encoded in the consumer.
    onfly = lf.zhr_source == "onfly"
    store: LatentStore | None = None
    sids: list[str] = []
    if not onfly:
        if latent_dir is None:
            raise ValueError("zhr_source='store' requires latent_dir")
        store = LatentStore(latent_dir, bucket_hr)
        doc = read_index(latent_dir)
        sids = sorted(doc["samples"].keys())

    # P1 ⑤: pin torch CPU intra-op threads. The thread-pool producer runs
    # many workers IN THIS process, each issuing small CPU torch ops
    # (degrade / stack); if every worker inherited the default intra-op
    # pool the cores oversubscribe (workers × N threads) and steal cycles
    # from the HCU. Parallelism comes from the worker count, not from
    # intra-op threads.
    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)

    # P1-4 (2026-08-29): the clean score is a FROZEN offline sidecar
    # (cli/clean_score_precompute). Training is READ-ONLY: no compute, no
    # O_APPEND races. The sidecar feeds (a) the start-up distribution
    # report (rank 0) and (b) the optional clean_score_min gate, which
    # filters ds.samples identically on every rank BEFORE the stream is
    # built. Report-only by default (clean_score_min = -1.0): the user
    # picks the threshold from the report numbers, never an auto-drop.
    clean_scores = (
        CleanScoreCache(index_dir)
        if cfg.filter.clean_score_stage == "lazy" and cfg.filter.clean_score_cache
        else None
    )
    ds = SRDataset(index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="train")
    train_ids = [m.sample_id for m in ds.samples]
    if clean_scores is not None and rank == 0:
        report = build_clean_score_report(
            index_dir, train_ids, cfg.filter.clean_score_candidates
        )
        p = report["percentiles"]
        thr = report["candidate_thresholds"]
        thr_txt = " ".join(
            f"{t}: keep {v['kept']}/excl {v['excluded']}" for t, v in thr.items()
        )
        print(
            f"[latent] clean-score (frozen sidecar, {report['n_covered']}/"
            f"{report['n_requested']} covered, coverage "
            f"{report['coverage']:.1%}): "
            f"p10={p['p10']:.3f} p25={p['p25']:.3f} p50={p['p50']:.3f} "
            f"p75={p['p75']:.3f} p90={p['p90']:.3f} mean={p['mean']:.3f}",
            flush=True,
        )
        if thr_txt:
            print(f"[latent] clean-score candidate thresholds -> {thr_txt}", flush=True)
        if cfg.filter.clean_score_min >= 0:
            print(
                f"[latent] clean-score GATE ACTIVE: min={cfg.filter.clean_score_min} "
                f"(samples below, or without a sidecar row, are excluded)",
                flush=True,
            )
        else:
            print(
                "[latent] clean-score gate DISABLED (report-only; set "
                "[filter] clean_score_min to enable)",
                flush=True,
            )
        report_path = Path(out_dir) / "clean-score-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[latent] clean-score report -> {report_path}", flush=True)

    # the gate filter (identical on all ranks: frozen sidecar + config)
    retained = (
        clean_score_gate_retained(train_ids, index_dir, cfg.filter.clean_score_min)
        if clean_scores is not None
        else None
    )
    if retained is not None:
        n_before = len(ds.samples)
        ds.samples = [m for m in ds.samples if m.sample_id in retained]
        if not ds.samples:
            raise RuntimeError(
                f"clean-score gate (min={cfg.filter.clean_score_min}) removed all "
                f"{n_before} train samples — check the sidecar coverage and threshold"
            )
        if rank == 0:
            print(
                f"[latent] clean-score gate: {n_before - len(ds.samples)}/{n_before} "
                f"samples excluded, {len(ds.samples)} remain",
                flush=True,
            )

    if onfly:
        # P1 ④: no store — the stream is the train-index order (deterministic)
        order = list(range(len(ds.samples)))
        n = len(order)
        if rank == 0:
            print(
                f"[latent] zhr_source=onfly: {n} train crops (bucket {bucket_hr}), "
                f"z_hr encoded on the fly by the frozen VAE (no store)",
                flush=True,
            )
    else:
        # gate-excluded samples are gone from ds.samples: intersect the
        # store ids so their latent rows are simply unused
        sids = [s for s in sids if s in {m.sample_id for m in ds.samples}]
        sid_to_idx = {m.sample_id: i for i, m in enumerate(ds.samples)}
        missing = [s for s in sids if s not in sid_to_idx]
        if missing:
            raise RuntimeError(
                f"{len(missing)} latent sample ids missing from the train index "
                f"(e.g. {missing[:3]}); rebuild the latent store from this index"
            )
        order = [sid_to_idx[s] for s in sids]
        n = len(order)

    # P1 pool sampler (2026-08-29): the TRAIN stream is a per-cycle
    # pool-mixed permutation (config [sampling]); `order` above is kept as
    # the legacy stream for the val probe's separate contract. When
    # sampling is disabled the slot map degenerates to `order[slot % n]`.
    slot_map = _build_slot_map(ds, cfg, order)
    if rank == 0:
        rep = slot_map.pool_report()
        if slot_map.enabled:
            print(
                f"[latent] pool sampler ON: per-cycle {n} slots = "
                f"priority {rep.get('priority', 0)} / regular "
                f"{rep.get('regular', 0)} / aux {rep.get('aux', 0)} "
                f"(targets core/regular/aux = "
                f"{cfg.sampling.core_fraction:g}/"
                f"{cfg.sampling.regular_fraction:g}/"
                f"{cfg.sampling.aux_fraction:g}, aux cap "
                f"{cfg.filter.aux_max_fraction:g})",
                flush=True,
            )
        else:
            print(
                "[latent] pool sampler OFF: legacy stream "
                "(index/store order, straight read)",
                flush=True,
            )

    # P1 ①: held-out validation split. Val samples have no LatentStore rows
    # (the store covers train crops), so _validate_heldout encodes z_hr on
    # the fly with the frozen VAE. 0 disables the held-out probe entirely;
    # a missing/empty validation split soft-disables it (logged, not fatal).
    val_ds: SRDataset | None = None
    if lf.val_heldout_samples > 0:
        try:
            val_ds = SRDataset(
                index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="validation",
            )
        except (RuntimeError, ValueError):
            val_ds = None
        if val_ds is not None and rank == 0:
            print(
                f"[latent] held-out val: {len(val_ds.samples)} validation-split "
                f"samples for bucket {bucket_hr} "
                f"(n={min(lf.val_heldout_samples, len(val_ds.samples))} per probe)",
                flush=True,
            )
        elif val_ds is None and lf.val_heldout_samples > 0 and rank == 0:
            print(
                "[latent] held-out val disabled: no validation-split samples "
                "eligible for this bucket",
                flush=True,
            )

    # P1: the process-pool producer must fork BEFORE any HCU context exists
    # (the workers are CPU-only and never touch the HCU; forking after the
    # device init would inherit the accelerator runtime state). Thread mode
    # creates its pool at the classic spot below, after the model is on
    # device — threads inheriting the HCU context is harmless.
    pool: ThreadPoolExecutor | _MPPool | None = None
    if lf.producer == "process":
        global _PRODUCER_CTX  # _PRE_FORK_POOL is only read here
        if _PRE_FORK_POOL is not None:
            # P1-WEDGE-FIX: the pool (and its inherited ctx) was created in
            # prepare_producer_prefork() BEFORE dist.init_process_group, so
            # the workers carry no NCCL/HCU runtime state.
            pool = _PRE_FORK_POOL
            if rank == 0:
                print(
                    "[latent] producer=process: reusing pre-NCCL pool "
                    "(fork-before-init_process_group; ctx inherited at fork)",
                    flush=True,
                )
        else:
            _PRODUCER_CTX = {
                "ds": ds,
                "order": order,
                "n": n,
                "slot_map": slot_map,
                "cfg": cfg,
                "store": store,
                "global_seed": ds.global_seed,
                "bucket_hr": bucket_hr,
                "exposure_per_cycle": _EXPOSURE_PER_CYCLE,
            }
            n_pp = max(1, (lf.prefetch_depth or 1) * lf.batch_size)
            pool = _make_process_pool(n_pp)
            if rank == 0:
                print(
                    f"[latent] producer=process: {n_pp} forked workers "
                    f"(intra-op re-tuned from OMP_NUM_THREADS)",
                    flush=True,
                )

    vae = load_frozen_vae(vae_path or cfg.vae.path, device, dtype=dtype)
    if lf.pixel_features:
        model = AnimeSRModel(
            cfg.model, zero_init_pixel=cfg.model.zero_init_pixel
        ).to(device, dtype=dtype)
    else:
        model = UFlowSR(cfg.model.uflow, cfg.model.output_head).to(device, dtype=dtype)
    n_params = count_parameters(model)
    opt = _optimizer_for(cfg, model)
    # P0-2: --init-trunk (stage transition) and --resume (same-stage
    # recovery) are fully separate paths and mutually exclusive.
    if init_trunk is not None and resume is not None:
        raise ValueError(
            "--init-trunk (stage transition) and --resume (same-stage "
            "recovery) are mutually exclusive"
        )
    if init_trunk is not None:
        if not lf.pixel_features:
            raise ValueError(
                "--init-trunk is the trunk-only -> pixel-stage transition and "
                "requires [latent_flow].pixel_features = true"
            )
        start_step = _apply_init_trunk(model, init_trunk, cfg, device, rank)
    elif resume is not None:
        # Same-stage full resume. EMA is wired in by the P2 section below;
        # pass it once created (the trainer keeps ``self.ema`` there).
        start_step, _v2_meta = _apply_resume(
            model, opt, None, Path(resume), device, rank, lf.pixel_features
        )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[rank % max(1, torch.cuda.device_count())]
            if torch.cuda.is_available()
            else None,
        )
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (cfg.hardware.dtype == "bf16" and device.type == "cuda")
        else (lambda: nullcontext())
    )

    bs = lf.batch_size
    if rank == 0:
        print(
            f"[latent] {n_params / 1e6:.2f}M params, {n} crops (bucket {bucket_hr}), "
            f"bs={bs} x world={world_size}, steps {start_step}..{total} "
            f"({p1.exposure_target} samples), zhr={lf.zhr_source}, "
            f"producer={lf.producer}, prefetch_depth={lf.prefetch_depth}, "
            f"device={device}, dtype={dtype}"
        )

    # ------------------------------------------------------------------
    # CPU producer / accelerator consumer (M1 #8 data-wait gate): a
    # producer (thread pool by default, or a forked process pool when
    # producer="process") keeps ``prefetch_depth`` step-batches ready ahead
    # of the consumer (2 = double-buffered M3 default, 4 = quad buffer).
    # Every fetch is a pure function of (step, slot) -> bit-exact §11.5
    # stream in either backend; the producer records per-stage wall-times
    # and the loop exposes producer/consumer throughput + ready-queue
    # occupancy.
    # ------------------------------------------------------------------

    @dataclass
    class _Prepared:
        hr: torch.Tensor | None  # None: store mode w/ process pool (consumer never reads it)
        lq: torch.Tensor
        z_hr: torch.Tensor | None  # None in P1 ④ on-fly mode (consumer encodes)
        meta: SampleMeta
        stages: dict[str, float]

    def _fetch(slot: int, step: int) -> _Prepared:
        j = slot_map[slot]  # P1 pool stream (legacy order[slot % n] when disabled)
        meta = ds.samples[j]
        st: dict[str, float] = {}
        hr_full, dec = ds.decode_hr_timed(meta)  # shard/decode stage split
        st["shard"] = dec["shard"]
        st["decode"] = dec["decode"]
        t_c0 = time.perf_counter()
        x, y = _train_crop_box(ds, meta, store, step, _EXPOSURE_PER_CYCLE)
        hr_crop = hr_full[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()
        st["crop"] = time.perf_counter() - t_c0
        t_d0 = time.perf_counter()
        lq, _ = degrade_hr(
            hr_crop,
            cfg,
            global_seed=ds.global_seed,
            sample_id=meta.sample_id,
            data_cycle=step // _EXPOSURE_PER_CYCLE,
            exposure_index=step % _EXPOSURE_PER_CYCLE,
        )
        st["degradation"] = time.perf_counter() - t_d0
        t_z0 = time.perf_counter()
        if store is not None:
            z_hr_s = store.read(meta.sample_id)  # fp16 CPU; read is thread-safe
        else:
            z_hr_s = None  # P1 ④ on-fly: the consumer encodes z_hr
        st["z_hr"] = time.perf_counter() - t_z0
        return _Prepared(hr=hr_crop, lq=lq, z_hr=z_hr_s, meta=meta, stages=st)

    depth = max(0, lf.prefetch_depth)
    if lf.producer == "thread":
        pool = ThreadPoolExecutor(
            max_workers=max(1, (depth or 1) * bs), thread_name_prefix="lfetch"
        )
    assert pool is not None, "producer pool must be created (thread here, process above)"

    _is_proc = lf.producer == "process"

    def _submit_batch(step: int) -> list[Future | _MPAsyncResult]:
        pairs = [
            (latent_sample_index(step, rank, i, bs, world_size, n), step)
            for i in range(bs)
        ]
        if _is_proc:
            ppool = cast("_MPPool", pool)
            # apply_async(func, args): args is the tuple of positional args,
            # so the single-tuple payload ((slot, st),) binds to
            # _pp_fetch(args) exactly once (p1formal P1-PRODUCER-PORT-V2
            # verified form; a flattened (slot, st) raises TypeError in the
            # worker and the batch is lost).
            futs: list[Future | _MPAsyncResult] = [
                ppool.apply_async(_pp_fetch, ((slot, st),)) for slot, st in pairs
            ]
            for f, task in zip(futs, pairs):
                inflight[f] = task  # P1-WORKER-RECOVERY: track for resubmit
            return futs
        tpool = cast("ThreadPoolExecutor", pool)
        return [tpool.submit(_fetch, slot, st) for slot, st in pairs]

    # ready queue: the producer keeps `depth` step-batches queued ahead of
    # the consumer (prefilled at start; refilled as each batch is consumed).
    ready: deque[list[Future | _MPAsyncResult]] = deque()
    inflight: dict[Any, tuple[int, int]] = {}  # P1-WORKER-RECOVERY: future -> (slot, step)
    crash_state = [0]  # P1-WORKER-RECOVERY: worker deaths seen by this rank
    seen_workers: set = set()  # P1-WORKER-RECOVERY r2: live worker Process objects
    if _is_proc:
        seen_workers.update(cast(Any, pool)._pool)  # Pool impl detail (typeshed-undeclared)
    for k in range(depth):
        if start_step + k < total:
            ready.append(_submit_batch(start_step + k))

    t0 = time.time()
    t_data_cum = 0.0  # M1 #8 telemetry: data-wait fraction of step time
    t_comp_cum = 0.0
    stage_cum: dict[str, float] = {
        s: 0.0
        for s in ("shard", "decode", "crop", "degradation", "z_hr", "zhr_enc", "stack", "H2D")
    }
    n_produced = 0
    ready_occ_sum = 0  # per-step count of queued (non-front) batches already fetched
    n_wait = 0  # steps where the consumer blocked on the front batch

    def data_snapshot(done_steps: int, elapsed: float) -> str:
        """M1 #8 producer/consumer snapshot, printed at val milestones.

        ``prod_margin`` = (consumer time per batch) / (producer time to make
        one batch serially): >= 1.25x means even a serial producer keeps
        ahead of the consumer (with ``depth*bs`` parallel workers the real
        margin is larger). ``ready_avg`` is the mean number of queued
        batches already fully fetched (0 = consumer starved)."""
        denom = max(1e-9, t_data_cum + t_comp_cum)
        wait_pct = 100.0 * t_data_cum / denom
        # per-rank consumption (the counters are this rank's; the job rate is
        # world_size x this, all ranks run the symmetric loop)
        cons_img_s = done_steps * bs / max(1e-9, elapsed)
        prod_sample = (
            sum(stage_cum[s] for s in ("shard", "decode", "crop", "degradation", "z_hr"))
            / max(1, n_produced)
        )
        comp_batch = t_comp_cum / max(1, done_steps)
        margin = comp_batch / max(1e-9, prod_sample * bs) if done_steps >= 50 else -1.0
        occ = ready_occ_sum / max(1, done_steps)
        empty = 100.0 * n_wait / max(1, done_steps)
        ms = {s: 1000.0 * stage_cum[s] / max(1, n_produced) for s in stage_cum}
        return (
            f"data_wait={wait_pct:.1f}% prod_margin={margin:.2f}x "
            f"consumer={cons_img_s:.1f}img/s ready_avg={occ:.2f}/{depth} "
            f"starve={empty:.1f}% stage_ms="
            f"shard:{ms['shard']:.0f} decode:{ms['decode']:.0f} crop:{ms['crop']:.1f} "
            f"degrad:{ms['degradation']:.0f} z_hr:{ms['z_hr']:.0f} "
            f"zhr_enc:{ms['zhr_enc']:.1f} "
            f"stack:{ms['stack']:.1f} h2d:{ms['H2D']:.1f}"
        )

    for step in range(start_step, total):
        if depth > 0:
            futs = ready.popleft()
            ns = step + depth
            if ns < total:
                ready.append(_submit_batch(ns))
        else:
            futs = _submit_batch(step)  # sync mode: no look-ahead

        # ready-queue telemetry, before the consumer waits on the front batch
        if depth > 0:
            ready_occ_sum += sum(1 for q in ready if all(_fut_done(f) for f in q))
            if not all(_fut_done(f) for f in futs):
                n_wait += 1

        tf = time.perf_counter()
        if _is_proc:
            # P1-WORKER-RECOVERY: poll with worker-liveness checks so a dead
            # worker (lost in-flight task) is detected and its task
            # resubmitted instead of blocking forever.
            while not all(_fut_done(f) for f in futs):
                _pp_recover_lost_tasks(
                    cast("_MPPool", pool), futs, ready, inflight, crash_state,
                    seen_workers,
                )
                time.sleep(_PP_RECOVER_POLL_S)
            # process pool hands back the (hr, lq, z_hr, meta, stages) tuple;
            # rewrap into _Prepared (hr is None in store mode)
            prepared = [
                _Prepared(hr=r[0], lq=r[1], z_hr=r[2], meta=r[3], stages=r[4])
                for r in (_fut_result(f) for f in futs)
            ]
            for f in futs:
                inflight.pop(f, None)
        else:
            prepared = [_fut_result(f) for f in futs]
        n_produced += bs
        for p in prepared:
            for k, v in p.stages.items():
                stage_cum[k] += v
        # store mode: hr_crop is consumed by the producer (degradation draw)
        # and is not needed by the compute phase — stack/H2D only lq + z_hr.
        # P1 ④ on-fly: the hr crop feeds the consumer-side VAE encode.
        t_s0 = time.perf_counter()
        lq = torch.stack([p.lq for p in prepared])
        # p.z_hr/p.hr are `Tensor | None` (the other mode's field), narrowed
        # to Tensor by the store/onfly branch above.
        if store is not None:
            z_hr = torch.stack([cast(torch.Tensor, p.z_hr) for p in prepared])
        else:
            z_hr = None
        hr_b = (
            None
            if store is not None
            else torch.stack([cast(torch.Tensor, p.hr) for p in prepared])
        )
        stage_cum["stack"] += time.perf_counter() - t_s0
        t_h0 = time.perf_counter()
        lq = lq.to(device, non_blocking=True)
        if z_hr is not None:
            z_hr = z_hr.to(device, non_blocking=True)
        if hr_b is not None:
            hr_b = hr_b.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )  # attribute the H2D transfer to the data phase
        stage_cum["H2D"] += time.perf_counter() - t_h0
        t_data_cum += time.perf_counter() - tf

        tc = time.perf_counter()
        with autocast():
            # plan §4.3 anchor: z_lr = E_Mage(Bicubic4x(LQ)), frozen VAE
            lq_up = F.interpolate(
                lq.float(), size=(bucket_hr, bucket_hr), mode="bicubic"
            )
            z_lr = vae.encode(lq_up.to(dtype))
            if z_hr is not None:
                z_hr = z_hr.to(dtype)
            else:
                # P1 ④: on-fly z_hr — frozen VAE encodes the HR crop here
                # (inputs are requires_grad-free leaves; §4.3).
                assert hr_b is not None  # set in the stack block above
                t_e0 = time.perf_counter()
                z_hr = vae.encode(hr_b.to(dtype))
                stage_cum["zhr_enc"] += time.perf_counter() - t_e0
            rt, v_star, sigma, _t = build_flow_targets(z_hr, z_lr, cfg, device=device)
            if lf.pixel_features:
                # Phase I-P: the pixel path consumes the degraded LQ directly
                # (the full model computes the encoder features internally).
                v_hat = model(rt, z_lr, lq, _t, sigma)  # DDP syncs gradients on the backward
            else:
                v_hat = model(rt, z_lr, _t, sigma)  # DDP syncs gradients on the backward
            loss = F.mse_loss(v_hat.float(), v_star.float())
        opt.zero_grad()
        loss.backward()
        lr = _cosine_lr(step, total, cfg.optimizer.lr, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        if cfg.gradient.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient.clip_norm)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_comp_cum += time.perf_counter() - tc

        if (step + 1) % lf.val_every_steps == 0:
            _validate_latent(
                model,
                vae,
                ds,
                order,
                n,
                store,
                cfg,
                device,
                rank,
                step + 1,
                bucket_hr,
                autocast,
            )
            # M1 #8 / revised-#3 milestone: data pipeline + producer/consumer
            # telemetry alongside the model metrics above.
            if rank == 0:
                done = step + 1 - start_step
                print(
                    f"[latent] val-data step {step + 1}: "
                    f"{data_snapshot(done, time.time() - t0)}",
                    flush=True,
                )
        # P1 ①: held-out val on its own cadence AND at run end (dedup)
        if val_ds is not None and (
            (step + 1 == total)
            or ((step + 1) % lf.val_heldout_every_steps == 0 and step + 1 < total)
        ):
            _validate_heldout(
                model,
                vae,
                val_ds,
                cfg,
                device,
                rank,
                step + 1,
                bucket_hr,
                autocast,
            )
        if rank == 0 and (step + 1) % lf.save_every_steps == 0:
            _save_ckpt(out / f"step-{step + 1:07d}.pt", step + 1, model, opt)
        if rank == 0 and ((step + 1) % 50 == 0 or step + 1 == total):
            done = step + 1 - start_step
            wait_pct = (
                100.0 * t_data_cum / max(1e-9, t_data_cum + t_comp_cum)
                if done >= 50
                else -1.0
            )
            extra = f" data_wait={wait_pct:.1f}%" if done >= 50 else ""
            print(
                f"[latent] step {step + 1}/{total} loss={loss.item():.4f} lr={lr:.2e} "
                f"({(step + 1 - start_step) / max(1e-3, time.time() - t0):.1f} it/s){extra}",
                flush=True,
            )

    if _is_proc:
        ppool = cast("_MPPool", pool)
        ppool.close()
        ppool.join()  # mirrors pool.shutdown(wait=True): wait for the drain
    else:
        tpool = cast("ThreadPoolExecutor", pool)
        tpool.shutdown(wait=True)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        _save_ckpt(out / "latest.pt", total, model, opt)
        done = total - start_step
        elapsed = max(1e-9, time.time() - t0)
        prod_sample = (
            sum(stage_cum[s] for s in ("shard", "decode", "crop", "degradation", "z_hr"))
            / max(1, n_produced)
        )
        comp_batch = t_comp_cum / max(1, done)
        meta = {
            "iterations": total,
            "bucket_hr": bucket_hr,
            "n_crops": n,
            "n_params_m": round(n_params / 1e6, 2),
            "batch_size": bs,
            "producer": lf.producer,
            "prefetch_depth": lf.prefetch_depth,
            "exposure_target_samples": p1.exposure_target,
            "zhr_source": lf.zhr_source,
            "data_wait_pct": 100.0 * t_data_cum / max(1e-9, t_data_cum + t_comp_cum),
            "producer_margin_x": round(
                comp_batch / max(1e-9, prod_sample * bs), 3
            ),
            "consumer_img_s_per_rank": round(done * bs / elapsed, 3),
            "ready_occ_avg": round(ready_occ_sum / max(1, done), 3),
            "queue_starve_pct": round(100.0 * n_wait / max(1, done), 3),
            "stage_ms_per_sample": {
                s: round(1000.0 * stage_cum[s] / max(1, n_produced), 3)
                for s in stage_cum
            },
        }
        (out / "train-meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[latent] done: {out / 'latest.pt'}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return total
