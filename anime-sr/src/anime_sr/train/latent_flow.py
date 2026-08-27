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
CPU thread-pool producer keeps ready ahead of the accelerator consumer
(2 = double-buffered, the M3 default; 4 = quad buffer, the M1 #8
data-wait fix for Phase I; 0 = synchronous, M2-style canary). Every fetch
is a pure function of ``(step, slot)``, so the §11.5 stream stays
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
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from anime_sr.config.schema import Config
from anime_sr.data.degradation import degrade_hr
from anime_sr.data.latent_store import LatentStore, read_index
from anime_sr.data.pipeline import _EXPOSURE_PER_CYCLE, SampleMeta, SRDataset
from anime_sr.flow.path import (
    interpolate,
    sample_sigma,
    sample_source_noise,
    target_velocity,
)
from anime_sr.flow.sampling import FlowSampler
from anime_sr.flow.solver import euler_trajectory, heun_trajectory
from anime_sr.model.uflow import UFlowSR, count_parameters
from anime_sr.train.pixel_baseline import (
    _cosine_lr,
    _load_ckpt,
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
) -> dict[str, float]:
    """Endpoint consistency of the learned field (revised plan §13 #3).

    With r0 = 0 (Faithful sigma=0) the exact path is r_t = t*delta, so the
    endpoint estimate is ``delta_hat_t = r_t + (1-t) v_theta(r_t, t)`` and
    the endpoint L1 ``|delta_hat_t - delta|`` is reported at t in
    {0, .25, .5, .75} (at t=0 it equals the 1-step L1; for an exact field
    it goes to 0 as t -> 1). Deterministic: fixed t grid, sigma=0.
    """
    b = z_hr.shape[0]
    delta = z_hr - z_lr
    sigma = torch.zeros(b, device=device)
    out: dict[str, float] = {}
    with torch.no_grad(), autocast():
        for t in _ENDPOINT_TS:
            r_t = t * delta
            t_vec = torch.full((b,), t, device=device, dtype=torch.float32)
            v_t = mod(r_t, z_lr, t_vec, sigma)
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
    store: LatentStore,
    cfg: Config,
    device: torch.device,
    rank: int,
    step: int,
    bucket_hr: int,
    autocast: Callable[[], Any],
) -> None:
    """Quantitative M3 checklist subset on a fixed val slice (rank 0 only)."""
    if rank != 0:
        return
    lf = cfg.latent_flow
    mod = _unwrap(model)
    was_training = mod.training
    mod.eval()
    sampler = FlowSampler(_LatentVelocity(cast(UFlowSR, mod)))
    g = torch.Generator(device=str(device)).manual_seed(0x5EED ^ (step % (2**31)))
    n_val = min(lf.val_samples, n)
    js = [order[k % n] for k in range(n_val)]
    z_hrs: list[torch.Tensor] = []
    z_lrs: list[torch.Tensor] = []
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
            z_hr = store.read(meta.sample_id).to(vae.dtype)
            z_hrs.append(z_hr)
            z_lrs.append(z_lr)
        z_hr_b = torch.stack(z_hrs).to(device)
        z_lr_b = torch.stack(z_lrs).to(device)
        # Run the model forwards under the same autocast as the train loop:
        # build_flow_targets emits fp32 ``t``/``rt`` (the §5.3 uniform-t draw
        # is fp32) which would otherwise hit the bf16 trunk weights raw.
        with autocast():
            z1 = sampler.one_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
            z4 = sampler.four_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
        m1 = latent_val_metrics(z_hr_b, z_lr_b, z1)
        m4 = latent_val_metrics(z_hr_b, z_lr_b, z4)
        # flow-direction probe at a random t (reproducible per step)
        with autocast():
            rt, v_star, sigma, t = build_flow_targets(
                z_hr_b, z_lr_b, cfg, generator=g, device=device
            )
            v_hat = mod(rt, z_lr_b, t, sigma)
        cos_v = velocity_cosine(v_hat, v_star)
        # Revised M3 #3: 4-step stability is judged by ratio + trajectory
        # behavior (all deterministic: fixed val slice, sigma=0, fixed t grid;
        # the per-step seeded draw above only feeds the cos_v probe).
        ep = endpoint_consistency(mod, z_hr_b, z_lr_b, autocast, device)
        d4 = trajectory_deviation(
            mod, z_hr_b, z_lr_b, solver="heun", n_steps=4, autocast=autocast, device=device
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


def run_latent_flow(
    cfg: Config,
    *,
    index_dir: str | Path,
    webp_dir: str | Path,
    latent_dir: str | Path,
    out_dir: str | Path,
    vae_path: str | None = None,
    bucket_hr: int = 1024,
    rank: int = 0,
    world_size: int = 1,
    start_step: int = 0,
    resume: str | Path | None = None,
) -> int:
    """Train (or resume) the M3/M4 latent flow model; returns the final step."""
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
    store = LatentStore(latent_dir, bucket_hr)
    doc = read_index(latent_dir)
    sids = sorted(doc["samples"].keys())
    ds = SRDataset(index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="train")
    sid_to_idx = {m.sample_id: i for i, m in enumerate(ds.samples)}
    missing = [s for s in sids if s not in sid_to_idx]
    if missing:
        raise RuntimeError(
            f"{len(missing)} latent sample ids missing from the train index "
            f"(e.g. {missing[:3]}); rebuild the latent store from this index"
        )
    order = [sid_to_idx[s] for s in sids]
    n = len(order)

    vae = load_frozen_vae(vae_path or cfg.vae.path, device, dtype=dtype)
    model = UFlowSR(cfg.model.uflow, cfg.model.output_head).to(device, dtype=dtype)
    n_params = count_parameters(model)
    opt = _optimizer_for(cfg, model)
    if resume is not None:
        start_step = _load_ckpt(Path(resume), model, opt, device)
        if rank == 0:
            print(f"[latent] resumed at step {start_step} from {resume}")
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
            f"({p1.exposure_target} samples), "
            f"prefetch_depth={lf.prefetch_depth}, device={device}, dtype={dtype}"
        )

    # ------------------------------------------------------------------
    # CPU producer / accelerator consumer (M1 #8 data-wait gate): a
    # thread-pool producer keeps ``prefetch_depth`` step-batches ready ahead
    # of the consumer (2 = double-buffered M3 default, 4 = quad buffer).
    # Every fetch is a pure function of (step, slot) -> bit-exact §11.5
    # stream; the producer records per-stage wall-times and the loop
    # exposes producer/consumer throughput + ready-queue occupancy.
    # ------------------------------------------------------------------

    @dataclass
    class _Prepared:
        hr: torch.Tensor
        lq: torch.Tensor
        z_hr: torch.Tensor
        meta: SampleMeta
        stages: dict[str, float]

    def _fetch(slot: int, step: int) -> _Prepared:
        j = order[slot % n]
        meta = ds.samples[j]
        st: dict[str, float] = {}
        hr_full, dec = ds.decode_hr_timed(meta)  # shard/decode stage split
        st["shard"] = dec["shard"]
        st["decode"] = dec["decode"]
        t_c0 = time.perf_counter()
        x, y = ds.crop(meta, 0, 0)  # pinned (0,0) box — matches the pre-encoded z_hr
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
        z_hr_s = store.read(meta.sample_id)  # fp16 CPU; read is thread-safe
        st["z_hr"] = time.perf_counter() - t_z0
        return _Prepared(hr=hr_crop, lq=lq, z_hr=z_hr_s, meta=meta, stages=st)

    depth = max(0, lf.prefetch_depth)
    pool = ThreadPoolExecutor(
        max_workers=max(1, (depth or 1) * bs), thread_name_prefix="lfetch"
    )

    def _submit_batch(step: int) -> list[Future]:
        return [
            pool.submit(
                _fetch, latent_sample_index(step, rank, i, bs, world_size, n), step
            )
            for i in range(bs)
        ]

    # ready queue: the producer keeps `depth` step-batches queued ahead of
    # the consumer (prefilled at start; refilled as each batch is consumed).
    ready: deque[list[Future]] = deque()
    for k in range(depth):
        if start_step + k < total:
            ready.append(_submit_batch(start_step + k))

    t0 = time.time()
    t_data_cum = 0.0  # M1 #8 telemetry: data-wait fraction of step time
    t_comp_cum = 0.0
    stage_cum: dict[str, float] = {
        s: 0.0 for s in ("shard", "decode", "crop", "degradation", "z_hr", "stack", "H2D")
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
            ready_occ_sum += sum(1 for q in ready if all(f.done() for f in q))
            if not all(f.done() for f in futs):
                n_wait += 1

        tf = time.perf_counter()
        prepared = [f.result() for f in futs]
        n_produced += bs
        for p in prepared:
            for k, v in p.stages.items():
                stage_cum[k] += v
        # hr_crop is consumed by the producer (degradation draw) and is not
        # needed by the compute phase — stack/H2D only lq + z_hr.
        t_s0 = time.perf_counter()
        lq = torch.stack([p.lq for p in prepared])
        z_hr = torch.stack([p.z_hr for p in prepared])
        stage_cum["stack"] += time.perf_counter() - t_s0
        t_h0 = time.perf_counter()
        lq = lq.to(device, non_blocking=True)
        z_hr = z_hr.to(device, non_blocking=True)
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
            z_hr = z_hr.to(dtype)
            rt, v_star, sigma, _t = build_flow_targets(z_hr, z_lr, cfg, device=device)
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

    pool.shutdown(wait=True)
    _save_ckpt(out / "latest.pt", total, model, opt)
    if rank == 0:
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
            "prefetch_depth": lf.prefetch_depth,
            "exposure_target_samples": p1.exposure_target,
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
