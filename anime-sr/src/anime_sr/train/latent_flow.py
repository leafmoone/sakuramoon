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

Prefetch: with ``[latent_flow].prefetch`` a double-buffered CPU thread
pool fetches the next step's decode+crop+degrade batch while the current
step computes on the accelerator (M1 #8 data-wait gate); the stream
stays bit-exact because every fetch is a pure function of
``(step, slot)``. With ``prefetch = false`` the same calls run
synchronously (M2-style) for small canary runs.

Flow math (plan §5, ``anime_sr.flow``):

    delta = z_hr - z_lr ; r0 = sigma * eps ; rt = (1-t) r0 + t delta
    v*    = delta - r0   ; loss = MSE(v_hat, v*)      (v-prediction)

M3 checklist mapping (plan §13, "M3 未通过，不启动正式模型"):

1. flow direction correct      -> ``cos_v`` rises above ~0.5 in validation
2. 1-step moves toward HR      -> ``toward_frac_1step`` > 0.5 and rising
3. 4-step not worse than 1-step-> ``l1_4step`` <= ``l1_1step`` once trained
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
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import cast

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
            lq_up = F.interpolate(lq, size=(bucket_hr, bucket_hr), mode="bicubic")
            z_lr = vae.encode(lq_up.to(vae.dtype))
            z_hr = store.read(meta.sample_id).to(vae.dtype)
            z_hrs.append(z_hr)
            z_lrs.append(z_lr)
        z_hr_b = torch.stack(z_hrs).to(device)
        z_lr_b = torch.stack(z_lrs).to(device)
        z1 = sampler.one_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
        z4 = sampler.four_step(z_lr_b, z_lr_b, sigma=0.0, generator=g)
        m1 = latent_val_metrics(z_hr_b, z_lr_b, z1)
        m4 = latent_val_metrics(z_hr_b, z_lr_b, z4)
        # flow-direction probe at a random t (reproducible per step)
        rt, v_star, sigma, t = build_flow_targets(
            z_hr_b, z_lr_b, cfg, generator=g, device=device
        )
        v_hat = mod(rt, z_lr_b, t, sigma)
        cos_v = velocity_cosine(v_hat, v_star)
    if was_training:
        mod.train()
    print(
        f"[latent] val step {step}: l1_1={m1['l1']:.4f} l1_4={m4['l1']:.4f} "
        f"l1_anchor={m1['l1_anchor']:.4f} toward_1={m1['toward_frac']:.2f} "
        f"toward_4={m4['toward_frac']:.2f} cos_v={cos_v:.3f} (n={n_val})",
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
    total = p1.exposure_target
    if not (p1.exposure_min <= total <= p1.exposure_max):
        raise ValueError(
            f"phase1.exposure_target {total} outside the frozen "
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
            f"bs={bs} x world={world_size}, steps {start_step}..{total}, "
            f"prefetch={lf.prefetch}, device={device}, dtype={dtype}"
        )

    # ------------------------------------------------------------------
    # double-buffered CPU prefetch (decode + (0,0) crop + degradation draw)
    # ------------------------------------------------------------------
    def _fetch(slot: int, step: int) -> tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        j = order[slot % n]
        meta = ds.samples[j]
        hr_full = ds.decode_hr(meta)
        x, y = ds.crop(meta, 0, 0)  # pinned (0,0) box — matches the pre-encoded z_hr
        hr_crop = hr_full[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()
        lq, _ = degrade_hr(
            hr_crop,
            cfg,
            global_seed=ds.global_seed,
            sample_id=meta.sample_id,
            data_cycle=step // _EXPOSURE_PER_CYCLE,
            exposure_index=step % _EXPOSURE_PER_CYCLE,
        )
        return hr_crop, lq, meta

    pool = ThreadPoolExecutor(max_workers=max(1, bs), thread_name_prefix="lfetch")

    def _submit(step: int) -> list:
        return [
            pool.submit(
                _fetch, latent_sample_index(step, rank, i, bs, world_size, n), step
            )
            for i in range(bs)
        ]

    cur = _submit(start_step) if start_step < total else []
    t0 = time.time()
    t_data_cum = 0.0  # M1 #8 telemetry: data-wait fraction of step time
    t_comp_cum = 0.0
    for step in range(start_step, total):
        futs = cur
        if lf.prefetch and step + 1 < total:
            cur = _submit(step + 1)  # next batch fetches while this step computes

        tf = time.time()
        prepared = [f.result() for f in futs]
        z_hr = torch.stack([store.read(m.sample_id) for _, _, m in prepared]).to(
            device, non_blocking=True
        )
        lq = torch.stack([p[1] for p in prepared]).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )  # attribute the H2D transfer to the data phase
        t_data_cum += time.time() - tf
        if not lf.prefetch and step + 1 < total:
            cur = _submit(step + 1)  # sync mode: next fetch starts after the compute

        tc = time.time()
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
        t_comp_cum += time.time() - tc

        if (step + 1) % lf.val_every_steps == 0:
            _validate_latent(
                model, vae, ds, order, n, store, cfg, device, rank, step + 1, bucket_hr
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
        meta = {
            "iterations": total,
            "bucket_hr": bucket_hr,
            "n_crops": n,
            "n_params_m": round(n_params / 1e6, 2),
            "batch_size": bs,
            "prefetch": lf.prefetch,
            "phase1_target": total,
            "data_wait_pct": 100.0 * t_data_cum / max(1e-9, t_data_cum + t_comp_cum),
        }
        (out / "train-meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[latent] done: {out / 'latest.pt'}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return total
