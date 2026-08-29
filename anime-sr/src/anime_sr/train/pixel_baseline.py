"""M2 Pixel Baseline training loop (plan §M2, step 7).

Deterministic, resumable schedule (plan §11.5 contract):

    step s (global, across all ranks)
        exposure_index = s % _EXPOSURE_PER_CYCLE
        data_cycle     = s // _EXPOSURE_PER_CYCLE
        sample index for rank r, local slot i:
            (s * (bs * world) + r * bs + i) % len(ds)

So a resume at step ``s0`` reproduces the exact data stream (bit-exact
exposures), matching the M1 degradation/codec-bank determinism.

The train stream is assembled via ``SRDataset.fetch`` on a fixed-size
thread pool (batch-parallel: worker ``i`` always serves the ``i``-th
stream slot, so the exposure stream stays bit-exact vs the sequential
schedule); the validation stream uses a plain ``DataLoader`` (no
shuffle). No shuffling anywhere: the exposure seed is the only
randomness, and it is a pure function of (seed, sample, cycle,
exposure).
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from anime_sr.config.schema import Config
from anime_sr.data.pipeline import _EXPOSURE_PER_CYCLE, SRDataset, make_loader
from anime_sr.model.pixel_baseline import PixelBaseline


class _ValView(torch.utils.data.Dataset):
    """(hr, lq) only — the SampleMeta third element is not collatable."""

    def __init__(self, ds: SRDataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        hr, lq, _ = self.ds[i]
        return hr, lq


def _optimizer_for(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    o = cfg.optimizer
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        (no_decay if (p.ndim <= 1 or any(t in name for t in o.no_decay)) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": o.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=o.lr, betas=(o.betas[0], o.betas[1]), eps=o.eps)


def _cosine_lr(step: int, total: int, base_lr: float, cfg: Config) -> float:
    s = cfg.scheduler
    warm = max(1, int(total * s.warmup_fraction))
    if step < warm:
        return base_lr * (step + 1) / warm
    t = min(1.0, (step - warm) / max(1, total - warm))
    return base_lr * (s.min_lr_ratio + (1.0 - s.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def _sample_index(step: int, rank: int, i: int, bs: int, world: int, n: int) -> int:
    return (step * (bs * world) + rank * bs + i) % n


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


def _save_ckpt(path: Path, step: int, model: nn.Module, opt: torch.optim.Optimizer) -> None:
    payload = {"step": step, "model": _unwrap(model).state_dict(), "optimizer": opt.state_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic; overwrites an existing file (unlike Path.rename on Windows)


def _load_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, device: torch.device) -> int:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    return int(payload["step"])


def run_pixel_baseline(
    cfg: Config,
    *,
    index_dir: str | Path,
    webp_dir: str | Path,
    out_dir: str | Path,
    bucket_hr: int = 1024,
    rank: int = 0,
    world_size: int = 1,
    start_step: int = 0,
    resume: str | Path | None = None,
) -> int:
    """Train (or resume) the M2 pixel baseline; returns the final global step."""
    pb = cfg.pixel_baseline
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    bank = None
    if cfg.data.bank_dir:
        from anime_sr.data.codec_bank import CodecBank

        bank = CodecBank(cfg.data.bank_dir)

    # §10.5 clean score is a frozen offline sidecar (P1-4, 2026-08-29) —
    # the pixel baseline does not gate on it; the latent_flow trainer owns
    # the report + gate.
    ds = SRDataset(
        index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="train",
        bank=bank,
    )
    val_ds = SRDataset(
        index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="validation",
        bank=bank,
    )
    val_loader = make_loader(_ValView(val_ds), batch_size=pb.batch_size, shuffle=False) if pb.val_every_steps > 0 else None

    model = PixelBaseline(pb.base_channels, pb.depth).to(device)
    opt = _optimizer_for(cfg, model)
    if resume is not None:
        start_step = _load_ckpt(Path(resume), model, opt, device)
        if rank == 0:
            print(f"[baseline] resumed at step {start_step} from {resume}")
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[rank % max(1, torch.cuda.device_count())] if torch.cuda.is_available() else None)
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (cfg.hardware.dtype == "bf16" and device.type == "cuda")
        else (lambda: nullcontext())
    )

    total = pb.iterations
    base_lr = cfg.optimizer.lr
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[baseline] {n_params / 1e6:.2f}M params, {len(ds)} train samples, "
            f"bucket {bucket_hr}, bs={pb.batch_size} x world={world_size}, "
            f"steps {start_step}..{total}, device={device}"
        )

    # Batch-parallel fetch: CPU decode + degradation is the step bottleneck;
    # each worker serves exactly its stream slot, so the exposure stream is
    # unchanged (determinism comes from the slot assignment, not the thread).
    pool = ThreadPoolExecutor(max_workers=max(1, pb.batch_size), thread_name_prefix="fetch")
    t0 = time.time()
    t_data_cum = 0.0  # M1 stress telemetry (plan §M1#8): data-wait fraction of step time
    t_comp_cum = 0.0
    for step in range(start_step, total):
        exp = step % _EXPOSURE_PER_CYCLE
        cyc = step // _EXPOSURE_PER_CYCLE
        lr = _cosine_lr(step, total, base_lr, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        tf = time.time()
        futs = [
            pool.submit(ds.fetch, _sample_index(step, rank, i, pb.batch_size, world_size, len(ds)), exp, cyc)
            for i in range(pb.batch_size)
        ]
        hrs: list[torch.Tensor] = []
        lqs: list[torch.Tensor] = []
        for f in futs:  # ordered: slot i -> futs[i] (same stream as sequential)
            hr, lq, _ = f.result()
            hrs.append(hr)
            lqs.append(lq)
        hr = torch.stack(hrs).to(device, non_blocking=True)
        lq = torch.stack(lqs).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # attribute the H2D transfer to the data phase
        t_data_cum += time.time() - tf
        tc = time.time()
        with autocast():
            out_t = model(lq)
            loss = pb.l1_weight * (out_t - hr).abs().mean() + pb.l2_weight * ((out_t - hr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if cfg.gradient.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient.clip_norm)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_comp_cum += time.time() - tc

        if val_loader is not None and (step + 1) % pb.val_every_steps == 0:
            _validate(model, val_loader, device, rank, step)
        if rank == 0 and (step + 1) % pb.save_every_steps == 0:
            _save_ckpt(out / f"step-{step + 1:07d}.pt", step + 1, model, opt)
        if rank == 0 and ((step + 1) % 50 == 0 or step + 1 == total):
            done = step + 1 - start_step
            wait_pct = 100.0 * t_data_cum / max(1e-9, t_data_cum + t_comp_cum) if done >= 50 else -1.0
            extra = f" data_wait={wait_pct:.1f}%" if done >= 50 else ""
            print(
                f"[baseline] step {step + 1}/{total} loss={loss.item():.4f} lr={lr:.2e} "
                f"({(step + 1 - start_step) / max(1e-3, time.time() - t0):.1f} it/s){extra}",
                flush=True,
            )

    pool.shutdown(wait=True)
    _save_ckpt(out / "latest.pt", total, model, opt)
    if rank == 0:
        meta = {
            "iterations": total,
            "bucket_hr": bucket_hr,
            "base_channels": pb.base_channels,
            "depth": pb.depth,
            "data_wait_pct": 100.0 * t_data_cum / max(1e-9, t_data_cum + t_comp_cum),
        }
        (out / "train-meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[baseline] done: {out / 'latest.pt'}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return total


def _validate(model: nn.Module, loader, device: torch.device, rank: int, step: int) -> None:
    if rank != 0:
        return
    mod = _unwrap(model)
    mod.eval()
    n = 0
    l1 = 0.0
    mse = 0.0
    with torch.no_grad():
        for hr, lq in loader:
            hr = hr.to(device)
            lq = lq.to(device)
            o = mod(lq)
            l1 += (o - hr).abs().mean().item()
            mse += ((o - hr) ** 2).mean().item()
            n += hr.shape[0]
            if n >= 64:
                break
    l1 /= max(1, n)
    psnr = 10.0 * math.log10(4.0 / max(1e-12, mse))  # [-1, 1] range
    mod.train()
    print(f"[baseline] val step {step}: L1={l1:.4f} PSNR={psnr:.2f}dB (n={n})", flush=True)
