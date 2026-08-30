#!/usr/bin/env python3
"""Phase I-P RGB inference spot check (p1formal latest.pt + frozen Mage-VAE).

Sets
  A: 128 held-out val crops (sr-validation, zero train overlap) x 5 profiles
     P0..P4  -> paired metrics vs HR GT (6-column outputs per pair)
  B: 56 stress-curated train crops (8 content categories x 7), profile P2
     (+ P4 for 8) -> paired metrics
  C: 60 real danbooru original small-web images -> LQ 256, outputs only,
     human evaluation (no paired metrics, no GT)

Paired outputs per A/B pair:  HR GT / LQ(256) / bicubic4x / Mage anchor
  = Decode(E(Bicubic4x(LQ))) / Phase I-P 1-step (Faithful) / Phase I-P 4-step
  (Experimental).  1-step=Faithful, 4-step=Experimental; 4-step may only be
  claimed as a Quality selling point if RGB metrics beat 1-step (plan 16.4).

Metrics (per method vs GT, bucketed by degradation profile):
  PSNR-RGB, PSNR-Y, SSIM-RGB, SSIM-Y, edge-F1 (3px dilation matching),
  edge displacement (px, one-way within 6px), line-width error (px, FWHM
  proxy at matched edges), flat-region HF energy + artifact ratio vs
  bicubic.  LPIPS/DISTS: NOT available in the DTK env (no torchvision) —
  disclosed in the report, not installed (user directive: no large new deps).

Run under DTK env (torch requires /opt/dtk-26.04).  HCU single device.

    source /opt/dtk-26.04/env.sh; export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
    PYTHONPATH=src /usr/local/bin/python3.11 -m anime_sr.cli.rgb_eval --set A ...
or standalone:
    PYTHONPATH=src python rgb_eval.py --set A
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from anime_sr.config.loader import load_config
from anime_sr.data.buckets import crop_box
from anime_sr.data.degradation import degrade_hr
from anime_sr.data.pipeline import box_seed
from anime_sr.flow.sampling import FlowSampler
from anime_sr.model.uflow import AnimeSRModel
from anime_sr.train.latent_flow import _PixelVelocity
from anime_sr.vae.mage import load_frozen_vae
from PIL import Image

REPO = Path("/root/anime-sr-p1formal")
CKPT = "/root/private_data/anime-sr/output_model/latent-flow-phase1-pi/latest.pt"
VAE = "/root/private_data/anime-sr/model/vae/mage-vae.safetensors"
WEBP_SHARDS = [
    # 08-30: webp staging now lives on the persistent volume (new sakrua10
    # container re-provision; /var/tmp is ephemeral and was lost)
    "/root/private_data/anime-sr/data/webp/shard-000000.tar",
    "/root/private_data/anime-sr/data/webp/shard-000001.tar",
    "/root/private_data/anime-sr/data/webp/shard-000002.tar",
]
VAL_JSON = "/root/private_data/anime-sr/data/index/sr-validation-v1.json"
CONFIGS = [
    "config/base.toml",
    "config/data.toml",
    "config/phase1-small.toml",
    "config/phase1-pi-formal.toml",
]
PROFILES = ["P0_clean", "P1_mild_web", "P2_normal_web", "P3_anime_codec", "P4_severe"]
GLOBAL_SEED = 42  # SRDataset default (latent_flow does not override)
BUCKET_HR = 1024  # p1formal pool: single 1024 bucket (latent_flow bucket_hr default)
EXPOSURE = (0, 0)  # (data_cycle, exposure_index) = val-probe default exposure


# ---------------------------------------------------------------------
# tensors: all model/VAE I/O in [-1, 1]; metrics in [0, 255] uint8
# ---------------------------------------------------------------------

def hr_crop_for(sid: str, webp_map: dict[str, Path]) -> tuple[torch.Tensor, tuple[int, int]]:
    """Exact p1formal pool crop for sample sid (bit-identical to training):
    original webp (full size) -> seeded center-jitter 1024^2 crop
    (crop_box + box_seed, pipeline.fetch rule) -> (1, 3, 1024, 1024) fp32
    [-1, 1] CPU.  Returns (hr_crop, (x, y))."""
    im = Image.open(str(webp_map[sid])).convert("RGB")
    w, h = im.size
    if w < BUCKET_HR or h < BUCKET_HR:
        raise ValueError(f"{sid}: original {w}x{h} < HR bucket {BUCKET_HR}")
    seed_box = box_seed(sid, *EXPOSURE)
    x, y = crop_box(w, h, BUCKET_HR, seed_box)
    arr = np.asarray(im, dtype=np.uint8).copy()
    arr = arr[y : y + BUCKET_HR, x : x + BUCKET_HR]
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return (t * (2.0 / 255.0) - 1.0).unsqueeze(0), (x, y)


def lq_for(hr_crop: torch.Tensor, cfg_p, sid: str) -> torch.Tensor:
    """degrade_hr on the unbatched [3, 1024, 1024] crop (exact pipeline call)."""
    lq, _ = degrade_hr(
        hr_crop[0],
        cfg_p,
        global_seed=GLOBAL_SEED,
        sample_id=sid,
        data_cycle=EXPOSURE[0],
        exposure_index=EXPOSURE[1],
    )
    assert tuple(lq.shape) == (3, 256, 256), f"unexpected lq shape {tuple(lq.shape)}"
    return lq  # unbatched [3, 256, 256]; callers add the batch dim as needed


def to_u8cpu(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) in [-1, 1] -> HxWx3 uint8 numpy (clamped)."""
    y = ((x[0].clamp(-1, 1) + 1.0) * 127.5).round().cpu().numpy().astype(np.uint8)
    return np.transpose(y, (1, 2, 0))


def gray01(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def save_png(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


# ---------------------------------------------------------------------
# metrics (torch/numpy only; no skimage/scipy/lpips/dists available)
# ---------------------------------------------------------------------

def _gauss2d(k: int = 11, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-((ax ** 2) / (2 * sigma * sigma)))
    return (torch.outer(g, g) / torch.outer(g, g).sum()).float()


def ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """x, y: (1, 3, H, W) in [0, 1]. Returns mean SSIM over channels."""
    win = _gauss2d().to(x.device).view(1, 1, 11, 11)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for i in range(x.shape[1]):
        xi, yi = x[0, i : i + 1], y[0, i : i + 1]
        mx = F.conv2d(xi, win, padding=5)[0, 0]
        my = F.conv2d(yi, win, padding=5)[0, 0]
        mxx = F.conv2d(xi * xi, win, padding=5)[0, 0]
        myy = F.conv2d(yi * yi, win, padding=5)[0, 0]
        mxy = F.conv2d(xi * yi, win, padding=5)[0, 0]
        vx = mxx - mx * mx
        vy = myy - my * my
        vxy = mxy - mx * my
        out.append(((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return float(torch.stack(out).mean().item())


def psnr(a: np.ndarray, b: np.ndarray, gray: bool = False) -> float:
    a = a[..., 0] if gray else a
    b = b[..., 0] if gray else b
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(255.0 ** 2 / mse))


def _sobel(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients via torch conv2d (scipy's convolve2d is unavailable in
    the DTK env).  conv2d is cross-correlation-free: kernels are flipped vs
    the scipy convention, but the result feeds |.| / hypot — invariant."""
    m = m.astype(np.float32)
    t = torch.from_numpy(m)[None, None]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = torch.nn.functional.conv2d(t, kx, padding=1).numpy()[0, 0]
    gy = torch.nn.functional.conv2d(t, ky, padding=1).numpy()[0, 0]
    return np.abs(gx), np.abs(gy)


def _nms8(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """8-direction non-maximum suppression (numpy, valid interior)."""
    h, w = mag.shape
    out = np.zeros_like(mag)
    d = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # vectorized: compare against all 8 neighbors via shifted views
    padded = np.pad(mag, 1, mode="constant", constant_values=-1.0)
    neigh = np.stack(
        [padded[dy + 1 : dy + 1 + h, dx + 1 : dx + 1 + w] for dy, dx in d], axis=0
    )  # (8, h, w): offset (dy, dx) neighbor views, aligned to the original grid
    m2 = mag[1 : h - 1, 1 : w - 1]
    # direction-aware NMS: keep pixel if mag >= both along-gradient neighbors
    ang = np.arctan2(gy[1 : h - 1, 1 : w - 1], gx[1 : h - 1, 1 : w - 1]) * 2  # /pi
    idx = ((ang + 4) % 8).astype(np.int64)  # (h-2, w-2)
    # per-pixel gather along axis 0 (a 2-D np.take here would materialize
    # (h-2, w-2, h, w) — ~4 TiB at 1024²; index the interior pixels directly)
    iy, ix = np.ogrid[1 : h - 1, 1 : w - 1]
    a1 = neigh[idx, iy, ix]
    a2 = neigh[(idx + 4) % 8, iy, ix]
    core = m2 >= a1
    core &= m2 >= a2
    out[1 : h - 1, 1 : w - 1] = np.where(core, m2, 0.0)
    return out


def edge_map(gray: np.ndarray, frac_hi: float = 0.5, frac_lo: float = 0.25) -> np.ndarray:
    """Canny-lite: sobel magnitude -> NMS -> double threshold (fraction of the
    image's own mean gradient) -> binary edges."""
    gx, gy = _sobel(gray)
    mag = np.hypot(gx, gy)
    nms = _nms8(mag, gx, gy)
    t_hi = frac_hi * mag.mean()
    t_lo = frac_lo * mag.mean()
    strong = nms >= t_hi
    weak = (nms >= t_lo) & ~strong
    out = strong.copy()
    # cheap connectivity: keep weak pixels adjacent (8-conn) to strong
    w2, h2 = weak.shape
    pad = np.pad(weak, 1)
    neigh_sum = np.zeros_like(weak)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh_sum += np.pad(strong, 1)[1 + dy : 1 + dy + h2, 1 + dx : 1 + dx + w2]
    out |= weak & (neigh_sum > 0)
    del w2, h2, pad, neigh_sum
    return out


def _dilate(a: np.ndarray, iters: int = 3) -> np.ndarray:
    """3x3 OR dilation, `iters` iterations (each iteration grows the mask by
    one pixel in every direction)."""
    out = a.astype(np.uint8).copy()
    h, w = out.shape
    for _ in range(iters):
        p = np.pad(out, 1)
        acc = np.zeros((h, w), np.uint8)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                acc |= p[dy : dy + h, dx : dx + w]
        out = acc
    return out.astype(bool)


def _dilate_dist(a: np.ndarray, iters: int = 6) -> np.ndarray:
    """distance transform approx: value = 0 on a, else dilation step count."""
    cur = a.astype(bool)
    dist = np.zeros(a.shape, np.float32)
    frontier = ~a
    unvisited = frontier.copy()
    for step in range(1, iters + 1):
        grown = _dilate(cur, 1) & unvisited
        dist[grown] = step
        unvisited &= ~grown
        cur = grown | cur
        if not unvisited.any():
            break
    dist[unvisited] = iters  # capped
    return dist


def edge_f1(ea: np.ndarray, eb: np.ndarray, dil: int = 3) -> float:
    da, db = _dilate(ea, dil), _dilate(eb, dil)
    tp = float((ea & db).sum() + (eb & da).sum())  # matched both ways
    fp = float(eb.sum() - (eb & da).sum() + ea.sum() - (ea & db).sum())
    fn = fp  # symmetric
    if tp + fp + fn == 0:
        return 1.0
    return float(tp / (tp + fp + fn))


def edge_disp(pred: np.ndarray, gt: np.ndarray, r: int = 6) -> float:
    """mean distance (px) from pred edge pixels to the nearest GT edge."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float(r)
    dist = _dilate_dist(gt, r)
    return float(dist[pred].mean())


def linewidth_err(pred_e: np.ndarray, gt_e: np.ndarray, gray_p: np.ndarray,
                  gray_g: np.ndarray) -> float:
    """FWHM proxy of the gradient profile along the normal at matched edges."""
    match = pred_e & _dilate(gt_e, 2)
    if match.sum() < 32:
        return float("nan")
    # sample up to 4000 matched pixels (deterministic stride)
    ys, xs = np.nonzero(match)
    if len(ys) > 4000:
        step = len(ys) // 4000
        ys, xs = ys[::step], xs[::step]
    errs = []
    for y, x in zip(ys, xs):
        for g in (gray_p, gray_g):
            prof = g[max(0, y - 4) : min(g.shape[0], y + 5), x]
            if len(prof) < 9:
                continue
            # width = count of consecutive samples above 0.5 * local max
            peak = prof.max()
            if peak < 2.0:
                continue
            above = prof > 0.5 * peak
            run = 0
            for i in range(len(above)):
                if above[i]:
                    run += 1
                else:
                    if run:
                        break
            if run:
                errs.append(run)
    if not errs:
        return float("nan")
    # pairs (pred, gt) alternating
    half = len(errs) // 2
    p_w = np.mean(errs[:half]) if half else float("nan")
    g_w = np.mean(errs[half:]) if len(errs) > half else p_w
    return float(abs(p_w - g_w))


def flat_hf(gray: np.ndarray) -> tuple[float, float]:
    """flat-region HF energy: mean |laplacian| over GT-flat 16x16 patches.
    Returns (flat_fraction, mean HF in 0-255 units)."""
    h, w = gray.shape
    g = gray.astype(np.float32)
    lap = np.zeros_like(g)
    lap[1 : h - 1, 1 : w - 1] = np.abs(
        4 * g[1 : h - 1, 1 : w - 1]
        - g[:-2, 1 : w - 1]
        - g[2:, 1 : w - 1]
        - g[1 : h - 1, :-2]
        - g[1 : h - 1, 2:]
    )
    # local std via box filters
    def box(a, k):
        c = F.avg_pool2d(
            torch.from_numpy(a).unsqueeze(0).unsqueeze(0), k, stride=k
        ).numpy()[0, 0]
        return c

    k = 16
    n = (h // k) * (w // k)
    gs = gray[: n * k, : n * k].reshape(n, k, k)
    std = gs.std(axis=(1, 2), ddof=0)
    flat = std < 2.0
    if flat.sum() < 8:
        return 0.0, float("nan")
    hfn = lap[: n * k, : n * k].reshape(n, k, k)
    return float(flat.mean()), float(hfn[flat].mean())


def compute_metrics(gt_u8: np.ndarray, methods: dict[str, np.ndarray]) -> dict:
    """methods: name -> HxWx3 uint8 (1024).  GT = 1024 HR crop."""
    g = gray01(gt_u8).astype(np.float32)
    gt_e = edge_map(g)
    out = {}
    flat_frac, _flat_hf_gt = flat_hf(g)
    hf = {m: flat_hf(gray01(methods[m]).astype(np.float32))[1] for m in methods}
    out["psnr_rgb"] = {m: psnr(a, gt_u8) for m, a in methods.items()}
    out["psnr_y"] = {m: psnr(a, gt_u8, gray=True) for m, a in methods.items()}
    out["ssim_rgb"] = {}
    out["ssim_y"] = {}
    for m, a in methods.items():
        af = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        gf = torch.from_numpy(gt_u8.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        out["ssim_rgb"][m] = ssim(af, gf)
        out["ssim_y"][m] = ssim(
            af.mean(0, keepdim=True).expand(af.shape), gf.mean(0, keepdim=True).expand(gf.shape)
        )
    for m, a in methods.items():
        pm = edge_map(gray01(a).astype(np.float32))
        out.setdefault("edge_f1", {})[m] = edge_f1(pm, gt_e)
        out.setdefault("edge_disp_px", {})[m] = edge_disp(pm, gt_e)
        out.setdefault("linewidth_err_px", {})[m] = linewidth_err(
            pm, gt_e, gray01(a).astype(np.float32), g
        )
    out["flat_hf_energy"] = hf
    out["flat_fraction_gt"] = flat_frac
    bic_hf = hf.get("bicubic4x", 0.0)
    out["flat_artifact_ratio_vs_bic"] = {
        m: (v / bic_hf if bic_hf > 1e-6 else float("nan")) for m, v in hf.items()
    }
    return out


# ---------------------------------------------------------------------
# inference core
# ---------------------------------------------------------------------

class _PixelVelocityBf16(_PixelVelocity):
    """:class:`_PixelVelocity` wrapped in the same ``autocast(bf16)`` context
    the training val probes use (latent_flow ~L399).  ``sinusoidal_embedding``
    builds an fp32 feature vector, which feeds the (bf16) TimestepEmbedding
    Linears — a bare bf16 model crashes on that matmul, while autocast runs
    the Linears in bf16 exactly like the training probes do."""

    def forward(self, rt, t, sigma, cond):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return super().forward(rt, t, sigma, cond)


class Evaluator:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cuda")
        torch.cuda.set_device(0)
        cfgs = [str(REPO / c) for c in CONFIGS]
        self.cfg = load_config(*cfgs)
        self.cfg_profiles = {p: 1.0 for p in PROFILES}

        model = AnimeSRModel(self.cfg.model, zero_init_pixel=self.cfg.model.zero_init_pixel)
        payload = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = payload["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, f"ckpt mismatch: {len(missing)} missing / {len(unexpected)} unexpected"
        self.n_model_keys = len(sd)
        model.eval().to(self.device, dtype=torch.bfloat16)
        self.model = model
        self.ckpt_step = payload.get("step", -1)

        self.vae = load_frozen_vae(VAE, device=self.device, dtype=torch.bfloat16)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.sampler = FlowSampler(_PixelVelocityBf16(model))
        self.profile_cfg = {
            p: self._one_hot(p) for p in PROFILES
        }

    def _one_hot(self, p: str):
        c = copy.deepcopy(self.cfg)
        c.degradation.profiles = {p: 1.0}
        return c

    def sha256_of(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _zero_grads(self) -> None:
        for p in self.model.parameters():
            p.grad = None
        for p in self.vae.parameters():
            p.grad = None

    def infer_image(
        self, hr: torch.Tensor, sid: str, profile: str
    ) -> dict[str, torch.Tensor]:
        """hr: (1,3,1024,1024) [-1,1] CPU. Returns dict of (1,3,1024,1024) CPU."""
        cfg_p = self.profile_cfg[profile]
        lq = lq_for(hr, cfg_p, sid).unsqueeze(0)
        dev, dt = self.device, torch.bfloat16
        bic4 = F.interpolate(lq, size=(1024, 1024), mode="bicubic", align_corners=False)
        with torch.no_grad():
            z_lr = self.vae.encode(bic4.to(dev, dt))
            anchor = self.vae.decode(z_lr)
            cond = (z_lr, lq.to(dev, dt))
            z1 = self.sampler.one_step(z_lr, cond, sigma=0.0)
            o1 = self.vae.decode(z1)
        # DTK/HCU driver-defect workaround (2026-08-30): sustained
        # forward-only dispatches balloon host-side driver staging
        # unbounded on this HCU pool (guard-kills at the 112 GiB cgroup cap,
        # plus a phantom ~62.8 GiB HBM-accounting OOM), while the identical
        # workload with a backward on the real graph every step stays flat
        # (bare5: 20000 iters flat vs bare2: dead in 32 s; a one-time
        # warmup (bare6) and a tiny dummy backward (bare7) both still die —
        # the backward must ride the real workload).  Phase I-P training
        # (same model fwd+bwd) ran clean for days on the same host, so the
        # four_step graph runs grad-enabled and one dummy backward fires on
        # the real output per image.  Forward numerics are unchanged —
        # only an extra backward's memory/time is added.
        z4 = self.sampler.four_step(z_lr, cond, sigma=0.0)
        o4 = self.vae.decode_with_grad(z4)
        o4.sum().backward()
        self._zero_grads()
        o4 = o4.detach()
        return {
            "lq": lq.float().cpu(),
            "bicubic4x": bic4.float().cpu(),
            "anchor": anchor.float().cpu(),
            "s1": o1.float().cpu(),
            "s4": o4.float().cpu(),
        }

    def infer_many(self, jobs: list[tuple[torch.Tensor, str, str]]) -> list[dict]:
        """jobs: (hr_cpu, sid, profile).  Per-profile (b1) inference, grouped by sid.

        08-30 OOM fix: the old batched-across-profiles attempt (b5 at
        1024x1024) needs ~63 GiB of HCU VRAM (measured on 2026-08-30:
        63.36/63.98 GiB allocated at the b5 four_step -> HIP OOM on a 10 MiB
        allocation) and therefore ALWAYS OOMs the 64 GiB HCU, and on the
        re-provisioned pods the HCU OOM was followed by unbounded host-side
        growth that crossed the 118 GiB cgroup cap (two pod OOM kills,
        ports 10357/10054).  Per-profile b1 is exactly the path the old OOM
        fallback already executed, so results are numerically the fallback
        path's results, and HCU VRAM stays ~12 GiB with headroom."""
        out: list[dict] = []
        # group by (sid) -> up to 5 profiles share one hr load
        from collections import defaultdict

        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        hrs: dict[str, torch.Tensor] = {}
        for i, (hr, sid, p) in enumerate(jobs):
            groups[sid].append((i, p))
            hrs[sid] = hr
        n_groups = len(groups)
        for gi, (sid, items) in enumerate(groups.items()):
            hr = hrs[sid]
            profiles = [p for _, p in items]
            if gi % 8 == 0:
                print(
                    f"[rgb-eval] infer group {gi}/{n_groups} (sid={sid}, "
                    f"{len(profiles)} profiles, b1)", flush=True
                )
            for j, p in items:
                out.append(self.infer_image(hr, sid, p))
            del hr
            gc.collect()
            torch.cuda.empty_cache()
        # restore job order
        final = [None] * len(jobs)
        # re-index: iterate groups in original order
        ptr = 0
        for sid, items in groups.items():
            for _, p in items:
                # find the job index for this (sid, p)
                j = next(
                    i for i, (h, s, pp) in enumerate(jobs)
                    if s == sid and pp == p
                )
                final[j] = out[ptr]
                ptr += 1
        return final

    # NOTE: no hand-rolled batched integrators — FlowSampler is batch-capable
    # (t/sigma are (B,) vectors; sigma=0 -> r0=0 via sample_source_noise), so the
    # batched path calls sampler.one_step / sampler.four_step directly, keeping
    # the call pattern bit-identical to the training val probes.


# ---------------------------------------------------------------------
# set construction
# ---------------------------------------------------------------------

def build_sid_webp_map() -> dict[str, Path]:
    m: dict[str, Path] = {}
    for shard in WEBP_SHARDS:
        d = Path(shard)
        for f in d.iterdir():
            if f.suffix == ".webp":
                m[f.stem] = f
    return m


def select_set_a(n: int = 128) -> list[str]:
    """First n val sids (index order) that (a) exist in the webp staging and
    (b) are eligible for the p1formal 1024 bucket (original >= 1024^2)."""
    doc = json.loads(Path(VAL_JSON).read_text())
    ids = doc["validation_ids"]
    assert doc.get("zero_overlap"), "val split must be zero-overlap with train"
    webp = build_sid_webp_map()
    sel: list[str] = []
    for s in ids:
        if len(sel) >= n:
            break
        if s not in webp:
            continue
        im = Image.open(str(webp[s]))
        if im.width >= BUCKET_HR and im.height >= BUCKET_HR:
            sel.append(s)
    return sel


def load_set_b(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())  # [{"sid":..., "category":...}]


def load_set_c(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())  # [{"sid":..., "src":..., "w":..., "h":...}]


# ---------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------

def contact_sheet(
    cells: list[list[dict]], out_path: Path, cell: int = 256
) -> None:
    """cells: rows of {name, img(HxWx3 uint8)}. 6 cols for A/B, 5 for C."""
    rows = [list(r.values()) for r in cells]
    ncols = len(rows[0])
    sheet = np.zeros((len(rows) * cell + (len(rows) + 1) * 4, ncols * cell + (ncols + 1) * 4, 3), np.uint8)
    for ri, row in enumerate(rows):
        for ci, img in enumerate(row):
            im = np.asarray(Image.fromarray(img).resize((cell, cell), Image.LANCZOS))
            y = 4 + ri * (cell + 4)
            x = 4 + ci * (cell + 4)
            sheet[y : y + cell, x : x + cell] = im
    save_png(sheet, out_path)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["A", "B", "C", "all"])
    ap.add_argument("--a-n", type=int, default=128)
    ap.add_argument("--b-curation", default="")
    ap.add_argument("--c-selection", default="")
    ap.add_argument("--out", default="/root/anime-sr-mig/rgb-eval-out")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap pairs")
    # 08-30 r5.5 process isolation (HSA/host-leak forensics): the DTK/HSA host
    # staging leak grows with per-process forward count and is invisible to the
    # GC (probe4: gc object count flat at 252k while cgroup climbed 4->99 GiB).
    # One fresh process per Set A chunk keeps any per-process leak bounded; the
    # sheet pass aggregates the per-chunk records.
    ap.add_argument("--chunk", type=int, default=-1, help="run only pair-chunk i (0-based) of the set's jobs")
    ap.add_argument("--chunksize", type=int, default=3, help="pairs per chunk (default 3)")
    ap.add_argument("--sheets", action="store_true", help="aggregate-only pass: union metrics.jsonl + sheets from chunk-*.jsonl / PNGs, no model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.sheets:
        # aggregate-only pass (no model, seconds): union metrics.jsonl from
        # per-chunk jsonls (falling back to per-set jsonls of full runs) and
        # rebuild all sheets from per-pair PNGs on disk.
        base_a, base_b, base_c = out / "A", out / "B", out / "C"
        recs_a: list[dict] = []
        for cf in sorted(base_a.glob("chunk-*.jsonl")):
            with open(cf) as fh:
                recs_a.extend(json.loads(line) for line in fh if line.strip())
        recs_b: list[dict] = []
        for cf in sorted(base_b.glob("chunk-*.jsonl")):
            with open(cf) as fh:
                recs_b.extend(json.loads(line) for line in fh if line.strip())
        recs_c: list[dict] = []
        for cf in sorted(base_c.glob("chunk-*.jsonl")):
            with open(cf) as fh:
                recs_c.extend(json.loads(line) for line in fh if line.strip())
        union = recs_a + recs_b + recs_c
        for pf, have in (
            (out / "metrics-B.jsonl", bool(recs_b)),
            (out / "metrics-C.jsonl", bool(recs_c)),
        ):
            if pf.exists() and not have:
                union.extend(
                    json.loads(line) for line in pf.read_text().splitlines() if line.strip()
                )
        (out / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m, default=str) for m in union)
        )
        print(
            f"[rgb-eval] metrics.jsonl union: {len(union)} records "
            f"(A {len(recs_a)}, B {len(recs_b)}, C {len(recs_c)})",
            flush=True,
        )
        sheets_list: list[str] = []
        if recs_a:
            _write_sheets_a(base_a, recs_a, sheets_list)
        # Set B category sheets: rebuild from per-pair PNGs (first 8 per category)
        if recs_b and args.b_curation:
            cur_b = load_set_b(args.b_curation)
            cat_of_b = {c["sid"]: c.get("category", "unknown") for c in cur_b}
            cats: dict[str, list[dict]] = {}
            for r in recs_b:
                cats.setdefault(cat_of_b.get(r["sid"], "unknown"), []).append(r)
            for cat, rows in cats.items():
                rows = sorted(rows, key=lambda r: (r["sid"], r["profile"]))[:8]
                cells_b = []
                for r in rows:
                    d = base_b / f"{r['sid']}__{r['profile']}"
                    cells_b.append({
                        "gt": _down256(np.asarray(Image.open(d / "gt.png"))),
                        "lq": _down256(np.asarray(Image.open(d / "lq256.png").resize((256, 256), Image.BICUBIC))),
                        "bic": _down256(np.asarray(Image.open(d / "bicubic4x.png"))),
                        "anchor": _down256(np.asarray(Image.open(d / "mage_anchor.png"))),
                        "s1": _down256(np.asarray(Image.open(d / "phase1p_s1.png"))),
                        "s4": _down256(np.asarray(Image.open(d / "phase1p_s4.png"))),
                    })
                sp = base_b / f"sheet_{cat}.png"
                contact_sheet(cells_b, sp)
                sheets_list.append(str(sp))
        # Set C: 10-row contact sheets from per-item PNGs in selection order
        if recs_c and args.c_selection:
            sel_c = load_set_c(args.c_selection)
            n_c = max((int(r["idx"]) for r in recs_c), default=-1) + 1
            for k in range(0, n_c, 10):
                cells_c = []
                for j in range(k, min(k + 10, n_c)):
                    cid = sel_c[j]["sid"]
                    d = base_c / f"{j:02d}_{cid}"
                    cells_c.append({
                        "lq": _down256(np.asarray(Image.open(d / "lq256.png"))),
                        "bic": _down256(np.asarray(Image.open(d / "bicubic4x.png"))),
                        "anchor": _down256(np.asarray(Image.open(d / "mage_anchor.png"))),
                        "s1": _down256(np.asarray(Image.open(d / "phase1p_s1.png"))),
                        "s4": _down256(np.asarray(Image.open(d / "phase1p_s4.png"))),
                    })
                sp = base_c / f"sheet_c_{k // 10:02d}.png"
                contact_sheet(cells_c, sp)
                sheets_list.append(str(sp))
        print(f"[rgb-eval] sheets written: {len(sheets_list)} files", flush=True)
        print(f"[rgb-eval] done in {time.time() - t0:.1f}s", flush=True)
        return
    ev = Evaluator(args)
    print(
        f"[rgb-eval] model keys={ev.n_model_keys} ckpt step={ev.ckpt_step} "
        f"vae fingerprint={json.dumps(ev.vae.fingerprint())}", flush=True
    )
    ckpt_sha = ev.sha256_of(CKPT)
    print(f"[rgb-eval] latest.pt sha256={ckpt_sha}", flush=True)
    git_rev = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"[rgb-eval] code rev={git_rev}", flush=True)

    webp = build_sid_webp_map()
    print(f"[rgb-eval] webp map: {len(webp)} crops", flush=True)
    all_metrics: list[dict] = []
    sheets: list[str] = []

    # ---------------- Set A (08-30: chunked 16 sids/chunk to bound host RSS) ----------------
    if args.set in ("A", "all"):
        sids_all = select_set_a(args.a_n)
        boxes: dict[str, tuple[int, int]] = {}
        # r5.5 process isolation: work at PAIR granularity (sid-major,
        # profile-minor). The DTK/HSA host-staging leak scales with per-process
        # forward count, so the driver (run_full_eval.sh) runs one fresh
        # process per pair-chunk (3 pairs: worst case ~78 GiB < 92 GiB guard).
        all_pairs = [(s, p) for s in sids_all for p in PROFILES]
        if args.limit:
            # limit counts pairs; whole sids only (limit 10 -> 2 sids x 5 profiles)
            all_pairs = all_pairs[
                : ((args.limit + len(PROFILES) - 1) // len(PROFILES)) * len(PROFILES)
            ]
        if args.chunk >= 0:
            cs = args.chunksize
            pairs = all_pairs[args.chunk * cs : (args.chunk + 1) * cs]
        else:
            pairs = all_pairs
        sids = list(dict.fromkeys(s for s, _ in pairs))
        print(
            f"[rgb-eval] set A: {len(sids)} sids, {len(pairs)} pairs "
            f"(chunk {args.chunk if args.chunk >= 0 else 'full'} of size "
            f"{args.chunksize if args.chunk >= 0 else '-'}, "
            f"crop rule: crop_box + box_seed, exposure {EXPOSURE})",
            flush=True,
        )
        base = out / "A"
        hrs: dict[str, torch.Tensor] = {}
        jobs: list[tuple[torch.Tensor, str, str]] = []
        for sid, p in pairs:
            if sid not in hrs:
                hr, box = hr_crop_for(sid, webp)
                boxes[sid] = box
                hrs[sid] = hr
            jobs.append((hrs[sid], sid, p))
        results = ev.infer_many(jobs)
        for (hr, sid, p), res in zip(jobs, results):
            d = base / f"{sid}__{p}"
            gt_u8 = to_u8cpu(hr)
            lq_u8 = to_u8cpu(res["lq"])
            methods = {
                "lq_up": np.asarray(Image.fromarray(lq_u8).resize((1024, 1024), Image.BICUBIC)),
                "bicubic4x": to_u8cpu(res["bicubic4x"]),
                "mage_anchor": to_u8cpu(res["anchor"]),
                "phase1p_s1": to_u8cpu(res["s1"]),
                "phase1p_s4": to_u8cpu(res["s4"]),
            }
            save_png(gt_u8, d / "gt.png")
            save_png(lq_u8, d / "lq256.png")
            for m, im in methods.items():
                save_png(im, d / f"{m}.png")
            met = compute_metrics(gt_u8, {k: v for k, v in methods.items() if k != "lq_up"})
            rec = {
                "set": "A", "sid": sid, "profile": p,
                "global_seed": GLOBAL_SEED,
                "data_cycle": EXPOSURE[0], "exposure_index": EXPOSURE[1],
                "crop_box": list(boxes[sid]),
                "code_rev": git_rev,
                "metrics": met,
            }
            all_metrics.append(rec)
        del jobs, results
        gc.collect()
        if args.chunk >= 0:
            # process-isolated chunk: persist records for the --sheets union
            # pass; no shared metrics.jsonl write, no sheets here.
            cf = base / f"chunk-{args.chunk}.jsonl"
            with open(cf, "w") as fh:
                for r in all_metrics:
                    fh.write(json.dumps(r) + "\n")
            print(
                f"[rgb-eval] set A chunk {args.chunk}: {len(all_metrics)} records -> {cf.name} "
                f"(done in {time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            _write_sheets_a(base, [r for r in all_metrics if r["set"] == "A"], sheets)

    # ---------------- Set B ----------------
    if args.set in ("B", "all"):
        assert args.b_curation, "--b-curation required for set B"
        cur = load_set_b(args.b_curation)
        cat_of = {c["sid"]: c.get("category", "unknown") for c in cur}
        n_cur_total = len(cur)
        if args.chunk >= 0:
            # process isolation: one fresh process per curation-chunk (leak
            # scales with per-process forward count); records -> chunk jsonl.
            cs = args.chunksize
            cur = cur[args.chunk * cs : (args.chunk + 1) * cs]
        jobs = []
        for c in cur:
            sid = c["sid"]
            profs = ["P2_normal_web", "P4_severe"] if c.get("stress_p4") else ["P2_normal_web"]
            hr, _ = hr_crop_for(sid, webp)
            for p in profs:
                jobs.append((hr, sid, p))
        if args.limit:
            jobs = jobs[: args.limit]
        print(
            f"[rgb-eval] set B: {len(cur)}/{n_cur_total} curated crops "
            f"(chunk {args.chunk if args.chunk >= 0 else 'full'} x {args.chunksize}), "
            f"{len(jobs)} pairs",
            flush=True,
        )
        results = ev.infer_many(jobs)
        base = out / "B"
        cat_sheets: dict[str, list] = {}
        for (hr, sid, p), res in zip(jobs, results):
            d = base / f"{sid}__{p}"
            gt_u8 = to_u8cpu(hr)
            lq_u8 = to_u8cpu(res["lq"])
            methods = {
                "lq_up": np.asarray(Image.fromarray(lq_u8).resize((1024, 1024), Image.BICUBIC)),
                "bicubic4x": to_u8cpu(res["bicubic4x"]),
                "mage_anchor": to_u8cpu(res["anchor"]),
                "phase1p_s1": to_u8cpu(res["s1"]),
                "phase1p_s4": to_u8cpu(res["s4"]),
            }
            save_png(gt_u8, d / "gt.png")
            save_png(lq_u8, d / "lq256.png")
            for m, im in methods.items():
                save_png(im, d / f"{m}.png")
            met = compute_metrics(gt_u8, {k: v for k, v in methods.items() if k != "lq_up"})
            cat = cat_of.get(sid, "unknown")
            rec = {
                "set": "B", "sid": sid, "profile": p, "category": cat,
                "code_rev": git_rev, "metrics": met,
            }
            all_metrics.append(rec)
            cat_sheets.setdefault(cat, []).append({
                "gt": gt_u8, "lq": np.asarray(Image.fromarray(lq_u8).resize((256, 256), Image.BICUBIC)),
                "bic": to_u8cpu(res["bicubic4x"]), "anchor": to_u8cpu(res["anchor"]),
                "s1": to_u8cpu(res["s1"]), "s4": to_u8cpu(res["s4"]),
            })
        if args.chunk >= 0:
            cf = base / f"chunk-{args.chunk}.jsonl"
            with open(cf, "w") as fh:
                for r in all_metrics:
                    fh.write(json.dumps(r) + "\n")
            print(
                f"[rgb-eval] set B chunk {args.chunk}: {len(all_metrics)} records -> {cf.name} "
                f"(done in {time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            for cat, rows in cat_sheets.items():
                rows = rows[:8]
                cells = [
                    {
                        "gt": _down256(r["gt"]),
                        "lq": r["lq"], "bic": _down256(r["bic"]), "anchor": _down256(r["anchor"]),
                        "s1": _down256(r["s1"]), "s4": _down256(r["s4"]),
                    }
                    for r in rows
                ]
                sp = base / f"sheet_{cat}.png"
                contact_sheet(cells, sp)
                sheets.append(str(sp))

    # ---------------- Set C ----------------
    if args.set in ("C", "all"):
        assert args.c_selection, "--c-selection required for set C"
        sel = load_set_c(args.c_selection)
        n_sel_total = len(sel)
        if args.limit:
            sel = sel[: args.limit]
        i_base = 0
        if args.chunk >= 0:
            # process isolation: one fresh process per item-chunk; dir names
            # and records carry the GLOBAL selection index.
            cs = args.chunksize
            i_base = args.chunk * cs
            sel = sel[i_base : i_base + cs]
        base = out / "C"
        c_rows: list[dict] = []
        for j, c in enumerate(sel):
            i = i_base + j
            src = Path(c["src"])
            im = Image.open(src).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            h0, w0 = arr.shape[:2]
            s = min(h0, w0)
            arr = arr[(h0 - s) // 2 : (h0 - s) // 2 + s, (w0 - s) // 2 : (w0 - s) // 2 + s]
            arr = np.asarray(Image.fromarray(arr).resize((256, 256), Image.LANCZOS))
            lq = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
            lq = lq / 255.0 * 2.0 - 1.0
            # C is human-eval only: no HR GT, no paired metrics.
            with torch.no_grad():
                dev, dt = ev.device, torch.bfloat16
                bic = F.interpolate(lq, size=(1024, 1024), mode="bicubic", align_corners=False)
                z_lr = ev.vae.encode(bic.to(dev, dt))
                anchor = ev.vae.decode(z_lr)
                cond = (z_lr, lq.to(dev, dt))
                z1 = ev.sampler.one_step(z_lr, cond, sigma=0.0)
                o1 = ev.vae.decode(z1)
            # Same forward-only workaround as Evaluator.infer_image (2026-08-30):
            # grad-enabled four_step + one dummy backward on the real output.
            z4 = ev.sampler.four_step(z_lr, cond, sigma=0.0)
            o4 = ev.vae.decode_with_grad(z4)
            o4.sum().backward()
            ev._zero_grads()
            o4 = o4.detach()
            d = base / f"{i:02d}_{c['sid']}"
            save_png(to_u8cpu(lq), d / "lq256.png")
            save_png(to_u8cpu(bic.float().cpu()), d / "bicubic4x.png")
            save_png(to_u8cpu(anchor.float().cpu()), d / "mage_anchor.png")
            save_png(to_u8cpu(o1.float().cpu()), d / "phase1p_s1.png")
            save_png(to_u8cpu(o4.float().cpu()), d / "phase1p_s4.png")
            c_rows.append({
                "lq": _down256(to_u8cpu(lq)), "bic": _down256(to_u8cpu(bic.float().cpu())),
                "anchor": _down256(to_u8cpu(anchor.float().cpu())),
                "s1": _down256(to_u8cpu(o1.float().cpu())), "s4": _down256(to_u8cpu(o4.float().cpu())),
            })
        if args.chunk >= 0:
            cf = base / f"chunk-{args.chunk}.jsonl"
            with open(cf, "w") as fh:
                for j, c in enumerate(sel):
                    rec = {
                        "set": "C", "idx": i_base + j, "sid": c["sid"],
                        "src": c["src"], "code_rev": git_rev,
                    }
                    fh.write(json.dumps(rec) + "\n")
            print(
                f"[rgb-eval] set C chunk {args.chunk}: {len(sel)}/{n_sel_total} items -> {cf.name} "
                f"(done in {time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            for k in range(0, len(c_rows), 10):
                rows = c_rows[k : k + 10]
                cells = [
                    {"lq": r["lq"], "bic": r["bic"], "anchor": r["anchor"], "s1": r["s1"], "s4": r["s4"]}
                    for r in rows
                ]
                sp = base / f"sheet_c_{k // 10:02d}.png"
                contact_sheet(cells, sp)
                sheets.append(str(sp))

    # ---------------- report ----------------
    if args.chunk >= 0:
        # process-isolated chunk: shared metrics.jsonl is built only by the
        # --sheets union pass; this chunk's records are in <SET>/chunk-<i>.jsonl.
        return
    (out / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m, default=str) for m in all_metrics)
    )
    (out / f"metrics-{args.set}.jsonl").write_text(
        "\n".join(json.dumps(m, default=str) for m in all_metrics)
    )
    _write_summary_report(out, ckpt_sha, git_rev, ev, sheets, t0)
    print(f"[rgb-eval] done in {time.time() - t0:.0f}s -> {out}", flush=True)


def _down256(a: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((256, 256), Image.LANCZOS))


def _write_sheets_a(base: Path, records: list[dict], sheets: list[str]) -> None:
    """Per-profile best/worst-8 sheets (by PSNR-Y of s1), 6-column rows."""
    per = {p: [r for r in records if r["profile"] == p] for p in PROFILES}
    for p in ("P0_clean", "P4_severe"):
        rows = per[p]
        if not rows:
            continue
        rows.sort(key=lambda r: r["metrics"]["psnr_y"]["phase1p_s1"], reverse=True)
        top = rows[:8]
        bottom = sorted(rows[-8:], key=lambda r: r["metrics"]["psnr_y"]["phase1p_s1"])
        for tag, sel in (("best", top), ("worst", bottom)):
            cells = []
            for r in sel:
                sid = r["sid"]
                d = base / f"{sid}__{p}"
                cells.append({
                    "gt": _down256(np.asarray(Image.open(d / "gt.png"))),
                    "lq": _down256(np.asarray(Image.open(d / "lq256.png").resize((256, 256), Image.BICUBIC))),
                    "bic": _down256(np.asarray(Image.open(d / "bicubic4x.png"))),
                    "anchor": _down256(np.asarray(Image.open(d / "mage_anchor.png"))),
                    "s1": _down256(np.asarray(Image.open(d / "phase1p_s1.png"))),
                    "s4": _down256(np.asarray(Image.open(d / "phase1p_s4.png"))),
                })
            sp = base / f"sheet_{tag}_{p}.png"
            contact_sheet(cells, sp)
            sheets.append(str(sp))


def _write_summary_report(out: Path, ckpt_sha: str, git_rev: str, ev, sheets: list[str], t0: float) -> None:
    mets: list[dict] = []
    p = out / "metrics.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                mets.append(json.loads(line))

    def agg(sub: list[dict], method: str) -> dict:
        if not sub:
            return {}
        vals = []
        for m in sub:
            for key in ("psnr_rgb", "psnr_y", "ssim_rgb", "ssim_y"):
                v = m["metrics"].get(key, {}).get(method)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append((key, v))
        flat = {}
        for key, v in vals:
            flat.setdefault(key, []).append(v)
        return {k: round(float(np.mean(v)), 4) for k, v in flat.items()}

    rep: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "path": CKPT, "sha256": ckpt_sha, "step": ev.ckpt_step,
            "git_rev": git_rev, "model_keys": ev.n_model_keys,
            "vae_fingerprint": ev.vae.fingerprint(),
        },
        "provenance": {
            "latest_sha_file": "provenance/latest.sha256",
            "git_tag": "p1formal-final-20260829",
            "manifest": "provenance/manifest-final-5a92ce1.sha256",
            "waivers": "provenance/waivers.md (CODE-DELTA: Pool shutdown compat)",
        },
        "eval_protocol": {
            "hr_crop": "p1formal pool rule: original webp -> crop_box(w, h, 1024, "
                       "box_seed(sid, 0, 0)) [seeded center-jitter, bit-exact]",
            "lq": "degrade_hr(hr_crop, one-hot profile cfg, global_seed=42, "
                  "data_cycle=0, exposure_index=0) -> 256^2",
            "exposure": list(EXPOSURE),
            "one_step": "sigma=0 deterministic (r0=0) = Faithful mode",
            "four_step": "Heun 3x + last-Euler, 7 evals = Experimental mode",
            "val_comparable": "Set A exposure (0,0) matches the training val "
                              "probe default exposure (anchor l1_1=0.3922 @18750)",
        },
        "disclosures": [
            (
                "LPIPS/DISTS not used: DTK env has no torchvision/scipy/lpips/dists; "
                "perceptual axis covered by flat-region HF energy + edge displacement. "
                "No new dependencies installed (user directive)."
            ),
            (
                "APISR / Real-CUGAN: not present on this machine; skipped per user "
                "directive (do not block on large new deps)."
            ),
            (
                "Set C source: danbooru-v2 original small-web images (400-1200px, "
                "real, not synthetic degradation); center-crop -> 256 LQ; human "
                "evaluation only, no paired metrics (no GT by design)."
            ),
            (
                "4-step marked Experimental; 1-step marked Faithful; 4-step is a "
                "Quality selling point only if RGB metrics beat 1-step (plan 16.4)."
            ),
        ],
    }
    for s in ("A", "B"):
        sub = [m for m in mets if m["set"] == s]
        if not sub:
            continue
        by_prof = {p: [m for m in sub if m["profile"] == p] for p in PROFILES}
        rep[f"set_{s}"] = {
            "n_pairs": len(sub),
            "aggregate_vs_gt": {mth: agg(sub, mth) for mth in
                                ("mage_anchor", "phase1p_s1", "phase1p_s4", "bicubic4x")},
            "per_profile": {
                p: {mth: agg(v, mth) for mth in ("mage_anchor", "phase1p_s1", "phase1p_s4", "bicubic4x")}
                for p, v in by_prof.items() if v
            },
            "failures_s1_psnry_bottom5": [
                {"sid": m["sid"], "profile": m["profile"],
                 "psnr_y_s1": m["metrics"]["psnr_y"]["phase1p_s1"]}
                for m in sorted(sub, key=lambda m: m["metrics"]["psnr_y"]["phase1p_s1"])[:5]
            ],
        }
    # 1-step vs 4-step paired comparison (per-metric delta s4 - s1)
    if mets:
        per_mth: dict[str, list[float]] = {}
        for m in mets:
            for mth in ("psnr_rgb", "psnr_y", "ssim_rgb", "ssim_y",
                        "edge_f1", "edge_disp_px", "linewidth_err_px"):
                a = m["metrics"][mth].get("phase1p_s1")
                b = m["metrics"][mth].get("phase1p_s4")
                if a is None or b is None:
                    continue
                if isinstance(a, float) and math.isnan(a):
                    continue
                per_mth.setdefault(mth, []).append(b - a)
        # "higher is better" metrics: psnr_*, ssim_*, edge_f1
        # "lower is better": edge_disp_px, linewidth_err_px
        higher_better = ("psnr_rgb", "psnr_y", "ssim_rgb", "ssim_y", "edge_f1")
        stats = {}
        for mth, ds in per_mth.items():
            mean_d = float(np.mean(ds))
            if mth in higher_better:
                wins = sum(1 for d in ds if d > 0.0005)
                losses = sum(1 for d in ds if d < -0.0005)
            else:
                wins = sum(1 for d in ds if d < -0.0005)
                losses = sum(1 for d in ds if d > 0.0005)
            stats[mth] = {
                "n": len(ds), "mean_delta_s4_minus_s1": round(mean_d, 5),
                "s4_wins": wins, "s4_losses": losses, "ties": len(ds) - wins - losses,
            }
        # aggregate verdict on quality-relevant metrics (higher-better only)
        hb = [d for mth in higher_better for d in per_mth.get(mth, [])]
        verdict = (
            "4-step (Experimental) beats 1-step (Faithful) on aggregate "
            "RGB quality -> Quality claim MAY proceed to user decision"
            if (hb and float(np.mean(hb)) > 0.002)
            else "4-step does NOT beat 1-step on aggregate RGB metrics -> "
                 "per plan 16.4, 4-step stays Experimental; 1-step is the "
                 "Faithful deliverable"
        )
        rep["one_vs_four_step"] = {
            "per_metric": stats,
            "mean_delta_higher_better": round(float(np.mean(hb)), 5) if hb else None,
            "verdict": verdict,
        }
    c_dir = out / "C"
    rep["set_C"] = {
        "note": "outputs only (human eval); see C/sheet_c_*.png + C/<n>_<sid>/ PNGs",
        "n": len([d for d in c_dir.iterdir() if d.is_dir() and not d.name.startswith("sheet")]) if c_dir.exists() else 0,
    }
    rep["sheets"] = sheets
    rep["wall_seconds"] = round(time.time() - t0, 1)
    (out / "rgb-eval-report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[rgb-eval] report -> {out / 'rgb-eval-report.json'}", flush=True)


if __name__ == "__main__":
    main()
