"""M3 checklist probe (plan §13): visual + structural canary for a latent-flow
checkpoint.

Covers the smoke.toml checklist items a bare training log cannot show:

- #5/#9 window mask / no seams: 1-step and 4-step outputs are decoded through
  the frozen VAE and saved next to the HR crop; the seam probe measures the
  128-px-aligned (window-8 on the 64x latent grid) grid-line discontinuity in
  the decoded output.
- #8 decoder gradient reaches the trunk: one L1 loss through
  ``vae.decode_with_grad(z_hat)`` must put non-zero gradient on the trunk
  (the Stage-II GAN path plumbing, plan §12.6).

Run after the smoke reaches its first validation (step >= val_every_steps):

    python -m anime_sr.cli.m3_probe \
        --config config/base.toml config/data.toml config/smoke.toml \
        --ckpt output_model/latent-flow-smoke/step-005000.pt \
        --latent-dir /root/private_data/anime-sr/latents-10k-1024 \
        --index-dir /root/private_data/anime-sr/data/index \
        --webp-dir /root/private_data/anime-sr/data/webp \
        --vae /root/private_data/anime-sr/model/vae/mage-vae.safetensors \
        --out-dir output_model/latent-flow-smoke/probe-step-005000 \
        --bucket-hr 1024 --n 4

The checkpoint must come from the single-card smoke run (the DDP-wrapped
payload uses "module."-prefixed keys and is not accepted here).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anime_sr.config.loader import load_config
from anime_sr.data.degradation import degrade_hr
from anime_sr.data.latent_store import LatentStore, read_index
from anime_sr.data.pipeline import SRDataset
from anime_sr.flow.sampling import FlowSampler
from anime_sr.model.uflow import UFlowSR
from anime_sr.train.latent_flow import _LatentVelocity
from anime_sr.vae.mage import load_frozen_vae


def _save_img(t: torch.Tensor, path: Path) -> None:
    """(C, H, W) in [-1, 1] -> PNG (imageio) or .npy fallback."""
    arr = ((t.detach().cpu().float().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    arr = arr.permute(1, 2, 0).numpy()
    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), arr)
    except Exception:  # noqa: BLE001 - a probe must never die on imaging libs
        np.save(str(path.with_suffix(".npy")), arr)
        print(f"[probe] {path.name}: imageio unavailable, saved .npy instead")


def _grid_seam(pix: torch.Tensor, ref: torch.Tensor) -> float:
    """#9: mean abs discontinuity across 128-px-aligned vertical lines in
    ``pix - ref``, vs an interior baseline. window-8 on the 64x grid gives a
    128-px period at 16x downscale; a ratio > ~1.5x flags window seams."""
    d = (pix - ref).abs().float()  # (C, H, W)
    w = d.shape[2]
    lines = [d[:, :, x].mean() for x in range(128, w - 128, 128)]
    inter = [
        d[:, y : y + 64, x : x + 64].mean()
        for y in (96, 320, 576)
        for x in (96, 320, 576)
    ]
    line_e = sum(lines) / max(1, len(lines))
    inter_e = sum(inter) / max(1, len(inter))
    return float(line_e / max(1e-9, inter_e))


def main() -> int:
    ap = argparse.ArgumentParser(description="M3 checklist probe")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = load_config(*a.config)
    device = torch.device(a.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # model (smoke dims come from the config overlay — must match the ckpt)
    model = UFlowSR(cfg.model.uflow, cfg.model.output_head)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device, dtype).eval()
    print(
        f"[probe] loaded {a.ckpt} ({state.get('step', '?')} steps) "
        f"-> {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"
    )

    # The trunk's TimestepEmbedder computes sinusoidal features in fp32, so
    # every trunk forward needs the train loop's autocast to downcast into
    # the bf16 weights — mirroring latent_flow._validate_latent.
    ac = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (dtype == torch.bfloat16 and device.type == "cuda")
        else (lambda: nullcontext())
    )

    vae = load_frozen_vae(a.vae, device, dtype)
    store = LatentStore(Path(a.latent_dir), a.bucket_hr)
    doc = read_index(Path(a.latent_dir))
    sids = sorted(doc["samples"].keys())
    ds = SRDataset(a.index_dir, a.webp_dir, cfg, bucket_hr=a.bucket_hr, split="train")
    sid_to_i = {m.sample_id: i for i, m in enumerate(ds.samples)}
    picks = [s for s in sids[: a.n] if s in sid_to_i]
    if not picks:
        raise RuntimeError("no probe samples: latent store and train index disjoint")

    def _prep(sid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_hr, z_lr, hr_crop) for one probe sample: pinned (0,0) crop box +
        the fixed probe exposure (dcyc=0, eidx=24) — the trainer's _fetch
        pattern."""
        meta = ds.samples[sid_to_i[sid]]
        hr_full = ds.decode_hr(meta)
        x, y = ds.crop(meta, 0, 0)
        hr_crop = hr_full[..., y : y + a.bucket_hr, x : x + a.bucket_hr].contiguous()
        # store/dataset tensors are unbatched ([128,h,w] / [3,B,B]); the
        # sampler and frozen VAE take a batch dim, so keep everything 4D.
        # decode_hr / degrade_hr stay on CPU; the frozen VAE and the model
        # live on ``device``, so all tensors fed to them move there.
        z_hr = store.read(sid).unsqueeze(0).to(device)
        lq, _ = degrade_hr(
            hr_crop,
            cfg,
            global_seed=ds.global_seed,
            sample_id=sid,
            data_cycle=0,
            exposure_index=24,
        )
        z_lr = vae.encode(
            F.interpolate(
                lq.float().unsqueeze(0),
                size=(a.bucket_hr, a.bucket_hr),
                mode="bicubic",
            ).to(vae.device, vae.dtype)
        )
        return z_hr, z_lr, hr_crop.to(device)

    sampler = FlowSampler(_LatentVelocity(model))
    with torch.no_grad():
        for k, sid in enumerate(picks):
            z_hr, z_lr, hr_crop = _prep(sid)
            g = torch.Generator(device=str(device)).manual_seed(0x5EED ^ k)
            with ac():
                z1 = sampler.one_step(z_lr, z_lr, sigma=0.0, generator=g)
                z4 = sampler.four_step(z_lr, z_lr, sigma=0.0, generator=g)
            l1_1 = (z1 - z_hr).abs().mean().item()
            l1_4 = (z4 - z_hr).abs().mean().item()
            print(f"[probe] {sid} l1_1={l1_1:.4f} l1_4={l1_4:.4f}")

            pix_hr = hr_crop.unsqueeze(0)  # (1, 3, B, B) in [-1, 1]
            pix1 = vae.decode(z1.to(vae.dtype))
            pix4 = vae.decode(z4.to(vae.dtype))
            seam1 = _grid_seam(pix1[0], pix_hr[0])
            seam4 = _grid_seam(pix4[0], pix_hr[0])
            print(f"[probe] {sid} seam1={seam1:.2f}x seam4={seam4:.2f}x (<=1.5 OK)")
            _save_img(pix_hr[0], out / f"{sid}-hr.png")
            _save_img(pix1[0], out / f"{sid}-1step.png")
            _save_img(pix4[0], out / f"{sid}-4step.png")

    # #8: decoder gradient must reach the trunk (Stage-II plumbing), on the
    # first probe sample, with a fresh graph. The forward AND the backward
    # run under the same autocast as the train loop.
    _, z_lr, hr_crop = _prep(picks[0])
    rt = torch.zeros_like(z_lr)  # sigma=0 => r_0 = 0
    t0 = torch.zeros(z_lr.shape[0], device=device, dtype=dtype)
    sg = torch.zeros(z_lr.shape[0], device=device, dtype=dtype)
    with ac():
        v_hat = model(rt, z_lr, t0, sg)
        z_hat = z_lr + v_hat  # Euler 1-step at t=0
        pix_hat = vae.decode_with_grad(z_hat.to(vae.dtype))
        (pix_hat - hr_crop.unsqueeze(0)).abs().mean().backward()
    gnorms = [
        p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None
    ]
    non_zero = sum(1 for x in gnorms if x > 0)
    print(
        f"[probe] decoder-grad: {non_zero}/{len(gnorms)} trunk params have grad, "
        f"max={max(gnorms) if gnorms else 0:.3e} {'OK' if non_zero > 0 else 'FAIL'}"
    )
    print(f"[probe] wrote {len(picks)} x 3 images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
