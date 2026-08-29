#!/usr/bin/env python3
"""Gradient-coverage probe for the p1formal final checkpoint (Phase I-P-1024).

Read-only evidence probe: builds AnimeSRModel exactly as training does
(same config load + zero_init_pixel flag), loads the final checkpoint
weights, runs ONE synthetic forward+backward on CPU, and reports how many
parameters received a non-zero gradient. No training code is modified.

Usage:
  PYTHONPATH=/root/anime-sr-p1formal/src python3 grad_coverage_probe.py <ckpt.pt>
"""
import sys
import torch

CKPT = sys.argv[1]
CONFIGS = (
    "/root/anime-sr-p1formal/config/base.toml",
    "/root/anime-sr-p1formal/config/data.toml",
    "/root/anime-sr-p1formal/config/phase1-small.toml",
    "/root/anime-sr-p1formal/config/phase1-pi-formal.toml",
)


def main() -> None:
    from anime_sr.config.loader import load_config
    from anime_sr.model.uflow import AnimeSRModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    cfg = load_config(*CONFIGS)
    model = AnimeSRModel(cfg.model, zero_init_pixel=cfg.model.zero_init_pixel)
    d = torch.load(CKPT, map_location="cpu")
    sd = d["model"] if isinstance(d, dict) and "model" in d else d
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt step={d.get('step') if isinstance(d, dict) else '?'} "
          f"keys={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(dev)

    B, H, W = 1, 128, 128  # z grid for 1024px HR (VAE downsample 8x)
    r_t = torch.randn(B, 128, H, W, device=dev)
    z_lr = torch.randn(B, 128, H, W, device=dev)
    lq = torch.randn(B, 3, 4 * H, 4 * W, device=dev) * 0.1
    t = torch.tensor([0.5], device=dev)
    sigma = torch.tensor([0.1], device=dev)
    out = model(r_t, z_lr, lq, t, sigma)
    out.abs().sum().backward()

    all_p = [(n, p) for n, p in model.named_parameters()]
    alive = [(n, p) for n, p in all_p if p.grad is not None and p.grad.abs().sum().item() > 0]
    n_total = len(all_p)
    n_alive = len(alive)
    print(f"grad coverage: {n_alive}/{n_total} params with non-zero grad")

    def show(name: str) -> None:
        for n, p in all_p:
            if name in n:
                ok = p.grad is not None and p.grad.abs().sum().item() > 0
                gl = p.grad.abs().sum().item() if p.grad is not None else 0.0
                print(f"  {'OK  ' if ok else 'ZERO'} {n} |grad|_1={gl:.6e}")

    print("=== transition + pixel weights ===")
    for suf in ("proj_p64.weight", "proj_p32.weight", "proj_p16.weight", "gap_proj.weight"):
        show(suf)
    pix = [n for n, _ in all_p if "pixel" in n]
    pix_alive = sum(1 for n, p in all_p if "pixel" in n and p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"  pixel_encoder range: {pix_alive}/{len(pix)} with non-zero grad")

    zeros = [n for n, p in all_p if p.grad is None or p.grad.abs().sum().item() == 0]
    print(f"zero-grad params ({len(zeros)}):")
    for n in zeros:
        print(f"  ZERO {n}")

    verdict = n_alive == n_total or (n_alive >= n_total - 10)
    print(f"GRAD-COVERAGE VERDICT: {'PASS' if verdict else 'REVIEW (check zero-grad list)'} "
          f"({n_alive}/{n_total})")
    sys.exit(0 if verdict else 2)


if __name__ == "__main__":
    main()
