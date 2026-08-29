"""HCU micro-benchmark for p1formal onfly smoke wedge bisection.

Runs single-process (no DDP, no pool) on cuda:0 and times each suspect
stage of the onfly consumer step:
  1. VAE load (HCU init)
  2. VAE encode HR 1024^2 (onfly z_hr path)
  3. VAE encode LQ-up 1024^2 (z_lr path, bicubic 256->1024)
  4. AnimeSRModel (trunk+pixel) forward at 1024 latent (pixel path)
  5. backward + optimizer step (zero-init pixel family included)

Each stage prints a wall-clock marker. A wedged stage never prints its
marker. Exit 0 = all stages completed.
"""
import sys, time

sys.path.insert(0, "/root/anime-sr-p1formal/src")
import torch
import torch.nn.functional as F

from anime_sr.config.loader import load_config
from anime_sr.model.uflow import (
    AnimeSRModel,
    apply_pixel_zero_init,
    count_parameters,
)
from anime_sr.train.latent_flow import build_flow_targets
from anime_sr.vae.mage import load_frozen_vae

VAE = "/root/private_data/anime-sr/model/vae/mage-vae.safetensors"
CFG = [
    "/root/anime-sr-p1formal/config/base.toml",
    "/root/anime-sr-p1formal/config/data.toml",
    "/root/anime-sr-p1formal/config/phase1-small.toml",
    "/root/anime-sr-p1formal/config/phase1-pi-formal.toml",
]
dev = torch.device("cuda:0")
dtype = torch.bfloat16
t0 = time.time()


def mark(name):
    torch.cuda.synchronize(dev)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: +{time.time()-t0:7.2f}s (cum)", flush=True)


print(f"[{time.strftime('%H:%M:%S')}] start (OMP={__import__('os').environ.get('OMP_NUM_THREADS','unset')})", flush=True)

cfg = load_config(*CFG)
print(f"[{time.strftime('%H:%M:%S')}] config loaded (pixel_features={cfg.latent_flow.pixel_features}, zhr={cfg.latent_flow.zhr_source})", flush=True)

vae = load_frozen_vae(VAE, dev, dtype=dtype)
mark("1. vae-load")

hr = torch.randint(0, 256, (1, 3, 1024, 1024), dtype=torch.uint8).to(dev)
z_hr = vae.encode(hr.to(dtype))
print(f"  z_hr shape={tuple(z_hr.shape)}", flush=True)
mark("2. vae-encode-hr-1024 (onfly z_hr)")

lq = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8).float().to(dev)
lq_up = F.interpolate(lq, size=(1024, 1024), mode="bicubic")
z_lr = vae.encode(lq_up.to(dtype))
print(f"  z_lr shape={tuple(z_lr.shape)}", flush=True)
mark("3. vae-encode-lq-up-1024 (z_lr)")

model = AnimeSRModel(cfg.model, zero_init_pixel=cfg.model.zero_init_pixel).to(dev, dtype=dtype)
model.requires_grad_(True)
print(f"  model {count_parameters(model)} params (zero_init_pixel={cfg.model.zero_init_pixel})", flush=True)
mark("4. model-build+zero-init")

rt, v_star, sigma, _t = build_flow_targets(z_hr, z_lr, cfg, device=dev)
lq_b = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8).to(dev).to(dtype)
v_hat = model(rt, z_lr, lq_b, _t, sigma)
loss = F.mse_loss(v_hat.float(), v_star.float())
print(f"  loss={loss.item():.4f} finite={torch.isfinite(loss).item()}", flush=True)
mark("5. forward (pixel path)")

model.zero_grad()
loss.backward()
gsum = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
nzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() == 0)
print(f"  grad abs-sum={gsum:.4e}, zero-grad params={nzero}", flush=True)
mark("6. backward")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
opt.step()
mark("7. optimizer-step")

# second pass: steady-state timing (no first-use autotune)
t_s = time.time()
for i in range(5):
    model.zero_grad()
    rt2, v2, s2, t2 = build_flow_targets(z_hr, z_lr, cfg, device=dev)
    v_hat2 = model(rt2, z_lr, lq_b, t2, s2)
    F.mse_loss(v_hat2.float(), v2.float()).backward()
    opt.step()
torch.cuda.synchronize(dev)
print(f"  steady-state fwd+bwd+step: {(time.time()-t_s)/5*1000:.1f} ms/step", flush=True)
mark("8. steady-state x5")
print("MICROBENCH-PASS", flush=True)
