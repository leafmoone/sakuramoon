"""Pure-eval ODE solver sweep (revised plan §13 #3 / small-Phase-I pre-gate).

Runs a fixed grid of solvers (Euler 1/2/4/8, Heun 2/4/8 — Heun in the
project's last-euler quality mode, so ``heun 4`` == the trainer's 4-step) on
a fixed validation slice of pre-encoded samples with a fixed seed, and
records per solver:

* ``l1`` / ``l1_anchor`` / ``toward_frac`` plus the ratio vs the 1-step
  baseline (revised #3: ``l1_4/l1_1 <= 1.05``, 1-step significantly better
  than the anchor, 4-step numerically stable)
* ``cos_v`` at every sub-step start (v* = delta, r0 = 0)
* per-sub-step state norm and velocity norm
* trajectory deviation ``D_t = mean|r_hat(t_k) - t_k*delta|``
* endpoint consistency (solver-independent, r0 = 0):
  ``delta_hat_t = r_t + (1-t) v_theta(r_t, t)``, endpoint L1 at
  t in {0, .25, .5, .75}
* on-path / off-path velocity error at the same probe grid (off-path adds a
  fixed-seed perturbation of ``0.1 * ||delta||`` per sample — the Heun
  corrector's evaluation neighborhood)

No training; single card. Writes a full JSON evidence bundle to
``--out-dir/sweep-results.json``. Usage:

    python -m anime_sr.cli.solver_sweep \
        --config config/base.toml config/data.toml \
        --ckpt output_model/latent-flow-smoke/step-025000.pt \
        --latent-dir /root/private_data/anime-sr/latents-10k-1024 \
        --index-dir /root/private_data/anime-sr/data/index \
        --webp-dir /root/private_data/anime-sr/data/webp \
        --vae /root/private_data/anime-sr/model/vae/mage-vae.safetensors \
        --out-dir output_model/latent-flow-smoke/sweep-step-025000 \
        --bucket-hr 1024
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from anime_sr.config.loader import load_config
from anime_sr.data.degradation import degrade_hr
from anime_sr.data.latent_store import LatentStore, read_index
from anime_sr.data.pipeline import SRDataset
from anime_sr.flow.path import sample_source_noise
from anime_sr.flow.solver import euler_trajectory, heun_trajectory
from anime_sr.model.uflow import UFlowSR
from anime_sr.train.latent_flow import (
    _LatentVelocity,
    latent_val_metrics,
    velocity_cosine,
)
from anime_sr.vae.mage import load_frozen_vae

#: Sweep grid: (solver, n_steps). Heun runs in the last-euler quality mode
#: (heun 4 == four_step_heun == the trainer's val 4-step, 7 evaluations).
_SOLVERS: tuple[tuple[str, int], ...] = (
    ("euler", 1),
    ("euler", 2),
    ("euler", 4),
    ("euler", 8),
    ("heun", 2),
    ("heun", 4),
    ("heun", 8),
)
_ENDPOINT_TS = (0.0, 0.25, 0.5, 0.75)
_ENDPOINT_LABELS = {0.0: "t0", 0.25: "t25", 0.5: "t50", 0.75: "t75"}
#: Off-path velocity probe: perturbation = 0.1 * per-sample ||delta||_L2.
_OFFPATH_PERTURB = 0.1


def _l2_norm(
    x: torch.Tensor, dims: tuple[int, ...], keepdim: bool = False
) -> torch.Tensor:
    """Per-row L2 norm over `dims`.

    Portable power-sum form: some torch builds (e.g. DTK 2.9.0) route
    ``Tensor.norm(p=2, dim=<tuple>)`` to ``linalg.matrix_norm``, which
    rejects dim tuples longer than 2.
    """
    return x.float().pow(2).sum(dim=dims, keepdim=keepdim).sqrt()


def _load_model(
    ckpt: str, cfg: Any, device: torch.device, dtype: torch.dtype
) -> tuple[UFlowSR, Any]:
    model = UFlowSR(cfg.model.uflow, cfg.model.output_head)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["model"]
    if any(str(k).startswith("module.") for k in sd):
        # DDP-wrapped payload: strip the wrapper prefix (the trainer saves
        # the unwrapped dict, so this only matters for foreign checkpoints).
        print("[sweep] note: stripping 'module.' prefix from DDP checkpoint keys")
        sd = {str(k)[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device, dtype).eval()
    return model, state.get("step", "?")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pure-eval ODE solver sweep (revised §13 #3)")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="val slice size (default [latent_flow].val_samples)",
    )
    ap.add_argument("--seed", type=int, default=0x5EED)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = load_config(*a.config)
    device = torch.device(a.device)
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and cfg.hardware.dtype == "bf16")
        else torch.float32
    )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt_step = _load_model(a.ckpt, cfg, device, dtype)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    # The trunk's TimestepEmbedder computes its sinusoidal features in fp32,
    # so every forward needs the train loop's autocast (mirrors m3_probe /
    # _validate_latent).
    ac: Callable[[], Any] = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if dtype == torch.bfloat16 and device.type == "cuda"
        else (lambda: nullcontext())
    )
    print(
        f"[sweep] loaded {a.ckpt} (step {ckpt_step}) "
        f"-> {n_p:.2f}M params, device={device} dtype={dtype}"
    )

    vae = load_frozen_vae(a.vae, device, dtype)
    store = LatentStore(Path(a.latent_dir), a.bucket_hr)
    doc = read_index(Path(a.latent_dir))
    sids = sorted(doc["samples"].keys())
    ds = SRDataset(a.index_dir, a.webp_dir, cfg, bucket_hr=a.bucket_hr, split="train")
    sid_to_i = {m.sample_id: i for i, m in enumerate(ds.samples)}
    n = a.n if a.n is not None else cfg.latent_flow.val_samples
    picks = [s for s in sids[:n] if s in sid_to_i]
    if not picks:
        raise RuntimeError("no sweep samples: latent store and train index disjoint")

    def _prep(sid: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(z_hr, z_lr) for one sweep sample: pinned (0,0) crop box + the
        fixed probe exposure (dcyc=0, eidx=24) — the trainer's _fetch pattern."""
        meta = ds.samples[sid_to_i[sid]]
        hr_full = ds.decode_hr(meta)
        x, y = ds.crop(meta, 0, 0)
        hr_crop = hr_full[..., y : y + a.bucket_hr, x : x + a.bucket_hr].contiguous()
        # store.read returns (C, g, g) — keep it 3D so torch.stack yields the
        # (B, C, g, g) batch the model expects (an extra per-sample dim 1
        # would produce a 5D tensor and crash the input convs).
        z_hr = store.read(sid).to(device, dtype)
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
        ).squeeze(0)  # -> (C, g, g) to match z_hr; stack then yields 4D (B, C, g, g)
        return z_hr, z_lr

    z_hrs: list[torch.Tensor] = []
    z_lrs: list[torch.Tensor] = []
    for sid in picks:
        zh, zl = _prep(sid)
        z_hrs.append(zh)
        z_lrs.append(zl)
    z_hr_b = torch.stack(z_hrs).to(dtype)
    z_lr_b = torch.stack(z_lrs).to(dtype)
    b = z_hr_b.shape[0]
    delta = z_hr_b - z_lr_b  # the exact (constant-in-t) target field, r0 = 0
    sigma_b = torch.zeros(b, device=device, dtype=torch.float32)
    gen = torch.Generator(device=str(device)).manual_seed(a.seed)
    r0 = sample_source_noise(
        sigma_b, z_lr_b.shape, generator=gen, dtype=dtype, device=device
    )  # sigma = 0 -> r0 = 0 (Faithful mode)

    adapter = _LatentVelocity(model)

    def v_fn(r: torch.Tensor, t: float) -> torch.Tensor:
        t_vec = torch.full((b,), t, device=device, dtype=torch.float32)
        with ac():
            return adapter(r, t_vec, sigma_b, z_lr_b)

    results: dict[str, Any] = {
        "ckpt": str(a.ckpt),
        "ckpt_step": ckpt_step,
        "n_params_m": round(n_p, 2),
        "n_val": b,
        "sids": picks,
        "seed": a.seed,
        "bucket_hr": a.bucket_hr,
        "device": str(device),
        "dtype": str(dtype),
    }

    with torch.no_grad():
        # --- solver-independent probes (computed once) -------------------
        ep: dict[str, float] = {}
        on_path: dict[str, float] = {}
        off_path: dict[str, float] = {}
        perturb = (_OFFPATH_PERTURB * _l2_norm(delta, (1, 2, 3), keepdim=True)).to(dtype)
        gen_off = torch.Generator(device=str(device)).manual_seed(a.seed ^ 0x3C5E)
        for t in _ENDPOINT_TS:
            r_t = t * delta  # exact path with r0 = 0
            t_vec = torch.full((b,), t, device=device, dtype=torch.float32)
            with ac():
                v_t = adapter(r_t, t_vec, sigma_b, z_lr_b)
            delta_hat = r_t + (1.0 - t) * v_t
            lab = _ENDPOINT_LABELS[t]
            ep[f"ep_l1_{lab}"] = (delta_hat - delta).abs().float().mean().item()
            on_path[lab] = (v_t - delta).abs().float().mean().item()
            # torch.randn_like(generator=...) is unsupported in some builds
            # (e.g. DTK 2.9.0); generate into an explicit shape instead.
            noise = torch.randn(r_t.shape, generator=gen_off, device=r_t.device, dtype=r_t.dtype)
            r_off = r_t + perturb * noise
            with ac():
                v_off = adapter(r_off, t_vec, sigma_b, z_lr_b)
            off_path[lab] = (v_off - delta).abs().float().mean().item()
        results["endpoint_l1"] = ep
        results["on_path_vel_err"] = on_path
        results["off_path_vel_err"] = off_path

        # --- per-solver runs --------------------------------------------
        per_solver: list[dict[str, Any]] = []
        for solver, n_steps in _SOLVERS:
            if solver == "euler":
                r_final, states = euler_trajectory(r0, v_fn, n_steps)
                n_evals = n_steps
            else:
                r_final, states = heun_trajectory(r0, v_fn, n_steps, last_euler=True)
                n_evals = 2 * (n_steps - 1) + 1
            z_hat = z_lr_b + r_final
            m = latent_val_metrics(z_hr_b, z_lr_b, z_hat)
            per_sample_l1 = (z_hat - z_hr_b).abs().float().mean(dim=(1, 2, 3)).tolist()
            # per-sub-step-start probes: state before sub-step k
            cos_per_t: dict[str, float] = {}
            vel_norm: dict[str, float] = {}
            state_norm: dict[str, float] = {}
            d_t: dict[str, float] = {}
            r_probe = r0
            for k in range(n_steps):
                t_k = k / n_steps
                with ac():
                    v_k = adapter(
                        r_probe,
                        torch.full((b,), t_k, device=device, dtype=torch.float32),
                        sigma_b,
                        z_lr_b,
                    )
                lab = f"t{round(t_k * 100):03d}"
                cos_per_t[lab] = velocity_cosine(v_k, delta)
                vel_norm[lab] = _l2_norm(v_k, (1, 2, 3)).mean().item()
                r_probe = states[k]  # state after sub-step k
            for k, r_k in enumerate(states):
                t_k = (k + 1) / n_steps
                lab = f"t{round(t_k * 100):03d}"
                state_norm[lab] = _l2_norm(r_k, (1, 2, 3)).mean().item()
                d_t[lab] = (r_k - t_k * delta).abs().float().mean().item()
            finite = bool(
                torch.isfinite(r_final).all().item()
                and all(torch.isfinite(s).all().item() for s in states)
            )
            per_solver.append(
                {
                    "solver": solver,
                    "n_steps": n_steps,
                    "n_evals": n_evals,
                    "l1": m["l1"],
                    "l1_anchor": m["l1_anchor"],
                    "toward_frac": m["toward_frac"],
                    "ratio_vs_1step": None,  # filled below vs the euler-1 baseline
                    "cos_v": cos_per_t,
                    "state_norm": state_norm,
                    "vel_norm": vel_norm,
                    "D_t": d_t,
                    "per_sample_l1": per_sample_l1,
                    "finite": finite,
                }
            )
        baseline = per_solver[0]["l1"]  # euler 1-step (grid is ordered)
        for e in per_solver:
            e["ratio_vs_1step"] = None if e is per_solver[0] else e["l1"] / max(1e-9, baseline)
        results["solvers"] = per_solver

    # --- summary + gate check (revised #3) --------------------------------
    base = per_solver[0]
    h4 = next(e for e in per_solver if e["solver"] == "heun" and e["n_steps"] == 4)
    print(f"[sweep] n={b} seed={a.seed:#x} l1_anchor={base['l1_anchor']:.4f} "
          f"l1_1={base['l1']:.4f} toward_1={base['toward_frac']:.2f}")
    print("[sweep] endpoint_l1 " + " ".join(f"{k.replace('ep_l1_', '')}={v:.4f}" for k, v in ep.items()))
    print("[sweep] on_path_vel  " + " ".join(f"{k}={v:.4f}" for k, v in on_path.items()))
    print("[sweep] off_path_vel " + " ".join(f"{k}={v:.4f}" for k, v in off_path.items()))
    for e in per_solver:
        ratio = "-" if e["ratio_vs_1step"] is None else f"{e['ratio_vs_1step']:.3f}x"
        cos = " ".join(f"{k}={v:.3f}" for k, v in e["cos_v"].items())
        dt = " ".join(f"{k}={v:.4f}" for k, v in e["D_t"].items())
        print(
            f"[sweep] {e['solver']}{e['n_steps']} ({e['n_evals']} evals): "
            f"l1={e['l1']:.4f} ratio={ratio} toward={e['toward_frac']:.2f} "
            f"cos[{cos}] D[{dt}] finite={e['finite']}"
        )
    all_finite = all(e["finite"] for e in per_solver)
    beat = base["l1"] < base["l1_anchor"]
    ratio_ok = h4["ratio_vs_1step"] is not None and h4["ratio_vs_1step"] <= 1.05
    print(
        f"[sweep] gate(revised #3): 1-step beats anchor? {beat} | "
        f"heun4/1step = {h4['ratio_vs_1step']:.4f} <= 1.05? {ratio_ok} | "
        f"no NaN/Inf? {all_finite}"
    )
    results["gate"] = {
        "one_step_beats_anchor": bool(beat),
        "heun4_ratio": h4["ratio_vs_1step"],
        "heun4_ratio_ok": bool(ratio_ok),
        "all_finite": all_finite,
    }
    (out / "sweep-results.json").write_text(json.dumps(results, indent=2))
    print(f"[sweep] results -> {out / 'sweep-results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
