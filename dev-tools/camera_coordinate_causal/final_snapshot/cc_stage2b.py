"""Stage 2b (OPTIONAL, spec s30) - coordinate-leaf gradient probe.

128 camera units, one stratum (median, k=2), CORRECT arm. Model parameters are
frozen (requires_grad=False); only the image-coordinate tensor is a leaf with
requires_grad=True. Records RMS of d(loss)/d(coordinate) per checkpoint.

SKIP (recorded, not a failure) if the RoPE/attention path lacks autograd support
(e.g. flash-attention backward unavailable): the spec anticipates this.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_prev = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", ""
)
_opts = [o.strip() for o in _prev.split(",") if o.strip() and not o.strip().startswith("expandable_segments:")]
if not any(o.startswith("max_split_size_mb:") for o in _opts):
    _opts.append("max_split_size_mb:512")
_opts.append("expandable_segments:True")
os.environ["PYTORCH_ALLOC_CONF"] = ",".join(_opts)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]

import torch  # noqa: E402

import cc_common as cc  # noqa: E402

N_PROBE = 128
STRATUM = 2  # median stratum


def log(msg: str) -> None:
    print(f"[stage2b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    from typing import cast

    from sakuramoon.checkpoint.load import load_inference_artifact, read_checkpoint_manifest
    from sakuramoon.cli.generation_eval import _resolve_alpha  # noqa: N812
    from sakuramoon.objective.flow import flow_matching_loss
    from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs

    device = torch.device("cuda", 0)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("stage2b requires exactly one visible CUDA device")

    with open(cc.OUT / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    meta = {m["unit"]: m for m in manifest["units"]}
    probe_units = [u for u in sorted(meta) if meta[u]["cohort"] == "camera"][:N_PROBE]
    log(f"probe units: {len(probe_units)} (stratum k={STRATUM}, t={manifest['seeds']['t_values'][STRATUM]:.6f})")

    out: dict = {"skipped": None, "per_ckpt": {}}
    for ck_name in ("PRE", "MID", "POST"):
        ckpt = cc.CKPTS[ck_name]
        log(f"=== {ck_name} ===")
        ckpt_manifest = read_checkpoint_manifest(ckpt)
        alpha = _resolve_alpha(ckpt, None)
        composite = load_inference_artifact(ckpt, ckpt_manifest.identity, device=device)
        composite = cast(TrainableComposite, composite)
        composite.eval()
        composite.requires_grad_(False)

        rms_vals: list[float] = []
        skipped_reason = None
        try:
            for n, u in enumerate(probe_units):
                b = torch.load(cc.OUT / "units" / f"unit-{u:05d}.pt", map_location="cpu", weights_only=False)
                x0 = b["x0"].to(device)
                state_k = b["states"][STRATUM].to(device).squeeze(0)  # 3D [C,H,W] — PackedDiT latents contract (v2)
                t_k = torch.tensor([b["t"][STRATUM]], device=device, dtype=torch.float32)
                coord = b["arms"]["CORRECT"].to(device, dtype=torch.float32).clone()
                coord.requires_grad_(True)
                inputs = TrainableCompositeInputs(
                    qwen_states=b["qwen_states"].to(device),
                    main_token_indices=b["main_token_indices"].to(device),
                    main_mask=b["main_mask"].to(device),
                    main_token_lengths=b["main_token_lengths"],
                    condition_token_indices=b["condition_token_indices"].to(device),
                    condition_mask=b["condition_mask"].to(device),
                    use_null_condition=b["use_null_condition"].to(device),
                    active_condition_sample_indices=b["active_condition_sample_indices"].to(device),
                    latents=(state_k,),
                    image_coordinates=(coord,),
                    timestep=t_k,
                    size_scale=torch.tensor([b["size_scale"]], device=device, dtype=torch.float32),
                    aspect=torch.tensor([b["aspect"]], device=device, dtype=torch.float32),
                    growth_alpha=alpha,
                )
                conditioning = composite.forward_conditioning(inputs)
                pred = composite.forward_dit(inputs, conditioning)[0]
                loss = flow_matching_loss(
                    pred.unsqueeze(0), state_k.unsqueeze(0), x0, t_k,
                    t_eps=cc.T_EPS,
                    noise_observation_boundary=cc.NOISE_OBSERVATION_BOUNDARY,
                ).per_sample[0]
                loss.backward()
                if coord.grad is None:
                    skipped_reason = "coordinate leaf received no gradient (RoPE/FA2 path lacks autograd support)"
                    break
                rms_vals.append(float(coord.grad.pow(2).mean().sqrt()))
                if n % 32 == 0:
                    log(f"  {ck_name} {n}/{len(probe_units)}")
        except Exception as e:  # noqa: BLE001
            skipped_reason = f"backward failed: {type(e).__name__}: {e}"
            log(f"  {ck_name} SKIP: {skipped_reason}")

        if skipped_reason is None:
            out["per_ckpt"][ck_name] = {
                "n": len(rms_vals),
                "grad_rms_mean": sum(rms_vals) / len(rms_vals),
                "grad_rms_max": max(rms_vals),
                "grad_rms_min": min(rms_vals),
            }
        else:
            out["skipped"] = skipped_reason
            out["per_ckpt"][ck_name] = {"skipped": skipped_reason}
        del composite
        torch.cuda.empty_cache()

    cc.write_json(cc.OUT / "stage2b-grad-probe.json", out)
    log(f"stage2b done: {out['skipped'] or out['per_ckpt']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
