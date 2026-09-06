"""Stage 2 - forward matrix: PRE/MID/POST x units x 4 JLT strata x arms.

Forward-only. Production TrainableComposite loaded read-only per checkpoint;
conditioning computed once per (checkpoint, unit); per (unit, stratum, arm) the
PackedDiT runs with the swapped image-coordinate tensor only. Loss = production
flow_matching_loss (x-pred velocity MSE, fp32, locked t_eps=0.05 / NOB=0.95).

Run one process per HCU: --worker 0|1 with HIP_VISIBLE_DEVICES=0|1.
Units are split by even/odd final-cohort index; both workers cover all checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
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


def log(msg: str) -> None:
    print(f"[stage2-w{os.environ.get('CC_WORKER', '?')} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arm_set_for(bundle: dict) -> list[str]:
    arms = ["CORRECT", "IDENTITY", "SAME"]
    if bundle["cohort"] == "camera":
        if not bundle["opp_na"]:
            arms.append("OPPOSITE")
        arms += ["SHUFFLED", "HALF", "OVER"]
    else:
        if "RANDOM" in bundle["arms"]:
            arms.append("RANDOM")
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    os.environ["CC_WORKER"] = str(args.worker)

    from typing import cast

    from sakuramoon.checkpoint.load import (
        load_inference_artifact,
        read_checkpoint_manifest,
    )
    from sakuramoon.cli.generation_eval import _resolve_alpha  # noqa: N812
    from sakuramoon.objective.flow import flow_matching_loss
    from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs

    device = torch.device("cuda", 0)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("stage2 worker requires exactly one visible CUDA device")

    with open(cc.OUT / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "OK":
        raise RuntimeError("stage1 manifest is not OK; refusing stage2")

    units_dir = cc.OUT / "units"
    all_units: list[int] = [meta["unit"] for meta in manifest["units"]]
    my_units = [u for i, u in enumerate(all_units) if i % 2 == args.worker]
    log(f"worker {args.worker}: {len(my_units)}/{len(all_units)} units")

    results_path = cc.OUT / f"results-worker{args.worker}.pt"
    done: dict = {}
    if results_path.is_file():
        done = torch.load(results_path, map_location="cpu", weights_only=False)
        log(f"resuming: {len(done)} units with partial checkpoint rows")

    determinism: dict[str, dict] = {}

    with torch.inference_mode():
        for ck_name in ("PRE", "MID", "POST"):
            ckpt = cc.CKPTS[ck_name]
            log(f"=== {ck_name} {ckpt} ===")
            ckpt_manifest = read_checkpoint_manifest(ckpt)
            alpha = _resolve_alpha(ckpt, None)
            log(f"update={ckpt_manifest.identity.update} growth_alpha={alpha}")
            composite = load_inference_artifact(ckpt, ckpt_manifest.identity, device=device)
            composite = cast(TrainableComposite, composite)
            composite.eval()
            t_ckpt = time.time()

            for n, unit in enumerate(my_units):
                if done.get(unit, {}).get(ck_name) is not None:
                    continue
                p = units_dir / f"unit-{unit:05d}.pt"
                b = torch.load(p, map_location="cpu", weights_only=False)
                arms = arm_set_for(b)
                x0 = b["x0"].to(device)  # 4D [1,C,H,W] — loss clean (batch-matched)
                x0_lat = x0.squeeze(0)  # 3D [C,H,W] — PackedDiT latents contract
                qwen_states = b["qwen_states"].to(device)
                arms_dev = {k: v.to(device) for k, v in b["arms"].items()}
                routing_dev = {
                    "main_token_indices": b["main_token_indices"].to(device),
                    "main_mask": b["main_mask"].to(device),
                    "condition_token_indices": b["condition_token_indices"].to(device),
                    "condition_mask": b["condition_mask"].to(device),
                    "use_null_condition": b["use_null_condition"].to(device),
                    "active_condition_sample_indices": b["active_condition_sample_indices"].to(device),
                }
                size_scale_t = torch.tensor([b["size_scale"]], device=device, dtype=torch.float32)
                aspect_t = torch.tensor([b["aspect"]], device=device, dtype=torch.float32)

                # Conditioning is independent of (stratum, arm) for a unit:
                # forward_conditioning reads only qwen_states + token routing
                # (train/step.py). Compute it once.
                inputs_base = TrainableCompositeInputs(
                    qwen_states=qwen_states,
                    main_token_indices=routing_dev["main_token_indices"],
                    main_mask=routing_dev["main_mask"],
                    main_token_lengths=b["main_token_lengths"],
                    condition_token_indices=routing_dev["condition_token_indices"],
                    condition_mask=routing_dev["condition_mask"],
                    use_null_condition=routing_dev["use_null_condition"],
                    active_condition_sample_indices=routing_dev["active_condition_sample_indices"],
                    latents=(x0_lat,),  # v2: 3D [C,H,W] per-sample (run1 passed 4D -> dit.py ValueError)
                    image_coordinates=(arms_dev["CORRECT"],),
                    timestep=torch.tensor([b["t"][0]], device=device, dtype=torch.float32),
                    size_scale=size_scale_t,
                    aspect=aspect_t,
                    growth_alpha=alpha,
                )
                conditioning = composite.forward_conditioning(inputs_base)

                # determinism probe: first unit of this checkpoint, stratum 0
                if not determinism.get(ck_name):
                    t0 = torch.tensor([b["t"][0]], device=device, dtype=torch.float32)
                    s0 = b["states"][0].to(device).squeeze(0)  # 3D [C,H,W]

                    def once(arm: str) -> tuple[float, torch.Tensor]:
                        inp = replace(
                            inputs_base,
                            latents=(s0,),
                            image_coordinates=(arms_dev[arm],),
                            timestep=t0,
                        )
                        pred = composite.forward_dit(inp, conditioning)[0]
                        loss = flow_matching_loss(
                            pred.unsqueeze(0), s0.unsqueeze(0), x0, t0,
                            t_eps=cc.T_EPS,
                            noise_observation_boundary=cc.NOISE_OBSERVATION_BOUNDARY,
                        )
                        return float(loss.per_sample[0].float().item()), pred

                    l1, p1 = once("CORRECT")
                    l2, p2 = once("CORRECT")
                    determinism[ck_name] = {
                        "unit": unit,
                        "loss_bitexact": l1 == l2,
                        "loss_absdiff": abs(l1 - l2),
                        "pred_bitexact": bool(torch.equal(p1, p2)),
                    }
                    log(f"  determinism {ck_name}: {determinism[ck_name]}")

                row: dict = {"loss": {}, "sens": {}, "correct_best3": []}
                for k in range(cc.N_STRATA):
                    t_k = torch.tensor([b["t"][k]], device=device, dtype=torch.float32)
                    state_k = b["states"][k].to(device).squeeze(0)  # 3D [C,H,W]
                    preds: dict[str, torch.Tensor] = {}
                    for arm in arms:
                        inputs_a = replace(
                            inputs_base,
                            latents=(state_k,),
                            image_coordinates=(arms_dev[arm],),
                            timestep=t_k,
                        )
                        pred = composite.forward_dit(inputs_a, conditioning)[0]
                        preds[arm] = pred
                        loss = flow_matching_loss(
                            pred.unsqueeze(0),
                            state_k.unsqueeze(0),
                            x0,
                            t_k,
                            t_eps=cc.T_EPS,
                            noise_observation_boundary=cc.NOISE_OBSERVATION_BOUNDARY,
                        )
                        row["loss"].setdefault(arm, []).append(
                            float(loss.per_sample[0].float().item())
                        )
                    pred_c = preds["CORRECT"]
                    rms_c = float(pred_c.float().pow(2).mean().sqrt())
                    for arm in arms:
                        if arm == "CORRECT":
                            continue
                        d = (preds[arm].float() - pred_c.float())
                        rel = float(d.pow(2).mean().sqrt()) / max(rms_c, cc.NORM_EPS)
                        cos = float(
                            torch.nn.functional.cosine_similarity(
                                preds[arm].float().flatten(),
                                pred_c.float().flatten(),
                                dim=0,
                            )
                        )
                        row["sens"].setdefault(arm, []).append([rel, cos])
                    l_id = row["loss"]["IDENTITY"][k]
                    l_opp = row["loss"]["OPPOSITE"][k] if "OPPOSITE" in row["loss"] else None
                    cb3 = 1.0 if (
                        row["loss"]["CORRECT"][k] <= l_id
                        and (l_opp is None or row["loss"]["CORRECT"][k] <= l_opp)
                    ) else 0.0
                    row["correct_best3"].append(cb3)

                done.setdefault(unit, {})[ck_name] = row
                if n % 64 == 0:
                    torch.save(done, results_path)
                    log(f"  {ck_name} {n}/{len(my_units)} units ({time.time()-t_ckpt:.0f}s)")
            torch.save(done, results_path)
            del composite
            torch.cuda.empty_cache()

    log("worker done")
    with open(cc.OUT / f"determinism-worker{args.worker}.json", "w", encoding="utf-8") as fh:
        json.dump(determinism, fh, indent=2)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except torch.cuda.OutOfMemoryError as e:
        log(f"OOM - audit batch is already 1; STOP per spec s28: {e}")
        sys.exit(3)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
