"""MBS Three-Way Same-Source Causal Audit - stage 3: paired 2x2 scoring worker.

Forward-only scoring of the frozen same-source mirrored pairs (512 pairs,
frozen VBS pair manifest + latents) on the THREE frozen checkpoints:
PRE U118100 / CONTROL U118200 / TREATMENT U118200.  One process per DCU;
pairs are split by manifest index parity (worker 0 = even pair_index,
worker 1 = odd).  Run once per checkpoint (``--ckpt``) so the PRE replay
gate can be enforced before CONTROL/TREATMENT results are accepted.

This file is a faithful adaptation of the FROZEN VBS scorer
(vertical_bottom_supervision/score_pairs.py,
sha256 32446b1bc4f478a6b090499b22481739948f874cf474184aea98e8c01f11db8a).
The numerical core - conditioning computed ONCE per pair from the anchor
bundle, per-(pair,stratum) eps from the pre-registered seed contract shared
by TOP and BOTTOM, the 8-arm 4-stratum loss loop with the production-locked
t_eps/noise_observation_boundary - is VERBATIM.  Documented adaptations:
  1. --ckpt selects ONE checkpoint per invocation (frozen tool looped all
     three old ckpts); ledger file is per-(worker,ckpt).
  2. Checkpoint gate = contracts.verify_checkpoint_identity (this audit's
     own expected manifest/model-tree/full-tree shas + update + alpha)
     instead of the old audit's stage1-manifest evidence (which referenced
     U116100/U117100).
  3. Reads pair manifest / latents / bundles from the FROZEN VBS runtime
     roots (VBS_ROOT, bundles via anchor unit_file); writes NEW raw
     artifacts only under AUDIT_ROOT (/tmp/camera-mbs-three-way-audit).
  4. No --no-id: the ID arms are REQUIRED (the prior 32-pair timing gate
     already passed; spec keeps all 8 arms).

No training, no optimizer, no backward, no checkpoint writes.
Resume-safe: completed (ckpt, pair) rows are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contracts as C  # new three-way contracts (re-exports frozen VBS helpers)

LOG_WORKER = 0
CK_NAME = "?"


def log(msg: str, log_path: Path) -> None:
    line = f"[score3 w{LOG_WORKER} {CK_NAME} {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    global LOG_WORKER, CK_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=(0, 1), required=True)
    ap.add_argument("--ckpt", choices=C.CKS, required=True)
    ap.add_argument("--repo", type=Path, default=C.C2_REPO,
                    help="original causal-audit code worktree (provenance + PYTHONPATH src)")
    ap.add_argument("--pair-root", type=Path, default=C.VBS_ROOT,
                    help="frozen VBS runtime root (pair-manifest.json, latents/)")
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT,
                    help="NEW raw artifact root (ledgers, gate docs, logs)")
    ap.add_argument("--device", default=None, help="default cuda:<worker>")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()
    LOG_WORKER = args.worker
    CK_NAME = args.ckpt

    log_path = args.out_root / f"score-three-way-w{args.worker}-{args.ckpt.lower()}.log"
    args.out_root.mkdir(parents=True, exist_ok=True)
    led_dir = args.out_root / "ledger"
    led_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = led_dir / f"ledger-w{args.worker}-{args.ckpt.lower()}.jsonl"

    with open(args.pair_root / "pair-manifest.json", "r", encoding="utf-8") as fh:
        pm = json.load(fh)
    if pm.get("status") != "OK":
        log(f"FATAL: pair manifest status {pm.get('status')}", log_path)
        return 2
    pairs = [p for p in pm["pairs"] if p["pair_index"] % 2 == args.worker]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    arms = list(C.ARMS)
    log(
        f"worker {args.worker} ck={args.ckpt}: {len(pairs)} pairs, arms={arms}, "
        f"device={args.device or f'cuda:{args.worker}'}, pair_root={args.pair_root}",
        log_path,
    )

    import torch

    from sakuramoon.checkpoint.load import read_checkpoint_manifest
    from sakuramoon.cli.generation_eval import _resolve_alpha
    from sakuramoon.conditioning.rope import (
        full_canvas_crop_coordinates,
        image_coordinates,
    )
    from sakuramoon.objective.flow import (
        flow_matching_loss,
        interpolate_state,
        sample_noise,
    )
    from sakuramoon.train.step import TrainableCompositeInputs

    device = torch.device(args.device or f"cuda:{args.worker}")
    if not torch.cuda.is_available():
        log("FATAL: no CUDA/HCU device", log_path)
        return 2

    ck = args.ckpt
    gate = C.verify_checkpoint_identity(ck)
    if gate["status"] != "PASS":
        log(f"FATAL: {ck} identity gate FAILED: {gate['problems']}", log_path)
        return 2
    log(
        f"{ck} gate: path+manifest+model-tree sha == expected, "
        f"update={C.CKPT_UPDATES[ck]}, alpha=1.0",
        log_path,
    )

    from sakuramoon.checkpoint.load import load_inference_artifact

    ckpt = C.CKPT_PATHS[ck]
    ck_manifest = read_checkpoint_manifest(ckpt)
    alpha = _resolve_alpha(ckpt, None)
    gate["growth_alpha"] = alpha
    gate["identity"] = {
        "checkpoint_id": ck_manifest.identity.checkpoint_id,
        "update": int(ck_manifest.identity.update),
    }
    if float(alpha) != 1.0:
        log(f"FATAL: resolved alpha {alpha} != 1.0", log_path)
        return 2
    composite = load_inference_artifact(ckpt, ck_manifest.identity, device=device)
    composite = composite.eval()
    log(f"{ck}: loaded (alpha={alpha})", log_path)

    # resume: read completed pairs from this worker's per-ckpt ledger
    done: set[int] = set()
    if ledger_path.is_file():
        with open(ledger_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("worker") == args.worker:
                    done.add(int(row["pair_index"]))
    log(f"{ck}: resume, {len(done)} pairs already done", log_path)

    t_ckpt_start = time.time()
    n = 0
    n_anchor_coord_ok = 0
    bundle_cache: dict[int, object] = {}
    for row in pairs:
        i = row["pair_index"]
        if i in done:
            continue
        geo = row["geometry"]
        F = geo["full_height"]
        anchor = row["anchor_unit"]
        if anchor["unit"] not in bundle_cache:
            b = torch.load(anchor["unit_file"], map_location="cpu", weights_only=False)
            if b["unit"] != anchor["unit"]:
                raise RuntimeError(f"pair {i} bundle unit mismatch")
            tvals = [float(t) for t in b["t"]]
            expect = C.t_values()
            for a_, e_ in zip(tvals, expect):
                if abs(a_ - e_) > 1e-9:
                    raise RuntimeError(f"pair {i} bundle t {tvals} != frozen {expect}")
            bundle_cache[anchor["unit"]] = b
        b = bundle_cache[anchor["unit"]]

        x0_top = torch.load(
            args.pair_root / "latents" / f"pair-{i:04d}-top.pt",
            map_location=device, weights_only=False,
        )
        x0_bot = torch.load(
            args.pair_root / "latents" / f"pair-{i:04d}-bottom.pt",
            map_location=device, weights_only=False,
        )
        if tuple(x0_top.shape) != (1, 128, 16, 16) or tuple(x0_bot.shape) != (1, 128, 16, 16):
            raise RuntimeError(f"pair {i} latent shape mismatch")

        k_top = geo["k_start"]
        k_bot = geo["k_end"]
        coord_top = full_canvas_crop_coordinates(
            C.LATENT_TOKENS, C.LATENT_TOKENS,
            full_height=F, full_width=C.VIEWPORT,
            crop_box=(0, k_top, C.VIEWPORT, k_top + C.VIEWPORT),
            device=device,
        )
        coord_bot = full_canvas_crop_coordinates(
            C.LATENT_TOKENS, C.LATENT_TOKENS,
            full_height=F, full_width=C.VIEWPORT,
            crop_box=(0, k_bot, C.VIEWPORT, k_bot + C.VIEWPORT),
            device=device,
        )
        coord_id = image_coordinates(C.LATENT_TOKENS, C.LATENT_TOKENS, device=device)

        # invariant: anchor-side reconstructed coords == frozen bundle CORRECT
        bundle_correct = b["arms"]["CORRECT"].to(device)
        anchor_coord = coord_top if row["geometry"]["anchor_is_top"] else coord_bot
        if not torch.equal(anchor_coord, bundle_correct):
            raise RuntimeError(f"pair {i} anchor coords differ from frozen bundle CORRECT")
        n_anchor_coord_ok += 1

        qwen_states = b["qwen_states"].to(device)
        x0_anchor = b["x0"].to(device)  # 4D [1,C,H,W] (loss side)
        x0_anchor_lat = x0_anchor.squeeze(0)  # 3D [C,H,W] (PackedDiT contract)
        t0 = torch.tensor([float(b["t"][0])], dtype=torch.float32, device=device)
        inputs_base = TrainableCompositeInputs(
            qwen_states=qwen_states,
            main_token_indices=b["main_token_indices"].to(device),
            main_mask=b["main_mask"].to(device),
            main_token_lengths=tuple(int(v) for v in b["main_token_lengths"]),
            condition_token_indices=b["condition_token_indices"].to(device),
            condition_mask=b["condition_mask"].to(device),
            use_null_condition=b["use_null_condition"].to(device),
            active_condition_sample_indices=b["active_condition_sample_indices"].to(device),
            latents=(x0_anchor_lat,),
            image_coordinates=(anchor_coord,),
            timestep=t0,
            size_scale=torch.tensor([float(b["size_scale"])], device=device),
            aspect=torch.tensor([float(b["aspect"])], device=device),
            growth_alpha=alpha,
        )
        with torch.inference_mode():
            conditioning = composite.forward_conditioning(inputs_base)
            losses: dict[str, list[float]] = {a: [] for a in arms}
            for k in range(4):
                t_k = torch.tensor([float(b["t"][k])], dtype=torch.float32, device=device)
                gen = torch.Generator(device=device)
                gen.manual_seed(C.noise_seed(i, k))
                eps = sample_noise(x0_top, noise_scale=C.NOISE_SCALE, generator=gen)
                state_top = interpolate_state(x0_top, eps, t_k)
                state_bot = interpolate_state(x0_bot, eps, t_k)
                for arm in arms:
                    side = C.arm_side(arm)
                    ckind = C.arm_coord(arm)
                    coords = (
                        coord_top if ckind == "top_correct"
                        else coord_bot if ckind == "bottom_correct"
                        else coord_id
                    )
                    lat = state_top if side == "top" else state_bot
                    lat = lat.squeeze(0)
                    x0_side = x0_top if side == "top" else x0_bot
                    inp = replace(
                        inputs_base,
                        latents=(lat,),
                        image_coordinates=(coords,),
                        timestep=t_k,
                    )
                    pred = composite.forward_dit(inp, conditioning)[0]
                    loss = flow_matching_loss(
                        pred.unsqueeze(0),
                        lat.unsqueeze(0),
                        x0_side,
                        t_k,
                        t_eps=C.T_EPS,
                        noise_observation_boundary=C.NOISE_OBSERVATION_BOUNDARY,
                    )
                    v = float(loss.per_sample[0])
                    if math.isnan(v):
                        raise RuntimeError(f"pair {i} {ck} {arm} k{k} NaN loss")
                    losses[arm].append(v)
        entry = {
            "worker": args.worker,
            "ck": ck,
            "pair_index": i,
            "unit": anchor["unit"],
            "elapsed_s": round(time.time() - t_ckpt_start, 1),
            "losses": {a: losses[a] for a in arms},
        }
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
        n += 1
        if n % 32 == 0:
            per_forward = (time.time() - t_ckpt_start) / (n * 4 * len(arms))
            log(
                f"{ck}: {n}/{len(pairs) - len(done)} new pairs "
                f"(~{per_forward:.3f}s/forward)",
                log_path,
            )
        del b, x0_top, x0_bot, x0_anchor, x0_anchor_lat, qwen_states, conditioning
        torch.cuda.empty_cache()
    done_marker = led_dir / f"w{args.worker}-done-{args.ckpt.lower()}.json"
    C.write_frozen(done_marker, {"worker": args.worker, "ck": ck,
                                  "n_pairs": len(pairs), "arms": arms,
                                  "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    del composite
    torch.cuda.empty_cache()

    gate_doc = {
        "label": "MBS THREE-WAY SCORING GATE",
        "worker": args.worker,
        "ckpt": ck,
        "identity_gate": gate,
        "arms": arms,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "pair_root": str(args.pair_root),
        "anchor_coord_invariant_pairs": n_anchor_coord_ok,
        "new_pairs_this_run": n,
        "total_new_pairs": n + len(done),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_frozen(args.out_root / f"scoring-gate-w{args.worker}-{args.ckpt.lower()}.json", gate_doc)
    log(f"worker {args.worker} {ck} DONE", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
