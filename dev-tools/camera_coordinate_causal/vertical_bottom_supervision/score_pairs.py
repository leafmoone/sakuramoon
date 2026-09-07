"""Vertical Bottom Supervision Audit - stage 3: paired 2x2 scoring worker.

Forward-only scoring of the frozen same-source mirrored pairs on the three
frozen checkpoints (PRE 116100 / MID 117100 / POST 118100).  One process per
DCU; pairs are split by manifest index parity (worker 0 = even indices).

Per (checkpoint, pair) the conditioning is computed ONCE (from the anchor
unit's frozen bundle routing + qwen states, exactly as stage2 built it) and
shared by all arms and strata.  Per (pair, stratum) one eps tensor is drawn
with the pre-registered seed contract.noise_seed and shared by the TOP and
BOTTOM arms.  Losses: flow_matching_loss(pred, state_side, x0_side, t_k) with
production locked t_eps/noise_observation_boundary.

Arms (spec s21):
  TT top-latents + top-correct coords      TB top-latents + bottom-correct
  BT bottom-latents + top-correct          BB bottom-latents + bottom-correct
  T_SAME / B_SAME exact duplicate forwards of TT / BB (numerics floor)
  T_ID / B_ID identity square (optional; --no-id to drop, decided by the
  32-pair smoke timing gate contracts.ID_ARMS_MAX_S_PER_FORWARD)

Writes ONLY under /tmp/camera-vertical-bottom/:
    ledger/ledger-w{worker}.jsonl     (one row per completed (ckpt, pair))
    ledger/w{worker}-done.json        (per-checkpoint completion marker)
    scoring-gate-{worker}.json        (ckpt identity / alpha / parity invariants)
    score-pairs-w{worker}.log

No training, no optimizer, no backward, no checkpoint writes.  Resume-safe:
completed (ckpt, pair) rows are skipped.
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

import contracts as C


def log(msg: str, log_path: Path) -> None:
    line = f"[score w{args_worker} {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


args_worker = 0  # set in main before first log()


def _tree_sha256(root: Path) -> str:
    """Verbatim copy of final_snapshot/cc_common.tree_sha256 (provenance)."""
    import hashlib

    files: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append((
                p.relative_to(root).as_posix(),
                hashlib.sha256(p.read_bytes()).hexdigest(),
            ))
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ckpt_gate(
    ck: str,
    stage1_ckpt_evidence: dict,
    log_path: Path,
) -> dict:
    """Verify one frozen checkpoint against the stage1 manifest evidence."""
    import hashlib

    ckpt = C.CKPT_PATHS[ck]
    if not ckpt.is_dir():
        raise RuntimeError(f"{ck} checkpoint missing: {ckpt}")
    complete = (ckpt / "COMPLETE").read_bytes()
    if complete != b"complete\n":
        raise RuntimeError(f"{ck} COMPLETE marker invalid")
    manifest_bytes = (ckpt / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    model_tree = _tree_sha256(ckpt / "model")
    ev = stage1_ckpt_evidence[ck]
    problems = []
    if str(ev["path"]) != str(ckpt):
        problems.append(f"path {ev['path']} != {ckpt}")
    if int(ev["update"]) != C.CKPT_UPDATES[ck]:
        problems.append(f"update {ev['update']} != {C.CKPT_UPDATES[ck]}")
    if ev["manifest_sha256"] != manifest_sha:
        problems.append("manifest sha mismatch")
    if ev["model_tree_sha256"] != model_tree:
        problems.append("model tree sha mismatch")
    if problems:
        raise RuntimeError(f"{ck} checkpoint gate FAILED: {problems}")
    log(f"{ck} gate: update={C.CKPT_UPDATES[ck]} path+manifest+tree sha == stage1 evidence", log_path)
    return {
        "ckpt": ck,
        "path": str(ckpt),
        "update": C.CKPT_UPDATES[ck],
        "manifest_sha256": manifest_sha,
        "model_tree_sha256": model_tree,
    }


def main() -> int:
    global args_worker
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, choices=(0, 1), required=True)
    ap.add_argument("--repo", type=Path, required=True, help="worktree (PYTHONPATH src)")
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT)
    ap.add_argument("--device", default=None, help="default cuda:<worker>")
    ap.add_argument("--no-id", action="store_true", help="skip T_ID/B_ID arms (smoke gate)")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()
    args_worker = args.worker

    log_path = args.out_root / f"score-pairs-w{args.worker}.log"
    led_dir = args.out_root / "ledger"
    led_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = led_dir / f"ledger-w{args.worker}.jsonl"

    with open(args.out_root / "pair-manifest.json", "r", encoding="utf-8") as fh:
        pm = json.load(fh)
    if pm.get("status") != "OK":
        log(f"FATAL: pair manifest status {pm.get('status')}", log_path)
        return 2
    pairs = [p for p in pm["pairs"] if p["pair_index"] % 2 == args.worker]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    arms = list(C.ARMS_NO_ID) if args.no_id else list(C.ARMS)
    log(
        f"worker {args.worker}: {len(pairs)} pairs, arms={arms}, device={args.device or f'cuda:{args.worker}'}",
        log_path,
    )

    import torch

    from sakuramoon.checkpoint.load import (
        load_inference_artifact,
        read_checkpoint_manifest,
    )
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

    with open(C.STAGE1_MANIFEST, "r", encoding="utf-8") as fh:
        stage1 = json.load(fh)
    stage1_ckpt_evidence = {
        name: entry for name, entry in stage1["checkpoints"].items()
    } if isinstance(stage1.get("checkpoints"), dict) else {}

    gates: dict[str, dict] = {}
    bundle_cache: dict[int, object] = {}

    for ck in C.CKS:
        gates[ck] = ckpt_gate(ck, stage1_ckpt_evidence, log_path)
        ckpt = C.CKPT_PATHS[ck]
        ck_manifest = read_checkpoint_manifest(ckpt)
        alpha = _resolve_alpha(ckpt, None)
        gates[ck]["growth_alpha"] = alpha
        gates[ck]["identity"] = {
            "checkpoint_id": ck_manifest.identity.checkpoint_id,
            "update": int(ck_manifest.identity.update),
        }
        composite = load_inference_artifact(ckpt, ck_manifest.identity, device=device)
        composite = composite.eval()
        log(f"{ck}: loaded (alpha={alpha})", log_path)

        # resume: read completed (pair,ck) from this worker's ledger
        done: set[int] = set()
        if ledger_path.is_file():
            with open(ledger_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("ck") == ck and row.get("worker") == args.worker:
                        done.add(int(row["pair_index"]))
        log(f"{ck}: resume, {len(done)} pairs already done", log_path)

        t_ckpt_start = time.time()
        n = 0
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
                args.out_root / "latents" / f"pair-{i:04d}-top.pt",
                map_location=device, weights_only=False,
            )
            x0_bot = torch.load(
                args.out_root / "latents" / f"pair-{i:04d}-bottom.pt",
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
                latents=(x0_anchor_lat,),  # 3D [C,H,W] per-sample (PackedDiT contract)
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
                        lat = lat.squeeze(0)  # 3D [C,H,W] for the DiT forward
                        x0_side = x0_top if side == "top" else x0_bot  # 4D for the loss
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
                        if math.isnan(v):  # NaN guard
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
        done_marker = led_dir / f"w{args.worker}-done-{ck}.json"
        C.write_frozen(done_marker, {"worker": args.worker, "ck": ck,
                                      "n_pairs": len(pairs), "arms": arms,
                                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        del composite
        torch.cuda.empty_cache()
        log(f"{ck}: worker {args.worker} done", log_path)

    gate_doc = {
        "label": "VERTICAL BOTTOM SUPERVISION SCORING GATE",
        "worker": args.worker,
        "gates": gates,
        "arms": arms,
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_frozen(args.out_root / f"scoring-gate-w{args.worker}.json", gate_doc)
    log(f"worker {args.worker} ALL DONE", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
