"""Posthoc SAME microprobe - SMALL read-only HCU numerical probe.

Purpose (section 22-25 of the posthoc review prompt):
  Determine whether the SAME-arm difference (Loss(SAME) - Loss(CORRECT) with
  byte/float-exact identical coordinate maps) is a bounded runtime numeric /
  order effect (the numerical floor) or a hidden mutable-state / harness
  defect. It uses ONLY the existing frozen unit bundles; no full rerun.

Design:
  - 32 camera units + 32 ordinary units, deterministic selection
    (round-robin over zoom-band x offset-low/high cells, unit id ascending).
  - Checkpoints PRE/MID/POST; two representative timestep strata (0 and 2).
  - Execution-sequence patterns for the exact same coordinate/input:
      P1: CORRECT, CORRECT                 (pure repeat - runtime floor)
      P2: CORRECT, IDENTITY, CORRECT       (arm interleaving)
      P3: CORRECT, SAME
      P4: SAME, CORRECT
  - Per pattern: loss mean-delta / p95 / p99 / max (absolute and relative to
    the CORRECT loss scale), prediction torch.equal rate, maxabs, relRMS.
  - The SAME map must be byte/float exact == CORRECT in every bundle
    (fail-closed if not).

Interpretation (section 24):
  - bounded zero-mean small jitter  -> quantified runtime numeric floor;
  - large systematic loss bias, order-dependent directional drift, hidden
    state accumulation, or coordinate map mutation -> hidden_state_detected
    = true, which forces NUMERICS_CLEAN = NO / BLOCKED_NUMERICS upstream.
  - The historical 6/6 determinism probes (first-unit immediate replay,
    bitexact) are a different sequence than the interleaved-arm patterns
    here; if they appear to disagree, the data in this probe output explains
    which sequence shows the effect (not hidden in either report).

Read-only: writes ONLY /tmp/camera-coordinate-causal/posthoc-microprobe.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# Locked constants mirroring final_snapshot/cc_common.py (do not import it).
T_EPS = 0.05
NOISE_OBSERVATION_BOUNDARY = 0.95
STRATA = (0, 2)
N_CAM = 32
N_ORD = 32
OUT_DIR = Path("/tmp/camera-coordinate-causal")
PROBE_JSON = OUT_DIR / "posthoc-microprobe.json"

CKPTS: dict[str, Path] = {
    "PRE": Path("/sakuramoon-runtime/output_model/g1/ckpt_116100_raw-116100-update-cadence"),
    "MID": OUT_DIR / "ckpts" / "MID" / "ckpt_117100_raw-117100-update-cadence",
    "POST": Path("/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence"),
}

# Hidden-state thresholds (documented, locked by tests via the probe output).
P1_REL_CAP = 1e-6          # pure repeat must be (near) bitexact
SAME_REL_MAX_CAP = 5e-4    # vs historical SAME maxabs ~1.57e-4 abs on L~0.55
DIRECTION_FRAC = 0.25      # |signed mean| / max  -> directional drift
ACCUM_RATIO_CAP = 3.0      # drift growth with sequence length


def log(msg: str) -> None:
    print(msg, flush=True)


def zoom_band(z: float) -> str:
    if z < 1.20:
        return "mild"
    if z < 1.35:
        return "medium"
    return "strong"


def select_units(meta: list[dict], n: int) -> list[int]:
    """Deterministic stratified selection: round-robin over the 6 cells
    (mild/medium/strong x low/high offset), unit id ascending within a cell."""
    cells: dict[tuple[str, str], list[int]] = {
        (b, o): [] for b in ("mild", "medium", "strong") for o in ("low", "high")
    }
    for row in meta:
        b = zoom_band(float(row["zoom"]))
        o = "low" if float(row["norm_offset"]) < 0.5 else "high"
        cells[(b, o)].append(int(row["unit"]))
    for lst in cells.values():
        lst.sort()
    order = list(cells.keys())  # fixed dict order
    picked: list[int] = []
    cursors = {k: 0 for k in order}
    while len(picked) < n:
        progressed = False
        for k in order:
            if len(picked) >= n:
                break
            if cursors[k] < len(cells[k]):
                picked.append(cells[k][cursors[k]])
                cursors[k] += 1
                progressed = True
        if not progressed:
            break  # fewer units than requested in the cohort
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree (not used beyond arg parity)")
    ap.add_argument("--dry-run", action="store_true", help="selection + bundle checks only; no forwards")
    args = ap.parse_args()

    from sakuramoon.checkpoint.load import (
        load_inference_artifact,
        read_checkpoint_manifest,
    )
    from sakuramoon.cli.generation_eval import _resolve_alpha
    from sakuramoon.objective.flow import flow_matching_loss
    from sakuramoon.train.step import TrainableCompositeInputs

    device = torch.device("cuda", 0)
    if not torch.cuda.is_available():
        raise RuntimeError("microprobe requires exactly one visible CUDA device")

    with open(OUT_DIR / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "OK":
        raise SystemExit("stage1 manifest not OK")
    rows = manifest["units"]
    cam_meta = [r for r in rows if r["cohort"] == "camera"]
    ord_meta = [r for r in rows if r["cohort"] == "ordinary"]
    cam_units = select_units(cam_meta, N_CAM)
    ord_units = select_units(ord_meta, N_ORD)
    log(f"selected camera units: {cam_units}")
    log(f"selected ordinary units: {ord_units}")

    # ---- SAME == CORRECT exactness (fail closed) across all selected bundles ----
    same_exact = True
    tvals: dict[int, list[float]] = {}
    for u in cam_units + ord_units:
        b = torch.load(OUT_DIR / "units" / f"unit-{u:05d}.pt", map_location="cpu", weights_only=False)
        if not torch.equal(b["arms"]["SAME"], b["arms"]["CORRECT"]):
            same_exact = False
            log(f"SAME != CORRECT exact for unit {u} - FAIL CLOSED")
        tvals[int(u)] = [float(x) for x in b["t"]]
    if not same_exact:
        log("ABORT: coordinate map mutation detected (SAME != CORRECT exact)")
        return 6

    if args.dry_run:
        log("dry-run OK (selection + same-exactness)")
        return 0

    # ---- forward collection ----
    rec: dict = {
        "p1_pairs": [],            # loss delta CORRECT repeat (P1)
        "p2_repeats": [],          # loss delta CORRECT repeat inside P2 (3rd vs 1st)
        "p2_identity": [],         # loss(IDENTITY) - loss(CORRECT) in P2
        "sc_pairs": [],            # loss delta SAME vs CORRECT, P3+P4 pooled
        "sc_rel": [],              # relative (delta / ref) SAME vs CORRECT
        "pred_p1": {"equal": 0, "n": 0, "maxabs": 0.0, "relrms": 0.0},
        "pred_p2": {"equal": 0, "n": 0, "maxabs": 0.0, "relrms": 0.0},
        "pred_sc": {"equal": 0, "n": 0, "maxabs": 0.0, "relrms": 0.0, "relrms_vals": []},
    }
    l_cor_ref: list[float] = []

    def rel(delta: float, ref: float) -> float:
        return float(delta) / max(float(ref), 1e-12)

    with torch.inference_mode():
        for ck_name in ("PRE", "MID", "POST"):
            ckpt = CKPTS[ck_name]
            ckpt_manifest = read_checkpoint_manifest(ckpt)
            alpha = _resolve_alpha(ckpt, None)
            log(f"=== {ck_name} {ckpt.name} (update={ckpt_manifest.identity.update} alpha={alpha}) ===")
            composite = load_inference_artifact(ckpt, ckpt_manifest.identity, device=device)
            composite = composite.eval()

            for u in cam_units + ord_units:
                b = torch.load(OUT_DIR / "units" / f"unit-{u:05d}.pt", map_location="cpu", weights_only=False)
                x0 = b["x0"].to(device)                       # 4D [1,C,H,W]
                x0_lat = x0.squeeze(0)                        # 3D [C,H,W]
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
                inputs_base = TrainableCompositeInputs(
                    qwen_states=qwen_states,
                    main_token_indices=routing_dev["main_token_indices"],
                    main_mask=routing_dev["main_mask"],
                    main_token_lengths=b["main_token_lengths"],
                    condition_token_indices=routing_dev["condition_token_indices"],
                    condition_mask=routing_dev["condition_mask"],
                    use_null_condition=routing_dev["use_null_condition"],
                    active_condition_sample_indices=routing_dev["active_condition_sample_indices"],
                    latents=(x0_lat,),
                    image_coordinates=(arms_dev["CORRECT"],),
                    timestep=torch.tensor([b["t"][0]], device=device, dtype=torch.float32),
                    size_scale=size_scale_t,
                    aspect=aspect_t,
                    growth_alpha=alpha,
                )
                conditioning = composite.forward_conditioning(inputs_base)

                def once(
                    arm: str, k: int,
                    _b=b, _inputs_base=inputs_base, _arms=arms_dev,
                    _composite=composite, _conditioning=conditioning, _x0=x0,
                ) -> tuple[float, torch.Tensor, float]:
                    t_k = torch.tensor([_b["t"][k]], device=device, dtype=torch.float32)
                    s_k = _b["states"][k].to(device).squeeze(0)
                    inp = replace(_inputs_base, latents=(s_k,), image_coordinates=(_arms[arm],), timestep=t_k)
                    pred = _composite.forward_dit(inp, _conditioning)[0]
                    loss = flow_matching_loss(
                        pred.unsqueeze(0), s_k.unsqueeze(0), _x0, t_k,
                        t_eps=T_EPS,
                        noise_observation_boundary=NOISE_OBSERVATION_BOUNDARY,
                    )
                    return float(loss.per_sample[0].float().item()), pred, float(t_k[0])

                for k in STRATA:
                    # P1: CORRECT, CORRECT
                    l1, p1, _ = once("CORRECT", k)
                    l2, p2, _ = once("CORRECT", k)
                    ref = l1
                    l_cor_ref.append(ref)
                    d12 = l2 - l1
                    rec["p1_pairs"].append(d12)
                    e = bool(torch.equal(p1, p2))
                    ma = float((p1 - p2).abs().max())
                    rr = float(((p1 - p2).norm() / p1.norm()).item())
                    rec["pred_p1"]["equal"] += int(e)
                    rec["pred_p1"]["n"] += 1
                    rec["pred_p1"]["maxabs"] = max(rec["pred_p1"]["maxabs"], ma)
                    rec["pred_p1"]["relrms"] = max(rec["pred_p1"]["relrms"], rr)

                    # P2: CORRECT, IDENTITY, CORRECT
                    l3, p3, _ = once("CORRECT", k)
                    l4, _, _ = once("IDENTITY", k)
                    l5, p5, _ = once("CORRECT", k)
                    d35 = l5 - l3
                    rec["p2_repeats"].append(d35)
                    rec["p2_identity"].append(l4 - l3)
                    e = bool(torch.equal(p3, p5))
                    ma = float((p3 - p5).abs().max())
                    rr = float(((p3 - p5).norm() / p3.norm()).item())
                    rec["pred_p2"]["equal"] += int(e)
                    rec["pred_p2"]["n"] += 1
                    rec["pred_p2"]["maxabs"] = max(rec["pred_p2"]["maxabs"], ma)
                    rec["pred_p2"]["relrms"] = max(rec["pred_p2"]["relrms"], rr)

                    # P3: CORRECT, SAME
                    l6, p6, _ = once("CORRECT", k)
                    l7, p7, _ = once("SAME", k)
                    # P4: SAME, CORRECT
                    l8, p8, _ = once("SAME", k)
                    l9, p9, _ = once("CORRECT", k)
                    for (l_a, l_b, p_a, p_b) in ((l6, l7, p6, p7), (l8, l9, p8, p9)):
                        d = l_b - l_a
                        sc_ref = max(l_a, 1e-12)
                        rec["sc_pairs"].append(d)
                        rec["sc_rel"].append(d / sc_ref)
                        e = bool(torch.equal(p_a, p_b))
                        ma = float((p_a - p_b).abs().max())
                        rr = float(((p_a - p_b).norm() / p_a.norm()).item())
                        rec["pred_sc"]["equal"] += int(e)
                        rec["pred_sc"]["n"] += 1
                        rec["pred_sc"]["maxabs"] = max(rec["pred_sc"]["maxabs"], ma)
                        rec["pred_sc"]["relrms"] = max(rec["pred_sc"]["relrms"], rr)
                        rec["pred_sc"]["relrms_vals"].append(rr)

            del composite
            torch.cuda.empty_cache()

    # ---- aggregation ----
    def agg(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95.0)),
            "p99": float(np.percentile(a, 99.0)),
            "max_abs": float(np.abs(a).max()),
            "signed_mean": float(a.mean()),
        }

    p1_abs = np.abs(np.asarray(rec["p1_pairs"]))
    p1_rel_max = float(p1_abs.max() / max(float(np.mean(l_cor_ref)), 1e-12))
    p2_abs = np.abs(np.asarray(rec["p2_repeats"]))
    p2_rel_max = float(p2_abs.max() / max(float(np.mean(l_cor_ref)), 1e-12))
    sc_rel = np.asarray(rec["sc_rel"], dtype=np.float64)
    accum_ratio = float(p2_abs.max() / p1_abs.max()) if p1_abs.max() > 0 else float(p2_abs.max() > 0)

    direction_drift = bool(np.abs(sc_rel.mean()) > DIRECTION_FRAC * max(float(np.abs(sc_rel).max()), 1e-30))
    hidden = (
        not same_exact
        or p1_rel_max > P1_REL_CAP
        or float(np.abs(sc_rel).max()) > SAME_REL_MAX_CAP
        or direction_drift
        or accum_ratio > ACCUM_RATIO_CAP
    )

    out = {
        "status": "OK",
        "present": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_camera": len(cam_units),
        "n_ordinary": len(ord_units),
        "camera_units": cam_units,
        "ordinary_units": ord_units,
        "selection": {
            "zoom_bands": {"mild": "zoom < 1.20", "medium": "1.20 <= zoom < 1.35", "strong": "zoom >= 1.35"},
            "offset": {"low": "norm_offset < 0.5", "high": "norm_offset >= 0.5"},
            "method": "round-robin over 6 cells (mild/medium/strong x low/high), unit id ascending",
        },
        "strata": list(STRATA),
        "t_values": {str(k): tvals[cam_units[0]][k] for k in STRATA},
        "checkpoints": ["PRE", "MID", "POST"],
        "same_exact_equal_correct": same_exact,
        "p1": {
            "n_pairs": len(rec["p1_pairs"]),
            "loss_delta": agg(rec["p1_pairs"]),
            "max_abs_loss_delta_rel": p1_rel_max,
            "pred_bitexact_rate": rec["pred_p1"]["equal"] / max(rec["pred_p1"]["n"], 1),
            "pred_maxabs": rec["pred_p1"]["maxabs"],
            "pred_relrms_max": rec["pred_p1"]["relrms"],
            "bitexact_all": rec["pred_p1"]["equal"] == rec["pred_p1"]["n"],
        },
        "p2": {
            "repeat_loss_delta": agg(rec["p2_repeats"]),
            "max_abs_loss_delta_rel": p2_rel_max,
            "identity_shift_mean": float(np.mean(rec["p2_identity"])) if rec["p2_identity"] else None,
            "order_drift_vs_p1_max_ratio": accum_ratio,
            "pred_bitexact_rate": rec["pred_p2"]["equal"] / max(rec["pred_p2"]["n"], 1),
            "pred_relrms_max": rec["pred_p2"]["relrms"],
        },
        "same_vs_correct": {
            "n": len(rec["sc_pairs"]),
            "loss_delta": agg(rec["sc_pairs"]),
            "rel_p99": float(np.percentile(np.abs(sc_rel), 99.0)),
            "rel_max": float(np.abs(sc_rel).max()),
            "signed_mean_rel": float(sc_rel.mean()),
            "directional_drift": direction_drift,
            "pred_bitexact_rate": rec["pred_sc"]["equal"] / max(rec["pred_sc"]["n"], 1),
            "pred_relrms_p99": float(np.percentile(rec["pred_sc"]["relrms_vals"], 99.0)) if rec["pred_sc"]["relrms_vals"] else None,
        },
        "accumulation_ratio": accum_ratio,
        "thresholds": {
            "p1_rel_cap": P1_REL_CAP,
            "same_rel_max_cap": SAME_REL_MAX_CAP,
            "direction_frac": DIRECTION_FRAC,
            "accum_ratio_cap": ACCUM_RATIO_CAP,
        },
        "hidden_state_detected": bool(hidden),
        "interpretation": (
            "hidden mutable state / order defect suspected - see threshold breaches"
            if hidden
            else "bounded zero-mean runtime numeric jitter; quantify as the runtime numeric floor"
        ),
    }
    with open(PROBE_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    log(f"probe written to {PROBE_JSON}")
    log(f"hidden_state_detected={hidden} p1_rel_max={p1_rel_max:.3e} sc_rel_max={float(np.abs(sc_rel).max()):.3e} accum_ratio={accum_ratio:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
