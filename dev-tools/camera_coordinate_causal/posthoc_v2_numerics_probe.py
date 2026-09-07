"""Posthoc V2 SAME microprobe - SMALL read-only HCU numerical probe (v2).

Purpose (sections 10-18 of the posthoc V2 review prompt):
  Quantify the ordinary runtime nondeterminism floor (P0 immediate repeats)
  and use repeated-sequence structure to distinguish kernel nondeterminism
  from mutable/order state:

    P0: CORRECT x8                        (immediate repeats - runtime floor)
    P1: C ID C ID C ID C                   (arm interleave; C-position drift)
    P2: C SAME C SAME C SAME C             (SAME interleave; SAME==CORRECT exact)
    P3: fresh-load RUN A vs reload RUN B   (POST ck, first 8 camera + 8
        ordinary units, stratum 0): statistical consistency of the whole
        14-position sequence, first-forward systematic difference,
        reproducibility of the sequence-position effect. NO bitexact
        requirement across reload.

  Plus, per checkpoint load:
    - model parameter / persistent buffer immutability (fingerprinted
      after the first unit completes, re-checked after the last);
    - CPU + device RNG state audit (consumption is recorded, never an
      automatic training-mechanism failure);
    - coordinate map mutation check (arms tensors bit-stable across each
      block, SAME == CORRECT exact in every bundle - fail closed).

  V2 terminology (sections 7-8):
    A REPEAT_NUMERIC_JITTER_PRESENT  = P0 not 100% bitexact OR max relative
      repeat jitter > P1_REL_CAP_DIAGNOSTIC (1e-6). DIAGNOSTIC ONLY.
    B COORDINATE_MAP_MUTATION_DETECTED
    C PERSISTENT_BUFFER_MUTATION_DETECTED (buffers or parameters)
    D ORDER_DEPENDENT_DRIFT_DETECTED  = interleave C drift systematic beyond
      the P0 floor (signed-mean CI excl 0, slope CI beyond the P0 slope
      floor, or p99 magnitude ratio > MAG_FLOOR_RATIO). Distributional,
      never single-max based.
    E ACCUMULATING_DRIFT_DETECTED     = CI-backed positive interleave slope
      beyond the P0 floor (drift grows with call index).
    F HIDDEN_MUTABLE_STATE_DETECTED   = B OR C OR D OR E. A alone NEVER implies F.

  The V1 gate bug (p1_rel_max > 1e-6 => hidden_state=True) is not repeated:
  the 1e-6 cap is recorded as a diagnostic flag only.

Read-only: writes ONLY /tmp/camera-coordinate-causal/posthoc-v2-microprobe.json
(never the V1 posthoc-microprobe.json, whose sha is verified before writing).
"""
from __future__ import annotations

import argparse
import hashlib
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
N_P0 = 8
P1_SEQ = ("CORRECT", "IDENTITY", "CORRECT", "IDENTITY", "CORRECT", "IDENTITY", "CORRECT")
P2_SEQ = ("CORRECT", "SAME", "CORRECT", "SAME", "CORRECT", "SAME", "CORRECT")
C_POSITIONS = (0, 2, 4, 6)
P3_SEQ = (
    "CORRECT", "CORRECT", "CORRECT", "CORRECT",
    "CORRECT", "IDENTITY", "CORRECT", "IDENTITY", "CORRECT",
    "CORRECT", "SAME", "CORRECT", "SAME", "CORRECT",
)
OUT_DIR = Path("/tmp/camera-coordinate-causal")
PROBE_JSON = OUT_DIR / "posthoc-v2-microprobe.json"
V1_PROBE_JSON = OUT_DIR / "posthoc-microprobe.json"
V1_PROBE_SHA = "85346492a3500454b285e76a0d6539b0227d44593adcd22cec2fc99734068720"
BOOT_SEED = 20260907
N_BOOT = 10000

# Diagnostic thresholds (documented). None of them alone can set hidden state.
P1_REL_CAP_DIAGNOSTIC = 1e-6
MAG_FLOOR_RATIO = 3.0  # p99(interleave |delta|) / p99(P0 |delta|) magnitude check

CKPTS: dict[str, Path] = {
    "PRE": Path("/sakuramoon-runtime/output_model/g1/ckpt_116100_raw-116100-update-cadence"),
    "MID": OUT_DIR / "ckpts" / "MID" / "ckpt_117100_raw-117100-update-cadence",
    "POST": Path("/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence"),
}


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
    (mild/medium/strong x low/high offset), unit id ascending within a cell.
    IDENTICAL to the V1 probe so the unit populations match exactly."""
    cells: dict[tuple[str, str], list[int]] = {
        (b, o): [] for b in ("mild", "medium", "strong") for o in ("low", "high")
    }
    for row in meta:
        b = zoom_band(float(row["zoom"]))
        o = "low" if float(row["norm_offset"]) < 0.5 else "high"
        cells[(b, o)].append(int(row["unit"]))
    for lst in cells.values():
        lst.sort()
    order = list(cells.keys())
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
            break
    return picked


def _pct(vals: np.ndarray, q: float) -> float:
    return float(np.percentile(vals, q)) if vals.size else float("nan")


def _fingerprint(model: torch.nn.Module) -> dict[str, tuple | None]:
    """Deterministic per-tensor (shape, dtype, sum, sumsq) float64
    fingerprint of all named parameters and named buffers."""
    out: dict[str, tuple | None] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            t = p.detach().to(torch.float64)
            out[f"param:{name}"] = (tuple(p.shape), str(p.dtype), float(t.sum()), float((t * t).sum()))
            del t
        for name, buf in model.named_buffers():
            if buf is None:
                out[f"buffer:{name}"] = None
                continue
            t = buf.detach().to(torch.float64)
            out[f"buffer:{name}"] = (tuple(buf.shape), str(buf.dtype), float(t.sum()), float((t * t).sum()))
            del t
    return out


def _fingerprint_diff(before: dict[str, tuple | None], after: dict[str, tuple | None]) -> dict[str, list[str]]:
    changed: dict[str, list[str]] = {"param": [], "buffer": []}
    for key, val in before.items():
        new = after.get(key, "<missing>")
        if new != val:
            kind = "param" if key.startswith("param:") else "buffer"
            changed[kind].append(key)
    for key in after:
        if key not in before:
            kind = "param" if key.startswith("param:") else "buffer"
            changed[kind].append(key + " [new]")
    return changed


def _rng_hashes(device: torch.device) -> dict[str, str]:
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state(device)
    return {
        "cpu_sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "gpu_sha256": hashlib.sha256(gpu.cpu().numpy().tobytes()).hexdigest(),
    }


def _ci95(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _block_mean_ci(vals: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = vals.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return _ci95(vals[idx].mean(axis=1))


def _block_slope_ci(slopes: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = slopes.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return _ci95(slopes[idx].mean(axis=1))


def _ci_excl0(ci: list[float]) -> bool:
    return ci[0] > 0.0 or ci[1] < 0.0


def _slope_beyond_floor(ci: list[float], floor: tuple[float, float]) -> bool:
    return ci[0] > floor[1] or ci[1] < floor[0]


def _mean_stat(vals: np.ndarray, rng: np.random.Generator) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    if not a.size:
        return {"signed_mean": None, "signed_mean_ci95": None, "abs_p99": None, "n": 0}
    return {
        "signed_mean": float(a.mean()),
        "signed_mean_ci95": _block_mean_ci(a, rng),
        "abs_p99": float(np.percentile(np.abs(a), 99.0)),
        "n": int(a.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree (not used beyond arg parity)")
    ap.add_argument("--dry-run", action="store_true", help="selection + bundle checks only; no forwards")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import posthoc_v2_contracts as v2c

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
    all_units = cam_units + ord_units
    log(f"selected camera units: {cam_units}")
    log(f"selected ordinary units: {ord_units}")

    # ---- SAME == CORRECT exactness (fail closed) across all selected bundles ----
    same_exact = True
    tvals: dict[int, list[float]] = {}
    for u in all_units:
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
    p0: list[dict] = []    # per block: 7 signed deltas vs first + pred stats
    p1c: list[dict] = []   # per block: 3 C-position deltas + C pred relrms
    p2c: list[dict] = []   # per block: 3 C-position deltas (P2 sequence)
    p2s: list[dict] = []   # per block: 3 SAME-position deltas vs first C
    ck_identity: dict[str, dict] = {}
    param_mutation: list[str] = []
    buffer_mutation: list[str] = []
    arms_stable = True

    def collect(unit: int, b: dict, arms_dev: dict, composite, conditioning, x0, inputs_base, k: int) -> None:
        nonlocal arms_stable
        t_k = torch.tensor([b["t"][k]], device=device, dtype=torch.float32)
        before_arms = {a: v.clone() for a, v in arms_dev.items()}

        def once(arm: str) -> tuple[float, torch.Tensor]:
            s_k = b["states"][k].to(device).squeeze(0)
            inp = replace(inputs_base, latents=(s_k,), image_coordinates=(arms_dev[arm],), timestep=t_k)
            pred = composite.forward_dit(inp, conditioning)[0]
            loss = flow_matching_loss(
                pred.unsqueeze(0), s_k.unsqueeze(0), x0, t_k,
                t_eps=T_EPS,
                noise_observation_boundary=NOISE_OBSERVATION_BOUNDARY,
            )
            return float(loss.per_sample[0].float().item()), pred

        # P0: CORRECT x8 (immediate repeats)
        ref_l, ref_p = once("CORRECT")
        ls_p0 = [ref_l]
        ps_p0 = [ref_p]
        bitexact = 0
        pred_maxabs = 0.0
        pred_relrms_max = 0.0
        for _ in range(N_P0 - 1):
            l, p = once("CORRECT")
            ls_p0.append(l)
            if torch.equal(ref_p, p):
                bitexact += 1
            pred_maxabs = max(pred_maxabs, float((ref_p - p).abs().max()))
            pred_relrms_max = max(pred_relrms_max, float(((ref_p - p).norm() / ref_p.norm()).item()))
        d = np.asarray(ls_p0, dtype=np.float64)[1:] - float(ls_p0[0])
        p0.append({
            "deltas": d.tolist(),
            "bitexact": int(bitexact),
            "n": N_P0 - 1,
            "pred_maxabs": pred_maxabs,
            "pred_relrms_max": pred_relrms_max,
            "ref": ref_l,
        })
        del ps_p0

        # P1: C ID C ID C ID C
        ls1 = [0.0] * len(P1_SEQ)
        ps1: list[torch.Tensor] = []
        for i, a in enumerate(P1_SEQ):
            ls1[i], p = once(a)
            ps1.append(p)
        c_l = np.asarray([ls1[i] for i in C_POSITIONS], dtype=np.float64)
        c_pred_relrms = [
            float((ps1[0] - ps1[i]).norm().item() / ps1[0].norm().item())
            for i in C_POSITIONS[1:]
        ]
        p1c.append({
            "c_deltas": (c_l[1:] - c_l[0]).tolist(),
            "identity_shifts": [float(ls1[i] - ls1[i - 1]) for i in (1, 3, 5)],
            "c_pred_relrms": c_pred_relrms,
        })
        del ps1

        # P2: C SAME C SAME C SAME C
        ls2 = [0.0] * len(P2_SEQ)
        for i, a in enumerate(P2_SEQ):
            ls2[i], _ = once(a)
        c_l2 = np.asarray([ls2[i] for i in C_POSITIONS], dtype=np.float64)
        s_l2 = np.asarray([ls2[i] for i in (1, 3, 5)], dtype=np.float64)
        p2c.append({"c_deltas": (c_l2[1:] - c_l2[0]).tolist()})
        p2s.append({"same_deltas": (s_l2 - c_l2[0]).tolist()})

        for a, before_v in before_arms.items():
            if not torch.equal(before_v, arms_dev[a]):
                arms_stable = False
                log(f"arms[{a}] mutated for unit {unit} stratum {k}")
        del before_arms

    def run_p3(ck_name: str) -> list[dict]:
        """Fresh load of the checkpoint, then the fixed 14-position P3
        sequence per unit (stratum 0). Returns per-unit positional losses.
        The composite is deleted and the cache emptied before return, so a
        second call is a genuinely fresh reload (RUN B)."""
        ckpt = CKPTS[ck_name]
        ckpt_manifest = read_checkpoint_manifest(ckpt)
        alpha = _resolve_alpha(ckpt, None)
        composite = load_inference_artifact(ckpt, ckpt_manifest.identity, device=device)
        composite = composite.eval()
        records: list[dict] = []
        with torch.inference_mode():
            for u in cam_units[:8] + ord_units[:8]:
                b = torch.load(OUT_DIR / "units" / f"unit-{u:05d}.pt", map_location="cpu", weights_only=False)
                x0 = b["x0"].to(device)
                x0_lat = x0.squeeze(0)
                qwen_states = b["qwen_states"].to(device)
                arms_dev = {k: v.to(device) for k, v in b["arms"].items()}
                t0 = torch.tensor([b["t"][0]], device=device, dtype=torch.float32)
                size_scale_t = torch.tensor([b["size_scale"]], device=device, dtype=torch.float32)
                aspect_t = torch.tensor([b["aspect"]], device=device, dtype=torch.float32)
                inputs_base = TrainableCompositeInputs(
                    qwen_states=qwen_states,
                    main_token_indices=b["main_token_indices"].to(device),
                    main_mask=b["main_mask"].to(device),
                    main_token_lengths=b["main_token_lengths"],
                    condition_token_indices=b["condition_token_indices"].to(device),
                    condition_mask=b["condition_mask"].to(device),
                    use_null_condition=b["use_null_condition"].to(device),
                    active_condition_sample_indices=b["active_condition_sample_indices"].to(device),
                    latents=(x0_lat,),
                    image_coordinates=(arms_dev["CORRECT"],),
                    timestep=t0,
                    size_scale=size_scale_t,
                    aspect=aspect_t,
                    growth_alpha=alpha,
                )
                conditioning = composite.forward_conditioning(inputs_base)
                losses: list[float] = []
                for arm in P3_SEQ:
                    s_k = b["states"][0].to(device).squeeze(0)
                    inp = replace(inputs_base, latents=(s_k,), image_coordinates=(arms_dev[arm],), timestep=t0)
                    pred = composite.forward_dit(inp, conditioning)[0]
                    loss = flow_matching_loss(pred.unsqueeze(0), s_k.unsqueeze(0), x0, t0, t_eps=T_EPS, noise_observation_boundary=NOISE_OBSERVATION_BOUNDARY)
                    losses.append(float(loss.per_sample[0].float().item()))
                records.append({"unit": int(u), "losses": losses})
        del composite
        torch.cuda.empty_cache()
        return records

    rng_before = _rng_hashes(device)
    with torch.inference_mode():
        for ck_name in ("PRE", "MID", "POST"):
            ckpt = CKPTS[ck_name]
            ckpt_manifest = read_checkpoint_manifest(ckpt)
            alpha = _resolve_alpha(ckpt, None)
            ck_identity[ck_name] = {"path": str(ckpt), "update": int(ckpt_manifest.identity.update), "alpha": float(alpha)}
            log(f"=== {ck_name} {ckpt.name} (update={ckpt_manifest.identity.update} alpha={alpha}) ===")
            composite = load_inference_artifact(ckpt, ckpt_manifest.identity, device=device)
            composite = composite.eval()
            fp_before: dict[str, tuple | None] | None = None
            for u in all_units:
                b = torch.load(OUT_DIR / "units" / f"unit-{u:05d}.pt", map_location="cpu", weights_only=False)
                x0 = b["x0"].to(device)
                x0_lat = x0.squeeze(0)
                qwen_states = b["qwen_states"].to(device)
                arms_dev = {k: v.to(device) for k, v in b["arms"].items()}
                size_scale_t = torch.tensor([b["size_scale"]], device=device, dtype=torch.float32)
                aspect_t = torch.tensor([b["aspect"]], device=device, dtype=torch.float32)
                t0 = torch.tensor([b["t"][0]], device=device, dtype=torch.float32)
                inputs_base = TrainableCompositeInputs(
                    qwen_states=qwen_states,
                    main_token_indices=b["main_token_indices"].to(device),
                    main_mask=b["main_mask"].to(device),
                    main_token_lengths=b["main_token_lengths"],
                    condition_token_indices=b["condition_token_indices"].to(device),
                    condition_mask=b["condition_mask"].to(device),
                    use_null_condition=b["use_null_condition"].to(device),
                    active_condition_sample_indices=b["active_condition_sample_indices"].to(device),
                    latents=(x0_lat,),
                    image_coordinates=(arms_dev["CORRECT"],),
                    timestep=t0,
                    size_scale=size_scale_t,
                    aspect=aspect_t,
                    growth_alpha=alpha,
                )
                conditioning = composite.forward_conditioning(inputs_base)
                for k in STRATA:
                    collect(u, b, arms_dev, composite, conditioning, x0, inputs_base, k)
                if fp_before is None:
                    fp_before = _fingerprint(composite)  # after first unit (warm)
            fp_after = _fingerprint(composite)
            diff = _fingerprint_diff(fp_before or {}, fp_after)
            if diff["param"]:
                param_mutation.append(f"{ck_name}: {diff['param'][:5]}")
                log(f"PARAMETER MUTATION in {ck_name}: {diff['param'][:5]}")
            if diff["buffer"]:
                buffer_mutation.append(f"{ck_name}: {diff['buffer'][:5]}")
                log(f"BUFFER MUTATION in {ck_name}: {diff['buffer'][:5]}")
            del fp_before, fp_after, composite
            torch.cuda.empty_cache()

        log("=== P3 RUN A (fresh POST load) ===")
        p3_a = run_p3("POST")
        log("=== P3 RUN B (reloaded POST) ===")
        p3_b = run_p3("POST")
    rng_after = _rng_hashes(device)

    # ---- aggregation ----
    def pool(blocks: list[dict], key: str) -> np.ndarray:
        return np.concatenate([np.asarray(bd[key], dtype=np.float64) for bd in blocks]) if blocks else np.array([])

    p0_d = pool(p0, "deltas")
    p0_ref = np.asarray([bd["ref"] for bd in p0], dtype=np.float64).repeat(N_P0 - 1)
    p0_rel = p0_d / np.maximum(p0_ref, 1e-12) if p0_d.size else np.array([])
    p0_bitexact_rate = float(sum(bd["bitexact"] for bd in p0) / max(sum(bd["n"] for bd in p0), 1))
    p0_max_rel = float(np.abs(p0_rel).max()) if p0_rel.size else 0.0

    rng_stat = np.random.default_rng(BOOT_SEED)
    p0_slope_vals = np.asarray([
        float(np.polyfit(np.arange(1, N_P0), bd["deltas"], 1)[0]) for bd in p0
    ])
    p1c_d = pool(p1c, "c_deltas")
    p1_slope_vals = np.asarray([
        float(np.polyfit(np.arange(1, 4), bd["c_deltas"], 1)[0]) for bd in p1c
    ])
    p2c_d = pool(p2c, "c_deltas")
    p2_slope_vals = np.asarray([
        float(np.polyfit(np.arange(1, 4), bd["c_deltas"], 1)[0]) for bd in p2c
    ])
    p2s_d = pool(p2s, "same_deltas")

    p0_slope_ci = _block_slope_ci(p0_slope_vals, rng_stat) if p0_slope_vals.size >= 2 else [float("nan"), float("nan")]
    p0_slope_floor = (p0_slope_ci[0], p0_slope_ci[1])
    p0_p99 = float(np.percentile(np.abs(p0_d), 99.0)) if p0_d.size else float("nan")

    p0_stat = _mean_stat(p0_d, rng_stat)
    p1_stat = _mean_stat(p1c_d, rng_stat)
    p2_stat = _mean_stat(p2c_d, rng_stat)
    p2s_stat = _mean_stat(p2s_d, rng_stat)
    p1_slope_ci = _block_slope_ci(p1_slope_vals, rng_stat) if p1_slope_vals.size >= 2 else [float("nan"), float("nan")]
    p2_slope_ci = _block_slope_ci(p2_slope_vals, rng_stat) if p2_slope_vals.size >= 2 else [float("nan"), float("nan")]

    def ratio(v: float) -> float:
        if np.isnan(v) or np.isnan(p0_p99) or p0_p99 <= 0:
            return float("nan")
        return float(v / p0_p99)

    p1_p99_ratio = ratio(p1_stat["abs_p99"] or 0.0)
    p2_p99_ratio = ratio(p2_stat["abs_p99"] or 0.0)
    p2s_p99_ratio = ratio(p2s_stat["abs_p99"] or 0.0)

    def mag_exceeds(r: float) -> bool:
        return bool(not np.isnan(r) and r > MAG_FLOOR_RATIO)

    order_drift = bool(
        (p1_stat["signed_mean_ci95"] and _ci_excl0(p1_stat["signed_mean_ci95"]))
        or (p2_stat["signed_mean_ci95"] and _ci_excl0(p2_stat["signed_mean_ci95"]))
        or (p1_slope_ci[0] == p1_slope_ci[0] and _slope_beyond_floor(p1_slope_ci, p0_slope_floor))
        or (p2_slope_ci[0] == p2_slope_ci[0] and _slope_beyond_floor(p2_slope_ci, p0_slope_floor))
        or mag_exceeds(p1_p99_ratio)
        or mag_exceeds(p2_p99_ratio)
    )
    acc_pos = max(0.0, p0_slope_floor[1]) if p0_slope_floor[1] == p0_slope_floor[1] else 0.0
    accumulating = bool(
        (p1_slope_ci[0] == p1_slope_ci[0] and p1_slope_ci[0] > acc_pos)
        or (p2_slope_ci[0] == p2_slope_ci[0] and p2_slope_ci[0] > acc_pos)
    )

    repeat_jitter = bool(p0_bitexact_rate < 1.0 or p0_max_rel > P1_REL_CAP_DIAGNOSTIC)
    param_mut = bool(param_mutation)
    buf_mut = bool(buffer_mutation)
    coord_mut = bool(not arms_stable)

    state = v2c.classify_microprobe_state(
        coordinate_map_mutated=coord_mut,
        persistent_buffer_mutated=buf_mut,
        parameter_mutated=param_mut,
        order_dependent_drift=order_drift,
        accumulating_drift=accumulating,
        repeat_numeric_jitter_present=repeat_jitter,
    )

    # ---- P3 analysis ----
    p3_analysis: dict = {}
    if p3_a and p3_b and len(p3_a) == len(p3_b):
        ra = np.asarray([r["losses"] for r in p3_a], dtype=np.float64)
        rb = np.asarray([r["losses"] for r in p3_b], dtype=np.float64)
        n_u, pos_n = ra.shape
        diff_curve = rb - ra
        rng_p3 = np.random.default_rng(BOOT_SEED)
        idx = rng_p3.integers(0, n_u, size=(N_BOOT, n_u))
        pos_ci = diff_curve[idx].mean(axis=1)
        p0_floor_abs = float(np.abs(p0_d).max()) if p0_d.size else 0.0
        positions = []
        flag = False
        for j in range(pos_n):
            lo, hi = float(np.percentile(pos_ci[:, j], 2.5)), float(np.percentile(pos_ci[:, j], 97.5))
            systematic = (lo > 0.0 or hi < 0.0) and abs((lo + hi) / 2.0) > 0.25 * max(p0_floor_abs, 1e-30)
            if systematic:
                flag = True
            positions.append({
                "pos": int(j),
                "mean_A": float(ra[:, j].mean()),
                "mean_B": float(rb[:, j].mean()),
                "delta_B_minus_A": float(diff_curve[:, j].mean()),
                "ci95": [lo, hi],
                "systematic": bool(systematic),
            })
        ff = diff_curve[:, 0]
        ff_ci = _ci95(ff[idx].mean(axis=1))
        p3_analysis = {
            "n_units": int(n_u),
            "units": [r["unit"] for r in p3_a],
            "positions": positions,
            "first_forward_delta": {"point": float(ff.mean()), "ci95": ff_ci},
            "state_history_effect": bool(flag),
            "p0_floor_abs_max": p0_floor_abs,
            "note": "no bitexact requirement across reload; systematic state-history effect = a sequence position whose B-A delta CI excludes 0 beyond 0.25x the P0 abs floor",
        }

    rng_consumption = rng_before != rng_after

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
            "method": "round-robin over 6 cells (mild/medium/strong x low/high), unit id ascending (IDENTICAL to the V1 probe)",
        },
        "strata": list(STRATA),
        "t_values": {str(k): tvals[cam_units[0]][k] for k in STRATA},
        "checkpoints": ck_identity,
        "sequences": {
            "P0": "CORRECT x8 (immediate repeats; runtime nondeterminism floor; P0 by itself NEVER judges hidden state)",
            "P1": "CORRECT IDENTITY CORRECT IDENTITY CORRECT IDENTITY CORRECT (C-position drift vs call index)",
            "P2": "CORRECT SAME CORRECT SAME CORRECT SAME CORRECT (SAME==CORRECT exact; C positions + SAME positions)",
            "P3": "fresh POST load RUN A vs del+empty_cache+reload RUN B; first 8 camera + 8 ordinary units; stratum 0; per unit: CORRECT x4, C ID C ID C, C SAME C SAME C (14 positions)",
        },
        "same_exact_equal_correct": same_exact,
        "arms_stable": arms_stable,
        "p0": {
            "n_blocks": len(p0),
            "n_pairs": int(p0_d.size),
            "pred_bitexact_rate": p0_bitexact_rate,
            "bitexact_all": p0_bitexact_rate == 1.0,
            "signed_mean": p0_stat["signed_mean"],
            "signed_mean_ci95": p0_stat["signed_mean_ci95"],
            "abs_p50": _pct(np.abs(p0_d), 50) if p0_d.size else None,
            "abs_p90": _pct(np.abs(p0_d), 90) if p0_d.size else None,
            "abs_p95": _pct(np.abs(p0_d), 95) if p0_d.size else None,
            "abs_p99": p0_p99,
            "abs_max": float(np.abs(p0_d).max()) if p0_d.size else None,
            "rel_p99": float(np.percentile(np.abs(p0_rel), 99.0)) if p0_rel.size else None,
            "max_abs_loss_delta_rel": p0_max_rel,
            "pred_maxabs": max((bd["pred_maxabs"] for bd in p0), default=None),
            "pred_relrms_max": max((bd["pred_relrms_max"] for bd in p0), default=None),
            "slope": {"point": float(p0_slope_vals.mean()) if p0_slope_vals.size else None, "ci95": p0_slope_ci, "n_blocks": int(p0_slope_vals.size)},
        },
        "p1": {
            "n_blocks": len(p1c),
            "c_signed": p1_stat,
            "c_slope": {"point": float(p1_slope_vals.mean()) if p1_slope_vals.size else None, "ci95": p1_slope_ci, "n_blocks": int(p1_slope_vals.size)},
            "p99_ratio_vs_p0": p1_p99_ratio,
            "identity_shift_mean": float(np.mean(np.concatenate([bd["identity_shifts"] for bd in p1c]))) if p1c else None,
            "c_pred_relrms_max": max((max(bd["c_pred_relrms"]) for bd in p1c), default=None),
            "slope_beyond_p0_floor": bool(p1_slope_ci[0] == p1_slope_ci[0] and _slope_beyond_floor(p1_slope_ci, p0_slope_floor)),
            "signed_mean_ci_excl0": bool(p1_stat["signed_mean_ci95"] and _ci_excl0(p1_stat["signed_mean_ci95"])),
            "magnitude_exceeds_p0_floor": mag_exceeds(p1_p99_ratio),
        },
        "p2": {
            "n_blocks": len(p2c),
            "c_signed": p2_stat,
            "c_slope": {"point": float(p2_slope_vals.mean()) if p2_slope_vals.size else None, "ci95": p2_slope_ci, "n_blocks": int(p2_slope_vals.size)},
            "p99_ratio_vs_p0": p2_p99_ratio,
            "same_vs_first_c": p2s_stat,
            "same_p99_ratio_vs_p0": p2s_p99_ratio,
            "slope_beyond_p0_floor": bool(p2_slope_ci[0] == p2_slope_ci[0] and _slope_beyond_floor(p2_slope_ci, p0_slope_floor)),
            "signed_mean_ci_excl0": bool(p2_stat["signed_mean_ci95"] and _ci_excl0(p2_stat["signed_mean_ci95"])),
            "magnitude_exceeds_p0_floor": mag_exceeds(p2_p99_ratio),
        },
        "p3": p3_analysis or {"status": "not_run"},
        "parameter_mutation": {"detected": param_mut, "details": param_mutation},
        "persistent_buffer_mutation": {"detected": buf_mut, "details": buffer_mutation},
        "state": state,
        "repeat_numeric_jitter_present": repeat_jitter,
        "hidden_mutable_state_detected": state["HIDDEN_MUTABLE_STATE_DETECTED"],
        "rng": {
            "before": rng_before,
            "after": rng_after,
            "consumption_detected": bool(rng_consumption),
            "note": "RNG consumption is recorded and must be explained, but is not by itself a hidden-state or training-mechanism failure (section 16)",
        },
        "thresholds": {
            "p1_rel_cap_diagnostic": P1_REL_CAP_DIAGNOSTIC,
            "mag_floor_ratio": MAG_FLOOR_RATIO,
            "boot_seed": BOOT_SEED,
            "n_boot": N_BOOT,
        },
        "interpretation": (
            "HIDDEN MUTABLE STATE EVIDENCE PRESENT - see state.reasons (B/C/D/E)"
            if state["HIDDEN_MUTABLE_STATE_DETECTED"]
            else (
                "No hidden mutable-state evidence (B/C/D/E all False). "
                + ("Repeat numeric jitter is present (A) and is reported as a diagnostic runtime floor ONLY; per V2 semantics it does not block numerics."
                   if repeat_jitter
                   else "Repeats are effectively bitexact; the runtime floor is negligible.")
            )
        ),
    }

    # defensive: verify the V1 probe json is still untouched
    if V1_PROBE_JSON.is_file():
        h = hashlib.sha256(V1_PROBE_JSON.read_bytes()).hexdigest()
        if h != V1_PROBE_SHA:
            raise RuntimeError(f"V1 probe json sha changed: {h} != {V1_PROBE_SHA}")
    with open(PROBE_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    log(f"probe v2 written to {PROBE_JSON}")
    log(f"hidden_mutable_state={state['HIDDEN_MUTABLE_STATE_DETECTED']} jitter={repeat_jitter} p0_bitexact={p0_bitexact_rate:.4f} p0_max_rel={p0_max_rel:.3e} order_drift={order_drift} accumulating={accumulating} param_mut={param_mut} buf_mut={buf_mut} rng_consumed={rng_consumption}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
