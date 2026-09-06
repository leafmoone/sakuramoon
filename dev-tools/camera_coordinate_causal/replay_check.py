"""Read-only replay of the final Camera Coordinate Causal audit core numbers.

Recomputes, from the FINAL stage2 ledgers (results-worker{0,1}.pt) and the
stage1 unit manifest, using the exact aggregation semantics of the final
cc_stage3.py (v5) snapshot (see final_snapshot/cc_stage3.py):

  * per-unit stratum means: cam_loss[ck][arm][i] = mean(row['loss'][arm])
  * camera margins: M_{arm}_raw = nanmean((w - c)[sel]) with sel = opp_mask
    for OPPOSITE and ~isnan(w) otherwise
  * cluster bootstrap: seed 20260906, n=10000, units = clusters, v5 finite
    mask (a unit is dropped unless its margin is finite at ALL checkpoints),
    resample WITHIN the selected subspace (rng.integers(0, k, (n_boot, k))),
    call order OPPOSITE -> IDENTITY -> SHUFFLED (the rng stream is
    order-sensitive, so the order is part of the contract), paired deltas

Every value is compared EXACTLY (full float64 equality after the JSON
round-trip) against the committed report
reports/camera-coordinate-causal-audit.json and written to
/tmp/camera-causal-review-replay.json.

Exit codes: 0 = REPLAY PASS (all values equal the report),
            2 = REPLAY FAIL (mismatch),
            3 = REPLAY = SKIPPED_NOT_REQUIRED (stage2 ledgers not present).

This script is read-only: it never writes into the repository, never touches
training/checkpoints/exposure state. It is a review-time verification aid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MASTER_SEED = 20260906
CKS = ("PRE", "MID", "POST")
CAM_ARMS = ("CORRECT", "IDENTITY", "OPPOSITE", "SHUFFLED", "HALF", "OVER")
REQUIRED_ARMS = ("OPPOSITE", "IDENTITY", "SHUFFLED")
N_BOOT = 10000


def ci95(samples: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def load_all(out: Path):
    manifest_path = out / "stage1-manifest.json"
    p0 = out / "results-worker0.pt"
    p1 = out / "results-worker1.pt"
    if not (manifest_path.is_file() and p0.is_file() and p1.is_file()):
        return None
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    meta = {m["unit"]: m for m in manifest["units"]}
    done: dict = {}
    for w in (0, 1):
        part = torch.load(out / f"results-worker{w}.pt", map_location="cpu", weights_only=False)
        for u, rows in part.items():
            done.setdefault(int(u), {}).update(rows)
    units = sorted(meta.keys())
    missing = [u for u in units if any(ck not in done[u] for ck in CKS)]
    if missing:
        raise RuntimeError(str(len(missing)) + " units missing checkpoint rows, e.g. " + str(missing[:5]))
    camera_units = [u for u in units if meta[u]["cohort"] == "camera"]
    return manifest, done, camera_units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-out", default="/tmp/camera-coordinate-causal")
    ap.add_argument("--repo", default="/sakuramoon-runtime/sakuramoon-camera-causal-review")
    ap.add_argument("--report-out", default="/tmp/camera-causal-review-replay.json")
    args = ap.parse_args()

    out = Path(args.audit_out)
    loaded = load_all(out)
    if loaded is None:
        print("REPLAY = SKIPPED_NOT_REQUIRED (stage2 ledgers not present)")
        return 3
    manifest, done, camera_units = loaded
    meta = {m["unit"]: m for m in manifest["units"]}
    n_cam = len(camera_units)
    assert camera_units == list(range(n_cam)), "camera units must be 0..C-1"

    # ---- exact cc_stage3 v5 aggregation (final_snapshot/cc_stage3.py) ----
    cam_loss = {ck: {a: np.full(n_cam, np.nan) for a in CAM_ARMS} for ck in CKS}
    for i, u in enumerate(camera_units):
        for ck in CKS:
            row = done[u][ck]
            for a in CAM_ARMS:
                if a in row["loss"]:
                    cam_loss[ck][a][i] = np.mean(row["loss"][a])
    opp_mask = np.array([not meta[u]["opp_na"] for u in camera_units], dtype=bool)

    replay: dict = {"point_estimates": {}, "bootstrap": {}}
    for ck in CKS:
        cor = cam_loss[ck]["CORRECT"]
        for a in REQUIRED_ARMS:
            w = cam_loss[ck][a]
            sel = opp_mask if a == "OPPOSITE" else ~np.isnan(w)
            replay["point_estimates"][a + "|" + ck] = float(np.nanmean((w - cor)[sel]))

    rng = np.random.default_rng(MASTER_SEED)

    def boot_margin(arm: str, sel: np.ndarray) -> dict:
        valid = sel.copy()
        for ck in CKS:
            valid &= np.isfinite(cam_loss[ck][arm] - cam_loss[ck]["CORRECT"])
        sel_idx = np.flatnonzero(valid)
        k = int(sel_idx.size)
        if k < 2:
            raise RuntimeError("arm " + arm + ": k=" + str(k) + " < 2")
        boot_idx_sub = rng.integers(0, k, size=(N_BOOT, k))
        vals: dict = {}
        for ck in CKS:
            w = cam_loss[ck][arm]
            c = cam_loss[ck]["CORRECT"]
            m = (w - c)[sel_idx]
            vals[ck] = {"mean": float(m.mean()), "boot": m[boot_idx_sub].mean(axis=1)}
        d_pm = vals["POST"]["boot"] - vals["PRE"]["boot"]
        d_mm = vals["MID"]["boot"] - vals["PRE"]["boot"]
        d_mp = vals["POST"]["boot"] - vals["MID"]["boot"]
        result = {ck: {"mean": vals[ck]["mean"], "ci95": ci95(vals[ck]["boot"])} for ck in CKS}
        result["delta_POST_PRE"] = {
            "point": float(vals["POST"]["boot"].mean() - vals["PRE"]["boot"].mean()),
            "ci95": ci95(d_pm),
        }
        result["delta_MID_PRE"] = {
            "point": float(vals["MID"]["boot"].mean() - vals["PRE"]["boot"].mean()),
            "ci95": ci95(d_mm),
        }
        result["delta_POST_MID"] = {
            "point": float(vals["POST"]["boot"].mean() - vals["MID"]["boot"].mean()),
            "ci95": ci95(d_mp),
        }
        return result

    # EXACT final call order (the rng stream is order-sensitive)
    ones = np.ones(n_cam, dtype=bool)
    replay["bootstrap"]["M_OPPOSITE_raw"] = boot_margin("OPPOSITE", opp_mask)
    replay["bootstrap"]["M_IDENTITY_raw"] = boot_margin("IDENTITY", ones)
    replay["bootstrap"]["M_SHUFFLED_raw"] = boot_margin("SHUFFLED", ones)

    # ---- exact comparison against the committed report ----
    with open(Path(args.repo) / "reports/camera-coordinate-causal-audit.json", encoding="utf-8") as fh:
        audit = json.load(fh)
    problems: list[str] = []
    for key, val in replay["point_estimates"].items():
        arm, ck = key.split("|")
        rep = audit["camera_metrics"][ck]["M_" + arm + "_raw"]
        if val != rep:
            problems.append("point " + key + ": replay=" + repr(val) + " report=" + repr(rep) + " abs=" + format(abs(val - rep), ".3e"))
    for arm in REQUIRED_ARMS:
        rb = replay["bootstrap"]["M_" + arm + "_raw"]
        ab = audit["bootstrap"]["M_" + arm + "_raw"]
        for ck in CKS:
            if rb[ck]["mean"] != ab[ck]["mean"]:
                problems.append("boot mean " + arm + " " + ck + ": " + repr(rb[ck]["mean"]) + " vs " + repr(ab[ck]["mean"]))
            if list(rb[ck]["ci95"]) != list(ab[ck]["ci95"]):
                problems.append("boot ci95 " + arm + " " + ck + ": " + repr(rb[ck]["ci95"]) + " vs " + repr(ab[ck]["ci95"]))
        for dkey in ("delta_POST_PRE", "delta_MID_PRE", "delta_POST_MID"):
            if rb[dkey]["point"] != ab[dkey]["point"]:
                problems.append("boot " + dkey + " point " + arm + ": " + repr(rb[dkey]["point"]) + " vs " + repr(ab[dkey]["point"]))
            if list(rb[dkey]["ci95"]) != list(ab[dkey]["ci95"]):
                problems.append("boot " + dkey + " ci95 " + arm + ": " + repr(rb[dkey]["ci95"]) + " vs " + repr(ab[dkey]["ci95"]))

    replay["n_camera_units"] = n_cam
    replay["n_opp_mask_true"] = int(opp_mask.sum())
    replay["n_opp_na"] = int((~opp_mask).sum())
    replay["bootstrap_seed"] = MASTER_SEED
    replay["n_boot"] = N_BOOT
    Path(args.report_out).write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if problems:
        print("REPLAY = FAIL")
        for p in problems[:20]:
            print("  ", p)
        return 2
    print("REPLAY = PASS: 9 point estimates + 3 arms x (3 checkpoint means+ci95 + 3 delta point+ci95), all exactly equal to the committed report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
