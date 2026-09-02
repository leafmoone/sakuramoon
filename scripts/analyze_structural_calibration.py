#!/usr/bin/env python3
"""Offline analysis of the structural/SNR pre-NS classifier calibration
(D1 round, 08-31, spec sections 5-8).

Inputs: the per-rank JSONL files from the shadow run (features + K-run
NS4 safety labels) plus the P3 refs file.

Pipeline:
  1. Load all observations from all ranks.  Features are deterministic
     for identical chunks (verified by the run's cross-rank consistency
     gate), so per (obs, fqn, chunk) we MERGE the ranks: label runs are
     pooled (K per rank), hazard = pooled catastrophic rate, label =
     DANGEROUS if any pooled run is catastrophic.
  2. Rule families (deterministic, interpretable only):
       A amplitude only:        rel_sig < tau
       B top1 only:             top1_energy > theta
       C stable-rank only:      stable_rank < rho
       D amplitude + top1:      rel_sig < tau AND top1_energy > theta
       E amplitude + top1 + temporal: D AND cos_grad_nest < gamma
     Thresholds are grid-swept; the objective is lexicographic:
       (a) dangerous false negatives on the HOLDOUT = 0,
       (b) then minimal safe false-skip count on the holdout.
  3. Holdout protocols (spec section 8 — never select and evaluate on
     the same events):
       primary:  SLOT holdout — rules are fit on rows EXCLUDING the
                 known-danger slots (slot_07/slot_08), then evaluated on
                 them; also reported for the non-held-out remainder.
       secondary: TIME holdout — fit on the first half of observations,
                 evaluate on the second half (and the reverse), full data.
  4. Counterexamples: SAFE rows with top1 above the dangerous threshold
     and DANGEROUS rows below it (the classifier-killer set).
  5. Grouping: per role / per slot / per shape — is the 88.9% top-1
     q_proj a cluster or an outlier?
  6. Cost aggregation from the JSONL cost_ms fields.
  7. Amplitude-floor baseline (the 64-70% skip of the failed v1) for
     comparison.

Output: a JSON report + a human-readable markdown summary on stdout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(paths: list[str]) -> list[dict]:
    """Merge rank JSONLs into per-(obs, fqn, chunk) rows with pooled runs."""
    by_key: dict[tuple[int, str, int], dict] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            obs = rec["obs"]
            rank = rec.get("rank", 0)
            for row in rec["rows"]:
                key = (obs, row["fqn"], row["chunk"])
                if key not in by_key:
                    entry = dict(row)
                    entry["runs"] = list(row["ns_runs"])
                    entry["obs"] = obs
                    entry["ranks"] = {rank}
                    by_key[key] = entry
                else:
                    by_key[key]["runs"].extend(row["ns_runs"])
                    by_key[key]["ranks"].add(rank)
    out = []
    for entry in by_key.values():
        n = len(entry["runs"])
        cat = sum(1 for r in entry["runs"] if r["catastrophic"])
        entry["n_runs"] = n
        entry["hazard_rate"] = cat / n if n else 0.0
        entry["label"] = "DANGEROUS" if cat > 0 else "SAFE"
        entry.pop("ns_runs", None)
        entry.pop("runs", None)
        out.append(entry)
    return out


def rel_sig(row: dict) -> float:
    v = row.get("rel_sig")
    return v if v is not None else 0.0


# ---------------------------------------------------------------- rules
#
# Rule families (deterministic, interpretable; spec section 5):
#   A amplitude only:        rel_sig < tau
#   B top1 energy only:      top1_energy > theta
#   C stable-rank only:      stable_rank < rho
#   D amplitude + top1:      A AND B
#   E amplitude + top1 + temporal: D AND (cos_grad_nest is not None
#                                     AND cos_grad_nest < gamma)
#
# Evaluation is bitset-accelerated: each row is a bit of a Python int, a rule
# fires on row i iff bit i is set in its mask, and the family masks compose
# with bitwise AND (1000-bit AND is O(1)), so the full E grid (60x60x40)
# sweeps in well under a second.  A row with a MISSING feature is never
# skippable by any rule that needs that feature (bit cleared in every mask
# for that feature) — a classifier must not act on data it cannot evaluate.


FAMILIES = [
    "A_amplitude",
    "B_top1",
    "C_stablerank",
    "D_amp_top1",
    "E_amp_top1_cos",
]


def quantile_grid(values: list[float], n: int) -> list[float]:
    vals = sorted(set(values))
    if len(vals) <= n:
        return vals
    step = (len(vals) - 1) / (n - 1)
    return [vals[round(i * step)] for i in range(n)]


def thresholds_from(rows: list[dict]) -> dict:
    """Grid values derived from the FIT rows only (no eval leakage)."""
    return {
        "taus": quantile_grid(
            [
                r["rel_sig"]
                for r in rows
                if r.get("rel_sig") is not None and r["rel_sig"] > 0
            ],
            60,
        ),
        "thetas": quantile_grid(
            [r["top1_energy"] for r in rows if r.get("top1_energy") is not None],
            60,
        ),
        "rhos": quantile_grid(
            [r["stable_rank"] for r in rows if r.get("stable_rank") is not None],
            60,
        ),
        "gammas": quantile_grid(
            [r["cos_grad_nest"] for r in rows if r.get("cos_grad_nest") is not None],
            40,
        ),
    }


def masks_for(rows: list[dict], th: dict) -> dict:
    """Fire-bitsets (one int per grid value) computed on this row set."""
    n = len(rows)
    dn = 0
    for i, r in enumerate(rows):
        if r["label"] == "DANGEROUS":
            dn |= 1 << i
    rs = [r.get("rel_sig") for r in rows]
    t1 = [r.get("top1_energy") for r in rows]
    sr = [r.get("stable_rank") for r in rows]
    cg = [r.get("cos_grad_nest") for r in rows]
    return {
        "n": n,
        "dn": dn,
        "T": [
            sum(1 << i for i in range(n) if rs[i] is not None and rs[i] < t)
            for t in th["taus"]
        ],
        "TH": [
            sum(1 << i for i in range(n) if t1[i] is not None and t1[i] > t)
            for t in th["thetas"]
        ],
        "R": [
            sum(1 << i for i in range(n) if sr[i] is not None and sr[i] < r_)
            for r_ in th["rhos"]
        ],
        "G": [
            sum(1 << i for i in range(n) if cg[i] is not None and cg[i] < g)
            for g in th["gammas"]
        ],
    }


def _stats(mask: int, m: dict) -> dict:
    """(fn, fs, skip_rate) for a fire-bitset mask: fn = dangerous rows the
    rule did NOT fire on (missed danger), fs = safe rows it fired on."""
    n = m["n"]
    fired = mask.bit_count()
    dn_hit = (mask & m["dn"]).bit_count()
    n_dn = m["dn"].bit_count()
    return {
        "fn": n_dn - dn_hit,
        "fs": fired - dn_hit,
        "fired": fired,
        "skip_rate": fired / n if n else 0.0,
    }


def best_params(family: str, fit_rows: list[dict], eval_rows: list[dict]) -> dict:
    """Lexicographic objective: 0 FNs on EVAL, then minimal FS on EVAL.

    Candidates must also achieve 0 FNs on the FIT set (a rule that misses
    in-sample danger is never a candidate).  Grid values come from FIT rows
    only; both fit and eval masks use those same values (spec section 8:
    selection and evaluation never share the same events).
    """
    th = thresholds_from(fit_rows)
    F = masks_for(fit_rows, th)
    E = masks_for(eval_rows, th)
    n_dn_eval = E["dn"].bit_count()
    n_dn_fit = F["dn"].bit_count()

    def finalize(params: dict, fstats: dict, estats: dict) -> dict:
        return {
            "family": family,
            "feasible": True,
            "params": params,
            "fit": {
                "fn": fstats["fn"],
                "fs": fstats["fs"],
                "n_dangerous": n_dn_fit,
                "skip_rate": fstats["skip_rate"],
                "dangerous_recall": 1.0,
            },
            "eval": {
                "fn": estats["fn"],
                "fs": estats["fs"],
                "n_dangerous": n_dn_eval,
                "skip_rate": estats["skip_rate"],
                "safe_false_skip_rate": (
                    estats["fs"] / max(1, E["n"] - n_dn_eval)
                ),
                "dangerous_recall": (
                    (n_dn_eval - estats["fn"]) / n_dn_eval if n_dn_eval else None
                ),
            },
        }

    best: tuple[tuple, dict] | None = None

    def consider(params: dict, fmask: int, emask: int) -> None:
        nonlocal best
        fstats = _stats(fmask, F)
        if fstats["fn"] > 0:
            return
        estats = _stats(emask, E)
        key = (estats["fn"], estats["fs"], estats["skip_rate"])
        if best is None or key < best[0]:
            best = (key, finalize(params, fstats, estats))

    if family == "A_amplitude":
        for i, t in enumerate(th["taus"]):
            consider({"tau": t}, F["T"][i], E["T"][i])
    elif family == "B_top1":
        for i, t in enumerate(th["thetas"]):
            consider({"theta": float(t)}, F["TH"][i], E["TH"][i])
    elif family == "C_stablerank":
        for i, r_ in enumerate(th["rhos"]):
            consider({"rho": float(r_)}, F["R"][i], E["R"][i])
    elif family == "D_amp_top1":
        for i, t in enumerate(th["taus"]):
            for j, tt in enumerate(th["thetas"]):
                consider(
                    {"tau": t, "theta": float(tt)},
                    F["T"][i] & F["TH"][j],
                    E["T"][i] & E["TH"][j],
                )
    else:  # E_amp_top1_cos
        for i, t in enumerate(th["taus"]):
            fi = F["T"][i]
            ei = E["T"][i]
            for j, tt in enumerate(th["thetas"]):
                fid = fi & F["TH"][j]
                eid = ei & E["TH"][j]
                if fid & F["dn"]:
                    continue  # D-part already misses fit danger; E can't help
                for k, g in enumerate(th["gammas"]):
                    consider(
                        {"tau": t, "theta": float(tt), "gamma": float(g)},
                        fid & F["G"][k],
                        eid & E["G"][k],
                    )
    if best is None:
        return {"family": family, "feasible": False}
    return best[1]


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-slots", default="slot_07,slot_08")
    args = ap.parse_args()

    refs = {
        (k.rpartition("#chunk")[0], int(k.rpartition("#chunk")[2])): float(v)
        for k, v in json.loads(Path(args.refs).read_text()).items()
    }
    rows = load_rows(args.jsonl)
    for r in rows:
        if r.get("ref") is None:
            r["ref"] = refs.get((r["fqn"], r["chunk"]))
            if r["ref"]:
                r["rel_sig"] = r["rms"] / r["ref"]
    obs_ids = sorted({r["obs"] for r in rows})
    heldout = set(args.heldout_slots.split(","))

    dangerous = [r for r in rows if r["label"] == "DANGEROUS"]
    safe = [r for r in rows if r["label"] == "SAFE"]
    report: dict = {
        "rows": len(rows),
        "observations": obs_ids,
        "n_dangerous": len(dangerous),
        "n_safe": len(safe),
        "dangerous_by_slot": dict(
            Counter(r["slot"] for r in dangerous)
        ),
        "dangerous_by_obs": dict(
            sorted(Counter(r["obs"] for r in dangerous).items())
        ),
        "families": {},
        "counterexamples": {},
        "groups": {},
        "cost": {},
        "baseline_amplitude_floor": {},
    }

    # ---- rule families under both holdout protocols ----
    for family in FAMILIES:
        slot_fit = [r for r in rows if r["slot"] not in heldout]
        slot_hold = [r for r in rows if r["slot"] in heldout]
        half = (obs_ids[len(obs_ids) // 2] + obs_ids[0]) // 2
        time_fit = [r for r in rows if r["obs"] <= half]
        time_hold = [r for r in rows if r["obs"] > half]
        time_fit2 = [r for r in rows if r["obs"] > half]
        time_hold2 = [r for r in rows if r["obs"] <= half]

        slot_primary = best_params(family, slot_fit, slot_hold)
        slot_reverse = best_params(family, slot_hold, slot_fit)
        time_a = best_params(family, time_fit, time_hold)
        time_b = best_params(family, time_fit2, time_hold2)
        report["families"][family] = {
            "slot_holdout_primary": slot_primary,
            "slot_holdout_reverse": slot_reverse,
            "time_holdout_first_half_fit": time_a,
            "time_holdout_second_half_fit": time_b,
        }

    # ---- counterexamples for the D-family best rule (typical winner) ----
    d = report["families"]["D_amp_top1"]["slot_holdout_primary"]
    if d.get("feasible"):
        theta = d["params"]["theta"]
        ce = []
        for r in rows:
            top1 = r.get("top1_energy") or 0.0
            if r["label"] == "SAFE" and top1 > theta:
                ce.append(
                    {
                        "kind": "safe_high_top1",
                        "obs": r["obs"],
                        "fqn": r["fqn"],
                        "slot": r["slot"],
                        "top1": top1,
                        "rel_sig": rel_sig(r),
                    }
                )
            if r["label"] == "DANGEROUS" and top1 < theta:
                ce.append(
                    {
                        "kind": "dangerous_low_top1",
                        "obs": r["obs"],
                        "fqn": r["fqn"],
                        "slot": r["slot"],
                        "top1": top1,
                        "rel_sig": rel_sig(r),
                        "hazard_rate": r["hazard_rate"],
                    }
                )
        report["counterexamples"] = {
            "rule": d["params"],
            "safe_high_top1": [c for c in ce if c["kind"] == "safe_high_top1"],
            "dangerous_low_top1": [
                c for c in ce if c["kind"] == "dangerous_low_top1"
            ],
        }

    # ---- grouping: role / slot / shape ----
    for field in ("role", "slot", "shape"):
        groups: dict[str, dict] = {}
        for r in rows:
            g = r[field] if field != "shape" else "x".join(map(str, r["shape"]))
            e = groups.setdefault(
                g, {"n": 0, "dangerous": 0, "top1_max": 0.0, "rel_sig_max": 0.0}
            )
            e["n"] += 1
            e["dangerous"] += r["label"] == "DANGEROUS"
            e["top1_max"] = max(e["top1_max"], r.get("top1_energy") or 0.0)
            e["rel_sig_max"] = max(e["rel_sig_max"], rel_sig(r))
        report["groups"][field] = dict(sorted(groups.items()))

    # ---- cost ----
    costs = []
    for p in args.jsonl:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            c = rec.get("cost_ms")
            if c:
                costs.append(c)
    if costs:
        keys = ("feat_pi", "feat_reduce", "feat_cos", "ns_label", "step_total")
        report["cost"] = {
            k: sum(c[k] for c in costs) / len(costs) for k in keys
        }
        report["cost"]["observations"] = len(costs)

    # ---- amplitude-floor baseline (v1's 64-70% skip) ----
    for floor in (1.2e-7, 1.35e-7):
        # exact v1 semantics: skip when signal rms < absolute floor
        skip = sum(1 for r in rows if r["rms"] < floor)
        dn_caught = sum(1 for r in dangerous if r["rms"] < floor)
        report["baseline_amplitude_floor"][str(floor)] = {
            "skip_rate": skip / len(rows),
            "dangerous_caught": dn_caught,
            "dangerous_missed": len(dangerous) - dn_caught,
        }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out}")
    print(
        f"rows={len(rows)} obs={len(obs_ids)} dangerous={len(dangerous)} "
        f"safe={len(safe)}"
    )
    for family, res in report["families"].items():
        pr = res["slot_holdout_primary"]
        if pr.get("feasible"):
            ev = pr["eval"]
            print(
                f"{family}: slot-holdout FEASIBLE params={pr['params']} "
                f"FN={ev['fn']} FS={ev['fs']} skip={ev['skip_rate']:.1%} "
                f"dfs_rate={ev['safe_false_skip_rate']:.1%}"
            )
        else:
            print(f"{family}: slot-holdout INFEASIBLE (no rule hits all fit danger)")
    ce = report["counterexamples"]
    if ce.get("safe_high_top1") is not None:
        print(
            "counterexamples: safe_high_top1="
            f"{len(ce['safe_high_top1'])} dangerous_low_top1="
            f"{len(ce['dangerous_low_top1'])}"
        )


if __name__ == "__main__":
    main()
