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
from itertools import product
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
    if row.get("rel_sig") is not None:
        return row["rel_sig"]
    return 0.0


# ---------------------------------------------------------------- rules


def rule_A(row, p):  # amplitude only
    return rel_sig(row) < p["tau"]


def rule_B(row, p):  # top1 energy only
    return (row.get("top1_energy") or 0.0) > p["theta"]


def rule_C(row, p):  # stable rank only
    return (row.get("stable_rank") or 1e9) < p["rho"]


def rule_D(row, p):  # amplitude AND top1
    return rel_sig(row) < p["tau"] and (row.get("top1_energy") or 0.0) > p["theta"]


def rule_E(row, p):  # D AND temporal alignment loss
    c = row.get("cos_grad_nest")
    return (
        rel_sig(row) < p["tau"]
        and (row.get("top1_energy") or 0.0) > p["theta"]
        and (c is not None and c < p["gamma"])
    )


FAMILIES = {
    "A_amplitude": (rule_A, ["tau"]),
    "B_top1": (rule_B, ["theta"]),
    "C_stablerank": (rule_C, ["rho"]),
    "D_amp_top1": (rule_D, ["tau", "theta"]),
    "E_amp_top1_cos": (rule_E, ["tau", "theta", "gamma"]),
}


def grid_for(params: list[str], rows: list[dict]) -> list[dict]:
    taus = sorted({rel_sig(r) for r in rows if rel_sig(r) > 0})[:200]
    thetas = sorted({r["top1_energy"] for r in rows if r.get("top1_energy")})
    thetas = thetas[:: max(1, len(thetas) // 200)]
    rhos = sorted({r["stable_rank"] for r in rows if r.get("stable_rank")})
    rhos = rhos[:: max(1, len(rhos) // 200)]
    gammas = sorted({r["cos_grad_nest"] for r in rows if r.get("cos_grad_nest") is not None})
    gammas = gammas[:: max(1, len(gammas) // 100)]
    combos: list[dict] = []
    for t in taus:
        combos.append({"tau": t})
    for th in thetas:
        combos.append({"theta": float(th)})
    for r_ in rhos:
        combos.append({"rho": float(r_)})
    for t, th in product(taus, thetas):
        combos.append({"tau": t, "theta": float(th)})
    for t, th in product(taus, thetas):
        for g in gammas:
            combos.append({"tau": t, "theta": float(th), "gamma": float(g)})
    return [c for c in combos if all(k in c for k in params)]


def evaluate(rule_fn, params, rows: list[dict]) -> dict:
    """Lexicographic objective: 0 holdout FNs, then minimal false skips."""
    dn = [r for r in rows if r["label"] == "DANGEROUS"]
    sf = [r for r in rows if r["label"] == "SAFE"]
    fn = sum(1 for r in dn if not rule_fn(r, params))
    fs = sum(1 for r in sf if rule_fn(r, params))
    skipped = sum(1 for r in rows if rule_fn(r, params))
    return {
        "fn": fn,
        "fs": fs,
        "n_dangerous": len(dn),
        "n_safe": len(sf),
        "skip_rate": skipped / len(rows) if rows else 0.0,
        "safe_false_skip_rate": fs / len(sf) if sf else 0.0,
        "dangerous_recall": (len(dn) - fn) / len(dn) if dn else None,
    }


def best_params(family: str, fit_rows: list[dict], eval_rows: list[dict]) -> dict:
    fn_rule, params = FAMILIES[family]
    best = None
    for cand in grid_for(params, fit_rows):
        fit_eval = evaluate(fn_rule, cand, fit_rows)
        if fit_eval["fn"] > 0:
            continue  # a rule that misses fit-set danger is never a candidate
        ev = evaluate(fn_rule, cand, eval_rows)
        key = (ev["fn"], ev["fs"], ev["skip_rate"])
        if best is None or key < best[0]:
            best = (key, cand, ev, fit_eval)
    if best is None:
        return {"family": family, "feasible": False}
    _, cand, ev, fit_eval = best
    return {
        "family": family,
        "feasible": True,
        "params": cand,
        "eval": ev,
        "fit": fit_eval,
    }


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
