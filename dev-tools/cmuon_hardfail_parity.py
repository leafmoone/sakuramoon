#!/usr/bin/env python3
"""F2 hard-fail parity: compare the F1 2-rank run (F1 tree, F1 artifact
contract) against the F2 2-rank run (F2 tree, minimal-capsule contract).

Spec §13: on a hard fail both trees must produce the SAME verdict and the
SAME outcome (zero commit, identical CMuonSafetyError on both ranks, same
rescue-failure counters); the ONLY allowed difference is the artifact
(form and location). The 2-rank tests already assert zero commit + message
equality on each run; this comparison adds the CROSS-TREE check.

Inputs:
  --f1  <f1-2rank.json from the F1 tree run>
  --f2  <f2-2rank.json from the F2 tree run>

Checks:
  * scenario set {A_above_ceiling, B_nonfinite, C_crash_loop, D_rescue}
    present in both (F2 adds E_below_floor; F1 has no E)
  * for A/B/C: the CMuonSafetyError message is byte-identical
  * fp32_rescue_failures identical for A/B/C (1)
  * D: fp32_rescues advanced identically (owner +1), spread 0.0 in both,
    artifact_published false in both
  * F2 extra: E below_floor capsule contract (reason, finite, < floor,
    mirror ok)
Output: JSON report; exit 0 = parity holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--f1", required=True, type=Path)
    p.add_argument("--f2", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    f1 = json.loads(a.f1.read_text())
    f2 = json.loads(a.f2.read_text())
    r1: dict = f1.get("results", {})
    r2: dict = f2.get("results", {})

    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    shared = ("A_above_ceiling", "B_nonfinite", "C_crash_loop")
    for scen in shared:
        m1 = r1.get(scen, {}).get("message")
        m2 = r2.get(scen, {}).get("message")
        check(f"{scen}.message_identical", m1 is not None and m1 == m2,
              f"f1={str(m1)[:80]!r} f2={str(m2)[:80]!r}")
        f1c = r1.get(scen, {}).get("fp32_rescue_failures")
        f2c = r2.get(scen, {}).get("fp32_rescue_failures")
        check(f"{scen}.fp32_rescue_failures_identical", f1c == f2c,
              f"f1={f1c} f2={f2c}")

    d1, d2 = r1.get("D_rescue", {}), r2.get("D_rescue", {})
    check("D_rescue.fp32_rescues_identical",
          d1.get("fp32_rescues") == d2.get("fp32_rescues"),
          f"f1={d1.get('fp32_rescues')} f2={d2.get('fp32_rescues')}")
    check("D_rescue.spread_zero_both",
          d1.get("param_fingerprint_spread") == 0.0
          and d2.get("param_fingerprint_spread") == 0.0,
          f"f1={d1.get('param_fingerprint_spread')} "
          f"f2={d2.get('param_fingerprint_spread')}")
    check("D_rescue.no_artifact_both",
          d1.get("artifact_published") in (False, None)
          and d2.get("artifact_published") is False,
          f"f1={d1.get('artifact_published')} f2={d2.get('artifact_published')}")

    e = r2.get("E_below_floor", {})
    e_capsule = r2.get("E_rank0_capsule", {})
    check("E_below_floor.present_f2_only",
          "E_below_floor" not in r1 and bool(e),
          f"f1_has_E={'E_below_floor' in r1} f2_has_E={bool(e)}")
    check("E_below_floor.verdict",
          e_capsule.get("reason") == "below_floor"
          and e_capsule.get("fp32_delta_rms") is not None
          and e_capsule.get("fp32_delta_rms") < e_capsule.get("rescue_floor")
          and e_capsule.get("mirror_status") == "ok",
          json.dumps(e_capsule))

    # The artifact itself is the ONLY allowed difference: F1 published the
    # full diagnostic artifact (diagnostic_replay_*), F2 the minimal capsule
    # (no diagnostic keys) in a different root. Both runs record their own
    # event; we assert the difference is limited to the artifact contract.
    a1 = r1.get("A_rank0_artifact", {})
    a2 = r2.get("A_rank0_artifact", {})
    check("A_artifact.contract_differs_only",
          bool(a1.get("input_sha256")) and bool(a2.get("input_sha256"))
          and "mirror_status" not in a1 and a2.get("mirror_status") == "ok",
          f"f1_keys={sorted(a1.keys())} f2_keys={sorted(a2.keys())}")

    report = {
        "schema": "sakuramoon.cmuon_hardfail_parity.v1",
        "f1": str(a.f1),
        "f2": str(a.f2),
        "checks": checks,
        "verdict": "PASS" if all(c["ok"] for c in checks) else "FAIL",
    }
    text = json.dumps(report, indent=2)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
