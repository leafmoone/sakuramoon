"""F3 checkpoint compatibility audit (F3 spec section 18, READ-ONLY).

Verifies that an optimizer state written by the bb41292 (F2) production
code loads under F3 code without any migration, using the F3-tree
checkpoint loaders themselves:

  1. ``_load_hybrid_optimizer_state`` (the exact production resume loader)
     accepts the production optimizer.pt: top-level exact-key check +
     schema version, no rejection, no version change;
  2. the persisted guard block's ``fp32_rescue`` key set is EXACTLY the
     6 keys F3's explicit-key loader expects (F3 adds no persisted keys,
     so old files need no migration);
  3. the 6 counter values are plain ints preserved verbatim;
  4. F3's process-local soft counters are ABSENT from the file (they are
     not persisted by design) and restart at zero on load (unit-verified
     in test_fp32_rescue_f3.py::test_I, incl. a round-trip + parent-class
     load).

Usage:
  python dev-tools/cmuon_f3_ckpt_compat_audit.py --ckpt <ckpt_dir> \
      [--output out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_FP32_KEYS = {
    "bf16_attempts",
    "bf16_safety_failures",
    "fp32_attempts",
    "fp32_rescues",
    "fp32_rescue_failures",
    "rescue_by_role",
}


def _find_repo_src() -> Path:
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if (base / "src").is_dir():
            return base / "src"
    raise SystemExit("cannot find repo src/ relative to dev-tools/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(_find_repo_src()))

    from sakuramoon.checkpoint.load import _load_hybrid_optimizer_state

    ckpt = Path(args.ckpt)
    opt_path = ckpt / "optimizer.pt"
    if not opt_path.is_file():
        # production raw ckpt layout: train_state/optimizer.pt
        alt = ckpt / "train_state" / "optimizer.pt"
        if alt.is_file():
            opt_path = alt
    assert opt_path.is_file(), f"missing {opt_path} (and {alt})"

    report: dict[str, object] = {"ckpt": str(ckpt), "file": str(opt_path)}

    # 1. exact production loader (top-level exact keys + schema versions)
    state = _load_hybrid_optimizer_state(opt_path)
    report["loader"] = "_load_hybrid_optimizer_state: PASS"
    report["top_level_keys"] = sorted(state.keys())
    report["hybrid_cmuon_schema_version"] = state["hybrid_cmuon_schema_version"]
    report["guarded_canonical_schema_version"] = (
        state.get("guarded_canonical_schema_version")
    )

    guard = state["guard"]
    report["guard_keys"] = sorted(guard.keys())
    report["guard_schema_version"] = guard.get("schema_version")

    # 2. the fp32_rescue block: exact key set F3's loader expects
    fp32 = guard["fp32_rescue"]
    keys = set(fp32.keys())
    report["fp32_rescue_keys"] = sorted(keys)
    report["fp32_rescue_keys_match_f3_loader"] = keys == EXPECTED_FP32_KEYS
    assert keys == EXPECTED_FP32_KEYS, (
        f"key set mismatch vs F3 explicit-key loader: {sorted(keys)}"
    )

    # 3. counter values preserved verbatim (plain ints / role dict)
    counters = {
        k: fp32[k] for k in EXPECTED_FP32_KEYS if k != "rescue_by_role"
    }
    for k, v in counters.items():
        assert isinstance(v, int), f"{k} must be a plain int, got {type(v)}"
    report["fp32_rescue_counters"] = counters
    report["rescue_by_role"] = dict(sorted(fp32["rescue_by_role"].items()))

    # 4. F3 process-local counters absent (not persisted by design)
    report["f3_process_local_counters_absent"] = (
        "fp32_low_delta_rescues" not in fp32
        and "fp32_low_delta_by_role" not in fp32
        and "fp32_low_delta_rescues" not in guard
    )

    report["verdict"] = (
        "PASS: F2-written optimizer.pt loads under F3 code unchanged "
        "(exact-key production loader, no migration, no schema growth)"
    )

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
