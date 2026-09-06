"""Read-only restart-readiness audit of the clean iREPA restart candidates.

Audits (no writes, in-memory only):
  1. original no-iREPA ckpt_113400  (p6b-source/ckpt_113400_raw-113400-...)
  2. migrated ckpt_113400_irepa     (p6b/migrated/ckpt_113400_irepa)

Evidence produced (each line PASS/FAIL):
  - trunk model shards bit-identical between original and migrated
  - RNG state (rank-0 + optimizer_sr) bit-identical between original/migrated
  - trainer_state identical: successful_updates == 113400 (zero updates since)
  - migrated optimizer: CMuon FQN set unchanged; AdamW state for every source
    parameter bit-identical after id renumbering; projector FQNs present in
    groups but WITHOUT any AdamW state entry (lazy init = pristine)
  - migrated CMuon/sr_rng/transition blocks bit-identical to source
  - migrated projector shard == deterministic reconstruction from
    migration_seed (20260904) via IRepaAlignment init (bit-exact)
  - irepa_state: source_update 113400, start_successful_update 113401
    (lambda binds exactly to 0.0 at the first update 113401)

Usage:
  PYTHONPATH=src python scripts/diag_restart_readiness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SOURCE = Path("/sakuramoon-runtime/p6b-source/ckpt_113400_raw-113400-update-cadence")
MIGRATED = Path("/sakuramoon-runtime/p6b/migrated/ckpt_113400_irepa")

PROJECTOR_WEIGHT_FQN = "irepa_alignment.projector.weight"
PROJECTOR_BIAS_FQN = "irepa_alignment.projector.bias"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def _bit_equal(a: object, b: object, label: str) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if type(a) is torch.Tensor and type(b) is torch.Tensor:
            return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        # Wrapper Tensor subclass (torchao OptimState8bit): the outer tensor
        # is storageless; the payload is the attrs named in tensor_attrs plus
        # scalar metadata (signed, block_size).  Compare all of them exactly.
        if type(a) is not type(b) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        attrs = getattr(type(a), "tensor_attrs", None)
        if attrs is None:
            return False
        for attr in attrs:
            if not _bit_equal(getattr(a, attr), getattr(b, attr), f"{label}.{attr}"):
                return False
        for attr, va in vars(a).items():
            if attr in attrs:
                continue
            if type(va) is not type(getattr(b, attr)) or va != getattr(b, attr):
                return False
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_bit_equal(a[k], b[k], f"{label}.{k}") for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _bit_equal(x, y, f"{label}[{i}]") for i, (x, y) in enumerate(zip(a, b))
        )
    return type(a) is type(b) and a == b


def main() -> int:
    # 1) trunk model shards bit-identical
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        a = (SOURCE / "model" / shard).read_bytes()
        b = (MIGRATED / "model" / shard).read_bytes()
        check(f"trunk shard {shard} identical", a == b, f"{len(a)} bytes")

    # 2) RNG state bit-identical
    for name in ("rank-0.safetensors", "optimizer_sr.safetensors"):
        a = (SOURCE / "train_state" / "rng" / name).read_bytes()
        b = (MIGRATED / "train_state" / "rng" / name).read_bytes()
        check(f"rng {name} identical", a == b, f"{len(a)} bytes")

    # 3) trainer_state identical, zero updates since 113400
    ts_s = json.loads((SOURCE / "train_state" / "trainer_state.json").read_text())
    ts_m = json.loads((MIGRATED / "train_state" / "trainer_state.json").read_text())
    check("trainer_state identical", ts_s == ts_m)
    check(
        "successful_updates == 113400 (zero training updates since save)",
        ts_m.get("successful_updates") == 113400
        and ts_m.get("attempted_updates") == 113400,
    )

    # 4) optimizer state audit
    # The production hybrid state embeds torchao 8-bit optimizer classes;
    # allowlist exactly those (same as the production loader path) and keep
    # weights_only=True.
    import torchao.optim.subclass_8bit as _t8

    _safe = _t8.OptimState8bit
    with torch.serialization.safe_globals([_safe]):
        opt_s = torch.load(
            SOURCE / "train_state" / "optimizer.pt", map_location="cpu", weights_only=True
        )
        opt_m = torch.load(
            MIGRATED / "train_state" / "optimizer.pt", map_location="cpu", weights_only=True
        )

    routing_s, routing_m = opt_s["routing"], opt_m["routing"]

    def _names(routing, key):
        return {spec["name"] for spec in routing[key]}

    cmuon_s, cmuon_m = _names(routing_s, "cmuon"), _names(routing_m, "cmuon")
    adamw_s, adamw_m = _names(routing_s, "adamw"), _names(routing_m, "adamw")
    check("CMuon FQN set unchanged", cmuon_s == cmuon_m, f"{len(cmuon_s)} params")
    check(
        "AdamW set == source + exactly the 2 projector FQNs",
        adamw_m - adamw_s == {PROJECTOR_WEIGHT_FQN, PROJECTOR_BIAS_FQN}
        and adamw_s - adamw_m == set(),
    )

    inner_s, inner_m = opt_s["optimizer"], opt_m["optimizer"]
    state_s, state_m = inner_s["state"], inner_m["state"]
    groups_s, groups_m = inner_s["param_groups"], inner_m["param_groups"]

    old_id_by_name = {
        n: pid
        for g in groups_s
        for n, pid in zip(g["param_names"], g["params"], strict=True)
    }
    new_id_by_name = {
        n: pid
        for g in groups_m
        for n, pid in zip(g["param_names"], g["params"], strict=True)
    }
    mismatched = []
    missing = []
    for name, old_id in old_id_by_name.items():
        new_id = new_id_by_name.get(name)
        if new_id is None:
            missing.append(name)
            continue
        if old_id not in state_s:
            continue  # parameter had no AdamW state in the source
        if new_id not in state_m or not _bit_equal(state_s[old_id], state_m[new_id], name):
            mismatched.append(name)
    check("every source AdamW state bit-identical after renumbering", not mismatched and not missing,
          f"state entries compared: {sum(1 for g in groups_s for n in g['param_names'] if old_id_by_name[n] in state_s)}")
    check(
        "projector FQNs have NO AdamW state entry (pristine lazy init)",
        new_id_by_name[PROJECTOR_WEIGHT_FQN] not in state_m
        and new_id_by_name[PROJECTOR_BIAS_FQN] not in state_m,
    )
    check(
        "migrated AdamW state set == source set (no extra entries)",
        {name for name in old_id_by_name if old_id_by_name[name] in state_s}
        == {name for name, nid in new_id_by_name.items() if nid in state_m},
    )

    check("CMuon optimizer state block bit-identical", _bit_equal(opt_s["cmuon"], opt_m["cmuon"], "cmuon"))
    check("sr_rng block bit-identical", _bit_equal(opt_s["sr_rng"], opt_m["sr_rng"], "sr_rng"))
    check("transition block identical", opt_s.get("transition") == opt_m.get("transition"))

    # 5) projector shard == deterministic reconstruction from migration_seed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sakuramoon.checkpoint.migrate_irepa_checkpoint import (
        _deterministic_projector,
        _projector_tensors,
    )

    irepa_state = json.loads((MIGRATED / "train_state" / "irepa_state.json").read_text())
    check(
        "irepa_state anchors",
        irepa_state.get("source_update") == 113400
        and irepa_state.get("start_successful_update") == 113401
        and irepa_state.get("schema_version") == 1,
        f"start_successful_update={irepa_state.get('start_successful_update')}",
    )
    seed = int(irepa_state["migration_seed"])
    from safetensors.torch import load_file

    saved_rng = torch.get_rng_state()
    try:
        rebuilt = _projector_tensors(_deterministic_projector(2560, seed))
    finally:
        torch.set_rng_state(saved_rng)
    shard = load_file(str(MIGRATED / "model" / "model-irepa-projector.safetensors"))
    check(
        "projector shard == deterministic init from migration_seed (bit-exact)",
        set(shard) == set(rebuilt)
        and all(_bit_equal(rebuilt[k], shard[k], k) for k in rebuilt),
        f"seed={seed}",
    )
    w = rebuilt[PROJECTOR_WEIGHT_FQN]
    b = rebuilt[PROJECTOR_BIAS_FQN]
    check(
        "projector numel == 2560*768*3*3 + 768",
        w.numel() == 2560 * 768 * 3 * 3 and b.numel() == 768
        and w.numel() + b.numel() == 17695488,
        f"total={w.numel() + b.numel()}",
    )

    print()
    fails = [r for r in _results if not r[1]]
    print(f"TOTAL: {len(_results)} checks, {len(fails)} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
