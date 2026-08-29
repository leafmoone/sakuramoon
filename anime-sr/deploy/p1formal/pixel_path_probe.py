#!/usr/bin/env python3
"""Pixel Path Activation Probe (Phase I-P-1024, p1formal).

Read-only probe over a training checkpoint. Verifies that the pixel-feature
path (zero-init transition weights + PixelConditionEncoder body) has started
learning, using two independent evidences:
  (a) weight evidence: zero-init params must have moved away from exactly 0
      (an AdamW update with an all-zero gradient leaves the weight at 0, so
      w != 0 implies a non-zero gradient was consumed),
  (b) optimizer-state evidence: exp_avg of the relevant params is non-zero.
No training code is modified; run after the first checkpoint (step 1000).

Usage: PYTHONPATH=/root/anime-sr-p1formal/src python3 pixel_path_probe.py <ckpt.pt>
"""
import sys
import torch

TARGET_SUFFIXES = ("proj_p64.weight", "proj_p32.weight", "proj_p16.weight", "gap_proj.weight")


def nz(v: torch.Tensor) -> bool:
    return bool(v.abs().sum().item() > 0)


def main() -> None:
    ckpt_path = sys.argv[1]
    d = torch.load(ckpt_path, map_location="cpu")
    sd = d["model"] if isinstance(d, dict) and "model" in d else d
    step = d.get("step") if isinstance(d, dict) else "?"
    print(f"ckpt={ckpt_path} step={step} n_keys={len(sd)}")

    # (a) weight evidence for the 4 zero-init transition weights
    print("=== (a) zero-init transition weights (w != 0  <=>  non-zero grad consumed) ===")
    hits = 0
    for suf in TARGET_SUFFIXES:
        for k in sd:
            if k.endswith(suf):
                v = sd[k]
                ok = nz(v)
                hits += int(ok)
                print(f"  {'OK ' if ok else 'DEAD'} {k} shape={tuple(v.shape)} l1={v.abs().sum().item():.6e} nonzero={ok}")
    print(f"  transition-weights alive: {hits}/4")

    # (b) PixelConditionEncoder body: count non-zero params among pixel keys
    pix = [(k, v) for k, v in sd.items() if "pixel" in k]
    pix_alive = [k for k, v in pix if nz(v)]
    print(f"=== (b) pixel_encoder body: {len(pix_alive)}/{len(pix)} params non-zero ===")
    for k in pix_alive[:8]:
        print(f"  alive {k} l1={sd[k].abs().sum().item():.6e}")

    # (c) optimizer-state corroboration (exp_avg non-zero where grads flowed)
    opt = d.get("optimizer") if isinstance(d, dict) else None
    print("=== (c) optimizer state (exp_avg) ===")
    if isinstance(opt, dict) and "state" in opt:
        state = opt["state"]
        total = len(state)
        alive = sum(1 for s in state.values() if nz(s.get("exp_avg", torch.zeros(1))))
        print(f"  optimizer params: {alive}/{total} with non-zero exp_avg")
        # index i in optimizer state == i-th key of model.state_dict() (optimizer
        # built on model.parameters(); named_parameters order matches state_dict)
        idx = {k: i for i, k in enumerate(sd)}
        for suf in TARGET_SUFFIXES:
            for k in sd:
                if k.endswith(suf):
                    s = state.get(idx[k], {})
                    ea = s.get("exp_avg")
                    print(f"  opt[{idx[k]}] {k} exp_avg_nonzero={bool(ea is not None and nz(ea))}")
        pix_idx = [i for k, i in idx.items() if "pixel" in k]
        pix_opt_alive = sum(1 for i in pix_idx if nz(state.get(i, {}).get("exp_avg", torch.zeros(1))))
        print(f"  optimizer pixel-range alive: {pix_opt_alive}/{len(pix_idx)}")
    else:
        print("  (no optimizer state in ckpt; weight evidence (a)/(b) stands alone)")

    verdict = hits == 4 and len(pix_alive) > 0
    print(f"PIXEL-PATH VERDICT: {'ALIVE (learning)' if verdict else 'DEAD / INSUFFICIENT EVIDENCE -> STOP'}")
    sys.exit(0 if verdict else 2)


if __name__ == "__main__":
    main()
