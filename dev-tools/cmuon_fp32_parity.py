#!/usr/bin/env python3
"""Base vs forensic BIT/semantic parity harness for the F1 telemetry patch.

Spec §14: for the fixed workload (no hard failures), the parent baseline
(tree @ f045f87) and the forensic-patched tree must produce
``torch.equal``-identical results for:

  * updated parameters (all, every step)
  * CMuon momentum buffers
  * AdamW8bit state (exp_avg / exp_avg_sq / step)
  * SR RNG state
  * guard references
  * rescue counters (bf16/fp32 attempts, rescues, failures, by-role)
  * staged deltas (verified indirectly via parameter equality: any staged-
    delta divergence would change the parameters)
  * rank broadcast / optimizer state_dict (single-rank here; the 2-rank
    contract is covered by ``cmuon_fp32_ddp2.py``)

How it works (two processes, one per tree; NOTHING is imported from both
trees in a single process, so there is no module-identity risk):

  1. RUN (once per tree):
       python dev-tools/cmuon_fp32_parity.py run \
           --tree <SRC_DIR> --out parity-<tree>.json --device cuda
     builds the fixed mini module, the production optimizer from THAT
     tree's source, runs the fixed step sequence (including a
     pathological-scale gradient step that exercises the BF16-fail -> FP32
     rescue path), and fingerprints the complete optimizer state after
     every step (SHA256 over the exact bytes of every relevant tensor).

  2. COMPARE:
       python dev-tools/cmuon_fp32_parity.py compare \
           --base parity-base.json --forensic parity-forensic.json
     field-by-field equality (torch.equal semantics = byte equality).
     Exit code 0 = PASS, 1 = FAIL (first mismatch printed).

The workload is deliberately hard-failure-FREE: step 9 scales the
content_gate gradient to the 112126-class pathologic RMS (~1.6e-7) so the
BF16 NS trips the safety band and the FP32 RESCUE path commits; if a tree
hard-fails, the run aborts with a distinct error (that itself is a parity
violation and is reported).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

POISON_SHAPE = (96, 96)  # content_gate: unique shape in the fixed module


def _find_src(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if (p / "sakuramoon").is_dir():
            return p
        raise SystemExit(f"--tree {explicit} has no sakuramoon/ package")
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "src"
    if (candidate / "sakuramoon").is_dir():
        return candidate
    raise SystemExit(f"cannot locate src/ next to {here.parent}")


# --------------------------------------------------------------------------
# Fixed module (mirrors the unit-test mini DiT: 14 NS inputs, 1 sensitive,
# 1 AdamW matrix; unique attention shapes; locked BF16/FP32 dtype policy).
# --------------------------------------------------------------------------


def _build_fixed_module(device):
    import torch
    from torch import nn

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(128, 128, bias=False)
            self.k_proj = nn.Linear(120, 120, bias=False)
            self.v_proj = nn.Linear(112, 112, bias=False)
            self.content_gate = nn.Linear(96, 96, bias=False)
            self.out_proj = nn.Linear(104, 104, bias=False)
            self.q_norm = nn.Parameter(torch.ones(128))

        def forward(self, x):  # pragma: no cover
            return x

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Linear(128, 256, bias=False)
            self.down_proj = nn.Linear(256, 128, bias=False)

        def forward(self, x):  # pragma: no cover
            return x

    class _MiniBlock(nn.Module):
        # The CMuon allowlist anchors on the ``.attention.`` / ``.mlp.``
        # FQN segments; a flat block would route every projection to the
        # AdamW fallback and trip the 8bit quantizability check.
        def __init__(self) -> None:
            super().__init__()
            self.attention = _Attention()
            self.mlp = _MLP()

        def forward(self, x):  # pragma: no cover
            return x

    class _MiniDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleDict({"slot_00": _MiniBlock()})
            self.conditioner = nn.ModuleDict(
                {"shared_block_projection": nn.Linear(144, 768, bias=False)}
            )
            self.final_layer = nn.ModuleDict(
                {"linear": nn.Linear(128, 128, bias=False)}
            )

        def forward(self, x):  # pragma: no cover
            return x

    class _Composite(nn.Module):
        def __init__(self, dit) -> None:
            super().__init__()
            self.dit = dit

        def forward(self, x):  # pragma: no cover
            return self.dit(x)

    module = _Composite(_MiniDiT()).to(device)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.02).mul_(0.1)
    for p in module.parameters():
        p.data = p.data.to(torch.bfloat16 if p.ndim == 2 else torch.float32)
    return module


def _bootstrap_refs() -> dict[str, float]:
    base = 1e-4
    keys = [
        "dit.blocks.slot_00.attention.q_proj.weight#chunk0",
        "dit.blocks.slot_00.attention.k_proj.weight#chunk0",
        "dit.blocks.slot_00.attention.v_proj.weight#chunk0",
        "dit.blocks.slot_00.attention.content_gate.weight#chunk0",
        "dit.blocks.slot_00.attention.out_proj.weight#chunk0",
        "dit.blocks.slot_00.mlp.in_proj.weight#chunk0",
        "dit.blocks.slot_00.mlp.in_proj.weight#chunk1",
        "dit.blocks.slot_00.mlp.down_proj.weight#chunk0",
        *(
            f"dit.conditioner.shared_block_projection.weight#chunk{i}"
            for i in range(6)
        ),
    ]
    return {k: base for k in keys}


def _set_gradients(module, seed: int, pathologic_scale: float | None) -> None:
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        scale = 0.1
        if pathologic_scale is not None and tuple(p.shape) == POISON_SHAPE:
            scale = pathologic_scale
        # Gradients are recorded in the PARAMETER dtype (the production
        # autograd graph emits grads matching the leaf dtype).
        p.grad = (
            torch.randn(p.shape, generator=g) * scale
        ).to(device=p.device, dtype=p.dtype)


def _tensor_sha(t) -> str:
    """Deterministic content hash for ANY tensor (incl. quantized
    torchao subclasses): full value listing via tolist() (small module).

    torchao 8bit optimizer states (OptimState8bit) do not support
    .tolist() on the subclass. Fall back to the official
    ``__tensor_flatten__`` storage: codes (uint8) + scale (fp32) + the
    flattened attributes (signed) fully determine the tensor. qmap is a
    process-global constant LUT (identical on every run) and is hashed by
    shape only.
    """
    h = hashlib.sha256()
    h.update(str(type(t).__name__).encode())
    try:
        values = t.detach().cpu().flatten().tolist()
        h.update(repr(values).encode())
        return h.hexdigest()
    except RuntimeError:
        attrs, extra = t.__tensor_flatten__()
        for name in attrs:
            v = getattr(t, name)
            if name == "qmap":
                h.update(
                    f"{name}=shape{tuple(v.shape)},dtype{v.dtype}\n".encode()
                )
            else:
                h.update(
                    f"{name}={v.detach().cpu().flatten().tolist()!r}\n".encode()
                )
        h.update(f"extra={extra!r}\n".encode())
        return h.hexdigest()


def _walk(obj, path: str, acc: list[tuple[str, str]]) -> None:
    import torch

    if torch.is_tensor(obj):
        acc.append((path, _tensor_sha(obj)))
    elif isinstance(obj, (int, float, bool, str)):
        acc.append((path, f"S:{obj!r}"))
    elif isinstance(obj, dict):
        for k in sorted(obj, key=lambda x: str(x)):
            _walk(obj[k], f"{path}/{k}", acc)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", acc)


def _state_sha(state: dict) -> str:
    acc: list[tuple[str, str]] = []
    _walk(state, "", acc)
    h = hashlib.sha256()
    for path, fp in sorted(acc):
        h.update(f"{path}={fp}\n".encode())
    return h.hexdigest()


def _fingerprint(module, opt, step_no: int) -> dict[str, object]:
    name_by_id = {id(p): n for n, p in module.named_parameters()}
    # CMuon momentum buffers, in stable parameter-name order.
    momentum_fp = _state_sha(
        {name_by_id[id(p)]: buf for p, buf in opt._momenta.items()}
    )
    # Full AdamW8bit state (raw torch optimizer state incl. 8bit moments +
    # SR RNG), fingerprinted by deterministic key path.
    adamw_fp = _state_sha(opt.optimizer.state_dict())
    refs = [
        f"{fqn}#{ci}={ref!r}" for (fqn, ci), ref in sorted(opt._refs.items())
    ]
    return {
        "step": step_no,
        "params_sha256": _state_sha(
            {n: p for n, p in sorted(module.named_parameters())}
        ),
        "momentum_sha256": momentum_fp,
        "adamw_state_sha256": adamw_fp,
        "guard_refs": refs,
        "counters": {
            "observations": opt.observations,
            "bf16_attempts": opt.bf16_attempts,
            "bf16_safety_failures": opt.bf16_safety_failures,
            "fp32_attempts": opt.fp32_attempts,
            "fp32_rescues": opt.fp32_rescues,
            "fp32_rescue_failures": opt.fp32_rescue_failures,
            "rescue_by_role": dict(sorted(opt.rescue_by_role.items())),
            "max_delta_rank_spread": opt.max_delta_rank_spread,
            "max_param_rank_diff": opt.max_param_rank_diff,
        },
    }


def run(src: Path, out: Path, device: str, steps: int) -> int:
    import torch

    sys.path.insert(0, str(src))
    from sakuramoon.optim.fp32_rescue import build_fp32_rescue
    from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig

    dev = torch.device(device)
    torch.manual_seed(1234)
    module = _build_fixed_module(dev)

    guard_cfg = GuardedCanonicalGuardConfig(
        guard_ratio=0.1,
        reference_decay=0.999,
        min_reference=3.096e-08,
        numerical_floor=6.575e-07,
        warmup_observations=50,
        invariant_check=False,
    )
    opt = build_fp32_rescue(  # type: ignore[call-overload]
        module,
        lr=1.5625e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,  # locked policy: the AdamW8bit build rejects anything else
        bf16_stochastic_round=True,
        matrix_weight_decay=0.1,
        sensitive_weight_decay=0.0,
        sr_seed=1234,
        ns_steps_by_role={
            "attention_q": 4,
            "attention_k": 4,
            "attention_v": 4,
            "attention_content_gate": 4,
            "attention_out": 4,
            "ffn_in": 4,
            "ffn_down": 4,
            "adaln_shared": 4,
        },
        guard_cfg=guard_cfg,
        guard_bootstrap_refs=_bootstrap_refs(),
        rank=0,
        world_size=1,
    )

    fingerprints: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        # Step 9: the 112126-class pathologic signal (RMS ~1.6e-7 on the
        # content_gate chunk) -> BF16 NS trips the band -> FP32 rescue path.
        pathologic = 1.6e-7 if step == 9 else None
        _set_gradients(module, seed=100 + step, pathologic_scale=pathologic)
        try:
            opt.step()
        except Exception as exc:  # noqa: BLE001 - a hard fail here is a parity violation
            print(
                f"[parity] HARD FAILURE at step {step} on tree {src}: {exc!r}",
                file=sys.stderr,
            )
            return 2
        fingerprints.append(_fingerprint(module, opt, step))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "tree": str(src.resolve()),
                "device": device,
                "steps": steps,
                "fingerprints": fingerprints,
            },
            indent=1,
        )
    )
    print(f"[parity] wrote {out} ({steps} steps, no hard failures)")
    return 0


def compare(base_path: Path, forensic_path: Path) -> int:
    base = json.loads(base_path.read_text())
    forensic = json.loads(forensic_path.read_text())
    ok = True
    if base["steps"] != forensic["steps"]:
        print(f"FAIL: step count differs ({base['steps']} vs {forensic['steps']})")
        return 1
    for fb, ff in zip(base["fingerprints"], forensic["fingerprints"], strict=True):
        if fb["step"] != ff["step"]:
            print(f"FAIL: step index mismatch {fb} vs {ff}")
            return 1
        # The SR (stochastic-round) RNG is part of the AdamW state_dict, so
        # adamw_state_sha256 already covers it; no separate sr_rng field.
        for key in ("params_sha256", "momentum_sha256", "adamw_state_sha256",
                    "guard_refs"):
            if fb[key] != ff[key]:
                ok = False
                print(f"FAIL: step {fb['step']} {key} differs")
                if key in ("params_sha256", "momentum_sha256",
                           "adamw_state_sha256"):
                    print(f"    base:     {fb[key]}")
                    print(f"    forensic: {ff[key]}")
                else:
                    for rb, rf in zip(fb[key], ff[key], strict=True):
                        if rb != rf:
                            print(f"    base:     {rb}")
                            print(f"    forensic: {rf}")
                            break
        if fb["counters"] != ff["counters"]:
            ok = False
            print(f"FAIL: step {fb['step']} counters differ: "
                  f"{fb['counters']} vs {ff['counters']}")
    if ok:
        print(f"PASS: {base['steps']} steps, all fields byte-identical "
              f"(base={base_path.name}, forensic={forensic_path.name})")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run_p = sub.add_parser("run", help="run the fixed workload on one tree")
    run_p.add_argument("--tree", default=None, help="path to the tree's src/")
    run_p.add_argument("--out", required=True, type=Path)
    run_p.add_argument("--device", default="cuda")
    run_p.add_argument("--steps", type=int, default=10)
    cmp_p = sub.add_parser("compare", help="compare two parity fingerprints")
    cmp_p.add_argument("--base", required=True, type=Path)
    cmp_p.add_argument("--forensic", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        return run(_find_src(args.tree), args.out, args.device, args.steps)
    return compare(args.base, args.forensic)


if __name__ == "__main__":
    raise SystemExit(main())
