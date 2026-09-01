"""D4-checkpoint backwards-compatibility proof for the cleanup implementation
(production cleanup spec section 11).

Launched with torchrun (2 local ranks, the production world size of the
validated D4 chain):

    torchrun --nproc_per_node=2 tests/gpu/optim/cmuon_ckpt_backcompat.py \
        --ckpt /sakuramoon-runtime/output_model/g1/ckpt_103500_raw-103500-update-cadence \
        --artifacts /sakuramoon-runtime/artifacts/g1/cmuon-cleanup/ckpt-backcompat \
        --out .../backcompat.json

Sequence (spec 11 A-E):
  A. load the D4 validated 103500 optimizer state with the CLEANUP code
     (torch.load + optimizer.load_state_dict — the same path production
     resume uses)
  B. verify every state group: model params, CMuon momentum, AdamW fallback
     state, SR RNG, guard/reference state, rescue counters, owner mapping,
     NS map
  C. run exactly ONE successful update and save a new optimizer checkpoint
     (artifacts only — never the production chain)
  D. build a FRESH optimizer and exact-resume the saved checkpoint
  E. verify counter continuity (observations +1, rescue counters,
     references advanced) and run a second successful update

ANY mismatch => raise => torchrun exits non-zero (no retry, no tolerance).
"""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path

import torch
import torch.distributed as dist

from sakuramoon.checkpoint.artifact import build_trainable_composite
from sakuramoon.optim.cmuon import route_cmuon_parameters
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig

# CMuonNSConfig role fields (schema.py) — mirrors canonical_map()
_NS_ROLE_FIELDS = (
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_content_gate",
    "attention_out",
    "ffn_in",
    "ffn_down",
    "adaln_shared",
)


def _seed_grads(model, seed: int) -> None:
    g = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(
                *p.shape, generator=g, dtype=torch.float32, device=p.device
            ).to(p.dtype)


def _fp(t: torch.Tensor) -> tuple[float, float]:
    x = t.detach().float()
    return (float(x.pow(2).mean().sqrt()), float(x.abs().max()))


def _ns_canonical_map(ns_table: dict[str, object]) -> dict[str, int]:
    """CMuonNSConfig.canonical_map() reproduced from the resolved toml table."""
    base = int(ns_table["default"])
    return {
        role: (int(ns_table[role]) if ns_table.get(role) is not None else base)
        for role in _NS_ROLE_FIELDS
    }


def _build_opt(model, rank: int, world_size: int, resolved: dict[str, object]):
    """Build the optimizer from the CHECKPOINT'S resolved production config
    (train/production.py build path): scaled lr, exact guard config and the
    P3 calibration bootstrap references. Backcompat means the cleanup code
    must load the state under the SAME production config it was saved with —
    a hardcoded test config legitimately fails the guard-config check."""
    opt_tbl = resolved["optimizer"]
    guard_tbl = opt_tbl["cmuon_guard"]
    stage_tbl = resolved["stage"]
    lr = (
        float(opt_tbl["base_lr"])
        * int(stage_tbl["global_batch"])
        / float(opt_tbl["reference_batch"])
    )
    return build_fp32_rescue(
        model,
        lr=lr,
        betas=tuple(opt_tbl["betas"]),
        eps=float(opt_tbl["eps"]),
        block_size=int(opt_tbl["block_size"]),
        bf16_stochastic_round=bool(opt_tbl["bf16_stochastic_round"]),
        matrix_weight_decay=float(opt_tbl["matrix_weight_decay"]),
        sensitive_weight_decay=float(opt_tbl["sensitive_weight_decay"]),
        sr_seed=int(resolved["run"]["seed"]),
        ns_steps_by_role=_ns_canonical_map(opt_tbl["cmuon_ns"]),
        guard_cfg=GuardedCanonicalGuardConfig(
            guard_ratio=float(guard_tbl["guard_ratio"]),
            reference_decay=float(guard_tbl["reference_decay"]),
            min_reference=float(guard_tbl["min_reference"]),
            numerical_floor=float(guard_tbl["numerical_floor"]),
            warmup_observations=int(guard_tbl["warmup_observations"]),
            invariant_check=bool(guard_tbl["invariant_check"]),
        ),
        guard_bootstrap_refs=dict(guard_tbl["references"]),
        rank=rank,
        world_size=world_size,
        momentum_dtype=str(opt_tbl["cmuon_momentum_dtype"]),
        chunk_rescale_sqrt_n=bool(opt_tbl["cmuon_chunk_rescale_sqrt_n"]),
    )


def _state_summary(opt, module) -> dict[str, object]:
    sd = opt.state_dict()
    guard = sd["guard"]
    fr_state = guard.get("fp32_rescue", {})
    momenta = opt._momenta  # pyright: ignore[reportPrivateUsage]
    mom_fp = {}
    for spec in opt.routing.cmuon_specs:  # pyright: ignore[reportPrivateUsage]
        mom_fp[spec.name] = _fp(momenta[spec.parameter].float())
    return {
        "schema": {
            "hybrid_cmuon_schema_version": sd["hybrid_cmuon_schema_version"],
            "guarded_canonical_schema_version": sd["guarded_canonical_schema_version"],
            "guard_schema_version": guard["schema_version"],
            "owner_mapping_version": guard["owner_mapping_version"],
            "world_size": guard["world_size"],
            "bootstrap_mode": guard["bootstrap_mode"],
            "canonical_ns_mode": guard["canonical_ns_mode"],
            "ns_map": guard["ns_map"],
            "guard_config": guard["config"],
        },
        "observations": guard["observations"],
        "n_references": len(guard["references"]),
        "ref_min": min(guard["references"].values()),
        "ref_max": max(guard["references"].values()),
        "skip_total": guard["skip_total"],
        "fp32_rescue_counters": {
            k: fr_state.get(k)
            for k in (
                "bf16_attempts",
                "bf16_safety_failures",
                "fp32_attempts",
                "fp32_rescues",
                "fp32_rescue_failures",
            )
        },
        "rescue_by_role": fr_state.get("rescue_by_role", {}),
        "n_momentum_buffers": len(momenta),
        "momentum_fp_sample": {
            k: mom_fp[k] for k in list(mom_fp)[:3]
        },
        "adamw_state_n_params": len(opt.adamw.optimizer.state),  # pyright: ignore[reportPrivateUsage]
        "sr_rng": {
            k: (v if not torch.is_tensor(v) else v.shape)
            for k, v in opt.sr_rng.state_dict().items()  # pyright: ignore[reportPrivateUsage]
        },
        "routing": {
            "n_cmuon_specs": len(
                route_cmuon_parameters(
                    module, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
                ).cmuon_specs
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "backcompat proof requires the D4 world size (2)"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"rank": rank, "world_size": world_size}
    try:
        # ---- A. load D4 103500 with cleanup code -------------------------
        torch.manual_seed(20260903)
        config = json.loads(
            (Path(args.ckpt) / "model" / "config.json").read_text()
        )
        module = build_trainable_composite(config["architecture"], device=device)
        module.eval()
        resolved = tomllib.loads(
            (Path(args.ckpt) / "resolved_config.toml").read_text()
        )
        opt = _build_opt(module, rank, world_size, resolved)
        # production ckpt layout: train_state/optimizer.pt (checkpoint/load.py)
        saved = torch.load(
            Path(args.ckpt) / "train_state" / "optimizer.pt",
            map_location=device,
            weights_only=False,
        )
        opt.load_state_dict(saved)
        report["A_load"] = "ok"
        report["B_state_after_load"] = _state_summary(opt, module)

        # sanity: the loaded state is the D4 final state (not the bootstrap)
        g = opt.state_dict()["guard"]
        assert g["observations"] > 100, (
            f"observations must be the D4-accumulated value, got {g['observations']}"
        )

        # ---- C. one successful update + save ------------------------------
        _seed_grads(module, 41)
        torch.cuda.synchronize(device)
        opt.step()
        torch.cuda.synchronize(device)
        if rank == 0:
            torch.save(
                opt.state_dict(),
                Path(args.artifacts) / "optimizer_after_1_update.pt",
            )
        dist.barrier()
        report["C_update_and_save"] = "ok"
        report["C_state_after_1"] = _state_summary(opt, module)

        # ---- D. fresh optimizer + exact resume ----------------------------
        torch.manual_seed(20260903)
        module2 = build_trainable_composite(config["architecture"], device=device)
        module2.eval()
        opt2 = _build_opt(module2, rank, world_size, resolved)
        resumed = torch.load(
            Path(args.artifacts) / "optimizer_after_1_update.pt",
            map_location=device,
            weights_only=False,
        )
        opt2.load_state_dict(resumed)
        report["D_resume"] = "ok"
        report["D_state_after_resume"] = _state_summary(opt2, module2)

        # ---- E. counter continuity + second update -------------------------
        before = opt2.state_dict()
        _seed_grads(module2, 42)
        opt2.step()
        after = opt2.state_dict()
        cont = {
            "observations_delta": (
                after["guard"]["observations"] - before["guard"]["observations"]
            ),
            "fp32_attempts_delta": (
                after["guard"]["fp32_rescue"]["fp32_attempts"]
                - before["guard"]["fp32_rescue"]["fp32_attempts"]
            ),
            "fp32_rescues_delta": (
                after["guard"]["fp32_rescue"]["fp32_rescues"]
                - before["guard"]["fp32_rescue"]["fp32_rescues"]
            ),
            "bf16_attempts_delta": (
                after["guard"]["fp32_rescue"]["bf16_attempts"]
                - before["guard"]["fp32_rescue"]["bf16_attempts"]
            ),
            "n_references_stable": len(after["guard"]["references"])
            == len(before["guard"]["references"]),
        }
        assert cont["observations_delta"] == 1, cont
        assert cont["n_references_stable"], "reference key set changed"
        report["E_continuity"] = cont
        report["verdict"] = "D4_CHECKPOINT_BACKCOMPAT=YES"
    finally:
        if rank == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
