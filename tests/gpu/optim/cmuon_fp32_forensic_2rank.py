"""F2 2-rank HCU forensic test for the FP32-rescue minimal hard-fail capsules.

Companion to ``fp32_rescue_2rank.py`` (the frozen cleanup-regression
scenario matrix). THIS script covers the F2 telemetry contract (fast minimal local-first capsules):

  A. hard fail, owner rank0 (BF16 ceiling + FP32 ABOVE_CEILING):
       * both ranks get the SAME verdict: identical CMuonSafetyError
         message (all_gather comparison)
       * ZERO commit (parameter bytes + observations untouched, both ranks)
       * cross-rank delta fingerprint mechanism intact (spread == 0)
       * OWNER (rank0) publishes exactly one MINIMAL local
         capsule (emergency root): input tensor + minimal
         metadata with the ORIGINAL fp32 verdict values and
         fp32_failure_reason = "above_ceiling"; best-effort
         shared mirror + mirror.json ok; NO diagnostic replay/trace
       * NON-OWNER (rank1) publishes NOTHING (no fabricated capsule)
       * legacy forensic JSON redirected to the per-rank test dir, owner
         record carries the fp32 fields, non-owner record nulls them
  B. hard fail, NONFINITE category (BF16 NaN + FP32 NaN):
       * strict metadata.json is valid JSON with null fp32_delta_rms
         (nonfinite is unrepresentable; the *_finite flags carry the state)
       * legacy JSON (allow_nan default) carries Infinity for the same
  C. crash loop: a second fresh-optimizer hard fail into the SAME root:
       * older capsule NEVER overwritten (metadata sha256 unchanged)
       * new capsule gets the -r2 suffix (unique per (obs, rank, fqn, chunk))
  D. successful rescue under instrumentation:
       * spread == 0, param diff == 0, fp32_rescues advanced (owner only)
       * NO capsule (local or mirror), NO legacy FP32 failure fields
         E. hard fail, BELOW_FLOOR (the 112105 production class):
               * owner capsule reason=below_floor, fp32 finite,
                 delta < floor, BF16 above ceiling; mirror ok

Launched with torchrun (2 local ranks), NOT pytest:

    torchrun --nproc_per_node=2 tests/gpu/optim/cmuon_fp32_forensic_2rank.py \
        --out /sakuramoon-runtime/artifacts/g1/cmuon-hard-fail-f1-2rank.json

Exit code 0 = every assertion passed on BOTH ranks (clean exit per spec).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Self-contained: put THIS tree's src first so torchrun workers (any cwd,
# any interpreter) import the tree under test, not a system install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import sakuramoon.optim.fp32_rescue as fr
from sakuramoon.optim.cmuon import (
    cmuon_zeroth_power_bf16,
    cmuon_zeroth_power_fp32,
    resolve_ns_map,
    route_cmuon_parameters,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import build_fp32_rescue
from sakuramoon.optim.guarded_canonical import (
    GuardedCanonicalGuardConfig,
    stable_owner,
)

NS4 = resolve_ns_map(None, 4)
LR = 0.00015625


# ---------------------------------------------------------------------------
# Mock module (same canonical-FQN layout as fp32_rescue_2rank.py).
# ---------------------------------------------------------------------------


class _MockDiTBlock(torch.nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.q_proj = torch.nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.k_proj = torch.nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.v_proj = torch.nn.Linear(
            hidden, hidden // 4, bias=False, dtype=torch.bfloat16
        )
        self.attention.content_gate = torch.nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention.out_proj = torch.nn.Linear(
            hidden, hidden, bias=False, dtype=torch.bfloat16
        )
        self.mlp = torch.nn.Module()
        self.mlp.in_proj = torch.nn.Linear(
            hidden, 2 * inter, bias=False, dtype=torch.bfloat16
        )
        self.mlp.down_proj = torch.nn.Linear(
            inter, hidden, bias=False, dtype=torch.bfloat16
        )
        self.attention_norm = torch.nn.Parameter(
            torch.ones(hidden, dtype=torch.float32)
        )
        self.mlp_norm = torch.nn.Parameter(torch.ones(hidden, dtype=torch.float32))


class _GlobalConditioner(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.shared_block_projection = torch.nn.Linear(
            hidden // 4, 6 * hidden, bias=False, dtype=torch.bfloat16
        )
        self.final_projection = torch.nn.Linear(
            hidden // 4, 2 * hidden, bias=True, dtype=torch.bfloat16
        )
        self.final_projection.bias.data = self.final_projection.bias.data.to(
            torch.float32
        )


class _MockDiT(torch.nn.Module):
    def __init__(self, hidden: int, inter: int, n_blocks: int) -> None:
        super().__init__()
        self.input_projection = torch.nn.Linear(
            128, hidden, bias=False, dtype=torch.bfloat16
        )
        self.conditioner = _GlobalConditioner(hidden)
        self.blocks = torch.nn.ModuleDict(
            {f"slot_{i:02d}": _MockDiTBlock(hidden, inter) for i in range(n_blocks)}
        )
        self.modality_image = torch.nn.Parameter(
            torch.zeros(hidden, dtype=torch.float32)
        )


class _MockComposite(torch.nn.Module):
    def __init__(self, dit: _MockDiT) -> None:
        super().__init__()
        self.dit = dit


def _seed_grads(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator(device=model.dit.input_projection.weight.device).manual_seed(
        seed
    )
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(
                *p.shape, generator=g, dtype=torch.float32, device=p.device
            ).to(p.dtype)


def _bootstrap_refs(model) -> dict[str, float]:
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    refs: dict[str, float] = {}
    for spec in routing.cmuon_specs:
        g = torch.randn_like(spec.parameter)
        for ci in range(spec.chunk_count):
            chunk_size = spec.chunk_size()
            if spec.chunk_count == 1:
                sig = g.float().pow(2).mean().sqrt().item()
            else:
                start = ci * chunk_size
                end = start + chunk_size
                sl = [slice(None)] * g.ndim
                sl[spec.chunk_dim] = slice(start, end)
                sig = g[tuple(sl)].float().pow(2).mean().sqrt().item()
            refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)
    return refs


def _param_fingerprint_spread(
    model: torch.nn.Module, device: torch.device
) -> float:
    fp: list[torch.Tensor] = []
    for _, p in model.named_parameters():
        x = p.detach().float()
        fp.append(x.pow(2).mean().sqrt())
        fp.append(x.abs().max())
    flat = torch.stack(fp).to(device)
    lo = flat.clone()
    hi = flat.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return float((hi - lo).max().item())


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gather_all(obj: object) -> list:
    """all_gather_object of ONE object per rank (NOT a list of them)."""
    gathered: list = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, obj)
    return gathered


class _ForcedNS:
    """Call-order forced NS (same mechanism as fp32_rescue_2rank.py): each
    rank's NS calls arrive in the order of THAT rank's owned chunks
    (flat-order filter), so the forced chunk is identified without any
    shape/name heuristics inside the stub.

    The FP32 path is forced with a per-step KIND FLAG (like the existing
    regression's ``fail_fp32``): every F1 scenario here forces exactly ONE
    chunk in BF16, so exactly ONE FP32 call happens per hard-fail step and
    the flag is unambiguous. ``fp32_kind=None`` = run the real NS (the
    successful-rescue scenario)."""

    def __init__(self, owned: list[int]) -> None:
        self.owned = owned
        self.pattern: dict[int, str] = {}
        self.call = 0
        self.fp32_kind: str | None = None

    def _apply(self, out: torch.Tensor, kind: str | None) -> torch.Tensor:
        if kind is None:
            return out
        if kind == "nan":
            return torch.full_like(out, float("nan"))
        if kind == "inf":
            # Clean inf FILL (not out*inf, which yields 0*inf=nan and would
            # poison the rms mean to nan): the FP32 rms reads back as +inf,
            # which the legacy dump serializes as Infinity (allow_nan=True) —
            # the exact 112126-class shape of a BF16-overflowing magnitude.
            return torch.full_like(out, float("inf"))
        if kind == "huge":
            return out * 1e9
        if kind == "tiny":
            # FP32 delta ~1e-9 x a normal NS delta: far BELOW the rescue
            # floor (finite) — the 112105 production failure class.
            return out * 1e-9
        raise ValueError(kind)

    def bf16(self, grad, ns_steps, ns_coefficients, eps):
        i = self.call
        self.call += 1
        kind = self.pattern.get(self.owned[i])
        out = cmuon_zeroth_power_bf16(grad, ns_steps, ns_coefficients, eps)
        return self._apply(out, kind)

    def fp32(self, grad, ns_steps, ns_coefficients, eps):
        out = cmuon_zeroth_power_fp32(grad, ns_steps, ns_coefficients, eps)
        return self._apply(out, self.fp32_kind)


def _build(
    model,
    rank: int,
    world_size: int,
    artifact_root: str,
    legacy_dir: str,
    forced: _ForcedNS,
    emergency_root: str | None = None,
):
    fr.cmuon_zeroth_power_bf16 = forced.bf16
    fr.cmuon_zeroth_power_fp32 = forced.fp32
    return build_fp32_rescue(
        model,
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
        sr_seed=44,
        ns_steps_by_role=NS4,
        guard_cfg=GuardedCanonicalGuardConfig(
            guard_ratio=0.05,
            reference_decay=0.999,
            min_reference=1e-12,
            numerical_floor=1e-20,
            warmup_observations=0,
            invariant_check=True,
        ),
        guard_bootstrap_refs=_bootstrap_refs(model),
        rank=rank,
        world_size=world_size,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        hard_fail_artifact_root=artifact_root,
        legacy_forensic_dir=legacy_dir,
        emergency_capsule_root=emergency_root,
    )


def _run_hard_fail(
    model,
    opt,
    forced: _ForcedNS,
    pattern: dict[int, str],
    fp32_kind: str,
    seed: int,
    report: dict,
    scenario: str,
) -> None:
    """One hard-fail step: identical verdict on both ranks, zero commit,
    owner-only artifact, cross-rank message equality."""
    forced.pattern = pattern
    forced.fp32_kind = fp32_kind
    forced.call = 0
    before_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    obs_before = opt.observations
    try:
        _seed_grads(model, seed)
        opt.step()
    except CMuonSafetyError as e:
        msg = str(e)
    else:
        raise AssertionError(f"{scenario}: no CMuonSafetyError on rank {opt.rank}")
    # zero commit (this rank)
    for name, p in model.named_parameters():
        assert torch.equal(before_params[name], p.detach()), (
            f"{scenario}: {name} changed on hard fail (partial commit)"
        )
    assert opt.observations == obs_before, f"{scenario}: observations advanced"
    # identical verdict message on both ranks
    msgs = _gather_all(msg)
    assert len(set(msgs)) == 1, f"{scenario}: messages differ across ranks: {msgs}"
    # fingerprint mechanism intact: no spread message appended
    assert "spread" not in msg, f"{scenario}: fingerprint spread leaked: {msg}"
    assert opt.max_delta_rank_spread == 0.0, (
        f"{scenario}: delta rank spread != 0"
    )
    report[scenario] = {
        "rank": opt.rank,
        "message": msg[:300],
        "fp32_rescue_failures": opt.fp32_rescue_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "this regression requires exactly 2 ranks"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    out_dir = Path(args.out).parent / f".f1-2rank-rank{rank}"
    # Fresh per run (each rank owns its own dir, no cross-rank race): stale
    # events from a previous crashed run would collide with this run's event
    # names and break the "exactly one event" asserts.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"rank": rank, "world_size": world_size}
    results: dict[str, object] = {}

    try:
        torch.manual_seed(20260903)
        model = _MockComposite(_MockDiT(256, 512, 2)).to(device)
        routing = route_cmuon_parameters(
            model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
        )
        flat_keys: list[tuple[str, int]] = []
        for spec in routing.cmuon_specs:
            for ci in range(spec.chunk_count):
                flat_keys.append((spec.name, ci))
        owner_of_flat = [
            stable_owner(fqn, ci, world_size) for fqn, ci in flat_keys
        ]
        owned0 = [i for i, o in enumerate(owner_of_flat) if o == 0]
        owned1 = [i for i, o in enumerate(owner_of_flat) if o == 1]
        assert owned0 and owned1, "need chunks owned by both ranks"
        poison_pos = owned0[0]  # spec: owner rank0 hard failure
        poison_fqn, poison_chunk = flat_keys[poison_pos]

        # ---- A. hard fail, above-ceiling (owner rank0) --------------------
        forced = _ForcedNS(owned0 if rank == 0 else owned1)
        opt_a = _build(
            model,
            rank,
            world_size,
            str(out_dir / "hf-a"),
            str(out_dir / "legacy"),
            forced,
            emergency_root=str(out_dir / "hf-a-emergency"),
        )
        _run_hard_fail(
            model,
            opt_a,
            forced,
            {poison_pos: "huge"},
            "huge",
            seed=61,
            report=results,
            scenario="A_above_ceiling",
        )
        # owner (rank0) F2 capsule contract: the MINIMAL capsule lands in
        # the LOCAL emergency root; the shared root carries the best-effort
        # mirror.
        if rank == 0:
            local = out_dir / "hf-a-emergency"
            events = list(local.iterdir()) if local.exists() else []
            events = [e for e in events if e.is_dir() and not e.name.startswith(".")]
            assert len(events) == 1, f"rank0 must publish exactly one local capsule, got {events}"
            ev = events[0]
            meta_path = ev / "metadata.json"
            assert meta_path.is_file(), f"missing {meta_path}"
            meta = json.loads(meta_path.read_text())  # strict JSON (allow_nan=False)
            assert meta["schema"] == "sakuramoon.cmuon_minimal_hardfail_capsule.v1"
            assert meta["fp32_failure_reason"] == "above_ceiling"
            assert meta["original_fp32_finite"] is True
            assert meta["original_fp32_delta_rms"] is not None
            assert meta["fp32_delta_rms"] == meta["original_fp32_delta_rms"]
            assert meta["fp32_delta_rms"] > meta["ceiling"]
            assert meta["owner"] == 0 and meta["this_rank"] == 0
            assert meta["fqn"] == poison_fqn and meta["chunk"] == poison_chunk
            # F2: NO diagnostic replay/trace in the critical-path capsule.
            assert "diagnostic_replay_bf16" not in meta
            assert "diagnostic_replay_fp32" not in meta
            assert "forensic_trace_error" not in meta
            assert meta["bf16_delta_rms"] is not None
            assert meta["bf16_delta_rms"] > meta["ceiling"]
            assert isinstance(meta["pid"], int) and meta["process_steps"] >= 1
            input_files = [
                f for f in ("input.safetensors", "input.pt") if (ev / f).is_file()
            ]
            assert len(input_files) == 1, f"exactly one input tensor: {input_files}"
            input_sha_local = _sha_file(ev / input_files[0])
            assert meta["tensor_sha256"] == input_sha_local
            # Best-effort shared mirror + mirror.json status.
            mirror_json = json.loads((ev / "mirror.json").read_text())
            assert mirror_json["status"] == "ok", mirror_json
            mirror_ev = Path(mirror_json["shared_path"])
            assert mirror_ev.is_dir() and (mirror_ev / "metadata.json").is_file()
            assert _sha_file(mirror_ev / input_files[0]) == input_sha_local
            results["A_rank0_artifact"] = {
                "event": ev.name,
                "input_file": input_files[0],
                "input_sha256": input_sha_local,
                "reason": meta["fp32_failure_reason"],
                "mirror_status": mirror_json["status"],
            }
        else:
            local1 = out_dir / "hf-a-emergency"
            assert not local1.exists() or not any(
                e.is_dir() and not e.name.startswith(".") for e in local1.iterdir()
            ), "non-owner must not publish a local capsule (fabrication)"
            assert not (out_dir / "hf-a").exists() or not any(
                e.is_dir() and not e.name.startswith(".") for e in (out_dir / "hf-a").iterdir()
            ), "non-owner must not publish a mirror (fabrication)"
            results["A_rank1_artifact"] = {"published": False}
        # legacy JSON (redirected per rank): owner has fp32 fields,
        # non-owner nulls them.
        legacy_path = out_dir / "legacy" / f"guard-forensic-rank{rank}.json"
        assert legacy_path.is_file(), f"missing redirected legacy JSON {legacy_path}"
        recs = json.loads(legacy_path.read_text())["records"]
        rec = next(
            r for r in recs if str(r.get("fqn", "")) == poison_fqn
        )
        if rank == 0:
            assert rec["fp32_failure_reason"] == "above_ceiling"
            assert rec["fp32_delta_rms"] is not None
            assert rec["bf16_delta_rms"] == rec["delta_rms"]
        else:
            assert rec["fp32_delta_rms"] is None
            assert rec["fp32_finite"] is None
            assert rec["delta_rms"] is None
        # cross-rank: rank0 published, rank1 did not
        published = _gather_all(rank == 0)
        assert published[0] is True and published[1] is False

        # ---- B. hard fail, nonfinite category -----------------------------
        forced_b = _ForcedNS(owned0 if rank == 0 else owned1)
        opt_b = _build(
            model,
            rank,
            world_size,
            str(out_dir / "hf-b"),
            str(out_dir / "legacy"),
            forced_b,
            emergency_root=str(out_dir / "hf-b-emergency"),
        )
        _run_hard_fail(
            model,
            opt_b,
            forced_b,
            {poison_pos: "inf"},
            "inf",
            seed=62,
            report=results,
            scenario="B_nonfinite",
        )
        if rank == 0:
            evb = next(iter((out_dir / "hf-b-emergency").iterdir()))
            metab = json.loads((evb / "metadata.json").read_text())
            assert metab["fp32_failure_reason"] == "nonfinite"
            assert metab["original_fp32_finite"] is False
            # strict JSON: a nonfinite rms is recorded as null (the *_finite
            # flags carry the state)
            assert metab["fp32_delta_rms"] is None
            legacy_b = json.loads(
                (out_dir / "legacy" / f"guard-forensic-rank{rank}.json").read_text()
            )["records"]
            recb = next(
                r for r in legacy_b if str(r.get("fqn", "")) == poison_fqn
            )
            assert recb["fp32_finite"] is False
            # the legacy dump uses default json.dump (allow_nan=True):
            # Infinity round-trips as inf
            assert math.isinf(float(recb["fp32_delta_rms"]))

        # ---- C. crash loop: second hard fail into the SAME root ------------
        # Only the owner (rank0) publishes to hf-a, so only it hashes the
        # older event; the non-owner has no hf-a directory at all.
        meta_a_path = None
        sha_before = None
        if rank == 0:
            meta_a_path = next((out_dir / "hf-a-emergency").glob("*/metadata.json"))
            sha_before = _sha_file(meta_a_path)
        forced_c = _ForcedNS(owned0 if rank == 0 else owned1)
        opt_c = _build(
            model,
            rank,
            world_size,
            str(out_dir / "hf-a"),
            str(out_dir / "legacy"),
            forced_c,
            emergency_root=str(out_dir / "hf-a-emergency"),
        )
        _run_hard_fail(
            model,
            opt_c,
            forced_c,
            {poison_pos: "huge"},
            "huge",
            seed=63,
            report=results,
            scenario="C_crash_loop",
        )
        if rank == 0:
            assert meta_a_path is not None and sha_before is not None
            events_c = sorted(
                p.name
                for p in (out_dir / "hf-a-emergency").iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            assert len(events_c) == 2, f"crash loop must add a 2nd local capsule: {events_c}"
            assert any(name.endswith("-r2") for name in events_c), (
                f"second event must carry the -r2 suffix: {events_c}"
            )
            assert _sha_file(meta_a_path) == sha_before, (
                "older event metadata was modified by the crash loop"
            )
            results["C_rank0"] = {"events": events_c}

        # ---- D. successful rescue under instrumentation --------------------
        forced_d = _ForcedNS(owned0 if rank == 0 else owned1)
        opt_d = _build(
            model,
            rank,
            world_size,
            str(out_dir / "hf-d"),
            str(out_dir / "legacy"),
            forced_d,
            emergency_root=str(out_dir / "hf-d-emergency"),
        )
        forced_d.pattern = {poison_pos: "huge"}  # BF16 trips; FP32 runs REAL
        forced_d.call = 0
        _seed_grads(model, 71)
        rescues_before = opt_d.fp32_rescues
        opt_d.step()  # must NOT raise
        fp_spread = _param_fingerprint_spread(model, device)
        assert opt_d.max_delta_rank_spread == 0.0
        assert opt_d.max_param_rank_diff == 0.0
        assert fp_spread == 0.0, f"rescue: param fingerprint spread {fp_spread}"
        if rank == 0:
            assert opt_d.fp32_rescues == rescues_before + 1, (
                f"owner must record the rescue: {opt_d.fp32_rescues}"
            )
        else:
            assert opt_d.fp32_rescues == rescues_before, (
                "non-owner must not record a rescue for the owner's chunk"
            )
        for d_root in ("hf-d-emergency", "hf-d"):
            ddir = out_dir / d_root
            assert not ddir.exists() or not any(
                e.is_dir() and not e.name.startswith(".") for e in ddir.iterdir()
            ), f"successful rescue must not publish a capsule under {d_root}"
        results["D_rescue"] = {
            "rank": rank,
            "fp32_rescues": opt_d.fp32_rescues,
            "param_fingerprint_spread": fp_spread,
            "artifact_published": False,
        }

        # ---- E. hard fail, BELOW_FLOOR (the 112105 production class) ------
        # BF16 trips (huge); the REAL FP32 NS is scaled to 1e-9x so the
        # FP32 delta is finite but far below the rescue floor — exactly the
        # 112105 production failure (below_floor, finite, ~floor-graze class).
        forced_e = _ForcedNS(owned0 if rank == 0 else owned1)
        opt_e = _build(
            model,
            rank,
            world_size,
            str(out_dir / "hf-e"),
            str(out_dir / "legacy"),
            forced_e,
            emergency_root=str(out_dir / "hf-e-emergency"),
        )
        _run_hard_fail(
            model,
            opt_e,
            forced_e,
            {poison_pos: "huge"},
            "tiny",
            seed=77,
            report=results,
            scenario="E_below_floor",
        )
        if rank == 0:
            eve = next(iter((out_dir / "hf-e-emergency").iterdir()))
            metae = json.loads((eve / "metadata.json").read_text())
            assert metae["fp32_failure_reason"] == "below_floor"
            assert metae["original_fp32_finite"] is True
            assert metae["fp32_delta_rms"] is not None
            assert metae["fp32_delta_rms"] < metae["rescue_floor"]
            assert metae["bf16_delta_rms"] > metae["ceiling"]
            mirror_e = json.loads((eve / "mirror.json").read_text())
            assert mirror_e["status"] == "ok"
            results["E_rank0_capsule"] = {
                "event": eve.name,
                "reason": metae["fp32_failure_reason"],
                "fp32_delta_rms": metae["fp32_delta_rms"],
                "rescue_floor": metae["rescue_floor"],
                "mirror_status": mirror_e["status"],
            }
        else:
            results["E_rank1_capsule"] = {"published": False}

        report["poison"] = f"{poison_fqn}#chunk{poison_chunk}"
        report["results"] = results
        report["verdict"] = "PASS"
    finally:
        if rank == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2))
        dist.barrier()
        dist.destroy_process_group()

    if report.get("verdict") != "PASS":
        print(f"[F2-2rank] rank{rank} FAIL", file=sys.stderr)
        sys.exit(1)
    if rank == 0:
        print(f"[F2-2rank] PASS: {json.dumps(report['results'], indent=1)[:400]}")


if __name__ == "__main__":
    main()
