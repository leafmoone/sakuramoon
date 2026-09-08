"""Three-way audit stage-1 RECONSTRUCTION (verbatim copy of frozen cc_stage1.py).

Provenance:
  frozen source : dev-tools/camera_coordinate_causal/final_snapshot/cc_stage1.py
  frozen sha256 : 384d58e49b3c08b345525aded9485c1bb38c310d426d399aa07acf6182aa65b0
  (branch camera-v2-mbs-corrected-100u-rerun2-evidence @ 1b8fce4; identical at 34f646ab)

Exactly three deviations from the frozen source, all documented inline:
  1. sys.path insert targets the frozen final_snapshot/ directory (this copy
     lives in mbs_three_way/).
  2. The prior-audit checkpoint-evidence loop is reduced to PRE (U116100,
     recorded verbatim) plus an explicit UNAVAILABLE note for MID (U117100,
     not present on come7).  Provenance-only; zero effect on pair inputs.
  3. The manifest script_sha256 block hashes the frozen scripts from their
     final_snapshot/ location (semantics unchanged: same frozen files).
Every other line - pipeline wiring, quota, freeze, SHUFFLED derangement,
RANDOM arms, arm validation, manifest schema - is byte-identical to the
frozen tool.  Output: /tmp/camera-coordinate-causal/{units/,stage1-manifest.json}
(same OUT root the frozen build_pairs/encode_pairs read from).

Stage 1 - cohort construction + frozen input build (forward-only, read-only).

Drives the production WebDatasetPipeline over the fixed s0-validation-50k-v1
shards (real ordinary admission + Camera v2 planner, p=0.25), selects the
CAMERA_APPLIED and ORDINARY cohorts, and freezes every unit's inputs:
VAE x0, production JLT timestep strata, per-(unit,stratum) noise, x_t states,
Qwen text features, token routing, and all arm coordinate maps.

Artifacts: /tmp/camera-coordinate-causal/{units/unit-*.pt, stage1-manifest.json}
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final_snapshot"))  # deviation 1: frozen final_snapshot module (sha256 in README)

# Allocator config must be set before importing torch (production pattern).
_prev = os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", ""
)
_opts = [o.strip() for o in _prev.split(",") if o.strip() and not o.strip().startswith("expandable_segments:")]
if not any(o.startswith("max_split_size_mb:") for o in _opts):
    _opts.append("max_split_size_mb:512")
_opts.append("expandable_segments:True")
os.environ["PYTORCH_ALLOC_CONF"] = ",".join(_opts)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]

import torch  # noqa: E402

import cc_common as cc  # noqa: E402


def log(msg: str) -> None:
    print(f"[stage1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_state(repo: Path) -> dict:
    import subprocess

    out: dict = {}
    r = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    out["status_porcelain"] = [ln for ln in r.stdout.splitlines() if ln.strip()]
    r2 = subprocess.run(
        ["git", "-C", str(repo), "diff", "--stat",
         f"{cc.ENTRANCE_HEAD}..HEAD", "--", "src", "tests", "config"],
        capture_output=True, text=True, check=True,
    )
    out["tracked_diff_src_tests_config"] = r2.stdout.strip()
    r3 = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    out["head"] = r3.stdout.strip()
    return out


def build_token_routing(sample, padding_token_id: int) -> dict:
    """Batch-1 reproduction of data/collate.py collate_samples field mapping."""
    from sakuramoon.data.collate import _active_condition_sample_indices, _index_tensor

    cap = sample.caption
    dense_length = int(cap.dense_length)
    input_ids = torch.full((1, dense_length), padding_token_id, dtype=torch.long)
    attention_mask = torch.zeros((1, dense_length), dtype=torch.bool)
    length = len(cap.input_ids)
    if length > dense_length:
        raise ValueError("serialized caption exceeds its dense bucket")
    input_ids[0, :length] = torch.tensor(cap.input_ids, dtype=torch.long)
    attention_mask[0, :length] = True
    main_indices, main_mask = _index_tensor((tuple(cap.main_token_indices),))
    cond_indices, cond_mask = _index_tensor((tuple(cap.condition_token_indices),))
    active = _active_condition_sample_indices((sample,))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "main_token_indices": main_indices,
        "main_mask": main_mask,
        "main_token_lengths": (int(len(cap.main_token_indices)),),
        "condition_token_indices": cond_indices,
        "condition_mask": cond_mask,
        "use_null_condition": torch.tensor([bool(cap.use_null_condition)]),
        "active_condition_sample_indices": active,
        "dense_length": dense_length,
    }


def build_arms(
    sample, device: torch.device
) -> dict:
    from sakuramoon.conditioning.camera import (
        camera_transform_params,
        transform_camera_coordinates,
    )
    from sakuramoon.conditioning.rope import (
        full_canvas_crop_coordinates,
        image_coordinates,
    )

    th = sample.target_height // 16
    tw = sample.target_width // 16
    audit = sample.audit
    base = image_coordinates(th, tw, device=device)
    correct = full_canvas_crop_coordinates(
        th,
        tw,
        full_height=audit.resized_height,
        full_width=audit.resized_width,
        crop_box=audit.crop_box,
        device=device,
    )
    arms = {"CORRECT": correct, "IDENTITY": base, "SAME": correct.clone()}
    zoom = xs = ys = None
    if audit.camera_applied:
        left, top, _right, _bottom = audit.crop_box
        zoom, xs, ys = camera_transform_params(
            viewport=sample.target_width,
            full_width=audit.resized_width,
            full_height=audit.resized_height,
            left=left,
            top=top,
        )
        opposite = transform_camera_coordinates(
            base, zoom=zoom, x_shift=-xs, y_shift=-ys
        )
        half = base + 0.5 * (correct - base)
        over = base + 1.5 * (correct - base)
        arms.update(
            OPPOSITE=opposite, HALF=half, OVER=over,
        )
    return arms, zoom, xs, ys


def main() -> int:
    t_start = time.time()
    from sakuramoon.checkpoint.load import read_checkpoint_manifest
    from sakuramoon.conditioning.rope import (
        full_canvas_crop_coordinates,
        image_coordinates,
    )
    from sakuramoon.encoders.mage_vae import load_local_mage_vae
    from sakuramoon.encoders.qwen import load_local_qwen
    from sakuramoon.objective.flow import interpolate_state, sample_noise

    device = torch.device("cuda", 0)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("stage1 requires exactly one visible CUDA device")
    log("loading config / qwen / vae ...")
    config = cc.load_runtime_config().config
    qwen = load_local_qwen(cc.RUNTIME_ROOT, device)
    vae = load_local_mage_vae(cc.RUNTIME_ROOT, device)
    padding_token_id = qwen.tokenizer.pad_token_id
    if type(padding_token_id) is not int:
        raise RuntimeError("tokenizer padding identity unavailable")
    log(
        f"config: stage={config.stage.name} resolution={config.stage.resolution} "
        f"seed={config.run.seed} camera={config.data.camera_viewport}"
    )

    cc.OUT.mkdir(parents=True, exist_ok=True)
    units_dir = cc.OUT / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    # ---------- entrance gate evidence ----------
    log("recording entrance gate evidence ...")
    gate = git_state(cc.REPO)
    log(f"git head={gate['head']} untracked={len(gate['status_porcelain'])} "
        f"src/tests/config diff=[{gate['tracked_diff_src_tests_config'] or 'empty'}]")

    # --- DOCUMENTED DEVIATION (three-way audit reconstruction, come7) ---
    # Frozen cc_stage1 records model evidence for the OLD causal audit's three
    # checkpoints.  On come7 the MID checkpoint (U117100, which was a /tmp copy
    # on the original audit host) is not present, so the frozen evidence loop
    # would abort.  This block records the PRE (U116100) evidence verbatim and
    # substitutes an explicit UNAVAILABLE note for MID.  This field is
    # PROVENANCE RECORDING for the prior causal audit only; it does not affect
    # any pair input, selection, anchor bundle, or VBS/three-way scoring.
    ckpt_evidence = {}
    for name, path in (("PRE", cc.CKPTS["PRE"]),):
        model_dir = path / "model"
        manifest = read_checkpoint_manifest(path)
        tree = cc.tree_sha256(model_dir)
        ckpt_evidence[name] = {
            "path": str(path),
            "update": manifest.identity.update,
            "model_tree_sha256": tree,
            "manifest_sha256": cc.sha256_file(path / "manifest.json"),
        }
        log(f"ckpt {name}: update={manifest.identity.update} model_tree={tree[:16]}")
    ckpt_evidence["MID"] = {
        "path": str(cc.CKPTS["MID"]),
        "update": None,
        "model_tree_sha256": None,
        "manifest_sha256": None,
        "note": (
            "UNAVAILABLE_ON_COME7: U117100 ckpt was a /tmp copy on the original "
            "audit host; evidence is provenance-only for the prior causal audit "
            "and does not affect VBS pair inputs"
        ),
    }
    log("ckpt MID evidence: UNAVAILABLE_ON_COME7 (provenance-only note recorded)")

    # ---------- validation shard plan ----------
    shard_plan = cc.load_validation_shard_plan()
    log(f"validation shards: {len(shard_plan)}")

    # ---------- pipeline loop ----------
    tvals = cc.t_values()
    # --- hard sanity gate on the timestep strata (must match the production JLT) ---
    from sakuramoon.objective.flow import sample_jlt_timesteps

    # 1) Acklam reference values
    ref = {0.10: -1.281552, 0.35: -0.385320, 0.50: 0.0, 0.65: 0.385320, 0.90: 1.281552}
    for q, z in ref.items():
        got = cc.normal_cdf_inv(q)
        if abs(got - z) > 1e-5:
            raise RuntimeError(f"normal_cdf_inv({q}) = {got}, expected {z}")
    # 2) monotone, within (0,1)
    if not all(0.0 < t < 1.0 for t in tvals) or tvals != tuple(sorted(tvals)):
        raise RuntimeError(f"t strata not monotone in (0,1): {tvals}")
    # 3) empirical cross-check: 2M draws from the PRODUCTION sampler
    gen_chk = torch.Generator(device=device)
    gen_chk.manual_seed(987654321)
    draws = sample_jlt_timesteps(
        2_000_000, p_mean=cc.P_MEAN, p_std=cc.P_STD, device=device, generator=gen_chk
    ).cpu().numpy()
    import numpy as _np

    for q, t_analytic in zip(cc.T_QUANTILES, tvals):
        t_emp = float(_np.quantile(draws, q))
        if abs(t_emp - t_analytic) > 0.005:
            raise RuntimeError(
                f"JLT quantile mismatch at q={q}: analytic {t_analytic:.6f} vs empirical {t_emp:.6f}"
            )
    del draws
    log(f"t strata (JLT quantiles {cc.T_QUANTILES}): {[f'{t:.6f}' for t in tvals]} - verified vs production sampler")

    rejections: dict[str, int] = {}

    def rejection_observer(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    dist = {
        "admitted": 0, "camera_selected": 0, "camera_applied": 0,
        "camera_fallback": {}, "camera_orientation": {},
        "camera_zoom_hist": {}, "camera_latent_shift_hist": {},
    }
    camera_samples: list = []
    ordinary_samples: list = []
    ORD_CAP = 2048

    def camera_zoom_band(z: float) -> str:
        if z < 1.20:
            return "mild"
        if z < 1.35:
            return "medium"
        return "strong"

    def latent_shift_band(s: float) -> str:
        if s < 2.0:
            return "lt2"
        if s < 4.0:
            return "2to4"
        return "ge4"

    for shard_i, plan in enumerate(shard_plan):
        log(f"shard {shard_i+1}/{len(shard_plan)}: {plan['rel']}")
        pipeline = cc.build_validation_pipeline(
            plan["local"], plan["rel"], plan["bytes"],
            config, qwen, padding_token_id, rejection_observer,
        )
        shard_admitted = 0
        for sample in pipeline:
            audit = sample.audit
            dist["admitted"] += 1
            shard_admitted += 1
            if audit.camera_selected:
                dist["camera_selected"] += 1
            if audit.camera_applied:
                dist["camera_applied"] += 1
                dist["camera_orientation"][audit.camera_orientation] = (
                    dist["camera_orientation"].get(audit.camera_orientation, 0) + 1
                )
                dist["camera_zoom_hist"][camera_zoom_band(audit.camera_equivalent_zoom)] = (
                    dist["camera_zoom_hist"].get(camera_zoom_band(audit.camera_equivalent_zoom), 0) + 1
                )
                band = latent_shift_band(audit.camera_latent_center_shift)
                dist["camera_latent_shift_hist"][band] = (
                    dist["camera_latent_shift_hist"].get(band, 0) + 1
                )
                if len(camera_samples) < cc.N_CAMERA_TARGET:
                    camera_samples.append(sample)
            else:
                reason = audit.camera_fallback_reason if audit.camera_selected else "not_selected"
                dist["camera_fallback"][reason] = (
                    dist["camera_fallback"].get(reason, 0) + 1
                )
                if len(ordinary_samples) < ORD_CAP:
                    ordinary_samples.append(sample)
            if (
                len(camera_samples) >= cc.N_CAMERA_TARGET
                and len(ordinary_samples) >= ORD_CAP
                and shard_admitted % 1000 == 0
            ):
                log(f"  quotas met; continuing for full distribution (admitted={dist['admitted']})")
        log(f"  shard done: admitted={shard_admitted} "
            f"camera_pool={len(camera_samples)} ordinary_pool={len(ordinary_samples)}")
    log(
        f"pipeline done: admitted={dist['admitted']} selected={dist['camera_selected']} "
        f"applied={dist['camera_applied']} ({dist['camera_applied']/max(dist['admitted'],1):.4f})"
    )

    # ---------- freeze units ----------
    log("freezing units (VAE + Qwen + arms + states) ...")
    t_freeze = time.time()
    bundles: list[dict] = []

    def freeze_one(idx: int, sample, cohort: str, arms: dict, zoom, xs, ys) -> dict:
        routing = build_token_routing(sample, padding_token_id)
        image = (
            sample.image.to(device, non_blocking=True)
            .to(torch.bfloat16)
            .div(127.5)
            .sub(1.0)
            .unsqueeze(0)
        )
        with torch.no_grad():
            x0 = vae.encode(image)  # [1,128,h/16,w/16] bf16
            qwen_out = qwen.encoder(
                routing["input_ids"].to(device),
                routing["attention_mask"].to(device),
                dense_lengths=(routing["dense_length"],),
            )
            qwen_states = qwen_out.hidden_states
            if not isinstance(qwen_states, torch.Tensor):
                raise TypeError("Qwen encoder did not return a hidden_states tensor")
            eps_list = []
            states = []
            for k in range(cc.N_STRATA):
                gen = torch.Generator(device=device)
                gen.manual_seed(cc.eps_seed(idx, k))
                eps = sample_noise(x0, noise_scale=cc.NOISE_SCALE, generator=gen)
                state = interpolate_state(
                    x0, eps,
                    torch.tensor([tvals[k]], device=device, dtype=torch.float32),
                )
                eps_list.append(eps)
                states.append(state)
        size_scale, aspect = cc.size_scale_aspect(
            sample.target_height, sample.target_width
        )
        audit = sample.audit
        arms_cpu = {name: t.detach().cpu() for name, t in arms.items()}
        correct_cpu = arms_cpu["CORRECT"]
        identity_cpu = arms_cpu["IDENTITY"]
        strict_identity = bool((correct_cpu == identity_cpu).all().item())
        opp_na = bool(audit.camera_applied and xs == 0.0 and ys == 0.0)
        bundle = {
            "unit": idx,
            "cohort": cohort,
            "sample_id": int(sample.sample_id),
            "source_shard": str(sample.source_shard),
            "target_h": int(sample.target_height),
            "target_w": int(sample.target_width),
            "resized_h": int(audit.resized_height),
            "resized_w": int(audit.resized_width),
            "crop_box": list(audit.crop_box),
            "crop_retention": float(audit.crop_retention),
            "crop_policy": str(audit.crop_policy),
            "camera_applied": bool(audit.camera_applied),
            "camera_fallback_reason": str(audit.camera_fallback_reason),
            "camera_orientation": str(audit.camera_orientation),
            "camera_equivalent_zoom": float(audit.camera_equivalent_zoom),
            "camera_final_retention": float(audit.camera_final_retention),
            "camera_normalized_offset": float(audit.camera_normalized_offset),
            "camera_pixel_center_shift": float(audit.camera_pixel_center_shift),
            "camera_latent_center_shift": float(audit.camera_latent_center_shift),
            "camera_shift_x": float(audit.camera_shift_x),
            "camera_shift_y": float(audit.camera_shift_y),
            "zoom": None if zoom is None else float(zoom),
            "x_shift": None if xs is None else float(xs),
            "y_shift": None if ys is None else float(ys),
            "opp_na": opp_na,
            "strict_identity": strict_identity,
            "t": list(tvals),
            "size_scale": size_scale,
            "aspect": aspect,
            "x0": x0.detach().cpu(),
            "states": [s.detach().cpu() for s in states],
            "eps": [e.detach().cpu() for e in eps_list],
            "qwen_states": qwen_states.detach().cpu(),
            **routing,
            "arms": arms_cpu,
        }
        return bundle

    # camera cohort first (indices 0..C-1), then ordinary (C..)
    n_cam = len(camera_samples)
    strict_idx: list[int] = []
    log(f"freezing {n_cam} camera units ...")
    for i, sample in enumerate(camera_samples):
        arms, zoom, xs, ys = build_arms(sample, device)
        bundle = freeze_one(i, sample, "camera", arms, zoom, xs, ys)
        torch.save(bundle, units_dir / f"unit-{i:05d}.pt")
        if i % 128 == 0:
            log(f"  camera {i}/{n_cam} ({time.time()-t_freeze:.0f}s)")

    log(f"freezing {len(ordinary_samples)} ordinary units ...")
    for j, sample in enumerate(ordinary_samples):
        idx = n_cam + j
        arms, zoom, xs, ys = build_arms(sample, device)
        bundle = freeze_one(idx, sample, "ordinary", arms, zoom, xs, ys)
        torch.save(bundle, units_dir / f"unit-{idx:05d}.pt")
        if bundle["strict_identity"]:
            strict_idx.append(idx)
        if j % 128 == 0:
            log(f"  ordinary {j}/{len(ordinary_samples)} ({time.time()-t_freeze:.0f}s)")
    n_ord_pool = len(ordinary_samples)

    # ---------- final cohort selection ----------
    log(f"strict-identity ordinary units: {len(strict_idx)}")

    n_strict_take = min(len(strict_idx), max(cc.N_STRICT_TARGET, cc.N_ORDINARY_TARGET))
    strict_take = strict_idx[:n_strict_take]
    strict_set = set(strict_take)
    ordinary_final: list[int] = list(strict_take)
    for j in range(n_ord_pool):
        idx = n_cam + j
        if idx in strict_set:
            continue
        if len(ordinary_final) >= cc.N_ORDINARY_TARGET:
            break
        ordinary_final.append(idx)
    unit_order = list(range(n_cam)) + ordinary_final
    n_units = len(unit_order)
    log(f"final cohort: {n_cam} camera + {len(ordinary_final)} ordinary = {n_units}")

    # ---------- SHUFFLED arm: seeded derangement over camera cohort ----------
    perm = cc.derangement(n_cam, seed=cc.MASTER_SEED)
    log(f"shuffle derangement first 12: {perm[:12]}")
    for i in range(n_cam):
        p = units_dir / f"unit-{i:05d}.pt"
        q = units_dir / f"unit-{perm[i]:05d}.pt"
        b = torch.load(p, map_location="cpu", weights_only=False)
        s = torch.load(q, map_location="cpu", weights_only=False)
        b["arms"]["SHUFFLED"] = s["arms"]["CORRECT"]
        b["shuffle_from_unit"] = perm[i]
        b["shuffle_from_sample_id"] = s["sample_id"]
        torch.save(b, p)
    log("SHUFFLED arm written (no self-assignment by construction)")

    # ---------- RANDOM arm on first 64 ordinary 256x256 units ----------
    random_targets = [
        idx for idx in ordinary_final
        if (units_dir / f"unit-{idx:05d}.pt") is not None
    ][:0]
    # load lazily to keep memory low
    from sakuramoon.conditioning.camera import transform_camera_coordinates
    from sakuramoon.conditioning.rope import image_coordinates as id_coords

    n_random = 0
    for idx in ordinary_final:
        if n_random >= cc.N_RANDOM_GEOM:
            break
        p = units_dir / f"unit-{idx:05d}.pt"
        b = torch.load(p, map_location="cpu", weights_only=False)
        if (b["target_h"], b["target_w"]) != (256, 256):
            continue
        rng = torch.Generator(device="cpu")
        rng.manual_seed(cc.random_geom_seed(idx))
        zoom = 1.10 + float(torch.rand((), generator=rng)) * 0.40
        span = zoom - 1.0
        xs = -span + float(torch.rand((), generator=rng)) * 2.0 * span
        ys = -span + float(torch.rand((), generator=rng)) * 2.0 * span
        base = id_coords(b["target_h"] // 16, b["target_w"] // 16, device="cpu")
        b["arms"]["RANDOM"] = transform_camera_coordinates(
            base, zoom=zoom, x_shift=xs, y_shift=ys
        )
        b["random_geom"] = {"zoom": zoom, "x_shift": xs, "y_shift": ys}
        torch.save(b, p)
        n_random += 1
    log(f"RANDOM arm written on {n_random} ordinary 256x256 units")

    # ---------- arm validation (>=32 camera units, before any mass forward) ----------
    log("arm validation (production-path parity) ...")
    from sakuramoon.conditioning.camera import (
        camera_transform_params,
        transform_camera_coordinates,
    )

    n_val = min(64, n_cam)
    val = {
        "n_units_checked": n_val,
        "correct_vs_affine_maxdiff": [],
        "identity_vs_fullcanvas_maxdiff": [],
        "opposite_vs_affine_maxdiff": [],
        "zoom_vs_audit_maxdiff": [],
        "shift_params_vs_audit_maxdiff": [],
        "maps_finite": True,
        "no_self_assignment": True,
        "derangement_reproducible": True,
    }
    for i in range(n_val):
        b = torch.load(units_dir / f"unit-{i:05d}.pt", map_location="cpu", weights_only=False)
        th, tw = b["target_h"] // 16, b["target_w"] // 16
        base = id_coords(th, tw, device="cpu")
        correct = b["arms"]["CORRECT"]
        identity = b["arms"]["IDENTITY"]
        opposite = b["arms"]["OPPOSITE"]
        shuffled = b["arms"]["SHUFFLED"]
        z, xs, ys = b["zoom"], b["x_shift"], b["y_shift"]
        aff = transform_camera_coordinates(base, zoom=z, x_shift=xs, y_shift=ys)
        val["correct_vs_affine_maxdiff"].append(float((correct - aff).abs().max()))
        fc = full_canvas_crop_coordinates(
            th, tw, full_height=th * 16, full_width=tw * 16,
            crop_box=(0, 0, tw * 16, th * 16), device="cpu",
        )
        val["identity_vs_fullcanvas_maxdiff"].append(float((identity - fc).abs().max()))
        opp_aff = transform_camera_coordinates(base, zoom=z, x_shift=-xs, y_shift=-ys)
        val["opposite_vs_affine_maxdiff"].append(float((opposite - opp_aff).abs().max()))
        z2, xs2, ys2 = camera_transform_params(
            viewport=b["target_w"], full_width=b["resized_w"],
            full_height=b["resized_h"], left=b["crop_box"][0], top=b["crop_box"][1],
        )
        val["zoom_vs_audit_maxdiff"].append(abs(z - b["camera_equivalent_zoom"]))
        val["shift_params_vs_audit_maxdiff"].append(
            max(abs(xs - b["camera_shift_x"]), abs(ys - b["camera_shift_y"]))
        )
        for name, m in (("CORRECT", correct), ("IDENTITY", identity),
                        ("OPPOSITE", opposite), ("SHUFFLED", shuffled),
                        ("HALF", b["arms"]["HALF"]), ("OVER", b["arms"]["OVER"])):
            if not torch.isfinite(m).all():
                val["maps_finite"] = False
        if b.get("shuffle_from_unit", i) == i:
            val["no_self_assignment"] = False
    val["derangement_reproducible"] = cc.derangement(n_cam) == perm
    for k in (
        "correct_vs_affine_maxdiff", "identity_vs_fullcanvas_maxdiff",
        "opposite_vs_affine_maxdiff", "zoom_vs_audit_maxdiff",
        "shift_params_vs_audit_maxdiff",
    ):
        val[f"{k}_max"] = max(val[k])
        del val[k]
    log(f"arm validation: {val}")

    if (
        val["correct_vs_affine_maxdiff_max"] > 1e-5
        or val["identity_vs_fullcanvas_maxdiff_max"] > 0.0
        or val["opposite_vs_affine_maxdiff_max"] > 1e-5  # v2: same tolerance class as CORRECT; strict-0.0 tripped on DCU-vs-CPU fp32 1-ulp (1.19e-07) FMA-contract noise in run1
        or not val["maps_finite"]
        or not val["no_self_assignment"]
        or not val["derangement_reproducible"]
    ):
        log("ARM VALIDATION FAILED — STOP before mass forward (spec s12)")
        cc.write_json(cc.OUT / "stage1-manifest.json", {
            "status": "FAILED_ARM_VALIDATION", "validation": val,
        })
        return 2

    # ---------- manifest ----------
    unit_meta = []
    for idx in unit_order:
        b = torch.load(units_dir / f"unit-{idx:05d}.pt", map_location="cpu", weights_only=False)
        unit_meta.append({
            "unit": idx, "cohort": b["cohort"], "sample_id": b["sample_id"],
            "source_shard": b["source_shard"],
            "target": [b["target_h"], b["target_w"]],
            "camera_applied": b["camera_applied"],
            "fallback": b["camera_fallback_reason"],
            "orientation": b["camera_orientation"],
            "zoom": b["camera_equivalent_zoom"],
            "latent_shift": b["camera_latent_center_shift"],
            "norm_offset": b["camera_normalized_offset"],
            "pixel_shift": b["camera_pixel_center_shift"],
            "shift_x": b["x_shift"], "shift_y": b["y_shift"],
            "opp_na": b["opp_na"], "strict_identity": b["strict_identity"],
            "random_geom": b.get("random_geom"),
            "shuffle_from_unit": b.get("shuffle_from_unit"),
            "file_sha256": cc.sha256_file(units_dir / f"unit-{idx:05d}.pt"),
        })
    manifest = {
        "status": "OK",
        "script_sha256": {
            n: cc.sha256_file(Path(__file__).resolve().parent.parent / "final_snapshot" / f"{n}.py")  # deviation 3: frozen scripts live in final_snapshot/
            for n in ("cc_common", "cc_stage1", "cc_stage2", "cc_stage3")
        },
        "code_head": gate["head"],
        "entrance_head": cc.ENTRANCE_HEAD,
        "git": gate,
        "checkpoints": ckpt_evidence,
        "config": {
            "path": str(cc.CONFIG_PATH),
            "sha256": cc.sha256_file(cc.CONFIG_PATH),
            "stage": config.stage.name,
            "resolution": config.stage.resolution,
            "run_seed": config.run.seed,
            "camera_viewport": {
                "enabled": config.data.camera_viewport.enabled,
                "probability": config.data.camera_viewport.probability,
                "min_zoom": config.data.camera_viewport.min_equivalent_zoom,
                "max_zoom": config.data.camera_viewport.max_equivalent_zoom,
            },
        },
        "seeds": {
            "master": cc.MASTER_SEED,
            "t_quantiles": list(cc.T_QUANTILES),
            "t_values": list(tvals),
            "eps_seed_formula": "MASTER_SEED*1000003 + unit_idx*100 + stratum",
            "shuffle_seed": cc.MASTER_SEED,
            "random_geom_seed_formula": "MASTER_SEED*1000003 + 900000 + unit_idx",
        },
        "validation_source": {
            "dir": str(cc.VALIDATION_DIR),
            "selection_seed": 44,
            "shards": [
                {"rel": p["rel"], "bytes": p["bytes"], "sha256": p["local_sha256"]}
                for p in shard_plan
            ],
        },
        "distribution": dist,
        "rejections": rejections,
        "cohort": {
            "camera": n_cam,
            "ordinary": len(ordinary_final),
            "strict_identity_ordinary": len(strict_take),
            "random_geom": n_random,
            "unit_order_camera_range": [0, n_cam - 1],
            "unit_order_ordinary": ordinary_final,
            "total": n_units,
        },
        "arm_validation": val,
        "units": unit_meta,
        "elapsed_s": time.time() - t_start,
    }
    cc.write_json(cc.OUT / "stage1-manifest.json", manifest)
    log(f"stage1 complete in {manifest['elapsed_s']:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
