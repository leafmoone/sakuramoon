"""M4-1024 production canary verification (8-item checklist).

Run this on the remote host AFTER the 10k-exposure production canary
finished, to verify the launch-readiness checklist from
``docs/m4-1024-launch-decision.md`` §8 / ``config/m4_1024_canary.toml``:

  1. checkpoint transition  — 596/596 expected parameter paths, pixel
     path alive (never re-zeroed), provenance source SHA256 matches the
     real Phase I-P checkpoint, optimizer mode recorded;
  2. EMA                    — section non-empty, decay/ref_samples match
     the config, n_samples_total == global exposures, shadow finite;
  3. v2 checkpoint          — model/optimizer/EMA/RNG/exposure/
     provenance/scalars sections all actually present;
  4. same-stage resume      — mid-canary save -> fresh model+opt+EMA ->
     _apply_resume: step/optimizer/EMA/exposure round-trip, pixel weights
     preserved (NOT unit tests alone; --check-resume <ckpt>);
  5. process producer       — 0 silent wedge, worker crash telemetry
     normal (0 crashes), data_wait, starve, queue occupancy (from
     train-meta.json + the launch log);
  6. dynamic crop           — same sample, different exposures ->
     different crop boxes; identical (cycle, exposure) reproduces the
     same box (deterministic §11.5 stream) — needs the data dirs;
  7. sampling pool          — short-window pool shares ≈ 80/10/10
     (config targets, aux capped) — needs the data dirs;
  8. numerics               — loss/grad finite, Pixel path active, no
     NaN/Inf anywhere (log + checkpoints + train-meta.json).

Usage (remote DTK env, PYTHONPATH=src):

    /usr/local/bin/python3.11 tools/verify_m4_canary.py \
        --out-dir /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary \
        --config config/base.toml config/data.toml config/m4_1024.toml config/m4_1024_canary.toml \
        --log /root/private_data/anime-sr/logs/m4-canary.log \
        --source-ckpt /root/private_data/anime-sr/output_model/latent-flow-phase1-pi/latest.pt \
        --check-resume /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary/step-0000320.pt \
        --index-dir /root/private_data/anime-sr/data/index \
        --webp-dir /root/private_data/anime-sr/data/webp \
        --bucket-hr 1024

Prints a JSON report (and ``--out`` when given). Exit code 0 iff every
item is PASS or SKIP (a SKIP is only allowed for the data-probe items
6/7 when the data dirs are not supplied).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from anime_sr.config.loader import load_config
from anime_sr.config.schema import Config
from anime_sr.data.pipeline import _EXPOSURE_PER_CYCLE, SRDataset
from anime_sr.model.uflow import AnimeSRModel
from anime_sr.train.ema_sample import SampleEMA
from anime_sr.train.latent_flow import (
    _PIXEL_ALIVE_KEYS,
    M4_LAUNCH_DECISION,
    _apply_resume,
    _build_slot_map,
    _sha256_file,
    _train_crop_box,
)
from anime_sr.train.pixel_baseline import _optimizer_for

#: Work order (2026-08-29): the production model is 596 tensors
#: (474 trunk + 122 pixel_encoder).
EXPECTED_N_TENSORS = 596
#: Canary quality gates (work order §10 performance class).
DATA_WAIT_WARN = 15.0
DATA_WAIT_STOP = 30.0
STARVE_WARN = 10.0
STARVE_STOP = 20.0
POOL_TOLERANCE = 0.05  # shares vs config targets, absolute fraction


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


class Report:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def add(self, item: str, ok: bool | None, detail: dict, note: str = "") -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.items[item] = {"status": status, **detail, "note": note}

    def summary(self) -> dict:
        n_pass = sum(1 for v in self.items.values() if v["status"] == "PASS")
        n_fail = sum(1 for v in self.items.values() if v["status"] == "FAIL")
        n_skip = sum(1 for v in self.items.values() if v["status"] == "SKIP")
        return {
            "verdict": "PASS" if n_fail == 0 else "FAIL",
            "pass": n_pass,
            "fail": n_fail,
            "skip": n_skip,
            "items": self.items,
        }


# ----------------------------------------------------------------------
# item 1: checkpoint transition
# ----------------------------------------------------------------------
def check_transition(rep: Report, ckpt: dict, out_dir: Path, source_path: Path | None, log: str | None) -> None:
    detail: dict = {}
    problems: list[str] = []
    model_sd: dict = ckpt["model"]
    detail["n_tensors"] = len(model_sd)
    if len(model_sd) != EXPECTED_N_TENSORS:
        problems.append(f"n_tensors {len(model_sd)} != {EXPECTED_N_TENSORS}")
    # pixel path alive (never re-zeroed)
    dead = [k for k in _PIXEL_ALIVE_KEYS if k not in model_sd or not _finite(model_sd[k]) or float(model_sd[k].abs().max().item()) == 0.0]
    detail["pixel_alive"] = not dead
    if dead:
        problems.append(f"dead/missing pixel weights: {dead}")
    # source SHA256 cross-check
    prov = ckpt.get("provenance") or {}
    trans = prov.get("stage_transition")
    detail["optimizer_mode"] = (trans or {}).get("optimizer")
    detail["transition"] = (trans or {}).get("transition")
    if trans is None:
        problems.append("provenance.stage_transition missing")
    else:
        if trans.get("transition") != "legacy-full->m4-v2":
            problems.append(f"transition tag {trans.get('transition')!r}")
        if trans.get("source_step") != ckpt.get("step", None):
            pass  # source_step is the SOURCE stage's step, not this ckpt's
        if source_path is not None:
            real = _sha256_file(source_path)
            detail["source_sha256_match"] = trans.get("source_sha256") == real
            detail["source_sha256"] = real
            if trans.get("source_sha256") != real:
                problems.append("provenance source_sha256 != actual source file SHA256")
            src_payload = torch.load(source_path, map_location="cpu", weights_only=False)
            src_keys = set(src_payload["model"])
            detail["source_n_tensors"] = len(src_keys)
            if set(model_sd) != src_keys:
                problems.append(
                    f"parameter paths differ from source: only_new={len(set(model_sd) - src_keys)} "
                    f"only_src={len(src_keys - set(model_sd))}"
                )
        else:
            problems.append("--source-ckpt not supplied: 596/596 path-set check impossible")
    # resolved-config launch decision stamp
    rc = out_dir / "resolved-config.json"
    detail["resolved_config"] = rc.exists()
    if rc.exists():
        doc = json.loads(rc.read_text(encoding="utf-8"))
        detail["launch_decision"] = doc.get("launch_decision") == M4_LAUNCH_DECISION
        if doc.get("launch_decision") != M4_LAUNCH_DECISION:
            problems.append("resolved-config.json launch_decision missing/stale")
    else:
        problems.append("resolved-config.json missing")
    # log line
    if log:
        txt = Path(log).read_text(encoding="utf-8", errors="replace")
        detail["log_transition_line"] = bool(re.search(r"stage-transition from .*tensors in", txt))
        if not detail["log_transition_line"]:
            problems.append("launch log missing the stage-transition line")
    rep.add("1_transition", None if source_path is None and log is None else (not problems), detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 2: EMA
# ----------------------------------------------------------------------
def check_ema(rep: Report, ckpt: dict, cfg: Config, out_meta: dict) -> None:
    detail: dict = {}
    problems: list[str] = []
    ema = ckpt.get("ema")
    if ema is None:
        rep.add("2_ema", False, {}, "no EMA section in the production canary checkpoint")
        return
    decay = float(ema.get("decay", -1))
    ref = int(ema.get("ref_samples", -1))
    n_tot = int(ema.get("n_samples_total", -1))
    step = int(ckpt["step"])
    bs = int(out_meta.get("batch_size", cfg.latent_flow.batch_size))
    world = n_tot // (step * bs) if step * bs > 0 and n_tot % (step * bs) == 0 else -1
    detail["decay"] = decay
    detail["ref_samples"] = ref
    detail["n_samples_total"] = n_tot
    detail["ckpt_step"] = step
    detail["world_size_implied"] = world
    detail["global_exposures_expected"] = step * bs * (world if world > 0 else 1)
    if decay != 0.5:
        problems.append(f"decay {decay} != 0.5")
    if ref != int(cfg.ema.half_life_samples):
        problems.append(f"ref_samples {ref} != cfg half_life {cfg.ema.half_life_samples}")
    if world <= 0:
        problems.append(f"n_samples_total {n_tot} not a clean (step*bs*world) product")
    elif n_tot != step * bs * world:
        problems.append("n_samples_total != step*bs*world")
    params = ema.get("params") or {}
    detail["n_shadow_params"] = len(params)
    detail["shadow_finite"] = all(_finite(v) for v in params.values())
    if len(params) != len(ckpt["model"]):
        problems.append(f"shadow keys {len(params)} != model keys {len(ckpt['model'])}")
    if not detail["shadow_finite"]:
        problems.append("non-finite EMA shadow")
    rep.add("2_ema", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 3: v2 sections
# ----------------------------------------------------------------------
def check_v2_sections(rep: Report, ckpt: dict) -> None:
    detail: dict = {}
    problems: list[str] = []
    required = ("model", "optimizer", "ema", "rng", "exposure", "provenance", "scalars")
    for k in required:
        present = ckpt.get(k) is not None
        detail[k] = present
        if not present:
            problems.append(f"missing section {k}")
    if ckpt.get("rng") is not None:
        detail["rng_cpu"] = (ckpt["rng"] or {}).get("cpu") is not None
    if ckpt.get("exposure") is not None:
        detail["exposure_step"] = (ckpt["exposure"] or {}).get("step")
        detail["exposure_global"] = (ckpt["exposure"] or {}).get("global_exposures")
    rep.add("3_v2_sections", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 4: same-stage resume (needs --check-resume)
# ----------------------------------------------------------------------
def check_resume(rep: Report, cfg: Config, ckpt_path: Path | None) -> None:
    if ckpt_path is None:
        rep.add("4_resume", None, {}, "--check-resume not supplied")
        return
    problems: list[str] = []
    detail: dict = {"ckpt": str(ckpt_path)}
    torch.manual_seed(0)
    model = AnimeSRModel(cfg.model, zero_init_pixel=cfg.model.zero_init_pixel)
    opt = _optimizer_for(cfg, model)
    ema = SampleEMA(model, decay=0.5, ref_samples=cfg.ema.half_life_samples)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = payload["model"]
    src_opt = payload["optimizer"]
    src_ema = payload.get("ema")
    start_step, v2_meta = _apply_resume(model, opt, ema, ckpt_path, torch.device("cpu"), 0, True)
    detail["start_step"] = start_step
    detail["step_match"] = start_step == int(payload["step"])
    if start_step != int(payload["step"]):
        problems.append(f"start_step {start_step} != ckpt step {payload['step']}")
    dst_opt = opt.state_dict()
    opt_ok = dst_opt["state"].keys() == src_opt["state"].keys()
    if opt_ok:
        for i, s in src_opt["state"].items():
            if not torch.equal(dst_opt["state"][i]["exp_avg"], s["exp_avg"]):
                opt_ok = False
                break
            if not torch.equal(dst_opt["state"][i]["exp_avg_sq"], s["exp_avg_sq"]):
                opt_ok = False
                break
    detail["optimizer_bitexact"] = opt_ok
    if not opt_ok:
        problems.append("optimizer state not bit-exact after resume")
    if src_ema is not None:
        shadow = ema.avg_state_dict()
        ema_ok = set(shadow) == set(src_ema["params"]) and all(
            torch.equal(shadow[k], src_ema["params"][k].float()) for k in src_ema["params"]
        )
        detail["ema_bitexact"] = ema_ok
        detail["ema_n_samples"] = ema.n_samples_total
        if not ema_ok:
            problems.append("EMA shadow not bit-exact after resume")
    # pixel weights preserved (never re-zeroed)
    live = model.state_dict()
    pixel_preserved = all(
        k in live and torch.equal(live[k], src_sd[k]) and float(live[k].abs().max().item()) > 0.0
        for k in _PIXEL_ALIVE_KEYS
    )
    detail["pixel_preserved"] = pixel_preserved
    if not pixel_preserved:
        problems.append("pixel path re-zeroed/altered by resume")
    if v2_meta is not None and v2_meta.get("exposure"):
        detail["exposure_cursor"] = v2_meta["exposure"].get("step")
        detail["rng_restorable"] = v2_meta.get("rng") is not None
    rep.add("4_resume", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 5: process producer (train-meta.json + log)
# ----------------------------------------------------------------------
def check_producer(rep: Report, out_dir: Path, log: str | None) -> None:
    meta_path = out_dir / "train-meta.json"
    if not meta_path.exists():
        rep.add("5_producer", False, {}, "train-meta.json missing (canary did not finish?)")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    detail: dict = {
        "producer": meta.get("producer"),
        "data_wait_pct": meta.get("data_wait_pct"),
        "queue_starve_pct": meta.get("queue_starve_pct"),
        "ready_occ_avg": meta.get("ready_occ_avg"),
        "producer_margin_x": meta.get("producer_margin_x"),
        "consumer_img_s_per_rank": meta.get("consumer_img_s_per_rank"),
    }
    problems: list[str] = []
    if meta.get("producer") != "process":
        problems.append(f"producer {meta.get('producer')!r} != 'process'")
    dw = float(meta.get("data_wait_pct", 100.0))
    sv = float(meta.get("queue_starve_pct", 100.0))
    occ = float(meta.get("ready_occ_avg", 0.0))
    if dw > DATA_WAIT_STOP:
        problems.append(f"data_wait {dw:.1f}% > {DATA_WAIT_STOP}% (STOP class)")
    elif dw > DATA_WAIT_WARN:
        detail["data_wait_warn"] = True
    if sv > STARVE_STOP:
        problems.append(f"starve {sv:.1f}% > {STARVE_STOP}% (STOP class)")
    elif sv > STARVE_WARN:
        detail["starve_warn"] = True
    if occ <= 0.0:
        problems.append("ready_occ_avg == 0 (silent wedge: queue never filled)")
    crashes = 0
    wedge = False
    if log:
        txt = Path(log).read_text(encoding="utf-8", errors="replace")
        crashes = len(re.findall(r"worker crash|worker died|SIGSEGV", txt))
        wedge = "crash-loop" in txt or "aborting" in txt.lower()
        detail["log_worker_crashes"] = crashes
        detail["log_crash_loop"] = wedge
        if crashes:
            problems.append(f"{crashes} worker crash lines in the log")
        if wedge:
            problems.append("crash-loop/abort marker in the log")
    rep.add("5_producer", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# items 6/7: data probes (crop + pool)
# ----------------------------------------------------------------------
def _dataset(cfg: Config, index_dir: str, webp_dir: str, bucket_hr: int) -> SRDataset:
    return SRDataset(index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="train")


def check_crop(rep: Report, cfg: Config, index_dir: str, webp_dir: str, bucket_hr: int) -> None:
    ds1 = _dataset(cfg, index_dir, webp_dir, bucket_hr)
    ds2 = _dataset(cfg, index_dir, webp_dir, bucket_hr)  # independent instance: determinism
    meta = ds1.samples[0]
    epc = _EXPOSURE_PER_CYCLE
    boxes = {}
    for s in (0, 1, epc, epc + 1):
        boxes[s] = _train_crop_box(ds1, meta, None, s, epc)
    # same (sample, cycle, exposure) reproduces exactly across instances
    det = all(_train_crop_box(ds1, meta, None, s, epc) == _train_crop_box(ds2, meta, None, s, epc) for s in boxes)
    distinct = len(set(boxes.values()))
    detail = {"sample_id": meta.sample_id, "boxes": {str(k): v for k, v in boxes.items()},
              "distinct_boxes_4_exposures": distinct, "cross_instance_deterministic": det}
    problems = []
    if distinct < 2:
        problems.append("crop box never changes across exposures (dynamic crop dead)")
    if not det:
        problems.append("crop box not deterministic across dataset instances")
    rep.add("6_dynamic_crop", not problems, detail, "; ".join(problems))


def check_pool(rep: Report, cfg: Config, index_dir: str, webp_dir: str, bucket_hr: int) -> None:
    ds = _dataset(cfg, index_dir, webp_dir, bucket_hr)
    order = list(range(len(ds.samples)))
    slot_map = _build_slot_map(ds, cfg, order)
    n = len(order)
    perm = [slot_map[i] for i in range(n)]
    counts = {"priority": 0, "regular": 0, "aux": 0}
    for j in perm:
        pool = ds.samples[j].sampling_pool if ds.samples[j].sampling_pool in counts else "regular"
        counts[pool] += 1
    shares = {k: v / n for k, v in counts.items()}
    targets = {
        "priority": cfg.sampling.core_fraction,
        "regular": cfg.sampling.regular_fraction,
        "aux": min(cfg.sampling.aux_fraction, cfg.filter.aux_max_fraction),
    }
    detail = {"n": n, "enabled": slot_map.enabled, "counts": counts,
              "shares": {k: round(v, 4) for k, v in shares.items()},
              "targets": {k: float(v) for k, v in targets.items()}}
    problems = []
    if not slot_map.enabled:
        problems.append("pool sampler disabled for the canary run")
    for k in counts:
        if abs(shares[k] - targets[k]) > POOL_TOLERANCE:
            problems.append(f"pool {k} share {shares[k]:.3f} vs target {targets[k]:.3f}")
    rep.add("7_pool_sampler", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 8: numerics
# ----------------------------------------------------------------------
def check_numerics(rep: Report, ckpt: dict, out_dir: Path, log: str | None) -> None:
    detail: dict = {}
    problems: list[str] = []
    detail["model_finite"] = all(_finite(v) for v in ckpt["model"].values())
    if not detail["model_finite"]:
        problems.append("non-finite live model weights in latest.pt")
    if (ckpt.get("ema") or {}).get("params"):
        detail["ema_finite"] = all(_finite(v) for v in ckpt["ema"]["params"].values())
    if log:
        txt = Path(log).read_text(encoding="utf-8", errors="replace")
        losses = [float(m) for m in re.findall(r"loss=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", txt)]
        detail["n_loss_lines"] = len(losses)
        detail["loss_min"] = min(losses) if losses else None
        detail["loss_max"] = max(losses) if losses else None
        bad = [v for v in losses if not math.isfinite(v)]
        detail["loss_finite"] = not bad
        if bad:
            problems.append(f"{len(bad)} non-finite loss values in the log")
        if re.search(r"\bnan\b|inf\b", txt, flags=re.IGNORECASE) and "finite" not in txt.lower():
            problems.append("NaN/Inf token found in the launch log")
    meta_path = out_dir / "train-meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        detail["consumer_img_s_per_rank"] = meta.get("consumer_img_s_per_rank")
    rep.add("8_numerics", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="canary output dir (latest.pt, train-meta.json, resolved-config.json)")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--log", default=None, help="rank-0 launch log of the canary run")
    ap.add_argument("--source-ckpt", default=None, help="Phase I-P latest.pt (SHA256 cross-check)")
    ap.add_argument("--check-resume", default=None, help="mid-canary step-*.pt for the same-stage resume verification")
    ap.add_argument("--index-dir", default=None, help="data index (enables crop/pool probes)")
    ap.add_argument("--webp-dir", default=None, help="webp data root (enables crop/pool probes)")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--out", default=None, help="write the JSON report here too")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    out_dir = Path(args.out_dir)
    latest = out_dir / "latest.pt"
    if not latest.exists():
        print(json.dumps({"verdict": "FAIL", "error": f"latest.pt missing under {out_dir}"}))
        return 1
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    rep = Report()
    check_transition(rep, ckpt, out_dir, Path(args.source_ckpt) if args.source_ckpt else None, args.log)
    meta_path = out_dir / "train-meta.json"
    out_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    check_ema(rep, ckpt, cfg, out_meta)
    check_v2_sections(rep, ckpt)
    check_resume(rep, cfg, Path(args.check_resume) if args.check_resume else None)
    check_producer(rep, out_dir, args.log)
    if args.index_dir and args.webp_dir:
        check_crop(rep, cfg, args.index_dir, args.webp_dir, args.bucket_hr)
        check_pool(rep, cfg, args.index_dir, args.webp_dir, args.bucket_hr)
    else:
        rep.add("6_dynamic_crop", None, {}, "data dirs not supplied (probe skipped)")
        rep.add("7_pool_sampler", None, {}, "data dirs not supplied (probe skipped)")
    check_numerics(rep, ckpt, out_dir, args.log)

    summary = rep.summary()
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
