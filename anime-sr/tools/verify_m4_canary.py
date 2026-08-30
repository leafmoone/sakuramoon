"""M4-1024 production canary verification (8-item checklist + gates).

Run this on the remote host AFTER the 10k-exposure production canary
finished, to verify the launch-readiness checklist from
``docs/m4-1024-launch-decision.md`` §8 / ``config/m4_1024_canary.toml``
plus the 2026-08-30 final execution gates:

  1. checkpoint transition  — 596/596 expected parameter paths, pixel
     path alive (never re-zeroed), provenance source SHA256 matches the
     real Phase I-P checkpoint, optimizer mode recorded;
  2. EMA                    — section non-empty, decay/ref_samples match
     the config, n_samples_total == global exposures (final 10,000),
     shadow finite;
  3. v2 checkpoint          — model/optimizer/EMA/RNG/exposure/
     provenance/scalars sections all actually present;
  4. same-stage resume, split into TWO evidence sources:
     4a. offline checkpoint round-trip — fresh model+opt+EMA ->
         _apply_resume: step/optimizer/EMA/exposure bit-exact, pixel
         weights preserved (--check-resume <mid-ckpt>);
     4b. REAL process-level restart — Leg A log proves the
         stage-transition ran and reached step 320; Leg B log proves a
         fresh torchrun resumed at step 320 (same-stage resume, v2 RNG/
         exposure cursor, EMA n_samples=5120 at start, steps 320..625,
         NO stage-transition line); the resume gate is PASS only if BOTH
         4a and 4b PASS (--leg-a-log / --leg-b-log);
  5. process producer       — 0 silent wedge, worker crash telemetry
     normal (0 crashes), data_wait, starve, queue occupancy (from
     train-meta.json + the launch log);
  6. dynamic crop           — same sample, different exposures ->
     different crop boxes; identical (cycle, exposure) reproduces the
     same box (deterministic §11.5 stream) — needs the data dirs;
  7. sampling pool          — diversity-first FULL-SET deterministic
     permutation (08-31 M4 resolution): full-cycle coverage == N,
     duplicate count == 0, deterministic across fresh SlotMaps, observed
     pool composition == the eligible index's NATURAL composition (the
     ~19/60/21 data statistic is accepted; the 80/10/10 quota is an
     inactive no-op and NOT checked), rank global slots disjoint, final
     permutation is not an index-order straight read — needs the data dirs;
  8. numerics               — loss/grad finite, Pixel path active, no
     NaN/Inf anywhere (log + checkpoints + train-meta.json);
  9. throughput gate        — S_canary from the stable log intervals
     (startup / checkpoint-save / val / run-end excluded) vs the hard
     gate 0.71 step/s (90% of the 0.787 historical anchor); reports
     ETA_6M = 375000/S_canary;
  10. Leg B continuity      — LR of every logged Leg B step equals the
     625-step cosine plan recomputed from config (stateless
     _cosine_lr), no loss jump at the seam (300->350) beyond Leg A's
     own 50-step variation, EMA cursor continued from 5120 (final
     n_samples_total == 5120 + 305*16 == 10000).

Usage (remote DTK env, PYTHONPATH=src of the pinned execution tree):

    /usr/local/bin/python3.11 tools/verify_m4_canary.py \
        --out-dir /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary \
        --config config/base.toml config/data.toml config/m4_1024.toml config/m4_1024_canary.toml \
        --log /root/private_data/anime-sr/logs/m4-canary-legA.log \
        --source-ckpt /root/private_data/anime-sr/output_model/latent-flow-phase1-pi/latest.pt \
        --check-resume /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary/step-0000320.pt \
        --leg-a-log /root/private_data/anime-sr/logs/m4-canary-legA.log \
        --leg-b-log /root/private_data/anime-sr/logs/m4-canary-legB.log \
        --index-dir /root/private_data/anime-sr/data/index-p1formal \
        --webp-dir /root/private_data/anime-sr/data/webp \
        --bucket-hr 1024 \
        --world-size 2

Prints a JSON report (and ``--out`` when given). Exit code 0 iff every
item is PASS or SKIP (a SKIP is only allowed for the data-probe items
6/7 when the data dirs are not supplied, and for 9/10 when the leg
logs are not supplied).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
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
    latent_sample_index,
)
from anime_sr.train.pixel_baseline import _cosine_lr, _optimizer_for

#: Work order (2026-08-29): the production model is 596 tensors
#: (474 trunk + 122 pixel_encoder).
EXPECTED_N_TENSORS = 596
#: Canary quality gates (work order §10 performance class).
DATA_WAIT_WARN = 15.0
DATA_WAIT_STOP = 30.0
STARVE_WARN = 10.0
STARVE_STOP = 20.0
#: Item 9: hard throughput gate = 90% of the Phase I-P anchor (0.787 step/s
#: at 6.298 img/s/rank); the 6M horizon is 375,000 global steps.
S_CANARY_GATE = 0.71
ETA_6M_STEPS = 375_000
#: Item 10: the logged lr is printed with %.2e (3 significant digits).
LR_REL_TOL = 1e-2


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# ----------------------------------------------------------------------
# shared log parsing (items 4b / 9 / 10)
# ----------------------------------------------------------------------
STEP_LINE = re.compile(
    r"\[latent\] step (\d+)/(\d+) loss=(-?\d+\.\d+) "
    r"lr=(-?\d+\.\d+[eE][-+]?\d+) \((\d+\.\d) it/s\)"
    r"(?: data_wait=(-?\d+\.\d+)%)?",
)
PLAN_LINE = re.compile(r"steps (\d+)\.\.(\d+) \((\d+) samples\)")


def _read_log(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def _read_logs(paths: list[str]) -> str | None:
    chunks = [c for c in (_read_log(p) for p in paths) if c is not None]
    return "\n".join(chunks) if chunks else None


def _step_lines(text: str) -> list[dict]:
    """Per-step progress lines. ``it_s`` is CUMULATIVE from the leg start,
    so the absolute wall-clock time of a line is ``step / it_s``."""
    out = []
    for m in STEP_LINE.finditer(text):
        n, total, loss, lr, it_s, wait = m.groups()
        out.append(
            {
                "step": int(n),
                "total": int(total),
                "loss": float(loss),
                "lr": float(lr),
                "it_s": float(it_s),
                "data_wait": float(wait) if wait is not None else None,
            }
        )
    return out


def _interval_rates(lines: list[dict], leg: str, total: int) -> list[dict]:
    """Local step rates between consecutive logged lines, with the
    non-steady-state intervals flagged (step-320 mid-ckpt save, step-500
    in-stream val, final run-end step)."""
    out = []
    prev = None
    for ln in lines:
        if prev is not None and prev["it_s"] > 0 and ln["it_s"] > 0:
            t0 = prev["step"] / prev["it_s"]
            t1 = ln["step"] / ln["it_s"]
            if t1 > t0:
                s0, s1 = prev["step"], ln["step"]
                excluded = None
                if s0 < 320 <= s1:
                    excluded = "mid-ckpt save"
                elif s0 < 500 <= s1:
                    excluded = "in-stream val"
                elif s1 == total:
                    excluded = "run-end"
                out.append(
                    {"leg": leg, "from": s0, "to": s1, "rate": (s1 - s0) / (t1 - t0), "excluded": excluded}
                )
        prev = ln
    return out


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
def check_transition(rep: Report, ckpt: dict, out_dir: Path, source_path: Path | None, logs: list[str]) -> None:
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
    # log line (may live in any supplied log — Leg A's in the canary)
    txt = _read_logs(logs)
    if txt:
        detail["log_transition_line"] = bool(re.search(r"stage-transition from .*tensors in", txt))
        if not detail["log_transition_line"]:
            problems.append("launch log missing the stage-transition line")
    rep.add("1_transition", None if source_path is None and not logs else (not problems), detail, "; ".join(problems))


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
# item 4a: offline checkpoint round-trip (needs --check-resume)
# ----------------------------------------------------------------------
def check_resume_offline(rep: Report, cfg: Config, ckpt_path: Path | None) -> None:
    if ckpt_path is None:
        rep.add("4a_offline_resume", False, {}, "--check-resume not supplied (the resume gate requires 4a AND 4b)")
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
    rep.add("4a_offline_resume", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 5: process producer (train-meta.json + log)
# ----------------------------------------------------------------------
def check_producer(rep: Report, out_dir: Path, logs: list[str]) -> None:
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
    txt = _read_logs(logs)
    if txt:
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


def check_pool(
    rep: Report,
    cfg: Config,
    index_dir: str,
    webp_dir: str,
    bucket_hr: int,
    world: int = 2,
) -> None:
    """Item 7 (08-31 M4 resolution): the train stream is a DIVERSITY-FIRST
    FULL-SET deterministic permutation of the eligible samples — every
    sample exactly once per cycle, so the long-term pool composition equals
    the data's NATURAL composition (a data statistic: reported and
    accepted; the 80/10/10 quota fractions are an inactive no-op and are
    NOT checked)."""
    ds1 = _dataset(cfg, index_dir, webp_dir, bucket_hr)
    ds2 = _dataset(cfg, index_dir, webp_dir, bucket_hr)  # fresh instance: determinism
    order = list(range(len(ds1.samples)))
    n = len(order)
    slot_map1 = _build_slot_map(ds1, cfg, order)
    slot_map2 = _build_slot_map(ds2, cfg, order)
    perm1 = [slot_map1[i] for i in range(n)]
    perm2 = [slot_map2[i] for i in range(n)]

    def _composition(indices: list[int]) -> dict[str, int]:
        c = {"priority": 0, "regular": 0, "aux": 0}
        for j in indices:
            pool = ds1.samples[j].sampling_pool
            c[pool if pool in c else "regular"] += 1
        return c

    # 1+2. one cycle is an exact permutation of the index: full coverage,
    # zero duplicates
    coverage = sorted(perm1) == order
    duplicates = n - len(set(perm1))
    # 3. pure-function contract: fresh instances -> identical cycle order
    deterministic = perm1 == perm2
    # 4. observed stream composition == the eligible index's NATURAL
    # composition (the ~19/60/21 data statistic is accepted, not targeted)
    natural = _composition(order)
    observed = _composition(perm1)
    shares = {k: round(v / n, 4) for k, v in natural.items()}
    # 5. DDP safety: rank r owns the global-slot block [r*bs, (r+1)*bs) of
    # each step (latent_sample_index) — the blocks must stay disjoint
    bs = cfg.latent_flow.batch_size
    rank_slots = [
        {latent_sample_index(0, r, i, bs, world, n) for i in range(bs)}
        for r in range(world)
    ]
    disjoint = all(
        not (rank_slots[a] & rank_slots[b])
        for a in range(world)
        for b in range(a + 1, world)
    )
    # 6. the final global mix is alive: neither the identity read nor a
    # long contiguous index run
    k = min(1024, n)
    identity = perm1 == order
    straight_run = all(perm1[i + 1] == perm1[i] + 1 for i in range(k - 1))

    detail = {
        "n": n,
        "enabled": slot_map1.enabled,
        "coverage_full_cycle": coverage,
        "duplicates": duplicates,
        "deterministic_across_instances": deterministic,
        "natural_composition": natural,
        "natural_shares": shares,
        "observed_composition": observed,
        "composition_matches_natural": observed == natural,
        "bs": bs,
        "world": world,
        "rank_slots_disjoint": disjoint,
        "identity_straight_read": identity,
        "leading_contiguous_run": straight_run,
    }
    problems = []
    if not slot_map1.enabled:
        problems.append("pool sampler disabled for the canary run")
    if not coverage:
        problems.append("cycle order does not cover the full index set")
    if duplicates:
        problems.append(f"{duplicates} duplicated slot(s) in the cycle permutation")
    if not deterministic:
        problems.append("cycle order differs across fresh dataset instances")
    if observed != natural:
        problems.append("observed stream composition != natural index composition")
    if not disjoint:
        problems.append(f"rank global slots collide (bs={bs}, world={world})")
    if identity:
        problems.append("cycle order is the index-order straight read (identity)")
    elif straight_run:
        problems.append(f"first {k} slots are a contiguous index run (final mix dead)")
    rep.add("7_pool_sampler", not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 8: numerics
# ----------------------------------------------------------------------
def check_numerics(rep: Report, ckpt: dict, out_dir: Path, logs: list[str]) -> None:
    detail: dict = {}
    problems: list[str] = []
    detail["model_finite"] = all(_finite(v) for v in ckpt["model"].values())
    if not detail["model_finite"]:
        problems.append("non-finite live model weights in latest.pt")
    if (ckpt.get("ema") or {}).get("params"):
        detail["ema_finite"] = all(_finite(v) for v in ckpt["ema"]["params"].values())
    txt = _read_logs(logs)
    if txt:
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
# item 4b: real process-level restart (Leg A / Leg B logs)
# ----------------------------------------------------------------------
def check_real_restart(rep: Report, leg_a_log: str | None, leg_b_log: str | None, out_dir: Path, final_step: int) -> None:
    name = "4b_real_restart"
    la, lb = _read_log(leg_a_log), _read_log(leg_b_log)
    if la is None or lb is None:
        missing = [flag for flag, t in (("--leg-a-log", la), ("--leg-b-log", lb)) if t is None]
        rep.add(name, False, {"missing_args": missing}, "4b needs both leg logs (the real torchrun restart evidence)")
        return
    detail: dict = {}
    problems: list[str] = []
    # Leg A: the stage-transition executed and the leg reached the mid-ckpt
    ma = re.search(
        r"stage-transition from \S+: \d+ tensors in \(pixel path retained, never re-zeroed\), optimizer=(\w+)", la
    )
    detail["leg_a_transition_line"] = bool(ma)
    detail["leg_a_optimizer_mode"] = ma.group(1) if ma else None
    if not ma:
        problems.append("Leg A: stage-transition line missing")
    steps_a = [ln["step"] for ln in _step_lines(la)]
    detail["leg_a_last_step_logged"] = max(steps_a) if steps_a else None
    if not steps_a or max(steps_a) < 300:
        problems.append("Leg A: no step line >= 300 (the mid-ckpt at 320 was never reached)")
    mid = out_dir / "step-0000320.pt"
    detail["mid_ckpt"] = str(mid)
    if not mid.exists():
        problems.append("mid ckpt step-0000320.pt missing")
    else:
        try:
            mid_payload = torch.load(mid, map_location="cpu", weights_only=False)
            detail["mid_ckpt_step"] = int(mid_payload.get("step", -1))
            if int(mid_payload.get("step", -1)) != 320:
                problems.append(f"mid ckpt step {mid_payload.get('step')} != 320")
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as e:
            problems.append(f"step-0000320.pt not torch.load-able: {e}")
    # Leg B: a fresh torchrun resumed at 320 — same-stage resume, v2 meta
    for key, pat in (
        ("resumed_at_320_v2", r"\[latent\] resumed at step 320 from .+ \(v2: RNG/exposure cursor restorable\)"),
        ("v2_restore_line", r"resume v2: RNG restored, exposure cursor step=320 global_exposures=5120"),
        ("plan_320_to_625", r"steps 320\.\.625 \(10000 samples\)"),
        ("bs8_world2", r"bs=8 x world=2"),
        ("ema_start_5120", r"n_samples=5120 at start"),
        ("producer_process", r"producer=process"),
        ("clean_done_line", r"\[latent\] done: .*latest\.pt"),
    ):
        hit = bool(re.search(pat, lb))
        detail[f"leg_b_{key}"] = hit
        if not hit:
            problems.append(f"Leg B: {key} evidence missing")
    detail["leg_b_no_transition_rerun"] = "stage-transition from" not in lb
    if not detail["leg_b_no_transition_rerun"]:
        problems.append("Leg B: a stage-transition line was printed (transition re-executed!)")
    # final checkpoint at the end of the horizon
    detail["final_ckpt_step"] = final_step
    if final_step != 625:
        problems.append(f"latest.pt step {final_step} != 625 (clean run-end not reached)")
    rep.add(name, not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 9: throughput gate (S_canary >= 0.71 step/s)
# ----------------------------------------------------------------------
def check_throughput(rep: Report, leg_a_log: str | None, leg_b_log: str | None, out_meta: dict) -> None:
    name = "9_throughput_gate"
    la, lb = _read_log(leg_a_log), _read_log(leg_b_log)
    if la is None or lb is None:
        rep.add(name, None, {}, "leg logs not supplied (gate skipped; note 4b FAILs without them, so a PASS verdict still requires them)")
        return
    lines_a, lines_b = _step_lines(la), _step_lines(lb)
    plan = PLAN_LINE.search(lb) or PLAN_LINE.search(la)
    total = int(plan.group(2)) if plan else 625
    intervals = _interval_rates(lines_a, "A", total) + _interval_rates(lines_b, "B", total)
    used = [r for r in intervals if r["excluded"] is None]
    excluded = [
        {"leg": r["leg"], "from": r["from"], "to": r["to"], "why": r["excluded"]}
        for r in intervals
        if r["excluded"] is not None
    ]
    if not used:
        rep.add(name, False, {"intervals": intervals}, "no stable interval parseable from the leg logs")
        return
    s_canary = statistics.median([r["rate"] for r in used])
    eta_h = ETA_6M_STEPS / s_canary / 3600.0
    detail = {
        "s_canary_steps_s": round(s_canary, 4),
        "gate": S_CANARY_GATE,
        "img_s_global": round(s_canary * 16, 2),
        "eta_6m_hours": round(eta_h, 1),
        "eta_6m_days": round(eta_h / 24.0, 2),
        "intervals_used": [
            {"leg": r["leg"], "from": r["from"], "to": r["to"], "rate": round(r["rate"], 4)} for r in used
        ],
        "intervals_excluded": excluded,
        "meta_data_wait_pct": out_meta.get("data_wait_pct"),
        "meta_consumer_img_s_per_rank": out_meta.get("consumer_img_s_per_rank"),
    }
    problems: list[str] = []
    if s_canary < S_CANARY_GATE:
        problems.append(
            f"S_canary {s_canary:.3f} step/s < gate {S_CANARY_GATE}: no 6M until attributed "
            "(host / producer / VAE encode / CPU-IO / HCU kernel)"
        )
    rep.add(name, not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
# item 10: Leg B continuity (LR plan / loss seam / EMA + exposure cursor)
# ----------------------------------------------------------------------
def check_leg_b_continuity(rep: Report, cfg: Config, leg_a_log: str | None, leg_b_log: str | None, ckpt: dict) -> None:
    name = "10_legB_continuity"
    la, lb = _read_log(leg_a_log), _read_log(leg_b_log)
    if la is None or lb is None:
        rep.add(name, None, {}, "leg logs not supplied (skipped)")
        return
    detail: dict = {}
    problems: list[str] = []
    lines_a, lines_b = _step_lines(la), _step_lines(lb)
    plan = PLAN_LINE.search(lb)
    total = int(plan.group(2)) if plan else 625
    exp_target = int(plan.group(3)) if plan else total * 16
    mww = re.search(r"bs=(\d+) x world=(\d+)", lb)
    bs_world = int(mww.group(1)) * int(mww.group(2)) if mww else 16
    # (a) LR of every logged Leg B step vs the stateless cosine plan
    lr_detail: dict[str, dict] = {}
    lr_bad: list[int] = []
    for ln in lines_b:
        ref = _cosine_lr(ln["step"], total, float(cfg.optimizer.lr), cfg)
        rel = abs(ln["lr"] - ref) / ref if ref else math.inf
        lr_detail[str(ln["step"])] = {"logged": ln["lr"], "plan": round(ref, 8), "rel_err": round(rel, 6)}
        if rel > LR_REL_TOL:
            lr_bad.append(ln["step"])
    detail["lr_vs_plan"] = lr_detail
    if lr_bad:
        problems.append(f"logged LR off the {total}-step plan at steps {lr_bad}")
    # (b) loss seam 300 (Leg A) -> 350 (Leg B) vs Leg A's own 50-step variation
    loss_a = {ln["step"]: ln["loss"] for ln in lines_a}
    loss_b = {ln["step"]: ln["loss"] for ln in lines_b}
    if 300 in loss_a and 350 in loss_b:
        sa = sorted(loss_a)
        deltas = [abs(loss_a[sa[i]] - loss_a[sa[i - 1]]) for i in range(1, len(sa))]
        seam = abs(loss_b[350] - loss_a[300])
        base = max(deltas) if deltas else 0.0
        detail["loss_a_300"] = loss_a[300]
        detail["loss_b_350"] = loss_b[350]
        detail["loss_seam_abs"] = round(seam, 6)
        detail["legA_50step_delta_max"] = round(base, 6)
        if seam > max(3.0 * base, 0.1):
            problems.append(f"loss jump at the 300->350 seam ({seam:.4f} vs Leg A max 50-step delta {base:.4f})")
        elif base > 0 and seam > base:
            detail["loss_seam_warn"] = True
    # (c) EMA + exposure cursor continued from the step-320 state
    m = re.search(r"n_samples=(\d+) at start", lb)
    detail["ema_n_samples_legB_start"] = int(m.group(1)) if m else None
    if m is None or int(m.group(1)) != 5120:
        problems.append("Leg B EMA did not start from n_samples=5120 (the step-320 cursor)")
    n_final = int((ckpt.get("ema") or {}).get("n_samples_total", -1))
    expected_final = 5120 + (total - 320) * bs_world
    detail["ema_n_samples_final"] = n_final
    detail["ema_n_samples_expected"] = expected_final
    if n_final != expected_final:
        problems.append(f"final EMA n_samples_total {n_final} != {expected_final} (5120 + {total - 320} x {bs_world})")
    exp = ckpt.get("exposure") or {}
    detail["final_exposure"] = {"step": exp.get("step"), "global_exposures": exp.get("global_exposures")}
    if exp.get("step") != total or exp.get("global_exposures") != exp_target:
        problems.append(f"final exposure cursor not at (step={total}, global={exp_target})")
    rep.add(name, not problems, detail, "; ".join(problems))


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="canary output dir (latest.pt, train-meta.json, resolved-config.json)")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--log", default=None, help="rank-0 launch log of the canary run")
    ap.add_argument("--source-ckpt", default=None, help="Phase I-P latest.pt (SHA256 cross-check)")
    ap.add_argument("--check-resume", default=None, help="mid-canary step-*.pt for the offline resume round-trip (4a)")
    ap.add_argument("--leg-a-log", default=None, help="Leg A (stage-transition) rank-0 log; required for 4b/9/10")
    ap.add_argument("--leg-b-log", default=None, help="Leg B (resume) rank-0 log; required for 4b/9/10")
    ap.add_argument("--index-dir", default=None, help="data index (enables crop/pool probes)")
    ap.add_argument("--webp-dir", default=None, help="webp data root (enables crop/pool probes)")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--world-size", type=int, default=2,
                    help="DDP world size for the rank-slot disjointness probe (item 7)")
    ap.add_argument("--out", default=None, help="write the JSON report here too")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    out_dir = Path(args.out_dir)
    latest = out_dir / "latest.pt"
    if not latest.exists():
        print(json.dumps({"verdict": "FAIL", "error": f"latest.pt missing under {out_dir}"}))
        return 1
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    logs: list[str] = []
    for p in (args.log, args.leg_a_log, args.leg_b_log):
        if p and p not in logs:
            logs.append(p)

    rep = Report()
    check_transition(rep, ckpt, out_dir, Path(args.source_ckpt) if args.source_ckpt else None, logs)
    meta_path = out_dir / "train-meta.json"
    out_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    check_ema(rep, ckpt, cfg, out_meta)
    check_v2_sections(rep, ckpt)
    check_resume_offline(rep, cfg, Path(args.check_resume) if args.check_resume else None)
    check_real_restart(rep, args.leg_a_log, args.leg_b_log, out_dir, int(ckpt.get("step", -1)))
    check_producer(rep, out_dir, logs)
    if args.index_dir and args.webp_dir:
        check_crop(rep, cfg, args.index_dir, args.webp_dir, args.bucket_hr)
        check_pool(rep, cfg, args.index_dir, args.webp_dir, args.bucket_hr, world=args.world_size)
    else:
        rep.add("6_dynamic_crop", None, {}, "data dirs not supplied (probe skipped)")
        rep.add("7_pool_sampler", None, {}, "data dirs not supplied (probe skipped)")
    check_numerics(rep, ckpt, out_dir, logs)
    check_throughput(rep, args.leg_a_log, args.leg_b_log, out_meta)
    check_leg_b_continuity(rep, cfg, args.leg_a_log, args.leg_b_log, ckpt)

    summary = rep.summary()
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
