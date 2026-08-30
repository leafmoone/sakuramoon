# M4-1024 launch decision (2026-08-29 M4 resolution revision)

> **2026-08-29 M4 resolution revision: primary production M4 uses 256→1024
> only. Mixed-resolution curriculum is deferred and is not silently implied
> by the current run.**

This statement is stamped verbatim into `resolved-config.json` and every v2
provenance block by the trainer (`M4_LAUNCH_DECISION` in
`src/anime_sr/train/latent_flow.py`). The old 512/768/1024 curriculum in
`config/stage1_flow.toml` is **kept, marked DEFERRED/inactive** — not
deleted — so documentation never drifts from what the run actually does
(the production trainer is a single `--bucket-hr` run; no mixed-bucket
scheduler exists).

## 1. Stage scope (frozen)

| item | value |
|---|---|
| name | **M4-1024** (primary production M4 stage) |
| model | AnimeSR-Mage-UFlow, full `AnimeSRModel` (122.67M trunk + PixelConditionEncoder, 596 tensors) |
| task | 4× blind anime SR, HR 1024 / LQ 256 (single bucket `--bucket-hr 1024`) |
| z_hr | on-the-fly (P1 ④; no latent store) |
| pixel features | enabled (trained path carried from Phase I-P) |
| producer | process pool (pre-fork before NCCL, canary #6 verified) |
| dynamic crop | enabled, deterministic (§11.5 exposure identity) |
| pool sampler | enabled — diversity-first FULL-SET deterministic permutation
  (08-31 resolution: quota inactive, natural ~19/60/21 composition; the
  80/10/10 target was never achievable under the full-set contract) |
| clean score | `clean_score_min = -1.0` (report-only; the stale buggy
  `clean-score-v1.jsonl` is NOT reused — see §6) |
| EMA | production `SampleEMA`, half-life 500k GLOBAL exposures |
| checkpoint | v2 (model/optimizer/EMA/step/exposure cursor/RNG/scalars/provenance) |
| attention backend | `sdpa-correctness` = the frozen verified EXPLICIT core
  (no kernel change at the stage boundary) |
| first-stage budget | **6,000,000 exposures (pinned)** — 10M/16M only after
  the 6M end-judgement, as an explicit separate decision; never auto-extended |

## 2. Source checkpoint & transition (Phase I-P → M4-1024)

Source: `/root/private_data/anime-sr/output_model/latent-flow-phase1-pi/latest.pt`
(SHA256 prefix `9c22a6d899dc66386771ed14…`, step 18,750).

Verified schema (2026-08-30): **v1 legacy** `{step, model, optimizer}` —
596 model tensors (474 trunk + 122 pixel_encoder), complete AdamW optimizer
(2 groups, 596 state entries, all with shape-exact `exp_avg`/`exp_avg_sq`,
betas (0.9, 0.95), wd 0.05/0.0), group LR at 1.5e-5 = the cosine floor of
the finished 300k pilot schedule. No EMA/v2 sections.

**Transition mode: `--stage-transition`** (new, M4-1024 work order) — an
explicit `legacy-full -> M4-v2` stage transition. It is deliberately NOT
`--resume` (that continues the SAME stage with step/exposure continuity —
disguising a stage start as a resume is prohibited) and NOT `--init-trunk`
(that path is for trunk-only→pixel with zero-init; loading the trained full
pixel checkpoint there is prohibited).

Semantics of the transition:

| aspect | behavior |
|---|---|
| model weights | **strict full load** — all 596 tensors in, no key filtering, **no `apply_pixel_zero_init` ever** (pixel-alive guard rejects an all-zero `trunk.proj_p64/p32/p16` source) |
| optimizer | **INHERITED** (the decision): the Phase I-P AdamW state passes the explicit compatibility gate — complete shape-exact moments for all 596 params + identical betas/eps/wd. If the gate ever fails, the run prints and records `optimizer=fresh` (a fresh-optimizer stage transition, never a same-stage resume) |
| scheduler | **FRESH** over the M4 horizon: step 0 starts the 3% warmup (11,250 steps at global batch 16), cosine to the 10% floor over 375,000 steps. The inherited group LR (1.5e-5, Phase I-P tail) is NOT carried — the trainer re-sets `g["lr"]` every step |
| warmup × inherited moments coexistence | Adam update = `lr · m̂/(√v̂+ε)` where m/v are **unscaled by lr** (raw gradient moments). Inheriting m/v while restarting the lr schedule is therefore well-defined: step 0 applies a ~1.3e-8 LR to warmed-up directions (small, safe update); the schedule is monotone non-decreasing through warmup (no LR jump), finite everywhere (tested) |
| EMA | fresh `SampleEMA` **seeded from the loaded (source) weights** — the M4 EMA has no Phase I-P history |
| step / exposure | **0 / 0** for the M4 stage (fresh cursor; the §11.5 stream identity is step-relative, so the M4 stream starts clean) |
| provenance | every M4 v2 checkpoint records `stage_transition = {transition: "legacy-full->m4-v2", source_sha256 (full 64-hex), optimizer: "inherited"|"fresh", ema: "seeded-from-source-weights", n_model_tensors, source_step}` — the source SHA256 is the explicit M4-1024 requirement (superseding the repo no-project-hash rule for this identifier per the launch work order) |

A legitimate production **v2** checkpoint (with an EMA section) is rejected
by `--stage-transition` and must use `--resume` — the two entry points
cannot be confused.

## 3. Batch, budget, schedule

* global batch = `batch_size × world_size` = 8 × 2 = **16 exposures/step**
* total optimizer steps = 6,000,000 / 16 = **375,000**
* warmup = ⌊0.03 × 375,000⌋ = **11,250 steps**, then cosine → `min_lr_ratio 0.10` (floor 1.5e-5)
* EMA: one step feeds `n_samples = 16`; `n_samples_total` after the run = 6,000,000 (= global exposures); retention = 0.5 after 500k, 0.25 after 1M, …

## 4. Checkpoint & validation cadence (world = 2)

**08-31 post-canary resolution**: the production v2 grid is the PERIODIC
`save_every_steps = 15,625` (= every **250k GLOBAL exposures**) PLUS the 100k
milestone. A step landing on both a periodic due and a milestone writes
exactly ONE `step-NNNNNNN.pt` (single `save_v2` call). Rationale: cap the
worst-case training loss on the 4.8-day run at 250k exposures. Volume:
24 periodic + 1 milestone-only ≈ 25 files × ~1.23 GiB ≈ **31 GiB** on the
output volume (fits with headroom — re-verify free space pre-launch).

| global exposures | step | production v2 ckpt | held-out probe (live + EMA) |
|---|---|---|---|
| 100k | 6,250 | `step-0006250.pt` (milestone only) | |
| 250k | 15,625 | `step-0015625.pt` (periodic) | |
| 500k | 31,250 | `step-0031250.pt` (periodic) | ✔ |
| 1M | 62,500 | `step-0062500.pt` (periodic) | ✔ |
| 1.5M | 93,750 | `step-0093750.pt` (periodic) | ✔ |
| 2M | 125,000 | `step-0125000.pt` (periodic) | ✔ |
| 2.5M | 156,250 | `step-0156250.pt` (periodic) | ✔ |
| 3M | 187,500 | `step-0187500.pt` (periodic) | ✔ |
| 3.5M | 218,750 | `step-0218750.pt` (periodic) | ✔ |
| 4M | 250,000 | `step-0250000.pt` (periodic) | ✔ |
| 4.5M | 281,250 | `step-0281250.pt` (periodic) | ✔ |
| 5M | 312,500 | `step-0312500.pt` (periodic) | ✔ |
| 5.5M | 343,750 | `step-0343750.pt` (periodic) | ✔ |
| 6M | 375,000 | `step-0375000.pt` (periodic) + `latest.pt` (final) | ✔ + 6M extras (RGB eval, seam probe, gradient coverage, EMA-vs-live, 1-step vs 4-step, stress set) |

Held-out cadence (2026-08-30 correction): `val_heldout_every_steps = 31,250`
at global batch 16 = **one probe every 500k exposures**, i.e. **12 held-out
nodes total: 0.5M / 1.0M / 1.5M / 2.0M / 2.5M / 3.0M / 3.5M / 4.0M / 4.5M /
5.0M / 5.5M / 6.0M** (12 × 31,250 = 375,000 = run end, deduplicated). The
production v2 ckpt grid (periodic 250k + the 100k milestone) is independent
of the probe grid — do not read the ckpt grid as the probe grid.

Probes: `l1_anchor`, `l1_1` (live **and** EMA), `l1_4`, `ratio_4_1`,
`toward_1`, `cos_v`, endpoint consistency, trajectory deviation, Pixel-path
health. `ratio_4_1 ≤ 1.05` remains the multi-step stability gate — do NOT
kill a healthy M4 just because `l1_4` has not beaten `l1_1` yet.

## 5. Attention backend (P2-2)

Default stays `sdpa-correctness` (frozen explicit core; Phase I-P trained
under it). SDPA variants are benchmark-ready only. Switching to
`sdpa-repeat`/`sdpa-native-gqa` in a main run requires, **on a healthy
host**: parity PASS + bf16 numeric PASS + a real throughput win, then a
separate decision. Benchmark commands (three backends side by side, numeric
diff + auto utilization sampling):

```bash
cd /root/anime-sr-p1formal   # or a healthy host
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
# fp32 numeric parity probe (SAFE on the current bad-state host: no bf16/conv trigger)
/usr/local/bin/python3.11 tools/bench_attention_backends.py \
    --H 64 --W 64 --dim 384 --heads 6 --kv 2 --dtype fp32 --iters 30 --out bench-m4-fp32.json
# bf16 throughput/latency/memory/utilization (HEALTHY HOST ONLY — the
# current sakrua10 host has the DTK bf16/conv driver-leak bad state)
/usr/local/bin/python3.11 tools/bench_attention_backends.py \
    --H 64 --W 64 --dim 384 --heads 6 --kv 2 --dtype bf16 --iters 60 --out bench-m4-bf16.json
# parity gate (fp32 fwd/bwd, shifted/unshifted, padded/exact, global,
# 20-step trajectory, bf16 rel-L2 report, model trajectory)
/usr/local/bin/python3.11 -m pytest tests/test_p2_sdpa_parity.py tests/test_p2_sdpa_backend.py -q
```

## 6. Clean score (report-only for M4-1024)

* The remote `data/index/clean-score-v1.jsonl` (34,901 rows) was produced by
  the **legacy buggy** lazy algorithm — it is prohibited from gating the
  M4-1024 set. `clean_score_min = -1.0` (report-only) is the default;
  **no hard gate is enabled without an explicit user-approved threshold.**
* The stale sidecar is backed up (`clean-score-v1.jsonl.buggy-<date>`) and
  the fixed CLI recomputes the eligible-train set (20,418 samples, CPU-only,
  incremental, single writer) in the background — non-blocking for the
  canary/6M. When done, the distribution report (p10/p25/p50/p75/p90 +
  priority/regular/aux buckets + candidate thresholds 0.5/0.6/0.7) is the
  input to a future gate decision.

```bash
# (already started; re-runs are incremental/skipping)
cd /root/anime-sr-p1formal
/usr/local/bin/python3.11 -m anime_sr.cli.clean_score_precompute \
    --config config/base.toml config/data.toml \
    --index-dir /root/private_data/anime-sr/data/index \
    --webp-dir /root/private_data/anime-sr/data/webp --every 500
```

## 7. Canary (10k exposures) — must pass BEFORE the 6M

```bash
cd /root/anime-sr-p1formal
env LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib \
torchrun --nproc_per_node=2 --master_port=29511 -m anime_sr.cli.train_latent_flow \
    --config config/base.toml config/data.toml config/m4_1024.toml config/m4_1024_canary.toml \
    --index-dir /root/private_data/anime-sr/data/index \
    --webp-dir /root/private_data/anime-sr/data/webp \
    --vae /root/private_data/anime-sr/model/vae/mage-vae.safetensors \
    --out-dir /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary \
    --stage-transition /root/private_data/anime-sr/output_model/latent-flow-phase1-pi/latest.pt
```

Checklist (verify with `tools/verify_m4_canary.py` + log inspection):

1. **checkpoint transition** — 596/596 param paths in; Pixel path alive
   (`trunk.proj_p64/p32/p16` non-zero in the saved ckpt); NEVER re-zeroed;
   provenance `source_sha256` = the Phase I-P sha.
2. **EMA** — first ckpt's EMA section non-empty; `n_samples_total` == global
   exposures at save (5,120 at the mid-canary ckpt); live/EMA finite;
   save/load round-trip.
3. **v2 sections** — model/optimizer/EMA/RNG/exposure cursor/provenance
   all present in `step-0000320.pt` and `latest.pt`.
4. **same-stage resume** — kill after the mid-canary save, relaunch with
   `--resume …/step-0000320.pt`, continue to 625 steps (NOT unit tests
   alone): RNG/exposure cursor restored, no step loss.
   ```bash
   # resume leg (after canary leg 1 stops at/past step 320)
   env LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib \
   torchrun --nproc_per_node=2 --master_port=29511 -m anime_sr.cli.train_latent_flow \
       --config config/base.toml config/data.toml config/m4_1024.toml config/m4_1024_canary.toml \
       --index-dir /root/private_data/anime-sr/data/index \
       --webp-dir /root/private_data/anime-sr/data/webp \
       --vae /root/private_data/anime-sr/model/vae/mage-vae.safetensors \
       --out-dir /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary \
       --resume /root/private_data/anime-sr/output_model/latent-flow-m4-1024-canary/step-0000320.pt
   ```
5. **process producer** — 0 silent wedge; worker crash telemetry sane;
   `data_wait` (gate: ≤15% normal / 15–20% WARN / >20% sustained
   investigate / >30% + HCU starvation STOP); starve; queue occupancy
   (run-end `train-meta.json`).
6. **dynamic crop** — same sample, different exposures → different crop
   boxes (`tools/verify_m4_canary.py --probe-data` uses the production
   `_train_crop_box` path).
7. **pool sampler** — short-window realized shares ≈ 80/10/10 (probe over
   the first stream slots; `verify_m4_canary.py --probe-data`).
8. **numerics** — loss finite, grad finite, Pixel path active, no NaN/Inf
   across all 625 steps.

## 8. 6M launch (only after the canary passes AND the HCU health gate is
green — see §10)

```bash
cd /root/anime-sr-p1formal
env LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib \
nohup torchrun --nproc_per_node=2 --master_port=29511 -m anime_sr.cli.train_latent_flow \
    --config config/base.toml config/data.toml config/m4_1024.toml \
    --index-dir /root/private_data/anime-sr/data/index \
    --webp-dir /root/private_data/anime-sr/data/webp \
    --vae /root/private_data/anime-sr/model/vae/mage-vae.safetensors \
    --out-dir /root/private_data/anime-sr/output_model/latent-flow-m4-1024 \
    --stage-transition /root/private_data/anime-sr/output_model/latent-flow-phase1-pi/latest.pt \
    > /root/private_data/anime-sr/rgb-eval-logs/m4-1024-6m.log 2>&1 &
```

## 9. ETA (measured on this host — no 20–60h guesses)

Phase I-P-1024 on THIS host (same model/resolution/config class):
18,000→18,750 steps took 8,477 s and 10,000→18,000 took 8,477 s +
9,000 steps in 9,660 s → **≈1.06–1.07 s/step** (late-run steady state);
the whole-run average (train-meta) is 16 / (2×6.298) = 1.27 s/step.

```
ETA(remaining) = (exposures_remaining / global_batch) × t_step
               = remaining_steps × t_step
t_step          = rolling mean of the trainer's own it/s log lines
                  (every 50 steps), last ~1,000 steps preferred
```

* at 1.07 s/step: 375,000 × 1.07 ≈ 401,000 s ≈ **111 h ≈ 4.6 days**
* at 1.27 s/step: 375,000 × 1.27 ≈ 476,000 s ≈ **132 h ≈ 5.5 days**
* plus <2 h validation/milestone overhead. **Planning value: ~5 days.**
* Re-measure `t_step` from the canary and the first 10k production steps;
  if the host state degrades (the DTK bad state), the formula still holds
  but `t_step` and the leak gate (§10) dominate — STOP conditions apply.

## 10. Stop conditions & host-state guard

STOP immediately: NaN/Inf; optimizer corruption; repeated process-worker
crash loop; resume state mismatch; Pixel path unexpectedly dead; data-stream
determinism violation; v2 checkpoint cannot fully restore.
Performance: `data_wait` ≤15% normal, 15–20% WARN, sustained >20%
investigate, sustained >30% with HCU starvation → STOP & investigate.

**Host guard:** sakrua10's host (f9648edc) carries the confirmed DTK/HSA
driver bad state (bf16/conv(MIOpen) allocation pattern → phantom HBM
accounting OOM). M4-1024 is a bf16 + conv training load — exactly the
trigger class. Preconditions for ANY M4 start: the `eval-gate-watch.sh`
probe (30-min 1-pair, peak ≤ 20 GiB) reports HEALTHY, and the run is
monitored (cgroup + VMA). If the host dies mid-run: the v2 checkpoints
(milestones above) + the Phase I-P source make a clean re-start possible on
a healthy host; nothing here trains anything yet — the canary is prepared
but NOT started (awaiting the explicit user go).

## 11. What this stage deliberately does NOT do

* no mixed-resolution curriculum (deferred; §1 above)
* no SDPA backend switch (P2-2: benchmark-ready only)
* no clean-score hard gate (report-only until an explicit threshold)
* no auto-extension beyond 6M (end-judgement: STOP→Stage II / EXTEND→10M /
  re-evaluate 16M)
* no Stage II, no new architecture/flow/loss changes (M4-1024 reuses the
  frozen Phase-I core verbatim)
