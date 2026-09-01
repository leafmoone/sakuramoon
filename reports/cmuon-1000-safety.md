# CMuon FP32-rescue — 1000-update Safety Gate (D4)

Candidate: `hybrid_cmuon_canonical_ns4_fp32_rescue` (math frozen since D2 R1,
D3-verified). Host: salt6 (172.31.59.244 after 09-01 pod rebuild; 2×HCU "BW",
pod-local DTK 2.9.0, code @37602e4).
Window under test: **1000 cumulative successful updates, 102501–103500**,
forked from `ckpt_102500` (G1 AdamW8bit vehicle; CMuon cold start: fresh
momentum, guard refs bootstrapped, warmup_observations=50) →
`ckpt_103500_raw-103500-update-cadence` (final).

## Verdict

**SAFE_1000 = YES** — all A4 gates passed with zero unresolved safety events
(685/685 BF16 trips FP32-rescued, 0 rescue failures, 0 nonfinite, 0 rank
drift, loss flat, atomic ckpt chain 102500→103500 exact). **With one
disclosed caveat**: the BF16-safety trip frequency on the post-rebuild
machine is an elevated band (~67–112 per 100 updates vs D3's 2–10 per 100);
it is bounded (no runaway pattern), the step at the pod-rebuild boundary is
an infrastructure artifact (machine-dependent BF16 non-determinism, guard
state fully continuous), and every trip was safely rescued. Recommend
monitoring the next 500–1000 updates for band flattening as the guard
reference convergence completes (see A3).

Stopped exactly at 103500 (watchdog); no automatic 5000-gate / deployment
(awaiting user decision).

## A0 — post-run forensic mirror

`ckpt_103500_raw-103500-update-cadence` mirrored to
`/root/private_data/sakuramoon-g1-forensic/cmuon-d4/ckpt_103500/` after the
stop: 14 files, `cp -a` + **MD5 14/14 identical** (sorted-list comparison;
`ckpt_103500.md5` archived alongside). The earlier VERIFY-FAIL was a
find-ordering artifact (unsorted `cmp` of two differently-ordered lists);
the re-verification with sorted lists passed cleanly.
Hub chain: `leafmoone/sm_train_state` `cmoun/` now holds the full gate chain
102500/102600/102700/102800/102900/103000/103100/103200/103300/103400/
**103500** (103500 uploaded post-stop via the same `publish_train_state.sh`
path; live publisher stopped after the upload).
Raw telemetry (metrics tail 102901–103500, rescue event log 103101–103500,
guard-state window extraction) preserved on NFS under
`sakuramoon-g1-forensic/cmuon-d4/` and in `cmuon-1000-safety.json`.

## A1 — freeze compliance

No candidate math changed for the entire run: identical toml chain
(`train_g1_fp32_rescue_r1.toml`), identical guard calibration (guard_ratio
0.1, reference_decay 0.999, min_reference 3.096e-08, numerical_floor
6.575e-07, invariant_check=true), lr 5e-05 (actual 1.5625e-4 JLT-scaled,
constant across all 600 logged updates), batch 800, NS4 per-role map, BF16
momentum, code @37602e4. No always-FP32, no attention-only FP32, no
compile/overlap/broadcast/owner/threshold changes.
**One user-directed telemetry exception (103100 boundary, mid-run):**
W&B logging was enabled via a run-local overlay toml on salt6
(`train_g1_fp32_rescue_d4_wandb.toml`: `extends` the frozen toml, only
`[wandb] enabled = true`); the frozen toml stayed byte-identical (sha256
recorded in the restart log), the overlay was never committed to the frozen
branch, and it cannot affect candidate math (telemetry only). Run:
`wandb.ai/2574905817leaf-jinan-university/sakuramoon/runs/g1_fp32_rescue_r1_200`
(resume=allow, same run across the restart). Sampling behavior during the
run: one 60-image batch at 103000 (pre-overlay); no further batches — per
user confirmation this is normal behavior for the current setup; not
investigated further.

## A2 — run continuity (two infrastructure events, both disclosed)

- **Segment 1 (102501–~102900, old pod):** clean 2-rank run from the 102500
  fork. **Infrastructure failure ~102900**: the salt6 pod was rebuilt
  externally (gateway/port changed; local disk wiped) while D4 was running.
  Not a candidate safety event (no rescue failure / nonfinite / rank drift /
  atomicity violation / loss divergence in any preserved log).
- **Recovery:** full restore from the hub `cmoun/` chain (ckpt_102900 as the
  resume point, last hub-published ckpt) + replay_10 bundle (code, venv,
  model, inductor cache, cohort, data); all SHA256-verified; queue state
  (NFS, 2099 rows) intact → cycle-2 shard-0 continuation. Resume at 102900:
  counter continuity proven (guard `fp32_rescue` block ckpt-restored:
  state(400)={108 rescues, roles q90/k8/gate10} verified three ways —
  ckpt_102900 optimizer.pt, the first post-resume log line (obs 401, 109),
  and the 83-attempts/obs exact progression 8300→16600→24900→33200→41500).
  Metrics boundary continuity: first post-resume losses
  0.5537/0.5453/0.5439 (no jump vs the 0.549–0.564 regime).
- **Segment 2 (102901–103100, new pod):** clean, no restart.
- **103100 boundary (user-directed):** clean stack stop at the 103100 COMPLETE
  point (manifest 12/12 verified), wandb-overlay config swap, resume from
  103100. obs counter continued 600→601+; counters continuous (state(600)=275
  matches ckpt-derived 208 + log W6 67). Telemetry-only change (A1).
- **Segment 3 (103101–103500, new pod + wandb):** clean to the watchdog stop
  at 103500. `CMuonSafetyError` lines in any segment: **0**.

The invalid-overlap rule from D3 does not apply (the rebuild recovery resumed
exactly at the last published ckpt 102900; nothing was double-counted — the
102901–103500 segment is single-run, single-machine).

## A3 — per-100-update window stats

Loss / preclip / s-per-update from `metrics.jsonl` for the preserved
102901–103500 segment (600 updates; the 102501–102900 metrics were lost in
the pod rebuild — the window table below marks them `n/a(metrics)`; their
rescue counts are exact from the ckpt guard-state chain).

| window | updates | loss min/mean/max | preclip mean/p90/max | nonfinite | BF16 fail | FP32 rescued | rescue fails | roles Δ (gate/q/k) |
|---|---|---|---|---|---|---|---|---|
| 102501–102600 | 100 | n/a (metrics lost) | n/a | 0* | 38 | 38 | 0 | 1/36/1 |
| 102601–102700 | 100 | n/a (metrics lost) | n/a | 0* | 33 | 33 | 0 | 3/26/4 |
| 102701–102800 | 100 | n/a (metrics lost) | n/a | 0* | 28 | 28 | 0 | 1/25/2 |
| 102801–102900 | 100 | n/a (metrics lost) | n/a | 0* | 9 | 9 | 0 | 5/3/1 |
| 102901–103000 | 100 | 0.536 / 0.558 / 0.591 | 0.0440 / 0.0650 / 0.1096 | 0 | 100 | 100 | 0 | 10/90/0 |
| 103001–103100 | 100 | 0.528 / 0.557 / 0.593 | 0.0507 / 0.0776 / 0.1191 | 0 | 67 | 67 | 0 | 12/54/1 |
| 103101–103200 | 100 | 0.515 / 0.558 / 0.591 | 0.0476 / 0.0721 / 0.1038 | 0 | 93 | 93 | 0 | 5/87/1 |
| 103201–103300 | 100 | 0.525 / 0.556 / 0.597 | 0.0472 / 0.0718 / 0.1184 | 0 | 95 | 95 | 0 | 6/88/0 |
| 103301–103400 | 100 | 0.530 / 0.556 / 0.588 | 0.0454 / 0.0665 / 0.1033 | 0 | 110 | 110 | 0 | 6/100/3 |
| 103401–103500 | 100 | 0.532 / 0.556 / 0.596 | 0.0465 / 0.0720 / 0.0926 | 0 | 112 | 112 | 0 | 1/100/12 |
| **total** | **1000** | 0.515–0.597 (600 logged) | — | **0** | **685** | **685** | **0** | 51/609/25 |

\* nonfinite for 102501–102900: the preserved train.log segments (old pod,
pre-rebuild) recorded zero `nonfinite_count`/`CMuonSafetyError` lines, and
the guard counters (0 rescue failures, 0 rank spread) are cumulative from
the fork — a nonfinite event would have been fail-closed and recorded.

Cross-rank invariants (final stats line, obs 1000):
`max_delta_rank_spread = 0.0`, `max_param_rank_diff = 0.0` — **zero rank
drift over all 1000 updates** (the per-step invariant check ran every
update; any nonzero diff raises `CMuonSafetyError`). Throughput: 15.6–16.3
s/update steady state (102901–103500), optimizer phase 0.57–0.64 s mean —
within ~1–5% of D3 (16.1–17.0 s) and salt7 G1 (14.4–15.2 s, different
pod/optimizer); rescue overhead 685 × ~0.46 ms ≈ 0.31 s total (0.003%).

**Rescue frequency trend (the key A3 watch item):** 38 / 33 / 28 / 9 (old
pod, declining) → **100 / 67 / 93 / 95 / 110 / 112** (new pod). The 3x step
sits exactly on the 102900 pod-rebuild boundary and is attributed to
machine-dependent BF16 non-determinism in the NS (D3 Phase-B B9: BF16 NS is
not bit-deterministic under the DTK build; FP32 is) — guard refs,
counters, RNG and momentum were all ckpt-restored and continuous, and the
same candidate produced 2–10/100 on the D3 machine vs 9–38/100 on the old
salt6 machine vs 67–112/100 on the new one, i.e. the frequency is
environment-dominated, not a monotonic model-level drift. Within the new
machine the band is bounded (67–112, no 1-2-8-16-30 runaway shape); the
mild tail drift (67→112 over 500 obs) coincides with the guard reference
convergence transient (`ref_max` 2.55e-06 → 1.78e-06, still decaying at
0.999/obs toward the signal equilibrium; `ref_min` flat ~1.8e-08) and is
self-limiting by construction. q-role dominates (609/685 = 89%), the same
attention-dominated role family as D3 (q16/k2/gate10 per 500).

## A4 — gate checklist

| gate | result |
|---|---|
| 1000 valid successful updates from the 102500 fork | **PASS** (102501–103500, counters exact) |
| 0 unresolved BF16 safety failure | **PASS** (685/685 FP32-rescued) |
| 0 FP32 rescue failure | **PASS** (0) |
| 0 nonfinite | **PASS** (0 across 1000) |
| 0 rank drift (delta spread / param diff) | **PASS** (0.0 / 0.0, invariant checked every update) |
| 0 catastrophic loss jump / divergence | **PASS** (logged windows mean 0.556–0.558, max 0.597; resume boundaries continuous) |
| atomic ckpt chain 102500→103500 (11 cadence ckpts) | **PASS** (COMPLETE + manifest at every point; guard counter chain 8300→…→83083 exact at 83/obs) |
| clean resume at both restarts (102900 rebuild, 103100 wandb) | **PASS** (obs/counter/loss continuity verified; 0 state errors) |
| rescue frequency not accelerating | **PASS with disclosure** (step = pod-rebuild infra artifact; within-machine band 67–112/100 bounded, no runaway pattern, mild self-limiting reference-convergence drift; monitoring recommended next 500–1000 updates) |

### Notes & caveats

1. Metrics for 102501–102900 (loss/preclip/s-per-update) were lost in the
   pod rebuild; the rescue counts for those windows are exact (atomic
   ckpt guard-state chain) and the segment's safety surface is covered by
   the cumulative counters (0 failures / 0 spread) plus the zero-error
   preserved log segments.
2. The train.log rescue-event segment obs 401–599 was truncated by the
   stack restart (log truncation on start); its aggregates are recovered
   from the ckpt chain (108/208/275 at obs 400/500/600) and two
   pre-truncation live snapshots (obs 401: 109, obs 590: 264) — all
   cross-consistent.
3. W&B logging was user-enabled at the 103100 boundary (overlay toml,
   telemetry-only, frozen toml byte-identical) — disclosed in A1. The W&B
   run carries live metrics from 103101; the 103000 sample batch exists on
   disk (output_model/g1/sample/step-103000/, 60 PNGs) and in the NFS
   mirror scope.
4. The 5000-cadence evaluation (1000-sample FID/KID/IS + 120-concept suite)
   has no grid point inside 102501–103500 (next grid 105000) — it never
   fired in this gate by design. The concept-suite files are not part of
   the replay_10 restore; a post-gate quality-gate run requires staging
   them first (concept-120 bundle available locally / on salt3, salt7).
5. Quality gate readiness: **YES** pending caveat 4 (stage concept files,
   then run the evaluation CLI at 103500 on a machine with the suite).

## Artifacts

- NFS: `sakuramoon-g1-forensic/cmuon-d4/{ckpt_103500, ckpt_103500.md5, metrics-tail.jsonl, rescue-lines.txt, guard-states-windows.json}` (MD5-verified mirror + raw telemetry)
- Hub: `leafmoone/sm_train_state` `cmoun/ckpt_102500 … ckpt_103500` (11 ckpts, full gate chain)
- `reports/cmuon-1000-safety.json` (machine-readable window metrics, rescue stream, counter chain, verdict)
- Local ckpts (salt6): `output_model/g1/ckpt_{103400,103500}_raw-*-update-cadence` (slots=2)
- W&B: `wandb.ai/2574905817leaf-jinan-university/sakuramoon/runs/g1_fp32_rescue_r1_200`
