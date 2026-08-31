# CMuon FP32-rescue — 500-update Safety Gate (D3)

Candidate: `hybrid_cmuon_canonical_ns4_fp32_rescue` (math frozen from D2 R1).
Host: salt6 (172.31.33.71, 2×HCU "BW"), pod-local DTK 2.9.0.
Window under test: **500 cumulative successful updates, 97101–97600**, forked
from `ckpt_97100_raw-97100-update-cadence` (R1) →
`ckpt_97300_raw-97300-update-cadence` (A0) → `ckpt_97400` (recovery, user-approved)
→ `ckpt_97600_raw-97600-update-cadence` (final).

## Verdict

**SAFE_500 = YES** — all A2/A3 gates passed with zero unresolved safety
events. **READY_FOR_1000_SAFETY_GATE = YES**. Stopped after 97600; no
automatic 1000 / 5000 / deployment (awaiting user decision).

## A0 — pre-gate NFS mirror

`ckpt_97300_raw-97300-update-cadence` mirrored to
`/root/private_data/sakuramoon-g1-forensic/ckpt-mirror-500/` before the gate:
14 files, `rsync -a --checksum`, **MD5-DIFF-OK**, COMPLETE marker, manifest
identity update=97300, guard block `fp32_rescue={bf16_attempts:16600,
bf16_safety_failures:12, fp32_attempts:12, fp32_rescues:12,
fp32_rescue_failures:0, rescue_by_role{gate:5,q:5,k:2}}`, trainer_state
successful_updates=97300. Training was not started before verification.
`ckpt_97400` (recovery point) and `ckpt_97600` (final) were also mirrored and
MD5-verified (14 files each).

## A1 — freeze compliance

No candidate math changed: identical toml (`train_g1_fp32_rescue_r1.toml`,
reused verbatim), identical guard calibration values (guard_ratio 0.1,
reference_decay 0.999, min_reference 3.096e-08, numerical_floor 6.575e-07,
invariant_check=true), identical lr 5e-05 / batch 800 / NS4 map / BF16
momentum. No always-FP32, no attention-only FP32, no optimizer/NS compile,
no bucketed broadcast, no async overlap, no owner remapping, no telemetry
removal, no threshold change. (Phase B is benchmark/design-only, post-gate.)

## A2 — run continuity

- R1 (D2): 97101–97300, 2 ranks, no restart, 12/12 rescues, 0 failures.
- 500-gate segment 1: resumed exactly from 97300 (pinned), ran 97301–97463.
  **Infrastructure failure at 97463** (17:16): the 50G per-pod NFS quota was
  exhausted (111G of legacy analysis artifacts predated the quota; new writes
  failed), the data-service queue-state save failed, the service failed
  closed by design, and the trainer died with `DataServiceUnavailable`.
  This is **not** a candidate safety event (no rescue failure / nonfinite /
  rank drift / atomicity violation / loss divergence).
- Recovery (user-approved): 104G of junk removed (92G full-sample tensor
  dumps of the completed D2 offline validation + 12.8G DTK core dumps),
  quota restored to 39G free, seg2 logs preserved, `ckpt_97400` (atomic
  cadence save from the crashed run, counters 12→16 verified) used as the
  resume point; segment 2 ran 97401–97600 on 2 ranks, no further restart.
  The invalid overlap 97401–97463 (seg2, different queue position, 13
  rescues) is excluded from the gate chain.
- Counter continuity (per-rank, authoritative from atomic ckpts):
  `bf16_attempts 16600→24900→33200→41500` (=83 inputs/rank/update, exact),
  `rescues 12→16→26→28`, `rescue_failures 0` throughout.

## A3 — per-100-update window stats

Loss / preclip from `metrics.jsonl` (500/500 updates, last occurrence wins);
rescue deltas from the ckpt counter chain, localized by `fp32_rescue_obs`
event lines (update ≈ 97100 + obs, ±1).

| window | updates | loss min/mean/max | preclip mean/p90/max | nonfinite | BF16 fail | FP32 rescued | rescue fails | rescues (updates) |
|---|---|---|---|---|---|---|---|---|
| 97101–97200 | 100 | 0.531 / 0.564 / 0.605 | 0.0477 / 0.0692 / 0.1262 | 0 | 2 | 2 | 0 | gate1, q1 — 97122, 97164 |
| 97201–97300 | 100 | 0.529 / 0.559 / 0.588 | 0.0431 / 0.0672 / 0.0868 | 0 | 10 | 10 | 0 | gate4, q4, k2 — 97203/04/10/12/35/37/42×2/43, 97292 |
| 97301–97400 | 100 | 0.525 / 0.560 / 0.602 | 0.0507 / 0.0738 / 0.1143 | 0 | 4 | 4 | 0 | gate3, q1 — 97325, 97334, 97337, 97391 |
| 97401–97500 | 100 | 0.500 / 0.549 / 0.594 | 0.0461 / 0.0738 / 0.1033 | 0 | 10 | 10 | 0 | q10 — 97417/21/30/31/33/36/37/38/39/40 |
| 97501–97600 | 100 | 0.539 / 0.559 / 0.584 | 0.0467 / 0.0719 / 0.0924 | 0 | 2 | 2 | 0 | gate2 — 97555, 97580 |

Cross-rank invariants (periodic stats, all windows):
`max_delta_rank_spread = 0.0`, `max_param_rank_diff = 0.0` — **zero rank
drift over all 500 updates**. `CMuonSafetyError` lines in any segment log: 0.
Optimizer phase mean 0.58–0.84 s, full update mean 16.1–17.0 s (new pod).

**Rescue frequency trend (the key A3 watch item):** 2 / 10 / 4 / 10 / 2 per
100 updates — small fluctuations, **no sustained acceleration** (the
"1,2,8,16,30" warning pattern is absent; the 10-rescue windows are separated
by 4 and 2). Total 28/500 = 0.056 rescues/update, in line with R1's
0.06/update. The q-role bursts cluster in data-region-localized windows
(the excluded seg2 overlap showed a similar 13-rescue cluster at comparable
queue positions), not a run-level trend.

## A4 — gate checklist

| gate | result |
|---|---|
| 0 unresolved BF16 safety failure | **PASS** (28/28 rescued) |
| 0 FP32 rescue failure | **PASS** (0) |
| 0 nonfinite | **PASS** (0 across 500) |
| 0 rank drift (delta spread / param diff) | **PASS** (0.0 / 0.0) |
| 0 catastrophic loss jump / divergence | **PASS** (window means 0.549–0.564, max 0.605, trending flat) |
| atomic ckpt chain 97300→97400→97500→97600 | **PASS** (COMPLETE + manifest + counters exact) |
| rescue frequency not accelerating | **PASS** (2/10/4/10/2) |

### Notes & caveats

1. The 97301–97400 metrics window comes from segment 1 (pre-crash); its
   rescue telemetry (4 events) is in the preserved seg2 log. The crash
   itself is an infra event, documented in A2, and does not touch the
   candidate safety surface.
2. Per-window BF16/FP32 `delta_rms` values are not in the telemetry
   (the candidate logs counters + event lines, not per-event deltas); the
   guard ceilings (target 1e-5, ceiling 1e-4 at lr 5e-05) were never
   exceeded post-rescue, and invariant checks held every step.
3. Throughput 16.1–17.0 s/update vs salt7 live 15.5 s/update (different pod,
   AdamW8bit): the candidate's full-update cost is within ~4–9% of the
   production baseline on this pod; the optimizer phase itself is
   ~0.58–0.84 s.

## Artifacts

- NFS: `ckpt-mirror-500/{ckpt_97300,ckpt_97400,ckpt_97600}_raw-*-update-cadence` (MD5-verified)
- `artifacts-fp32-rescue/perf-500-windows.json` (A3 metrics)
- `artifacts-fp32-rescue/perf-500-rescue-events.json` (A3 rescue stream + counter chain)
- `artifacts-fp32-rescue/seg2-97301-97463-train.log`, `seg2-data-service.log` (preserved crash segment)
- local ckpts: `output_model/g1/ckpt_{97300,97400,97500,97600}_raw-*-update-cadence`
