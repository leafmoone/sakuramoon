# CMuon Structural/SNR Pre-NS Classifier Calibration — D1 Round (FAIL)

Date: 2026-08-31 · Host: salt6 (2×HCU, DTK 26.04) · Branch: cmoun-guarded @ f397ca8
Goal: goal-a3a53bfb · Frozen spec: 12-section D1 protocol (08-31)

## 1. Objective (frozen, §1)

Before any "Guarded Canonical CMuon v2" (pre-NS dangerous-input classifier), prove that
dangerous NS inputs are **reliably identifiable without mass false-skips**:
100 real fwd/bwd observations from healthy ckpt_97100, **zero parameter updates**,
all 166 semantic CMuon chunks per observation, real safety labels from production-identical
bf16 NS4 (K=5 runs per input per rank, no clamp, no model write), candidate rule
families A–E, holdout-validated thresholds. PASS requires holdout FN=0 **and**
safe false-skip ≪ 64–70% (v1's failure level) and skip ≤ 50% overall.

## 2. Shadow run — clean (§1)

| item | value |
|---|---|
| source checkpoint | ckpt_97100_raw-97100-update-cadence (pristine, verified after run) |
| observations | 100 (updates 97101–97200), 2 ranks |
| parameter updates | **0** (momentum EMA only; weights bit-identical to ckpt) |
| inputs labeled | 166 chunks × 100 obs × 2 ranks × 5 NS4 runs = **166,000 NS4 runs** |
| cross-rank consistency | max nesterov_rms spread = **0.0** on all observations |
| stop | clean `StructuralCalibrationComplete` at obs 100, no failure bundle |
| side effects | no checkpoints published, no W&B (noop), S1 mainset untouched (647); structural mainset 647→657 (31 shards, must be reset) |

Data: pre-staged queue 101–300 shards (463.6 GiB cache, 200/200 byte-exact, mtime
queue-ordered; 136 shards relayed salt3→salt6 in 22 min at ~60 MB/s after the P3-chain
`[data.cache]` watermark bug LRU-evicted them on the first pilot start — toml now pins
low=192/high=500 like the S1 toml).

## 3. PI validation (§4) — done before trusting features

Power iteration with rank-1 deflation vs **exact torch SVD on 200 real saved tensors**
(CPU fp64 reference; HCU SVD was 3 s/input → ~14 h/100 obs, rejected):

| PI iters | σ1 rel err (median / max) | top1-energy abs err (median / max) |
|---|---|---|
| 5 | 2.0e-7 / **1.6e-1** | 1.7e-7 / **8.8e-2** |
| 10 | 1.1e-7 / 1.4e-1 | 6.0e-8 / 7.6e-2 |
| **20 (used)** | **9.5e-8 / 7.7e-4** | **4.2e-8 / 4.5e-4** |
| 30 | 9.1e-8 / 2.6e-5 | 3.4e-8 / 1.2e-6 |
| 50 | 8.5e-8 / 7.4e-7 | 3.3e-8 / 8.3e-7 |

PI@20 kept (production feat_pi = 2.62 s/obs HCU). PI@5/10 have unacceptable max errors
on slow-convergence (small spectral-gap) tensors. Audit: `pi-audit.log`, 200 tensors.

## 4. What the labels found (§3 labels, §7 grouping)

**108 dangerous input-observations out of 16,600 (0.65%)** — an input-observation is
DANGEROUS if any of its K=5 production-identical NS4 runs is catastrophic
(nonfinite or delta_rms > 10×0.2×lr).

- **By slot**: slot_21: 33 · slot_07: 29 · slot_02: 20 · slot_08: 16 · slot_01: 10 (all other 27 slots: 0)
- **By role**: attention_k 34 · content_gate 32 · q 31 · v 10 · out 1 · **ffn/adaLN: 0**
- **Temporal**: 77/100 obs contain ≥1 dangerous input; obs 3..100; **first 50 obs: 50 events, last 50 obs: 58 events — danger persists and accelerates**. The "momentum-from-zero early phase only" hypothesis (S1 crashed within obs 1–15) is **falsified** at this sensitivity.
- **§7 question answered**: the q_proj 88.9%-top-1 pathology is a **general cluster, not an outlier** — danger spans 5 slots and 4 attention roles; FFN and AdaLN never go dangerous.
- Known S1 forensics reproduced inside the shadow: obs 13 (slot_07 q_proj, S1 crash site) and obs 15-class events reappear; the S1-b crash tensor (obs 3, slot_07 k_proj) is the **lowest-top-1 dangerous input found (0.2583)** — the class boundary.

## 5. Rule families — holdout-validated (§5–§8)

Protocol: thresholds grid-swept on FIT rows only (slot holdout: fit on all slots except
07/08, evaluate on 07/08 — the known crash slots; time holdout: obs-half splits),
lexicographic objective (holdout FN=0 first, then minimal false skip). Never selected
and evaluated on the same events.

| family | rule | slot-holdout (07/08) | time-holdout skip | verdict |
|---|---|---|---|---|
| A amplitude | rel_sig < τ | FN=2, skip 99.9% | — | INFEASIBLE (v1 failure mode, confirmed on 100 obs) |
| B top1 | top1 > θ | **FN=0**, θ=0.2558, **false-skip 71.1%, skip 71.9%** | 68.1% / 86.2% | FN=0 but skip ≫ 50% |
| C stable-rank | stable_rank < ρ | FN=0, ρ=3.909, skip 71.9% | 68.1% / 86.2% | identical to B (rank≈top1 dominance) |
| D amp+top1 | A ∧ B | FN=**2**, skip 71.8% | — | amplitude AND-condition **loses** holdout danger |
| E amp+top1+temporal | D ∧ cos < γ | INFEASIBLE (no 3-threshold box hits all fit danger) | — | — |
| baseline: v1 amplitude floor | sig < 1.2e-7 | skip 70.3%, catches only 76/108 (32 FN) | — | v1 FAIL re-confirmed at larger scale |

**The core finding (feature separation)**: dangerous inputs have top1-energy 0.258–0.995,
but safe inputs span 0.01–**0.998** — 13,245 of 16,492 safe rows (80.3%) sit above the
best FN=0 threshold 0.2558. The dangerous class is *inside* the safe distribution.
High top1 energy is necessary but nowhere near sufficient (88.9%-top1 q_proj chunks that
never went catastrophic in 50 NS4 runs exist in every slot, e.g. slot_00 out_proj
obs 1 top1=0.930, 0/10 catastrophic). No deterministic threshold rule in the frozen
families separates the classes at acceptable skip rate.

## 6. Counterexamples (§12 record, on NFS)

`/root/private_data/sakuramoon-g1-forensic/artifacts-fail-record/`
(analysis JSON + both raw rank JSONLs + `counterexamples.json` + tensor dumps):

- **Dangerous, lowest top1 (class boundary)**: obs 3, `dit.blocks.slot_07.attention.k_proj.weight` c0 — top1=0.2583, rel_sig=0.432, hazard 5/5. This is the S1-b crash tensor; only 0.0025 above the best threshold. Tensor: `tensors/dangerous-min-top1-obs03-slot07-k_proj-c0.pt` (fp32, 6.3 MB).
- **Safe, first false-skips under the best FN=0 rule**: obs 1 slot_00 out_proj (top1=0.9304, rel_sig=0.1709, 0/10 catastrophic) and slot_00 v_proj (top1=0.9254). Tensor: `tensors/false-skip-obs01-slot00-out_proj-c0.pt` (26 MB).
- Total safe rows the best FN=0 rule must skip: **13,245** (80.3% of all safe rows).

## 7. Cost (§9)

HCU, median over 200 rank-observations:

| component | ms/obs |
|---|---|
| NS4 K=5 safety labels | 36,349 (74%) |
| power-iteration features (166 chunks @20 iters) | 2,622 |
| temporal cosines | 351 |
| amplitude reductions | 114 |
| **shadow step total** | **40,693** (vs ~16,000 ms production step → 2.54×) |

Hypothetical v2 classifier (no label runs): 3.09 s/update ≈ **17% overhead** — reported
for the record, moot under FAIL.

## 8. S1-c (§11)

**NOT RUN** — the spec authorizes S1-c only after a calibration PASS.

## 9. Verdict (§6/§10/§12)

**FAIL — structural/SNR pre-NS classifier, frozen rule families A–E.**

- The dangerous class is *identifiable in aggregate* (top1 > 0.2558 catches 108/108,
  holdout FN=0) but **not separable**: every FN=0 rule false-skips **68–86%** of all
  inputs (> 50% red line; not ≪ the 64–70% v1 level; §6/§10 both fail).
- Per §12: first counterexample full pre-NS features + tensor artifacts saved; **stop**.
- **v2 guard NOT implemented. S1-c NOT run. Nothing deployed.** Live G1 (salt3), S1
  mainset (647), and ckpt_97100 are untouched.

## 10. Open questions (for user)

1. **Danger is time-persistent and accelerating** (37/25 obs in the final window) — a
   time-limited guard ("skip only the first N updates") is not a viable alternative.
2. **Per-input dynamic baselines** (skip when an input's top1 deviates from its own
   rolling history) were outside the frozen A–E candidate set; static-threshold overlap
   does not prove they fail, but they were not authorized in this round.
3. **Intrinsic chaos ceiling**: if the catastrophic NS4 branch is selected by sub-ulp
   bf16 nondeterminism, no pre-NS feature can perfectly predict it. Higher-precision NS
   and NS-depth changes were hard-constrained out of this round (禁 FP32 fix / NS depth
   change) and remain the only unexplored structural levers.
4. Reset housekeeping: structural mainset 657→647; full-sample dumps (30 G) remain on
   volatile local disk if the dynamic-baseline option is chosen.

---

```
=== COPY TO CHATGPT: CMUON STRUCTURAL GUARD ===
[shadow] 100 real fwd/bwd obs from healthy ckpt_97100, 2 ranks, ZERO parameter updates (momentum only), all 166 CMuon chunks per obs, labels = K=5 production-identical bf16 NS4 runs per input per rank (166,000 runs), no clamp, no model write, cross-rank nesterov spread 0.0 all obs, clean StructuralCalibrationComplete stop, ckpt/S1-mainset untouched.
[known_pathology] near-zero nesterov -> Frobenius-normalize x2.0766e5 -> NS4 convergence-boundary bf16 bit chaos (~20% catastrophic branch) -> x1.675e5 total. S1 crashes obs 13/2/7/3 (slot_07 q_proj), S1-b obs 15 (slot_07 k_proj). Shadow found 108/16600 dangerous input-obs (0.65%): by slot 21:33 07:29 02:20 08:16 01:10; by role k:34 gate:32 q:31 v:10 out:1, FFN/AdaLN: 0; 77/100 obs hit, obs 3..100, first-50: 50 events vs last-50: 58 (persists + accelerates; early-phase hypothesis FALSIFIED). q_proj 88.9%-top1 case = general 5-slot/4-role cluster, not an outlier.
[feature_separation] PI@20 validated vs exact SVD on 200 real tensors (sigma1 rel err med 9.5e-8 / max 7.7e-4; top1-energy err med 4.2e-8 / max 4.5e-4; PI@5/10 max err 8.8%/7.6% = rejected). Dangerous top1 range 0.258-0.995 sits INSIDE safe range 0.01-0.998: 13245/16492 safe rows (80.3%) above best threshold 0.2558. Min dangerous top1 = 0.2583 (obs3 slot_07 k_proj, the S1-b crash tensor, hazard 5/5). High top1 is necessary, not sufficient.
[classifier] slot holdout (fit all except 07/08, eval 07/08) + time holdout, thresholds fit-only: A amplitude: FN=2 skip 99.9% (infeasible, v1 mode re-confirmed). B top1>0.2558: FN=0, false-skip 71.1%, skip 71.9% (time holdout 68.1%/86.2%). C stable_rank<3.909: identical to B. D amp+top1: FN=2 (AND-condition loses danger). E amp+top1+cos: infeasible. v1 amplitude-floor baseline: 70.3% skip catching only 76/108 (32 FN).
[counterexamples] FAIL record on NFS /root/private_data/sakuramoon-g1-forensic/artifacts-fail-record/: dangerous-lowest-top1 = obs3 slot_07 k_proj c0 (top1 0.2583, rel_sig 0.432, hazard 5/5, tensor saved); first safe false-skips = obs1 slot_00 out_proj (top1 0.9304, 0/10 catastrophic, tensor saved) + slot_00 v_proj (0.9254); 13245 safe rows must be skipped by any FN=0 rule.
[cost] HCU median/obs: NS labels K=5 36.3s (74%), PI features 2.62s, cosines 0.35s, reductions 0.11s, step total 40.7s vs 16s production (2.54x). Hypothetical v2 classifier (no labels): 3.1s/update ~17%.
[S1c] NOT RUN (spec: only after PASS).
[verdict] FAIL — dangerous class identifiable (top1>0.2558: 108/108, holdout FN=0) but NOT separable: every FN=0 rule false-skips 68-86% of inputs (>50% red line, not <<64-70%). Frozen families A-E exhausted. v2 NOT implemented, S1-c NOT run, NOTHING deployed. Live G1 / S1 mainset 647 / ckpt_97100 untouched.
[open_questions] (1) danger time-persistent + accelerating => time-limited guard not viable. (2) per-input dynamic baselines (top1 vs own rolling history) outside frozen A-E; not tested, not authorized. (3) if catastrophic branch is sub-ulp bf16 chaos, no pre-NS feature can fully predict it; fp32-NS / NS-depth changes were hard-constrained out and are the only unexplored levers. (4) housekeeping: structural mainset 657->647; 30G full-sample obs1-5 on volatile local disk if option (2) is chosen.
=== END ===
```
