# SakuraMoon Camera Coordinate Causal Audit - POSTHOC REVIEW

> **Label: POSTHOC REVIEW** - this report supersedes only the VERDICT / NUMERICS interpretation of
> the historical audit. It does NOT replace any historical evidence file; the historical raw numbers
> remain exactly as committed (bit-exact cross-check below).

- generated: 2026-09-06T17:33:45Z (UTC)
- base commit: 254e098e8941a4f47111d48fd37b858167116529 (prior review head)
- bootstrap: seed 20260906, n=10000, cluster=unit, ONE shared resample matrix per aggregate

## 1. HISTORICAL STATUS

- historical verdict: **INCONCLUSIVE** (recommendation: NO_MORE_EXPOSURE_YET)
- why the historical verdict was short-circuited: stage3 sets
  `numerics_ok = determinism AND max(M_SAME_maxabs over ck) < 1e-6`; with ordinary
  SAME maxabs PRE/MID/POST = 9.163e-05 / 1.565e-04 / 8.124e-05 (all > 1e-6) the gate forced
  `verdict = INCONCLUSIVE` before the effect logic (rising + CI exclusion) could execute.
- reported numerics mismatch: audit.md prints `harness numerics OK` from the
  determinism-probe non-empty check only - a DIFFERENT quantity from the gate boolean.
  Decision logic and reported numerics state were therefore inconsistent.

## 2. NUMERICAL FLOOR

Primary baseline = CAMERA SAME (Loss(SAME) - Loss(CORRECT); SAME is a byte-exact duplicate
of CORRECT coordinates). maxabs is an OUTLIER diagnostic, not a mean-effect gate.

| ck | camera SAME mean | 95% CI | abs p50 | abs p95 | abs p99 | maxabs |
|---|---|---|---|---|---|---|
| PRE | -5.693e-08 | [-2.631e-07, 1.509e-07] | 0.000e+00 | 0.000e+00 | 2.985e-05 | 1.859e-04 |
| MID | 4.837e-09 | [-2.166e-07, 2.390e-07] | 0.000e+00 | 0.000e+00 | 3.409e-05 | 4.311e-04 |
| POST | 2.181e-08 | [-2.040e-07, 2.559e-07] | 0.000e+00 | 0.000e+00 | 3.499e-05 | 3.122e-04 |

camera SAME paired deltas (per-unit, 4-strata paired, cluster bootstrap):

- MID_PRE: 6.177e-08  [-2.489e-07, 3.922e-07]
- POST_MID: 1.697e-08  [-2.967e-07, 3.270e-07]
- POST_PRE: 7.874e-08  [-2.323e-07, 3.898e-07]

Secondary = ordinary SAME replication (512 units):

- PRE: mean -2.675e-07 [-1.051e-06, 5.346e-07], maxabs 3.186e-04
- MID: mean 2.936e-07 [-4.596e-07, 1.212e-06], maxabs 6.856e-04
- POST: mean -8.877e-08 [-8.025e-07, 6.303e-07], maxabs 2.849e-04

Microprobe (32 camera + 32 ordinary units, patterns P1-P4, strata 0/2, PRE/MID/POST):
- status: OK ; SAME == CORRECT exact (torch.equal, all 64 bundles): True
- P1 pure repeat (384 pairs): pred bit-exact rate 0.9557291666666666; max |loss delta| 1.285e-04 (rel 2.279e-04); mean delta -8.131e-07
- P2 sandwich (CORRECT, IDENTITY, CORRECT): IDENTITY-shift mean 8.727e-05; order-drift-vs-P1 max ratio 2.644
- SAME vs CORRECT order pairs (n=768): rel p99 1.968e-04, rel max 3.659e-04 (cap 5e-04); signed mean rel -8.307e-07; directional drift False
- accumulation ratio 2.644 (cap 3)
- hidden_state_detected = True (locked fail-closed rule; tripped by: P1 pure-repeat rel max 2.279e-04 > 1e-06 cap)
- data pattern: 95.6% of pure repeats bit-exact, no directional/order drift, accumulation within cap -> consistent with SPORADIC FORWARD NON-DETERMINISM on the HCU (a small fraction of repeats is non-bit-exact), NOT with KV-cache-style hidden-state accumulation. The fail-closed 1e-06 pure-repeat cap nonetheless blocks NUMERICS_CLEAN per the locked gate rules.

Historical control context (unchanged, cited from committed audit.json): ordinary SAME means
-2.68e-07/2.94e-07/-8.88e-08; strict-identity ordinary (n=146) maxabs 3.5e-05/5.8e-05/3.1e-05; RANDOM-geometry mean 1.100e-02/4.090e-03/4.024e-03.

## 3. RAW CAUSAL (historical values, recomputed bit-exact)

Sign convention (correction): **M > 0 = wrong/alternative coordinates have HIGHER loss,
i.e. the correct coordinates are PREFERRED.** (The historical README prose said the opposite;
the implementation and tests were already correct - see README erratum.)

| arm | PRE | MID | POST | n |
|---|---|---|---|---|
| M_OPPOSITE | 0.001706 | 0.002025 | 0.002099 | 2040 |
| M_IDENTITY | -0.001722 | 0.002232 | 0.002249 | 2048 |
| M_SHUFFLED | 0.001284 | 0.001916 | 0.002040 | 2048 |

correct_best3 (committed): 0.2428 -> 0.4191 -> 0.4329

## 4. NOISE-ADJUSTED (difference-of-differences: D_arm - D_same, shared bootstrap indices)

| arm | interval | raw arm delta | raw SAME delta | adjusted | 95% CI | n(common) |
|---|---|---|---|---|---|---|
| OPPOSITE | MID_PRE | 0.000320 | 6.201e-08 | 0.000319 | [0.000186, 0.000456] | 2040 |
| OPPOSITE | POST_MID | 0.000073 | 1.704e-08 | 0.000073 | [0.000014, 0.000132] | 2040 |
| OPPOSITE | POST_PRE | 0.000393 | 7.905e-08 | 0.000392 | [0.000244, 0.000544] | 2040 |
| IDENTITY | MID_PRE | 0.003954 | 6.177e-08 | 0.003952 | [0.003694, 0.004222] | 2048 |
| IDENTITY | POST_MID | 0.000017 | 1.697e-08 | 0.000017 | [-0.000039, 0.000073] | 2048 |
| IDENTITY | POST_PRE | 0.003971 | 7.874e-08 | 0.003969 | [0.003702, 0.004250] | 2048 |
| SHUFFLED | MID_PRE | 0.000631 | 6.177e-08 | 0.000629 | [0.000281, 0.000973] | 2048 |
| SHUFFLED | POST_MID | 0.000125 | 1.697e-08 | 0.000125 | [0.000068, 0.000182] | 2048 |
| SHUFFLED | POST_PRE | 0.000756 | 7.874e-08 | 0.000753 | [0.000396, 0.001115] | 2048 |

OPPOSITE common subspace excludes opp_na units (n recorded explicitly per arm above).

## 5. LOSS PREFERENCE (causal classification)

- PRIMARY arms: OPPOSITE (directional geometry), SHUFFLED (sample-specific mapping);
  SUPPORTIVE: IDENTITY (transform removed wholesale; never the sole driver).
- interval classes (CI-backed): OPPOSITE=MONOTONIC_GAIN, IDENTITY=PLATEAU, SHUFFLED=MONOTONIC_GAIN
- classification: **BLOCKED_NUMERICS**

## 6. PREDICTION DISPLACEMENT SENSITIVITY (separate from loss preference)

| arm | PRE | MID | POST | POST-PRE |
|---|---|---|---|---|
| OPPOSITE relRMS | 0.060505 | 0.042835 | 0.043014 | -0.017491 |
| IDENTITY relRMS | 0.068053 | 0.055694 | 0.055724 | -0.012329 |
| SHUFFLED relRMS | 0.063050 | 0.044200 | 0.044546 | -0.018504 |

- classification: **LOSS_ALIGNMENT_GAIN_WITH_REDUCED_PREDICTION_DISPLACEMENT**
- loss preference and raw displacement are reported as SEPARATE metrics; the
  historical single-sentence coordinate-sensitivity framing is withdrawn.

## 7. OFFSET (universal tertiles + orientation-specific, OPPOSITE arm)

Universal naming: START (<1/3) / CENTER [1/3,2/3) / END (>=2/3) of the NORMALIZED
offset. Historical overall L/C/R were exactly these tertiles and do not by themselves
mean physical left/center/right; the historical right-edge finding is re-stated as an
**END-offset regression candidate** and decomposed below.

Physical-axis convention VERIFIED from production source (base commit b2443af):
horizontal normalized_offset = left/available (0 = LEFT, 1 = RIGHT); vertical =
top/available (0 = TOP, 1 = BOTTOM). Both physical and normalized labels are reported.

| cohort | n | raw PRE | raw MID | raw POST | adjusted POST-PRE | 95% CI | physical (H/V) |
|---|---|---|---|---|---|---|---|
| universal START | 692 | 0.001111 | 0.002418 | 0.002670 | 0.001560 | [0.001277, 0.001860] | H:LEFT / V:TOP |
| universal CENTER | 670 | 0.000260 | 0.000305 | 0.000351 | 0.000090 | [-0.000047, 0.000221] | H:CENTER / V:CENTER |
| universal END | 686 | 0.003701 | 0.003290 | 0.003208 | -0.000493 | [-0.000779, -0.000213] | H:RIGHT / V:BOTTOM |
| horizontal START | 171 | 0.002178 | 0.002822 | 0.002844 | 0.000663 | [0.000214, 0.001124] | - |
| horizontal CENTER | 146 | 0.000511 | 0.000509 | 0.000520 | 0.000010 | [-0.000162, 0.000192] | - |
| horizontal END | 174 | 0.002297 | 0.001893 | 0.002221 | -0.000076 | [-0.000499, 0.000321] | - |
| vertical START | 521 | 0.000761 | 0.002285 | 0.002614 | 0.001852 | [0.001519, 0.002215] | - |
| vertical CENTER | 524 | 0.000190 | 0.000248 | 0.000304 | 0.000113 | [-0.000054, 0.000270] | - |
| vertical END | 512 | 0.004178 | 0.003765 | 0.003544 | -0.000634 | [-0.000982, -0.000293] | - |

- END-offset systematic negative (adjusted CI upper < 0): **True**

## 8. BEHAVIOR (no recomputation)

- expanded behavioral effectiveness verdicts remain NULL_EFFECT / WEAK as committed;
  not recomputed in this pass; behavioral effectiveness is excluded from the causal
  classification by design.

## 9. NUMERICS GATE

- NUMERICS_CLEAN = **False** (same boolean drives both the verdict and this report)
  - R1: PASS
  - R2: PASS
  - R3: PASS
  - R4: PASS
  - R5: FAIL
  - reason: R5: microprobe absent or flagged hidden-state/order defect
  - same/causal scale ratio (|SAME POST-PRE| / min |primary adjusted POST-PRE|): 0.00020092189071057868

## 10. POSTHOC VERDICT & RECOMMENDATION

- POSTHOC_VERDICT = **BLOCKED_NUMERICS**
- RECOMMENDATION = **FIX_AUDIT_HARNESS_BEFORE_TRAINING** (recommendation only; NO training authorized)
- LONGER_P25_AUTHORIZED=NO, P50_AUTHORIZED=NO, PRODUCTION_CAMERA=OFF

## 11. IMMUTABILITY & PROVENANCE

- final_snapshot 5/5 bit-identical to committed manifest: **YES**
- historical causal + expanded reports sha-checked: **ALL UNCHANGED**
- stage1 manifest sha: YES (matches committed manifest)
- RAW cross-check vs committed audit.json: EXACT (all arms x checkpoints, S_ relRMS)

## 12. METHOD & SEMANTICS (locked in posthoc_contracts.py + tests)

- M_arm = Loss(arm) - Loss(CORRECT); M > 0 = correct coordinates preferred.
- N_same = Loss(SAME) - Loss(CORRECT); numerical floor, mean/CI-based, NOT maxabs-gated.
- Common-finite subspace per arm (arm + SAME finite at all 3 checkpoints; OPPOSITE also
  excludes opp_na); n explicit. ONE shared resample matrix per aggregate applies the
  same replicate indices to PRE/MID/POST/SAME/ARM; D_adj computed per replicate.
- Interval classes are CI-backed (EARLY_GAIN/LATE_GAIN/MONOTONIC_GAIN/PLATEAU/FLAT/
  REVERSED/NON_MONOTONIC), never point-sign only.
- Historical INCONCLUSIVE is retained as history; only its interpretation is superseded.

