# SakuraMoon Camera Coordinate Causal Audit - POSTHOC V2 REVIEW

> **Label: POSTHOC V2 REVIEW** - supersedes ONLY the V1 posthoc NUMERICS CLASSIFICATION and the
> downstream recommendation. V1 reports, V1 tooling, and all historical evidence remain immutable.

- generated: 2026-09-07T02:53:46Z
- base commit: 0caf0a156f45eefc0986829005e49e59a7301082 (V1 posthoc review head)
- bootstrap: seed 20260907, n=10000, cluster=unit, ONE shared resample matrix
- point estimates: EXACT observed sample means; bootstrap for CIs only (section 21)

## 1. V1 FALSE-POSITIVE CORRECTION

- V1 rule: `p1_rel_max > 1e-6 => hidden_state_detected=True`. A single worst repeat outlier
  was treated as a hidden mutable-state defect. External review: false-positive design risk.
- V2 rule: `HIDDEN_MUTABLE_STATE_DETECTED = B OR C OR D OR E` (coordinate map mutation,
  parameter/buffer mutation, order-dependent drift beyond the P0 floor, CI-backed accumulating
  drift beyond the P0 floor). `REPEAT_NUMERIC_JITTER_PRESENT` (A) is a DIAGNOSTIC ONLY and can
  never by itself set hidden state or NUMERICS_CLEAN=False.

## 2. NUMERICAL FLOOR (primary basis unchanged)

camera SAME (primary):

| ck | mean | 95% CI | abs p95 | abs p99 | maxabs |
|---|---|---|---|---|---|
| PRE | -5.693e-08 | [-2.651e-07, 1.502e-07] | 0.000e+00 | 2.985e-05 | 1.859e-04 |
| MID | 4.837e-09 | [-2.222e-07, 2.361e-07] | 0.000e+00 | 3.409e-05 | 4.311e-04 |
| POST | 2.181e-08 | [-2.059e-07, 2.605e-07] | 0.000e+00 | 3.499e-05 | 3.122e-04 |

camera SAME MID_PRE: 6.177e-08 [-2.518e-07, 3.888e-07]
camera SAME POST_MID: 1.697e-08 [-2.946e-07, 3.338e-07]
camera SAME POST_PRE: 7.874e-08 [-2.281e-07, 3.955e-07]

ordinary SAME (secondary):
- PRE: mean -2.675e-07 [-1.071e-06, 5.402e-07], maxabs 3.186e-04
- MID: mean 2.936e-07 [-4.716e-07, 1.201e-06], maxabs 6.856e-04
- POST: mean -8.877e-08 [-7.945e-07, 6.549e-07], maxabs 2.849e-04

scale comparison (section 20):
- |camera SAME POST-PRE mean| / min(|primary adjusted POST-PRE|) = **2.006e-04**
- SAME POST-PRE CI width = 6.236e-07 (0.00x the smallest primary effect)

## 3. MICROPROBE V2 (state vs jitter)

| concept | value |
|---|---|
| REPEAT_NUMERIC_JITTER_PRESENT | True |
| COORDINATE_MAP_MUTATION_DETECTED | False |
| PERSISTENT_BUFFER_MUTATION_DETECTED | False |
| ORDER_DEPENDENT_DRIFT_DETECTED | False |
| ACCUMULATING_DRIFT_DETECTED | False |
| HIDDEN_MUTABLE_STATE_DETECTED | False |

parameter mutation: False; persistent buffer mutation: False
P0 immediate repeats: bitexact rate 0.9363839285714286/2688; signed mean -7.956266580593018e-07; max rel jitter 0.0005562964412705204 (diagnostic cap 1e-6: exceeded - WARNING only)
P1 interleave C drift: signed mean 2.893050097756916e-07 CI [-8.898211591359642e-07, 1.538271317258477e-06]; slope 4.0275820841391853e-07 CI [-8.583000938718524e-07, 1.7560193858419843e-06]; beyond P0 floor: False
P2 SAME interleave: C signed mean 6.783908853928248e-07 CI [-4.3732127071254783e-07, 1.8369613422287835e-06]; SAME-vs-C p99 ratio vs P0 1.2704505252795555
P3 reload consistency: state_history_effect = False
RNG consumption: True

interpretation: No hidden mutable-state evidence (B/C/D/E all False). Repeat numeric jitter is present (A) and is reported as a diagnostic runtime floor ONLY; per V2 semantics it does not block numerics.

## 4. CAUSAL V2 (exact point estimates, paired shared bootstrap)

| arm | interval | raw arm | raw same | adjusted |
|---|---|---|---|---|
| OPPOSITE | MID_PRE | 3.195e-04 | 6.201e-08 | 3.195e-04 [1.801e-04, 4.570e-04] |
| OPPOSITE | POST_MID | 7.305e-05 | 1.704e-08 | 7.303e-05 [1.502e-05, 1.310e-04] |
| OPPOSITE | POST_PRE | 3.926e-04 | 7.905e-08 | 3.925e-04 [2.410e-04, 5.408e-04] |
| IDENTITY | MID_PRE | 3.954e-03 | 6.177e-08 | 3.954e-03 [3.691e-03, 4.223e-03] |
| IDENTITY | POST_MID | 1.711e-05 | 1.697e-08 | 1.710e-05 [-3.931e-05, 7.330e-05] |
| IDENTITY | POST_PRE | 3.971e-03 | 7.874e-08 | 3.971e-03 [3.695e-03, 4.245e-03] |
| SHUFFLED | MID_PRE | 6.311e-04 | 6.177e-08 | 6.311e-04 [2.904e-04, 9.819e-04] |
| SHUFFLED | POST_MID | 1.247e-04 | 1.697e-08 | 1.246e-04 [6.924e-05, 1.812e-04] |
| SHUFFLED | POST_PRE | 7.558e-04 | 7.874e-08 | 7.557e-04 [4.043e-04, 1.112e-03] |

intervals: OPPOSITE=MONOTONIC_GAIN  IDENTITY=PLATEAU  SHUFFLED=MONOTONIC_GAIN

## 5. PREDICTION DISPLACEMENT (separate metric)

OPPOSITE: PRE 0.0605 / MID 0.0428 / POST 0.0430 (POST-PRE -0.0175)
IDENTITY: PRE 0.0681 / MID 0.0557 / POST 0.0557 (POST-PRE -0.0123)
SHUFFLED: PRE 0.0631 / MID 0.0442 / POST 0.0445 (POST-PRE -0.0185)

classification: **LOSS_ALIGNMENT_GAIN_WITH_REDUCED_PREDICTION_DISPLACEMENT**

## 6. OFFSET CELLS (OPPOSITE adjusted POST-PRE)

| scope | tertile | n | n_common | adjusted | CI |
|---|---|---|---|---|---|
| universal | START | 692 | 692 | 1.559e-03 | [1.276e-03, 1.863e-03] |
| universal | CENTER | 670 | 662 | 9.052e-05 | [-4.522e-05, 2.243e-04] |
| universal | END | 686 | 686 | -4.928e-04 | [-7.754e-04, -2.135e-04] |
| horizontal | START | 171 | 171 | 6.659e-04 | [2.244e-04, 1.125e-03] |
| horizontal | CENTER | 146 | 144 | 8.475e-06 | [-1.632e-04, 1.897e-04] |
| horizontal | END | 174 | 174 | -7.538e-05 | [-5.021e-04, 3.322e-04] |
| vertical | START | 521 | 521 | 1.852e-03 | [1.508e-03, 2.225e-03] |
| vertical | CENTER | 524 | 518 | 1.133e-04 | [-4.897e-05, 2.724e-04] |
| vertical | END | 512 | 512 | -6.346e-04 | [-9.713e-04, -2.894e-04] |

universal END systematic negative: **True**

## 7. OFFSET BALANCE AUDIT (see offset-balance report)

balance classification: **EFFECT_PERSISTS_AFTER_GEOMETRY_BALANCE**
- counts and geometry balanced; BOTTOM negative persists after balance -> directional/content interaction, NOT 'sampling imbalance'

- exposure count imbalance (planner z >= 3): False
- geometry imbalanced (|SMD| >= 0.1): False (largest: main_tokens)
- reweighted TOP: 1.825e-03 [0.0014895396485939993, 0.0021860739800730494]; BOTTOM: -6.443e-04 [-0.0009951498707519459, -0.00028926600606747605]
- bottom negative after reweight: **True**; geometry explains bottom: **False**
- Spearman(norm_offset, D_opp) vertical: -0.3129336662056765; shard imbalance: False

## 8. NUMERICS V2 GATE

NUMERICS_V2_CLEAN = **True**
- R1: PASS
- R2: PASS
- R3: PASS
- R4: PASS
- R5: PASS
- R6: PASS
- R7: PASS
- R8: PASS
- R9: PASS

repeat jitter warning: True (diagnostic only - never a fail condition)

## 9. VERDICT + RECOMMENDATION

POSTHOC_V2_VERDICT = **POSITIVE_LOSS_PREFERENCE_LEARNING**
RECOMMENDATION = **REVIEW_VERTICAL_END_SUPERVISION_FIRST** (case B)

> RECOMMENDATION is not AUTHORIZATION. No training of any kind is authorized.

## 10. IMMUTABILITY

- final_scripts: 5/5 unchanged
- causal_reports: 5/5 unchanged
- expanded_effectiveness_reports: 6/6 unchanged
- v1_posthoc_reports: 4/4 unchanged
- v1_tooling: 3/3 unchanged
