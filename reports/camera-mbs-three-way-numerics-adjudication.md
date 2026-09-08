# MBS Three-Way Numerics Adjudication (replicate-robustness, numerics-v2)

Pre-registered protocol frozen at H1 (see `mbs_three_way_numerics/README.md`). NO per-pair maxabs hard gate; per-pair tail statistics are diagnostics only.

**Question.** Does HCU/DCU forward numerical nondeterminism change the intervention-associated estimator classification under a same-source frozen evaluation cohort?

**NUMERICAL_ADJUDICATION: PASS** — NUMERICALLY_STABLE, common classification: CLEAR_HARM, conservative envelope: CLEAR_HARM.

## 1. Replicate tuples (TARGETED latent>=2, n=312)

| tuple | I_BOTTOM point [CI] | I_G point [CI] | class |
|---|---|---|---|
| P0 C0 T0 | -0.000439 [-0.000618, -0.000270] | 0.000721 [0.000483, 0.000984] | CLEAR_HARM |
| P0 C0 T1 | -0.000438 [-0.000617, -0.000269] | 0.000720 [0.000482, 0.000983] | CLEAR_HARM |
| P0 C1 T0 | -0.000439 [-0.000618, -0.000270] | 0.000721 [0.000483, 0.000984] | CLEAR_HARM |
| P0 C1 T1 | -0.000438 [-0.000617, -0.000269] | 0.000720 [0.000482, 0.000983] | CLEAR_HARM |
| P1 C0 T0 | -0.000439 [-0.000618, -0.000270] | 0.000721 [0.000483, 0.000984] | CLEAR_HARM |
| P1 C0 T1 | -0.000438 [-0.000617, -0.000269] | 0.000720 [0.000482, 0.000983] | CLEAR_HARM |
| P1 C1 T0 | -0.000439 [-0.000618, -0.000270] | 0.000721 [0.000483, 0.000984] | CLEAR_HARM |
| P1 C1 T1 | -0.000438 [-0.000617, -0.000269] | 0.000720 [0.000482, 0.000983] | CLEAR_HARM |

## 2. Conservative envelope (4 unique C/T combinations, TARGETED)

- **i_top**: point in [0.000282, 0.000283], envelope CI [0.000124, 0.000472]
- **i_bot**: point in [-0.000439, -0.000438], envelope CI [-0.000618, -0.000269]
- **i_g**: point in [0.000720, 0.000721], envelope CI [0.000482, 0.000984]

## 3. PRE cancellation (diagnostic proof, protocol s24)

tolerance 1e-12; max |I(P_A)-I(P_B)| per combination: C0/T0: i_top=0.00e+00, i_bot=0.00e+00, i_g=0.00e+00; C0/T1: i_top=0.00e+00, i_bot=0.00e+00, i_g=0.00e+00; C1/T0: i_top=0.00e+00, i_bot=0.00e+00, i_g=0.00e+00; C1/T1: i_top=0.00e+00, i_bot=0.00e+00, i_g=0.00e+00. **PASS**

## 4. Replicate A vs B per-pair diagnostics (DIAGNOSTICS ONLY)

- PRE/m_top: bit-exact 467/512 (91.2%), maxabs 1.083e-04, rms 9.187e-06, >1e-6: 43, >1e-5: 28, >1e-4: 1
- PRE/m_bot: bit-exact 464/512 (90.6%), maxabs 5.822e-05, rms 8.555e-06, >1e-6: 47, >1e-5: 33, >1e-4: 0
- PRE/f_top: bit-exact 472/512 (92.2%), maxabs 9.029e-05, rms 8.110e-06, >1e-6: 40, >1e-5: 26, >1e-4: 0
- PRE/f_bot: bit-exact 468/512 (91.4%), maxabs 6.827e-05, rms 7.525e-06, >1e-6: 39, >1e-5: 29, >1e-4: 0
- CONTROL/m_top: bit-exact 467/512 (91.2%), maxabs 7.763e-05, rms 7.272e-06, >1e-6: 45, >1e-5: 27, >1e-4: 0
- CONTROL/m_bot: bit-exact 462/512 (90.2%), maxabs 1.608e-04, rms 1.232e-05, >1e-6: 47, >1e-5: 29, >1e-4: 2
- CONTROL/f_top: bit-exact 468/512 (91.4%), maxabs 1.040e-04, rms 8.538e-06, >1e-6: 44, >1e-5: 25, >1e-4: 1
- CONTROL/f_bot: bit-exact 466/512 (91.0%), maxabs 1.675e-04, rms 9.798e-06, >1e-6: 43, >1e-5: 25, >1e-4: 1
- TREATMENT/m_top: bit-exact 464/512 (90.6%), maxabs 1.035e-04, rms 1.055e-05, >1e-6: 46, >1e-5: 32, >1e-4: 1
- TREATMENT/m_bot: bit-exact 460/512 (89.8%), maxabs 2.404e-04, rms 1.481e-05, >1e-6: 51, >1e-5: 31, >1e-4: 1
- TREATMENT/f_top: bit-exact 466/512 (91.0%), maxabs 1.254e-04, rms 9.259e-06, >1e-6: 44, >1e-5: 34, >1e-4: 1
- TREATMENT/f_bot: bit-exact 468/512 (91.4%), maxabs 3.648e-05, rms 5.978e-06, >1e-6: 41, >1e-5: 26, >1e-4: 0

## 5. Estimator numerical spread (diagnostic only)

- i_top: point spread 3.572e-07 vs median CI width 3.476e-04 (ratio 0.001)
- i_bot: point spread 1.306e-06 vs median CI width 3.481e-04 (ratio 0.004)
- i_g: point spread 1.327e-06 vs median CI width 5.006e-04 (ratio 0.003)

## 6. Other cohorts (points + CIs, all tuples)

- all (n=512), first tuple: i_top: 0.000206, i_bot: -0.000310, i_g: 0.000516
- latent_lt2 (n=200), first tuple: i_top: 0.000086, i_bot: -0.000109, i_g: 0.000195
- latent_2to4 (n=275), first tuple: i_top: 0.000200, i_bot: -0.000360, i_g: 0.000560
- latent_ge4 (n=37), first tuple: i_top: 0.000894, i_bot: -0.001024, i_g: 0.001919
- content_balanced (n=256), first tuple: i_top: 0.000230, i_bot: -0.000333, i_g: 0.000563

## 7. Statement

The numerical nondeterminism is **NUMERICALLY_STABLE** across all 8 replicate tuples: the robust classification is **CLEAR_HARM**, and it agrees with the conservative replicate envelope. This is a **replicate-robust intervention-associated contrast under a same-source frozen evaluation cohort** — NOT an exact causal effect. Root cause wording: localized HCU/DCU forward numerical nondeterminism; root operator/kernel family not isolated.

## 8. Checkpoint identity (re-hashed before/after TREATMENT_B scoring)

status: **PASS** — all checkpoints verified before and after scoring; manifest and model-tree hashes unchanged.

- pre_scoring PRE: manifest `821a8c12546a0948...`, tree `32b76aff83a92b26...`, updates 118100
- pre_scoring CONTROL: manifest `f014827fb441aca8...`, tree `5347def81c6c8627...`, updates 118200
- pre_scoring TREATMENT: manifest `11deada823da8217...`, tree `22a802f22987a679...`, updates 118200
- post_scoring PRE: manifest `821a8c12546a0948...`, tree `32b76aff83a92b26...`, updates 118100
- post_scoring CONTROL: manifest `f014827fb441aca8...`, tree `5347def81c6c8627...`, updates 118200
- post_scoring TREATMENT: manifest `11deada823da8217...`, tree `22a802f22987a679...`, updates 118200

## 9. All-cohort detail

See `camera-mbs-three-way-numerics-metrics.json` (`per_tuple`) for every cohort x tuple x metric point + CI, and `camera-mbs-three-way-numerics-pairs.csv` (self-replay verified) for the per-pair replicate margins and C/T combination contrasts.
