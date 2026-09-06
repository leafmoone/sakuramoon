# SakuraMoon Camera Viewport v2 - Coordinate Causal Audit

**Fixed question.** Did P25 (2000U, U116100->U118100) training increase the model's *causal dependence on image absolute coordinates*? Design: freeze the complete training input (x0 from production preprocessing + Mage-VAE, text conditioning, timestep strata, per-unit noise and x_t); change ONLY the image coordinate tensor; compare PRE U116100 / MID U117100 / POST U118100 on the production forward and the production JLT x-pred loss.

- Code: `b2443af436b268fafb2cb6c05d724e3ab0d6042c` (branch camera-v2-c2), entrance gate `b2443af436b268fafb2cb6c05d724e3ab0d6042c`, `git diff b2443af..HEAD -- src tests config` = empty.
- Cohort: **2048 CAMERA_APPLIED** (natural, via real admission + Camera v2 planner, p=0.25) + **512 ORDINARY** controls, from the fixed s0-validation-50k-v1 shards (selection seed 44, 7 shards; held out from training).
- Timesteps: 4 deterministic JLT quantiles (10/35/65/90%): 0.138806, 0.248196, 0.379483, 0.556073. Noise: per-(unit, stratum) seeded draws (master seed 20260906), frozen across arms and checkpoints.
- Arms: CORRECT (production) / IDENTITY / OPPOSITE (sign-flipped shift, zoom kept; 8 units N/A with exact-zero shift) / SHUFFLED (derangement, seed 20260906, no self-assignment) + interpretive HALF (0.5x) / OVER (1.5x) interpolation.
- Power gate: **FORMAL** (2048 usable CAMERA_APPLIED units (target 2048, floor 1024).)

## 1. Design and red lines

Forward-only, read-only: no training, no optimizer step, no checkpoint save or modification, no src/tests/config/geometry/RoPE/packing/loss/JLT/sampler/CMuon changes, no commit/push. Reports are UNSTAGED. LONGER_P25_AUTHORIZED=NO, P50_AUTHORIZED=NO (recommendation only). Same code path, dtype (bf16 linears, fp32 loss), device, and eager mode across all arms and checkpoints; model dropout is enforced zero in this codebase (no nn.Dropout; dropout raises). Conditioning (Qwen + text adapter + condition tokens) is computed once per unit and shared across arms: forward_conditioning does not read latents, timestep or image_coordinates (train/step.py), so sharing is mathematically exact.

## 2. Input parity (spec s29)

All units share one frozen input cache (per-unit .pt bundles: x0, per-stratum noise/eps, x_t states, Qwen states, token routing, arm maps). Only checkpoint weights vary across PRE/MID/POST. Input manifest hash and per-unit file hashes are in the provenance. Checkpoint model-tree sha256:
- MID U117100: `/tmp/camera-coordinate-causal/ckpts/MID/ckpt_117100_raw-117100-update-cadence` model_tree=036fbb43ff4a5413 manifest=ac1193931337f37e
- POST U118100: `/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence` model_tree=dfbe25e4ada0ad69 manifest=821a8c12546a0948
- PRE U116100: `/sakuramoon-runtime/output_model/g1/ckpt_116100_raw-116100-update-cadence` model_tree=4f8efca407e692ca manifest=7edc89cfaf832cc6
- Forward determinism probe (identical forward twice): worker0: PRE bitexact=True, MID bitexact=True, POST bitexact=True; worker1: PRE bitexact=True, MID bitexact=True, POST bitexact=True.
- Arm validation (first 64 camera units, before mass forward): correct==affine maxdiff 2.38e-07, identity==full-canvas maxdiff 0.00e+00, opposite==affine(-shift) maxdiff 1.19e-07, zoom==audit maxdiff 0.00e+00, shift params==audit maxdiff 0.00e+00, maps finite=True, no self-assignment=True, derangement reproducible=True -> PASS.

## 3. Negative controls (spec s20)

| control | PRE | MID | POST | requirement |
|---|---|---|---|---|
| SAME arm (CORRECT duplicated) mean | -2.675e-07 | 2.936e-07 | -8.877e-08 | numerically zero |
| SAME arm max abs | 9.163e-05 | 1.565e-04 | 8.124e-05 | ~0 (harness plumbing) |
Strict-identity ordinary subset (n=146): IDENTITY vs CORRECT margin PRE mean -2.345e-07 / maxabs 3.497e-05, MID mean 3.481e-07 / maxabs 5.825e-05, POST mean -9.308e-08 / maxabs 3.120e-05 -> zero as required.
All-ordinary IDENTITY vs CORRECT (diagnostic, includes aspect-crop zoom units): PRE 0.00039, MID 0.00041, POST 0.00039.
RANDOM-geometry sensitivity on 64 ordinary 256x256 units (must be NON-zero, proving the harness can detect a coordinate effect when one exists): PRE 0.011002 (n=64), MID 0.004090 (n=64), POST 0.004024 (n=64).

## 4. CORRECT vs IDENTITY / OPPOSITE / SHUFFLED (camera cohort)

Margins M = Loss(wrong) - Loss(correct); raw and normalized (÷ max(L_correct, 1e-8)). Positive = wrong coordinates are penalized (the model reads absolute coordinates).

| metric | PRE | MID | POST | POST-PRE delta (95% CI) | trend |
|---|---|---|---|---|---|
| M_IDENTITY raw | -0.001722 | 0.002232 | 0.002249 | +0.003971 [0.003708, 0.004249] | MONOTONIC_GAIN |
| M_OPPOSITE raw | 0.001706 | 0.002025 | 0.002099 | +0.000392 [0.000244, 0.000544] | MONOTONIC_GAIN |
| M_SHUFFLED raw | 0.001284 | 0.001916 | 0.002040 | +0.000756 [0.000393, 0.001109] | MONOTONIC_GAIN |
| M_IDENTITY norm | -0.003425 | 0.004381 | 0.004403 | +0.007828 | - |
| M_OPPOSITE norm | 0.003226 | 0.003962 | 0.004114 | +0.000888 | - |
| M_SHUFFLED norm | 0.002474 | 0.003689 | 0.003930 | +0.001456 | - |

Scale: MAIN correct loss mean = PRE 0.55505, MID 0.55110, POST 0.55053. Effect size as % of MAIN loss (M_OPPOSITE): PRE 0.3073%, MID 0.3675%, POST 0.3812%.

Correct-best-among-{correct, identity, opposite} fraction: PRE 0.2428, MID 0.4191, POST 0.4329.

Optional interpolation arms (interpretive only; linear meaning holds because coordinates enter as raw RoPE angles and the camera affine is linear in the map): M_HALF raw PRE=-0.002561, M_HALF raw MID=0.000365, M_HALF raw POST=0.000408, M_OVER raw PRE=0.015274, M_OVER raw MID=0.003313, M_OVER raw POST=0.002792.

## 5. Prediction sensitivity (spec s17)

Per-sample rel-RMS = RMS(pred_wrong - pred_correct) / max(RMS(pred_correct), eps); cosine on flattened preds.

| arm | PRE relRMS | POST relRMS | PRE cos | POST cos |
|---|---|---|---|---|
| IDENTITY | 0.068053 | 0.055724 | 0.996248 | 0.997009 |
| OPPOSITE | 0.060505 | 0.043014 | 0.996232 | 0.997649 |
| SHUFFLED | 0.063050 | 0.044546 | 0.996428 | 0.997709 |
| HALF | 0.047107 | 0.032314 | 0.998195 | 0.998971 |
| OVER | 0.075695 | 0.038375 | 0.995740 | 0.998583 |

## 6. Timestep strata (raw M_OPPOSITE per stratum)

| t quantile | PRE | MID | POST |
|---|---|---|---|
| t0.1388 | 0.005798 | 0.006568 | 0.006833 |
| t0.2482 | 0.000873 | 0.001065 | 0.001077 |
| t0.3795 | 0.000172 | 0.000337 | 0.000340 |
| t0.5561 | -0.000019 | 0.000133 | 0.000144 |

(The JLT distribution puts <5% of training mass above t=0.626 (95th pct); the 'late/clean' stratum is therefore the 90th percentile of the real sampler, t=0.5561 - inside, not beyond, the training support.)

## 7. Geometry strata (M_OPPOSITE, POST-PRE)

| stratum | level | n | PRE | POST | delta (95% CI) |
|---|---|---|---|---|---|
| zoom | mild | 1371 | 0.001656 | 0.001885 | +0.000230 [0.000096, 0.000364] |
| zoom | medium | 597 | 0.001873 | 0.002454 | +0.000579 [0.000222, 0.000938] |
| zoom | strong | 80 | 0.001314 | 0.003101 | +0.001788 [0.000380, 0.003296] |
| shift | lt2 | 1206 | 0.000850 | 0.000981 | +0.000131 [0.000012, 0.000246] |
| shift | 2to4 | 710 | 0.002939 | 0.003405 | +0.000465 [0.000172, 0.000763] |
| shift | ge4 | 132 | 0.002844 | 0.005217 | +0.002375 [0.001133, 0.003634] |
| orientation | horizontal | 491 | 0.001729 | 0.001938 | +0.000206 [-0.000019, 0.000430] |
| orientation | vertical | 1557 | 0.001699 | 0.002149 | +0.000449 [0.000272, 0.000636] |
| offset | low | 1008 | 0.000760 | 0.001933 | +0.001172 [0.000964, 0.001389] |
| offset | high | 1040 | 0.002630 | 0.002260 | -0.000370 [-0.000564, -0.000174] |
| edge | L | 692 | 0.001111 | 0.002670 | +0.001560 [0.001280, 0.001864] |
| edge | C | 670 | 0.000260 | 0.000351 | +0.000089 [-0.000045, 0.000221] |
| edge | R | 686 | 0.003701 | 0.003208 | -0.000494 [-0.000764, -0.000217] |

Shuffled-arm test stratification: all n=2048 PRE=0.001284 POST=0.002040 delta=+0.000755 [0.000397, 0.001108]; large_shift n=842 PRE=-0.001404 POST=0.002318 delta=+0.003721 [0.003086, 0.004380]; strong_zoom n=80 PRE=-0.016668 POST=0.001040 delta=+0.017714 [0.014753, 0.020744].

## 8. Optional coordinate-leaf gradient probe

See `cc_stage2b` results (128 camera units, median stratum, CORRECT arm, model params frozen, d(loss)/d(coordinate leaf) RMS per checkpoint). Secondary evidence only; SKIP if the RoPE/FA2 path lacks autograd support (recorded in results if run).

## 9. Statistics (spec s25-s27)

Unit = top-level cluster (image unit); timestep strata and arms are paired within unit. Cluster bootstrap: seed 20260906, n=10000, units resampled with replacement; CIs are percentiles of the resampled mean (paired POST-PRE uses the same resample for both checkpoints).

| quantity | PRE | MID | POST |
|---|---|---|---|
| M_OPPOSITE raw mean (95% CI) | 0.001706 [0.001519, 0.001902] | 0.002025 [0.001845, 0.002211] | 0.002099 [0.001917, 0.002287] |
| M_IDENTITY raw mean (95% CI) | -0.001722 [-0.002004, -0.001450] | 0.002232 [0.002065, 0.002405] | 0.002249 [0.002082, 0.002422] |
| M_SHUFFLED raw mean (95% CI) | 0.001284 [0.000898, 0.001685] | 0.001916 [0.001753, 0.002087] | 0.002040 [0.001875, 0.002217] |

Power: **FORMAL** - 2048 usable CAMERA_APPLIED units (target 2048, floor 1024).

## 10. 2000U INTERPRETATION (spec s33)

Trend of M_OPPOSITE: **MONOTONIC_GAIN** -> **CASE_B**.

Margins still rise from MID to POST at the end of the 2000U window. The trajectory is not yet flat, so 2000U cannot be treated as conclusive in either direction: a longer exposure could plausibly change the verdict. No absolute claim about sufficiency is warranted.

## 11. CAUSAL VERDICT (spec s32)

**INCONCLUSIVE**

Decision inputs: M_OPPOSITE POST-PRE delta +0.000392 (95% CI [0.000244, 0.000544]); trend classes OPPOSITE=MONOTONIC_GAIN, IDENTITY=MONOTONIC_GAIN, SHUFFLED=MONOTONIC_GAIN; sensitivity OPPOSITE relRMS PRE 0.060505 -> POST 0.043014; direction structure at POST: M_OPPOSITE 0.002099 vs M_IDENTITY 0.002249; harness numerics OK=True.

## 12. Relation to the behavioral null (task 1)

The expanded behavioral evaluation (2808 images, generation-level) found NULL_EFFECT / CASE B with WEAK effectiveness: no stable behavioral gain from 2000U of camera exposure. This causal audit tests whether the *mechanism* (learned dependence on absolute coordinates) moved at all, independent of downstream generation quality. A null behavioral result is consistent with NO_DETECTED_COORDINATE_LEARNING; a positive causal result would mean the supervision signal was absorbed without (yet) changing generation, and would shift the recommendation toward longer exposure; a flat causal result means the 25% coordinate-perturbation supervision produced no measurable coordinate dependence in 2000U.

## 13. Recommendation and authorization (spec s34, s38)

RECOMMENDATION: **NO_MORE_EXPOSURE_YET** (recommendation only; all authorization = NO).

AUTHORIZATION: LONGER_P25_AUTHORIZED = NO; P50_AUTHORIZED = NO; no production camera change by this audit.

## 14. Git immutability (spec s37)

head=b2443af436b268fafb2cb6c05d724e3ab0d6042c == entrance b2443af436b268fafb2cb6c05d724e3ab0d6042c; tracked changes now: NONE; `git diff b2443af..HEAD -- src tests config`: EMPTY; untracked: 16 files (11 prior-task reports + 5 this audit); commit/push: NONE. Gate: PASS.

## 15. Provenance (spec s36)

See `camera-coordinate-causal-audit.json` -> provenance (code head, checkpoint trees + manifest hashes, validation source + shard hashes, unit manifest hash + per-unit file hashes, Qwen/VAE asset sha256, coordinate impl file:line, timestep sampler file:line + locked floats, arm definitions, all seeds, script sha256s, runtime mode + determinism probe). Asset hashes: model.safetensors=a71a234d04b9a026…, diffusion_pytorch_model.safetensors=34e076dc1e8a1532… (full values in audit.json).

## 16. NEXT

**STOP AT USER GATE.** No further autonomous training, exposure, or deployment action. Await user GO/NO-GO on the recommendation.
