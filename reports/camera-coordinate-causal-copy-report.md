== SakuraMoon Camera Viewport v2 · Coordinate Causal Audit ==

DESIGN: PRE U116100 / MID U117100 / POST U118100; 2048 CAMERA_APPLIED + 512 ORDINARY units from s0-validation-50k-v1 (seed-44 fixed shards); frozen production inputs (VAE x0, text, 4 JLT-quantile timesteps, per-unit noise); only the image coordinate tensor changes between arms; production PackedDiT forward + JLT x-pred loss; power=FORMAL.

INPUT PARITY: one frozen input cache; only weights vary across checkpoints. Arm validation PASS (64 units, maxdiffs in report §2). Determinism: worker0/PRE bitexact=True, worker0/MID bitexact=True, worker0/POST bitexact=True, worker1/PRE bitexact=True, worker1/MID bitexact=True, worker1/POST bitexact=True.

NEGATIVE CONTROL: SAME-arm maxabs = PRE 9.2e-05 / MID 1.6e-04 / POST 8.1e-05 (zero as required). Strict-identity ordinary subset (n=146) IDENTITY-CORRECT maxabs = PRE 3.5e-05 / MID 5.8e-05 / POST 3.1e-05 RANDOM-geometry sensitivity (64 units) POST 0.00402 (non-zero, harness detects effects).

CORRECT vs IDENTITY: M_raw PRE/MID/POST = -0.00172 / 0.00223 / 0.00225; POST-PRE +0.003971 [0.003708, 0.004249] (MONOTONIC_GAIN).
CORRECT vs OPPOSITE: M_raw PRE/MID/POST = 0.00171 / 0.00203 / 0.00210; POST-PRE +0.000392 [0.000244, 0.000544] (MONOTONIC_GAIN; 8 zero-shift units excluded as N/A).
CORRECT vs SHUFFLED: M_raw PRE/MID/POST = 0.00128 / 0.00192 / 0.00204; POST-PRE +0.000756 [0.000393, 0.001109] (MONOTONIC_GAIN).

PREDICTION SENSITIVITY (OPPOSITE, relRMS / cos): PRE 0.06051 / 0.99623 -> POST 0.04301 / 0.99765. Correct-best-3 frac: 0.243 / 0.419 / 0.433.

TIMESTEP (M_OPPOSITE raw): t0.1388: +0.00580/+0.00657/+0.00683; t0.2482: +0.00087/+0.00106/+0.00108; t0.3795: +0.00017/+0.00034/+0.00034; t0.5561: -0.00002/+0.00013/+0.00014 (PRE/MID/POST).

GEOMETRY STRATA: see report §7 (zoom mild/medium/strong; latent shift <2/2-4/>=4; orientation; offset low/high; edge L/C/R; shuffled-test splits).

OPTIONAL COORD GRAD: see report §8 / cc_stage2b output.

STATISTICS: cluster bootstrap seed 20260906 n=10000 (units = clusters; strata + arms paired); power=FORMAL (2048 usable camera units). Effect size M_OPPOSITE/MAIN loss: PRE 0.307% / MID 0.368% / POST 0.381%.

2000U INTERPRETATION: trend MONOTONIC_GAIN -> CASE_B. Margins still rise from MID to POST at the end of the 2000U window. The trajectory is not yet flat, so 2000U cannot be treated as conclusive in either direction: a longer exposure could plausibly change the verdict. No absolute claim about sufficiency is warranted.

CAUSAL VERDICT: INCONCLUSIVE

RELATION TO BEHAVIORAL NULL: task-1 expanded eval was NULL_EFFECT/CASE B/WEAK; a flat causal result confirms the null is not a downstream-generation artifact (the mechanism itself did not move); a rising causal result would mean absorbed supervision not yet visible in generation.

RECOMMENDATION: NO_MORE_EXPOSURE_YET

AUTHORIZATION: LONGER_P25_AUTHORIZED = NO; P50_AUTHORIZED = NO; no production change by this audit.

GIT: head=b2443af436b268fafb2cb6c05d724e3ab0d6042c (== entrance); src/tests/config diff EMPTY; tracked changes: NONE; untracked: 16 (prior 11 + this audit 5); commit/push: NONE. Gate PASS.

NEXT: STOP AT USER GATE. Awaiting GO/NO-GO on the recommendation.
