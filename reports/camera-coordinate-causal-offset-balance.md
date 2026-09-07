# Camera Coordinate Causal - Offset Balance Audit (Posthoc V2)

- generated: 2026-09-07T02:49:28Z
- evidence: /tmp/camera-coordinate-causal (manifest sha d000f58f394d42c0…, worker0 455df176a0f23364…, worker1 80b5b7a73c3525fb…)
- camera units: 2048; valid D_opp units: 2040
- D_opp_unit = (M_OPP_POST - M_OPP_PRE) - (M_SAME_POST - M_SAME_PRE) (no new forwards)

## 1. Planner discrete symmetry

sampler: left/top = random.Random(offset_seed).randrange(available + 1) on the long axis (camera_viewport.py plan_camera_viewport)
reconstruction verified exact for all units: True

| scope | tertile | observed | expected | sd | z |
|---|---|---|---|---|---|
| all | START | 692 | 684.08 | 21.34 | +0.371 |
| all | CENTER | 670 | 674.81 | 21.27 | -0.226 |
| all | END | 686 | 689.11 | 21.38 | -0.145 |
| horizontal | START | 171 | 164.11 | 10.45 | +0.659 |
| horizontal | CENTER | 146 | 161.69 | 10.41 | -1.507 |
| horizontal | END | 174 | 165.20 | 10.47 | +0.840 |
| vertical | START | 521 | 519.97 | 18.61 | +0.055 |
| vertical | CENTER | 524 | 513.12 | 18.55 | +0.586 |
| vertical | END | 512 | 523.90 | 18.64 | -0.638 |

EXPOSURE_COUNT_IMBALANCE (|z| >= 3.0): **False**

Mirror symmetry (k vs available-k): uniform sampler => P(k) = 1/(available+1) = P(available-k) EXACTLY (static proof).

## 2. Vertical TOP vs BOTTOM geometry

TOP n = 521; BOTTOM n = 512

| covariate | TOP mean | TOP p50 | TOP p90 | BOTTOM mean | BOTTOM p50 | BOTTOM p90 | SMD |
|---|---|---|---|---|---|---|---|
| zoom | 1.2047e+00 | 1.1891e+00 | 1.3035e+00 | 1.2020e+00 | 1.1891e+00 | 1.2913e+00 | 4.143e-02 |
| latent_shift | 2.4293e+00 | 2.2188e+00 | 3.6562e+00 | 2.3912e+00 | 2.2188e+00 | 3.6844e+00 | 3.575e-02 |
| pixel_shift | -3.8869e+01 | -3.5500e+01 | -2.0500e+01 | 3.8260e+01 | 3.5500e+01 | 5.8950e+01 | 4.523e+00 |
| abs_displacement | 3.8869e+01 | 3.5500e+01 | 5.8500e+01 | 3.8260e+01 | 3.5500e+01 | 5.8950e+01 | 3.575e-02 |
| canvas_aspect | 1.4559e+00 | 1.4141e+00 | 1.6992e+00 | 1.4490e+00 | 1.4141e+00 | 1.6676e+00 | 4.143e-02 |
| retention | 6.9488e-01 | 7.0718e-01 | 7.7812e-01 | 6.9750e-01 | 7.0718e-01 | 7.7552e-01 | 3.815e-02 |
| available | 1.1671e+02 | 1.0600e+02 | 1.7900e+02 | 1.1495e+02 | 1.0600e+02 | 1.7090e+02 | 4.143e-02 |
| norm_offset | 1.6248e-01 | 1.6541e-01 | 3.0033e-01 | 8.3537e-01 | 8.3615e-01 | 9.6226e-01 | 6.954e+00 |
| main_tokens | 1.9760e+02 | 1.9200e+02 | 3.1100e+02 | 1.9279e+02 | 1.9050e+02 | 3.2060e+02 | 5.257e-02 |
| condition_tokens | 9.4146e+00 | 8.0000e+00 | 1.7000e+01 | 9.4141e+00 | 8.0000e+00 | 1.6900e+01 | 6.597e-05 |

largest non-definitional imbalance: **main_tokens** (SMD 5.257e-02); geometry_imbalanced (>= 0.1): **False**
split-definitional SMDs (tautological, excluded from gate): norm_offset=6.954e+00, pixel_shift=4.523e+00, abs_displacement=3.575e-02
null-condition fraction: TOP 0.1017274472168906 / BOTTOM 0.1328125

## 3. Reference effects (unbalanced valid vertical units)

TOP adjusted D_opp: 1.8521e-03 CI [0.0015031297982539874, 0.0022260830928920094]
BOTTOM adjusted D_opp: -6.3460e-04 CI [-0.0009846982115050197, -0.0002936238153779416] (CI<0: True)

## 4. Reweighted (exact cell reweighting, shared cells only)

shared cells: 11; retained units TOP 521 / BOTTOM 512; dropped units {'top': 0, 'bottom': 0}
TOP ESS = 518.6; BOTTOM ESS = 510.3
TOP reweighted: 1.8248e-03 CI (0.0014895396485939993, 0.0021860739800730494)
BOTTOM reweighted: -6.4434e-04 CI (-0.0009951498707519459, -0.00028926600606747605) (negative persists: True)
effect shrink after reweight: -1.534e-02

## 5. Matched geometry audit (1:1 without replacement)

pairs: 512; post-match SMD: {"zoom_val": 0.003769628703117375, "latent_shift": 0.011033118937671267, "canvas_aspect": 0.0036770096240450532, "available": 0.003677009624045054}; max = 1.103e-02
TOP - BOTTOM paired difference: 2.4373e-03 CI (0.0019195196646251134, 0.002968733798843459)

## 6. Continuous offset relationship (vertical)

Spearman(norm_offset, D_opp) = -3.129e-01 CI (-0.3650075099400405, -0.2593693676734586)
deciles (by normalized offset): n / mean D_opp / CI
  [0.0,0.1): n=168 mean=2.5253e-03 ci=(0.0018993108782784215, 0.0032019202026449855)
  [0.1,0.2): n=150 mean=2.3764e-03 ci=(0.0015769364306082328, 0.0033427121862769126)
  [0.2,0.3): n=150 mean=9.1960e-04 ci=(0.0005244172078867753, 0.0013393312835445008)
  [0.3,0.4): n=148 mean=7.0391e-04 ci=(0.00036412615052147494, 0.0010737362670133245)
  [0.4,0.5): n=153 mean=1.9327e-04 ci=(-2.19900288856497e-05, 0.00043363392876546373)
  [0.5,0.6): n=166 mean=2.5087e-05 ci=(-0.0001331230468419661, 0.0001786606840744436)
  [0.6,0.7): n=152 mean=-3.2915e-04 ci=(-0.0008185181828019649, 0.00011898209157056722)
  [0.7,0.8): n=144 mean=-4.8522e-04 ci=(-0.0011120223854151037, 0.00010991104516304213)
  [0.8,0.9): n=164 mean=-3.4386e-04 ci=(-0.000919920671163354, 0.0002344381490086274)
  [0.9,1.0): n=156 mean=-1.1635e-03 ci=(-0.001880700618792803, -0.00041967970020591436)
mirror bins (low - high):
  [0.0,0.1) vs [0.9,1.0): n 168/156 diff=1.8696e-03 ci=(0.0013751019033844823, 0.0023723690554982536)
  [0.1,0.2) vs [0.8,0.9): n 150/164 diff=1.3148e-03 ci=(0.0008108992589293582, 0.0018549555693130205)
  [0.2,0.3) vs [0.7,0.8): n 150/144 diff=7.0684e-04 ci=(0.0003604623828032593, 0.0010705022068478827)
  [0.3,0.4) vs [0.6,0.7): n 148/156 diff=5.0672e-04 ci=(0.00022508288537593263, 0.0008121136624954249)
  [0.4,0.5) vs [0.5,0.6): n 153/162 diff=8.1178e-05 ci=(-5.552913786636458e-05, 0.00022158875322294622)
interpretation: monotone-ish decline from TOP to BOTTOM

## 7. Shard / cohort / content

shard chi2 = 0.20096594855721459 df=1 p≈0.658130468176348; SHARD_COHORT_IMBALANCE: False
content: AVAILABLE; raw caption: NOT_AVAILABLE

## 8. Balance classification

**EFFECT_PERSISTS_AFTER_GEOMETRY_BALANCE**
- counts and geometry balanced; BOTTOM negative persists after balance -> directional/content interaction, NOT 'sampling imbalance'

Caveat: only a genuine exposure or geometry imbalance is called a 'sampling balance' issue. If counts and geometry are balanced and the BOTTOM effect persists, the correct statement is that a directional/content interaction remains after balance (section 43).
