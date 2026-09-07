# SakuraMoon Camera Vertical Bottom Supervision Audit

> **Label: VERTICAL BOTTOM SUPERVISION AUDIT** - same-source mirrored vertical viewport
> pairs; 2x2 TOP/BOTTOM x correct/mirrored-correct coordinate factorial on three frozen
> checkpoints (PRE U116100 / MID U117100 / POST U118100). Forward-only, read-only, no
> training. Supersedes nothing; complements the Camera Coordinate Causal audit and its
> posthoc V2 round (base `34f646ab`).

## 1. Pair design
- candidate vertical units: 1030
- unique same-source: 1030 (dedup keep smallest unit id)
- selected pairs: **512** (seed 20260907)
- original START anchors: 259; original END anchors: 253
- zoom bands: mild 350 / medium 141 / strong 21
- pair manifest sha256: `7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297`
- anchor reconstruction parity: 480/480 bitexact, worst maxabs 0.000e+00, rel_rms 0.000e+00

## 2. Geometry (per pair, verified for all pairs)
- same source / full canvas / zoom / available / |shift| / target (256,256): YES
- signed_shift_top == -signed_shift_bottom (exact): True
- TOP crop = (0, k_start, 256, k_start+256); BOTTOM crop = (0, k_end, 256, k_end+256)
- all pair invariants: True

## 3. Conditioning / timestep / noise
- conditioning = frozen anchor unit bundle (qwen states + all routing), shared by TOP and BOTTOM; NO caption regeneration
- t strata: [0.1388061520926247, 0.24819609741338816, 0.37948289980451305, 0.5560734456280833]
- one deterministic eps per (pair, stratum), shared by TOP and BOTTOM; same inputs across PRE/MID/POST

## 4. Numerics prerequisite (spec s27)
- V2 numerics clean: True; hidden mutable state: False
- P3 state_history_effect: False (wired into this gate: True)
- repeat jitter (diagnostic only): True

## 5. Causal results (point = exact mean; CI = paired bootstrap seed 20260907, n=10000)

| transition | D_TOP | CI | D_BOTTOM | CI | G = D_TOP - D_BOTTOM | CI |
|---|---|---|---|---|---|---|
| PRE->MID | 0.00141766 | [0.00111898, 0.0017271] | -0.000649035 | [-0.000996853, -0.000310067] | **0.0020667** | [0.00155563, 0.00260406] |
| MID->POST | 0.000307636 | [0.000111816, 0.000505215] | -0.000130995 | [-0.000288734, 2.26394e-05] | **0.00043863** | [0.000184761, 0.000705578] |
| PRE->POST | 0.0017253 | [0.00136104, 0.00212742] | -0.00078003 | [-0.00115646, -0.000421724] | **0.00250533** | [0.00191989, 0.00316719] |

### M levels (4-stratum mean margin, all pairs)
- M_TOP: PRE 0.000389735 [5.94804e-05, 0.000724385]; MID 0.00180776 [0.00148623, 0.00214115]; POST 0.00211501 [0.00177065, 0.0024898]
- M_BOTTOM: PRE 0.00432636 [0.00383759, 0.00482942]; MID 0.00367702 [0.00325391, 0.00412115]; POST 0.00354669 [0.00311532, 0.00399634]

## 6. Natural cohort vs same-source (diagnostic, no mixed bootstrap)
- historical natural TOP: 0.00182477 [0.00148954, 0.00218607]
- historical natural BOTTOM: -0.000644338 [-0.00099515, -0.000289266]
- historical gap: 0.00243725 [0.00191952, 0.00296873]
- same-source gap (PRE->POST): 0.00250533 [0.00191989, 0.00316719]
- confounding attenuation: -0.028

## 7. Content audit
- CLIP text available: True (AVAILABLE)
- CLIP text TOP-BOTTOM delta: 0.00058445 [-0.000605104, 0.00179019]
- CLIP image sim(full,crop) TOP-BOTTOM delta: 0.0232431 [0.019882, 0.0265391]
- PE retained-content TOP-BOTTOM delta: 0.00894915 [0.00586515, 0.0119405]
- systematic BOTTOM content loss: True
- content-balanced subset: n=256 (pre-registered, uses causal results: False)
- balanced-subset PRE->POST gap: 0.00216326 [0.00128263, 0.00321546]; attenuation vs all: 0.137
- Spearman(content asymmetry, G_PRE->POST): 0.13141971295247248
- quartile trend (G mean by content-asymmetry quartile): {'q0': {'n': 128, 'mean_g': 0.0016953338345047086}, 'q1': {'n': 128, 'mean_g': 0.0026311911933589727}, 'q2': {'n': 128, 'mean_g': 0.0019249830802436918}, 'q3': {'n': 128, 'mean_g': 0.0037698114465456456}}

## 8. Correct-loss baseline (L_BB - L_TT)
- PRE: -0.0012917 [-0.00510967, 0.00241804]
- MID: -0.00132927 [-0.00513982, 0.00237037]
- POST: -0.00147198 [-0.00527929, 0.0022599]

## 9. Numerics floor (SAME duplicate forwards; mean + CI, not a hard gate)
- PRE: floor TOP 1.52606e-07 [-1.73389e-07, 4.7824e-07]; floor BOTTOM -2.08791e-07 [-8.81503e-07, 4.39717e-07]
- MID: floor TOP 5.17e-07 [9.53136e-08, 9.70401e-07]; floor BOTTOM -5.16258e-07 [-1.14135e-06, 4.27721e-08]
- POST: floor TOP 1.31513e-07 [-3.10172e-07, 6.06093e-07]; floor BOTTOM 1.47651e-07 [-3.79354e-07, 6.73968e-07]

## 10. Strata (formal CI only for n>=30, else SMALL_N)
- anchor_START: n=259 G=0.00273076 CI=[0.00182915, 0.00380564]
- anchor_END: n=253 G=0.00227456 CI=[0.0015661, 0.00300143]
- zoom_mild: n=350 G=0.000684171 CI=[0.000245959, 0.00111553]
- zoom_medium: n=141 G=0.00571216 CI=[0.00424948, 0.0075245]
- zoom_strong: n=21 G=0.0113264 CI=SMALL_N
- latent_lt2: n=200 G=0.000286756 CI=[-0.000143409, 0.000730861]
- latent_2to4: n=275 G=0.00285134 CI=[0.00199536, 0.00384116]
- latent_ge4: n=37 G=0.0119259 CI=[0.00938192, 0.0147221]

## 11. Classification & recommendation (spec s45/s46)
- **VERDICT: MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY** (case D)
- RECOMMENDATION: **REVIEW_CROP_AND_VERTICAL_SUPERVISION** (longer_p25_authorized=False, p50_authorized=False)

## 12. Validation / immutability / security
- validation: {"anchor_parity": "480/480 bitexact vs frozen stage1 bundles (worst maxabs 0.0, worst rel_rms 0.0; 32 smoke pairs cached from the 32-pair smoke run)", "caption_validation": "3/3 pairs bit-exact vs frozen bundles (production-verbatim caption reconstruction)", "ckpt_gates": "PASS x3 (PRE 116100 / MID 117100 / POST 118100: path + checkpoint manifest sha + full model-tree sha == stage1 manifest evidence; growth_alpha 1.0)", "forward_only": true, "no_training": true, "prerequisite_gate": "PASS (frozen microprobe sha verified; hidden_mutable_state=false; P3 wired & false; NUMERICS_V2_CLEAN=true; verdict POSITIVE_LOSS_PREFERENCE_LEARNING)", "pytest": "135 passed (20 audit + 21 V1 + 49 V2 + 45 VBS), 0 skip, 0 xfail", "replay": "REPLAY = PASS (9 point estimates + 3 arms x (3 checkpoint means+ci95 + 3 delta point+ci95) exactly equal to committed report)", "ruff": "All checks passed (ruff 0.16.x, worktree context: new tooling 15 files + new test)", "smoke_timing_gate": "32-pair smoke with 8 arms: ~0.064 s/forward <= 1.18 s/forward pre-registered bound -> T_ID/B_ID arms kept"}
- immutability: {"frozen_pair_manifest_sha_verified_at_analyze": true, "git_diff_base_head_reports_lines_before_evidence_commit": 0, "git_diff_base_head_src_config_lines": 0, "git_diff_base_to_tooling_head_files": "15 new files, 5251 insertions, 0 deletions (no modification of any committed file)", "prior_committed_reports_untouched": true, "raw_causal_ledger_untouched": true, "stage1_units_ckpt_ckpts_untouched": true}
- security: {"max_new_file_bytes": 49075, "no_png_latent_weights_dataset_staged": true, "secret_scan_note": "crude-grep hits are ML 'token' terminology (input_ids/tokenizers) and pre-registered repository/asset sha constants; no credentials, keys, or connection strings", "secrets_in_new_files": 0, "single_file_over_5mib": false, "staged_file_types": "python source, README, LICENSE/NOTICE, markdown/json/csv reports only"}

## 13. Next
- HARD STOP; SEND SHA TO EXTERNAL REVIEWER; EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING GO

