# SakuraMoon Camera · Visual Content Residual Audit

Same-source vertical mirror pairs (512), real visual-content metrics only (CLIP image-full + PE retained; CLIP text secondary). CPU only; frozen feature JSON + frozen per-pair statistics; no forwards, no re-encode, no training.

## 1. Inputs & integrity
- pairs = 512; clip-image rows = 512; PE rows = 512; CLIP text available = True (rows 512)
- frozen main == committed audit (exact); committed CSV 12-sig-digit cross-check = PASS
- pair manifest sha = 7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297
- subset freeze sha = 15506002a0480e29e0b87c147bb018432349b99fe8d1af85b24be196b88db0ca
- cached note: this audit uses ONLY frozen feature JSON and frozen per-pair statistics (no re-encode, no forwards)

## 2. Phase separation
- subset frozen before mirror load: YES (PHASE A reads manifest + CLIP + PE only)
- freeze script reads mirror data: NO (static source lock, see tests)

## 3. Visual asymmetry (TOP-BOTTOM signed; positive = TOP preserves more)
- ALL (n=512): CLIP-image signed = 0.0232431 [0.019882, 0.0265391]; PE signed = 0.00894915 [0.00586515, 0.0119405]; A_visual mean/p50/p90 = 0.5/0.501957/0.80411
- V50 (n=256): CLIP-image signed = 0.0141699 [0.0112551, 0.0172162]; PE signed = 0.00231353 [-4.49402e-05, 0.00474361]; A_visual mean/p50/p90 = 0.329726/0.325832/0.515166
- V25 (n=128): CLIP-image signed = 0.00963654 [0.00643882, 0.0136448]; PE signed = 0.00140684 [-0.000736482, 0.00363509]; A_visual mean/p50/p90 = 0.23329/0.22456/0.347945

## 4. Geometry preservation (composition vs proportional expectation)
- ALL: START/END = 259/253; zoom mild/medium/strong = 350/141/21; latent lt2/2to4/ge4 = 200/275/37; mean zoom = 1.20334, mean latent = 2.43268, mean |shift| px = 38.9229
- V50: START/END = 129/127; zoom mild/medium/strong = 175/71/10; latent lt2/2to4/ge4 = 100/138/18; mean zoom = 1.20193, mean latent = 2.36145, mean |shift| px = 37.7832
- V25: START/END = 65/63; zoom mild/medium/strong = 88/36/4; latent lt2/2to4/ge4 = 50/69/9; mean zoom = 1.2018, mean latent = 2.34351, mean |shift| px = 37.4961

## 5. Causal residual (G = D_TOP - D_BOTTOM, per source pair)
- PRE|MID: ALL = 0.0020667 [0.00155563, 0.00260406]; V50 = 0.0017454 [0.00109455, 0.00244141]; V25 = 0.00204399 [0.00114788, 0.00307982]
- MID|POST: ALL = 0.00043863 [0.000184761, 0.000705578]; V50 = 0.000263242 [5.05912e-05, 0.000488717]; V25 = 0.000233175 [-5.04156e-05, 0.00050861]
- PRE|POST: ALL = 0.00250533 [0.00191989, 0.00316719]; V50 = 0.00200864 [0.00134108, 0.00271416]; V25 = 0.00227716 [0.00132526, 0.00333803]
- RETENTION_50 = 0.801748; RETENTION_25 = 0.908928
- ATTEN_50 = 0.198252; ATTEN_25 = 0.0910721 (negative allowed, not clipped)

## 6. Correlations (Spearman)
- rho_C_img_signed_vs_G = 0.158126
- rho_C_pe_signed_vs_G = 0.131312
- rho_A_img_vs_absG = 0.231713
- rho_A_pe_vs_absG = 0.0914757
- rho_A_visual_vs_absG = 0.202585
- CLIP text (secondary): rho_C_text_signed_vs_G = -0.0489796, rho_abs_C_text_signed_vs_absG = 0.0757013

## 7. Quartiles of A_visual (Q0 lowest -> Q3 highest)
- Q0: n=128, mean G = 0.00133608 [0.000605977, 0.00210295], A_visual mean = 0.215463
- Q1: n=128, mean G = 0.00148051 [0.000596172, 0.00238372], A_visual mean = 0.414277
- Q2: n=128, mean G = 0.00289857 [0.00181662, 0.0040636], A_visual mean = 0.578622
- Q3: n=128, mean G = 0.00430616 [0.00267015, 0.00633162], A_visual mean = 0.791639

## 8. Signed visual-loss groups (G per group)
- CLIP image: TOP_preserves_more n=421 G=0.00271544 [0.00203555, 0.00348219]; BOTTOM_preserves_more n=91 G=0.0015333 [0.000654086, 0.00245486]; tied n=0
- PE retained: TOP_preserves_more n=309 G=0.00330581 [0.00247095, 0.0042799]; BOTTOM_preserves_more n=203 G=0.00128687 [0.000635592, 0.0019587]; tied n=0

## 9. Joint-low intersection (secondary stringent subset)
- JOINT_LOW_VISUAL_50_INTERSECTION: n=145, mean G = 0.00146127, CI = [0.0006656295497869623, 0.002306399845100682]

## 10. Sensitivity subsets (G PRE|POST)
- GLOBAL_VISUAL_50: n=256, G = 0.0014083 [0.000829487, 0.00199553]
- GLOBAL_VISUAL_25: n=128, G = 0.00133608 [0.000594442, 0.00209593]
- CLIP_IMAGE_ONLY_GEO_50: n=256, G = 0.00233519 [0.00166366, 0.00303565]
- CLIP_IMAGE_ONLY_GEO_25: n=128, G = 0.00262889 [0.00167325, 0.00369502]
- PE_ONLY_GEO_50: n=256, G = 0.00199405 [0.00131066, 0.00270219]
- PE_ONLY_GEO_25: n=128, G = 0.00208636 [0.00109378, 0.00317313]

## 11. In-subset geometry confirmation
- V50: latent_2to4 n=138 G=0.00234047 [0.00143692, 0.00330044]; latent_ge4 n=18 G=0.0105065; latent_lt2 n=100 G=2.11108e-05 [-0.000486969, 0.000532885]; zoom_medium n=71 G=0.00457586 [0.00312725, 0.00623517]; zoom_mild n=175 G=0.000470518 [-8.58522e-05, 0.00104432]; zoom_strong n=10 G=0.0106986
- V25: latent_2to4 n=69 G=0.00229159 [0.0010774, 0.00351589]; latent_ge4 n=9 G=0.0125265; latent_lt2 n=50 G=0.00041237 [-0.00028993, 0.00115531]; zoom_medium n=36 G=0.00516833 [0.00300201, 0.007871]; zoom_mild n=88 G=0.00073863 [-5.31712e-05, 0.00155265]; zoom_strong n=4 G=0.0101044

## 12. Classification & recommendation
- DIRECTIONAL RESIDUAL = CONFIRMED
- VISUAL CONTENT CONTRIBUTION = STRONG_SUPPORTED
- OVERALL = MIXED_VISUAL_CONTENT_AND_DIRECTIONAL
- RECOMMENDATION = DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION_WITH_CONTENT_GUARDS (longer_p25_authorized=False, p50_authorized=False)

## 13. Interpretation
- primary: a directional residual persists after visual-content balancing; visual content asymmetry alone does not explain the TOP-BOTTOM gap

## 14. Legacy tool note
- the prior-generation legacy helper is not imported or called by this audit; this tool implements only paired_mean_difference and two_sample_mean_difference.

## 15. Validation / immutability / security (gate runner)
- validation: {"causal_id": "512/512; frozen vertical-bottom-main per_pair == committed audit per_pair (exact); committed pairs.csv G_PRE_POST 12-sig-digit cross-check PASS (|delta| <= 1e-12)", "feature_id": "512/512 CLIP image rows + 512/512 PE rows + 512/512 CLIP text rows (AVAILABLE), pair_index exactly 0..511, no duplicates, all finite", "phase_separation": "subset freeze executed BEFORE any per-pair mirror load; freeze source static-locked (tests)", "prerequisite_gate": "PASS (frozen pair manifest sha == pinned 7f3a81b2...; v2_numerics_clean=true; hidden_mutable_state=false; P3 wired into gate & p3_state_history_effect=false; selected_pairs=512; committed same-source PRE->POST gap CI lower 0.001919886 > 0)", "py_compile": "PASS (4 new python files)", "pytest": "175 passed (20 audit + 21 V1 + 49 V2 + 45 VBS + 40 VCR), 0 skip, 0 xfail (worktree context, DTK env sourced)", "replay": "REPLAY = PASS (37 checks: 3 all-set gaps + 6 arm M levels + 9 adjusted deltas, each point+ci95; content-balanced subset point+ci95; spearman; 4 quartile means - all bit-exact vs committed VBS audit from frozen per-pair data)", "ruff": "All checks passed (ruff worktree context: 4 new tooling files + 1 new test)"}
- immutability: {"frozen_pair_manifest_sha_verified_at_analyze": true, "git_diff_base_head_src_config_lines": 0, "git_diff_base_to_tooling_head_files": "5 new files, 2205 insertions, 0 deletions (no modification of any committed file)", "prior_committed_files_untouched": "23/23 pinned sha256 re-verified by the 40-test suite in the worktree (5 VBS reports + 4 prior test suites + 14 VBS tooling files)", "raw_causal_ledger_untouched": true, "tmp_evidence_retained": true}
- security: {"max_new_file_bytes": 39648, "no_png_latent_weights_dataset_staged": true, "secret_scan_note": "crude grep over the 5 new files: only hits are a copy-report template placeholder and a forbidden-token loop variable; no credentials, keys, or connection strings", "secrets_in_new_files": 0, "single_file_over_5mib": false, "staged_file_types": "python source, README, markdown/json/csv reports only"}
