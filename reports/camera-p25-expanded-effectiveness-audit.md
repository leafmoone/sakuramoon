# SakuraMoon Camera Viewport v2 — P25 EXPANDED EFFECTIVENESS AUDIT

24 prompts × 3 seeds × 13 points (12 training-coupled + identity) × 3 checkpoints (PRE U116100 / MID U117100 / POST U118100) = 2808 images, salt15 evaluation-only.

## 1. Design & anti-cherry-pick

- design frozen before render: `True` (design.json), master seed 20260905, deterministic prompt selection (audit of repo prompt sources; lexicographic within stratum; no result-based selection).
- strata (5 × 4): mixed_other=6, multi_character=6, scenery_environment=4, single_close_upper=4, single_full_body=4
- legacy4: validation-fac64aa1d, validation-c17a74b54, validation-fe7761d38, validation-d985f354c
- seeds: v0 = bank/legacy seed per prompt; v1/v2 = splitmix64-derived (frozen noise-hashes.json). Cherry-picked = NO.

## 2. Integrity

- images: 2808/2808 expected; checks: count_2808=True, all_valid_png=True, all_dimensions_256=True, no_blank=True, no_corrupt=True, no_missing=True, no_duplicate_logical_keys=True, manifest_identity=True
- per-image ledger: /tmp/camera-p25-expanded/render-integrity.json (png sha256 + recomputed initial-noise sha256, 72/72 match frozen noise-hashes.json; resolved-config anchor all 18 chunk manifests = YES)
### 2.1 §10 legacy protocol continuity — ADJUDICATED

- new 12-case chunks vs old 4-case matrix: **0/52 byte-exact** (legacy4 × 13 points, PRE).
- Matched batch context re-render (validation-prompts.json, count=4, same 13 points, PRE ckpt): **52/52 byte-exact** → BIT_EXACT_REPRODUCTION = PASS under matched batch context.
- Root cause: batched CFG DiT forward is numerically batch-size dependent on HCU; per-case pixels depend on co-batched cases. Deterministic within a fixed batch composition (same-file re-render byte-identical; HCU0 ≡ HCU1).
- Consequence per spec §10: **old and new statistics are NOT mixed**. Legacy4 replication below is assessed within the new 12-case context. Internal PRE/MID/POST pairing is valid (identical 12-case chunk files across checkpoints; only variable = checkpoint weights).

## 3. Primary statistics (hierarchical bootstrap, prompt → seed → paired points, n=10000, seed 20260905)

| endpoint | direction | PRE | MID | POST | POST−PRE Δ | 95% hierarchical CI |
|---|---|---|---|---|---|---|
| shift_dir_cos | higher-better | 0.6059 | 0.6259 | 0.5937 | -0.0122 | [-0.0796, 0.0561] |
| shift_sign | higher-better | 0.8368 | 0.8420 | 0.8368 | -0.0005 | [-0.0417, 0.0399] |
| shift_resp_err | lower-better | 0.7611 | 0.7623 | 0.7534 | -0.0076 | [-0.0264, 0.0110] |
| zoom_log_err | lower-better | 0.1917 | 0.1926 | 0.1766 | -0.0151 | [-0.0403, 0.0121] |
| zoom_mono | higher-better | 0.8588 | 0.7477 | 0.7639 | -0.0946 | [-0.1782, -0.0139] |
| content_clip | higher-better | 0.9116 | 0.9171 | 0.9129 | 0.0013 | [-0.0051, 0.0073] |
| content_dense | higher-better | 0.8852 | 0.8911 | 0.8866 | 0.0014 | [-0.0037, 0.0065] |

Interpretation follows spec §33: CI including 0 = *no detected stable gain at this evaluation scale* (effect size reported), NOT proof of null; a narrow CI around 0 is stronger null evidence.

## 4. Splits

- **PRE**: legacy4 dir=0.5274 mono=0.9028 | expanded20 dir=0.6216 mono=0.8500 | left-edge dir=0.6438 right-edge dir=0.5681 | per-seed dir v0/v1/v2=0.6026/0.6011/0.6140
- **MID**: legacy4 dir=0.4638 mono=0.8194 | expanded20 dir=0.6584 mono=0.7333 | left-edge dir=0.6326 right-edge dir=0.6193 | per-seed dir v0/v1/v2=0.6029/0.6035/0.6715
- **POST**: legacy4 dir=0.4646 mono=0.7222 | expanded20 dir=0.6195 mono=0.7722 | left-edge dir=0.6381 right-edge dir=0.5493 | per-seed dir v0/v1/v2=0.5788/0.5817/0.6205

## 5. PRE ceiling analysis (PRE-defined tertiles; §25)

score = 0.25·dir + 0.25·sign + 0.25·mono + 0.25·max(0,1−zoom_err), per prompt (mean of 3 seeds), PRE only.
- **low** (n=8): dir 0.4081->0.4207 (d=0.0126 CI[-0.1181, 0.1418]) | zoom_err 0.2075->0.1976 (d=-0.0099 CI[-0.0444, 0.0246]) | mono d=-0.0139 | clip d=-0.0053
- **mid** (n=8): dir 0.6240->0.5650 (d=-0.0590 CI[-0.1724, 0.0670]) | zoom_err 0.2015->0.1724 (d=-0.0291 CI[-0.0767, 0.0228]) | mono d=-0.1667 | clip d=0.0085
- **high** (n=8): dir 0.7857->0.7954 (d=0.0097 CI[-0.0729, 0.0883]) | zoom_err 0.1662->0.1596 (d=-0.0066 CI[-0.0499, 0.0466]) | mono d=-0.1042 | clip d=0.0006
- **ceiling verdict**: low-PRE CI-backed gain = False (dir CI [-0.1181, 0.1418], zoom-err CI [-0.0444, 0.0246]); high-PRE capped (no detectable gain) = True (dir d=0.0097); aggregate dir CI includes 0; content healthy = True.

## 6. Content (§21)

- case2 (original, validation-c17a74b546e841cbac26930ea4119aef): content PRE->POST all-seed mean = 0.8302->0.7986 (delta -0.0316); per seed v0/v1/v2 = -0.085 / 0.017 / -0.027 (v0 = bank seed, the first-audit seed: 0.7789->0.6941, d=-0.085)
  strata mean delta: mixed_other=-0.004, multi_character=0.005, scenery_environment=0.011, single_close_upper=0.002, single_full_body=-0.007
- systematic content regression: see PRIMARY classification conditions.

## 7. Legacy4 replication (§26, within new 12-case context)

- validation-fac64aa1d4574f386dec1fefcb2eaca0 [multi_character]: content delta v0/v1/v2 = -0.001 / 0.000 / -0.008 (all-seed mean -0.003) | right-edge dir delta v0/v1/v2 = -0.113 / -0.613 / 0.201 (all-seed mean -0.175)
- validation-c17a74b546e841cbac26930ea4119aef [mixed_other]: content delta v0/v1/v2 = -0.085 / 0.017 / -0.027 (all-seed mean -0.032) | right-edge dir delta v0/v1/v2 = -0.373 / 0.515 / 0.393 (all-seed mean 0.178)
- validation-fe7761d38db73e6709e5afac5a476aea [mixed_other]: content delta v0/v1/v2 = -0.020 / -0.021 / 0.006 (all-seed mean -0.012) | right-edge dir delta v0/v1/v2 = -0.697 / -0.016 / 0.063 (all-seed mean -0.217)
- validation-d985f354c95ef3d2137bd3b5468adb43 [multi_character]: content delta v0/v1/v2 = 0.000 / 0.009 / 0.020 (all-seed mean 0.010) | right-edge dir delta v0/v1/v2 = -0.080 / -1.507 / -0.303 (all-seed mean -0.630)
- **classification: REPRODUCED** (first-audit findings: case2 content regression; right-edge case2/3 regression)
- note: original-seed (v0) result and new-seed (v1/v2) result reported per seed above; pixel-level legacy reproduction see §2.1.

## 8. Edge asymmetry gate (§27)

- per tag left/right dir: PRE L=0.6438 R=0.5681 | MID L=0.6326 R=0.6193 | POST L=0.6381 R=0.5493
- POST: mean left Δ=-0.0057, right Δ=-0.0187, (L−R) Δ=0.0130 CI[-0.1802, 0.2134]; prompts right-worse=12/24 → **CAMERA_EDGE_ASYMMETRY = INCONCLUSIVE**

## 9. Center vs edge + old-regressed-point replication (§28)

- z=1.1: center zoom_err PRE→POST 0.0860→0.0912 (Δ0.0052) | edge dir 0.6935→0.6082 (Δ-0.0853)
- z=1.225: center zoom_err PRE→POST 0.1701→0.1508 (Δ-0.0193) | edge dir 0.6230→0.6344 (Δ0.0113)
- z=1.4142135623730951: center zoom_err PRE→POST 0.2434→0.2140 (Δ-0.0294) | edge dir 0.5751→0.5316 (Δ-0.0435)
- z=1.5: center zoom_err PRE→POST 0.2675→0.2502 (Δ-0.0172) | edge dir 0.5320→0.6006 (Δ0.0686)
- old regressed points replicated in expanded cohort: coupled_z1.4142_center=no(Δ-0.029), coupled_z1.5_center=no(Δ-0.017), coupled_z1.4142_right_edge=YES(Δ-0.025), coupled_z1.225_right_edge=YES(Δ-0.037), coupled_z1.1_right_edge=YES(Δ-0.133)

## 10. Strata interaction (§29)

- mixed_other: camera gain Δ=0.0193 | content Δ=-0.0038 | right-edge Δ=0.0037
- multi_character: camera gain Δ=-0.0643 | content Δ=0.0049 | right-edge Δ=-0.1504
- scenery_environment: camera gain Δ=0.0077 | content Δ=0.0111 | right-edge Δ=0.0329
- single_close_upper: camera gain Δ=0.0038 | content Δ=0.0022 | right-edge Δ=0.0038
- single_full_body: camera gain Δ=-0.0175 | content Δ=-0.0072 | right-edge Δ=0.0709

## 11. Trajectories (§30)

- aggregate: {"shift_dir_cos": "NON_MONOTONIC", "zoom_log_err_neg": "NO_GAIN", "zoom_mono_pair_acc": "NON_MONOTONIC", "content_clip": "NO_GAIN"}
- per prompt: see reports/camera-p25-expanded-effectiveness-prompts.csv (traj_* columns)

## 12. Classification (§31)

- counts: {"POSITIVE": 14, "STABLE_GOOD": 2, "NULL": 2, "TRADEOFF": 2, "NEGATIVE": 4}
- per prompt: see audit.json classification_per_prompt / prompts.csv

## 13. Primary evidence classification (§32)

- **NULL_EFFECT** — primary POST-PRE deltas around 0 (CIs include 0); low-PRE group: no clear gain

- decision mapping (spec 34): CASE B -> **KEEP_P25_OFF_PRODUCTION**. no clear gain in any PRE tertile (all tertile CIs include 0) and no clear aggregate gain => exposure-only hypothesis weakened; do not extend exposure now.
## 14. Outliers (§37)

- top10_camera_gains: validation-06d95#v2:coupled_z1.4142_right_edge(Δ1.945), validation-033c6#v0:coupled_z1.5_right_edge(Δ1.934), validation-d985f#v0:coupled_z1.225_left_edge(Δ1.934), validation-0cc20#v2:coupled_z1.5_left_edge(Δ1.928), validation-06d95#v2:coupled_z1.5_right_edge(Δ1.903), validation-c17a7#v1:coupled_z1.1_left_edge(Δ1.881), validation-0935c#v1:coupled_z1.225_right_edge(Δ1.862), validation-01c4c#v2:coupled_z1.5_right_edge(Δ1.808), validation-00ef6#v1:coupled_z1.225_right_edge(Δ1.788), validation-033c6#v0:coupled_z1.225_left_edge(Δ1.770)
- top10_camera_regressions: validation-0cc20#v0:coupled_z1.1_left_edge(Δ-1.996), validation-02376#v2:coupled_z1.4142_right_edge(Δ-1.974), validation-02b45#v0:coupled_z1.1_right_edge(Δ-1.966), validation-c17a7#v0:coupled_z1.4142_right_edge(Δ-1.891), validation-0020f#v0:coupled_z1.4142_left_edge(Δ-1.889), validation-d985f#v1:coupled_z1.1_right_edge(Δ-1.880), validation-033c6#v1:coupled_z1.225_right_edge(Δ-1.867), validation-fe776#v0:coupled_z1.1_right_edge(Δ-1.820), validation-fac64#v0:coupled_z1.1_right_edge(Δ-1.793), validation-01c0a#v1:coupled_z1.5_left_edge(Δ-1.747)
- top10_content_drops: validation-c17a7#v0:coupled_z1.5_center(Δ-0.175), validation-c17a7#v0:coupled_z1.1_center(Δ-0.109), validation-c17a7#v0:coupled_z1.225_left_edge(Δ-0.104), validation-01c0a#v2:coupled_z1.4142_left_edge(Δ-0.101), validation-c17a7#v2:coupled_z1.225_right_edge(Δ-0.098), validation-c17a7#v2:coupled_z1.225_center(Δ-0.097), validation-c17a7#v0:coupled_z1.5_left_edge(Δ-0.097), validation-c17a7#v0:coupled_z1.5_right_edge(Δ-0.096), validation-0cc20#v0:coupled_z1.225_left_edge(Δ-0.096), validation-0cc20#v2:coupled_z1.5_center(Δ-0.094)
- top10_left_right_asymmetry: validation-0cc20#v2:1.5(Δ2.907), validation-d985f#v1:1.4142135623730951(Δ2.616), validation-d985f#v2:1.1(Δ2.612), validation-d985f#v1:1.1(Δ2.183), validation-005d6#v2:1.5(Δ2.097), validation-02b45#v0:1.1(Δ1.986), validation-033c6#v1:1.225(Δ1.920), validation-d985f#v0:1.225(Δ1.887), validation-01af6#v2:1.1(Δ1.886), validation-fe776#v0:1.1(Δ1.828)

## 15. Canonical checkpoint drift (§22)

- identity↔identity cosine (72 units): {"clip_PRE_MID": {"mean": 0.9487033258587432, "median": 0.9616283178319851, "min": 0.7804876304246287}, "clip_MID_POST": {"mean": 0.9491514786225179, "median": 0.9532212572424792, "min": 0.800443339373464}, "clip_PRE_POST": {"mean": 0.9484281772443186, "median": 0.9560300979617538, "min": 0.8525352898273633}, "dense_PRE_MID": {"mean": 0.851212403988022, "median": 0.8645711099631506, "min": 0.6526737581594881}, "dense_MID_POST": {"mean": 0.8524069419701008, "median": 0.8677976475213339, "min": 0.6607331398883152}, "dense_PRE_POST": {"mean": 0.8463601866378091, "median": 0.8496994951983378, "min": 0.6270660527724025}}
- explanation: base-checkpoint movement over 2000U, NOT camera learning; used only as context.

## 16. Base quality context (§35 — cited, not re-run)

- FID 53.5928→52.8045, KID 0.021868→0.020761, IS 4.3949→4.6909, CMMD 0.1593→0.1602, concept 0.043610→0.044143 — **NO-REGRESSION CONTEXT ONLY; NOT CAMERA-EFFECT EVIDENCE**.

## 17. P50 / decision (§34)

- CASE B → RECOMMENDATION: **KEEP_P25_OFF_PRODUCTION**
- P25 EFFECTIVENESS: WEAK | EXPOSURE HYPOTHESIS: NO
- P50_AUTHORIZED = NO (locked, unchanged by this task).

## 18. Provenance (§40)

```json
{
 "evaluator_code": {
  "expanded_analyzer": "exp-analyze.py (this script) sha256=6251b5c28cab7dfa7f6d73eb493f3ebf44a59a6c4db1793b9a42fce6e795c0a1",
  "locked_core_source": "/tmp/camera-p25-effectiveness/scripts/p25c_analyze.py (verbatim correspond/fit_similarity/cos)",
  "point_typing_source": "/tmp/camera-p25-effectiveness/scripts/p25a_artifact.py (verbatim rules)"
 },
 "pe_spatial": {
  "weights": "/tmp/camera-p25-effectiveness/refs/model/pe_spatial_b16_512/PE-Spatial-B16-512.pt",
  "sha256": "86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5",
  "note": "same asset + sha as first audit; verified by exp-features.py at load"
 },
 "clip": {
  "dir": "/sakuramoon-runtime/model/clip-vit-large-patch14-336",
  "note": "project-validated local weights, same loading path as first audit (local_files_only)"
 },
 "checkpoints": {
  "PRE": "/sakuramoon-runtime/output_model/g1/ckpt_116100_raw-116100-update-cadence",
  "MID": "/tmp/camera-p25-expanded/ckpt-restore/s0/ckpt_117100_raw-117100-update-cadence",
  "POST": "/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence",
  "note": "MID restored from hub leafmoone/sm_train_state s0/ (canary publisher artifact); integrity: size + LFS sha256 vs hub API + identity.update==117100 + resolved_config.toml sha256 must equal matrix anchor 933b9ad9c80cee6f..."
 },
 "resolved_config_anchor_sha256": "933b9ad9c80cee6fc315f604512dd3234e74ebe1cf0169c1dfbd97290067562a",
 "resolved_config_anchor_all_18_chunk_manifests_match": true,
 "checkpoint_path_all_18_chunk_manifests_match": true,
 "prompt_manifests": {
  "chunk_01.json": "2836a91ae8e8ee2e",
  "chunk_02.json": "9920184bdccdff6b",
  "chunk_03.json": "91ac94a520c76f73",
  "chunk_04.json": "1254c71f8f302123",
  "chunk_05.json": "4dce90aa360a1743",
  "chunk_06.json": "063e53a221fcc417"
 },
 "prompt_selection_sha16": "bf8067c4eba32686",
 "design_json_sha16": "43105306e2b7b649",
 "noise_hashes_sha16": "1b113e34874388d3",
 "noise_protocol": "CLI normalizes every case to unified 1:1 resolution 256x256 before render (camera_eval.py '统一 1:1 生成分辨率'); then frozen C2 _initial_noise: per-case fresh torch.Generator(device=cuda).manual_seed(case.seed); torch.randn((128, 256//16, 256//16), generator, cuda, fp32) = (128,16,16); sha256 of contiguous bytes. Native case h/w (e.g. 208x320) is NOT used for noise.",
 "camera_point_geometry_sha16": "5b0b5dc9d00e0384 (points_geometry inside render-integrity.json)",
 "render_manifests": {
  "PRE/chunk-01": "3d80ea54a26edee5",
  "PRE/chunk-02": "fc39bc5c5bc6c51f",
  "PRE/chunk-03": "c635576643fdb129",
  "PRE/chunk-04": "0c0b8373915cfe4c",
  "PRE/chunk-05": "91cb65ec6882f11d",
  "PRE/chunk-06": "3ab5279ff4c5ceea",
  "MID/chunk-01": "9a3f3d14e719c4f8",
  "MID/chunk-02": "6d5dc7e15d86fdb8",
  "MID/chunk-03": "a0b9130d9eb1cea2",
  "MID/chunk-04": "5282af1f84221ed3",
  "MID/chunk-05": "ae24bd5c2d2d1498",
  "MID/chunk-06": "42f1d6d74284332e",
  "POST/chunk-01": "caa4752e545fd265",
  "POST/chunk-02": "1e77be2cc411de90",
  "POST/chunk-03": "9d301a0143757fb8",
  "POST/chunk-04": "760ba27895c1b41b",
  "POST/chunk-05": "2f54ff1e5a25b30d",
  "POST/chunk-06": "07ddea577077ab31"
 },
 "render_integrity_sha16": "5b0b5dc9d00e0384",
 "feature_cache": {
  "dir": "/tmp/camera-p25-expanded/features",
  "pe_files": 2808,
  "clip_files": 2808,
  "manifest_sha256": "97e71a6eebc2339cd1e7fdca51035f2549fb4112542844498e86c7c6f08a07cb",
  "note": "per-image sha256-keyed; manifest (name,size list) sha256 = feature_cache.manifest_sha256; listing at analysis/feature-cache-manifest.json"
 },
 "bootstrap": {
  "seed": 20260905,
  "n": 10000,
  "unit": "prompt (top cluster) -> seed (within cluster) -> paired points retained",
  "secondary": "prompt x seed iid cluster bootstrap"
 },
 "legacy_protocol_continuity": {
  "new_12case_chunk_vs_old_4case_matrix": "0/52 byte-exact (expected: batch-context dependent numerics)",
  "matched_batch_context_rerender": "52/52 byte-exact (validation-prompts.json count=4, 13 points, PRE ckpt) -> BIT_EXACT_REPRODUCTION=PASS under matched batch context",
  "root_cause": "batched CFG DiT forward is numerically batch-size dependent on HCU; per-case results depend on co-batched cases; deterministic within a fixed batch composition (A/B re-render byte-identical; HCU0==HCU1)",
  "consequence": "old/new statistics NOT mixed; legacy4 replication assessed within the new 12-case context; internal PRE/MID/POST pairing valid (identical 12-case chunks across checkpoints)"
 }
}
```

## 19. Git immutability (§41)

- pre-report git status: `?? reports/camera-p25-effectiveness-audit.json | ?? reports/camera-p25-effectiveness-audit.md | ?? reports/camera-p25-effectiveness-copy-report.md | ?? reports/camera-p25-effectiveness-metrics.json | ?? reports/camera-p25-effectiveness-points.csv | ?? reports/camera-p25-expanded-effectiveness-audit.json | ?? reports/camera-p25-expanded-effectiveness-audit.md | ?? reports/camera-p25-expanded-effectiveness-copy-report.md | ?? reports/camera-p25-expanded-effectiveness-metrics.json | ?? reports/camera-p25-expanded-effectiveness-points.csv | ?? reports/camera-p25-expanded-effectiveness-prompts.csv`
- post-report git status: see analysis/git-status.json; src/tests/config = zero modifications; reports UNSTAGED; no commit; no push.

## 20. Task questions A-E (explicit answers)

- **A. 4-case power vs true null**: no detected stable gain at this evaluation scale (aggregate shift dir POST-PRE delta = -0.0122, CI [-0.0796, 0.0561] includes 0; per spec 33 this is 'no detected stable gain at this evaluation scale' with effect size reported, NOT proof of null). The 24-prompt x 3-seed matrix (2808 imgs, 72 independent units) is the higher-power read replacing the first audit's 4-case picture: the shift-dir CI width shrinks from ~0.63 (4-case audit, CI [-0.306, +0.320]) to 0.14 here.
- **B. case2 content drop specific vs reproducible**: REPRODUCED (rule: all-seed mean d=-0.032 < -0.02) and case2-SPECIFIC. Bank seed v0 (the seed the first audit used): 0.7789 -> 0.6941 (d=-0.085; first audit: 0.781->0.70). New seeds: v1 d=0.017, v2 d=-0.027. Strata means all within +/-0.02 => specific to case2, not composition-wide; SEED-DEPENDENT (weaker on new seeds).
- **C. right-edge regression specific vs systematic**: bank seed v0 REPRODUCES the first-audit pattern (case2 d=-0.373, case3 d=-0.697 — the two worst on v0, as in the 4-case audit). New seeds break the pattern: v1 (case1..4) d = [-0.613, 0.515, -0.016, -1.507], v2 d = [0.201, 0.393, 0.063, -0.303] (case4 worst on new seeds); all-seed mean (case1..4) = [-0.175, 0.178, -0.217, -0.630]. Edge-asymmetry gate = INCONCLUSIVE (12/24 prompts right-worse, (L-R) CI includes 0). => real at the bank seed, NOT seed-robust — prompt/seed-specific, not a structural left/right asymmetry.
- **D. PRE ceiling coverage**: NOT confirmed as a differential-gain ceiling: no tertile shows a CI-backed gain (low d=0.0126 CI[-0.1181, 0.1418], mid d=-0.0590 CI[-0.1724, 0.0670], high d=0.0097 CI[-0.0729, 0.0883] — all CIs include 0). The high tertile is simply FLAT (starts near the ceiling, dir PRE=0.7857) — the ceiling exists as 'no room to gain', but the expected low-group compensating gain is not detected.
- **E. low-PRE-response gain**: NO detectable gain: low tertile dir d=0.0126 CI[-0.1181, 0.1418] (includes 0); zoom-err d=-0.0099 CI[-0.0444, 0.0246] (includes 0). => the exposure-only hypothesis is NOT supported by the expanded matrix.
