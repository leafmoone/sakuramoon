# Vertical Bottom Supervision Audit (same-source mirrored viewport pairs)

Forward-only, read-only audit of whether the p25 camera stage's vertical
TOP-vs-BOTTOM loss preference is a **same-source geometric/directional effect**
or an artifact of the historical *natural cohort* (different sources / content
per TOP vs BOTTOM group).  Companion to the Camera Coordinate Causal audit
(`final_snapshot/` + posthoc V2, base `34f646abdb1e64f45dfddd47cf7e8b9247a7a979`).

This package adds **new files only**; it never modifies production
`src/`, `config/`, checkpoints, or prior evidence.

## Stages

| file | stage | device |
|---|---|---|
| `contracts.py` | frozen geometry / selection / seed / arm / margin / inference contracts (numpy + stdlib only) | CPU |
| `build_pairs.py` | pair candidate sweep, dedup, stratified selection, source extraction, **frozen pair manifest** | CPU |
| `encode_pairs.py` | production-equivalent canvas rebuild + crop + frozen Mage-VAE encoding; anchor reconstruction parity vs the frozen stage1 unit bundles | GPU |
| `score_pairs.py` | 2x2 coordinate factorial (TT/TB/BT/BB) + SAME duplicate arms (+ optional ID arms), 3 checkpoints, per-stratum seeded noise shared by TOP/BOTTOM | GPU x2 |
| `feature_audit.py` | PE-Spatial-B16-512 dense features + MNN correspondence (p25c), CLIP image/text features, production-verbatim caption reconstruction (fail-soft), content-balanced subset (written before analysis) | GPU |
| `analyze.py` | paired bootstrap inference (point = exact mean; CI = bootstrap only), strata, classification A-F, the 5 reports + contact sheets | CPU |
| `pe_ref/` | self-contained vendored PE tower (see below) | - |

Runtime artifacts live under `/tmp/camera-vertical-bottom/` (pair manifest,
latents + hash receipts, images, ledgers, features, bootstrap artifacts,
sheets). **Nothing under that path is staged into git.**

## `pe_ref/` provenance

- `config.py`, `rope.py`, `pe_vision.py` - vendored from
  `facebookresearch/perception_models` commit
  `3e352cca660658d4b5c90f42a7808b11469e4c66` (the exact copy vendored by the
  iprea branch at `src/sakuramoon/pe_spatial` in the come2 irepa-prod
  worktree, HEAD recorded there).  Class bodies are verbatim; import paths
  are re-based to `pe_ref`, the timm `DropPath` import re-based to the
  locally vendored `.drop_path`, unused imports trimmed, the huggingface_hub
  fetcher removed (`from_config` now requires a local `checkpoint_path`),
  and the SDPA call wrapped in a MATH-backend context (the DTK/ROCm torch
  build registers the flash SDPA backend without its kernel library).
  Apache-2.0 - see `LICENSE.PE`.
- `drop_path.py` - `timm/layers/drop_path.py::DropPath`, vendored because
  `timm` is not installed in the audit venv.  In PE-Spatial-B16-512
  `drop_path=0.0`, so the module loads but is never active.  MIT.
- Weights: frozen local asset
  `/sakuramoon-runtime/model/pe_spatial_b16_512/PE-Spatial-B16-512.pt`
  (sha256 `86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5`,
  345,783,707 bytes), strict-loaded with module./visual. key stripping.
- All vendored files carry `# ruff: noqa` (verbatim, not re-formatted).

## Pre-registered choices (no post-hoc tuning)

- **Pairs**: stage1 camera units, `orientation=vertical`, OPPOSITE
  applicable, `available = F-256 > 0`, anchor tertile START or END (V2
  tertile semantics on the stored rounded `norm_offset`); mirror
  `k_mirror = available - k` must land on the opposite side (exact integer
  test; an END anchor at exactly 2/3 mirrors to CENTER and is excluded);
  CENTER never enters the primary cohort.  Geometry invariant:
  `signed_shift(k, F) = k + 128 - F/2` equals the manifest `pixel_shift`
  EXACTLY (V2 offset-balance verified formula; the manifest stores rounded
  zoom/norm_offset, so no integrality check on the rounded value).
- **Dedup / selection**: unique by `(source_shard, sample_id)`, keep
  smallest unit id; strata = anchor side x zoom band (mild <1.20 <= medium
  <1.35 <= strong); proportional largest-remainder allocation; within-
  stratum order deterministic by sample key; target 512, minimum 256 else
  STOP.  Master/selection seed 20260907 (recorded; ordering is
  sample-key deterministic by construction).
- **Checkpoints**: PRE `g1/ckpt_116100`, MID `camera-coordinate-causal/
  ckpts/MID/ckpt_117100`, POST `g1_camera_v2_p25/ckpt_118100` - each gated
  (COMPLETE marker, checkpoint manifest sha, full model-tree sha vs the
  stage1 manifest evidence, alpha from growth_state.json).
- **Timesteps**: the 4 frozen JLT quantiles (10/35/65/90%) -
  `0.1388061520926247, 0.24819609741338816, 0.37948289980451305,
  0.5560734456280833` (production P_MEAN=-0.8, P_STD=0.8).
- **Noise**: one deterministic eps per (pair, stratum),
  `seed = 20260907*1_000_003 + pair_index*100 + stratum`, scale 1.0, shared
  by TOP and BOTTOM; identical inputs across PRE/MID/POST.
- **Arms**: TT/TB/BT/BB (2x2) + T_SAME/B_SAME duplicate forwards
  (numerics floor, reported as aggregate mean + CI only - never a hard
  gate) + optional T_ID/B_ID **iff** the 32-pair smoke measures
  <= 1.18 s/forward (timing gate, pre-registered).
- **Margins**: per (pair, ckpt) M_TOP = mean_k(L_TB - L_TT), M_BOTTOM =
  mean_k(L_BT - L_BB) over the 4 t-strata; per pair
  D_top[a->b] = (M_b - M_a) - (F_b - F_a) (SAME-corrected),
  G = D_top - D_bottom.  **Inference unit = source pair.**
- **Estimator** (the mirror-estimator correction): point =
  `mean(TOP - BOTTOM)` over paired values; paired bootstrap
  (seed 20260907, n=10000, shared source-pair index matrix) provides the
  95% CI only.  The old pooled-sign formula (single mean over the
  concatenated signed pool; 1.0 on the [1,2,3]/[0,0,0] fixture) is retained
  in `contracts.pooled_sign_mean` **as a regression reference only** and is
  never reported.  Unequal-count two-sample comparisons use
  `mean(low) - mean(high)`.
- **Content audit**: PE dense features on the full canvas (256xF resized
  256x256 LANCZOS, pre-registered) and each crop; p25c mutual
  nearest-neighbor patch matching (fp32 L2-normed cosine, deterministic
  argmax), retained = inlier_ratio * matched_cos.  CLIP ViT-L/14@336
  frozen fp32: sim(full, top/bottom).  CLIP text uses the **exact
  production caption** (shard metadata -> `parse_modelscope_caption_fields`
  -> `build_caption_plan(seed=_domain_seed(run.seed, stage, 0, sample_id,
  "caption"))` -> `serialize_caption` with the production tokenizer and
  framing contract), verified bit-exact against frozen unit bundles for
  32 pairs; any failure or missing dependency -> `CLIP_TEXT =
  NOT_AVAILABLE` (fail-soft, image+PE continue).
- **Content asymmetry**: `|CLIP text delta|` when text is available, else
  `|CLIP image delta| + |PE retained delta|` (TOP - BOTTOM; positive =
  BOTTOM preserves less).
- **Content-balanced subset**: lowest 50% by asymmetry (deterministic
  (asym, index) order); defined from content fields ONLY - the selector
  receives pair indices + one asymmetry scalar, so no causal field can
  enter; written to `features/content-balanced-subset.json` before any
  analysis.
- **Classification** (spec s45 tree): A same-source gap collapses (CI has
  0) while the natural-cohort gap is clear; B gap persists and the
  balanced-subset gap collapses (CI has 0) or attenuates >= 50%, or a
  consistent content relationship (|Spearman| >= 0.30) / systematic
  BOTTOM content loss (any pre-registered content delta CI > 0); C gap and
  balanced gap both CI > 0 with no content explanation (|Spearman| < 0.20
  and no systematic loss); D content difference present but directional
  gap persists after balancing; E no same-source gap and the natural
  result does not reproduce; F power/artifacts/metrics insufficient.
- **Recommendation mapping** (recommendation only; LONGER_P25_AUTHORIZED =
  NO, P50_AUTHORIZED = NO): A -> REVIEW_EVALUATION_COHORT, B ->
  REVIEW_CROP_CONTENT_SUPERVISION, C -> REVIEW_VERTICAL_COORDINATE_
  SUPERVISION, D -> REVIEW_CROP_AND_VERTICAL_SUPERVISION, E ->
  LONGER_P25_REVIEW_CANDIDATE, F -> NO_MORE_EXPOSURE_YET.
- **Prerequisite gate** (`validate_prior_numerics_v2`): frozen
  `posthoc-v2-microprobe.json` sha-checked (no re-run), requires
  `state.HIDDEN_MUTABLE_STATE_DETECTED == false` **and**
  `p3.state_history_effect == false` (P3 is wired into the gate; a true P3
  -> BLOCKED_NUMERICS) and the V2 review report
  `NUMERICS_V2_CLEAN == true` with verdict
  `POSITIVE_LOSS_PREFERENCE_LEARNING`.

## Red lines

forward-only (no training / backward / optimizer); no checkpoint writes; no
resume of 118100, no new exposures (P50 / longer p25) - the audit only
recommends; no new model downloads (existing local assets only); no PNG /
latent / weight / dataset files in git; `git diff 34f646ab..HEAD -- src
config` must stay EMPTY.
