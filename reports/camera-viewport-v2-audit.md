# SakuraMoon Camera Viewport v2 — Phase C1 Audit

Scope: Phase C1 (contract + implementation + tests + audits). C2 (production
cutover integration) is explicitly OUT of scope and was not performed.

- BASE: origin/dev = 3a341c0efa8aa6c82b41e508cf5fa2730e20fddb (expected == actual)
- BRANCH: camera-v2
- WORKTREE: /sakuramoon-runtime/sakuramoon-camera-v2
- HDM reference commit (read-only): 5fef7c4b71fe8386b497176021fe458810fdb7c0
- F3 context: F3 semantic commit 147355d, remote F3 head b4f3709,
  F3 verdict F3_CANARY_PASS_RESUME_MAIN_TRAINING, F3 inherited from base = YES,
  Camera changes to F3 = NO.

## HDM EVIDENCE STATUS:

- public code = NO (HDM public code was NOT used as a correctness oracle)
- oracle = NO (HDM code is a historical/public-mechanism reference only)
- conflicts found = true (the 7 discrepancies below)
- private behavior = UNKNOWN (HDM private production behavior is not observable)
- clean-room = YES (SakuraMoon Camera C1 is a clean-room implementation;
  HDM code was read for reference only, no code was copied)
- invariants authoritative = YES (SakuraMoon invariants, not HDM behavior,
  are the authority for this implementation)
- exact parity claimed = NO (no exact HDM parity is claimed, visual or otherwise)

## HDM report/source discrepancies (7)

1. The initial HDM 256 config references
   `hdm.data.danbooru512.Danbooru512Dataset`, which does not exist in the
   current public repo (exact initial dataset source unavailable).
2. The HDM TechReport's square-root-area normalization description does not
   fully match the current public HDM `bounding_box` implementation.
3. The current public HDM Kohya crop `randint` does not include the last
   legal offset (endpoint unreachable in the public code).
4. HDM pixel crop offset vs position offset `// vae_scale` can produce an
   alignment error of up to `vae_scale - 1` pixels.
5. HDM training Kohya `addon_info` is linear W/H, while the Diffusers
   pipeline currently has a `log(W/H)` path (aspect-conditioning mismatch).
6. Recent HDM history once fixed the training preview position/aspect path
   (a historical fix not reflected in all public artifacts).
7. HDM AxialRoPE frequencies are a trainable Parameter; SakuraMoon RoPE is
   fixed (not trainable). SakuraMoon does NOT adopt the trainable behavior.

None of the 7 discrepancies was copied. Where a public HDM bug exists (e.g.
the missing last offset), SakuraMoon deliberately does the correct thing
(inclusive offset, endpoint reachability proven by audit + brute-force test).

## Mandatory disclosures

- HDM exact initial dataset source unavailable (see discrepancy 1).
- HDM report/source discrepancies (the 7 above).
- Clean-room implementation (code copied = NO).
- No architecture change (DiT / attention / GlobalConditioner unchanged).
- No parameter change (parameters added = 0).
- No checkpoint schema change.
- No optimizer change.
- Current SakuraMoon VAE is x16 vs HDM x8: latent center-shift units are
  `abs_pixel_shift / 16.0` (VAE x16), NOT x8.
- Fixed SakuraMoon RoPE vs trainable HDM RoPE: no RoPE frequency is trained.
- p25 NOT deployed (canary requires C1 PASS + explicit user GO + C2).
- p50 LOCKED.
- Caption/crop mismatch remains a scientific risk: the camera crops the
  image while the caption describes the full image; this is an accepted
  open research risk, not a correctness defect of C1.
- Exact HDM visual parity NOT claimed.

## Camera geometry (mode = hdm_shifted_square_v2)

- viewport = stage-scaled square edge R = 512 (17-bucket vocab, G1 stage).
- ordinary admission preserved (assign_bucket first; camera never re-admits).
- no upscale (short edge only ever reduced to R; sub-stage sources are
  rejected by the ordinary `no_upscale` gate and never reach the camera).
- min zoom = 1.10, max zoom = 1.50 (equivalent zoom band).
- retention floor = 1/2.25 = 0.4444 (R^2 / (full_w * full_h) >= floor).
- inclusive offset: uniform inclusive `randrange(available + 1)` (both
  endpoints reachable — proven by endpoint probe + brute-force).
- horizontal: x_shift = 2*left/R + 1 - full_w/R, y_shift = 0; vertical symmetric.
- fallback: 7 fixed reasons (none / not_selected / short_edge_too_small /
  near_square_below_min / aspect_above_max / quantized_no_effect /
  no_square_bucket). `quantized_no_effect` is provably unreachable at R=512
  (half-up quantized rounding cannot escape the zoom band — brute-force
  tested in `tests/unit/data/test_camera_viewport.py`).

## Coordinates

- formula: `transform_camera_coordinates(base, *, zoom, x_shift, y_shift) =
  (base + [y_shift, x_shift]) / zoom` on FP32 [T,2] (y,x order); identity
  fast path returns the same tensor.
- training/full-canvas equivalence max error = 1.192e-07 (36-geometry probe,
  non-square canvases 640..1152 in both orientations, offsets 0..full-avail;
  unit tests assert <= 2e-6 for 1024x512 / 512x1024 offsets + boundary zooms).
- text anchor = 0 (packed layout: text coordinate 0).
- condition anchor = 0 (condition coordinate 0).
- x/y sign: +x/+y = increasing crop left/top (center shift is exactly 0.0).
- sub-latent precision: shifts are expressed in full-canvas pixels and in
  latent cells (/16.0, VAE x16); the affine is exact (single zoom + shift),
  no sub-latent rounding is introduced by the transform itself.

## Pipeline

- applied physical crop: one resize (source -> full canvas, LANCZOS) then one
  crop to the R x R viewport; audit `resized_*` = full canvas dims,
  `crop_box` = camera crop, `crop_policy = "camera_viewport"`.
- fallback bit-identical: absent / disabled / not-selected / infeasible paths
  are bit-identical to the pre-camera pipeline (tested).
- double resize: NO (performance audit: exactly 1 resize per applied image).
- EXIF: preserved by the existing decode/normalize path (unchanged).
- draft JPEG: unchanged (no new draft path introduced).
- transparent-white order: compositing precedes all crop paths (unchanged).
- MemoryError behavior preserved (no new allocation path of different class).

## Telemetry (schema 11)

- schema: 9 (legacy) -> 10 (spatial) -> 11 (camera-viewport C1).
- zoom bands: [1.10,1.20) / [1.20,1.35) / [1.35,1.501].
- shift-token bins: 6 fixed bins (token cells).
- band loss: 4 flat per-band fields
  `camera_{mild,medium,strong,ordinary}_{loss_sum,loss_count}`; the
  partition invariant mild+medium+strong+ordinary == effective_batch always
  holds (observer always populates; band -1 maps to ordinary).
- legacy zero semantics: with no camera activity, every camera field is 0
  except `camera_ordinary_loss_count == effective_batch` and
  `camera_ordinary_loss_sum == JLT main loss`; fallback table
  `none == effective_batch`. W&B exposes flat auto fields plus namespaced
  `camera_fallback_reasons/`, `camera_orientation_counts/`,
  `camera_zoom_histogram/`, `camera_shift_token_histogram/`.

## Config

- optional table: `data.camera_viewport` (StrictModel; 8 fields; validators:
  min<max, max<=1.5, enabled=>p>0, disabled=>p==0; mutual exclusion with
  `data.spatial_crop`).
- legacy resolved TOML byte-identical = YES
  (sha256 prefix 833850a7d63f6e79c76bc46330c5d06be /
  f285b635a60ea427c17f5ba70accd4b, split constant in the test).
- v1 spatial preserved = YES (spatial tests updated only for schema-11
  ordinary-band population; spatial semantics unchanged).
- mutual exclusion = YES (camera + spatial both enabled is rejected;
  production chain already has spatial enabled, camera configs disable it).
- p25 = config/train_g1_camera_v2_p25.toml (p=0.25, NOT deployed).
- p50 = config/train_g1_camera_v2_p50.toml (p=0.50, LOCKED).

## Distribution (synthetic audit, seed 20260905)

- synthetic samples = 1,000,000 (log-uniform aspect [1,4], short edge
  [256,2048], 50/50 orientation) through the REAL assign_bucket (17 buckets
  @512, retention 0.8) + planner.
- admitted = 932,773 (93.28%); selected = 232,882 (rate 0.2497 vs p=0.25,
  binomial 5-sigma PASS); applied = 95,952 (9.60% of total, 10.29% of
  admitted).
- mild/medium/strong (zoom bands) = 27,202 / 36,376 / 32,374 (ordinary =
  863,819).
- latent shift p50/p90/max = 4.375 / 11.6875 / 20.0 (pixel p50/p90/max =
  70 / 187 / 320).
- square bucket delta = 4.16% -> 14.44% (of admitted).
- real metadata audit = PENDING (synthetic audit only; the
  `--real-metadata` tar-JSON scan was not run on production shards in C1).
- invariants (8/8 True): crop_inside_full_canvas, no_upscale,
  no_distortion_except_integer_rounding, zoom_in_band,
  retention_above_floor, all_applied_outputs_square,
  ordinary_fallback_unchanged, inclusive_endpoints_reachable.

## Performance (CPU, PIL resize/crop instrumented)

- ordinary = 1 resize per image (13 resizes over 13 timed calls) for all
  geometries; sub-stage 256px sources rejected `no_upscale` (documented).
- v1 (spatial/ordinary path) = the ordinary column (no camera):
  512 square ~0.9-7.7 ms, 2:1 geometries ~13-49 ms (first geometry is a cold
  measurement).
- camera applied = exactly 1.0 resize per image (no double resize) for all
  applied 2:1 geometries; applied-path cost ~0.35-9.1 ms
  (first measurement cold).
- camera p25 / p50 = expected amortized overhead at p=0.25 / p=0.50 is
  p x applied_fraction (10.29%) x per-image delta — sub-millisecond per
  effective image at p25.
- no_double_resize = True.

## Camera eval suite

- 23-point manifest (legacy 1 + hdm_demo 6 + training_coupled 12 +
  diagonal 4); all 5 manifest classes used; classifier rules pinned by
  `tests/unit/cli/test_camera_eval_manifest.py`.

## Transition contract (spec section 22)

- `_record_data_policy_resume_transition` records `camera_viewport`
  (model_dump json) + skip_if_duplicate_of_last; resolved-TOML sidecar diff
  recorded as `resolved_config_changed_toml_paths`; artifact kind
  `sakuramoon.data_policy_transition.v1`.
- cutover diff allowlist (test-enforced): production sidecar -> p25 changes
  exactly `data.spatial_crop.enabled` + `data.camera_viewport.*`; every
  changed path is within the allowed prefixes
  (`data.spatial_crop.`, `data.camera_viewport.`, `run.`, `paths.`,
  `logging.`, `wandb.`, `evaluation.`); no forbidden prefix touched
  (optimizer/stage/model/objective/sampling/timestep/gpu/data.caption_dropout/
  data.image./data.buckets/data.spatial_crop.zoom).

## Tests

- ruff: clean on the C1 diff surface. The single remaining flag in the tree
  (`tests/gpu/optim/cmuon_capsule_teardown.py` EXE001) is a pre-existing
  baseline file not touched by C1.
- pyright: controlled comparison against a full 3a341c0 worktree — every
  file NOT modified by C1 has byte-identical per-file error counts; C1 files
  carry only the same error classes already present in their files'
  established patterns (metrics.py camera section mirrors the existing
  spatial section's `Mapping | None` fill; camera_eval's
  Module->TrainableComposite mirrors the baseline generation_eval; residual
  torch/numpy stub noise).
- targeted: 78 new camera-viewport tests (7 files) green; the 10
  pre-existing telemetry/spatial tests updated for the schema-11 ordinary
  band are green (28-test affected set green).
- full pytest: see C1 final report (parity vs baseline 9F + 1E).
- baseline failures: 9 failed + 1 collection error (onnxruntime import),
  all pre-existing and shared.
- new functional failures: 0 (after the 10 expected schema-11 test updates).
- new skips: 0.

## Commits

- C1_DATA_COMMIT: (see final report)
- C1_TELEMETRY_COMMIT: (see final report)
- No push. No production touch. No p25 run.

## Pipeline target fix (C1 internal finding)

`PipelineSample.target_height/target_width` are set to the actual crop-box
size (from `audit.crop_box`) for every path (ordinary / spatial / camera).
Rationale: the collate shape check and the runtime full-canvas invariant
`(bottom - top, right - left) == (target_height, target_width)` require the
target to equal the crop box; for a camera-applied 512x512 image inside an
ordinary non-square bucket the old value (assignment.bucket.height/width)
would have violated the invariant. All three paths agree by construction
after the fix (ordinary/spatial crop boxes already equal their bucket size).
