# SakuraMoon Camera Viewport v2 — Phase C1 Audit

base = 3a341c0efa8aa6c82b41e508cf5fa2730e20fddb (origin/dev, expected == actual)
branch = camera-v2
worktree = /sakuramoon-runtime/sakuramoon-camera-v2
HEAD = 3a341c0 -> 88250e9 (C1 data) -> 75e6b6d (C1 telemetry) ->
       fix commit "fix: scale camera viewport to active stage resolution"
       (this audit's revision; hash reported in the C1 final report)

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

## Current-stage bucket provenance (2026-09-05 correction audit)

Correction gate: the 512 in base.toml
(`data.buckets.base_area_px = 262144 = 512^2`) is the **512-equivalent BASE
bucket family definition only**, not the current G1 target edge. The
current G1 stage is `config/train_g1.toml [stage].resolution = 256`.

Runtime call chain (PROVEN correct, no code change needed):

    config/train_g1.toml:86  resolution = 256
    config/base.toml:95      base_area_px = 262144 (512-equivalent base family)
    src/sakuramoon/data/production.py:808-811
        buckets = scale_buckets(
            generate_base_buckets(self.config.data.buckets),
            self.config.stage.resolution,        # = 256 at current G1
        )
    src/sakuramoon/data/production.py:821-836
        WebDatasetPipeline(buckets=buckets, ..., camera_policy=camera_policy)
    src/sakuramoon/data/pipeline.py:446
        self._camera_stage_edge = camera_stage_edge(buckets)
    src/sakuramoon/data/pipeline.py:605-613
        plan_camera_viewport(..., buckets=self.buckets,
                             stage_edge=self._camera_stage_edge, ...)
    src/sakuramoon/data/camera_viewport.py:382-383
        viewport = discover_square_bucket(buckets, stage_edge=stage_edge)

`discover_square_bucket` (camera_viewport.py:131-148) is vocabulary-driven
and fail-closed: exactly one square bucket must exist in the vocabulary
passed to the planner AND it must equal the claimed stage edge on both
edges; otherwise the plan falls back to `no_square_bucket`. No code path
infers R from `data.buckets.base_area_px`, and no code path feeds the
unscaled 512-equivalent base family to the planner as the current
training vocabulary.

Answers to the correction audit questions:

- A. The first C1 report wrote "viewport = 512" because the offline
  distribution/performance audits were run with `--stage-edge 512` (the
  audit script default at the time) and the report inherited those
  numbers; the unit tests use 512 as a stage-vocabulary geometry constant.
  The runtime was never 512 at G1.
- B. The runtime receives the SCALED current-stage vocabulary
  (production.py:808-811 -> pipeline:446), not the unscaled base family.
- C. p25/p50 configs do not override `[stage]`; they inherit
  `resolution = 256`, so runtime R resolves to **256** at current G1.
- D. The offline distribution audit got R=512 because it was invoked with
  `--stage-edge 512` (script default at the time); corrected: the default
  is now 256 and the re-run used the current G1 stage explicitly.
- E. "256px sub-stage source = no_upscale" came from
  `src/sakuramoon/data/buckets.py:221-244` (`assign_bucket`: no-upscale,
  nearest-aspect, cover, then retention, in locked order;
  `return BucketRejection("no_upscale")` at line 244). That rejection is
  stage-relative: a 256px source is sub-stage ONLY at stage edge 512. At
  current G1 (256) a 256x256 source is ordinarily admitted (256x256
  bucket, retention 1.0) and the camera falls back with
  `near_square_below_min`; a 128px source is the sub-stage case at G1.

Effect of the correction: only the offline audit script defaults, the
offline audit outputs, this report, and the test coverage (new
stage-scaling tests) changed. The frozen geometry semantics (z formula,
shift formula, coordinate transform, inclusive endpoints, ordinary
admission, RNG domains, no-upscale rule, fallback semantics, telemetry
schema, architecture zero-diff, optimizer/CMuon zero-diff) are untouched.

## Camera geometry (mode = hdm_shifted_square_v2)

- R = width of the unique square bucket of the CURRENT-STAGE vocabulary
  `scale_buckets(generate_base_buckets(data.buckets), stage.resolution)`;
  discovered vocabulary-driven and fail-closed (no hardcoded R).
- Current G1: `stage.resolution = 256` => **R = 256** (square 256x256).
  The 512-equivalent base edge was NOT used as the runtime edge (NO).
- Future stages (same planner, verified): 512 => square 512, 768 =>
  square 768, 1024 => square 1024.
- viewport (current G1) = 256x256.
- ordinary admission preserved = YES; no upscale = YES.
- min zoom = 1.10 / max zoom = 1.50 (z^2 = full_w*full_h / R^2, aspect
  capped at 2.25 in either stage).
- retention floor = 1/2.25 = 0.4444.
- inclusive offset = YES (uniform inclusive randrange, both endpoints
  reachable — proven by probe and brute-force test).
- horizontal x_shift = 2*left/R + 1 - full_w/R; vertical symmetric.
- fallback = 7 fixed reasons (`quantized_no_effect` unreachable at
  R=256 and at R=512; brute-force proof at R=512 retained).
- Current G1 feasibility (new tests): 512x256 -> full canvas 512x256,
  viewport 256x256, z = sqrt(2), applied (bucket 368x176, retention vs
  resized ~0.96); 256x512 vertical mirror; 384x256 / 256x384 applied
  (z = sqrt(1.5)); 256x256 admitted and falls back
  `near_square_below_min`; 128x128 ordinarily rejected `no_upscale`.

## Coordinates

- formula = (base + [y_shift, x_shift]) / zoom on FP32 [T,2] (y,x order),
  identity fast path returns the input tensor.
- training/full-canvas equivalence max error = 1.192e-07 at BOTH stage
  edges: 36-geometry probes at R=512 and R=256 (non-square canvases,
  offsets 0..full-avail both orientations); unit tests tolerance 2e-6.
- text anchor = 0 / condition anchor = 0.
- x/y sign = +x/+y = increasing crop left/top; center shift exactly 0.0.
- sub-latent precision = affine exact; latent units = pixels / 16.0
  (VAE x16, not HDM x8).

## Pipeline

- applied physical crop = one resize (source -> full canvas, LANCZOS) +
  one R x R crop; audit.resized_* = full canvas, crop_box = camera crop
  (reuses the existing full-canvas mechanism, zero new runtime wiring).
- fallback bit-identical = YES (absent/disabled/not-selected/infeasible
  all test-locked).
- double resize = NO (perf audit: exactly 1 resize per applied image at
  both audited stages).
- EXIF preserved by the existing decode path / draft JPEG unchanged.
- transparent-white order = compositing precedes all crop paths (unchanged).
- MemoryError behavior preserved = YES.
- target_* = crop-box size (collate shape check + runtime full-canvas
  invariant require it; consistent by construction on all paths).

## Telemetry (schema 11)

- schema = 11 (9 -> 10 spatial -> 11 camera).
- zoom bands = [1.10,1.20) / [1.20,1.35) / [1.35,1.501].
- shift-token bins = 6 fixed.
- band loss = 4 flat field pairs
  `camera_{mild,medium,strong,ordinary}_{loss_sum,loss_count}`; partition
  invariant mild+medium+strong+ordinary == effective_batch always holds.
- legacy zero semantics = all camera fields 0; only the ordinary band
  carries the whole effective batch (count = effective_batch, sum = JLT
  main loss); W&B flat + 4 namespaced tables.

## Config

- optional table = `data.camera_viewport` (StrictModel, 8 fields;
  min<max / max<=1.5 / enabled=>p>0 / disabled=>p=0).
- legacy resolved TOML byte-identical = YES
  (sha256 833850a7d63f6e79c76bc46330c5d06be
  + f285b635a60ea427c17f5ba70accd4b).
- v1 spatial preserved = YES; camera+spatial mutual exclusion = YES.
- p25 = config/train_g1_camera_v2_p25.toml (NOT deployed);
  p50 = config/train_g1_camera_v2_p50.toml (LOCKED).
  Both inherit [stage].resolution (256 at current G1); neither hardcodes
  a viewport.

## Distribution (synthetic audit, seed 20260905, 1,000,000 samples)

CURRENT G1 stage (stage_edge = 256, real assign_bucket + planner):

- admitted = 1,000,000 / 1,000,000 (100%; the 256 family admits all
  synthetic sources, including 256px sources that the 512 family rejects)
- selected = 249,762 => 0.249762 vs p = 0.25 (binomial 5-sigma PASS,
  sigma = 433.0, observed-expected = -238.0)
- applied = 112,123 => 11.21% of total (= of admitted)
- zoom bands: mild 31,503 / medium 42,673 / strong 37,947
- fallback reasons: none 112,123 / not_selected 750,238 /
  near_square_below_min 34,326 / aspect_above_max 103,313 /
  short_edge_too_small 0 / quantized_no_effect 0 / no_square_bucket 0
- applied orientation: horizontal 55,926 / vertical 56,197
- latent shift (tokens, /16.0): p50 2.21875 / p90 5.84375 / max 10.0
- pixel shift: mean_abs 43.47 / p50 35.5 / p90 93.5 / max 160
- square bucket share: 4.53% -> 15.74%
- zoom mean 1.289 / max 1.5; shift-token bins: [0,1) 25,712 / [1,2) 25,506
  / [2,4) 33,522 / [4,8) 25,134 / [8,16) 2,249 / [16,inf) 0
- inclusive endpoints reachable: zero 159 / full 172
- invariants 8/8 True; real metadata audit = PENDING (the --real-metadata
  hook is implemented; C1 is synthetic only)

The old R=512 distribution numbers (admitted 932,773 / applied 95,952)
were a FUTURE-STAGE-512 measurement and are NOT a basis for current G1 p25.

## Performance (CPU, PIL resize/crop instrumented; WARMUP=3 + REPEATS=10)

CURRENT_G1 (stage_edge = 256):

| geometry / format   | ordinary bucket | camera   | full canvas | ordinary ms | camera ms | resize/image |
|---------------------|-----------------|----------|-------------|-------------|-----------|--------------|
| square_128 (JPEG/PNG) 128x128 | — | REJECTED no_upscale (sub-stage, never reaches camera) |
| square_256 (JPEG/PNG) 256x256 | 256x256 | not applied | — | 0.11-0.12 | 0.0 | — |
| wide_2to1 (JPEG/PNG) 512x256 | 368x176 | applied | 512x256 | 3.39-3.41 | 0.107-0.109 | 1.0 |
| tall_2to1 (JPEG/PNG) 256x512 | 176x368 | applied | 256x512 | 3.38-3.39 | 0.109 | 1.0 |
| wide_3to2 (JPEG/PNG) 384x256 | 320x208 | applied | 384x256 | 2.74-2.96 | 0.092-0.093 | 1.0 |
| tall_3to2 (JPEG/PNG) 256x384 | 208x320 | applied | 256x384 | 2.72-2.73 | 0.090-0.093 | 1.0 |

FUTURE_STAGE_512 (stage_edge = 512, explicitly NOT current G1):

| geometry / format   | ordinary bucket | camera   | full canvas | ordinary ms | camera ms | resize/image |
|---------------------|-----------------|----------|-------------|-------------|-----------|--------------|
| square_256 (JPEG/PNG) 256x256 | — | REJECTED no_upscale (sub-stage at 512 only) |
| square_512 (JPEG/PNG) 512x512 | 512x512 | not applied | — | 0.90-1.13 | 0.0 | — |
| wide_2to1 (JPEG/PNG) 1024x512 | 736x352 | applied | 1024x512 | 13.0-14.6 | 0.33-1.43 (first = cold) | 1.0 |
| tall_2to1 (JPEG/PNG) 512x1024 | 352x736 | applied | 512x1024 | 12.97-13.04 | 0.33-0.34 | 1.0 |

- no_double_resize = True at both stages (applied path: exactly 1.0
  resize per image).
- camera p25 amortized cost (current G1) = 0.25 x 11.21% x ~0.11 ms
  per admitted image ~ 0.003 ms per effective batch image.

## Camera eval suite

- manifest = 23 fixed points, 5 classes (in_distribution_coupled,
  interpolation, extrapolation, zoom_out_extrapolation,
  decoupled_shift_extrapolation); sign convention pinned; PNGs + manifest
  JSON per checkpoint.
- resolution = `config.stage.resolution` (camera_eval.py:286), i.e. 256 at
  current G1 — stage-driven, not hardcoded.

## Transition contract (spec section 22)

- record = data_policy_transition.json (sakuramoon.data_policy_transition.v1)
- cutover diff = data.spatial_crop.enabled + data.camera_viewport.* only
- allowlist enforced by test; no forbidden prefixes touched.

## Tests

- new camera tests = 88 across 8 files (78 from C1 + 10 new
  stage-scaling tests: current-G1 config fact 256; stage vocabulary
  square discovery at 256/512/768/1024; fail-closed vocabulary/stage
  mismatch; G1 feasibility of 512x256 / 256x512 / 384x256 / 256x384;
  G1 256x256 not-upscaled fallback; G1 128x128 ordinary no_upscale;
  pipeline-level G1 512x256 -> 256x256 viewport).
- affected pre-existing tests updated = 10 (schema-11 ordinary band).
- ruff = clean on the C1 diff surface (only the pre-existing baseline
  flag in tests/gpu/optim/cmuon_capsule_teardown.py remains, untouched).
- pyright = controlled per-file comparison vs the full 3a341c0 worktree
  (/tmp/c1-wt), verified in two independent environments:
  (a) codex-session run (recorded with 103b12b): tree totals 820 on
      both sides; every non-C1 file identical; the initial +27
      camera-attributable delta (camera_eval +1, metrics +24, observer
      +2, all in src) closed in 103b12b by typing-only changes:
      optional camera tables narrowed through unreachable-None locals +
      a _required_camera_table helper, the DTK tolist() stubs boundary
      replaced by a single .cpu() transfer + per-element .item(), and a
      local cast(TrainableComposite) at the camera_eval load boundary.
      No pyright ignores added (two pre-existing targeted ignores in the
      camera observer path were removed); no config/strictness changes.
  (b) final closure re-verification (documented environment): pyright
      1.1.411 (DTK venv CLI) with the repo pyproject strict config;
      import resolution through the workspace .venv ->
      /sakuramoon-runtime/venv-pyright-union (472 package symlinks: DTK
      site-packages first, system /usr/local site-packages fill;
      required because the DTK venv lacks torch/PIL while the system
      python lacks pytest, and pyright does not honor
      include-system-site-packages). Tree totals 3296 on both sides;
      all 103 non-C1 files byte-identical per-file (line, rule,
      message) = 0 mismatches; C1-added/modified files = 0 new
      diagnostics (message-level multiset diff). In this environment
      the pre-closure C1 tree carried 124 new diagnostics, all in six
      C1 test files (BUCKETS/fixture typing, untyped lambdas, protected
      access, unnecessary casts, StageEdge literal, dict[str, object]
      **kwargs); all closed in C1_TYPES_TESTS_COMMIT by typing-only
      edits: explicit annotations, local casts, typed helper defs, and
      targeted reportPrivateUsage ignores matching the repo's
      established pattern.
- targeted suite (13 files, camera + stage-scaling + affected telemetry/
  spatial/pipeline) = 116 passed; re-verified on the final tree across
  14 files (superset incl. test_spatial_crop.py + test_pipeline.py)
  = 152 passed, 0 failed.
- full pytest (corrected tree) = 9 failed / 872 passed / 2
  skipped / 1 error (914.55s); 872 = 784 baseline + 78 C1 + 10
  stage-scaling; failure set BYTE-IDENTICAL to the baseline
  (9 failed + 1 onnxruntime collection error, all pre-existing); 0 new
  functional failures.
- full pytest (final tree, after the types commits,
  --continue-on-collection-errors) = 9 failed / 872 passed / 2
  skipped / 1 error (926.36s); the 10 failed/errored items are
  byte-identical to the dev@3a341c0 baseline run (9 failed + 1
  onnxruntime collection error, all pre-existing); 872 = 784 baseline
  + 88 C1; 0 new functional failures, 0 new skips.
- new functional failures = 0 (verified on the corrected tree).
- new skips = 0.

## Commits

- C1_DATA_COMMIT = 88250e9 (data: add HDM-style shifted-square camera viewport)
- C1_TELEMETRY_COMMIT = 75e6b6d (telemetry: add camera viewport observability and evaluation suite)
- C1_CORRECTION_COMMIT = 8b525ef
  (fix: scale camera viewport to active stage resolution)
- C1_TYPES_COMMIT = 103b12b (types: close camera viewport C1
  type-check gate; src typing-only narrowing + this audit update)
- C1_TYPES_TESTS_COMMIT = this commit
  (types: close camera viewport C1 test-file type-check gate;
  typing-only cleanup of the six C1 test files + final audit numbers)
- No push; no origin/camera-v2 remote branch; `model` and `.venv`
  symlinks untracked.

## Pipeline target fix (C1 internal finding)

`target_height`/`target_width` = crop-box size (the camera crop), required
by the collate shape check and the runtime full-canvas invariant;
consistent by construction across all three pipeline paths (ordinary /
spatial / camera). Discovered and fixed inside C1; no external consumer
was affected.
