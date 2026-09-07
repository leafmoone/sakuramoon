# SakuraMoon Camera v2 — Vertical Mirror-Balanced Supervision (V1) Design

- Repository: `leafmoone/sakuramoon`
- Base SHA: `ff020ad5ae02859f1c853b5ecb80fb3b523a4baf` (VCR review head)
- Branch: `camera-v2-mirror-balanced-supervision-design`
- Status: **design + minimal training path, REVIEW ONLY — nothing trained, nothing launched**

---

## CURRENT PROBLEM

Directional residual evidence (frozen, forward-only audits on the
camera-v2 review lineage; no production change in any of them):

The visual-content-residual audit (`reports/` of the VCR round, base
`f3d183a35b3123e0b9c2a301fbc68ff5a276d345`, created
`2026-09-07T08:47:18Z`, 512 frozen same-source pairs, 10000-iteration
bootstrap, seed 20260907, `directional_residual = CONFIRMED`) measured the
same-source TOP-vs-BOTTOM loss gap before and after visual-content
balancing:

| Quantity | Point | 95% CI |
| --- | --- | --- |
| Gap PRE→POST, ALL | +2.505e-3 | [1.920e-3, 3.167e-3] |
| Gap PRE→POST, V25 | +2.277e-3 | [1.325e-3, 3.338e-3] |
| Gap PRE→POST, V50 | +2.009e-3 | [1.341e-3, 2.714e-3] |
| CLIP-image signed gap, BOTTOM preserves more | +1.533e-3 | [6.541e-4, 2.455e-3] |
| Gap in `latent_ge4` subset | +1.253e-2 (mean, n=9) | — |

- Every gap CI excludes zero: the directional residual persists **after**
  visual-content balancing, so visual-content asymmetry alone does not
  explain the TOP-BOTTOM gap.
- The gap grows with camera shift severity (`latent_ge4` subset mean gap
  ≈ 1.25e-2 vs the overall ≈ 2.5e-3), i.e. the strongest-shifted vertical
  crops — the ones that cut off the most canvas — carry the most residual.
- `RETENTION_25 = 0.9089`: even the 25th-percentile-shift vertical crops
  retain only ~91% of the canvas; the un-retained band is always the
  single side away from the crop center.
- Audit recommendation (verbatim):
  `DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION_WITH_CONTENT_GUARDS`.

Interpretation recorded by the audit: a same-source crop and its mirror
share content but disagree on **which** canvas side is supervised; the
model is trained, in each vertical crop, preferentially toward the kept
side. Mirroring the crop and supervising both views with equal weight
removes the one-sided supervision bias without discarding any content and
without changing the camera policy itself.

## INTERVENTION

A new, **disabled-by-default** data-section activates vertical
mirror-paired supervision:

```toml
[data.camera_mirror_balance]
enabled = true                 # false by default (field omitted)
mode = "vertical_mirror_pair_v1"   # only accepted literal
min_latent_shift = 2.0         # exact value (ge = le = 2.0)
pair_probability = 1.0         # [0, 1]; 1.0 in the canary proposal
pair_weight = 1.0              # exact value (ge = le = 1.0)
```

Eligibility for one sample, evaluated per sample in the data pipeline:

- the camera viewport plan **applied** on this sample, and
- `orientation == "vertical"`, and
- the viewport edge is even (whole-pixel mirror geometry), and
- `latent_center_shift >= min_latent_shift` (2.0 in V1; equality exact).

Horizontal, fallback, near-square, not-selected, and low-shift
(`latent < 2.0`) samples are never mirrored. An enabled
`camera_mirror_balance` requires `camera_viewport.enabled` (config
validation); `pair_probability` draws one uniform in an **isolated RNG
domain `camera-mirror-policy`** so the caption/crop/camera-selection/
camera-offset streams are bit-identical with the policy on or off.

Mirror failures degrade the single sample to the ordinary path (audit
`mirror_degraded` trace) — a mirror failure never rejects a sample.

## GEOMETRY

For a vertical plan on a full canvas of height `F` with viewport height
`R` (both even; `available = F - R`):

- original crop top: `k` (0 ≤ k ≤ available)
- mirror crop top: `k_mirror = available - k`
- mirror crop box: `(0, k_mirror, R, k_mirror + R)` (left anchored 0,
  vertical plans are full-width)
- signed pixel-center shift: `signed(k) = (k + R/2) - F/2` (exact float;
  integer-valued for even `R`)
- **antisymmetry (v1 hard gate, verified exactly):**
  `signed(k_mirror) == -signed(k)` — enforced at plan time; a violation
  yields `None` (no mirror) instead of an invalid geometry.
- camera `shift_y(k_mirror) = 2·k_mirror/R + 1 - F/R`,
  normalized offset `k_mirror / available` (its own geometric value;
  equals `1 - offset` up to division rounding).

Both views are produced from **one** LANCZOS resize of the source into
the full canvas (no second decode, no second resize): the original crop
box and the mirror crop box are two windows of the same canvas. The
mirror image is stored as a uint8 CHW payload alongside the original
audit; it is never a new source sample.

## LOGICAL WEIGHT

A mirror pair is **one logical source sample**, not two. The physical
per-view loss `(L_original, L_mirror)` reduces to one logical per-sample
value:

```
L_pair = 0.5 * L_original + 0.5 * L_mirror      (pair_weight = 1.0)
L_ordinary = L_original                          (weight 1.0)
```

Proof of conservation: the existing step reduction is
`sum(per_sample) / B_logical` (step.py is **unchanged**); a pair
contributes `0.5·L_o + 0.5·L_m` — exactly the source weight one unpaired
sample carries, so the DDP/global denominator and the per-logical-sample
semantics are identical with or without mirroring. Gradient check:
`∂(0.5·x₀ + 0.5·x₁)/∂x = (0.5, 0.5)` — each physical view of a pair
receives exactly half the weight inside its logical sample.

Discriminating synthetic fixture (unit-tested): batch
`(ordinary, pair)` with physical losses `(10, 0, 2)` gives logical
`(10.0, 1.0)`, logical mean **5.5** — while the incorrect
all-physical-view mean is **4.0**. Batch A `(ordinary, ordinary)` and
batch B `(ordinary, pair)` both carry exactly two logical sample weights.

## TIMESTEP/NOISE

- Timesteps are drawn **once per logical sample**
  (`sample_jlt_timesteps(B_logical)`), then expanded to physical rows by
  `index_select(phys_to_log)`; a pair's two views read the same drawn
  value (unit-tested with `torch.equal`).
- Noise is sampled from the **logical** clean latents
  (`sample_noise(clean_logical)`) and expanded the same way, so both
  views of a pair are corrupted by the **identical** noise tensor
  (unit-tested with `torch.equal`; with clean=0 the two physical states
  are bit-identical).
- The pair-selection RNG lives in the isolated `camera-mirror-policy`
  domain; with the mirror policy disabled or with zero eligible samples,
  the logical timestep/noise streams are bit-identical to the
  pre-mirror path (unit-tested).

## CONDITIONING

- The Qwen encoder runs **once per logical sample**; conditioning rows
  (main/condition token indices, masks, token lengths, null-condition
  flags, active-condition indices) are row-duplicated by
  `index_select(phys_to_log)` — never re-encoded. Unit test: the encoder
  call counter stays at 1 for a logical batch; the number of unique
  conditioning rows equals the logical batch size, not the physical one.
- Mirror views reuse the **same** full-canvas coordinate math
  (`full_canvas_crop_coordinates`) with the mirror crop box; no new
  position encoding, no new camera embedding, no new direction token, no
  TOP/BOTTOM label is introduced.
- `rope.py` and all conditioning architecture are **unchanged**.

## COMPUTE

**Estimate** (not a measurement; derived from the frozen P25 per-sample
audit of 2560 units = 2048 camera + 512 ordinary,
`camera-coordinate-causal-units.csv`), at `pair_probability = 1.0`:

- Vertical among camera samples: 1557/2048 = 76.0%.
- Eligible (vertical ∧ even viewport ∧ latent ≥ 2.0): **630/2048 =
  30.76%** of camera samples (24.61% of all units).
- Expected physical-view inflation: `+1.3076` views per camera-cohort
  logical sample (`+1.2461` for the mixed 80/20 population).
- Expected VAE overhead: one extra VAE encode per pair ⇒ ~+30.8% VAE
  encode work on the camera cohort (same cost per view).
- Expected DiT overhead: one extra DiT row per pair ⇒ ~+30.8% DiT
  forward (and backward) rows on the camera cohort; Qwen unchanged.
- Expected walltime multiplier: **≈ 1.25–1.31×** on the forward/backward
  path for the canary's P25 population (24.6% mixed-population lower
  bound, 30.8% camera-cohort upper bound). Data-side cost (one extra
  crop of an already-resized canvas) is negligible against the model.

Clearly labeled: these are **estimates from frozen audits**; no
training, benchmarking, or long forward run was performed.

## CONTENT GUARD

- **Telemetry only; no online semantic model.** No CLIP, no PE model, no
  content scoring, and no sample rejection run in the training path.
- Per-batch aggregate `CameraMirrorCounts` (16 fixed keys with a hard
  conservation invariant: `applied ≤ selected ≤ eligible ≤ vertical ≤
  logical`; severity bands `lt2/2to4/ge4` sum to `vertical`; side bands
  sum to `applied`; `physical = logical + extra`) is carried on the
  training batch and is available to the existing observation
  machinery. The pinned `TrainingMetric` key set is untouched:
  mirror counters stop at the batch/observation layer by design.
- Per-sample audit fields record eligibility/selection, the mirror crop
  box, signed shift, `shift_y`, normalized offset, and degrade traces,
  so post-hoc directional audits (VBS/VCR-style) can be re-run on
  whatever the canary trains.
- The known limitation is explicit: mirroring balances **supervision
  direction**, not **content** — a pair can still disagree on which
  canvas side matters for the caption. That residual is why the audit
  recommended "… WITH CONTENT GUARDS"; the guard in V1 is the
  telemetry/post-hoc-audit loop, not an online model.

## RISKS

- **Increased physical-view compute:** up to ~+31% forward/backward rows
  on the camera cohort (see COMPUTE). Bounded by `pair_probability`.
- **Pair correlation:** the two views share source, caption, timestep,
  and noise; they are a supervision symmetry, not two independent
  samples — the 0.5/0.5 weighting is what keeps the effective sample
  count honest, and any change to it changes batch semantics.
- **Content mismatch still exists:** mirroring does not verify that the
  kept side is the caption-relevant side; the residual measured by
  VBS/VCR may only be partially closed.
- **Possible overcorrection:** equal-weight supervision of both sides
  could dilute cases where one side is genuinely the subject; the
  `min_latent_shift = 2.0` gate restricts V1 to the strongest-shifted
  vertical crops where the audit found the largest residual.
- **DDP denominator / batch semantics:** protected by leaving `step.py`
  untouched and by reducing the loss to logical length **before**
  reduction; the synthetic weighting fixture (5.5 vs 4.0) and the
  two-logical-samples contract test pin this.
- **RNG-stream perturbation:** structurally impossible by design (new
  isolated domain; disabled path bit-identical, unit-tested), but the
  canary must still verify stream parity empirically before any run.

## CANARY PROPOSAL

`config/train_g1_camera_v2_p25_mirror_canary.toml` — derives from the
current P25 config and changes **only** run/artifact identity and the
`[data.camera_mirror_balance]` table (unit-tested: a resolved-TOML diff
against the P25 baseline contains nothing else).

**NOT AUTHORIZED TO RUN.** This config is REVIEW ONLY. It has not been
launched, and launching it (or any training, resume, longer P25/P50, or
deployment) is explicitly outside this round's authorization.
