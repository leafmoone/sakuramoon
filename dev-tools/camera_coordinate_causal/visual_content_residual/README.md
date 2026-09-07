# Visual Content Residual Audit (Camera Viewport v2)

Re-scores the already-reviewed 512-pair same-source vertical mirror audit
(base commit `f3d183a`, branch `camera-v2-vertical-bottom-supervision-review`)
with real visual-content metrics and geometry-stratified visual balancing.

## What this round answers

The prior generation's content-balanced 50% subset was defined from
`|delta_CLIP_TEXT|` only (CLIP text was available), so it did not control the
actual VISUAL content asymmetry measured by CLIP image-full preservation and
PE-Spatial retained content. This round:

1. ranks every pair by a composite visual asymmetry score built ONLY from
   the two visual metrics (CLIP text is secondary and never enters the
   primary score),
2. freezes geometry-stratified visual-balanced subsets (50% / 25%) so the
   balance cannot silently select milder geometry,
3. re-measures the paired TOP-BOTTOM causal gap on those frozen subsets and
   classifies the residual.

## Inputs (all frozen, read-only)

- `/tmp/camera-vertical-bottom/pair-manifest.json` (sha pinned in `contracts.py`)
- `/tmp/camera-vertical-bottom/features/clip-audit.json`
- `/tmp/camera-vertical-bottom/features/pe-audit.json`
- `/tmp/camera-vertical-bottom/analysis/vertical-bottom-main.json` (frozen per-pair G)
- committed `reports/camera-vertical-bottom-supervision-audit.json` + `...-pairs.csv` (exact cross-check)

CPU only. No forwards, no encodes, no latents, no captions, no training.

## Phase separation (pre-registered)

- `freeze_visual_subsets.py` (PHASE A) reads ONLY the pair manifest and the
  two feature audits. It emits `visual-subsets.json` and its SHA freeze.
  A static source lock in the test suite forbids it from referencing any
  per-pair mirror statistic or prior report.
- `analyze.py` (PHASE B) verifies the SHA, then combines the frozen subsets
  with the frozen per-pair statistics.

## Pre-registered choices

- composite: `A_visual = 0.5*(R_img + R_pe)`, percentile ranks with average
  ranks for ties, in `[0, 1]` (deterministic, seed-invariant point);
- geometry cells: `original_side x zoom_band x latent_shift_band` with the
  frozen VBS band definitions (mild <1.20 <= medium <1.35 <= strong;
  latent <2, [2,4), >=4);
- quotas: proportional + largest remainder, exact 256 / 128 totals;
  within-cell selection ascending by `(A_visual, pair_index)`;
- bootstrap: source-pair resampling, seed 20260907, n=10000; point is always
  the exact float64 observed mean (seed-invariant);
- retention/attenuation are never clipped (negative allowed).

## Legacy tool note

The prior-generation legacy helper (`pooled_sign_mean`) is NOT imported or
called by this audit. This tool implements only `paired_mean_difference`
(`mean(top-bottom)`) and `two_sample_mean_difference` (`mean(A)-mean(B)`),
both locked by tests.

## Outputs

Runtime (not committed): `/tmp/camera-visual-content-residual/{visual-subsets.json,
visual-subsets-frozen.json, analysis.json, bootstrap.json, logs/}`.

Reports (new files only, in `reports/`):
`camera-vertical-bottom-visual-content-residual-{audit.md, audit.json,
metrics.json, subsets.csv, copy-report.md}`.
