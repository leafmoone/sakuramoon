# SakuraMoon Camera — MBS Three-Way Same-Source Causal Audit

**Status: STOPPED AT PRE-SCORE REPLAY GATE (spec §7) — CONTROL/TREATMENT scored but NOT interpreted.**

- Date: 2026-09-08/09 (UTC+8 local; timestamps below UTC unless noted)
- Host: crdnotebook-2097137043750113282-come7-21657 (come7, SCNet HCU; DTK 26.04, torch 2.9.0, 2× DCU "BW")
- Base reviewed SHA: `1b8fce49031b854b835d9f47ea65a3f77b83a2f9`
- Branch: `camera-v2-mbs-three-way-same-source-audit-review`
- Tooling SHA (H1): `9d428faf01ed5843002eb5246f2d688b13de7418`
- Production diff (`src/ config/ scripts/` vs base): EMPTY
- Frozen `vertical_bottom_supervision/` + `final_snapshot/`: untouched

## 1. Design

Same-source 3-way comparison on the frozen 512-pair Vertical Bottom Supervision (VBS) cohort:
PRE U118100 vs CONTROL U118200 vs TREATMENT U118200. Forward-only: no training, no
backward, no optimizer steps, no checkpoint writes, no FID/generation eval. 8 arms ×
4 timestep strata per pair per checkpoint (49,152 forwards total). PRE doubles as a
replay/numerics gate against the frozen VBS audit's U118100 (POST) values (spec §7).

## 2. Checkpoint identity (gates, pre- and post-scoring)

| ckpt | update | manifest sha256 | model tree sha256 | full tree sha256 | status |
| --- | --- | --- | --- | --- | --- |
| PRE `/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence` | 118100 | `821a8c12…b892` | `32b76aff…a370` | — | PASS |
| CONTROL `/sakuramoon-runtime/output_model/g1_camera_v2_p25_mirror/ckpt_118200_raw-118200-update-cadence` | 118200 | `f014827f…57f7` | `5347def8…c241` | — | PASS |
| TREATMENT `/sakuramoon-runtime/output_model/g1_camera_v2_p25_mirror_v2/ckpt_118200_raw-118200-update-cadence` | 118200 | `11deada8…9824` | `22a802f2…0fb9` | `d2673914…06c74` | PASS |

growth_alpha = 1.0 for all. Re-hashed after scoring: identical (immutable).
Tree-hash algorithm: evidence-lineage canonical `find <abs-root> -type f | sort |
xargs sha256sum | sha256sum` (TREATMENT full-tree reproduces the RERUN #2 report value
`d2673914…` exactly).

## 3. Pair cohort (frozen 512, reconstructed)

- Pair manifest sha256 (reconstructed): `cbde1a735fec05e2c9025555c0797ae94a6911f2c3ad8934cb2b2c75580f76c9`
  (the original manifest sha `7f3a81b2…` is not byte-reproducible — build_pairs embeds created_utc)
- **Pair-cohort identity gate: PASS — 512/512 pairs field-identical to the committed
  frozen CSV** (source_shard, sample_id, original_side, k_start/k_end, available,
  full_height, signed_shift_start).
- Cohort sizes (pre-registered): all 512 · targeted latent≥2 312 · latent<2 200 ·
  2-to-4 275 · latent≥4 37 · content-balanced 256.
- Stage-1 reconstruction: 2560 units (2048 camera + 512 ordinary), 4831 s wall;
  arm validation all-pass (derangement reproducible, no self-assignment, fp32-eps maxdiffs).

## 4. PRE replay gate (spec §7) — **FAIL**

New PRE (U118100) pair-level values vs frozen VBS POST (U118100) values:

| check | value | bound | pass |
| --- | --- | --- | --- |
| maxabs per-pair | 2.7345e-04 | 3.3698e-06 | **no** |
| RMS per-pair | 2.0120e-05 | 3.3698e-06 | **no** |
| point Δ M_TOP | 5.09e-07 | 3.3698e-06 | yes |
| point Δ M_BOTTOM | 5.44e-07 | 3.3698e-06 | yes |

- mean per-pair delta: 5.27e-07 (no systematic bias)
- bound = 5 × max(frozen POST SAME-floor CI95 upper) from committed audit.json
  (floor CI95 upper: M_TOP 6.06e-07, M_BOT 6.74e-07)
- New points: M_TOP 0.0021155239082872868, M_BOT 0.0035472347371978685
- Frozen points: M_TOP 0.002115014685841743, M_BOT 0.003546690451912582

**Per-pair delta distribution (the decisive pattern):**

| | M_TOP | M_BOT |
| --- | --- | --- |
| median \|Δ\| | 1.98e-15 | 2.64e-15 |
| \|Δ\| > 1e-6 | 57/512 | 60/512 |
| \|Δ\| > 1e-5 | 39/512 | 46/512 |
| \|Δ\| > 1e-4 | 4/512 | 4/512 |
| max \|Δ\| | 2.73e-04 (pair 298) | 2.30e-04 (pair 400) |

~89% of pairs reproduce the frozen values to 1e-6 (median 2e-15, i.e. bit-level);
the deviation mass sits in a small tail, with mixed signs and no correlation with
content features (corr(|signed_shift|, |Δ_top|) = 0.025).

## 5. Forensics — why the gate failed

8 tail pairs were re-scored on come7 in an isolated scratch out-root
(`/tmp/mbs3-forensic`, original ledgers untouched):

| pair | Δ_top orig | Δ_top rerun | Δ_bot orig | Δ_bot rerun |
| --- | --- | --- | --- | --- |
| 298 | +2.735e-04 | +2.735e-04 (exact) | +3.801e-05 | +3.801e-05 (exact) |
| 203 | +2.654e-04 | +2.654e-04 (exact) | −1.007e-04 | −1.007e-04 (exact) |
| 459 | −1.237e-04 | −1.237e-04 (exact) | +9.675e-05 | +9.675e-05 (exact) |
| 371 | +1.083e-04 | **+1.52e-16 (bit-exact vs frozen)** | −4.162e-05 | +4.51e-15 (bit-exact) |
| 109 | +9.874e-05 | +9.874e-05 (exact) | +8.661e-05 | +8.661e-05 (exact) |
| 400 | −9.845e-05 | −9.845e-05 (exact) | +2.302e-04 | +2.302e-04 (exact) |
| 43 | +9.298e-05 | **+1.71e-15 (bit-exact vs frozen)** | +1.505e-05 | +3.14e-15 (bit-exact) |
| 278 | −8.754e-05 | −8.754e-05 (exact) | +5.735e-05 | +5.735e-05 (exact) |

- 6/8 re-runs bit-exact with the original run (per-input determinism for most inputs).
- 2/8 (pairs 371, 43) re-ran **bit-exact against the frozen values** although the
  original run differed ~1e-4 — the same host, same inputs, same code produced
  different results across runs for those inputs.

**Conclusion:** the pipeline is a bit-faithful reproduction (median per-pair delta
2e-15; two tail pairs bit-exact against the frozen audit on re-run). The residual tail
is fp32-level (1e-5…1e-4) deviation of a small input subset attributable to
non-deterministic DCU (ROCm) kernel reduction paths — some stable across runs on come7
(cross-host difference vs the original audit host), some run-to-run varying. This
exceeds the frozen host's own numerics floor by up to ~81× in the tail, which is
precisely what the pre-registered gate is designed to catch.

Full re-verification run (all 512 pairs, PRE + CONTROL, scratch out-root
`/tmp/mbs3-rerun-full`): **PRE (512/512 re-scored):** 456/512 bit-exact vs the original run · 72/512 bit-exact vs the frozen reference · of the 57 pairs with |Δ_top|>1e-6: 12 stable (identical re-run), 45 flipped · max |rerun Δ vs frozen| = 0.000273 (pair 298).  **CONTROL:** 458/512 bit-exact vs its original run.

## 6. Scoring run record (factual; no interpretation)

- 49,152 forwards total (512 pairs × 8 arms × 4 strata × 3 checkpoints)
- Per-checkpoint wall (both DCUs in parallel, 256 pairs/worker):
  PRE 22:32:58→22:41:36Z · CONTROL 22:41:36→22:51:09Z · TREATMENT 22:51:09→23:29:13Z
  (the TREATMENT interval includes one supervisor relaunch after a local ssh session
  drop; resume-safe ledgers, no pair re-scored)
- Anchor-coord invariant (reconstructed anchor coords == frozen bundle CORRECT,
  raise-on-mismatch): held for every scored pair — worker completion without exception
  is the proof (the check raises on any mismatch). Regenerated 256/256 counters in the
  re-verification run: PRE/CONTROL regenerated at 256/256 per worker by the re-verification run (the original PRE/CONTROL gate docs were overwritten by the resume launch, which re-verified identity but scored 0 new pairs and thus reset the counter); TREATMENT gate docs intact from the original run.
- Per-worker: | ckpt | worker | device | pairs | finished (UTC) | anchor-invariant | source |
| --- | --- | --- | --- | --- | --- | --- |
| PRE | w0 | cuda:0 | 256 | 2026-09-08T15:19:04Z | 256 | re-verification gate doc (counter regenerated; original doc overwritten by resume) |
| PRE | w1 | cuda:1 | 256 | 2026-09-08T15:19:04Z | 256 | re-verification gate doc (counter regenerated; original doc overwritten by resume) |
| CONTROL | w0 | cuda:0 | 256 | 2026-09-08T15:19:38Z | 256 | re-verification gate doc (counter regenerated; original doc overwritten by resume) |
| CONTROL | w1 | cuda:1 | 256 | 2026-09-08T15:19:38Z | 256 | re-verification gate doc (counter regenerated; original doc overwritten by resume) |
| TREATMENT | w0 | cuda:0 | 256 | 2026-09-08T15:28:54Z | 256 | original-run gate doc (intact) |
| TREATMENT | w1 | cuda:1 | 256 | 2026-09-08T15:29:13Z | 256 | original-run gate doc (intact) |

## 7. Outcome per spec §7

> "If replay exceeds the previously established numerical floor in a scientifically
> material way: STOP. do not interpret treatment/control."

**This audit therefore records: (a) full identity/cohort/immutability PASS, (b) replay
gate FAIL (maxabs 81× bound), (c) CONTROL/TREATMENT scored but NOT interpreted — no
D_TOP/D_BOTTOM/G/I contrast is reported.** Raw ledgers remain in
`/tmp/camera-mbs-three-way-audit/ledger` (never committed). The per-pair PRE replay
values (new + frozen + deltas) are committed in
`camera-mbs-three-way-same-source-pairs.csv` so the replay statistics can be
independently recomputed.

## 8. Validation

- Focused tests: 35/35 pass (come7 DTK venv, pytest 9.1.1; no torch required)
- ruff (repo config, 0.16): new tooling files clean; `reconstruct_stage1.py`
  verbatim-inherits the 11 frozen `cc_stage1.py` findings (documented in the tooling README)
- py_compile: OK · `git diff --check`: clean
- Immutability: `git diff 1b8fce49..HEAD -- src config scripts` empty; frozen VBS +
  final_snapshot untouched; all three checkpoints re-hashed post-scoring (unchanged)
- No installs, no reboot, no DTK changes; only come7 used; no /tmp artifacts staged

## 9. Files

- `camera-mbs-three-way-same-source-audit.md` (this file)
- `camera-mbs-three-way-same-source-audit.json` (machine-readable audit record)
- `camera-mbs-three-way-same-source-metrics.json` (run metrics, gates, forensics)
- `camera-mbs-three-way-same-source-pairs.csv` (512 rows: per-pair PRE margins, frozen
  references, replay deltas, cohort flags — 17-significant-digit float64 round-trip)
