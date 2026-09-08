# MBS Three-Way Numerics Adjudication (numerics-v2)

Pre-registered replicate-robustness protocol. This README and
`contracts.py` **freeze the protocol BEFORE `TREATMENT_B` is scored and
before any new C/T value is interpreted** (round rule 0). The round
implements the external review verdict of 2026-09-09: the previous
replay-gate FAIL (`replay-gate-v1`) was a gate-calibration artifact —
it compared different statistical objects (aggregate SAME mean CI vs
per-pair max/RMS of the new-minus-frozen delta) — and is NOT a valid
scientific hard gate. The old results remain `SCORED_NOT_INTERPRETED`;
the old gate is not relaxed post-hoc and the old results are not
released directly. This round asks the correct scientific question:

> **Does HCU/DCU forward numerical nondeterminism change the
> intervention-associated estimator classification
> (I_TOP / I_BOTTOM / I_G, TARGETED latent>=2) under a same-source
> frozen evaluation cohort?**

**This is an EVALUATION round.** No training, no backward, no
optimizer, no checkpoint write, no FID, no P50, no production, no
canonical control training. Exactly one new scoring run: `TREATMENT_B`
(2 workers, frozen pair root, fresh root, no overwriting).

## 0. Scope and hard rules (verbatim protocol)

* No per-pair maxabs hard gate; no aggregate-SAME-CI -> per-pair
  threshold conversion; no post-hoc per-pair limit.
* Per-pair numerical tail statistics (bit-exact rate, mean/max/RMS
  differences, tail counts) are **diagnostics only** and never enter
  the final decision.
* The final decision is made **only** on: (a) the classification of all
  8 replicate tuples, (b) their conservative envelope, and (c)
  agreement between them.
* Binding scientific wording (if the adjudication passes): "**a
  replicate-robust intervention-associated contrast under a same-source
  frozen evaluation cohort**". It is NOT an "exact causal effect" and
  not "causally proven".
* Root-cause wording (fixed): "**localized HCU/DCU forward numerical
  nondeterminism; root operator/kernel family not isolated**". Never
  "proven ROCm reduction bug".
* Immutability: `src/`, `config/`, `scripts/`, existing tests, prior
  reports, `mbs_three_way/`, `vertical_bottom_supervision/`
  (`final_snapshot/`) are never modified. Checkpoints are read-only
  and are re-hashed before and after. `/tmp` artifacts, checkpoints,
  weights, images, latents, caches, secrets are never staged.
* Branch: `camera-v2-mbs-three-way-numerics-adjudication-review` from
  `26b9a196f50658a4d0ae21a5c584c881d445ea47`. Two commits: H1 (this
  tooling, "tools: add MBS three-way numerics adjudication") and H2
  (evidence, "docs: record MBS three-way numerics adjudication").
  Push only via bundle -> local -> origin, no force push, no dev/main/
  master/prod push.

## 1. Evaluation object and estimators (frozen, reused)

Same source pairs / anchors / arms / cohorts / formulas as the original
VBS audit (frozen package `vertical_bottom_supervision/` + reviewed
three-way tooling `mbs_three_way/`, both imported read-only, unmodified):

* 512 source pairs; 8 arms (BB, BT, BB+TT ID, SAME x top/bottom) x 4
  strata each.
* Per-pair margins (frozen verbatim expression, sums before
  subtraction): `m_top=(sum(TB)-sum(TT))/4`, `m_bot=(sum(BT)-sum(BB))/4`,
  `f_top=(sum(T_SAME)-sum(TT))/4`, `f_bot=(sum(B_SAME)-sum(BB))/4`.
* Per-pair adjusted deltas (SAME-corrected), for X in {CONTROL,
  TREATMENT}: `D_TOP_X=(m_top_X - m_top_PRE) - (f_top_X - f_top_PRE)`,
  `D_BOTTOM_X=(m_bot_X - m_bot_PRE) - (f_bot_X - f_bot_PRE)`,
  `G_X=D_TOP_X-D_BOTTOM_X`.
* Intervention endpoints: `I_TOP=D_TOP_T-D_TOP_C`,
  `I_BOTTOM=D_BOTTOM_T-D_BOTTOM_C`, `I_G=G_T-G_C`.
* Point estimate = exact float64 mean over the cohort.
* Cohorts: `all` (512), `latent_ge2` (312, TARGETED), `latent_lt2`
  (200), `latent_2to4` (275), `latent_ge4` (37), `content_balanced`
  (256). Cohort membership is frozen via the committed pair CSV +
  `pair_cohort_flags`; sizes are hard-gated (identity + completeness
  gates only).

## 2. Replicate inventory (frozen)

`A` = original three-way scoring run; `B` = independent fresh-process
numerical realization. Each replicate = 2 ledger files x 256 rows = 512
pairs, worker split by pair parity (w0 even, w1 odd). All six must be
complete (no missing/duplicate pair, 8 arms x 4 strata, all finite).

| replicate | root | ledger file (w0 / w1) | sha256 (w0 / w1) |
| --- | --- | --- | --- |
| PRE_A | `/tmp/camera-mbs-three-way-audit` | `ledger-w0-pre.jsonl` / `ledger-w1-pre.jsonl` | `4f780e3578df506b77b14620ec0abec5e285e50391266eb71d0d82daa8ab51a1` / `59918d1aa3b821f4a350af72b6c636746a0f7a7230ebf266664ab67c066c4316` |
| PRE_B | `/tmp/mbs3-rerun-full` | `ledger-w0-pre.jsonl` / `ledger-w1-pre.jsonl` | `5a70b4c29c5a3d414589855a22bbae80286ffdc9fa7bfc8a2a4dbea4a0dd051d` / `e548a27c838cf6b0cc75057152c65b9c2e61e3dd3f23b84f2216b81951ec963a` |
| CONTROL_A | `/tmp/camera-mbs-three-way-audit` | `ledger-w0-control.jsonl` / `ledger-w1-control.jsonl` | `3cf918a0774d48861cf69e9a3d9047b6ecef7719831f5b07f2fc6e322bd20613` / `0616622d67217f7cbcb1d88ef068e6419fc5a077ea254f95d94c7883b53831e7` |
| CONTROL_B | `/tmp/mbs3-rerun-full` | `ledger-w0-control.jsonl` / `ledger-w1-control.jsonl` | `779860ceccfa4e9ec938b49f5389ca3a2b9b00f876966409b025be33236c18f0` / `1fdd17d733e5fdc290aae1e0649eff290020b8888bc671f571a91b89ba030a2b` |
| TREATMENT_A | `/tmp/camera-mbs-three-way-audit` | `ledger-w0-treatment.jsonl` / `ledger-w1-treatment.jsonl` | `8c6866b293a94061e97b8d1a3e38e35e8fc7ccb46cdd4cc333f1784a1e1db751` / `fcfaec5964e2067eae01c93b592eed4464c8fbf800fec87dbb85f9eaf5761960` |
| TREATMENT_B | `/tmp/camera-mbs-three-way-numerics/treatment-b` | `ledger-w0-treatment.jsonl` / `ledger-w1-treatment.jsonl` | filled at the raw-hash freeze step (after scoring, before analysis) |

Frozen evaluation cohort reference: pair manifest
`/tmp/camera-vertical-bottom/pair-manifest.json` (512 pairs) with
cohort identity field-verified against the committed frozen CSV
`reports/camera-vertical-bottom-supervision-pairs.csv` (sha256
`1a08b68de84db872e67b3fd3044929884d89c1ae2b6e8b317611be4f1221c4a8`)
by the reviewed `check_pair_cohort_identity` gate (reused, unmodified).

After `TREATMENT_B` completes, all 12 ledger files (6 replicates) are
hash-frozen into `/tmp/camera-mbs-three-way-numerics/raw-replicates.json`
(sha256 + row count per file). `analyze_replicates.py` re-verifies the
freeze **before** reading any scientific value and exits 2 on mismatch.

## 3. Replicate combinations (pre-registered)

* 8 full tuples (P_i, C_j, T_k), i,j,k in {0,1} (A=0, B=1), fixed
  enumeration order, all reported:
  `P0 C0 T0, P0 C0 T1, P0 C1 T0, P0 C1 T1, P1 C0 T0, P1 C0 T1, P1 C1 T0,
  P1 C1 T1`.
* 4 unique C/T combinations (PRE cancels from I; the 8-tuple analysis
  is the diagnostic proof of that fact): `C0/T0, C0/T1, C1/T0, C1/T1`.

## 4. Bootstrap (pre-registered)

* Unit = source pair (512 pairs, not 4096 strata, not arms).
* Paired bootstrap over the cohort's per-pair values: n=10000
  resamples, seed=20260907, percentile 2.5/97.5.
* **One deterministic index matrix per cohort**
  (`rng = np.random.default_rng(20260907); idx =
  rng.integers(0, n, size=(10000, n))`) — it depends only on (seed,
  cohort size), never on the values — and is **reused exactly** across
  all checkpoint replicates, all replicate tuples, and all C/T
  combinations. This makes tuple-level CIs directly paired. Mechanically
  identical to the frozen `VBS.bootstrap_ci` (tested for bit-equality).
* No additional confidence level is added; 95% only, as originally
  frozen.

## 5. PRE cancellation (diagnostic proof, pre-registered)

`I` is algebraically PRE-free. The analysis computes `I` through the
UNFACTORED frozen formulas with `PRE_A` and with `PRE_B` for every C/T
combination and every metric, and requires
`max_pairs |I(PRE_A) - I(PRE_B)| <= 1e-12`.

Rationale: values are O(1e-3); float64 eps at that scale is ~1e-19, so
1e-12 is 7 orders of magnitude above roundoff and ~6 orders below any
scientifically meaningful contrast. Exceeding the tolerance means an
implementation bug in this new tooling — the analysis exits 2 and no
result is produced. This check is an implementation/roundoff gate, NOT
a scientific per-pair gate.

## 6. Classification (unchanged pre-registered decision tree)

Per tuple, TARGETED cohort (`latent_ge2`, n=312), using the two 95%
CIs:

* `CLEAR_DIRECTIONAL_CORRECTION` iff CI(I_BOTTOM).lower > 0 AND
  CI(I_G).upper < 0.
* `CLEAR_HARM` iff CI(I_BOTTOM).upper < 0 OR CI(I_G).lower > 0.
* otherwise `BORDERLINE`.

Implemented by importing the exact reviewed
`mbs_three_way.contracts.classify_contrast` (unmodified); a focused
test pins the decision tree on boundary cases.

## 7. Conservative envelope (pre-registered)

For each cohort and metric, over all 4 unique C/T combinations:
`envelope = (min of CI lows, max of CI highs)` plus min/max point. The
envelope classification applies the SAME decision tree to the envelope
CI bounds (I_BOTTOM envelope CI + I_G envelope CI).

## 8. Final adjudication (pre-registered)

* `NUMERICALLY_STABLE` iff ALL 8 TARGETED replicate tuples produce the
  SAME classification.
* If stable: robust classification = the common class. If the common
  class == the conservative envelope class:
  **`NUMERICAL_ADJUDICATION = PASS`** and the report may use the
  binding wording of §0. If they disagree:
  `NUMERICAL_ADJUDICATION = AMBIGUOUS`; no directional release.
* If not stable: `NUMERICALLY_AMBIGUOUS`, `NUMERICAL_ADJUDICATION =
  AMBIGUOUS`; no directional release.
* The adjudication consumes ONLY combination-level classification
  labels. It never consults per-pair maxabs/RMS/tail counts (a focused
  test enforces that `adjudicate()` has no per-pair input).

## 9. Diagnostics (required output, no decision role)

* Per-pair A-vs-B differences for M_TOP, M_BOTTOM, F_TOP, F_BOTTOM per
  checkpoint: mean signed diff, mean abs, RMS, median/p95/p99/max abs,
  bit-exact count+fraction, counts > 1e-6 / 1e-5 / 1e-4.
* Estimator numerical spread: range of I_BOTTOM point and I_G point
  across the 4 unique C/T combinations (TARGETED), and
  point-spread / median-bootstrap-CI-width ratio.
* PRE-cancellation max-abs per combination (implementation proof, §5).
* The report must state that these diagnostics do not enter the
  decision.

## 10. Required reports (committed at H2)

1. `reports/camera-mbs-three-way-numerics-adjudication.md` — the
   adjudication narrative: verdict, 8-tuple table, envelope,
   diagnostics, binding statement.
2. `reports/camera-mbs-three-way-numerics-adjudication.json` — machine
   verdict + replicate inventory (hash freeze) + tuple classes.
3. `reports/camera-mbs-three-way-numerics-metrics.json` — all cohorts x
   8 tuples x {I_TOP, I_BOTTOM, I_G}: point + 95% CI; envelope;
   diagnostics; bootstrap parameters.
4. `reports/camera-mbs-three-way-numerics-pairs.csv` — per-pair
   replicate margins/floors (all 6 replicates x 4 metrics, 17-sig-fig
   exact round-trip) + per-pair C/T combination contrasts; the
   analysis tool independently replays ALL reported point estimates
   from this CSV and requires exact reproduction before writing the
   reports.

## 11. Execution order (pre-registered)

1. This H1 tooling commit (protocol frozen) — BEFORE scoring
   `TREATMENT_B`.
2. `TREATMENT_B` scoring: fresh root, 2 DCU workers (cuda:0/cuda:1),
   frozen pair root, resume-safe ledgers, then the 512/512 pair-cohort
   identity gate and completeness checks.
3. Raw hash freeze of all 6 replicates (`raw-replicates.json`).
4. Analysis (`analyze_replicates.py`): freeze verify -> pair-cohort
   identity -> load 6 replicates -> 8 tuples -> PRE-cancellation proof
   -> shared-matrix bootstrap -> classification + envelope +
   adjudication -> diagnostics -> CSV -> CSV self-replay -> 4 reports.
5. Checkpoint identity re-hash (PRE/CONTROL/TREATMENT) before and
   after; embed in the evidence.
6. H2 evidence commit (explicit paths only) -> bundle push -> origin.
7. HARD STOP after the successful push; deliver the final copy.

## 12. Tooling notes

* No torch import in the analysis tooling; CPU float64 only. The
  scoring run reuses the reviewed, unmodified
  `mbs_three_way/score_three_way.py` with the frozen scorer
  environment (`env-cc.sh`, `PYTHONPATH` to the C2 repo src, DTK venv).
* `contracts.py` and `analyze_replicates.py` import the reviewed
  `mbs_three_way` modules under their OWN bare names with the mbs
  directory first on `sys.path` (that code uses bare `import
  contracts`), while this package's own `contracts.py` is loaded only
  under the unique name `mbs3w_numerics_contracts`, so the bare name
  `contracts` never binds to this package — the reverse of the
  shadowing failure mode found and fixed in the previous round
  (mbs's VBS import uses the explicit-path distinct-name pattern;
  that pattern does not work in the reverse direction because
  analyze_three_way's bare `import contracts` would otherwise resolve
  to this package's module).
* Tests: `test_replicate_adjudication.py`, 10 focused tests (see its
  header). No GPU, no network, deterministic.
