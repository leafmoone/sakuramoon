# iREPA ramp-out anchor validation fix — dev merge readiness

Date: 2026-09-06 (UTC, host clock 2026-09-06 08:18)
Base: `fix/irepa-token-spatial-layout` @ `fa7096d6804088165976f5a35ee1c4784ec67f6a`
Fix commit: the commit containing this file (HEAD of
`fix/irepa-token-spatial-layout` after this change).

## 1. What was fixed (config-correctness only, no re-development)

`ramp_out_after_updates` is an **absolute successful-update number**; the
anchored ramp-in ends at `start_successful_update + ramp_in_updates`.
The previous checks in both entry points compared an absolute update number
against the *duration* `ramp_in_updates`, so an absurdly early ramp-out
start (e.g. `5000` with anchor `113401` + ramp-in `1000`) was silently
accepted.

New validation (shared private helper `_validate_ramp_out_anchor`, called
from **both** entries before any weight computation or early return):

- `ramp_out_after_updates` not `None` and
  `< start_successful_update + ramp_in_updates` → `ValueError`.
- **Equality allowed** (ramp-out starts exactly at the ramp-in end;
  continuous, no extra hold required).
- Existing type/positive/finite checks and the anchor-free base relation
  (`ramp_out_after_updates > ramp_in_updates`) are **preserved unchanged**
  (no previously rejected combination is relaxed).
- The half-cosine formula and operation order are **untouched**; all legal
  schedule values remain bit-identical (pinned by tests F/G against the
  pre-fix formula reference).
- Validation cannot be masked by the `target_weight == 0.0` or
  pre-anchor early returns (case H).

Modified runtime file: `src/sakuramoon/objective/irepa.py` only
(+44 lines: helper, two call sites, docstring note).
`src/sakuramoon/config/schema.py`: comment-only clarification (+12); the
schema validator is anchor-free by design (no start anchor in the config)
and is unchanged.

## 2. RED -> GREEN

New test file: `tests/unit/objective/test_irepa_lambda_schedule_anchor.py`
(cases A-I per the fix spec; common params start=113401, target=0.5,
ramp_in=1000, ramp_out_updates=1000, ramp_in_end=114401).

- **RED (unmodified fa7096d)**: `7 failed, 6 passed` — every failure is
  `DID NOT RAISE ValueError` (the expected missing validation), i.e. cases
  A (5000), B (114000), C (114400), H (bypass via target=0.0 / pre-anchor
  update) and I at anchors 113401/500000/999999937. I at anchor=1 already
  passed pre-fix because the preserved legacy relation check happens to
  catch that specific boundary. Log:
  `/sakuramoon-runtime/merge-evidence/red-base.log`.
- **GREEN (fix applied)**: `34 passed` (13 new + 21 existing objective
  tests). Log: `green-anchor-objective.log`.
- Targeted regression (objective + anchor + config + lambda binding +
  alignment + spatial layout + zero-impact + production gate + composite +
  shadow audit): **105 passed, 0 failed** in 90.17s.
  Log: `targeted-regression.log`.

Invocation: `source /opt/dtk/env.sh; PYTHONPATH=src
/sakuramoon-runtime/sakuramoon-dtk-venv/bin/python -m pytest <files>`.

## 3. Static checks (same venv, same environment as base)

- `ruff check` (0.16.1) on the three changed Python files:
  **All checks passed**.
- `pyright src tests` (explicit scope): base `4951 errors` on clean
  fa7096d, post-fix `4951 errors` — **delta 0** (line-level `comm`
  comparison of full diagnostic lists is empty in both directions).
  The new test file initially added 16 diagnostics that are the same
  environment noise already present across the baseline (84
  `Import "pytest" could not be resolved`, 406 `Type of "raises" is
  unknown` in existing files — pyright cannot resolve the venv's pytest in
  this environment). They are suppressed with **targeted per-line**
  `# pyright: ignore[...]` comments in the new file only, consistent with
  existing codebase practice; no pyright config change, no blanket ignore,
  no skip/xfail.

## 4. Default-off gate

- All 20 mainline configs (`train_s0.toml` + 19 `train_g1*.toml`) parse
  with `config.irepa is None` (no `[irepa]` table anywhere in the
  mainline tree).
- Existing tests (all green in the targeted regression above):
  `test_legacy_config_without_irepa_parses_as_absent`,
  `test_explicit_disabled_irepa_parses_and_stays_legacy` (explicit
  `enabled=false` full legal table → composite spec `schema_version == 3`,
  no `training_auxiliaries`),
  `test_enabled_irepa_parses_with_v1_defaults` (enabled → v4 +
  `training_auxiliaries.irepa` 2560→768 k3),
  `test_absent_irepa_passes_production_readiness`,
  `test_disabled_irepa_passes_production_readiness`,
  `test_enabled_lambda_zero_is_bit_identical_to_disabled`.

## 5. Compatibility disclosures (iREPA line vs old code, unchanged by this fix)

- Training telemetry schema: `TRAINING_METRIC_SCHEMA_VERSION = 11`
  (`src/sakuramoon/telemetry/metrics.py`).
- Timing vocabulary: `FIXED_TIMING_PHASES` = **27 phases**
  (`src/sakuramoon/config/schema.py`); the config loader rejects other
  phase sets.
- Log format is **not** byte-identical to pre-iREPA logs (extra phases /
  metric fields); old standalone resolved TOMLs pinned at 25 phases need
  re-parse before reuse.
- Do **not** rewrite old checkpoints' `resolved_config` to the new
  vocabulary (they stay readable via their own snapshot).
- `λ = 0` (e.g. before the anchor, or a ramp-out that has finished) is
  **not** the same as the feature being disabled (the projector/teacher
  path exists; only the gradient weight is zero).

## 6. Scope & non-intrusion

Files touched by the fix commit (explicit add, nothing else):

- `src/sakuramoon/objective/irepa.py` (validation + docs)
- `src/sakuramoon/config/schema.py` (comments only)
- `tests/unit/objective/test_irepa_lambda_schedule_anchor.py` (new)
- `reports/irepa-come2-migration/irepa-come2-migration.md` (correction:
  "1000u 线性到 0.5" → "1000u 半余弦到 0.5", 2 places)
- this report

Untouched (verified by diff scope): the token-spatial-layout fix
(`transpose(1,2)` + reshape, commit `177c7a7`), the half-cosine lambda
formula, CMuon/optimizer code, attention code, PE-Spatial teacher code,
JLT/flow code, checkpoint save/load code, all production configs, data
service code.

## 7. Integration rehearsal (performed after this commit)

Procedure: from `dev` @ `3a341c0efa8aa6c82b41e508cf5fa2730e20fddb`
(verified as the merge base and ancestor of the fix branch), create
`integration/irepa-dev-ready` and `git merge --ff-only <this commit>`.
Invariants asserted: integration HEAD == this commit, trees identical, no
squash/cherry-pick. Results are recorded in the session merge-readiness
deliverable (the rehearsal runs on the committed HEAD and therefore cannot
quote its own SHA inside this file).

## 8. Prohibited actions (observed)

No trainer/canary/effectiveness runs; no 200u/1000u/10k rerun; no image
generation or PRE/POST eval; no checkpoint/optimizer/RNG changes; no
data-service/publisher start; no production worktree touched; no push and
no remote-dev movement; no amend/squash/rebase/force; no unrelated edits.
This change prepares the merge only; it does not move local or remote
`dev`.

Evidence directory: `/sakuramoon-runtime/merge-evidence/`
(red-base.log, green-anchor-objective.log, targeted-regression.log,
pyright base/after logs, ruff + default-off parse logs).
