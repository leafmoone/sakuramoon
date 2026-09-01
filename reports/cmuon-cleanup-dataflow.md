# CMuon production cleanup — hot-path dataflow audit

Scope: `hybrid_cmuon_canonical_ns4_fp32_rescue` production hot path at D4
frozen head `37602e4` (branch `cmoun`, baseline `fd1131e`).

Files audited:
- `src/sakuramoon/optim/fp32_rescue.py` (entire `step()`, lines 118-587)
- `src/sakuramoon/optim/guarded_canonical.py` (helpers the subclass calls:
  `_sync_learning_rate`/`_validate_finite_gradients` [inherited from
  `HybridCMuon`], `owner_of`, `_first_device`, `_input_key`, `_chunk_shape`,
  `_guard_state`, `state_dict`, `load_state_dict`, `_f32`, `_PreparedSpec`;
  base `step()` 274-628 = separate legacy candidate path, NOT in the frozen
  candidate's hot path, audited for contrast only)
- `src/sakuramoon/optim/cmuon.py` (NS functions 112-243, `cmuon_moonlight_alpha`,
  `HybridCMuon` interface/state 795-1130)
- call paths: `train/production.py` `_build_optimizer` (377-513, name dispatch
  472-493), `checkpoint/save.py:239` (passthrough `optimizer.state_dict()`),
  `checkpoint/load.py:1134/1189/1213` (passthrough `load_state_dict`),
  `config/schema.py` (OptimizerConfig 794-880, frozen).

## 1. Host-sync inventory (baseline, per rank, per step)

Line refs = `fp32_rescue.py` unless noted.

| # | line | call | syncs/rank | class |
|---|---|---|---|---|
| 1 | 119 | `_sync_learning_rate` | 0 (lr is a Python float in param_groups; `.item()` only if a Tensor) | — |
| 2 | 120 | `_validate_finite_gradients` (cmuon.py:887 `bool(finite.item())`) | 1 (already batched across all grads) | A |
| 3 | 161 | per-chunk device RMS `c.pow(2).mean().sqrt()` | 0 (device) | B (feeds refs) |
| 4 | 162 | per-chunk device L2 `c.norm()` | 0 (device) | D (DEAD, §2) |
| 5 | 169 | `torch.stack(sig_flat).tolist()` | 1 | B (feeds ref update) + C (forensic) |
| 6 | 170 | `torch.stack(sigf_flat).tolist()` — **bare expression, result discarded** | 1 | D (DEAD) |
| 7 | 201 | `bool(torch.isfinite(delta).all())` — **per owned chunk** | 83 (owner rank only) | A |
| 8 | 212 | per-chunk device `delta.float().pow(2).mean().sqrt()` | 0 (device) | A+C |
| 9 | 226 | `torch.stack(d_rms_list).tolist()` | 1 (owner rank) | A |
| 10 | 238 | `int(fail_flags[idx].item())` — **per chunk, every rank** (166 iterations) | **166** | A |
| 11 | 251/253 | rescue: `float(delta32.pow(2).mean().sqrt())` + `bool(torch.isfinite(delta32).all())` | 2 × (rescues this step) | A |
| 12 | 283 | `dist.all_reduce(fail_flags, MAX)` | 0 (collective) | A |
| 13 | 285 | `fail_flags.tolist()` (failure messages) | 1 | A |
| 14 | 294 | `fail_flags.tolist()` **again** (builds `failed` set) — duplicate host read of the same tensor | 1 | A (duplicate) |
| 15 | 304/307 | per-chunk `dist.broadcast` (×166) | 0 (collectives) | A (frozen: keep per-chunk) |
| 16 | 325-330 | fingerprint MIN/MAX all_reduce + `float((hi-lo).max().item())` | 1 | A (frozen invariant) |
| 17 | 449-461 | param fingerprint all_reduce + `float((hi-lo).max().item())` | 1 | A (frozen invariant) |
| 18 | 341 | forensic dump `d_rms tolist` | failure path only | C |

Baseline totals: non-owner rank ≈ **172** host syncs (166 + 6); owner rank ≈
**256** (166 + 83 + 7). D3's "~168" counted the dominant per-chunk loop
conservatively; the exact instrumented number is recorded in the final
report (section 14 of the cleanup spec).

Dominant costs: #10 (166/rank), #7 (83 owner), #9/#6 (2), #14 (1 duplicate).

## 2. Intermediate-state classification (A/B/C/D)

| state | computed | consumed by | class | disposition |
|---|---|---|---|---|
| `buf` momentum EMA | PREPARE 142 | nesterov; ckpt momentum | A+B | keep |
| `nesterov` / `chunks` | PREPARE | NS input | A | keep |
| `alphas` (Moonlight) | PREPARE 150 | delta scale, commit | A | keep |
| `cf_chunks` (FP32 copies) | PREPARE 154 | `sig` only (after sigf removal) | B (feeds ref state) | keep (shrinks to sig-only use) |
| `sig` (per-chunk FP32 RMS) | PREPARE 161 device | (1) per-step reference update 440-442 → `_refs` ckpt contract; (2) forensic `u_t_rms` | **B + C** | keep device compute + 1 batched readback (state semantics, §3) |
| `sigf` (per-chunk L2) | PREPARE 162 device | **none in this candidate** (only base `step()`/`_is_low_signal` reads it — different candidate, line 344/349) | **D** | **remove** compute + stack + tolist (line 170) + `_PreparedSpec` fill (pass `[]`); field stays for the base candidate |
| `fail_flags` | 202/228 owner-local; rescue 261; all_reduce 283 | rescue decision, broadcast skip, fingerprint skip, failure raise, zero-commit | **A** | keep tensor semantics exactly; batch the reads (166 → 1) |
| `isfinite(delta)` | 201 owner | `fail_flags` | **A** | device-side flag (no `bool()` sync) |
| `d_rms` (delta RMS) | 212 owner device | ceiling flag (A); forensic dump (C) | **A + C** | device-side ceiling compare; host tolist only on the failure path |
| `staged` deltas | 214/260 | broadcast, fingerprint, commit | **A** | keep |
| `owners` / `is_active` | PREPARE | rescue gate, broadcast src, commit identity | A (routing identity, frozen) | keep |
| `rescue_meta` | 203/215 | rescue | **A** | keep |
| rescue counters (`bf16_attempts` … `rescue_by_role`) | 196/243-244/258/262-264 | ckpt `fp32_rescue` block (B); event/stats lines (C) | **B + C** | keep |
| `_refs` (per-input FP32 refs) | 440-442 (host update from batched `sig`) | ckpt guard `references` table (B); stats `ref_min/max` (C). **NOT a rescue decision input in this candidate** — ceiling (124) and rescue floor (125) are fixed `lr`-derived constants; `guard_ratio`/`min_reference`/`numerical_floor` are consumed only by base `_is_low_signal`, which `step()` never calls | **B + C** | keep update + storage exactly (checkpoint contract); the D4-observed ref evolution is real state motion but decision-inert here |
| `skip_total` / `skip_by_role` / `skip_by_fqn` | base `__init__` (0 in this candidate) | ckpt schema (B) | **B** | keep at 0 (schema fields) |
| `observations` | 445 | ckpt (B); stats cadence (C) | **B + C** | keep |
| `bootstrap_mode` | `__init__` | ckpt (B) | **B** | keep |
| fingerprint `fp` list | 317-323 | spread invariant 330 | **A** | keep (frozen invariant) |
| param fingerprint | 449-457 | invariant 459-461 | **A** | keep (frozen invariant) |
| `max_delta_rank_spread` / `max_param_rank_diff` | 331/460 | invariant raises (A); stats line (C) | **A + C** | keep |
| stats/event log lines | 265-279, 468-487 | report only (host floats/ints, no sync) | **C** | keep (already 1/10-obs cadence + event lines) |
| forensic dump | 338-394 | failure path only | **C** | keep (not a per-step cost) |

### Key proofs required by the spec before touching anything

1. **`sigf` is DEAD in this candidate.** Computed (162), stacked + read (170,
   bare expression with discarded result), stored in `_PreparedSpec.sigf` but
   never read anywhere in `HybridCMuonCanonicalNS4FP32Rescue.step()` or its
   helper calls. The only reader in the whole module tree is the base
   `step()` (guarded canonical candidate, line 344) via `_is_low_signal`
   (248-270) — a different production candidate whose behavior is untouched.
   Removal changes no safety decision, no checkpoint byte, no telemetry
   output. **D class.**
2. **`sig` is B (checkpoint-critical), not telemetry.** The per-step
   reference update (440-442) consumes the batched `sig` values and mutates
   `_refs`, which is persisted into the guard `references` table
   (`_guard_state`, guarded_canonical.py:665-668) and strictly validated on
   load (key-set match, 728-729). Removing the readback would freeze the
   reference table and break resume-state continuity (the D4 observation of
   decaying `ref_max` is this update running). Disposition: keep the single
   batched readback (1 sync, already minimal); do NOT move to checkpoint-time
   (the update is per-observation by contract).
3. **`_refs` is decision-inert in this candidate** (proof above, line
   "consumed by"): it can be safely kept/updated, and no rescue predicate may
   be assumed to depend on it. Cleanup keeps the update bit-for-bit.
4. **`fail_flags` semantics are frozen and rank-agreement is exact.** Owner
   sets codes device-side (`_NONFINITE`=1, `_CEILING`=2, cleared to 0 on
   rescue), `all_reduce(MAX)` makes every rank see every flag before
   broadcast/fingerprint/raise. The batched rewrite must preserve: code
   values, MAX reduction, rescue-before-reduce ordering, and the flag-clear
   on successful rescue.

## 3. Cleanup plan derived from the audit

### Cleanup A — remove retired hot-path telemetry (commit 1)
- `fp32_rescue.step()`: delete the `sigf` compute (line 162), delete the
  discarded bare-expression readback (line 170), pass `sigf=[]` to
  `_PreparedSpec` (field retained for the base candidate).
- Net effect: −1 host sync/rank, −166 device L2 norms/step, −1 stack alloc.
- `SAFETY_DECISION_CHANGE = NO`.

### Cleanup B — batch post-NS safety host verdicts (commit 2)
Design (all on device until the marked reads):
1. Owner NS loop: stage every owned delta; set `fail_flags[fi-1]` with
   device-side `torch.where` on `~isfinite(delta).all()` (`_NONFINITE`) and
   on `rms > ceiling` (`_CEILING`, only where flag still 0) — replaces
   syncs #7 (83) and #9 (1). `d_rms_list` kept device-side for the failure
   forensic dump only.
2. Owner rescue decision: one packed read `fail_flags.tolist()` (1 sync)
   replaces the 166-iteration `.item()` loop (#10). Non-owner ranks need no
   read (they skip rescue by construction and reach the same
   `all_reduce`).
3. Rescue internals: combine the two per-rescue reads (finite + rms) into
   one stacked `tolist` (2 → 1 sync per rescue).
4. After `all_reduce`: compute `failure_msgs` and the `failed` set from ONE
   `fail_flags.tolist()` (removes duplicate read #14).
5. Fingerprint spread read (#16) and param invariant read (#17) unchanged
   (frozen invariants, section 2 of the spec); broadcast phase unchanged
   (per-chunk, frozen).

Expected cleanup totals: non-owner ≈ **5** syncs/step (grad-validation,
sig, verdict-to-lish, fingerprint, param-invariant); owner ≈ **6** (+ packed
rescue-mask read); +1 per rescue event. Every remaining sync is justified:
4 are pre-existing batched reads (1 frozen base check + 1 checkpoint-state
feed + 2 frozen invariants), 1-2 are the spec's own "packed unsafe mask" and
"final host verdict".

### Untouched (frozen contract)
All optimizer math, NS functions, coefficients, owner mapping
(`stable_owner`, OWNER_MAPPING_VERSION), reference-update formula,
rescue math (`cmuon_zeroth_power_fp32` + re-check band), two-phase commit
structure, per-chunk broadcast payload/owner/dtype, checkpoint schema
(`GUARD_SCHEMA_VERSION`, `guarded_canonical_schema_version`,
`hybrid_cmuon_schema_version=1`, `fp32_rescue` counter block, `references`
table, `observations`, skip fields), config schema, base-class legacy
candidate behavior.

## 4. Production dependency audit (section 22/23 inputs)

- `PRODUCTION_DEPENDS_ON_RETIRED_D1 = NO` — no low-signal skip, no amplitude
  gate, no structural classifier call in `fp32_rescue.step()`; `sigf`
  (the D1 fro-floor input) is removed by Cleanup A.
- `PRODUCTION_DEPENDS_ON_FORENSIC = NO` (with note) — the ONLY production
  import from `cmuon_forensic.py` is the `CMuonSafetyError` exception class
  (fp32_rescue.py:61, also re-exported by guarded_canonical.py:61-66 for
  test parity). `ForensicMonitor`/`ForensicConfig` are not constructed on
  the guarded path (`production.py:455-460` actively rejects the forensic
  config for it). The in-step forensic DUMP (fp32_rescue.py:338-394) is
  failure-only, writes to the artifacts dir, and never gates the
  fail-closed raise. → exception-type import compatibility only, explicitly
  marked.
- D1 structural classifier / pre-NS failed-guard experiments / forensic
  dump path: NOT in the production normal hot path (the base `step()` and
  `ForensicMonitor` belong to other, non-active candidates).
