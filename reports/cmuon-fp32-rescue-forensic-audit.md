# CMuon FP32-rescue forensic dataflow audit (F1, spec §4)

Scope: full read of `src/sakuramoon/optim/cmuon.py`, `fp32_rescue.py`,
`guarded_canonical.py`, `cmuon_forensic.py` and the related tests at BASE
`f045f87` (production as of the 112126 hard failure), plus the F1 telemetry
delta on branch `cmuon-fp32-forensic`. Line numbers below refer to the
forensic tree; the base/forensic difference is inventoried in §4 of this
document and is telemetry-only.

## 1. Production dataflow (base, unchanged by F1)

```
step()  (fp32_rescue.py:152)
│
├─ PHASE 1: PREPARE (per CMuon spec, in parameter-name order)  [fp32_rescue.py:172-205]
│    grad (BF16 leaf, finite-checked at :154)
│      → buf = self._momenta[param]; buf.lerp_(grad, 1-mu)      ← IN-PLACE momentum EMA
│      → nesterov = grad.lerp(buf, mu)
│      → chunks = split(nesterov)                               (chunk_size / chunk_dim from routing)
│      → alphas = cmuon_moonlight_alpha per chunk               (cmuon.py:246)
│      → sig  = per-chunk RMS (fp32 read)
│    one batched signal sync (guarded canonical base)
│
├─ PHASE 2: OWNER NS + SAFETY (per flat chunk input)            [fp32_rescue.py:211-355]
│    if stable_owner(fqn,chunk) == rank:                        (FNV-1a 64 % world_size)
│      ns    = cmuon_zeroth_power_bf16(chunk)                   (cmuon.py:112; quintic, Frobenius clamp)
│      delta = (-alpha) * ns                                    (BF16)
│      staged[idx] = delta
│      flag = NONFINITE if ~isfinite(delta).all()
│               else ABOVE_CEILING if rms(delta).double() > ceiling   (fp64 compare, :162-165)
│      d_rms_list.append(rms(delta))                            ← owner-local BF16 delta rms
│    FP32 rescue (owner rank, flagged inputs only)              [fp32_rescue.py:283-340]
│      ns32    = cmuon_zeroth_power_fp32(chunk.float())         (cmuon.py:163; same coeffs)
│      delta32 = (-alpha) * ns32                                (FP32)
│      rms32   = delta32.pow(2).mean().sqrt()                   ← local variable (loop scope)
│      finite32 = torch.isfinite(delta32).all()                 ← on the FP32 delta, pre-BF16 staging
│      if finite and floor ≤ rms32 ≤ ceiling:  rescue: staged[idx] = delta32.bfloat16(); flag = 0
│      else: hard fail — flag stays > 0
│            (BASE: rms32 / finite32 / reason are consumed by the branch only — NOT SAVED)
│
├─ RANK-CONSISTENT VERDICT                                       [fp32_rescue.py:357-373]
│    all_reduce(fail_flags, MAX)    ← every rank learns every failure + severity
│    failure_msgs = [...f"{name}: {fqn}#chunk{c}"] (deterministic flat-index order)
│
├─ OWNER BROADCAST (world_size > 1)                              [fp32_rescue.py:374-392]
│    per active NON-failed chunk: owner broadcasts staged BF16 delta to all ranks
│    failed chunks: staged = None on every rank (consensus zero delta)
│
├─ CROSS-RANK FINGERPRINT                                        [fp32_rescue.py:395-415]
│    per-rank delta rms (float) → all_reduce MIN/MAX → spread
│    spread != 0 → failure_msgs += "cross-rank delta fingerprint spread ..."
│
├─ HARD FAIL branch (if failure_msgs)                            [fp32_rescue.py:417-533]
│    legacy JSON dump (per failed input): failure, fqn, chunk, owner, shape,
│      u_t_rms (sig), lr, target, ceiling, delta_rms (= d_rms_by_idx, BF16 rms,
│      owner-only; null on non-owner)
│    [F1: + fp32 fields + exact-input artifact publish + diagnostic trace — §4 below]
│    → RAISE CMuonSafetyError                                    ← BEFORE PHASE 3
│
└─ PHASE 3: COMMIT (two-phase atomic, never reached on hard fail) [fp32_rescue.py:535-600]
     AdamW part: self.sr_rng.run_step(self.optimizer.step)       (isolated SR RNG)
     CMuon part: per spec: parameter.mul_(decay); parameter.add_(reassembled delta)
     reference table update (_refs decay/max)
     self.observations += 1
     optional post-commit invariant check (all-rank parameter fingerprint)
```

## 2. Spec §4 audit questions (answers, base semantics)

1. **Where does the legacy forensic record's `delta_rms` come from?**
   From `d_rms_list` (fp32_rescue.py:247-249): the RMS of the **BF16 attempt**
   delta, `delta.pow(2).mean().sqrt()` with `delta = (-alpha) *
   cmuon_zeroth_power_bf16(chunk)`, computed on the owner rank only. Non-owner
   ranks never run the BF16 NS, so the field is `null` for them. The legacy
   name `delta_rms` is misleading: it is the BF16-attempt value, **not** the
   FP32 rescue value. (F1 renames it `bf16_delta_rms` and keeps `delta_rms`
   as a deprecated alias with the identical value.)

2. **In what scope is `rms32` lost?**
   `rms32 = delta32.pow(2).mean().sqrt()` is a local variable in the per-input
   FP32-rescue branch (fp32_rescue.py:296, owner rank only). The base code
   consumes it solely for the reason classification (`rms32 > ceiling` /
   `< rescue_floor`); it is never written to the legacy dump or anywhere
   else. This is the observation gap F1 closes.

3. **How is `finite32` computed?**
   `bool(torch.isfinite(delta32).all().item())` on the **FP32 rescue delta**
   `delta32 = (-alpha) * cmuon_zeroth_power_fp32(chunk.float())` — i.e. the
   finiteness of the FP32 value itself, before any BF16 staging cast.

4. **`rescue_floor` constant and numeric values.**
   `_RESCUE_SANITY_LOW = 0.05` (fp32_rescue.py:93-94, validated constant, not
   a tunable); `rescue_floor = 0.05 * (0.2 * lr)`. With the recorded
   production lr = `1.5624999650754035e-04` (HCU rounds the literal 1.5625e-4
   to BF16 before the band math — tests pin against the recorded value):
   target = `3.124999930150807e-05`, ceiling = `3.124999930150807e-04`,
   rescue_floor = `1.5624999650754036e-06`.

5. **Is the parameter commit not yet happened at hard fail?**
   Yes. The commit is PHASE 3 (fp32_rescue.py:535-564), strictly after the
   hard-fail `raise CMuonSafetyError` (:533). Verified: unit test G
   (hard-fail commits nothing: parameters, AdamW state, SR RNG untouched) and
   2-rank scenarios A/B/C (identical parameter fingerprint before/after on
   both ranks).

6. **Is the momentum buffer already updated in memory before the hard fail?**
   **Yes — this is the one optimizer state that advances on a hard fail.**
   The momentum EMA `buf.lerp_(grad_md, 1.0 - mu)` is in-place in PHASE 1
   (fp32_rescue.py:180), before NS/verdict. So at hard-fail time: CMuon
   momentum advanced by one step (in memory), parameters unchanged, AdamW
   state + SR RNG unchanged (the AdamW step is PHASE 3, :537), `_refs`
   unchanged (:579-581), `observations` unchanged (:584). This is the
   pre-existing production semantics; F1 neither changes nor masks it.

7. **Owner-only rescue / all-reduce semantics.**
   Only the owner rank (FNV-1a 64 of (fqn, chunk) mod world_size) runs the NS
   for a chunk (BF16 and FP32). `fail_flags` is a per-flat-input int64 tensor
   all-reduced with MAX, so every rank learns every failure with its severity
   class; the `CMuonSafetyError` message is built from the all-reduced flags
   in deterministic flat-index order, so **all ranks raise the identical
   message** (2-rank test asserts byte-identical messages via
   `all_gather_object`). Non-failed active chunks are broadcast owner→all as
   plain BF16 tensors; failed chunks are a consensus zero (None staged on
   every rank). A cross-rank fingerprint (MIN/MAX all-reduce of per-rank
   delta rms) makes broadcast corruption an independent hard-fail reason.
   Because the raise happens before PHASE 3 on every rank, a partial commit
   across ranks is structurally impossible.

## 3. F1 telemetry delta (telemetry-only inventory)

All changes in `fp32_rescue.py` are (a) dormant on the success path or
(b) on the hard-fail branch only. Proven by §14 bit-parity: base vs forensic
trees are byte-identical across 10 steps including a forced 112126-class
pathologic step (BF16 band trip → FP32 rescue success) — params, CMuon
momenta, AdamW8bit state (incl. 8bit moments + SR RNG), guard refs, counters.

| Change | Location | Path | Feeds verdict? |
|---|---|---|---|
| `fp32_verdicts` dict capture `{original_fp32_delta_rms, original_fp32_finite, fp32_failure_reason}` | fp32_rescue.py rescue branch | hard-fail inputs, owner only | NO (reads of local `rms32`/`finite32` after the branch) |
| legacy record fields: `bf16_delta_rms` (+ `delta_rms` alias), `fp32_delta_rms`, `fp32_finite`, `fp32_failure_reason`, `fp32_rescue_floor`, `fp32_ceiling` | legacy dump builder | hard-fail branch | NO |
| exact-input artifact publish (`input.safetensors`/`.pt` + `metadata.json`, owner-only, atomic tmp+fsync+rename, `-rN` collision suffix) | `cmuon_hardfail.py` + publish loop | hard-fail branch, after legacy dump, before raise | NO (publish errors are logged/wrapped, never mask the `CMuonSafetyError`) |
| diagnostic NS trace (`cmuon_ns_trace.trace_ns_replay`, both working dtypes, on a clone of the saved input) | forensic dump path | hard-fail branch | NO (op-exact read-only replay; result stored in metadata, never written back) |
| constructor kwargs `hard_fail_artifact_root`, `legacy_forensic_dir` | `__init__`/`build_fp32_rescue` | construction | NO (defaults = production paths: `/sakuramoon-runtime/artifacts/g1/cmuon-hard-fail`, `/sakuramoon-runtime/artifacts/g1`) |

`cmuon.py` and `guarded_canonical.py`: **zero diff vs f045f87** (spec §19
frozen-algorithm gate; verified by `git diff f045f87 -- <files>` = empty).

## 4. Diagnostic trace guarantees (referenced as "§7" by `cmuon_ns_trace.py`)

* The replay op sequence (cast → transpose-to-wide → Frobenius-clamp
  normalization → quintic `addmm` iterations) replicates
  `cmuon_zeroth_power_bf16` / `cmuon_zeroth_power_fp32` exactly; the extra
  per-iteration norm/rms/max/finite reductions are side-effect-free reads and
  never feed back into the working matrix.
* The replay output is **never written back**: it cannot change the fail
  flag, staged delta, momentum, parameters, owner broadcast, or commit.
  Production results already happened before the trace runs.
* If the replay itself fails, the caller records `forensic_trace_error` in
  the artifact metadata and the original `CMuonSafetyError` is still raised —
  an analysis failure never masks a production failure.
* `cmuon.py` is untouched; the trace lives in `cmuon_ns_trace.py`, imported
  only by the forensic dump path and the dev-tools.

## 5. Why recorded-original, in-process diagnostic replay, and offline replay can differ

Three numbers exist by design and are kept separate in every record:

1. **recorded original** (`original_fp32_delta_rms`/`bf16_delta_rms`): what
   the production verdict actually compared — captured at fail time.
2. **diagnostic replay at fail time** (`diagnostic_replay_*` in metadata):
   op-exact trace re-run on the same device, same process, microseconds
   later.
3. **offline replay** (dev-tool `cmuon_fp32_rescue_replay.py`): production NS
   re-run on the saved exact input in a later process (possibly another
   machine/driver).

Differences between (1) and (2)/(3) are expected in two documented cases:
(a) **HCU kernel nondeterminism** — the BF16 NS is nondeterministic across
identical calls on HCU (stress harness measured per-repeat spreads up to
≈3.3e-07 in delta rms at production shape); (b) **test injection** — the
2-rank test forces NS *outputs* (×1e9 / fill-inf / fill-nan), so the saved
*input* is clean and replays clean; the recorded/replay ratio is exactly the
injection factor (measured 1e9). Neither case is a product defect; on a real
production hard fail with no injection, (1)≈(2)≈(3) within kernel
nondeterminism.

## 6. Exact-input artifact (spec §9: atomic / fail-safe)

* Layout: `<root>/obs-<observations>-rank<R>-<fqn_dashes>-chunk<C>/`
  containing exactly one tensor file (`input.safetensors` preferred;
  `input.pt` fallback) + `metadata.json`.
* Owner-only publish; the non-owner never touches the root (2-rank test
  asserts the non-owner root stays empty — no fabrication).
* Atomic: temp file in the same directory → fsync → `rename`; a crashed
  publish leaves no torn event (the event dir appears only when complete).
* Crash loop: re-failure at the same (obs, rank, fqn, chunk) appends
  `-r2`, `-r3`, ... — earlier events are never modified (2-rank scenario C
  asserts the first event's metadata sha is unchanged after the second
  failure).
* `metadata.json` is strict JSON (`allow_nan=False`): nonfinite rms values
  are recorded as `null` with the `*_finite` flags carrying the state; the
  legacy JSON (pre-F1 consumer format) keeps `allow_nan=True`
  (`Infinity`/`NaN` round-trip) for backward compatibility.
* Publish failure can never mask the verdict: all non-
  `HardFailArtifactError` exceptions are wrapped into a
  `HardFailArtifactError` that is logged (via `stats_logger`) and swallowed;
  the `CMuonSafetyError` raise is unconditional.

## 7. 112126 — what F1 can and cannot answer

* **Cannot**: the 112126 NS input was never saved (pre-artifact), so an
  exact replay of that specific event is impossible. The legacy dump of the
  event gives: fqn `dit.blocks.slot_02.attention.content_gate.weight`,
  chunk 0, owner rank0, shape [2560,2560], `u_t_rms = 1.6460275276131142e-07`
  (input RMS — the "CENTER" scale of the stress scan), bf16 delta rms
  `9.00267114496e+11` (≈2.88e16 × target — far above the ceiling), and **no**
  FP32 rms/finite/reason (the gap).
* **Can (next real hard fail)**: the deployed telemetry captures the original
  FP32 rms + finite + reason class (NONFINITE / ABOVE_CEILING / BELOW_FLOOR),
  saves the exact input (owner, original dtype), and the offline replay CLI
  re-runs the production NS bit-for-bit reproducible (modulo HCU kernel
  nondeterminism, documented in §5).
* **Stress map (synthetic, production shape [2560,2560], 4 scales × 9
  families × 3 repeats, HCU)**: at scales 1e-8 … 1e-6, **no family trips the
  BF16 band** (BF16 failures = 0; scale alone is not the BF16 failure dial —
  NS normalizes by Frobenius norm, so the 112126-class BF16 blow-up is a
  structure/conditioning phenomenon beyond the RMS scale scan); **FP32
  below-floor = a whole input class**: `constant` (all scales) and `rank-1` /
  `rank-8` (all scales) collapse below `rescue_floor` — the category the
  legacy dump could not distinguish from above-ceiling/nonfinite. FP32
  above-ceiling and FP32 nonfinite were not produced at these scales.
  Repeat determinism: BF16 spread > 0 for all non-constant families
  (1e-10 … 3.3e-07), exactly 0 for constant — documenting HCU BF16 kernel
  nondeterminism that the recorded/diagnostic separation must absorb.

## 8. Verification index (gates → evidence)

* §14 bit-parity (base vs forensic): PASS — 10 steps, all fingerprint fields
  byte-identical (`dev-tools/cmuon_fp32_parity.py`;
  `/sakuramoon-runtime/cmuon-f1/out/parity-{base,forensic}.json`).
* §15/§18 2-rank HCU DDP: PASS — `tests/gpu/optim/cmuon_fp32_forensic_2rank.py`,
  4 scenarios (above-ceiling / nonfinite / crash-loop / successful-rescue),
  report `/sakuramoon-runtime/cmuon-f1/out/f1-2rank.json`.
* §11 stress: done — `/sakuramoon-runtime/cmuon-f1/out/stress/cmuon-fp32-rescue-stress.{json,md}`.
* §10 replay CLI: verified end-to-end on a real published artifact
  (`/sakuramoon-runtime/cmuon-f1/out/replay-a.json`).
* §19 frozen algorithm: zero diff (cmuon.py, guarded_canonical.py).
* Unit: `tests/unit/optim/test_cmuon_fp32_forensic.py` 11/11 (HCU).
