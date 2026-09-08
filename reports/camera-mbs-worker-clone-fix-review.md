# MBS worker-clone propagation fix — review evidence

**Branch**: `camera-v2-mbs-worker-clone-fix-review`
**Base (evidence SHA)**: `bbf744d204dd3cdd985cd13b3ab5298fe1b95b5b`
(hosts: `camera-v2-mbs-canary-100u-evidence` terminal)
**Mode**: review-only. No training, no backward, no optimizer step, no
checkpoint write, no P50, no production traffic.
**Date**: 2026-07-08 (runtime host: come7)

---

## 1. Errata to the 100U canary report

These correct the record for
`reports/camera-mbs-canary-100u-run.{md,json}` (file itself immutable).

### ERRATUM-1 — the old 100U exposure root-cause claim is FALSE

The 100U report attributed `mirror_eligible == mirror_selected ==
mirror_applied == 0` to a **dataset × policy mismatch**: "no severe
vertical rows existed in the first 100U exposure, so the mirror policy
correctly had nothing to do."

That is false. The canary metrics themselves (rank-0 telemetry cohort,
40000 logical rows) prove the severe population existed:

| counter (100U, vertical-applied population) | value |
|---|---|
| `camera_mirror_vertical_applied` | 5620 |
| `camera_mirror_severity_lt2` | 3399 |
| `camera_mirror_severity_2to4` | 1892 |
| `camera_mirror_severity_ge4` | 329 |
| **rows with latent shift ≥ 2.0** | **2221** |

The code invariant `severity_lt2 + severity_2to4 + severity_ge4 ==
vertical_applied` (enforced by the `CameraMirrorCounts` validator) holds
(3399 + 1892 + 329 = 5620), so the severity histogram is a partition of
the vertical-applied population — 2221 of them carried
`camera_latent_center_shift >= 2.0` and were eligibility-eligible under
the v1 policy (`min_latent_shift = 2.0`). A healthy pipeline would have
reported `eligible == 2221`, not 0.

**Corrected root cause**: a propagation bug.
`ProductionPipelineFactory.pipeline_for_lease()` passes `mirror_policy`
to the parent `WebDatasetPipeline`, but the persistent-worker clone
boundary `WebDatasetPipeline._with_local_shards()` copied
`spatial_policy`/`camera_policy`/`transparent_policy` and **omitted
`mirror_policy`**. Every spawned worker therefore ran with the
`__init__` default `mirror_policy = None`; the worker-side eligibility
gate (`if self.mirror_policy is not None and self.mirror_policy.enabled`)
never fired; every row came back `mirror_eligible = False` with no
mirror payload — `eligible/selected/applied` all 0 regardless of the
actual severe population.

### ERRATUM-2 — EARLY_ACTIVITY_GATE_ENFORCEMENT = FAILED

The 100U report left the monitoring-session death (~U118122, 03:24
local, liangshen `maxTokens=1024` reasoning-only) adjacent to the
early-activity gate in a way that could be read as "the gate could not
be enforced because the session died."

That reading is wrong. The gate's target was **U118103** (cumulative
`mirror_applied > 0` before update 118103, per the launch contract).
The session died ~19 updates **after** the gate point. The live session
had already observed `mirror_eligible == 0` through U118103 and
beyond and did **not** stop the run.

Correct phrasing: **EARLY_ACTIVITY_GATE_ENFORCEMENT = FAILED** — the
live session saw the gate condition fail (eligible=0 past 118103) and
did not act. The later session death does not explain, mitigate, or
relocate the failure.

### ERRATUM-3 — `effective_batch=400` terminology

The 100U report's `effective_batch=400` is the **rank-0 telemetry
cohort** (what `metrics.jsonl` aggregates), **not** a global batch of
80000. Accordingly, ratios such as 2221/40000 = 5.5525% are descriptive
statistics over the rank-0 cohort, not global rates. This does not
affect any verdict in this report (all gates are cohort-internal
conservation laws).

---

## 2. Root cause and fix

### 2.1 Root cause (code-confirmed)

Only two `WebDatasetPipeline(` construction sites exist in `src/`:

1. `src/sakuramoon/data/production.py` — `pipeline_for_lease()` (the
   parent pipeline; passes `mirror_policy=mirror_policy`).
2. `src/sakuramoon/data/pipeline.py` — `_with_local_shards()` (the
   **exact persistent-worker clone boundary**; spawned once per worker,
   per shard).

Site 2 omitted `mirror_policy` from the clone constructor call, so the
clone fell back to the `__init__` default `mirror_policy = None`. The
mirror eligibility gate inside `_process` is
`if self.mirror_policy is not None and self.mirror_policy.enabled:`,
which never fires in the workers — i.e. in exactly the population the
telemetry counts.

### 2.2 Fix (single line, production)

`src/sakuramoon/data/pipeline.py`, in
`WebDatasetPipeline._with_local_shards()`:

```diff
             cycle_index=cycle_index,
             spatial_policy=self.spatial_policy,
             camera_policy=self.camera_policy,
+            mirror_policy=self.mirror_policy,
             transparent_policy=self.transparent_policy,
             transparent_telemetry=self.transparent_telemetry,
         )
```

No other production change. No geometry, threshold, RNG, weighting,
RoPE/DiT/Qwen/VAE/conditioner/objective/optimizer/CMuon/scheduler/
checkpoint-schema change.

---

## 3. New test suite

`tests/unit/data/test_camera_mirror_worker_clone.py` — 7 tests, all
passing on the fixed tree, **0 skip / 0 xfail**:

| # | test | what it pins |
|---|---|---|
| 1 | `test_clone_preserves_policy_identity` | `_with_local_shards` clone keeps `is`-identity of all four policies (spatial/camera/mirror/transparent) + telemetry + cycle_index + lease-managed flag |
| 2 | `test_clone_preserves_none_policies` | `None` mirror/spatial/transparent are preserved as `None` (no accidental defaulting) |
| 3 | `test_v1_eligibility_boundary_values` | `camera_mirror_eligible` boundary: shift 1.99→False, 2.0→True, 4.5→True, horizontal 3.0→False, odd viewport→False, not-applied→False |
| 4 | `test_cloned_pipeline_applies_severe_vertical_rows` | through the **cloned** pipeline, row-level `mirror_eligible == (shift >= 2.0)` exactly; payload present iff severe; horizontal/non-applied never eligible |
| 5 | `test_canary_policy_invariant_on_cloned_pipeline` | loads the **v2 canary toml**, asserts policy values (2.0/1.0/1.0/v1 mode) and `eligible == severe_vertical` over a clone scan |
| 6 | `test_persistent_worker_exposes_mirror_through_clone` | **real spawn integration**: `_PersistentShardDataset → _with_local_shards → _iter_paths → collate_samples` over a real tar; asserts full §11 conservation chain on worker-produced batches |
| 7 | `test_v2_canary_config_differs_only_in_identity` | resolved-toml diff vs v1 canary == exactly the 7 identity/output keys |

### 3.1 BASE-FAIL proof (spec §6)

The same test 6 was run against the **unfixed base** `bbf744d`
(0 occurrences of `mirror_policy=self.mirror_policy` in
`pipeline.py`):

```
FAILED tests/unit/data/test_camera_mirror_worker_clone.py::test_persistent_worker_exposes_mirror_through_clone
AssertionError: row ImageAudit(..., camera_policy='hdm_shifted_square_v2',
  camera_selected=True, camera_applied=True, camera_orientation='vertical',
  ..., camera_latent_center_shift=7.0, ...) severe vertical row lost its
  mirror payload in the persistent worker
assert None is not None
1 failed in 8.19s
```

The failing row is a forensic specimen of the production symptom: a
severe vertical row (`camera_orientation='vertical'`,
`camera_latent_center_shift=7.0 ≥ 2.0`) whose mirror payload was lost in
the persistent worker. On the fixed tree the same test passes.

---

## 4. Real service-worker exposure smoke (spec §11)

Real production data path on the runtime host (come7), **data-only**
(no Qwen/VAE forward, no backward, no optimizer, no checkpoint):

```
ProductionPipelineFactory.from_config(v2 canary config)
  -> factory.batches(lease_client)
     -> ConfiguredDataLoader / iter_service_batches
        -> _PersistentShardDataset (spawned DataLoader workers,
           persistent_workers=True, multiprocessing_context=spawn)
           -> pipeline._with_local_shards
           -> shard_pipeline._iter_paths
           -> collate_samples
```

Leases were served by a local client over **3 real
`webdataset_danbooru_v2` shards** (6.44 GB, sha-verified against the
dataset manifest) using the repository's own
`ModelScopeDatasetTransport`; the client cycles shards with queue
semantics (a path is re-leased only after its previous lease is
acknowledged). Tool: `scripts/mbs_worker_exposure_smoke.py`.

Result (early-stopped at the spec's `severe_vertical >= 20` gate):

```json
{
  "schema": "mbs_worker_exposure_smoke_v1",
  "config": "train_g1_camera_v2_p25_mirror_v2_canary.toml",
  "batches_consumed": 8,
  "logical": 160,
  "camera_selected": 88,
  "camera_applied": 80,
  "vertical_applied": 60,
  "severity_lt2": 39,
  "severity_2to4": 19,
  "severity_ge4": 2,
  "severe_vertical": 21,
  "eligible": 21,
  "selected": 21,
  "applied": 21,
  "degraded": 0,
  "extra_views": 21,
  "physical_views": 181,
  "gates": {
    "severe_vertical_gt_0": true,
    "eligible_eq_severe_vertical": true,
    "selected_eq_eligible": true,
    "applied_eq_selected": true,
    "degraded_zero": true,
    "extra_views_eq_applied": true,
    "physical_eq_logical_plus_applied": true,
    "physical_gt_logical": true
  },
  "verdict": "PASS"
}
```

All §11 gates hold: `severe_vertical=21 > 0`;
`eligible(21) == severe_vertical(19+2=21)`; `selected == eligible`;
`applied == selected`; `degraded == 0`; `extra_views(21) ==
applied(21)`; `physical_views(181) == logical(160) + applied(21)` and
`181 > 160`. 16 transparent rejections on real danbooru rows (9
`ai_image_corrupted`, 5 `reject_missing_alpha`, 1 `no_upscale`, 1
`retention`) — expected on the real corpus.

## 5. Optional HCU end-to-end forward (spec §12)

**SKIPPED (optional per spec).** The §11 smoke already exercises the
complete real data path (production factory → spawn workers → clone →
collate) on real shards with the v2 config; the fix's correctness is
established without model-side evidence. A forward-only HCU pass
(Qwen+VAE+compiled DiT on one mirrored batch, no backward/optimizer/
checkpoint) would require re-assembling the trainer's private module
stack (encoder/VAE/DiT build, packing, conditioning) outside the
trainer, which risks a subtly-divergent harness proving less than the
spec requires. If the external reviewer wants it, it should be run
through a governed trainer entry point in a later, explicitly
authorized round.

---

## 6. Test suites and static checks

- New test file: **7 passed** (0 skip / 0 xfail) on the fixed tree.
- `tests/unit/data/` (related scope): **366 passed**
  (`test_deepghs_quality_pipeline.py` +
  `test_run_deepghs_quality_pipeline.py` excluded: pre-existing
  collection error, `onnxruntime` absent from the DTK venv —
  environment issue, unrelated to this change).
- Full `tests/unit/` suite: **962 passed, 4 failed** — the 4 failures
  are exactly the pre-existing cmuon forensic environment failures
  (`tests/unit/optim/test_cmuon_fp32_forensic.py` ×4: bcd_hard_fail,
  e_writer_failure, f_serialization_failure, m2_mirror_failure); the
  set is identical to the baseline, **no new failures**. Same two
  deepghs collection-error exclusions as above.
- `ruff check` (changed scope: `pipeline.py`, smoke tool, new test
  file): **All checks passed**.
- `py_compile` (all three changed files): OK.
- `git diff --check`: clean.
- Staged scope contains no secrets, no checkpoints, no binary files.

---

## 7. Control terminal & treatment source (immutable)

- U118200 control terminal:
  `output_model/g1_camera_v2_p25_mirror/ckpt_118200_raw-118200-update-cadence`
  — manifest `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7`
  (interpretation CONTROL_100U_CAMERA_ONLY). **Untouched.**
- Future corrected-treatment source:
  `output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence`
  — manifest `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892`.
  **Untouched.** No training, resume, or launch occurred in this round.

## 8. v2 review config

`config/train_g1_camera_v2_p25_mirror_v2_canary.toml` extends the v1
canary and overrides **only** the experiment identity and output
locations (resolved-toml diff = exactly these keys, pinned by test 7):

```
run.run_id
paths.run_dir
paths.checkpoint_dir
paths.artifact_dir
logging.local_jsonl_path
wandb.retry_jsonl_path
evaluation.output_dir
```

The mirror policy (v1 mode, `min_latent_shift = 2.0`,
`pair_probability = 1.0`, `pair_weight = 1.0`), camera viewport policy,
governed stop cap 118200, optimizer, LR, batch, scheduler, compile and
architecture are inherited unchanged. REVIEW ONLY — not deployed, not
launched.

## 9. Verdict

- Old 100U exposure root-cause claim: **FALSE** (erratum 1).
- EARLY_ACTIVITY_GATE_ENFORCEMENT: **FAILED** (erratum 2).
- New root cause (worker clone drops `mirror_policy`): **CONFIRMED** by
  code inspection + base-fail proof + real-worker smoke.
- Fix: single line, minimal, no side channels.
- §11 conservation chain on the real production worker path: **PASS**.
