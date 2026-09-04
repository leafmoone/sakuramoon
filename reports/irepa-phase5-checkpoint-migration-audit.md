# iREPA Phase 5 — Explicit Checkpoint Migration Audit

Phase 5 of the `iprea` branch adds the explicit, production-safe migration path
from a no-iREPA (v3) RAW checkpoint to an iREPA (v4) checkpoint: a
deterministic fresh projector in a new safetensors shard, canonical-FQN AdamW
parameter-ID remap (every pre-existing parameter keeps its exact numeric ID and
state), the persistent iREPA schedule anchor sidecar, and the fail-closed
load/save gates that bind them. No production code path changes behavior for a
checkpoint that is not being migrated.

## Heads

| ref | commit | role |
|---|---|---|
| source P4 head | `00bd795` | iprea HEAD the Phase 5 tree starts from ("docs: record iREPA phase 4 gate"); also the source from which the Phase 5 v3 checkpoints are built |
| P4 functional | `c56c7c1` | "iREPA phase 4: integrate representation alignment training graph" |
| P4 evidence head | `00bd795` | the P4 gate record commit; the Phase 4 audit reports live at this head |
| Phase 5 head | `00bd795` (parent) | the Phase 5 functional commit is created ON this parent and contains these reports (P4 convention: report records `parent_commit`; the functional SHA is unknowable inside its own report) |

Immutable chain (no amend/squash/rebase): `4b5bbd9 → ce7c36d → 91d340f →
14c9cee → c56c7c1 → 00bd795 → <phase 5 functional commit>`.

## Host

salt13, 2x BW DCU, DTK, system torch `2.9.0+das`, Python 3.11.9, venv
`/sakuramoon-runtime/sakuramoon-dtk-venv` (restored 2026-09-03 from the salt10
environment image; sha256-verified). All GPU evidence in this report was
produced on this host.

## What changed (working tree over `00bd795`)

**New (functional):**
- `src/sakuramoon/checkpoint/migrate_irepa_checkpoint.py` — the migration
  itself + CLI (`--dry-run` plan) + `validate_irepa_state_document` /
  `read_irepa_state` / `irepa_state_anchor` (the anchor API).

**Modified (functional):**
- `src/sakuramoon/checkpoint/load.py` — fail-closed iREPA gates on resume: a
  v4 composite requires the `train_state/irepa_state.json` sidecar; the
  schedule anchor is validated and bound to the resume lambda; a v3
  checkpoint under an iREPA-enabled config is rejected with an explicit
  "run the migration" error; routing-manifest comparison guards the hybrid
  optimizer state against silent positional AdamW mis-assignment.
- `src/sakuramoon/checkpoint/save.py` — v4 saves re-persist the anchor
  sidecar verbatim (anchor is immutable per migrated checkpoint); the save
  gate rejects an iREPA-enabled save without a valid sidecar.
- `src/sakuramoon/checkpoint/artifact.py` — `build_trainable_composite`
  builds the v4 architecture document (`training_auxiliaries.irepa`);
  optimizer coverage validation is name-sorted (order-independent).
- `src/sakuramoon/optim/groups.py` — §11 fix: `audit_trainable_parameters`
  appends the iREPA projector specs AFTER the FQN-sorted existing set, so the
  AdamW SR RNG consumption order is stable: every pre-existing parameter
  keeps its exact draw position and the projector's draw comes last.
- `src/sakuramoon/train/preflight.py` / `src/sakuramoon/train/production.py`
  — production readiness gate consumes the sidecar anchor (fail-closed) and
  binds the resume lambda schedule to it.
- `tests/unit/train/test_irepa_production_gate.py` — production-gate tests
  extended for the anchor/sidecar contract.

**New (tests):**
- `tests/unit/checkpoint/test_irepa_checkpoint_migration.py`
- `tests/gpu/checkpoint/test_irepa_checkpoint_save.py`
- `tests/gpu/irepa/test_irepa_zero_lambda_optimizer_parity.py`
  (spec-18 S18 gate, two modes) + `tests/gpu/irepa/s18_chain.py` (shared
  chain) + `tests/gpu/irepa/irepa_ddp_lambda_zero_smoke.py` (2-rank smoke,
  torchrun, not pytest).

## Checkpoint schema before/after

RAW manifest (`manifest.json`, `RAW_SCHEMA_VERSION = 4`): unchanged — the
migration rebuilds the file list (new projector shard + new sidecar) under the
same kind/identity. Before: `model/` shards with 287 FQNs + `train_state/`
(optimizer.pt, optimizer_schema.json, …). After: identical plus
`model/model-irepa-projector.safetensors` (2 FQNs) and
`train_state/irepa_state.json` (new sidecar, schema v1:
`start_successful_update`, `source_checkpoint_id`, `source_update`,
`migration_seed`).

Architecture document (`model/config.json → architecture`): before =
no-iREPA canonical schema **v3** (`{schema_version:3, class:
TrainableComposite, dit, text, condition_tokens}`); after = schema **v4**
adding `training_auxiliaries.irepa` (the projector metadata: in_channels
2560 → 768, k=3 s=1 p=1 d=1 g=1 bias, bf16 weight / fp32 bias).

Optimizer state (`train_state/optimizer.pt`, hybrid schema v1): before =
CMuon block (141 specs) + AdamW block with 146 parameters under integer IDs +
`sr_rng` + `routing` manifest. After = identical CMuon block and SR RNG;
AdamW block re-IDed by canonical FQN order with the 2 projector FQNs appended
as the LAST member of each group (existing parameters keep their state
verbatim, positions re-indexed but state matched 1:1 by name); `routing`
manifest rewritten to the v4 manifest (AdamW lists now include the projector
at the end; CMuon list byte-identical). `optimizer_schema.json` (v2:
`groups` + `hybrid_cmuon` + `guarded_canonical`): algorithm blocks preserved
verbatim, only the AdamW `groups` FQN lists change.

## Parameter counts

| | total | CMuon | AdamW |
|---|---|---|---|
| source v3 | 287 | 141 | 146 |
| migrated v4 | 289 | 141 | 148 |

Added FQNs: `irepa_alignment.projector.weight`, `irepa_alignment.projector.bias`
(both AdamW, neither CMuon — the CMuon allowlist is unchanged, 141 → 141,
identical FQN set, asserted inside the migration).

## Parity evidence (spec 18, S18 chain)

Chain: one no-iREPA source checkpoint N=2 (two real updates, production-shape
composite, RAW save) → Arm A (legacy) continues to update N+1; Arm B
(`migrate_irepa_checkpoint` + production `load_raw_checkpoint` into a fresh
v4 composite + the PRODUCTION optimizer class built from the real
`train_g1_fp32_rescue_r1.toml`) resumes and performs one update N+1 at
`lambda(N+1) == exact zero` with the FULL teacher/projector/cosine graph
running (no-skip contract). Controlled batch/seeds/timestep; identical
optimizer state (Arm B resumes the saved state); identical SR RNG.

**Resume parity (both arms, before the update):** every pre-existing
parameter bit-exact after resume; SR RNG state at resume bit-exact
(`torch.equal` on the saved/restored generator state).

**Zero-lambda one-step parity:**

- Primary gate (deterministic NS stand-in, bit-exact environment): **PASS** —
  old model tensor mismatches = **0** (all 287 pre-existing parameters
  `torch.equal`); CMuon state mismatches = **0** (141 bf16 momenta
  bit-exact); AdamW state mismatches = **0** (146 entries: step + exp_avg +
  exp_avg_sq bit-exact, torchao OptimState8bit compared via
  block_size/codes/qmap/scale); guard references/bookkeeping identical; MAIN
  JLT loss bit-exact; TOTAL == MAIN bit-exact; fp32-rescue counters equal and
  failure-free (`bf16_attempts=498=166 chunks×3 steps`, zero failures on both
  arms).
- Spec-17 HCU leg (UNPATCHED production optimizer, raw BF16-first NS):
  **PASS** — every deterministic component stays bit-exact (losses,
  non-CMuon parameters, AdamW state, CMuon momenta, guard references,
  counters); the NS-affected CMuon parameter values are within tolerance:
  worst measured relative RMS = **4.781e-03**
  (`dit.conditioner.shared_block_projection.weight`) vs tolerance 5e-2
  (~10x margin; the tolerance is ~30x the documented worst-case BF16-vs-FP32
  cross-NS-path delta and an O(1) divergence — which would indicate a
  checkpoint error — fails loudly per spec 17). No safety failure on either
  arm; rescue counters equal.

**SR RNG status:** the optimizer SR RNG state is checkpointed and restored
verbatim (bit-exact at resume, asserted in both gates). The §11 ordering fix
(`groups.py`) guarantees the projector's SR draw is appended after every
pre-existing AdamW draw, so the old-parameter SR stream is unshifted —
exercised by the production composite where `text.*` decay FQNs sort AFTER
`irepa_alignment.*` (the exact interleaving hazard the fix removes).

**Lambda anchor semantics:** the sidecar records
`start_successful_update = source_update + 1`;
`IRepaLambdaSchedule.weight_for_update(anchor) == 0` exactly, and the anchor
is persisted verbatim on every v4 save (immutability validated by
`validate_irepa_state_document`: `anchor == source_update + 1`). Resume and
the production readiness gate both bind the schedule to the persisted anchor.

## HCU / DDP result

HCU determinism facts (measured on this backend, see
`irepa-phase5-state-parity.json → hcu_determinism_facts`): bf16 `A@B` matmul,
bf16 reductions, and the DiT dense_sdpa forward/backward are bit-deterministic;
bf16 `torch.addmm` (the fused GEMM of the BF16 Newton-Schulz iteration) is
NON-deterministic across calls for identical inputs on every production chunk
shape (`use_deterministic_algorithms` has no effect); the FP32 path is
bit-deterministic. Consequence: the raw BF16-first NS is cross-call
non-deterministic on this HCU — which is why the primary gate runs a
test-only deterministic NS stand-in (production code untouched) and the
spec-17 leg tolerates the NS-affected values.

2-rank DDP smoke (`irepa_ddp_lambda_zero_smoke.py`, torchrun, nccl,
world_size=2, deterministic NS): the full S18 chain at world size 2 — source
chain, rank-0 save+migration, both-rank resume into the production optimizer
(`build_fp32_rescue(rank, world_size=2)`), one lambda=0 update. Proof
targets: per-rank spec-18 parity (identical comparisons to the single-rank
primary gate) AND cross-rank bit identity (fp32 rms/max fingerprints of every
parameter and CMuon momentum of both arms identical across ranks; the
optimizer's internal `invariant_check=True` additionally raises on any
staged-delta spread inside `step()`). The guard state is world-size stamped
(fail-closed on mismatch), so the whole chain runs at the target world size.

**Result: PASS** (221s). Per-rank lambda=0 parity bit-exact (all 287
pre-existing parameters, 141 CMuon momenta, 146 AdamW entries, guard
references, MAIN/TOTAL losses). Cross-rank: **0 mismatches of 864**
fingerprints (every parameter + CMuon momentum of both arms + both losses,
`all_gather_object` exact equality). Guard rescue counters per rank
`bf16_attempts=249` (=498/2: each rank owns half the NS chunks, owner-rank
computes once + broadcasts), zero safety failures.

Production 2-rank G1 is unaffected by the NS observation: the owner rank
computes each NS once per update and broadcasts it (cross-rank structural
consistency); the non-determinism is cross-call, not cross-rank, and not
within one update.

## Production checkpoint test status

- `tests/unit/checkpoint/test_irepa_checkpoint_migration.py` — migration
  happy path + every fail-closed branch (already-iREPA source, missing
  sidecar, anchor mismatch, unknown fields, non-RAW kind, destination
  exists, partial-failure atomicity, FQN-delta must be exactly the
  projector, CMuon-set-invariance, AdamW 1:1 state remap, ID stability,
  dry-run plan): **PASS** (unit, CPU).
- `tests/gpu/checkpoint/test_irepa_checkpoint_save.py` — v3 save rejects the
  anchor under iREPA-disabled/legacy config; v4 save re-persists the sidecar
  verbatim and load re-validates it: **PASS** (GPU).
- `tests/unit/train/test_irepa_production_gate.py` (extended) — the
  production readiness gate is fail-closed on missing/invalid sidecar and
  binds the anchor: **PASS** (unit).
- `tests/gpu/irepa/test_irepa_zero_lambda_optimizer_parity.py` — the S18
  gate above (both modes): **PASS** (GPU, HCU).
- No production (G1/salt11) checkpoint was read, written, or modified.

## CMuon forensic isolation

- The CMuon allowlist and routing are untouched by this phase (141 → 141,
  identical FQN set; asserted inside the migration and by the
  routing-manifest comparison at load).
- No NS algorithm, guard calibration, rescue logic, or optimizer numeric
  change. The deterministic NS stand-in is a **test-only monkeypatch** of a
  module attribute in the parity test; production source is untouched
  (verified: the 3 runtime files of the earlier CMuon F1 work are not part
  of this change set; `cmuon.py` / `guarded_canonical.py` / `fp32_rescue.py`
  have zero diff in this working tree).
- The HCU `bf16 addmm` non-determinism finding is recorded as an
  environment fact for the record only — no P5 action was taken on CMuon
  numerics (spec: HCU nondeterminism must not mask a checkpoint error; the
  tolerance leg is bounded and the O(1) divergence case fails loudly).

## Static checks

- `ruff check src tests`: All checks passed.
- `pyright` (P4 gate rules): **NEW-in-src = 0**, proven by diff against a
  clean `00bd795` worktree run in the identical salt13 environment:
  - `optim/groups.py`, `checkpoint/artifact.py`, `checkpoint/save.py`,
    `checkpoint/migrate_irepa_checkpoint.py` (new): **0 errors each**.
  - `train/production.py` (100), `train/preflight.py` (15),
    `checkpoint/load.py` (9): per-message diff vs baseline = **zero new
    signatures** (production 100→100, preflight 15→15, load 10→9 — the
    Phase 5 diff removes one pre-existing signature, adds none).
  - Test-file noise is limited to the pre-existing torch.distributed stub
    gap (same class as the committed 2-rank baseline script) + the same
    `torch.manual_seed` stub convention used across the repo.
- No new skip/xfail; no suppression of real errors; no unrelated cleanup.

## Full pytest parity (spec 30)

Final tree, `pytest tests/` on salt13 (1306s): **2 failed / 1016 passed /
2 skipped** — identical to the P4 baseline (2 failed / 992 passed / 2
skipped) except the +24 Phase 5 test additions. Both failures are the known
shared baseline, classified BASELINE_SHARED (failed on P4 baseline too):

- `tests/gpu/data/test_pipeline_encoders.py::test_real_pipeline_qwen_and_mage_encode_one_batch` (Qwen asset — pre-existing salt10/salt13 environment condition)
- `tests/gpu/fa4/test_varlen_attention.py::test_forged_boundary_handle_fails_before_native_kernel[host_metadata]` (host_metadata baseline bug — machine-independent)

**iprea Phase5-only failures = 0.**
