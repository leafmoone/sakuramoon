# T044 Infra/performance review

Status: **PASS for the implemented CPU/one-GPU scope after remediation acceptance**.
The independent review below found two blockers. The original reviewer follow-up did
not return a new task receipt, so the main agent performed and recorded the bounded
remediation acceptance without representing it as an independent rereview.

## Remediation acceptance

- `next_pass_index` and `next_seed` are removed from `StageTransitionRequest`, the
  production CLI and the durable transition plan. The plan test enforces an exact
  top-level key set and rejects pass, seed, mainset, tar cursor and tar order controls.
- Raw state validation now cross-binds the outer manifest, inner model manifest and
  model index to the exact config/index/shard roles. Coordinated outer/inner manifest
  attempts to add `model/data_state.json` or `model/opaque.bin` fail before mutation.
- `C12-012` and `OPEN-104` are `implemented`, not incorrectly `verified`, because the
  governed training-system profile still requires 4GPU evidence. T044 review artifacts
  remain mapped, while verification-only evidence fields stay empty.
- Final checks: 69 targeted CPU tests passed in 15.79 seconds; 25 real RTX 5090 raw,
  PMA, growth and full-production tests passed in 96.67 seconds; Ruff passed; Pyright
  reported 0 errors; traceability passed for 235 requirements and 102 production
  modules; `git diff --check` passed.

## Findings

### INFRA-001 (high): the production transition contract still carries the superseded data pass and seed

`src/sakuramoon/train/stage.py:51-52` still makes `next_pass_index` and
`next_seed` mandatory transition inputs. The production CLI then requires and writes
both values into every transition plan at `src/sakuramoon/cli/transition.py:33-34`
and `src/sakuramoon/cli/transition.py:106-107`. The updated tests continue to supply
these fields rather than proving their absence.

That is the old T043 data-reset control surface. It conflicts with the current
confirmed decision that transition does not control tar order or position and with
the closed open-item wording that a transition no longer uses a new
stage/pass/seed to reset data order. Removing `RawCheckpointState.data` is not enough
while the production transition artifact still advertises the superseded pass/seed
inputs to downstream T050 code. Remove these fields from the request, parser, plan
and tests, and add an exact-key contract showing that a transition plan contains no
pass, seed, cursor, order, mainset or data snapshot control. Because the current T044
allowed-path list omits `src/sakuramoon/cli/transition.py`, the main agent must make
that task-boundary adjustment explicitly rather than silently editing outside it.

### INFRA-002 (high): the T044 verified trace entries cannot pass the registry verifier

`docs/model-architecture/progress/traceability.toml:12710` and
`docs/model-architecture/progress/traceability.toml:12863` use
`evidence_hardware = "CPU+1GPU"`, but the verifier accepts only its governed hardware
levels (`CPU`, `1GPU`, or `4GPU`). The same entries cite the T044 test report while
their `artifacts` mappings at lines 12726 and 12878 exclude
`docs/model-architecture/reviews/T044/**`.

`uv run python tools/verify_traceability.py` therefore reports invalid hardware and
out-of-mapping evidence for both `C12-012` and `OPEN-104`; creation of the two review
files alone cannot clear those errors. Use the governed `1GPU` level, extend the
artifact mappings without deleting their existing mappings, and rerun the verifier
after both independent reviews exist.

## Verified behavior

- Raw manifests, trainer state and growth state use schema v2; raw schema v1 and a
  legacy `data_state.json` are explicitly rejected. Model-only, PMA and release
  artifacts remain schema v1.
- The raw sidecar set is exact. `resolved_config.toml` must be nonempty UTF-8 TOML
  and its exact bytes must hash to the checkpoint identity. PMA reaches the same raw
  validation through `read_raw_checkpoint_state` before accepting a source.
- The loader checks the complete physical file set, payload sizes and streaming
  SHA-256, expected identity, model metadata/tensors, optimizer coverage/schema/state,
  trainer/growth state and both RNG payloads before `_apply_model`, optimizer load or
  global RNG restoration. TorchAO deserialization remains `weights_only=True`.
- Save keeps the T042 task-owned temporary tree, file/directory fsync, `COMPLETE`,
  atomic rename, parent fsync and rollback behavior. Model I/O remains one shard at a
  time; the monolithic TorchAO optimizer sidecar and single-writer boundary are
  unchanged and are not represented as new performance improvements.
- The fixed external-batch GPU evidence covers exact output, per-sample/mean loss,
  all 239 FQN gradients, clip coefficient, update, optimizer state and training/SR
  RNG. The AF_UNIX contract shows that the client asks only for health and a worker
  lease at the service's current position.

## Independent verification

The reviewer ran the targeted CPU checkpoint/PMA/stage/transition/service suite:
`67 passed` in 14.29 seconds (four environment/deprecation warnings). Targeted Ruff
passed, Pyright reported `0 errors, 0 warnings`, and `git diff --check` passed. The
task-specific pytest basetemp was precisely removed. Trace verification failed with
the evidence errors described in INFRA-002. No GPU test was rerun because the task's
recorded RTX 5090 runs and full 5.14 GB round-trip were inspected and no GPU resource
was needed to reproduce either finding.

## Residual pending boundaries

Four-rank restore, DDP/NCCL, concurrent multi-rank publication, formal NVMe durability
and timing, measured production host RSS, cache-high-water checkpoint reservation,
long runs and formal stage canaries remain pending or blocked. The recorded one-GPU
and overlay/workspace results do not close those gates, and T044 claims no checkpoint
performance improvement.
