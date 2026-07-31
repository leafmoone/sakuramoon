# T044 AI/model correctness review

Status: PASS for the implemented CPU/one-GPU scope after remediation.

T044 establishes a service-decoupled raw-v2 continuation contract without
rewriting the historical T042 artifacts. The typed raw state contains trainer
counters and canonical stage/growth progress but no shard, mainset, cache,
lease, replay, prefetch or queue state. Model-only, PMA and release artifacts
remain schema v1. Four-rank restore, DDP/NCCL, formal stage canaries and long
runs remain explicitly pending or blocked and are not closed by this review.

## Remediation acceptance

### AI-001: superseded transition data controls removed

The initial review found that `StageTransitionRequest` and the published
transition plan still required T043's superseded `next_pass_index` and
`next_seed`. The request dataclass, production CLI arguments and plan payload
now omit both fields. The plan contract asserts the exact top-level key set and
rejects the presence of mainset, pass, seed, tar cursor or tar order controls.
Stage transition preserves trainer counters and changes only canonical growth
state; it cannot carry or reset data-service position.

### AI-002: nested opaque model payloads rejected

The initial review demonstrated that a raw artifact could hide
`model/data_state.json` under the broad `model/` prefix and still pass
`read_raw_checkpoint_state()`. Raw validation now cross-binds three independent
roles: the outer raw manifest, the inner model manifest and the model index.
The inner manifest must contain exactly `config.json`, the index and the shard
names referenced by that index; outer size/SHA records must match the inner
records. Extra `model/data_state.json` and `model/opaque.bin` are rejected even
when both manifests are rewritten to declare them. This reuses the outer
manifest's already completed payload checksum pass rather than hashing model
payloads again.

## Verified behavior

- Raw manifests plus trainer/growth documents use schema v2; raw-v1 manifests
  fail explicitly, while model-only/PMA/release v1 remains readable.
- Raw save requires nonempty parseable UTF-8 TOML bytes whose exact SHA-256
  matches `CheckpointIdentity.config_sha256` before publication.
- Raw load requires the exact sidecar and model payload sets. Legacy data state,
  root/train-state opaque files, nested model opaque files, missing/malformed
  config and config hash mismatch all fail before model, optimizer or global
  RNG mutation.
- PMA accepts only exact v2 raw sources with the resolved config and still
  publishes a non-resumable schema-v1 inference artifact.
- The fixed-external-batch fresh-process RTX 5090 evidence compares exact
  output, per-sample and mean loss, all 239 canonical-FQN gradients, clip
  values, parameter update, TorchAO optimizer state and both RNG streams. Live
  data continuity is correctly outside this equality contract.
- The real AF_UNIX client contract leases the service's current row using only
  session identity and worker ID; no historical checkpoint cursor is sent or
  requested.

## Independent verification

Commands run during final rereview:

```text
uv run pytest -q tests/unit/checkpoint/test_checkpoint.py tests/unit/checkpoint/test_pma_policy.py tests/unit/train/test_stage.py tests/unit/cli/test_transition.py tests/unit/data/test_d024_service.py --basetemp=docs/model-architecture/reviews/T044/.ai-rereview-pytest -p no:cacheprovider
uv run ruff check src/sakuramoon/checkpoint src/sakuramoon/train/stage.py src/sakuramoon/cli/transition.py tests/unit/checkpoint tests/unit/train/test_stage.py tests/unit/cli/test_transition.py
uv run pyright src/sakuramoon/checkpoint src/sakuramoon/train/stage.py src/sakuramoon/cli/transition.py tests/unit/checkpoint tests/unit/train/test_stage.py tests/unit/cli/test_transition.py
git diff --check
```

Results: 69 targeted CPU tests passed in 15.33 seconds; Ruff passed; strict
Pyright reported 0 errors; `git diff --check` passed. The recorded targeted
RTX 5090 evidence and its test implementation were inspected but not rerun
concurrently with the Infra reviewer. No blocking AI/model-correctness findings
remain in the implemented scope.
