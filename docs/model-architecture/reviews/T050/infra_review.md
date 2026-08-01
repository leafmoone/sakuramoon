# T050 independent Infra/performance final review

Reviewer: independent agent
`/root/t050_ai_final_review/t050_infra_final_review`.

Status: **PASS** for the implemented CPU and bounded single-GPU T050 scope.

Scope: frozen uncommitted T050 implementation and its published CPU/static/bounded
RTX 5090 evidence. This reviewer did not run a GPU workload, formal trace verifier,
long run, stage canary, DDP, or NCCL job and made no implementation change.

## Final verdict

No blocking Infra/performance finding remains in T050's implemented scope. The prior
review findings are closed:

- Accepted preflight is process-local, builder-produced, and bound to the exact
  resolved config, D025 stream, runtime, Qwen, VAE, trainable composite, optimizer,
  restored RAW handle, checkpoint publisher, and attention-backend artifact.
- Failed preflight, report publication, training completion, and training failure
  deterministically close the owned D025 stream. Simultaneous check, report, and
  stream-close failures are preserved rather than replacing one another.
- Backward/nonfinite failures retain cleanup failures, and the production loop wires
  its phase timer through the Qwen, VAE, conditioning, DiT, loss, clip, optimizer,
  zero-grad, data-wait, and checkpoint-publication paths without adding CUDA boundary
  synchronizations.
- Resolved checkpoint stage, growth, budget, backend, and cadence identities are
  bound before preflight acceptance and checked again at the training boundary
  before any batch is consumed.

## Worker, service, and resume controls

- The bounded integration requires exact DataLoader worker IDs `{0,1}`, two distinct
  non-parent PIDs per service process, and leases from both workers. It waits on
  marker/lease progress without consuming or ACKing an unused extra batch.
- Service readiness uses a bounded 30-second progress loop. An early child exit
  reports its exit code; timeout first requests stop/join and then uses bounded
  terminate/join cleanup. Normal stop requires exit code zero and AF_UNIX socket
  removal.
- Fresh resume loads local Qwen and Mage-VAE before RAW restore, restores successful
  update 1, replays the active shard, and completes update 2. RAW restore remains the
  last model-assembly operation that mutates training RNG state.
- After review, no pytest, GPU-test, or data-service child remained and the AF_UNIX
  service socket was absent. The zero-byte persistent service lock file is expected
  and is not an active-child or socket leak.

## Evidence reviewed

The frozen RTX 5090 full-chain command in `test_report.json` passed **2 tests with 17
warnings in 275.56 seconds** on driver `580.105.08` and CUDA `12.8`. It covers local
Qwen/Mage-VAE, the 16-layer packed DiT with resolved FA4 binding, strict JLT loss,
backward, FP32 clip, one TorchAO update, real AF_UNIX service, the exact two-worker
contract, durable RAW readback/retention, fresh restore, replay, and the next update.

The reviewer independently ran:

```text
uv run pytest -q tests/unit/train/test_preflight_failures.py \
  tests/unit/train/test_runtime.py tests/unit/train/test_loop.py \
  tests/unit/train/test_step.py \
  --basetemp cache/T050-infra-final-rereview-20260801
```

Result: **92 passed, 17 warnings in 13.85 seconds**.

Ruff passed for every changed T050 production/test file. Strict Pyright passed the
same selection with **0 errors, 0 warnings, and 0 informations**. `git diff --check`
also passed. A separate exact-file JSON/TOML structure check confirmed that T050's
`C12-006` mapping includes scheduler implementation/tests and both final review paths.
The reviewer did not rerun the formal trace verifier because that verifier traverses
the repository's prohibited archive; the frozen root evidence records its separate
40-test and live-verifier passes.

## Residual gates

C002 production config, overlay, and assembly binding is the only current blocker for
the production CLI. Text/Style and every dropout decision are already confirmed; no
parameter decision remains open and no engineering-smoke value is promoted.

The formal 1,000-update S0 canary, sustained maximum-batch-size/throughput and memory
qualification, FID/IS smoke, and all four-GPU DDP/NCCL gates remain pending or
blocked. The bounded one-GPU evidence closes none of those later gates.
