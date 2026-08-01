# D010-D025 DATA package Infra/performance review

Reviewer: independent agent `/root/data_package_review`

Scope: committed Data implementation through D025 at
`016b5263a0add818a1a9bd5efae5b3cdbf406e35`. The unrelated user roadmap edit and
uncommitted T050 work were excluded.

Overall Infra verdict: **CHANGES REQUIRED**. D010-D022 and the D024 service core pass
their implemented CPU scope, but D023's worker launch can deadlock and is inherited by
the D024 consumer and D025 production factory.

## Blocking finding

### High: persistent DataLoader workers use unsafe implicit `fork` and can hang

`src/sakuramoon/data/collate.py:478` constructs the DataLoader without an explicit
`multiprocessing_context`, while `src/sakuramoon/data/collate.py:156` creates command
and completion queues from the default multiprocessing context. On Linux this selects
`fork`. Python 3.12 emitted the repository's existing warning that the pytest/PyTorch
parent was multi-threaded and that `fork()` may deadlock in the child.

This review reproduced the warning as an actual hang twice:

1. The combined DATA CPU selector completed 84 tests, then remained blocked in
   `test_two_persistent_workers_coordinate_parent_state` at
   `src/sakuramoon/data/collate.py:593`. Both child workers were alive with zero CPU and
   waiting in poll. It was interrupted after 162.79 seconds; pytest reported
   `84 passed` before the interruption.
2. A reduced D022-D024 selector completed all 21 D022 state tests, entered the same
   first D023 worker test, produced no worker result, and was terminated by the
   reviewer's 120-second bound with exit code 124.

Production is at least as exposed as these selectors because the trainer process owns
PyTorch, Qwen, Mage-VAE, telemetry, and CUDA-facing libraries before or alongside the
loader lifecycle. A nondeterministic infinite wait violates the persistent two-worker,
bounded failure, and hard-failure requirements; it can stall training without leaving
a normal worker completion or actionable exception.

Required remediation:

- Select one explicit multiprocessing context suitable for a multi-threaded/CUDA
  parent (normally `spawn`) and use that same context for the DataLoader and every
  command/completion queue. Do not silently fall back to `fork` or one worker.
- Add a bounded regression that initializes representative parent library threads
  before starting the two persistent workers, proves both PIDs make progress across
  multiple shards, and fails instead of hanging if a worker cannot start or return.
- Rerun D023 worker exit/parent close/restart, D024 real AF_UNIX service consumer, and
  D025 governed factory-to-worker integration after the change.

## Per-task verdicts

| Task | Infra/performance verdict | Boundary |
|---|---|---|
| D010 | PASS | Bounded listing/streaming, verification, redaction, and durable rollback pass; live network performance remains pending. |
| D011 | PASS | Deterministic streaming bundle publication passes; production approximately 11M-row memory/time remains pending. |
| D012 | PASS | Explicit quota, protected LRU, manifest-bound publication, and rollback pass; production NVMe/quota evidence remains pending. |
| D013 | PASS | Scan memory is bounded by fixed counters/histogram; production decode throughput remains pending. |
| D014 | PASS | Work is bounded by 512 condition tokens and adds no model/network cache. |
| D015 | PASS | Ready budget and bucket fragments are bounded; production RSS/pinned/ready-wait evidence remains pending. |
| D016 | PASS | Startup-only strict schema has no material hot-path cost. |
| D017 | PASS | Sibling temporary, file fsync, replace, preservation, and cleanup pass. |
| D018 | PASS | Fixed 10,001-bin histogram is constant-memory. |
| D019 | PASS | Validation adds only bounded set traversal. |
| D020 | PASS | Fixed-key telemetry transfer is bounded. |
| D021 | PASS | Trusted mapping is bounded per sample; the recorded GPU result is engineering rather than throughput evidence. |
| D022 | PASS | Active cardinality, state publication, eviction protection, and recovery barrier pass (`21` independent tests in this review before D023). |
| D023 | FAIL | Default-fork persistent worker runtime hung twice; parent-only ownership and queue capacities do not make startup/progress bounded. |
| D024 | PARTIAL PASS / BLOCKED | Service/mainset/fault core passed `14` tests; the production two-worker consumer inherits D023 and must be rerun after remediation. |
| D025 | PARTIAL PASS / BLOCKED | Resolved loader-control binding passed `8` tests; `ProductionPipelineFactory.batches()` inherits D023 and lacks post-remediation real integration evidence. |

## Other validation

- Targeted Ruff passed.
- Targeted Pyright reported `0 errors, 0 warnings`.
- Direct trace verification passed with `235/235` requirements/source nodes.
- The exact D025 CPU selector passed `8/8`; no D025 parser/config failure was observed.
- The D024 service/mainset/fault selector excluding worker DataLoader passed `14/14`.

No network, GPU, production NVMe sweep, long training, DDP/NCCL, multi-GPU,
1,000-step canary, or formal stage was run. Cold-cache two-hour supply at
`>=12 samples/s`, ready wait `<2%`, bounded RSS/no swap, quota adherence, worker/queue
sweep, and fully-cached control comparison remain pending and cannot be closed before
the worker hang is remediated.
